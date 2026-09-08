# Owner(s): ["module: inductor"]
"""End-to-end tests for BACKEND="FLYDSL" flex attention on ROCm.

Forward only, and only where the kernel is known to be correct: bf16/f16 at head_dim 128 on
a validated architecture. The rejection cases are as much of the contract as the numerical
ones, because outside that range the kernel returns plausible-looking wrong numbers rather
than failing. Skipped unless the optional FlyDSL compiler/runtime is installed.
"""

import unittest
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    noop_mask,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.inductor_utils import HAS_GPU


try:
    from torch._inductor.kernel.flex.flydsl_flash_attention import (
        _arch_supported,
        flydsl_unavailable_reason,
    )
    from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (
        build_flex_flash_generic_module,
    )
    from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_interface import (
        prepare,
        regrid_block_mask,
    )

    FLYDSL_UNAVAILABLE_REASON = flydsl_unavailable_reason()
except ImportError as e:
    _arch_supported = None
    FLYDSL_UNAVAILABLE_REASON = f"could not import the FlyDSL flex backend: {e}"


def _flydsl_skip_reason():
    if not HAS_GPU:
        return "no GPU"
    if FLYDSL_UNAVAILABLE_REASON is not None:
        return FLYDSL_UNAVAILABLE_REASON
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    supported, reason = _arch_supported(arch)
    return None if supported else reason


SKIP_REASON = _flydsl_skip_reason()

# head_dim is not a parameter anywhere below: 128 is the only value the kernel computes
# correctly. See _SUPPORTED_HEAD_DIMS.
B, H, S, D = 2, 4, 512, 128


def _alibi(score, b, h, q_idx, kv_idx):
    return score + 0.125 * (h + 1) * (kv_idx - q_idx)


def _softcap(score, b, h, q_idx, kv_idx):
    return 20.0 * torch.tanh(score / 20.0)


def _times_two(score, b, h, q_idx, kv_idx):
    return score * 2.0


def _causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _sliding_window(b, h, q_idx, kv_idx):
    return (kv_idx <= q_idx) & (kv_idx > q_idx - 128)


SCORE_MODS = {"alibi": _alibi, "softcap": _softcap, "times_two": _times_two}
MASK_MODS = {"causal": _causal, "sliding_window": _sliding_window}

# Ops with no AMDGPU libcall, expanded in `flydsl_mod_runtime` over exp2/log2/sqrt. Each
# needs its own case because the expansions branch: `atan` reflects at |x| = 1, `atan2`
# has four quadrants plus the x = 0 axis, and the domain-restricted ones are squashed
# into range first (`asin`/`acos`/`atanh` onto [-1, 1], `acosh` onto x >= 1).
TRANSCENDENTAL_MODS = {
    "sinh": lambda s: torch.sinh(torch.tanh(s)),
    "cosh": lambda s: torch.cosh(torch.tanh(s)),
    "asinh": torch.asinh,
    "acosh": lambda s: torch.acosh(1.0 + torch.abs(s)),
    "atanh": lambda s: torch.atanh(0.9 * torch.tanh(s)),
    "atan": torch.atan,
    # Below and above the |x| > 1 reflection, which is a `where` on both branches.
    "atan_small": lambda s: torch.atan(0.5 * torch.tanh(s)),
    "atan_large": lambda s: torch.atan(8.0 * s),
    "asin": lambda s: torch.asin(torch.tanh(s)),
    "acos": lambda s: torch.acos(torch.tanh(s)),
    "erf": torch.erf,
    "erfc": torch.erfc,
    "atan2": lambda s: torch.atan2(s, 1.0 - s),
    # x == 0 resolves to +/-pi/2 rather than going through the division.
    "atan2_axis": lambda s: torch.atan2(s, torch.zeros_like(s)),
}

# Still rejected: each needs a long rational approximation rather than a few terms, and
# none is plausible in an attention score_mod.
STILL_UNSUPPORTED_MODS = {
    "erfinv": torch.erfinv,
    "lgamma": torch.lgamma,
    "digamma": torch.digamma,
}


@unittest.skipIf(SKIP_REASON is not None, f"FlyDSL flex unavailable: {SKIP_REASON}")
class TestFlyDSLFlexAttention(TestCase):
    def setUp(self):
        super().setUp()
        # Every test varies the mod or the shape on the same `flex_attention` code object,
        # so without a reset the later ones are rejected by Dynamo's recompile limit
        # rather than by anything under test.
        torch._dynamo.reset()

    def _tensors(
        self, dtype=torch.bfloat16, num_kv_heads=H, seq_len=S, head_dim=D, seq_len_kv=None
    ):
        seq_len_kv = seq_len if seq_len_kv is None else seq_len_kv
        torch.manual_seed(0)
        q = torch.randn(B, H, seq_len, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(B, num_kv_heads, seq_len_kv, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(B, num_kv_heads, seq_len_kv, head_dim, device="cuda", dtype=dtype)
        return q, k, v

    def _run(self, q, k, v, *, kernel_options=None, **kwargs):
        """Compile with BACKEND="FLYDSL" and return (flydsl_result, eager_result)."""
        options = {"BACKEND": "FLYDSL", **(kernel_options or {})}
        with torch.no_grad():
            expected = flex_attention(q, k, v, **kwargs)
            compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
            actual = compiled(q, k, v, kernel_options=options, **kwargs)
        return actual, expected

    def _assert_close(self, actual, expected, msg=""):
        self.assertFalse(torch.isnan(actual.float()).any(), f"NaNs in output {msg}")
        scale = expected.float().abs().max().clamp(min=1e-6)
        rel = (actual.float() - expected.float()).abs().max() / scale
        # bf16 attention against an fp32-accumulating eager reference; the tolerance is
        # the dtype's, not the kernel's.
        self.assertLess(rel.item(), 2e-2, f"relative error {rel:.3e} {msg}")

    @parametrize("name", sorted(SCORE_MODS))
    def test_score_mod(self, name):
        q, k, v = self._tensors()
        actual, expected = self._run(q, k, v, score_mod=SCORE_MODS[name])
        self._assert_close(actual, expected, f"for score_mod={name}")

    @parametrize("name", sorted(MASK_MODS))
    def test_mask_mod(self, name):
        q, k, v = self._tensors()
        block_mask = create_block_mask(MASK_MODS[name], B, H, S, S, device="cuda")
        actual, expected = self._run(q, k, v, block_mask=block_mask)
        self._assert_close(actual, expected, f"for mask_mod={name}")

    def test_prefetch_survives_a_walk_with_no_tiles(self):
        """A Q block whose mask leaves nothing to visit still primes the KV prefetch.

        Staging KV through registers asks for the *next* tile before knowing whether there
        is one, so the last iteration reaches one past the walk and a Q block with an empty
        walk reaches past it before starting. Under a block mask that indexes the tile list
        out of range, and the value there is not inert: the row clamp bounds only the top,
        so a stale negative int32 becomes a negative row and the load addresses off the
        front of the tensor. That fault is reachable only when the allocator happens to
        leave something negative in the slot, so it survived a full suite run and then
        killed a benchmark -- hence a test that names the shape rather than trusting luck.

        `kv_idx < 64` leaves every Q block past the first KV block with nothing to visit,
        and the rows it masks away entirely are the ones whose walk is empty.
        """

        def only_first_kv_block(b, h, q_idx, kv_idx):
            return kv_idx < 64

        q, k, v = self._tensors()
        block_mask = create_block_mask(only_first_kv_block, B, H, S, S, device="cuda")
        actual, expected = self._run(q, k, v, block_mask=block_mask)
        self._assert_close(actual, expected, "for a mask with empty walks")

    def test_score_mod_and_mask_mod(self):
        q, k, v = self._tensors()
        block_mask = create_block_mask(_causal, B, H, S, S, device="cuda")
        actual, expected = self._run(q, k, v, score_mod=_alibi, block_mask=block_mask)
        self._assert_close(actual, expected)

    def test_no_mod_is_plain_attention(self):
        """FLYDSL without a mod is served, not refused: the kernel takes none happily."""
        q, k, v = self._tensors()
        actual, expected = self._run(q, k, v)
        self._assert_close(actual, expected)

    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_dtypes(self, dtype):
        q, k, v = self._tensors(dtype=dtype)
        actual, expected = self._run(q, k, v, score_mod=_alibi)
        self.assertEqual(actual.dtype, dtype)
        self._assert_close(actual, expected, f"for {dtype}")

    @parametrize("seq_len", [128, 200, 500, 640])
    def test_ragged_seq_len(self, seq_len):
        """Sequence lengths that are not a multiple of the Q tile.

        The kernel documents `seq_len % 128 == 0` but enforces only `seq_len >= 1`. The
        ragged tail rows come out as accurate as the aligned ones, so that constraint is
        stale rather than unchecked -- pinned here because the head_dim 64 defect looked
        the same from the outside until the swizzle mask was fixed.
        """
        q, k, v = self._tensors(seq_len=seq_len)
        actual, expected = self._run(q, k, v, score_mod=_alibi)
        self._assert_close(actual, expected, f"for seq_len={seq_len}")

    @parametrize("seq_len", [200, 520])
    def test_ragged_seq_len_at_a_partial_load_geometry(self, seq_len):
        """A ragged tail on a head_dim whose KV load geometry is also partial.

        head_dim 96 idles part of the workgroup and takes a short final load batch, so
        its guard bounds the LDS row. A ragged sequence bounds the *global* row at the
        same time, and the two are computed separately -- so cover them together rather
        than trusting the aligned case plus the ragged case at head_dim 64.
        """
        q, k, v = self._tensors(seq_len=seq_len, head_dim=96)
        actual, expected = self._run(q, k, v, score_mod=_alibi)
        self._assert_close(actual, expected, f"for seq_len={seq_len}")

    def test_gqa(self):
        q, k, v = self._tensors(num_kv_heads=H // 2)
        actual, expected = self._run(q, k, v, score_mod=_alibi, enable_gqa=True)
        self._assert_close(actual, expected)

    @parametrize("mod_vec_size", [1, 2, 4])
    def test_mod_vec_size(self, mod_vec_size):
        """All three widths must agree; the autotuner is free to pick any of them."""
        q, k, v = self._tensors()
        actual, expected = self._run(
            q,
            k,
            v,
            score_mod=_alibi,
            kernel_options={"MOD_VEC_SIZE": mod_vec_size},
        )
        self._assert_close(actual, expected, f"for MOD_VEC_SIZE={mod_vec_size}")

    def test_rejects_bad_mod_vec_size(self):
        q, k, v = self._tensors()
        with self.assertRaisesRegex(Exception, "MOD_VEC_SIZE must be 1, 2 or 4"):
            self._run(q, k, v, score_mod=_alibi, kernel_options={"MOD_VEC_SIZE": 3})

    def test_logsumexp_is_natural_log(self):
        """The kernel writes ln(sum exp), not log2, so the wrapper must not rescale it.

        The failure is silent: log2 is off by a constant factor and still looks plausible.
        """
        q, k, v = self._tensors()
        with torch.no_grad():
            expected_out, expected_lse = flex_attention(
                q, k, v, score_mod=_alibi, return_lse=True
            )
            compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
            actual_out, actual_lse = compiled(
                q,
                k,
                v,
                score_mod=_alibi,
                return_lse=True,
                kernel_options={"BACKEND": "FLYDSL"},
            )
        self._assert_close(actual_out, expected_out)
        self._assert_close(actual_lse, expected_lse, "for logsumexp")

    def test_logsumexp_masked_rows(self):
        """A fully masked query row has no finite LSE, and the pattern must match eager."""
        q, k, v = self._tensors()
        block_mask = create_block_mask(_causal, B, H, S, S, device="cuda")
        with torch.no_grad():
            _, expected_lse = flex_attention(
                q, k, v, block_mask=block_mask, return_lse=True
            )
            compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
            _, actual_lse = compiled(
                q,
                k,
                v,
                block_mask=block_mask,
                return_lse=True,
                kernel_options={"BACKEND": "FLYDSL"},
            )
        self.assertEqual(torch.isfinite(actual_lse), torch.isfinite(expected_lse))

    def test_captured_tensor_broadcast_over_heads(self):
        """An [H] table: rank 1, and the stride spec has to land on the h axis."""
        q, k, v = self._tensors()
        slopes = torch.linspace(0.1, 0.4, H, device="cuda", dtype=torch.float32)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + slopes[h] * (kv_idx - q_idx)

        actual, expected = self._run(q, k, v, score_mod=score_mod)
        self._assert_close(actual, expected)

    def test_captured_tensor_fully_addressed(self):
        """A [B, H, S, S] bias: every axis of the stride spec in use."""
        q, k, v = self._tensors()
        bias = torch.randn(B, H, S, S, device="cuda", dtype=torch.float32) * 0.1

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]

        actual, expected = self._run(q, k, v, score_mod=score_mod)
        self._assert_close(actual, expected)

    def test_captured_tensor_read_by_row_and_by_column(self):
        """One capture, two index patterns, so two slots over the same tensor.

        A slot carries a single stride spec, so reading a table by ``q_idx`` and by
        ``kv_idx`` cannot share one -- they are the same data addressed two ways. Keying
        slots by tensor alone used to refuse the second read outright, which is what took
        document masking out.
        """
        q, k, v = self._tensors()
        doc_id = (torch.arange(S, device="cuda") // 128).to(torch.int32)

        def mask_mod(b, h, q_idx, kv_idx):
            return doc_id[q_idx] == doc_id[kv_idx]

        block_mask = create_block_mask(mask_mod, B, H, S, S, device="cuda")
        actual, expected = self._run(q, k, v, block_mask=block_mask)
        self._assert_close(actual, expected)

    def test_captured_tensor_indexed_by_a_loaded_value(self):
        """A gather: the index into one capture is read out of another.

        ``offsets[document_id[q_idx]]`` is the shape of it, and the reader has no gather
        entry point -- it sums ``stride * argument`` over four positions. A value handed to
        a position whose stride is the axis stride *is* the gather, so this pins that the
        spare position is found and the f32 the value arrives as becomes an index.

        Written out rather than taken from ``attn_gym`` so the suite keeps no dependency on
        it; this is that library's ``generate_doc_mask_mod`` with a causal inner mask.
        """
        q, k, v = self._tensors()
        doc_len = 128
        offsets = torch.arange(0, S + 1, doc_len, device="cuda", dtype=torch.int32)
        doc_id = (torch.arange(S, device="cuda") // doc_len).to(torch.int32)

        def mask_mod(b, h, q_idx, kv_idx):
            same_doc = doc_id[q_idx] == doc_id[kv_idx]
            q_logical = q_idx - offsets[doc_id[q_idx]]
            kv_logical = kv_idx - offsets[doc_id[kv_idx]]
            return same_doc & (q_logical >= kv_logical)

        block_mask = create_block_mask(mask_mod, B, H, S, S, device="cuda")
        actual, expected = self._run(q, k, v, block_mask=block_mask)
        self._assert_close(actual, expected)

    def test_backward_matches_eager_for_a_document_mask(self):
        """The same gather in the backward, where both mods re-render it."""
        doc_len = 128
        offsets = torch.arange(0, 512 + 1, doc_len, device="cuda", dtype=torch.int32)
        doc_id = (torch.arange(512, device="cuda") // doc_len).to(torch.int32)

        def mask_mod(b, h, q_idx, kv_idx):
            same_doc = doc_id[q_idx] == doc_id[kv_idx]
            q_logical = q_idx - offsets[doc_id[q_idx]]
            kv_logical = kv_idx - offsets[doc_id[kv_idx]]
            return same_doc & (q_logical >= kv_logical)

        self._assert_grads_match_eager("document mask", seq_len=512, mask_mod=mask_mod)

    @parametrize("seq_len", [63, 65, 127])
    def test_captured_bias_reaches_the_last_element_at_odd_seq_len(self, seq_len):
        """The very last bias element must be read, at a length where a dword straddles.

        A whole-tensor norm cannot see this. Upstream hit exactly this class: with a buffer
        bound ending on the last valid element, a wide load whose final dword crosses
        ``num_records`` is dropped by the hardware, so ``bias[..., sq-1, sk-1]`` silently
        reads zero. Their symptom was the last output row coming back bit-identical to the
        no-bias run at lengths 63 and 65 while moving at 64, and it survived their
        aggregate tolerance tests.

        Our aux resources are built with ``max_size=True``, so the bound is the whole
        allocation and there is nothing to straddle -- this is a movement assertion that
        pins that, and would fail if anyone tightened the bound to the tensor's extent.
        """
        q, k, v = self._tensors(seq_len=seq_len)
        bias = torch.zeros(B, H, seq_len, seq_len, device="cuda", dtype=torch.float32)
        # Only the final element is non-zero, so it alone can move the final output row.
        bias[-1, -1, -1, -1] = 8.0

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]

        biased, expected = self._run(q, k, v, score_mod=score_mod)
        self._assert_close(biased, expected, f"for seq_len={seq_len}")

        unbiased, _ = self._run(q, k, v)
        moved = (biased[-1, -1, -1] - unbiased[-1, -1, -1]).abs().max().item()
        self.assertGreater(
            moved,
            1e-3,
            f"bias[..., {seq_len - 1}, {seq_len - 1}] was not read: the last output row is "
            "unchanged by an 8.0 bias on the last score",
        )

    @parametrize("n_captures", [1, 2, 3, 4])
    def test_capture_count_up_to_the_slot_limit(self, n_captures):
        """The kernel signature carries four aux slots, so four captures must work.

        It carried two until the slots were widened, which made a score_mod and a
        mask_mod that each capture a couple of tensors unrepresentable. Slots past the
        last used one are handed a dummy, so the interesting case is the boundary: at
        four, nothing is left over to pad with.
        """
        q, k, v = self._tensors()
        tables = [
            torch.rand(H, device="cuda", dtype=torch.float32) for _ in range(n_captures)
        ]

        def score_mod(score, b, h, q_idx, kv_idx):
            for t in tables:
                score = score + t[h] * 0.1
            return score

        actual, expected = self._run(q, k, v, score_mod=score_mod)
        self._assert_close(actual, expected, f"with {n_captures} captures")

    def test_rejects_captures_past_the_slot_limit(self):
        """A fifth capture has nowhere to go and must say so."""
        q, k, v = self._tensors()
        tables = [torch.rand(H, device="cuda", dtype=torch.float32) for _ in range(5)]

        def score_mod(score, b, h, q_idx, kv_idx):
            for t in tables:
                score = score + t[h] * 0.1
            return score

        with self.assertRaisesRegex(Exception, "captured tensors"):
            self._run(q, k, v, score_mod=score_mod)

    def test_aux_slot_limit_matches_the_kernel_signature(self):
        """Lowering's copy of the slot count must match the kernel's own.

        Lowering cannot import the kernel module -- it runs where FlyDSL may not be
        installed -- so the limit is duplicated, and a mismatch would either reject
        captures the kernel could hold or pass more arguments than the signature takes.
        """
        from torch._inductor.kernel.flex.flydsl_flash_attention import (
            _MAX_AUX_TENSORS,
        )
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (
            MAX_AUX_TENSORS,
        )

        self.assertEqual(_MAX_AUX_TENSORS, MAX_AUX_TENSORS)

    def test_captured_view_built_inside_the_graph(self):
        """A capture that is a view produced by the traced graph, not a leaf tensor.

        These arrive as a ``ReinterpretView`` under a synthetic input name, so the node
        behind the name lives only in the graph's capture table.
        """
        q, k, v = self._tensors()

        def eager(q, k, v, big, kernel_options):
            bias = big[:, :, :, :S].transpose(-1, -2)

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + bias[b, h, q_idx, kv_idx]

            return flex_attention(
                q, k, v, score_mod=score_mod, kernel_options=kernel_options
            )

        big = torch.randn(B, H, S, S * 2, device="cuda", dtype=torch.float32) * 0.1
        with torch.no_grad():
            expected = eager(q, k, v, big, None)
            compiled = torch.compile(eager, fullgraph=True, dynamic=False)
            actual = compiled(q, k, v, big, {"BACKEND": "FLYDSL"})
        self._assert_close(actual, expected)

    def test_capture_indexed_by_an_offset_coordinate(self):
        """``table[q_idx + 1]``: an offset index, which the reader takes as a value.

        This was refused for as long as an axis had to be indexed by a bare coordinate.
        The same spare-position mechanism that document masking needs covers it, since a
        reader position multiplies its argument by that axis's stride and does not care
        whether the argument is a coordinate or something computed from one -- so the
        rejection was a limitation and not a property worth keeping.
        """
        q, k, v = self._tensors()
        table = torch.randn(S + 1, device="cuda", dtype=torch.float32) * 0.1

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + table[q_idx + 1]

        actual, expected = self._run(q, k, v, score_mod=score_mod)
        self._assert_close(actual, expected)

    def test_captured_device_scalar(self):
        """A 0-d capture on device is fine: rank 0 broadcasts over every axis."""
        q, k, v = self._tensors()
        scale = torch.tensor(2.0, device="cuda")

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * scale

        actual, expected = self._run(q, k, v, score_mod=score_mod)
        self._assert_close(actual, expected)

    def test_rejects_cpu_scalar_capture(self):
        """A 0-d CPU capture reaches the kernel as a host pointer and faults the GPU.

        Rejected before the captures are realized, since afterwards they no longer look
        like scalars.
        """
        q, k, v = self._tensors()
        cpu_scalar = torch.tensor(2.0)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * cpu_scalar

        with self.assertRaisesRegex(Exception, "0-dim CPU tensor scalar"):
            self._run(q, k, v, score_mod=score_mod)

    @parametrize("name", sorted(TRANSCENDENTAL_MODS))
    def test_transcendental_mod_ops(self, name):
        """Ops the shim expands rather than calling, checked against eager.

        These are polynomial and exp2/log2 expansions, so accuracy is a property of this
        code rather than of a vendor library and has to be measured. All of them land at
        the same ~0.004 relative error as plain attention, i.e. bf16 rounding dominates
        and the expansion contributes nothing measurable.
        """
        fn = TRANSCENDENTAL_MODS[name]

        def mod(score, b, h, q_idx, kv_idx):
            return fn(score)

        q, k, v = self._tensors()
        actual, expected = self._run(q, k, v, score_mod=mod)
        self._assert_close(actual, expected, f"for {name}")

    @parametrize("name", sorted(STILL_UNSUPPORTED_MODS))
    def test_rejects_unexpanded_mod_ops(self, name):
        """The three remaining ops must fail by name, not as an LLVM libcall error."""
        fn = STILL_UNSUPPORTED_MODS[name]

        def mod(score, b, h, q_idx, kv_idx):
            return fn(score)

        q, k, v = self._tensors()
        with self.assertRaisesRegex(Exception, name):
            self._run(q, k, v, score_mod=mod)

    @parametrize("head_dim", [64, 96, 128, 160, 192, 224, 256])
    def test_head_dim(self, head_dim):
        """64 is correct only because the K swizzle sizes its row mask to HEAD_DIM.

        With the mask hardcoded to 7 this built happily and returned ~44% relative
        error, which is why it is pinned rather than assumed.
        """
        q, k, v = self._tensors(head_dim=head_dim)
        actual, expected = self._run(q, k, v, score_mod=_alibi)
        self._assert_close(actual, expected, f"for head_dim={head_dim}")

    @parametrize("head_dim", [96, 160, 192, 224])
    def test_non_power_of_two_head_dim_disables_the_k_swizzle(self, head_dim):
        """These head_dims are correct only with K's XOR swizzle turned off.

        The swizzle permutes a row by ``col ^ ((row & mask) << 4)``, which is a
        permutation only over a power-of-two extent. ``HEAD_DIM // 16`` is 6, 10, 12 and
        14 here, so the mask is not contiguous and the XOR leaves the row -- at head_dim
        96, ``col ^ 80`` from col 32 up. It built and returned ~45% error, so pin that
        the mask is disabled rather than trusting the numbers alone.
        """
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (
            build_flex_flash_generic_module,
        )

        launcher = build_flex_flash_generic_module(
            num_heads=H,
            head_dim=head_dim,
            causal=False,
            dtype_str="bf16",
            layout="bhsd",
        )
        self.assertEqual(launcher.k_swizzle_rowmask, 0)

        q, k, v = self._tensors(head_dim=head_dim)
        actual, expected = self._run(q, k, v, score_mod=_alibi)
        self._assert_close(actual, expected, f"for head_dim={head_dim}")

    @parametrize("head_dim", [48, 80])
    def test_rejects_head_dim_off_the_32_grid(self, head_dim):
        """The O accumulators are 32-element chunks, so head_dim has to be a multiple.

        Rejected in the allowlist rather than left to fail a bare ``assert`` in the
        build, which is what the kernel itself does.
        """
        q, k, v = self._tensors(head_dim=head_dim)
        with self.assertRaisesRegex(Exception, "head_dim"):
            self._run(q, k, v, score_mod=_alibi)

    @parametrize("seq_len_kv", [128, 256, 1024, 300])
    def test_cross_attention_forward(self, seq_len_kv):
        """Q and KV lengths may differ. Both orders, and one that is a ragged pair.

        This is the test that earns the two separate extents. Held as a single `seq_len`,
        every one of these came back wrong rather than slow -- 0.97 relative error at
        `sk=256` and 1.57 at `sk=1024`, silently, because the KV walk ran to the *Q*
        length. Both orders are covered because they fail differently: a short KV reads
        past the real keys, a long one never reaches its tail.

        `seq_len_kv=300` against `seq_len=512` makes both extents ragged against the tile
        at once, which is where a bound that used the wrong one still looks plausible.
        """
        q, k, v = self._tensors(seq_len=512, seq_len_kv=seq_len_kv)
        actual, expected = self._run(q, k, v)
        self._assert_close(actual, expected, f"for seq_len_kv={seq_len_kv}")

    @parametrize("seq_len_kv", [256, 1024])
    def test_cross_attention_forward_with_a_mod_and_a_mask(self, seq_len_kv):
        """Cross attention through the mod call sites and a `mask_mod`'s BlockMask.

        The mask's grid is no longer square, so the regrid has to rescale each axis
        against its own tile count rather than reusing one.
        """
        q, k, v = self._tensors(seq_len=512, seq_len_kv=seq_len_kv)
        block_mask = create_block_mask(_causal, B, H, 512, seq_len_kv, device="cuda")
        actual, expected = self._run(q, k, v, score_mod=_alibi, block_mask=block_mask)
        self._assert_close(actual, expected, f"for seq_len_kv={seq_len_kv}")

    def test_rejects_key_and_value_disagreeing_on_length(self):
        """Q may differ from KV, but K and V may not differ from each other.

        One column index reads both, so there is no second extent to give them -- unlike
        the Q axis, which is exactly that. Checked against the interface helper rather than
        through `flex_attention`, because FlexAttention's own batched matmul rejects this
        shape first and a test routed through it would be pinning someone else's error.
        """
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_interface import (
            _seq_lens,
        )

        q, k, _ = self._tensors(seq_len=512, seq_len_kv=256)
        _, _, v = self._tensors(seq_len=512, seq_len_kv=128)
        with self.assertRaisesRegex(ValueError, "key and value"):
            _seq_lens(q, k, v, axis=2)
        self.assertEqual(_seq_lens(q, k, k, axis=2), (512, 256))

    def test_rejects_head_dim_past_the_lds_budget(self):
        """head_dim 288 needs 73728 B of LDS against gfx942's 65536 B.

        The kernel does not check this itself -- it surfaces as a compile-time
        ``local memory (73728) exceeds limit`` from the backend -- so the allowlist is
        what keeps it from getting that far.
        """
        q, k, v = self._tensors(head_dim=288)
        with self.assertRaisesRegex(Exception, "head_dim"):
            self._run(q, k, v, score_mod=_alibi)

    def test_head_dim_256_rescales_every_o_chunk(self):
        """head_dim 256 has 8 O accumulator chunks but only 4 PV k-steps.

        The online-softmax rescale of chunks 1.. is deferred into the GEMM2 loop to hide
        the multiply. Keyed off the k-step it capped at 4 rescales, so chunks 5-7 went
        unscaled and everything from d160 up was wrong by ~42% while d0-d128 stayed
        exact. A whole-tensor norm would catch that, but only barely at some shapes, so
        check the affected d range on its own.
        """
        head_dim = 256
        q, k, v = self._tensors(head_dim=head_dim)
        actual, expected = self._run(q, k, v, score_mod=_alibi)

        # d160 up is the range the deferred rescale used to miss.
        tail_a, tail_e = actual[..., 160:].float(), expected[..., 160:].float()
        rel = ((tail_a - tail_e).norm() / tail_e.norm()).item()
        self.assertLess(rel, 2e-2, f"d160+ disagrees for head_dim={head_dim}: {rel}")
        self._assert_close(actual, expected, f"for head_dim={head_dim}")

    def test_rejects_float32(self):
        """fp32 is refused on purpose, not pending.

        An fp32 MFMA exists, but the tile doubles with the element -- head_dim 128 lands
        exactly on the 65536 B LDS limit and 256 wants 131072 B -- and `32x32x2f32` is 8x
        the instructions of `32x32x16_bf16` for the same K extent. Triton's flex path
        handles fp32, so this rejection routes to a working kernel.
        """
        q, k, v = self._tensors(dtype=torch.float32)
        with self.assertRaisesRegex(Exception, "bf16/f16"):
            self._run(q, k, v, score_mod=_alibi)

    def _bshd_view_tensors(self, requires_grad=False, head_dim=D):
        """q/k/v whose sizes are BHSD and whose memory is BSHD.

        What a model produces: a projection reshaped to [B, S, H, D] and transposed for
        attention. FlexAttention passes the view through, so this is the layout the
        kernel is normally handed -- and the one every other test here does *not* cover,
        since they all allocate BHSD directly.
        """
        torch.manual_seed(0)
        return [
            torch.randn(B, S, H, head_dim, device="cuda", dtype=torch.bfloat16)
            .transpose(1, 2)
            .detach()
            .requires_grad_(requires_grad)
            for _ in range(3)
        ]

    def test_bshd_strided_inputs_are_indexed_where_they_are(self):
        """A BSHD-dense input builds a BSHD kernel instead of copying itself into BHSD.

        The kernel indexes either layout, so the choice is only about who copies. Getting
        it wrong is not a correctness bug, which is why it went unnoticed: it cost three
        contiguous copies of q/k/v and one of the output, 574 us against a 461 us kernel
        at 4x16x4096x128, and read as the sparse-mask forward deficit because a short KV
        walk has nothing to amortise it with.
        """
        from torch._inductor.utils import run_and_get_code

        for tensors, want in (
            (self._bshd_view_tensors(), 'layout="bshd"'),
            (self._tensors(), 'layout="bhsd"'),
        ):
            q, k, v = tensors
            with torch.no_grad():
                expected = flex_attention(q, k, v, score_mod=_alibi)
                torch._dynamo.reset()
                compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
                actual, (code,) = run_and_get_code(
                    compiled,
                    q,
                    k,
                    v,
                    score_mod=_alibi,
                    kernel_options={"BACKEND": "FLYDSL"},
                )
            self._assert_close(actual, expected, f"for {want}")
            self.assertIn(want, code)

    def test_bshd_strided_inputs_can_be_pinned_to_either_layout(self):
        """`LAYOUT` overrides the choice, for a shape whose own measurement disagrees.

        The copies grow with `S` where the kernel grows with `S**2`, so the default is a
        judgement that held over every shape measured rather than a proof.
        """
        q, k, v = self._bshd_view_tensors()
        for layout in ("bhsd", "bshd"):
            actual, expected = self._run(
                q, k, v, kernel_options={"LAYOUT": layout}, score_mod=_alibi
            )
            self._assert_close(actual, expected, f"for LAYOUT={layout}")
        with self.assertRaisesRegex(Exception, "LAYOUT must be"):
            self._run(q, k, v, kernel_options={"LAYOUT": "bsdh"}, score_mod=_alibi)

    def _grads(
        self,
        *,
        num_q_heads=H,
        num_kv_heads=H,
        seq_len=S,
        seq_len_kv=None,
        head_dim=D,
        score_mod=None,
        mask_mod=None,
        backend="FLYDSL",
        kernel_options=None,
    ):
        """Run forward+backward once through ``backend``, returning (dq, dk, dv)."""
        torch.manual_seed(0)
        gqa = num_q_heads != num_kv_heads
        seq_len_kv = seq_len if seq_len_kv is None else seq_len_kv

        def _rand(heads, grad, rows=seq_len):
            return torch.randn(
                B,
                heads,
                rows,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=grad,
            )

        q = _rand(num_q_heads, True)
        k = _rand(num_kv_heads, True, seq_len_kv)
        v = _rand(num_kv_heads, True, seq_len_kv)
        grad_out = _rand(num_q_heads, False)

        block_mask = (
            create_block_mask(mask_mod, None, None, seq_len, seq_len_kv, device="cuda")
            if mask_mod is not None
            else None
        )

        if backend is None:
            out = flex_attention(
                q, k, v, score_mod=score_mod, block_mask=block_mask, enable_gqa=gqa
            )
        else:
            torch._dynamo.reset()
            compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
            out = compiled(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=block_mask,
                enable_gqa=gqa,
                kernel_options={"BACKEND": backend, **(kernel_options or {})},
            )
        out.backward(grad_out)
        return q.grad, k.grad, v.grad

    def _assert_grads_match_eager(self, msg, **kwargs):
        """Assert FlyDSL's three gradients match eager's for the same inputs."""
        actual = self._grads(**kwargs)
        expected = self._grads(backend=None, **kwargs)
        for name, a, b in zip(("dq", "dk", "dv"), actual, expected):
            self.assertIsNotNone(a, f"{name} was not produced")
            self._assert_close(a, b, f"for {name}, {msg}")

    @parametrize("num_kv_heads", [H, 2, 1])
    def test_backward_matches_eager(self, num_kv_heads):
        """Gradients from the FlyDSL backward, against eager.

        The test this replaced asserted only that `q.grad` was not None and that the
        *forward* output was right, so it would have passed on any gradient at all.

        The three head counts are MHA, a GQA group of 2 and MQA. `dq` needs no reduction
        across a group -- each Q head owns its rows -- so what this covers on the kernel
        side is its KV-head indexing, not a group sum.
        """
        actual = self._grads(num_kv_heads=num_kv_heads)
        expected = self._grads(num_kv_heads=num_kv_heads, backend=None)
        for name, a, b in zip(("dq", "dk", "dv"), actual, expected):
            self.assertIsNotNone(a, f"{name} was not produced")
            self._assert_close(a, b, f"for {name}, num_kv_heads={num_kv_heads}")

    @parametrize("head_dim", [64, 96, 128, 256])
    def test_backward_matches_eager_across_head_dims(self, head_dim):
        """The backward reaches the same head_dim ladder as the forward.

        96 is the interesting one: not a power of two, so the K swizzle is off -- and off
        in the backward's *transposed* K tile as well as the row-major one, which is the
        backward's own and so not covered by the forward's head_dim tests. 256 only fits
        because the backward's KV tile is 32 where the forward's is 64; at 64 its three
        LDS tiles wanted 98304 B against gfx942's 65536 B.
        """
        actual = self._grads(head_dim=head_dim)
        expected = self._grads(head_dim=head_dim, backend=None)
        for name, a, b in zip(("dq", "dk", "dv"), actual, expected):
            self._assert_close(a, b, f"for {name}, head_dim={head_dim}")

    @parametrize("seq_len", [65, 200])
    def test_backward_matches_eager_for_ragged_seq_len(self, seq_len):
        """A partial KV tile reads clamped duplicate rows; their `p` must come out 0.

        Nothing downstream of the score site would notice if it did not -- `dq` has no
        cross-row reduction to poison -- so the duplicated rows would simply be added in.
        """
        actual = self._grads(seq_len=seq_len)
        expected = self._grads(seq_len=seq_len, backend=None)
        for name, a, b in zip(("dq", "dk", "dv"), actual, expected):
            self._assert_close(a, b, f"for {name}, seq_len={seq_len}")

    def test_backward_matches_eager_for_bshd_strided_inputs(self):
        """Gradients when the memory is BSHD, which is the layout a model hands over.

        The backward reads q/k/v/do and writes dq/dk/dv, so it had seven copies to make
        where the forward had four. It now indexes in place like the forward does, and
        that moves *both* backward kernels onto their transposed addressing at once --
        `dq` walking KV per Q tile and `dkdv` the other way round.
        """
        q, k, v = self._bshd_view_tensors(requires_grad=True)
        grad_out = torch.randn_like(q)

        expected = torch.autograd.grad(
            flex_attention(q, k, v, score_mod=_alibi), (q, k, v), grad_out
        )
        torch._dynamo.reset()
        compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
        actual = torch.autograd.grad(
            compiled(
                q, k, v, score_mod=_alibi, kernel_options={"BACKEND": "FLYDSL"}
            ),
            (q, k, v),
            grad_out,
        )
        for name, a, b in zip(("dq", "dk", "dv"), actual, expected):
            self._assert_close(a, b, f"for {name}, BSHD-strided inputs")

    @parametrize("tile", [(128, 32), (256, 32), (128, 64), (64, 32)])
    def test_backward_honours_an_explicit_dkdv_tile(self, tile):
        """Every tile the autotuner may pick has to give the same gradients.

        The tile changes which axis is 32-wide per wave and how much of Q is resident, so
        it moves both the LDS layout and the wave count -- the kind of thing that is right
        for one shape and off-by-a-strip for another.
        """
        actual = self._grads(kernel_options={"DKDV_TILE": tile})
        expected = self._grads(backend=None)
        for name, a, b in zip(("dq", "dk", "dv"), actual, expected):
            self._assert_close(a, b, f"for {name}, DKDV_TILE={tile}")

    @parametrize("seq_len_kv", [256, 1024, 300])
    def test_backward_matches_eager_for_cross_attention(self, seq_len_kv):
        """Cross attention through both backward kernels, which tile opposite axes.

        This is where the two extents are easiest to confuse: `dq` tiles Q and reduces
        over KV, `dkdv` tiles KV and reduces over Q, so each kernel's grid, resident tile
        and padding mask take a *different* one of the two. Getting a single site backwards
        leaves the other gradients intact, which is why all three are compared.
        """
        self._assert_grads_match_eager(
            f"seq_len_kv={seq_len_kv}", seq_len=512, seq_len_kv=seq_len_kv
        )

    @parametrize("seq_len_kv", [256, 1024])
    def test_backward_matches_eager_for_cross_attention_under_a_mask(self, seq_len_kv):
        """Cross attention plus a `mask_mod`, so both walks run off regridded block lists.

        The two lists are transposes of one occupancy on a grid that is no longer square,
        so a Q tile count used where a KV one belongs now changes the answer instead of
        being invisible.
        """
        self._assert_grads_match_eager(
            f"causal, seq_len_kv={seq_len_kv}",
            seq_len=512,
            seq_len_kv=seq_len_kv,
            mask_mod=_causal,
        )

    @parametrize("staged", [False, True])
    def test_dq_agrees_with_eager_under_either_kv_staging(self, staged):
        """Staging dq's KV tile through registers must not change the gradient.

        The staged path rewrites how a tile reaches LDS -- the global read becomes
        unconditional and row-clamped with the tile guard moved to the store side, and the
        loop carries the *next* tile's registers, so it also computes a next-tile index one
        step past the walk. A ragged KV length and a mask (which puts the walk on a block
        list, where that lookahead reads past the end of the list rather than past the
        tensor) are the two places that lookahead can go wrong.
        """
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels import (
            flex_flash_bwd_generic,
        )

        real = flex_flash_bwd_generic.build_flex_flash_bwd_dq_module

        def pinned(*args, **kwargs):
            kwargs["enable_kv_gpfetch"] = staged
            return real(*args, **kwargs)

        with mock.patch.object(
            flex_flash_bwd_generic, "build_flex_flash_bwd_dq_module", pinned
        ):
            self._assert_grads_match_eager(
                f"staged={staged}", seq_len=512, seq_len_kv=333, mask_mod=_causal
            )

    @parametrize("staged", [False, True])
    def test_dkdv_agrees_with_eager_under_either_q_staging(self, staged):
        """Staging dkdv's Q/DO tiles through registers must not change the gradient.

        The mirror of the dq case, and reached differently: dkdv stages by default only at
        head_dim 160/192/224, so both settings have to be pinned to exercise both paths at
        the head_dim the rest of this file uses. Its walk is over Q with the GQA group as
        the outer loop, so the lookahead one step past the walk has to be safe per Q head
        and the pipeline has to re-prime at each head rather than inherit the last one's
        in-flight tile -- a ragged Q length and a mask put both under test.
        """
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels import (
            flex_flash_bwd_generic,
        )

        real = flex_flash_bwd_generic.build_flex_flash_bwd_dkdv_module

        def pinned(*args, **kwargs):
            kwargs["enable_q_gpfetch"] = staged
            return real(*args, **kwargs)

        with mock.patch.object(
            flex_flash_bwd_generic, "build_flex_flash_bwd_dkdv_module", pinned
        ):
            self._assert_grads_match_eager(
                f"staged={staged}", seq_len=333, seq_len_kv=512, mask_mod=_causal
            )
            self._assert_grads_match_eager(
                f"staged={staged}, gqa", seq_len=512, seq_len_kv=256, num_kv_heads=2
            )

    def test_precompiling_choices_in_subprocesses_changes_nothing_but_speed(self):
        """Autotuning with the precompile workers on must pick the same kernel it would.

        The workers only warm FlyDSL's on-disk cache, so this is a no-op by construction
        -- which is worth a test precisely because it is invisible when it breaks. A pool
        that silently compiled nothing, or a parent that missed the cache the workers
        filled, both still produce correct output and would go unnoticed here; what this
        catches is the case that does not, where a worker's compile is not the compile the
        parent wanted and the answer changes.
        """
        q, k, v = self._tensors(seq_len=512)
        with config.patch({"flydsl.precompile_workers": 2, "max_autotune": True}):
            actual, expected = self._run(q, k, v, score_mod=SCORE_MODS["alibi"])
        self._assert_close(actual, expected, "with the precompile pool on")

    def test_disabling_the_precompile_pool_is_a_supported_configuration(self):
        """Zero workers has to keep working: it is the escape hatch.

        Eight CUDA-holding subprocesses is not always the right trade -- a small card
        autotuning inside a large model would rather have the memory -- so the serial path
        stays a first-class configuration rather than a fallback that rots untested.
        """
        from torch._inductor.autotune_process import _FlyDSLPrecompilePool

        q, k, v = self._tensors(seq_len=256)
        with config.patch({"flydsl.precompile_workers": 0}):
            self.assertIsNone(_FlyDSLPrecompilePool.get())
            actual, expected = self._run(q, k, v)
        self._assert_close(actual, expected, "with the precompile pool off")

    def test_backward_matches_eager_for_cross_attention_with_gqa(self):
        """GQA on top: dkdv's per-Q-head walk and the group loop, at two extents."""
        self._assert_grads_match_eager(
            "gqa cross attention", seq_len=512, seq_len_kv=256, num_kv_heads=2
        )

    def test_backward_rejects_a_malformed_dkdv_tile(self):
        with self.assertRaisesRegex(Exception, "DKDV_TILE"):
            self._grads(kernel_options={"DKDV_TILE": 128})

    def test_every_builder_requires_a_size_for_each_aux_tensor(self):
        """Strides without an element count is refused, in all three builders.

        The buffer descriptor is bounded with that count, and unbounded is not a milder
        version of the same thing: the mods run on the whole score tile, so the aux reader
        is called at rows and columns past ``seq_len`` and at the last ``(b, h)`` those
        offsets leave the tensor. Unbounded, they reached unmapped memory -- a GPU memory
        access fault whose reproducibility depended on what the caching allocator had
        placed next, so it survived this file run test-by-test and killed it run whole.

        Refused rather than defaulted, because a default here is either the fault we just
        removed or a guess at someone's tensor size.
        """
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_bwd_generic import (
            build_flex_flash_bwd_dkdv_module,
            build_flex_flash_bwd_dq_module,
        )
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (
            build_flex_flash_generic_module,
        )

        common = dict(
            num_heads=H,
            head_dim=D,
            num_kv_heads=H,
            dtype_str="bf16",
            sm_scale=1.0,
            num_aux_tensors=1,
            aux_specs=[[0, 1, 0, 0]],
            aux_numels=None,
        )
        for name, build in (
            ("forward", build_flex_flash_generic_module),
            ("dq", build_flex_flash_bwd_dq_module),
            ("dkdv", build_flex_flash_bwd_dkdv_module),
        ):
            with self.assertRaisesRegex(Exception, "aux_numels") as caught:
                build(**common)
            self.assertIn("read past the tensor", str(caught.exception), f"in {name}")

    def test_backward_matches_eager_with_an_additive_score_mod(self):
        """ALiBi: additive, so its joint graph is the identity.

        Which makes it the case a broken joint site would still pass, and why the
        soft-cap test below is the one that actually exercises the chain rule.
        """
        self._assert_grads_match_eager("score_mod=alibi", score_mod=_alibi)

    def test_backward_matches_eager_with_a_nonlinear_score_mod(self):
        """Soft-cap: the joint graph is ``grad * (1 - tanh²(s/cap))``, not the identity.

        This is the test with teeth. Drop the joint site entirely and the gradients come
        back wrong by exactly that factor -- plausible-looking numbers, no error -- which
        is why ``build_flex_flash_bwd_*`` refuses a ``score_mod`` without a ``joint_mod``
        rather than defaulting it to a passthrough.
        """
        self._assert_grads_match_eager("score_mod=softcap", score_mod=_softcap)

    @parametrize("mask_mod", [_causal, _sliding_window])
    def test_backward_matches_eager_with_a_mask_mod(self, mask_mod):
        """Causal is the case that matters: it is most of FlexAttention's traffic.

        A masked element gets ``p = 0`` and so contributes to no gradient, which is what
        makes this correct without a second site -- the joint graph is linear in its
        cotangent. The sliding window additionally masks on *both* sides, so it catches a
        mask read as a one-sided comparison.
        """
        self._assert_grads_match_eager(
            f"mask_mod={mask_mod.__name__}", mask_mod=mask_mod
        )

    def test_backward_matches_eager_with_both_mods(self):
        """A non-linear score_mod under a mask, which is where their order matters.

        score_mod runs first: a soft-cap applied *after* a mask would map -inf to -cap and
        resurrect the position with weight ~exp(-cap). Both backward kernels order these
        the way the forward does, and this is what would catch it if one stopped.
        """
        self._assert_grads_match_eager(
            "score_mod=softcap + causal", score_mod=_softcap, mask_mod=_causal
        )

    def test_backward_matches_eager_with_a_captured_tensor(self):
        """A mod reading a captured tensor. Refused is the *gradient* of one, not the read.

        The capture reaches both backward kernels through the same aux slots and the same
        stride specs as the forward, so a mod body lowered once runs unchanged at all
        three call sites.
        """
        bias = torch.randn(H, device="cuda", dtype=torch.float32)

        def head_bias(score, b, h, q_idx, kv_idx):
            return score + bias[h]

        self._assert_grads_match_eager("score_mod reads bias[h]", score_mod=head_bias)

    def test_backward_refuses_a_gradient_for_a_captured_tensor(self):
        """The one mod feature still missing, and it must refuse rather than fall back.

        A capture that requires grad needs the joint graph's ``zeros_and_scatter`` outputs
        accumulated with atomics. Falling back would pair this forward's natural-log LSE
        with Triton's log2 backward, so the refusal is loud. See Note [FlyDSL forward and
        backward must be chosen together].
        """
        bias = torch.randn(
            B, H, S, S, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        def learned_bias(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]

        with self.assertRaisesRegex(Exception, "captured buffers"):
            self._grads(score_mod=learned_bias)

    def test_backward_is_not_used_by_other_backends(self):
        """A Triton backward must still be reachable when FLYDSL was not asked for."""
        actual = self._grads(backend="TRITON")
        expected = self._grads(backend=None)
        for name, a, b in zip(("dq", "dk", "dv"), actual, expected):
            self._assert_close(a, b, f"for {name}")

    def test_other_backends_unaffected(self):
        q, k, v = self._tensors()
        actual, expected = self._run(
            q, k, v, score_mod=_alibi, kernel_options={"BACKEND": "TRITON"}
        )
        self._assert_close(actual, expected)

    def test_noop_mask_does_not_force_a_mask_mod(self):
        """A noop mask_mod is trivial, so it must not be lowered as a real one."""
        q, k, v = self._tensors()
        block_mask = create_block_mask(noop_mask, B, H, S, S, device="cuda")
        actual, expected = self._run(q, k, v, score_mod=_alibi, block_mask=block_mask)
        self._assert_close(actual, expected)

    @config.patch({"flydsl.autotune_mod_vec_size": False})
    def test_single_choice_without_autotuning(self):
        q, k, v = self._tensors()
        actual, expected = self._run(q, k, v, score_mod=_alibi)
        self._assert_close(actual, expected)

    @parametrize("depth", [1, 2, 3, 4, 5])
    def test_qk_prefetch_depth_agrees(self, depth):
        """Every prefetch depth must compute the same thing.

        Depth only moves when the K packs are read out of LDS relative to the MFMA chain,
        so it is a scheduling choice with no numerical content. Depths above 2 also emit a
        ``sched_group_barrier``; unhinted, a deep prefetch measured 6x slower on one shape,
        which is why the hint travels with the depth rather than being separately tunable.
        """
        torch.manual_seed(0)
        b, h, s, d = 2, 4, 512, 128
        q, k, v = (
            torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        launcher = build_flex_flash_generic_module(
            num_heads=h,
            head_dim=d,
            causal=False,
            dtype_str="bf16",
            return_lse=True,
            layout="bhsd",
            qk_prefetch_depth=depth,
        )
        self.assertEqual(launcher.qk_prefetch_depth, depth)

        out = torch.empty_like(q)
        lse = torch.empty((b, h, s), device="cuda", dtype=torch.float32)
        run, _, _ = prepare(launcher, q, k, v, out=out, lse=lse)
        run()
        torch.cuda.synchronize()

        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertLess(rel, 2e-2, f"depth {depth} disagrees with eager: {rel}")

    def test_qk_prefetch_depth_is_off_by_default(self):
        """The depth knob must not quietly multiply everyone's build count by four."""
        from torch._inductor.kernel.flex.flydsl_flash_attention import _qk_prefetch_depths

        with config.patch({"flydsl.autotune_qk_prefetch_depth": False}):
            self.assertEqual(_qk_prefetch_depths({}), [2])
        with config.patch({"flydsl.autotune_qk_prefetch_depth": True}):
            self.assertEqual(_qk_prefetch_depths({}), [2, 3, 4, 5])
        # An explicit request wins over either.
        with config.patch({"flydsl.autotune_qk_prefetch_depth": True}):
            self.assertEqual(_qk_prefetch_depths({"QK_PREFETCH_DEPTH": 3}), [3])
        with self.assertRaises(RuntimeError):
            _qk_prefetch_depths({"QK_PREFETCH_DEPTH": 0})

    def test_forward_q_tile_is_measured_where_the_default_reaches_for_256(self):
        """The 32-head default was a guess, and at head_dim 224/256 it cost 1.7-2.5x.

        The taller tile is only ever chosen at 32 heads or more, so that is the only place
        the two candidates differ and the only place worth paying a second build for. Below
        that the list must stay a single point: doubling the builds of the common 8-head
        case to re-answer a question every measurement agrees on is not free.
        """
        from torch._inductor.kernel.flex.flydsl_flash_attention import _forward_block_ms

        with config.patch({"flydsl.autotune_forward_block_m": True}):
            self.assertEqual(_forward_block_ms({}, num_heads=32), [128, 256])
            self.assertEqual(_forward_block_ms({}, num_heads=64), [128, 256])
            self.assertEqual(_forward_block_ms({}, num_heads=8), [128])
        # Off, both sides must reproduce the kernel's own default exactly, or an
        # autotune-off build silently changes tile.
        with config.patch({"flydsl.autotune_forward_block_m": False}):
            self.assertEqual(_forward_block_ms({}, num_heads=32), [256])
            self.assertEqual(_forward_block_ms({}, num_heads=8), [128])
        # An explicit request wins over either, including below 32 heads.
        with config.patch({"flydsl.autotune_forward_block_m": True}):
            self.assertEqual(_forward_block_ms({"BLOCK_M": 256}, num_heads=8), [256])
        for bad in (0, -128, 100, 1.5, "128"):
            with self.assertRaises(RuntimeError):
                _forward_block_ms({"BLOCK_M": bad}, num_heads=32)

    @parametrize("block_m", [128, 256])
    def test_forward_agrees_with_eager_at_either_q_tile(self, block_m):
        """Both swept tiles have to be answers, not just timings.

        The tile changes the grid, the LSE rows a workgroup owns, and the workgroup size
        (256 threads at 128 rows, 512 at 256), so a tile-dependent indexing bug would show
        up here as a wrong answer rather than as a slow one.
        """
        q, k, v = self._tensors()
        launcher = build_flex_flash_generic_module(
            num_heads=H,
            num_kv_heads=H,
            head_dim=D,
            causal=False,
            sm_scale=D**-0.5,
            dtype_str="bf16",
            return_lse=True,
            layout="bhsd",
            block_m=block_m,
        )
        self.assertEqual(launcher.q_block_size, block_m)

        out = torch.empty_like(q)
        lse = torch.empty((B, H, S), device="cuda", dtype=torch.float32)
        run, _, _ = prepare(launcher, q, k, v, out=out, lse=lse)
        run()
        torch.cuda.synchronize()

        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertLess(rel, 2e-2, f"BLOCK_M {block_m} disagrees with eager: {rel}")

    def test_both_kv_staging_strategies_are_offered(self):
        """Neither K staging strategy wins outright, so both have to reach the autotuner.

        Register-staged K wins 6-15% at head_dim 96/192/224 and loses 7-9% at 64 and 128
        dense, and nothing cheap about the shape says which case a given call is. Unlike
        the Q tile there is no head count to restrict the sweep to, so both points are
        always live -- and with the sweep off the list must still be the kernel's own
        default, or an autotune-off build silently changes staging.
        """
        from torch._inductor.kernel.flex.flydsl_flash_attention import (
            _forward_kv_gpfetch,
        )

        with config.patch({"flydsl.autotune_kv_gpfetch": True}):
            self.assertEqual(_forward_kv_gpfetch({}), [False, True])
        with config.patch({"flydsl.autotune_kv_gpfetch": False}):
            self.assertEqual(_forward_kv_gpfetch({}), [True])
        # An explicit request pins it either way, sweep on or off.
        for sweep in (True, False):
            with config.patch({"flydsl.autotune_kv_gpfetch": sweep}):
                self.assertEqual(_forward_kv_gpfetch({"ENABLE_KV_GPFETCH": False}), [False])
                self.assertEqual(_forward_kv_gpfetch({"ENABLE_KV_GPFETCH": True}), [True])

    @parametrize("gpfetch", [False, True])
    @parametrize("seq_len_kv", [S, S // 2 + 1])
    def test_forward_agrees_with_eager_under_either_kv_staging(self, seq_len_kv, gpfetch):
        """Staging K through registers must not change the answer, only the timing.

        The register path rewrites how a KV tile reaches LDS: the global read becomes
        unconditional and row-clamped with the tile guard moved to the store, and the loop
        carries the *next* tile's registers, which means it also computes a next-tile index
        one step past the walk. A ragged KV length exercises both -- the guard now decides
        what lands in LDS rather than what gets read, and the final iteration's lookahead
        runs off the end of the walk.
        """
        q, k, v = self._tensors(seq_len_kv=seq_len_kv)
        launcher = build_flex_flash_generic_module(
            num_heads=H,
            num_kv_heads=H,
            head_dim=D,
            causal=False,
            sm_scale=D**-0.5,
            dtype_str="bf16",
            return_lse=True,
            layout="bhsd",
            enable_kv_gpfetch=gpfetch,
        )
        run, out, _ = prepare(launcher, q, k, v)
        run()
        torch.cuda.synchronize()

        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertLess(rel, 2e-2, f"gpfetch={gpfetch} disagrees with eager: {rel}")

    @parametrize("zeroed", ["seq_len_q", "seq_len_kv"])
    def test_seq_lens_stay_at_their_positions_in_the_signature(self, zeroed):
        """The launcher reads both extents positionally, so nothing may shift them.

        Widening the aux slots moved the extent down the argument list once already, and
        splitting it into a Q and a KV one added a second position to get wrong. Getting
        either index wrong is quiet: it lands on a neighbouring argument, and `int()` of a
        1-element tensor succeeds, so the guard would compare against whatever happened to
        be there. Zero one extent at a time, leave every other slot non-zero, and require
        the error to name the one that was zeroed -- reading the wrong slot then cannot
        raise by accident, and cannot raise about the wrong axis either.
        """
        launcher = build_flex_flash_generic_module(
            num_heads=H,
            head_dim=64,
            causal=False,
            dtype_str="bf16",
            layout="bhsd",
        )
        filler = torch.full((1,), 5, device="cuda", dtype=torch.float32)
        lens = {"seq_len_q": 4, "seq_len_kv": 4} | {zeroed: 0}
        # Q, K, V, O, LSE, KV_NUM_BLOCKS, KV_INDICES, the aux slots, batch, then the two.
        args = (
            [filler] * (7 + launcher.max_aux_tensors)
            + [2]
            + [lens["seq_len_q"], lens["seq_len_kv"]]
        )

        # The guard raises before dispatch, so the kernel never runs with an extent of 0.
        with self.assertRaisesRegex(ValueError, f"{zeroed} must be >= 1"):
            launcher(*args)

    def test_v_swizzle_keeps_two_workgroups_per_cu(self):
        """The head_dim 128 tile must stay within half of a 64 KB CU.

        Occupancy here is decided by LDS, not registers: at 33280 B only one workgroup
        fits a CU, and the 512 B that made the difference was V's transpose padding.
        Swizzling V removed it, and the failure mode if someone reintroduces padding --
        or grows the tile -- is a quiet halving of occupancy that shows up only as a
        benchmark regression, so pin the number.
        """
        launcher = build_flex_flash_generic_module(
            num_heads=32,
            head_dim=128,
            causal=False,
            dtype_str="bf16",
            return_lse=True,
            layout="bhsd",
        )
        smem = launcher.smem_bytes
        self.assertLessEqual(
            smem,
            32768,
            f"LDS is {smem} B; over 32768 only one workgroup fits a 64 KB CU",
        )

    def test_kernels_do_not_import_buffer_ops_from_the_library(self):
        """`flydsl.expr.buffer_ops` exists in FlyDSL 0.2.4 and is gone in 0.3.1.

        Importing it directly does not fail loudly: the backend's import guard turns it
        into "flex unavailable", so every test here *skips* and the suite still reports
        green. That is how it went unnoticed. The kernels must go through
        `flex_kernels.buffer_ops`, which picks whichever copy the installed FlyDSL has.
        """
        import pathlib

        import torch._inductor.kernel.vendored_templates.flydsl as vendored

        root = pathlib.Path(vendored.__file__).parent
        offenders = []
        for path in root.rglob("*.py"):
            if path.name == "buffer_ops.py":
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "from flydsl.expr import" in stripped and "buffer_ops" in stripped:
                    offenders.append(f"{path.relative_to(root)}:{n}")
                elif "from flydsl.expr.buffer_ops" in stripped:
                    offenders.append(f"{path.relative_to(root)}:{n}")

        self.assertEqual(
            offenders,
            [],
            "import flex_kernels.buffer_ops instead; the library module is absent in "
            f"FlyDSL 0.3.1: {offenders}",
        )

    def test_layouts_agree(self):
        """The two global layouts must be two addressings of the same kernel.

        The flex path builds `layout="bhsd"` to index what FlexAttention hands it, while
        standalone callers still use `bshd`. Only the affine coefficients of the global
        address differ, so any divergence here is an addressing bug, not a numerics one.
        """
        torch.manual_seed(0)
        b, h, s, d = 2, 4, 256, 64
        bhsd = [
            torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        results = {}
        for layout in ("bshd", "bhsd"):
            launcher = build_flex_flash_generic_module(
                num_heads=h,
                head_dim=d,
                causal=False,
                dtype_str="bf16",
                return_lse=True,
                layout=layout,
            )
            tensors = (
                bhsd
                if layout == "bhsd"
                else [t.transpose(1, 2).contiguous() for t in bhsd]
            )
            out = torch.empty_like(tensors[0])
            lse = torch.empty((b, h, s), device="cuda", dtype=torch.float32)
            run, out, lse = prepare(launcher, *tensors, out=out, lse=lse)
            run()
            # bring bshd back to bhsd so the two are comparable
            results[layout] = (
                out if layout == "bhsd" else out.transpose(1, 2),
                lse,
            )

        self._assert_close(results["bshd"][0].contiguous(), results["bhsd"][0])
        self._assert_close(results["bshd"][1], results["bhsd"][1])

    def test_regrid_unions_lists_when_padding_collides(self):
        """A partial-list padding entry naming a block the full list visits must not drop it.

        `kv_indices` past `kv_num_blocks` holds real block ids, so the two lists collide on
        the same destination once they share one scatter. A plain `scatter_` of the validity
        mask would write whichever of True/False arrived last and could silently drop a
        visited block from the softmax, so the union is pinned here directly.
        """
        i32 = {"dtype": torch.int32, "device": "cuda"}
        # partial visits block 0; its padding names 2, 1, 3.
        partial_num = torch.tensor([[[1]]], **i32)
        partial_idx = torch.tensor([[[[0, 2, 1, 3]]]], **i32)
        # full visits block 2, which the partial padding also names.
        full_num = torch.tensor([[[1]]], **i32)
        full_idx = torch.tensor([[[[2, 0, 1, 3]]]], **i32)

        counts, indices = regrid_block_mask(
            partial_num,
            partial_idx,
            full_num,
            full_idx,
            batch=1,
            num_heads=1,
            sparse_q_block_size=128,
            sparse_kv_block_size=128,
            q_block_size=128,
            kv_block_size=64,
            num_q_tiles=1,
            num_kv_tiles=8,
        )

        # Blocks 0 and 2 at 128 wide are tiles 0,1 and 4,5 at 64 wide.
        self.assertEqual(int(counts[0, 0, 0]), 4)
        visited = sorted(int(x) for x in indices[0, 0, 0, :4])
        self.assertEqual(visited, [0, 1, 4, 5])

    def test_regrid_walk_q_transposes_the_same_occupancy(self):
        """``walk="q"`` must be the exact transpose of ``walk="kv"``, not a rebuild.

        The two backward kernels reduce over opposite axes, so one walks a KV list per Q
        tile and the other a Q list per KV tile. They are derived from one occupancy and
        transposed rather than from FlexAttention's own ``q_indices``, because a
        disagreement about which blocks exist would drop gradient contributions on one side
        only -- a wrong ``dk``/``dv`` with a right ``dq``, which is hard to read as a
        transpose bug.
        """
        i32 = {"dtype": torch.int32, "device": "cuda"}
        # A lower-triangular 4x4 mask: Q tile i visits KV tiles 0..i.
        num = torch.tensor([[[1, 2, 3, 4]]], **i32)
        idx = torch.tensor([[[[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 2, 0], [0, 1, 2, 3]]]], **i32)
        grid = {
            "batch": 1,
            "num_heads": 1,
            "sparse_q_block_size": 128,
            "sparse_kv_block_size": 128,
            "q_block_size": 128,
            "kv_block_size": 128,
            "num_q_tiles": 4,
            "num_kv_tiles": 4,
        }

        kv_counts, kv_indices = regrid_block_mask(num, idx, None, None, walk="kv", **grid)
        q_counts, q_indices = regrid_block_mask(num, idx, None, None, walk="q", **grid)

        # Transposed, the triangle inverts: KV tile j is visited by Q tiles j..3.
        self.assertEqual([int(x) for x in kv_counts[0, 0]], [1, 2, 3, 4])
        self.assertEqual([int(x) for x in q_counts[0, 0]], [4, 3, 2, 1])

        def visited(counts, indices):
            return {
                (row, int(b))
                for row in range(4)
                for b in indices[0, 0, row, : int(counts[0, 0, row])]
            }

        # The same set of (q, kv) pairs, read along either axis.
        self.assertEqual(
            visited(kv_counts, kv_indices),
            {(q, kv) for kv, q in visited(q_counts, q_indices)},
        )

    def test_backward_block_skip_is_gated_on_occupancy(self):
        """The gate, pinned: enough workgroups to fill the machine, or walk densely.

        Causal work per tile is triangular, so below a couple of workgroups per CU they are
        all resident and the kernel finishes when the *heaviest* one does -- which skipping
        does not move, while still paying the index loads. Measured 0.89-1.18x under the
        threshold against 1.54-2.39x over it.
        """
        from torch._inductor.kernel.flex.flydsl_flash_attention import (
            _BLOCK_SKIP_MIN_WGS_PER_CU,
            _enough_workgroups_to_skip,
        )

        cus = torch.cuda.get_device_properties("cuda").multi_processor_count
        enough = _BLOCK_SKIP_MIN_WGS_PER_CU * cus

        usable, reason = _enough_workgroups_to_skip(enough, "cuda")
        self.assertTrue(usable, reason)
        usable, reason = _enough_workgroups_to_skip(enough - 1, "cuda")
        self.assertFalse(usable)
        self.assertIn("imbalance", reason)

    @parametrize("mask_mod", [_causal, _sliding_window])
    def test_backward_block_skipping_matches_the_dense_walk(self, mask_mod):
        """Gradients under a mask big enough that both kernels *skip*, against eager.

        The other mask_mod backward tests run at the module's 2x4x512, which is 32
        workgroups and so under the occupancy gate -- they cover the dense walk with a
        per-element mask. This one uses 16 heads at 1024 to clear the gate, so the loop
        bounds come from the block lists instead. Skipping a block the mask did not prove
        empty loses that block's contribution silently, and only a shape on this side of
        the gate would show it.
        """
        self.assertGreaterEqual(
            B * -(-1024 // 128) * 16,
            2 * torch.cuda.get_device_properties("cuda").multi_processor_count,
            "shape no longer clears the occupancy gate, so this covers the dense walk",
        )
        self._assert_grads_match_eager(
            f"mask_mod={mask_mod.__name__} with block skipping",
            mask_mod=mask_mod,
            num_q_heads=16,
            num_kv_heads=16,
            seq_len=1024,
            head_dim=64,
        )

    def test_regrid_cache_respects_mutation(self):
        """The memo is keyed on tensor identity and version, so an in-place edit must miss."""
        i32 = {"dtype": torch.int32, "device": "cuda"}
        num = torch.tensor([[[1]]], **i32)
        idx = torch.tensor([[[[0, 1]]]], **i32)
        grid = {
            "batch": 1,
            "num_heads": 1,
            "sparse_q_block_size": 128,
            "sparse_kv_block_size": 128,
            "q_block_size": 128,
            "kv_block_size": 128,
            "num_q_tiles": 1,
            "num_kv_tiles": 2,
        }

        first = regrid_block_mask(num, idx, None, None, **grid)
        self.assertIs(regrid_block_mask(num, idx, None, None, **grid)[0], first[0])
        self.assertEqual(int(first[0][0, 0, 0]), 1)

        num.fill_(2)  # now visits both blocks; the stale memo would still say one
        second = regrid_block_mask(num, idx, None, None, **grid)
        self.assertIsNot(second[0], first[0])
        self.assertEqual(int(second[0][0, 0, 0]), 2)

    def test_regrid_caches_each_grid_separately(self):
        """One BlockMask is regridded to three grids, and they must not evict each other.

        The forward's tiles and the two backward walks all ask about the same mask at
        different granularities. With a single slot per mask they thrash and every step
        pays a full recompute -- correct answers, no error, just the sparsity win quietly
        spent on rebuilding the lists. So alternate two grids and require both to stay hot.
        """
        i32 = {"dtype": torch.int32, "device": "cuda"}
        num = torch.tensor([[[2]]], **i32)
        idx = torch.tensor([[[[0, 1]]]], **i32)
        base = {
            "batch": 1,
            "num_heads": 1,
            "sparse_q_block_size": 128,
            "sparse_kv_block_size": 128,
            "q_block_size": 128,
            "num_q_tiles": 1,
        }
        coarse = {**base, "kv_block_size": 128, "num_kv_tiles": 2}
        fine = {**base, "kv_block_size": 64, "num_kv_tiles": 4}

        first_coarse = regrid_block_mask(num, idx, None, None, **coarse)
        first_fine = regrid_block_mask(num, idx, None, None, **fine)
        # Asking for one must not have dropped the other.
        self.assertIs(regrid_block_mask(num, idx, None, None, **coarse)[0], first_coarse[0])
        self.assertIs(regrid_block_mask(num, idx, None, None, **fine)[0], first_fine[0])
        # And they are genuinely different lists, so this is not one entry serving both.
        self.assertEqual(int(first_coarse[0][0, 0, 0]), 2)
        self.assertEqual(int(first_fine[0][0, 0, 0]), 4)


instantiate_parametrized_tests(TestFlyDSLFlexAttention)


if __name__ == "__main__":
    run_tests()
