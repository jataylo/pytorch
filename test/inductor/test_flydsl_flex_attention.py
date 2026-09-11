# Owner(s): ["module: inductor"]
"""End-to-end tests for BACKEND="FLYDSL" flex attention on ROCm.

Forward and backward, over the head dims and dtypes the lowering admits. The rejection
cases are as much of the contract as the numerical ones, because outside that range the
kernel returns plausible-looking wrong numbers rather than failing -- so what is refused,
and with which message, is tested as deliberately as what is computed.

Everything numerical here runs on the GPU that is installed. The one exception is
``test_gfx950_builds_from_a_gfx942_host``, which builds for another architecture and never
launches; see its docstring for why that boundary is load-bearing. Skipped unless the
optional FlyDSL compiler/runtime is installed.
"""

import contextlib
import io
import os
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

    def test_cpu_scalar_capture_is_copied_rather_than_refused(self):
        """A 0-d CPU capture would reach the kernel as a host pointer, so it is copied.

        This used to raise and tell the caller to "pass the value as a tensor on device
        instead" -- a workaround describing a four-byte copy the lowering can insert
        itself, which is what it now does. Closing it properly mattered because
        `torch.tensor(2.0)` with no `device=` is the natural way to write a scalar, and a
        mod that works under `BACKEND='TRITON'` failing here reads as a broken backend
        rather than as a deliberate limit.
        """
        q, k, v = self._tensors()
        cpu_scalar = torch.tensor(2.0)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * cpu_scalar

        actual, expected = self._run(q, k, v, score_mod=score_mod)
        self._assert_close(actual, expected)


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

    def test_a_mod_capturing_a_dynamic_shape_value_stays_correct_across_values(self):
        """A score_mod closing over `seq_len // 4` -- a sympy capture, not a tensor.

        The value is baked into the generated module as a constant, so one length proves
        nothing: the question is whether a *second* length recompiles or silently reuses
        the module with the old window in it. See Note [a symbolic capture specializes
        the kernel]. Three lengths, each implying a different window, in one process.

        This also covers Note [k and v agreeing is asserted, not proven]: dynamic=True
        gives k and v unrelated symbols, and the gate used to refuse that pair. Reaching
        an answer at all proves it did not, since BACKEND="FLYDSL" raises rather than
        falling through when it cannot serve a graph.
        """

        def fn(q, k, v):
            window = q.shape[-2] // 4

            def score_mod(score, b, h, q_idx, kv_idx):
                return torch.where(
                    q_idx - kv_idx < window,
                    score,
                    torch.full_like(score, -float("inf")),
                )

            return flex_attention(
                q, k, v, score_mod=score_mod, kernel_options={"BACKEND": "FLYDSL"}
            )

        compiled = torch.compile(fn, dynamic=True)
        with torch.no_grad():
            for seq_len in (512, 1024, 256):
                q, k, v = self._tensors(seq_len=seq_len)
                window = seq_len // 4

                def reference(score, b, h, q_idx, kv_idx, window=window):
                    return torch.where(
                        q_idx - kv_idx < window,
                        score,
                        torch.full_like(score, -float("inf")),
                    )

                expected = flex_attention(q, k, v, score_mod=reference)
                self._assert_close(
                    compiled(q, k, v), expected, f"for seq_len={seq_len}"
                )

    def test_rejects_a_broadcast_kv_batch_in_both_directions(self):
        """A `Bkv=1` key/value broadcast across the Q batch, which neither kernel serves.

        Checked at the gate, and asserted for the *forward* as well as the backward, since
        the forward is the direction that used to reach the kernel: it addresses k and v at
        the Q batch index, so `Bkv=1` reads off the end of the allocation and takes the
        process down with a GPU memory fault. A refusal here is the whole fix -- there is
        no wrong number to catch downstream, and a test that ran the kernel would abort the
        run rather than fail.
        """
        for grad in (False, True):
            q = torch.randn(
                4, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=grad
            )
            k, v = (
                torch.randn(
                    1, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=grad
                )
                for _ in range(2)
            )
            block_mask = create_block_mask(_causal, 4, None, S, S, device="cuda")
            with self.assertRaises(Exception):
                out = self._run(q, k, v, block_mask=block_mask)
                if grad:
                    out.sum().backward()

    @parametrize(
        "qk_head_dim,v_head_dim",
        [
            # exact / exact lane geometry, both orders
            (128, 64),
            (64, 128),
            (256, 128),
            (128, 256),
            # partial K / exact V
            (192, 128),
            (96, 64),
            # exact K / partial V -- the opposite asymmetry of the guard state
            (128, 96),
            # two *different* partial geometries in one kernel: 96 idles 8 lanes of 512,
            # 160 idles 12, so K's guard and V's disagree about which lanes are live
            (96, 160),
        ],
    )
    def test_asymmetric_head_dims_forward(self, qk_head_dim, v_head_dim):
        """`qk_head_dim != v_head_dim` through the forward.

        The two extents are independent because the score tile sits between the GEMMs:
        GEMM1 contracts over the QK extent to produce it and GEMM2 contracts over the *KV*
        axis to consume it, so the V extent only ever appears as GEMM2's free axis. What
        had to be split to get here was not the loop structure but everything that had been
        sized once and used for both tensors -- the KV token stride, the KV global index,
        the num_records bound, and the cooperative load geometry, whose lane-to-(row, col)
        map is a function of the row width.

        The pairs are chosen for the load geometry rather than for plausibility, because
        that is the part with two states. A head_dim whose lane count divides the workgroup
        loads whole rows; 96, 160, 192 and 224 do not, and idle their remainder. So the
        cases that matter are the *combinations*: partial K against exact V, exact K against
        partial V, and -- the one nothing else reaches -- two different partial geometries
        in the same kernel, where K's idle-lane predicate and V's disagree about which
        lanes are live. `(96, 160)` is that case.

        Both orders are covered for the same reason. `qk > v` is the MLA-shaped case and the
        one upstream supports, but `qk < v` exercises the opposite side of every bound, and
        a constant left un-split shows up in only one of the two.
        """
        torch.manual_seed(0)
        q = torch.randn(B, H, S, qk_head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, qk_head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, v_head_dim, device="cuda", dtype=torch.bfloat16)

        actual, expected = self._run(q, k, v)
        # The output takes its width from V, not from Q.
        self.assertEqual(tuple(actual.shape), (B, H, S, v_head_dim))
        self._assert_close(actual, expected, f"for qk={qk_head_dim} v={v_head_dim}")

    def test_asymmetric_head_dims_with_a_mod_and_a_block_mask(self):
        """The asymmetric path with the mod machinery live, not just the plain GEMMs.

        A score_mod reads the score tile and a mask_mod drives the block walk, and both sit
        on the seam between the two extents, so this is where a mod site that had picked up
        the wrong width would show. Sharing `_run` with the symmetric tests is the point:
        nothing about a mod should know the two dims differ.
        """
        torch.manual_seed(0)
        q = torch.randn(B, H, S, 192, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, 192, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, 128, device="cuda", dtype=torch.bfloat16)

        actual, expected = self._run(q, k, v, score_mod=_alibi)
        self._assert_close(actual, expected, "for an asymmetric score_mod")

        torch._dynamo.reset()
        block_mask = create_block_mask(_causal, B, H, S, S, device="cuda")
        actual, expected = self._run(q, k, v, block_mask=block_mask)
        self._assert_close(actual, expected, "for an asymmetric block mask")

    @parametrize(
        "qk_head_dim,v_head_dim",
        [
            # exact / exact lane geometry, both orders
            (128, 64),
            (64, 128),
            (256, 128),
            # partial one side, exact the other, and the reverse
            (192, 128),
            (128, 96),
            # two different partial geometries at once, as in the forward test
            (96, 160),
        ],
    )
    def test_asymmetric_head_dims_backward(self, qk_head_dim, v_head_dim):
        """`qk_head_dim != v_head_dim` through both backward kernels.

        The backward holds the same property as the forward -- the two extents are
        independent loop bounds -- but spread over four GEMMs rather than two, so there are
        four places to get it wrong rather than one. In `dq`, `sᵀ = k qᵀ` reduces over the
        QK extent while `dpᵀ = v doᵀ` reduces over the V extent; in `dkdv`, `dv` and `dk`
        take the two extents as their *free* axes. Each pair had been one loop with one
        bound. See Note [the two head dims are two independent loop bounds here too].

        Checking the gradients rather than the forward output is the point: `dk` is written
        at the QK extent and `dv` at the V extent, out of accumulator banks that are no
        longer the same length, so a mixed-up bound lands in a gradient and nowhere else.

        The pairs mirror the forward test's, and for the same reason -- the interesting
        variable is the cooperative load geometry, whose lane-to-(row, col) map is a
        function of the row width, so what matters is the combinations of exact and partial
        rather than the head dims themselves.
        """
        self._assert_grads_match_eager(
            f"qk={qk_head_dim} v={v_head_dim}",
            head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
        )

    def test_asymmetric_head_dims_backward_with_a_mod_and_gqa(self):
        """The asymmetric backward with the mod machinery and the GQA group both live.

        `dkdv` folds the GQA group into an unrolled loop around the Q walk, accumulating
        into one set of registers, and that loop now reloads two tiles of *different*
        widths per Q head. A `score_mod` puts the joint-mod site on the seam between the
        two extents at the same time.
        """
        self._assert_grads_match_eager(
            "an asymmetric score_mod under GQA",
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=192,
            v_head_dim=128,
            score_mod=_alibi,
        )

    def test_rejects_head_dim_past_the_lds_budget(self):
        """head_dim 288 needs 73728 B of LDS against gfx942's 65536 B.

        The kernel does not check this itself -- it surfaces as a compile-time
        ``local memory (73728) exceeds limit`` from the backend -- so the allowlist is
        what keeps it from getting that far.

        The allowlist is one list for every architecture, and gfx950's 163840 B would hold
        this. Widening it there is a real opportunity and not this test's business: the
        ladder's upper rungs are where the LDS-dependent bugs were, and none of them can be
        caught without the hardware to run them on.
        """
        q, k, v = self._tensors(head_dim=288)
        with self.assertRaisesRegex(Exception, "head_dim"):
            self._run(q, k, v, score_mod=_alibi)

    def test_a_fully_masked_row_gives_zero_output_and_minus_inf_lse(self):
        """The values the fast-math flags could otherwise be allowed to discard.

        A row whose every KV column is masked has no valid score, so flex_attention is
        defined to return 0 for it and -inf for its LSE. Getting there depends on two
        guards that only exist to keep an infinity from becoming a NaN: the running max is
        seeded at a finite floor rather than -inf, because (-inf) - (-inf) is NaN and would
        poison the row's accumulator for the rest of the walk; and the final reciprocal is
        skipped when l == 0, because o * rcp(0) is 0 * inf, also NaN.

        Both guards are dead code to a compiler that has been told infinities do not occur,
        which `FastMathFlags.nnan|ninf` and `no-nans-fp-math` do say -- see Note [the
        fast-math flags stop short of nnan and ninf]. So this test is not really about
        masking, it is the thing that fails if those flags come back: without the seed the
        output is NaN rather than 0, and without the l == 0 check it is NaN rather than 0
        again. A tolerance check on a whole tensor would not catch either, since a NaN in
        one row is not a large relative error, which is why the assertions here are exact.
        """
        # Row 0 attends to nothing; every other row attends to column 0 only. Keeping the
        # rest of the tile alive matters -- a kernel that got this right only by way of an
        # entirely empty walk would still be wrong for the mixed case, which is the one a
        # document mask or a sliding window actually produces.
        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx > 0) & (kv_idx == 0)

        q, k, v = self._tensors(seq_len=128)
        block_mask = create_block_mask(mask_mod, B, H, 128, 128, device="cuda")
        with torch.no_grad():
            compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
            out, lse = compiled(
                q,
                k,
                v,
                block_mask=block_mask,
                return_lse=True,
                kernel_options={"BACKEND": "FLYDSL"},
            )

        self.assertFalse(torch.isnan(out.float()).any(), "NaN anywhere in the output")
        self.assertFalse(torch.isnan(lse.float()).any(), "NaN anywhere in the LSE")
        # The masked row: exactly zero, and -inf rather than a large finite number.
        self.assertEqual(
            out[:, :, 0].float().abs().max().item(), 0.0, "masked row is not exactly zero"
        )
        self.assertTrue(
            torch.isneginf(lse[:, :, 0]).all(), f"masked row LSE is {lse[:, :, 0]}"
        )
        # The live rows attend to column 0 alone, so O is V's first row and the LSE is the
        # single score. Both finite, which is what rules out the guards having been applied
        # too widely.
        self.assertTrue(torch.isfinite(lse[:, :, 1:]).all(), "live rows lost their LSE")
        self._assert_close(
            out[:, :, 1:],
            v[:, :, 0:1].expand(-1, -1, 127, -1),
            "live rows should equal V's first row",
        )

    def test_gfx950_builds_from_a_gfx942_host(self):
        """Every kernel we admit builds for gfx950, checked from a gfx942 host.

        gfx950 is served without a flag but nothing here can run it, so this stands in for
        hardware. The arch is a real input to the builders rather than a label: it decides
        MFMA K=16, the transposing LDS read, the permlane O store, DMA-to-LDS and the LDS
        budget, and `FLYDSL_GPU_ARCH` moves it without a device. What that buys is the
        checks the builders make -- geometry, LDS, instruction selection -- across the
        whole head-dim ladder in both directions.

        Building is as far as this can go, and the line is not caution but containment.
        FlyDSL has no compile-without-dispatch path: `flyc.compile` launches once on the
        way to returning a callable, a gfx950 binary cannot be dispatched here, and the
        attempt invalidates the HIP context so that every later GPU call in the process
        fails with `invalid resource handle` -- one poisoned test would take the rest of
        the file with it. Nothing below reaches a launch. `isa_stats.py --arch gfx950`
        carries the same configs through to ISA, one subprocess per build.
        """
        from torch._inductor.kernel.flex.flydsl_flash_attention import (
            _SUPPORTED_HEAD_DIMS,
        )
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_bwd_generic import (
            build_flex_flash_bwd_dkdv_module,
            build_flex_flash_bwd_dq_module,
        )

        def build_all(head_dim):
            """The forward launcher, having also built both backward kernels."""
            with contextlib.redirect_stdout(io.StringIO()):
                forward = build_flex_flash_generic_module(
                    num_heads=H,
                    head_dim=head_dim,
                    causal=True,
                    dtype_str="bf16",
                    layout="bhsd",
                )
                for build in (
                    build_flex_flash_bwd_dq_module,
                    build_flex_flash_bwd_dkdv_module,
                ):
                    build(
                        num_heads=H,
                        head_dim=head_dim,
                        dtype_str="bf16",
                        layout="bhsd",
                    )
            return forward

        host = build_all(D)
        with mock.patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx950"}):
            target = {dim: build_all(dim) for dim in sorted(_SUPPORTED_HEAD_DIMS)}

        # Without this the test quietly degrades into building for gfx942 twice: the arch
        # arrives through the environment, and nothing else here would notice if that
        # stopped working. LDS is the cheapest published thing that moves with it -- the
        # forward at D128 causal takes 32768 B on gfx942 and 49152 B on gfx950, because
        # the CDNA4 path stages a third K buffer for the DMA prefetch.
        self.assertNotEqual(
            host.smem_bytes,
            target[D].smem_bytes,
            "gfx950 built the same LDS footprint as gfx942; FLYDSL_GPU_ARCH did not apply",
        )

    def test_gfx950_backward_stops_holding_transposed_copies(self):
        """The transposing LDS read retires the Kᵀ and Qᵀ/DOᵀ tiles, which is the point.

        Both backward kernels held a second copy of a tile purely so GEMM2 could read it
        with the index roles swapped. `ds_read_b64_tr_b16` reads the row-major copy in
        that layout, so the copies go and the footprint falls by a third in `dq` (three
        tiles to two) and a half in `dkdv` (four to two). That is the measurable claim
        here; the register numbers are in `isa_stats.py --backward --arch gfx950`.

        Checked through `smem_bytes` rather than by inspecting ISA because the footprint
        is what the tile filter spends and therefore what a wrong answer here would cost.
        """
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_bwd_generic import (
            build_flex_flash_bwd_dkdv_module,
            build_flex_flash_bwd_dq_module,
        )

        def footprints(head_dim):
            with contextlib.redirect_stdout(io.StringIO()):
                dq = build_flex_flash_bwd_dq_module(
                    num_heads=H, head_dim=head_dim, dtype_str="bf16",
                    layout="bhsd", block_n=32,
                )
                dkdv = build_flex_flash_bwd_dkdv_module(
                    num_heads=H, head_dim=head_dim, dtype_str="bf16",
                    layout="bhsd", block_m=32,
                )
            return dq.smem_bytes, dkdv.smem_bytes

        for head_dim in (64, 128, 256):
            host_dq, host_dkdv = footprints(head_dim)
            with mock.patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx950"}):
                tgt_dq, tgt_dkdv = footprints(head_dim)
            # dq drops one of three equally sized tiles, dkdv two of four.
            self.assertEqual(
                tgt_dq * 3, host_dq * 2, f"dq at head_dim {head_dim}"
            )
            self.assertEqual(
                tgt_dkdv * 2, host_dkdv, f"dkdv at head_dim {head_dim}"
            )

    def test_backward_fused_output_store_is_written_but_off(self):
        """It is declined on evidence, not missing, so both states have to build.

        The forward's `permlane32_swap` + `cvt_pk_bf16_f32` store applies verbatim to dq
        and dk/dv -- same accumulator map -- and halves the store count. The ISA says it
        also costs 4-9 VGPRs and more spilling in the kernels that already spill, trading a
        once-per-workgroup store against scratch traffic inside the loop, so the default is
        off. What this pins is that the switch is the only thing deciding it: off it must
        build everywhere including gfx942, and on it must still build for a gfx950 target,
        or the declined path would rot into an unbuildable one.
        """
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_bwd_generic import (
            build_flex_flash_bwd_dkdv_module,
            build_flex_flash_bwd_dq_module,
        )

        def build(arch, **kwargs):
            with mock.patch.dict(os.environ, {"FLYDSL_GPU_ARCH": arch}):
                with contextlib.redirect_stdout(io.StringIO()):
                    return [
                        build_flex_flash_bwd_dq_module(
                            num_heads=H, head_dim=D, dtype_str="bf16",
                            layout="bhsd", block_n=32, **kwargs,
                        ),
                        build_flex_flash_bwd_dkdv_module(
                            num_heads=H, head_dim=D, dtype_str="bf16",
                            layout="bhsd", block_m=32, **kwargs,
                        ),
                    ]

        # gfx942 has no such instruction, so asking for it must not change the build --
        # the capability gates it before the switch does.
        off = [m.smem_bytes for m in build("gfx942")]
        forced = [m.smem_bytes for m in build("gfx942", enable_permlane_store=True)]
        self.assertEqual(off, forced)
        # And on gfx950 both states lower.
        build("gfx950", enable_permlane_store=False)
        build("gfx950", enable_permlane_store=True)

    def test_gfx950_keeps_the_wider_backward_q_tile_all_the_way_up(self):
        """The saved LDS has to reach the tile filter, or it buys nothing.

        A q tile of 64 costs more than gfx942's 65536 B past head_dim 128, so the sweep
        there narrows to the kv tile alone. Halving the footprint keeps it affordable
        across the whole ladder -- and the filter has to know that, because a footprint
        computed for the wrong arch either withholds a tile the builder would take or
        offers one it refuses.
        """
        from torch._inductor.kernel.flex.flydsl_flash_attention import (
            _backward_tiles,
            _dkdv_lds_bytes,
        )

        with config.patch({"flydsl.autotune_backward_tile": True}):
            with mock.patch(
                "torch._inductor.kernel.flex.flydsl_flash_attention._current_arch",
                return_value="gfx942",
            ):
                self.assertEqual(_backward_tiles({}, 256, 256), [(128, 32), (256, 32)])
            with mock.patch(
                "torch._inductor.kernel.flex.flydsl_flash_attention._current_arch",
                return_value="gfx950",
            ):
                self.assertEqual(
                    _backward_tiles({}, 256, 256), [(128, 32), (256, 32), (128, 64)]
                )
        # And the footprint itself, which is what the filter spends: the transposed copies
        # are exactly the row-major ones again at a power-of-two head dim.
        with_copies = _dkdv_lds_bytes(32, 128, 128, transposed_copies=True)
        without = _dkdv_lds_bytes(32, 128, 128, transposed_copies=False)
        self.assertEqual(with_copies, 2 * without)
        # Default is the conservative one, so a caller that forgets cannot overcommit.
        self.assertEqual(_dkdv_lds_bytes(32, 128, 128), with_copies)

    def test_taller_forward_tile_pads_the_k_row_where_it_could_swizzle(self):
        """The swizzle holds at four waves and stops holding at eight.

        Padding a head_dim whose granule count is a power of two used to be pointless --
        the swizzle was free and did the same job. At a 256-row tile it is not: twice the
        waves are reading LDS at once and the measurement flips to the padding by
        10.8-19.8% across every shape and mask tried. So the choice is the tile height's
        as well as the granule count's, and the footprint is where that is visible.

        head_dim 256 is the exception and has to stay one: unpadded it sits at exactly the
        gfx942 budget, so the pad is dropped rather than allowed to fail the build.
        """
        from torch._inductor.kernel.vendored_templates.flydsl.flex_kernels.flex_flash_generic import (  # noqa: B950
            build_flex_flash_generic_module,
        )

        def lds(head_dim, block_m):
            return build_flex_flash_generic_module(
                num_heads=8,
                head_dim=head_dim,
                causal=False,
                sm_scale=head_dim**-0.5,
                dtype_str="bf16",
                layout="bhsd",
                block_m=block_m,
            ).smem_bytes

        # A padded K row is 8 elements wider, which is `block_n` * 8 * 2 B = 1024 B.
        for head_dim in (64, 128):
            self.assertEqual(lds(head_dim, 256) - lds(head_dim, 128), 1024)
        # Already padded at every tile height, because they cannot swizzle at all.
        for head_dim in (96, 160, 192, 224):
            self.assertEqual(lds(head_dim, 256), lds(head_dim, 128))
        # And the one that must decline it.
        self.assertEqual(lds(256, 256), lds(256, 128))
        self.assertEqual(lds(256, 256), 65536)

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

    def test_bshd_strided_inputs_are_recognised_under_gqa(self):
        """A GQA capture is still BSHD, at a different head count for k/v than for q.

        Worth its own case because the strides are checked per tensor against that tensor's
        own sizes: k and v carry `num_kv_heads`, so a check written against q's head count
        would call them dense when they are not, or the reverse.
        """
        from torch._inductor.utils import run_and_get_code

        num_kv_heads = H // 4
        torch.manual_seed(0)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        k, v = (
            torch.randn(B, S, num_kv_heads, D, device="cuda", dtype=torch.bfloat16)
            .transpose(1, 2)
            for _ in range(2)
        )
        with torch.no_grad():
            expected = flex_attention(q, k, v, score_mod=_alibi, enable_gqa=True)
            torch._dynamo.reset()
            compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
            actual, (code,) = run_and_get_code(
                compiled,
                q,
                k,
                v,
                score_mod=_alibi,
                enable_gqa=True,
                kernel_options={"BACKEND": "FLYDSL"},
            )
        self._assert_close(actual, expected)
        self.assertIn('layout="bshd"', code)

    def test_disagreeing_input_layouts_fall_back_to_bhsd(self):
        """One launcher addresses all three, so a tensor that is not BSHD decides for them.

        The fallback is BHSD, which costs the copies but is never wrong. Getting this
        backwards would index k and v as though they were strided the way q is, which is
        silently wrong numbers rather than a fault.
        """
        from torch._inductor.utils import run_and_get_code

        torch.manual_seed(0)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        k, v = (
            torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )
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
        self._assert_close(actual, expected)
        self.assertIn('layout="bhsd"', code)

    def _grads(
        self,
        *,
        num_q_heads=H,
        num_kv_heads=H,
        seq_len=S,
        seq_len_kv=None,
        head_dim=D,
        v_head_dim=None,
        score_mod=None,
        mask_mod=None,
        backend="FLYDSL",
        kernel_options=None,
    ):
        """Run forward+backward once through ``backend``, returning (dq, dk, dv)."""
        torch.manual_seed(0)
        gqa = num_q_heads != num_kv_heads
        seq_len_kv = seq_len if seq_len_kv is None else seq_len_kv
        # V and grad_out take the V extent; Q, K and grad_q take the QK extent.
        v_head_dim = head_dim if v_head_dim is None else v_head_dim

        def _rand(heads, grad, rows=seq_len, width=head_dim):
            return torch.randn(
                B,
                heads,
                rows,
                width,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=grad,
            )

        q = _rand(num_q_heads, True)
        k = _rand(num_kv_heads, True, seq_len_kv)
        v = _rand(num_kv_heads, True, seq_len_kv, width=v_head_dim)
        grad_out = _rand(num_q_heads, False, width=v_head_dim)

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

    def _lse_grads(
        self,
        backend,
        *,
        score_mod=None,
        mask_mod=None,
        num_heads=H,
        seq_len=S,
        head_dim=D,
    ):
        """Gradients of a loss that reads the LSE as well as the output.

        Both terms are in the loss, so a dropped `dlse` shows up as a wrong gradient
        rather than as one that is merely scaled.
        """
        torch.manual_seed(0)
        q, k, v = (
            torch.randn(
                B,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        )
        kwargs = {
            "score_mod": score_mod,
            "block_mask": (
                create_block_mask(
                    mask_mod, None, None, seq_len, seq_len, device="cuda"
                )
                if mask_mod is not None
                else None
            ),
        }
        if backend is None:
            out, lse = flex_attention(q, k, v, return_lse=True, **kwargs)
        else:
            torch._dynamo.reset()
            compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
            out, lse = compiled(
                q,
                k,
                v,
                return_lse=True,
                kernel_options={"BACKEND": backend},
                **kwargs,
            )
        (out.float().sum() + lse.float().sum()).backward()
        return q.grad, k.grad, v.grad

    def _assert_lse_grads_match_eager(self, msg, **kwargs):
        actual = self._lse_grads("FLYDSL", **kwargs)
        expected = self._lse_grads(None, **kwargs)
        for name, a, b in zip(("dq", "dk", "dv"), actual, expected):
            self._assert_close(a, b, f"for {name}, {msg}")

    @parametrize("score_mod", [None, _alibi])
    @parametrize("mask_mod", [None, _causal])
    def test_lse_gradient_matches_eager(self, score_mod, mask_mod):
        """A loss that differentiates the returned LSE.

        `lse` is a log-sum-exp over the *post*-mod scores, so `d lse / d s` is `p` and the
        term folds into the row scalar the softmax derivative already subtracts. Both mods
        are swept because that folding happens before the joint graph runs: get the sign or
        the site wrong and a `score_mod` would push the wrong cotangent through its chain
        rule, which the plain case cannot see.
        """
        self._assert_lse_grads_match_eager(
            f"score_mod={score_mod is not None}, mask_mod={mask_mod is not None}",
            score_mod=score_mod,
            mask_mod=mask_mod,
        )

    def test_lse_gradient_under_block_skipping(self):
        """The LSE gradient where the loop bounds come from the block lists.

        The `dlse` plane is an extra declared input, and both the kernel signature and the
        positions `input_gen_fns` addresses shift by one when it is present -- so the list
        tensors are exactly what a wrong offset would corrupt. Same shape and same gate
        assertion as `test_backward_block_skipping_matches_the_dense_walk`, for the same
        reason: at the module's default shape neither kernel skips and this covers nothing.
        """
        self.assertGreaterEqual(
            B * -(-1024 // 128) * 16,
            2 * torch.cuda.get_device_properties("cuda").multi_processor_count,
            "shape no longer clears the occupancy gate, so this covers the dense walk",
        )
        self._assert_lse_grads_match_eager(
            "with block skipping",
            mask_mod=_causal,
            num_heads=16,
            seq_len=1024,
        )

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

        The taller tile used to be worth a second build only at 32 heads or more, and
        below that the list stayed a single point: doubling the builds of the common
        8-head case to re-answer a settled question is not free. The row padding at a
        256-row tile unsettled that at head_dim 128, which now sweeps at any head count.
        Padding alone is not the condition -- head_dim 64 is padded at the taller tile
        too, and the 128-row tile still wins every cell there below 32 heads.
        """
        from torch._inductor.kernel.flex.flydsl_flash_attention import _forward_block_ms

        def tiles(opts=None, *, num_heads=32, seq_len_q=S, head_dim=192):
            return _forward_block_ms(
                opts or {},
                num_heads=num_heads,
                seq_len_q=seq_len_q,
                head_dim=head_dim,
            )

        with config.patch({"flydsl.autotune_forward_block_m": True}):
            self.assertEqual(tiles(num_heads=32), [128, 256])
            self.assertEqual(tiles(num_heads=64), [128, 256])
            self.assertEqual(tiles(num_heads=8), [128])
            # head_dim 128 sweeps even at 8 heads, because that is where the padding
            # changed which tile wins. 64 is padded too and did not change, so it keeps
            # the shortcut rather than paying for a build that loses every cell.
            self.assertEqual(tiles(num_heads=8, head_dim=128), [128, 256])
            for head_dim in (64, 96, 160, 192, 224, 256):
                self.assertEqual(tiles(num_heads=8, head_dim=head_dim), [128])
                self.assertEqual(tiles(num_heads=32, head_dim=head_dim), [128, 256])
        # Off, both sides must reproduce the kernel's own default exactly, or an
        # autotune-off build silently changes tile.
        with config.patch({"flydsl.autotune_forward_block_m": False}):
            self.assertEqual(tiles(num_heads=32), [256])
            self.assertEqual(tiles(num_heads=8), [128])
        # An explicit request wins over either, including below 32 heads.
        with config.patch({"flydsl.autotune_forward_block_m": True}):
            self.assertEqual(tiles({"BLOCK_M": 256}, num_heads=8), [256])
        for bad in (0, -128, 100, 1.5, "128"):
            with self.assertRaises(RuntimeError):
                tiles({"BLOCK_M": bad})

        # A short Q sequence takes the shortest expressible tile instead, at any head
        # count and with the sweep either way: there is no KV walk to amortise over rows
        # that do not exist, so the padding is the cost. 64 rather than 32 because a wave
        # owns 32 rows and FlyDSL will not take a 64-thread workgroup.
        for autotune in (True, False):
            with config.patch({"flydsl.autotune_forward_block_m": autotune}):
                for heads in (8, 32):
                    for sq in (1, 4, 64):
                        self.assertEqual(tiles(num_heads=heads, seq_len_q=sq), [64])
                # 65 rows is past the floor and back on the normal rule.
                self.assertNotEqual(tiles(num_heads=32, seq_len_q=65), [64])
        # Still overridable there, or the knob would be a lie on decode shapes.
        self.assertEqual(tiles({"BLOCK_M": 128}, seq_len_q=1), [128])

    @parametrize("block_m", [64, 128, 256])
    def test_forward_agrees_with_eager_at_either_q_tile(self, block_m):
        """Every offered tile has to be an answer, not just a timing.

        The tile changes the grid, the LSE rows a workgroup owns, and the workgroup size
        (128 threads at 64 rows, 256 at 128, 512 at 256), so a tile-dependent indexing bug
        would show up here as a wrong answer rather than as a slow one. 64 is here because
        a decode shape now selects it, and it is the one height whose workgroup the builder
        had never been asked to derive.
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

    def test_gqa_packing_is_asked_only_where_it_is_legal_and_pays(self):
        """Packing is free where it does not help, so the rule is legality plus regime.

        Legality is three conditions: there has to be a group to pack, it has to divide
        the tile, and a BlockMask is regridded per ``(b, h, q_tile)`` and so cannot
        describe a tile spanning heads. The regime is the short Q sequence -- at prefill
        the packed and unpacked grids are the same size, so packing has nothing to win and
        the prefill path is the tuned one.
        """
        from torch._inductor.kernel.flex.flydsl_flash_attention import _pack_gqa

        def packed(opts=None, *, num_heads=32, num_kv_heads=8, block_m=64,
                   seq_len_q=1, use_block_mask=False):
            return _pack_gqa(
                opts or {},
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                block_m=block_m,
                seq_len_q=seq_len_q,
                use_block_mask=use_block_mask,
            )

        self.assertTrue(packed())
        self.assertTrue(packed(block_m=128))
        self.assertTrue(packed(num_kv_heads=1))  # MQA is a group of 32
        self.assertTrue(packed(seq_len_q=64))
        # Past the short-sequence floor, and where there is no group to pack.
        self.assertFalse(packed(seq_len_q=65))
        self.assertFalse(packed(num_kv_heads=32))
        # A group of 3 does not divide a 64-row tile, and every tile height the forward
        # offers is a multiple of 64, so these shapes simply do not pack.
        self.assertFalse(packed(num_heads=12, num_kv_heads=4))
        self.assertFalse(packed(use_block_mask=True))
        # Overridable, but not into an illegal build: the kernel would raise anyway, and
        # a refusal naming the reason beats a template failure.
        self.assertTrue(packed({"PACK_GQA": True}, seq_len_q=4096))
        self.assertFalse(packed({"PACK_GQA": False}))
        for illegal in (
            {"num_kv_heads": 32},
            {"num_heads": 12, "num_kv_heads": 4},
            {"use_block_mask": True},
        ):
            with self.assertRaises(RuntimeError):
                packed({"PACK_GQA": True}, **illegal)

    @parametrize(
        "num_heads,num_kv_heads,seq_len_q,block_m",
        [
            (8, 2, 1, 64),     # decode, group 4
            (8, 2, 3, 64),     # a partial packed tile
            (8, 2, 17, 64),    # rows spilling past one tile
            (8, 1, 5, 64),     # MQA, group 8
            (8, 2, 129, 64),   # many packed tiles
            (8, 2, 200, 128),  # taller tile
            (12, 6, 7, 64),    # group 2 at a head count that is not a power of two
        ],
    )
    def test_packed_gqa_matches_the_unpacked_kernel_bit_for_bit(
        self, num_heads, num_kv_heads, seq_len_q, block_m
    ):
        """Packing reorders which rows share a tile and must change nothing else.

        It is a pure remapping -- every lane still owns one Q row and the group already
        shared the KV head -- so the bar is equality with the unpacked kernel rather than
        a tolerance against eager. That is what catches the two things that do not follow
        from the remapping: the O store's new row predicate, which replaces a num_records
        bound that a packed row past seq_len no longer trips, and the num_records bound
        itself, which has to be taken at the last head of the group to stay uniform.
        """
        torch.manual_seed(0)
        head_dim = 64
        q = torch.randn(B, num_heads, seq_len_q, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, num_kv_heads, S, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, num_kv_heads, S, head_dim, device="cuda", dtype=torch.bfloat16)

        results = {}
        for pack in (False, True):
            launcher = build_flex_flash_generic_module(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                causal=False,
                sm_scale=head_dim**-0.5,
                dtype_str="bf16",
                return_lse=True,
                layout="bhsd",
                block_m=block_m,
                pack_gqa=pack,
            )
            out = torch.empty_like(q)
            lse = torch.empty(
                (B, num_heads, seq_len_q), device="cuda", dtype=torch.float32
            )
            run, _, _ = prepare(launcher, q, k, v, out=out, lse=lse)
            run()
            torch.cuda.synchronize()
            results[pack] = (out.clone(), lse.clone())

        self.assertEqual(results[True][0], results[False][0], atol=0, rtol=0)
        self.assertEqual(results[True][1], results[False][1], atol=0, rtol=0)
        # And the unpacked kernel is itself right, so equality is not two matching bugs.
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        rel = (
            (results[True][0].float() - ref.float()).norm() / ref.float().norm()
        ).item()
        self.assertLess(rel, 2e-2)

    def test_packing_refuses_what_it_cannot_express(self):
        """The builder's own guards, which the lowering is trusted not to reach."""
        def build(**kwargs):
            opts = dict(
                num_heads=8,
                num_kv_heads=2,
                head_dim=64,
                causal=False,
                sm_scale=D**-0.5,
                dtype_str="bf16",
                layout="bhsd",
                block_m=64,
                pack_gqa=True,
            )
            opts.update(kwargs)
            return build_flex_flash_generic_module(**opts)

        with self.assertRaisesRegex(ValueError, "needs a GQA group"):
            build(num_kv_heads=8)
        with self.assertRaisesRegex(ValueError, "divisible by the GQA group"):
            build(num_heads=12, num_kv_heads=4)
        # A block-masked build needs a mask_mod of its own before it gets this far.
        with self.assertRaisesRegex(ValueError, "cannot be combined with block_mask"):
            build(block_mask=True, mask_mod=_causal, mod_key="causal")

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

    def test_v_swizzle_keeps_eight_waves_per_cu_at_head_dim_128(self):
        """The head_dim 128 tile must keep eight waves resident on a 64 KB CU.

        Occupancy here is decided by LDS, not registers: at 33280 B only one workgroup
        fits a CU, and the 512 B that made the difference was V's transpose padding.
        Swizzling V removed it, and the failure mode if someone reintroduces padding --
        or grows the tile -- is a quiet halving of occupancy that shows up only as a
        benchmark regression, so pin the number.

        Pinned in *waves* rather than workgroups because the K row padding at a 256-row
        tile spends 1024 B and does drop that tile to one workgroup per CU. It costs no
        occupancy doing it: a 256-row workgroup is eight waves where a 128-row one is
        four, so one of the former is exactly two of the latter, and both land on eight.
        The byte count is the thing that must not creep; the workgroup count was only ever
        a proxy for it.
        """

        def waves_per_cu(block_m):
            launcher = build_flex_flash_generic_module(
                num_heads=32,
                head_dim=128,
                causal=False,
                dtype_str="bf16",
                return_lse=True,
                layout="bhsd",
                block_m=block_m,
            )
            smem = launcher.smem_bytes
            self.assertLessEqual(
                smem,
                65536 // 2 + (1024 if block_m == 256 else 0),
                f"LDS is {smem} B at block_m {block_m}, which is over budget",
            )
            return (block_m // 32) * (65536 // smem)

        for block_m in (128, 256):
            self.assertEqual(waves_per_cu(block_m), 8)

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
