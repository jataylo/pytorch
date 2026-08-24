# Owner(s): ["module: inductor"]
"""End-to-end tests for BACKEND="FLYDSL" flex attention on ROCm.

Forward only, and only where the kernel is known to be correct: bf16/f16 at head_dim 128
on a validated architecture. Everything outside that has to fall back or refuse rather
than run, so the rejection cases are as much of the contract as the numerical ones -- the
kernel returns plausible-looking wrong numbers at head_dim 64 rather than failing.

The flex-attention kernel sources are vendored inside PyTorch; these tests need only the
optional FlyDSL compiler/runtime installed. They skip when FlyDSL is unavailable.
"""

import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    noop_mask,
)
from torch.testing._internal.common_utils import parametrize, run_tests
from torch.testing._internal.inductor_utils import HAS_GPU


try:
    from torch._inductor.kernel.flex.flydsl_flash_attention import (
        _arch_supported,
        flydsl_unavailable_reason,
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


@unittest.skipIf(SKIP_REASON is not None, f"FlyDSL flex unavailable: {SKIP_REASON}")
class TestFlyDSLFlexAttention(TestCase):
    def setUp(self):
        super().setUp()
        # Every test varies the mod or the shape on the same `flex_attention` code object,
        # so without a reset the later ones are rejected by Dynamo's recompile limit
        # rather than by anything under test.
        torch._dynamo.reset()

    def _tensors(self, dtype=torch.bfloat16, num_kv_heads=H, seq_len=S):
        torch.manual_seed(0)
        q = torch.randn(B, H, seq_len, D, device="cuda", dtype=dtype)
        k = torch.randn(B, num_kv_heads, seq_len, D, device="cuda", dtype=dtype)
        v = torch.randn(B, num_kv_heads, seq_len, D, device="cuda", dtype=dtype)
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

        The kernel's module docstring claims `seq_len % 128 == 0` but only enforces
        `seq_len >= 1`, which reads like the head_dim 64 defect waiting to happen. It is
        not: the ragged tail rows come out as accurate as the aligned ones, so the
        documented constraint is stale rather than unchecked. Tested because a documented
        constraint nobody enforces is worth pinning down in one direction or the other.
        """
        q, k, v = self._tensors(seq_len=seq_len)
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

        Worth its own test because the failure is silent: log2 would be off by a constant
        factor, which looks like a plausible LSE.
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

    def test_captured_view_built_inside_the_graph(self):
        """A capture that is a view produced by the traced graph, not a leaf tensor.

        These arrive as a ``ReinterpretView`` under a synthetic input name, so the node
        behind the name lives only in the graph's capture table. Resolving it is what lets
        the call site pass the view rather than its base buffer.
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

    def test_rejects_offset_capture_index(self):
        """The aux reader resolves strides only, so an offset index has no encoding."""
        q, k, v = self._tensors()
        table = torch.randn(S + 1, device="cuda", dtype=torch.float32) * 0.1

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + table[q_idx + 1]

        with self.assertRaisesRegex(Exception, "indexed with"):
            self._run(q, k, v, score_mod=score_mod)

    def test_rejects_unsupported_head_dim(self):
        """head_dim 64 builds and returns wrong numbers, so it must be refused."""
        torch.manual_seed(0)
        q, k, v = (
            torch.randn(B, H, S, 64, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        with self.assertRaisesRegex(Exception, "head_dim"):
            self._run(q, k, v, score_mod=_alibi)

    def test_rejects_float32(self):
        q, k, v = self._tensors(dtype=torch.float32)
        with self.assertRaisesRegex(Exception, "bf16/f16"):
            self._run(q, k, v, score_mod=_alibi)

    def test_falls_back_for_backward(self):
        """No FlyDSL backward kernel exists, so a graph needing grads must use Triton."""
        torch.manual_seed(0)
        q, k, v = (
            torch.randn(
                B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            for _ in range(3)
        )
        compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
        out = compiled(
            q, k, v, score_mod=_alibi, kernel_options={"BACKEND": "FLYDSL"}
        )
        out.sum().backward()
        self.assertIsNotNone(q.grad)
        with torch.no_grad():
            expected = flex_attention(q.detach(), k.detach(), v.detach(), score_mod=_alibi)
        self._assert_close(out.detach(), expected)

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


from torch.testing._internal.common_utils import instantiate_parametrized_tests


instantiate_parametrized_tests(TestFlyDSLFlexAttention)


if __name__ == "__main__":
    run_tests()
