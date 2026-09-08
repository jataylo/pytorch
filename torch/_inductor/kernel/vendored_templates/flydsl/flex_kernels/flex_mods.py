# SPDX-License-Identifier: Apache-2.0
"""Reusable FlexAttention-style score/mask mods written in FlyDSL device values.

These are hand-written, but the signature and the value vocabulary (operators and
`fmath` calls on `ArithValue`) are exactly what Inductor's `modification()` hook would
emit from a lowered FX subgraph, so this doubles as the contract the lowering has to
target.

Two ABIs exist, selected by the kernel's `mod_vec_size`:

- ``mod_vec_size == 1``: ``score``/``kv_idx`` are scalars.
- ``mod_vec_size > 1``:  ``score``/``kv_idx`` are lists of that many contiguous-in-kv
  values, and the mod returns a list.

`@elementwise` / `@elementwise_mask` lift a scalar-shaped body to both, so a mod only
has to be written once. Note that lifting alone buys nothing at vec_size > 1 beyond
what LLVM's CSE already does — the real saving is a vectorised aux-tensor read, which
is why `aux_bias` uses ``aux[0].vec(...)`` when it can.
"""

from __future__ import annotations

import math

import flydsl.expr as fx
from flydsl.expr.utils.arith import ArithValue

_LOG2E = math.log2(math.e)


# ABI adapters


def elementwise(fn):
    """Lift a scalar score_mod body to also accept the vectorised list ABI."""

    def wrapper(score, b, h, q_idx, kv_idx, **kw):
        if isinstance(score, (list, tuple)):
            return [fn(s, b, h, q_idx, kv, **kw) for s, kv in zip(score, kv_idx)]
        return fn(score, b, h, q_idx, kv_idx, **kw)

    wrapper.__name__ = getattr(fn, "__name__", "mod")
    return wrapper


def elementwise_mask(fn):
    """Lift a scalar mask_mod body to also accept the vectorised list ABI."""

    def wrapper(b, h, q_idx, kv_idx, **kw):
        if isinstance(kv_idx, (list, tuple)):
            return [fn(b, h, q_idx, kv, **kw) for kv in kv_idx]
        return fn(b, h, q_idx, kv_idx, **kw)

    wrapper.__name__ = getattr(fn, "__name__", "mask_mod")
    return wrapper


def elementwise_joint(fn):
    """Lift a scalar joint_mod body to also accept the vectorised list ABI.

    A ``joint_mod`` is the chain rule through a ``score_mod``: given the *pre*-mod score
    and the gradient with respect to the *post*-mod score, it returns the gradient with
    respect to the pre-mod score. Only the backward kernels call it -- there is no forward
    equivalent -- and it is what makes a non-additive mod like a soft-cap differentiable
    rather than silently wrong by a factor of the mod's derivative.
    """

    def wrapper(score, b, h, q_idx, kv_idx, grad, **kw):
        if isinstance(score, (list, tuple)):
            return [
                fn(s, b, h, q_idx, kv, g, **kw) for s, kv, g in zip(score, kv_idx, grad)
            ]
        return fn(score, b, h, q_idx, kv_idx, grad, **kw)

    wrapper.__name__ = getattr(fn, "__name__", "joint_mod")
    return wrapper


# Math helpers


def tanh(x):
    """tanh via exp2.

    `math.tanh` does not lower on AMDGPU here ("no libcall available for ftanh"), so
    any soft-cap mod has to be expanded by hand. Uses |x| to keep exp2 from
    overflowing, then restores the sign:
        tanh(x) = sign(x) * (1 - 2 / (exp2(2*|x|*log2e) + 1))
    """
    x = ArithValue(x)
    neg = x < fx.Float32(0.0)
    ax = ArithValue(neg).select(-x, x)
    e = ArithValue(ax * fx.Float32(2.0 * _LOG2E)).exp2()
    t = fx.Float32(1.0) - fx.Float32(2.0) / (ArithValue(e) + fx.Float32(1.0))
    return ArithValue(neg).select(-ArithValue(t), ArithValue(t))


# Score mods


@elementwise
def identity(score, b, h, q_idx, kv_idx):
    """Isolates the cost of the hook itself from the cost of any mod body."""
    return score


def make_alibi(head_coeff=0.125, batch_coeff=0.03125):
    """ALiBi: a per-head linear penalty on query-key distance.

    Depends on all four coordinates and varies along kv_idx, which is what makes it a
    usable correctness test: softmax is shift-invariant per row, so a mod contributing
    only a per-row constant would be invisible in the output and a wrong coordinate
    derivation could still "pass".
    """

    @elementwise
    def alibi(score, b, h, q_idx, kv_idx):
        slope = fx.Float32(head_coeff) * (fx.Float32(h) + fx.Float32(1.0)) + fx.Float32(batch_coeff) * fx.Float32(b)
        return score + slope * (fx.Float32(kv_idx) - fx.Float32(q_idx))

    return alibi


def make_softcap(cap_value):
    """Gemma-2 style soft-cap: squash scores smoothly into +/-cap.

    `cap_value` is captured as a closure cell on purpose. FlyDSL's JIT cache key hashes
    traced source plus recursively-collected *scalar closure values*, so a constant read
    from a module global is invisible to the key: two soft-caps differing only in that
    constant collide and the stale binary is silently reused.
    """

    @elementwise
    def softcap(score, b, h, q_idx, kv_idx):
        cap = fx.Float32(cap_value)
        return cap * tanh(ArithValue(score) / cap)

    return softcap


def make_aux_bias(scale=1.0):
    """score + scale * bias[b, h, q_idx, kv_idx], reading a captured tensor.

    Uses the vectorised reader when the kernel hands it a group, which turns n dword
    loads into one dwordx2/x4 -- the actual payoff of mod_vec_size > 1.
    """

    def aux_bias(score, b, h, q_idx, kv_idx, aux=None):
        reader = aux[0]
        if isinstance(score, (list, tuple)):
            biases = reader.vec(b, h, q_idx, kv_idx[0], len(score))
            return [s + fx.Float32(scale) * bias for s, bias in zip(score, biases)]
        return score + fx.Float32(scale) * reader(b, h, q_idx, kv_idx)

    return aux_bias


# Mask mods (True keeps the element)


@elementwise_mask
def causal_mask(b, h, q_idx, kv_idx):
    return ArithValue(fx.Int32(kv_idx) <= fx.Int32(q_idx))


def make_sliding_window_mask(window):
    """Causal band of `window` keys. The case where block-skipping pays off most:
    the visited block count stops growing with sequence length."""

    @elementwise_mask
    def sliding_window(b, h, q_idx, kv_idx):
        kv = fx.Int32(kv_idx)
        q = fx.Int32(q_idx)
        return ArithValue(kv <= q) & ArithValue(kv > q - fx.Int32(window))

    return sliding_window


def make_prefix_lm_mask(prefix_len):
    """Bidirectional inside the first `prefix_len` tokens, causal after."""

    @elementwise_mask
    def prefix_lm(b, h, q_idx, kv_idx):
        kv = fx.Int32(kv_idx)
        return ArithValue(kv < fx.Int32(prefix_len)) | ArithValue(kv <= fx.Int32(q_idx))

    return prefix_lm
