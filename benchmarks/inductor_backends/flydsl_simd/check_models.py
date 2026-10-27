"""Numerical check of the model suite: FlyDSL and Triton against eager.

For each model, the same seeded model runs one step eagerly, compiled with Triton, and
compiled with FlyDSL. Training models compare the loss and every parameter's gradient
(the optimizer step is skipped: AdamW's first update is about sign(grad), so it hides
a gradient off by a constant factor); inference models compare their outputs. Reported
is the worst relative error ||compiled - eager|| / ||eager|| over the compared tensors,
for each backend, so FlyDSL can be held to Triton's own deviation from eager.

    python check_models.py [--only MODEL ...] [--factor 4]

Exits non-zero if FlyDSL's error exceeds ``factor`` times Triton's (plus a small floor).
"""

import argparse
import sys

import torch
import torch._inductor.config as ic

import models


def _grad_step(model, loss_fn, opt="adamw"):
    params = list(model.parameters())
    names = {id(p): n for n, p in model.named_parameters()}

    def step():
        for p in params:
            p.grad = None
        loss = loss_fn()
        loss.backward()
        out = {"loss": loss.detach().float()}
        for p in params:
            if p.grad is not None:
                out[names[id(p)]] = p.grad.detach().float().clone()
        return out

    return step


models._train = _grad_step


def _run(name: str, backend: str) -> dict:
    torch._dynamo.reset()
    ic.flydsl.simd = backend == "flydsl"
    torch.manual_seed(0)
    step = models.ALL_MODELS[name]()
    if backend != "eager":
        step = torch.compile(step)
    torch.manual_seed(1)
    out = step()
    if isinstance(out, torch.Tensor):
        out = {"out": out.detach().float()}
    elif isinstance(out, (tuple, list)):
        out = {f"out{i}": o.detach().float() for i, o in enumerate(out)}
    torch.cuda.synchronize()
    return out


def _worst(got: dict, ref: dict) -> tuple[float, str]:
    worst, where = 0.0, ""
    for k, r in ref.items():
        g = got.get(k)
        if g is None:
            return float("inf"), f"{k} missing"
        den = r.norm().item()
        err = (g - r).norm().item() / (den if den > 0 else 1.0)
        if not err <= worst:  # NaN counts as worst
            worst, where = err, k
    return worst, where


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--factor", type=float, default=4.0)
    a = p.parse_args()
    names = [n for n in (a.only or models.MODELS)]
    bad = []
    print(f"{'model':20s} {'triton err':>11s} {'flydsl err':>11s}  worst flydsl tensor")
    for n in names:
        ref = _run(n, "eager")
        t_err, _ = _worst(_run(n, "triton"), ref)
        f_err, where = _worst(_run(n, "flydsl"), ref)
        ok = f_err <= a.factor * t_err + 1e-3
        print(f"{n:20s} {t_err:11.2e} {f_err:11.2e}  {where}{'' if ok else '  <-- FAIL'}", flush=True)
        if not ok:
            bad.append(n)
    if bad:
        print("FAILED:", " ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
