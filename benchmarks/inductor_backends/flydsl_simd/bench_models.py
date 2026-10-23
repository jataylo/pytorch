"""Whole models: Inductor with FlyDSL pointwise/reduction kernels vs Triton.

For each model and backend this runs a fresh child process twice -- once with empty
Inductor, Triton and FlyDSL caches (cold compile) and once reusing them (warm) -- and
reports time to the first result, then the steady-state step time and the FlyDSL kernel
and fallback counts. A third run compiles with ``mode="reduce-overhead"`` (CUDA graphs),
which removes per-launch CPU cost and leaves the kernels' GPU time. Matmuls and convolutions are library calls under either backend, so
the difference in step time is entirely the generated pointwise/reduction kernels.

    python benchmarks/inductor_backends/flydsl_simd/bench_models.py [--only NAME ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile


CHILD = r"""
import json, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import torch._inductor.config as ic
from torch._dynamo.utils import counters

name, backend, mode = sys.argv[1], sys.argv[2], sys.argv[3]
ic.flydsl.simd = backend == "flydsl"
mode = None if mode == "default" else mode


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) * (D // self.h) ** -0.5
        att = att.masked_fill(torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1), float("-inf"))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, vocab=32000, d=768, h=12, layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.Sequential(*[Block(d, h) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        return self.head(self.ln(self.blocks(self.emb(idx))))


class ResNetish(nn.Module):
    def __init__(self, c=64, blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.layers = nn.ModuleList()
        for _ in range(blocks):
            self.layers.append(nn.ModuleList([nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c),
                                              nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c)]))
        self.fc = nn.Linear(c, 10)

    def forward(self, x):
        x = F.relu(self.stem(x))
        for c1, b1, c2, b2 in self.layers:
            x = F.relu(x + b2(c2(F.relu(b1(c1(x))))))
        return self.fc(x.mean((2, 3)))


torch.manual_seed(0)
dev = "cuda"
if name == "gpt_train_bf16":
    model = GPT().to(dev).to(torch.bfloat16)
    x = torch.randint(0, 32000, (8, 512), device=dev)
    tgt = torch.randint(0, 32000, (8, 512), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    def step():
        loss = F.cross_entropy(model(x).float().view(-1, 32000), tgt.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss
    run = torch.compile(step, mode=mode)
elif name == "gpt_infer_bf16":
    model = GPT().to(dev).to(torch.bfloat16).eval()
    x = torch.randint(0, 32000, (16, 512), device=dev)
    fwd = torch.compile(model, mode=mode)
    def run():
        with torch.no_grad():
            return fwd(x)
elif name == "resnet_train":
    model = ResNetish().to(dev).to(memory_format=torch.channels_last)
    x = torch.randn(64, 3, 64, 64, device=dev).to(memory_format=torch.channels_last)
    tgt = torch.randint(0, 10, (64,), device=dev)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, foreach=True)
    def step():
        loss = F.cross_entropy(model(x), tgt)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss
    run = torch.compile(step, mode=mode)
else:
    raise SystemExit(f"unknown model {name}")

compiled = run


def run():
    if mode is not None:
        torch.compiler.cudagraph_mark_step_begin()
    return compiled()


t0 = time.perf_counter()
run()
torch.cuda.synchronize()
first = time.perf_counter() - t0
for _ in range(5):
    run()
torch.cuda.synchronize()
best = float("inf")
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    best = min(best, (time.perf_counter() - t0) / 10)
c = counters["flydsl_simd"]
fallbacks = {k[len("fallback: "):]: v for k, v in c.items() if k.startswith("fallback: ")}
print("RESULT " + json.dumps({"first_s": first, "step_ms": best * 1e3, "flydsl": c["flydsl_kernels"],
                             "triton": c["triton_kernels"], "fallbacks": fallbacks}))
"""

MODELS = ["gpt_train_bf16", "gpt_infer_bf16", "resnet_train"]


def child(name, backend, env, cwd, mode="default"):
    # Run outside the repository so a source checkout's torch/ does not shadow the build.
    out = subprocess.run(
        [sys.executable, "-c", CHILD, name, backend, mode],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError(f"{name}/{backend} failed:\n{out.stderr[-4000:]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--json")
    a = p.parse_args()
    rows = []
    print(
        f"{'model':16s} {'backend':7s} {'cold s':>7s} {'warm s':>7s} {'step ms':>8s} {'graph ms':>9s}  kernels",
        flush=True,
    )
    for name in MODELS:
        if a.only and name not in a.only:
            continue
        for backend in ("triton", "flydsl"):
            with tempfile.TemporaryDirectory() as d:
                env = dict(os.environ)
                env.update(
                    TORCHINDUCTOR_CACHE_DIR=os.path.join(d, "inductor"),
                    TRITON_CACHE_DIR=os.path.join(d, "triton"),
                    FLYDSL_RUNTIME_CACHE_DIR=os.path.join(d, "flydsl"),
                )
                cold = child(name, backend, env, d)
                warm = child(name, backend, env, d)
                graphs = child(name, backend, env, d, "reduce-overhead")
            row = {
                "model": name,
                "backend": backend,
                "cold_s": cold["first_s"],
                "warm_s": warm["first_s"],
                "step_ms": min(cold["step_ms"], warm["step_ms"]),
                "graph_step_ms": graphs["step_ms"],
                "flydsl": cold["flydsl"],
                "triton": cold["triton"],
                "fallbacks": cold["fallbacks"],
            }
            rows.append(row)
            kern = (
                ""
                if backend == "triton"
                else f"flydsl={row['flydsl']} triton={row['triton']} {row['fallbacks'] or ''}"
            )
            print(
                f"{name:16s} {backend:7s} {row['cold_s']:7.1f} {row['warm_s']:7.1f} {row['step_ms']:8.2f}"
                f" {row['graph_step_ms']:9.2f}  {kern}",
                flush=True,
            )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
