"""A/B template variants on the loss and guard sets exported by report.py.

  python hillclimb.py --sets prefix_loss.json prefix_guard.json \\
      --variants current cfg:bt=128 tpl:skip_on "tpl:skip_on+cfg:w=8" [--gpus 8]

Variant specs (combine with "+"):
  current          the template as it is in the tree
  cfg:k=v,...      override the default config heuristic; keys:
                   bd (max BLOCK_D), tile (tile elements), bt (BLOCK_T), di (D_ITERS),
                   w (num_warps), s (num_stages)
  tpl:<name>       template source rewrite from TEMPLATE_VARIANTS
  icfg:<name>=<v>  patch an inductor config value (e.g. a preagg gate)
  mode:<base|ps>   a reference mode; when present it replaces the set's stored
                   best_ref_ms in the summary

All variants of a shape run in one process and are timed interleaved on the
same GPU. Results append to results/hillclimb.jsonl; the summary reports, per
set, the geomean of best(base, ps) / variant and guard-set regressions vs
"current".
"""

import argparse
import contextlib
import functools
import json
import math
import os
import subprocess
import sys
import time
from unittest import mock

import torch
import torch._inductor.config as icfg


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import mode_patch, run_shape, Shape


# name -> list of (old, new) replacements applied to the template source.
_SCAN = "            _, acc = tl.associative_scan((seg_key, val), 0, _preagg_segmented_sum)\n"
_SEG_KEY = "    seg_key = tl.broadcast_to(skey[:, None], (BLOCK_T, BLOCK_D))\n"
_SEG_FN = "    return key_b, tl.where(key_a == key_b, val_a + val_b, val_b)\n"
_MAX_FN = (
    _SEG_FN + "\n\n@triton.jit\ndef _preagg_max(a, b):\n    return tl.maximum(a, b)\n"
)
_SEG_START = (
    "    is_start = (skey != tl.gather(skey, tl.maximum(pos - 1, 0), 0)) | (pos == 0)\n"
    "    seg_start = tl.associative_scan(tl.where(is_start, pos, 0), 0, _preagg_max)\n"
    "    prev_row = tl.broadcast_to(tl.maximum(seg_start - 1, 0)[:, None], (BLOCK_T, BLOCK_D))\n"
    "    has_prev = (seg_start > 0)[:, None]\n"
)
TEMPLATE_VARIANTS: dict[str, list[tuple[str, str]]] = {
    # Atomics in the layout derived from load contiguity (ATOMIC_VEC off).
    "novec": [("            if ATOMIC_VEC:\n", "            if 0:\n")],
    # No-duplicate scan skip forced on / off regardless of SKIP_UNIQUE_SCAN.
    "skip_on": [("    if SKIP_UNIQUE_SCAN:\n", "    if 1:\n")],
    "skip_off": [("    if SKIP_UNIQUE_SCAN:\n", "    if 0:\n")],
    # Segment sums as differences of a plain prefix sum (rejected: ~3x slower).
    "cumsum": [
        (_SEG_FN, _MAX_FN),
        (_SEG_KEY, _SEG_KEY + _SEG_START),
        (
            _SCAN,
            "            csum = tl.cumsum(val, 0)\n"
            "            acc = csum - tl.where(has_prev, tl.gather(csum, prev_row, 0), 0)\n",
        ),
    ],
}


def _template_patch(name: str):
    from torch._inductor.kernel import scatter_preagg as sp

    src = sp._TEMPLATE_SOURCE
    for old, new in TEMPLATE_VARIANTS[name]:
        if old not in src:
            raise ValueError(
                f"template variant {name}: pattern not found: {old[:60]!r}"
            )
        src = src.replace(old, new)
    tpl = sp.ScatterPreaggTemplate(
        name=f"scatter_preagg_{name}",
        grid=sp.scatter_preagg_grid,
        always_freeze_layout=True,
        source=src,
    )
    return mock.patch.object(sp, "scatter_preagg_template", tpl)


def _config_patch(spec: str):
    from torch._inductor.heuristics.template import triton as heur
    from torch._inductor.runtime.runtime_utils import next_power_of_2

    kv = {k: int(v) for k, v in (item.split("=") for item in spec.split(","))}

    def configs(self, num_tokens, num_inner, dtype, inner_contiguous=True):
        if inner_contiguous:
            default_bd, default_tile = (32 if num_inner <= 128 else 64), 8192
        else:
            default_bd, default_tile = 8, 4096
        max_block_d = kv.get("bd", default_bd)
        tile = kv.get("tile", default_tile)
        if "bt" in kv:
            tile = kv["bt"] * min(next_power_of_2(num_inner), max_block_d)
        cfg = self._scatter_preagg_config(
            num_tokens, num_inner, max_block_d, tile, kv.get("w", 4)
        )
        if "bt" in kv:
            cfg = heur.ScatterPreaggConfig(
                kv["bt"], cfg.block_d, cfg.d_iters, cfg.num_stages, cfg.num_warps
            )
        if "di" in kv or "s" in kv:
            cfg = heur.ScatterPreaggConfig(
                cfg.block_t,
                cfg.block_d,
                kv.get("di", cfg.d_iters),
                kv.get("s", cfg.num_stages),
                cfg.num_warps,
            )
        return [cfg]

    return mock.patch.object(
        heur.BaseConfigHeuristic, "get_scatter_preagg_configs", configs
    )


@contextlib.contextmanager
def variant_ctx(spec: str):
    if spec.startswith("mode:"):
        with mode_patch(spec[5:]):
            yield
        return
    with contextlib.ExitStack() as stack:
        stack.enter_context(mode_patch("preagg"))
        # Template and heuristic patches are invisible to the FX graph cache key.
        stack.enter_context(icfg.patch(fx_graph_cache=False))
        for part in spec.split("+"):
            if part == "current":
                continue
            kind, _, arg = part.partition(":")
            if kind == "cfg":
                stack.enter_context(_config_patch(arg))
            elif kind == "tpl":
                stack.enter_context(_template_patch(arg))
            elif kind == "icfg":
                name, _, value = arg.partition("=")
                stack.enter_context(icfg.patch(**{name: type(getattr(icfg, name))(value)}))
            else:
                raise ValueError(part)
        yield


def load_set(path):
    with open(path) as f:
        return [
            (Shape(*e["shape"]), e["best_ref_ms"], os.path.basename(path))
            for e in json.load(f)
        ]


def worker(args):
    entries = []
    for path in args.sets:
        entries += load_set(path)
    entries = entries[args.shard :: args.nshards]
    with open(args.out, "a") as f:
        for s, best_ref, set_name in entries:
            res = run_shape(
                s, {v: functools.partial(variant_ctx, v) for v in args.variants}
            )
            f.writelines(
                json.dumps(
                    {
                        "run": args.run,
                        "set": set_name,
                        "variant": v,
                        "shape": list(s.__dict__.values()),
                        "best_ref_ms": best_ref,
                        **r,
                    }
                )
                + "\n"
                for v, r in res.items()
            )
            f.flush()


def summarize(args):
    rows = []
    with open(args.out) as f:
        for line in f:
            r = json.loads(line)
            if r["run"] == args.run:
                rows.append(r)
    by = {}
    for r in rows:
        by.setdefault((r["set"], tuple(r["shape"])), {})[r["variant"]] = r
    for set_name in sorted({k[0] for k in by}):
        shapes = {k: v for k, v in by.items() if k[0] == set_name}
        print(f"\n== {set_name} ({len(shapes)} shapes)")

        def best_ref(m, v):
            fresh = [
                m[r]["ms"] for r in ("mode:base", "mode:ps") if "ms" in m.get(r, {})
            ]
            return min(fresh) if fresh else m[v]["best_ref_ms"]

        for v in args.variants:
            sp = [
                best_ref(m, v) / m[v]["ms"]
                for m in shapes.values()
                if "ms" in m.get(v, {}) and best_ref(m, v) is not None
            ]
            vs_cur = [
                m["current"]["ms"] / m[v]["ms"]
                for m in shapes.values()
                if "ms" in m.get(v, {}) and "ms" in m.get("current", {})
            ]
            errs = sum("error" in m.get(v, {"error": 1}) for m in shapes.values())
            g = math.exp(sum(map(math.log, sp)) / len(sp)) if sp else float("nan")
            gc = (
                math.exp(sum(map(math.log, vs_cur)) / len(vs_cur))
                if vs_cur
                else float("nan")
            )
            print(
                f"  {v:40s} vs best(base,ps) {g:5.2f}x  vs current {gc:5.2f}x "
                f"(worst {min(vs_cur, default=float('nan')):.2f}x, "
                f">3% slower {sum(x < 0.97 for x in vs_cur)})  errors={errs}"
            )


def launch(args):
    env_var = "HIP_VISIBLE_DEVICES" if torch.version.hip else "CUDA_VISIBLE_DEVICES"
    gpus = args.gpu_ids or list(range(args.gpus))
    procs = []
    for i, gpu in enumerate(gpus):
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--shard",
            str(i),
            "--nshards",
            str(len(gpus)),
            "--out",
            args.out,
            "--run",
            args.run,
            "--sets",
            *args.sets,
            "--variants",
            *args.variants,
        ]
        os.makedirs(f"{args.out}.logs", exist_ok=True)
        log = f"{args.out}.logs/{args.run}.{i}.log"
        with open(log, "w") as lf:
            procs.append(
                subprocess.Popen(
                    cmd,
                    env={**os.environ, env_var: str(gpu)},
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            )
    t0 = time.time()
    for p in procs:
        p.wait()
    print(
        f"run {args.run} done in {time.time() - t0:.0f}s, rc={[p.returncode for p in procs]}"
    )
    summarize(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "results", "hillclimb.jsonl"),
    )
    p.add_argument("--run", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--worker", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--gpu-ids", type=int, nargs="+")
    args = p.parse_args()
    if "current" not in args.variants:
        args.variants = ["current", *args.variants]
    if args.worker:
        worker(args)
    elif args.summarize:
        summarize(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
