#!/usr/bin/env python3
"""
parse_bench_verbose.py  –  Extract configs from BENCH_VERBOSE log.

Usage:
    python parse_bench_verbose.py <log_file> [output.csv]            # fastest config
    python parse_bench_verbose.py <log_file> [output.csv] --chosen   # heuristic-chosen config

Output CSV columns (fastest mode):
    kernel_name, size_hints, fastest_us, config

Output CSV columns (chosen mode):
    kernel_name, size_hints, chosen_us, config, n_spills

If a kernel+size_hints combination appears multiple times (re-benchmarked),
only the last chosen / overall fastest finite result is kept.
"""

import csv
import re
import sys
from pathlib import Path

# ── patterns ──────────────────────────────────────────────────────────────────
RE_HEADER = re.compile(
    r'\[BENCH_VERBOSE\] kernel=(\S+)\s+size_hints=(\{[^}]+\})\s+n_configs=(\d+)'
)
RE_CONFIG = re.compile(
    r'\[BENCH_VERBOSE\]\s+\[\d+/\d+\] benchmarking (.+?) …\s+→ (.+?) µs'
)
# [HEURISTICS] ✓ Chosen: {'XBLOCK': 256}  num_warps=4  n_spills=0  time=0.006600ms ★ (from top-N)
RE_CHOSEN_PW = re.compile(
    r'\[HEURISTICS\].*?Chosen:\s*(\{[^}]+\})\s+num_warps=(\d+).*?n_spills=(\d+).*?time=([\d.]+)ms'
)
# 📊 HEURISTIC CHOSEN  (fastest from top-5 selection pool)   [reduction validation block]
RE_CHOSEN_RED_SECTION = re.compile(r'📊 HEURISTIC CHOSEN')
#     Config  : R0_BLOCK=4096 nw=4 XBLOCK=2
RE_CHOSEN_RED_CFG     = re.compile(r'Config\s*:\s*(.+)')
#     Measured: 0.009000 ms
RE_CHOSEN_RED_TIME    = re.compile(r'Measured\s*:\s*([\d.]+)\s*ms')

SKIP_PARAMS = {"num_ctas", "num_stages", "maxnreg"}


def parse_config_str(raw: str) -> str:
    """Normalise a raw BENCH_VERBOSE config param string into a compact form.

    Input:  'XBLOCK: 128, num_warps: 2, num_ctas: 1, num_stages: 1, maxnreg: None'
    Output: 'XBLOCK=128 num_warps=2'
    """
    parts = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            k, v = item.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in SKIP_PARAMS or v == "None":
                continue
            parts.append(f"{k}={v}")
    return " ".join(parts)


def parse_chosen_config(cfg_dict_str: str, num_warps: str) -> str:
    """Build a compact config string from the pointwise Chosen line.

    Input:  "{'XBLOCK': 256, 'R0_BLOCK': 2048}", "4"
    Output: "XBLOCK=256 R0_BLOCK=2048 num_warps=4"
    """
    parts = []
    for item in cfg_dict_str.strip("{}").split(","):
        item = item.strip()
        if not item:
            continue
        kv = item.split(":", 1)
        if len(kv) != 2:
            continue
        k = kv[0].strip().strip("'\"")
        v = kv[1].strip().strip("'\"")
        if k in SKIP_PARAMS or v == "None":
            continue
        parts.append(f"{k}={v}")
    parts.append(f"num_warps={num_warps}")
    return " ".join(parts)


def parse_red_config_str(raw: str) -> str:
    """Normalise a HEURISTIC CHOSEN Config line – handles two formats.

    Flat key=value (reduction):  "R0_BLOCK=4096 nw=4 XBLOCK=2"
    Dict literal (pointwise):    "{'XBLOCK': 512, 'num_warps': 8}"

    Output (either input): "XBLOCK=512 num_warps=8"   (canonical order)
    """
    raw = raw.strip()
    mapping: dict[str, str] = {}

    if raw.startswith("{"):
        # Dict literal produced by the pointwise HEURISTICS VALIDATION block.
        for item in raw.strip("{}").split(","):
            item = item.strip()
            if not item:
                continue
            kv = item.split(":", 1)
            if len(kv) != 2:
                continue
            k = kv[0].strip().strip("'\"")
            v = kv[1].strip().strip("'\"")
            if k not in SKIP_PARAMS and v != "None":
                mapping[k] = v
    else:
        # Flat key=value format used by reduction HEURISTIC CHOSEN blocks.
        for token in raw.split():
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            k = k.strip()
            if k == "nw":
                k = "num_warps"
            elif k == "wpe":
                k = "waves_per_eu"
            mapping[k] = v.strip()

    # Canonical order: XBLOCK, YBLOCK, R0_BLOCK, waves_per_eu, num_warps
    order = ["XBLOCK", "YBLOCK", "R0_BLOCK", "waves_per_eu", "num_warps"]
    parts = [f"{k}={mapping[k]}" for k in order if k in mapping]
    parts += [f"{k}={v}" for k, v in mapping.items() if k not in order]
    return " ".join(parts)


# ── fastest mode ──────────────────────────────────────────────────────────────

def parse_log_fastest(path: Path) -> list[dict]:
    """Return list of {kernel_name, size_hints, fastest_us, config} dicts."""
    best: dict[tuple, tuple] = {}
    current_kernel: str | None = None
    current_hints:  str | None = None

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_HEADER.match(line)
            if m:
                current_kernel, current_hints = m.group(1), m.group(2)
                continue

            if current_kernel is None:
                continue
            m = RE_CONFIG.match(line)
            if not m:
                continue

            raw_cfg, raw_time = m.group(1), m.group(2)
            if raw_time.strip() == "inf":
                continue
            try:
                us = float(raw_time)
            except ValueError:
                continue

            key = (current_kernel, current_hints)
            prev_us, _ = best.get(key, (float("inf"), ""))
            if us < prev_us:
                best[key] = (us, parse_config_str(raw_cfg))

    return [
        dict(kernel_name=k, size_hints=h, fastest_us=us, config=cfg)
        for (k, h), (us, cfg) in sorted(best.items())
    ]


# ── chosen mode ───────────────────────────────────────────────────────────────

def parse_log_chosen(path: Path) -> list[dict]:
    """Return list of {kernel_name, size_hints, chosen_us, config, n_spills} dicts.

    Two source formats are handled:
      • Pointwise:  [HEURISTICS] ✓ Chosen: {'XBLOCK': 256}  num_warps=4  n_spills=0  time=...ms
      • Reduction:  📊 HEURISTIC CHOSEN block inside [REDUCTION VALIDATION], with
                    separate  Config  : ...  and  Measured: ...  lines.
    """
    chosen: dict[tuple, dict] = {}
    current_kernel: str | None = None
    current_hints:  str | None = None

    # State for multi-line reduction chosen block
    in_red_chosen = False
    red_cfg: str | None = None

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # ── kernel header resets reduction state ───────────────────────────
            m = RE_HEADER.match(line)
            if m:
                current_kernel, current_hints = m.group(1), m.group(2)
                in_red_chosen = False
                red_cfg = None
                continue

            # ── pointwise: single-line Chosen ─────────────────────────────────
            m = RE_CHOSEN_PW.search(line)
            if m and current_kernel is not None:
                key = (current_kernel, current_hints)
                chosen[key] = dict(
                    kernel_name=current_kernel,
                    size_hints=current_hints,
                    chosen_us=float(m.group(4)) * 1000.0,
                    config=parse_chosen_config(m.group(1), m.group(2)),
                    n_spills=int(m.group(3)),
                )
                in_red_chosen = False
                red_cfg = None
                continue

            # ── reduction: enter HEURISTIC CHOSEN section ─────────────────────
            if RE_CHOSEN_RED_SECTION.search(line):
                in_red_chosen = True
                red_cfg = None
                continue

            if in_red_chosen:
                # First Config line after the section header is the chosen one
                if red_cfg is None:
                    mc = RE_CHOSEN_RED_CFG.search(line)
                    if mc:
                        red_cfg = parse_red_config_str(mc.group(1))
                        continue

                # Measured line gives the time (ms → µs)
                mm = RE_CHOSEN_RED_TIME.search(line)
                if mm and red_cfg is not None and current_kernel is not None:
                    key = (current_kernel, current_hints)
                    chosen[key] = dict(
                        kernel_name=current_kernel,
                        size_hints=current_hints,
                        chosen_us=float(mm.group(1)) * 1000.0,
                        config=red_cfg,
                        n_spills=0,
                    )
                    in_red_chosen = False
                    red_cfg = None
                    continue

                # Stop reading reduction block if we hit ACTUAL Best
                if "ACTUAL Best" in line or "🏆" in line:
                    in_red_chosen = False
                    red_cfg = None

    return [row for _, row in sorted(chosen.items())]


# ── writers ───────────────────────────────────────────────────────────────────

def write_fastest_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["kernel_name", "size_hints", "fastest_us", "config"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_chosen_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["kernel_name", "size_hints", "chosen_us", "config", "n_spills"]
        )
        writer.writeheader()
        writer.writerows(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    chosen_mode = "--chosen" in flags

    if not args:
        print(__doc__)
        sys.exit(1)

    log_path = Path(args[0])
    if not log_path.exists():
        print(f"ERROR: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    if len(args) >= 2:
        out_path = Path(args[1])
    else:
        suffix = "_chosen.csv" if chosen_mode else ".csv"
        out_path = log_path.with_name(log_path.stem + suffix)

    mode_label = "heuristic-chosen" if chosen_mode else "fastest"
    print(f"Parsing {log_path} [{mode_label} mode] …", flush=True)

    if chosen_mode:
        rows = parse_log_chosen(log_path)
        print(f"  {len(rows)} heuristic-chosen configs found.")
        write_chosen_csv(rows, out_path)
    else:
        rows = parse_log_fastest(log_path)
        print(f"  {len(rows)} unique kernel+shape combinations found.")
        write_fastest_csv(rows, out_path)

    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
