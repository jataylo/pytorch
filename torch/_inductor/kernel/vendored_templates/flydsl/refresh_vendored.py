# SPDX-License-Identifier: Apache-2.0
"""Refresh vendored FlyDSL kernel-side files against an upstream checkout.

Why this exists: the flex kernels here began as a copy of upstream FlyDSL kernel code and
then drifted. Over three weeks that fork silently missed four relevant upstream fixes,
because nothing recorded where a file came from or made it cheap to ask "what changed
upstream since?". Each vendored file therefore carries a provenance header naming its
upstream path, branch and commit, and this script reads those headers back.

    # what drifted, against the commit each file records
    python refresh_vendored.py --check --upstream /dockerx/xinya-flydsl

    # what upstream changed since that commit
    python refresh_vendored.py --log --upstream /dockerx/xinya-flydsl

    # pull the current upstream content in, rewriting the header's commit
    python refresh_vendored.py --update --upstream /dockerx/xinya-flydsl

`--update` deliberately does not try to merge: it overwrites and leaves the diff in the
working tree for review, because the local edits are the interesting part and a merge
would hide them. Files whose header says `local: true` are ours and never touched.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent

_HEADER = {
    "path": re.compile(r"^#\s*upstream:\s*\S+\s+`([^`]+)`", re.MULTILINE),
    "branch": re.compile(r"^#\s*branch:\s*(\S+)", re.MULTILINE),
    "commit": re.compile(r"^#\s*commit:\s*([0-9a-f]{7,40})", re.MULTILINE),
    "local": re.compile(r"^#\s*local:\s*true\s*$", re.MULTILINE),
}


def vendored_files():
    """Every .py under here that carries a provenance header, with what it claims."""
    found = []
    for path in sorted(HERE.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        head = "".join(path.read_text().splitlines(keepends=True)[:40])
        if _HEADER["local"].search(head):
            continue
        m = _HEADER["path"].search(head)
        if not m:
            continue
        found.append(
            {
                "local_path": path,
                "upstream_path": m.group(1),
                "branch": (b.group(1) if (b := _HEADER["branch"].search(head)) else None),
                "commit": (c.group(1) if (c := _HEADER["commit"].search(head)) else None),
            }
        )
    return found


def git(upstream: Path, *args):
    return subprocess.run(
        ["git", "-C", str(upstream), *args], capture_output=True, text=True, check=False
    )


def upstream_content(upstream: Path, rev: str, rel: str):
    got = git(upstream, "show", f"{rev}:{rel}")
    return got.stdout if got.returncode == 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", type=Path, required=True, help="path to the FlyDSL checkout")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report local drift from the recorded commit")
    mode.add_argument("--log", action="store_true", help="report upstream commits since the recorded one")
    mode.add_argument("--update", action="store_true", help="overwrite with current upstream content")
    args = ap.parse_args(argv)

    upstream = args.upstream.resolve()
    if not (upstream / ".git").exists():
        sys.exit(f"{upstream} is not a git checkout")

    files = vendored_files()
    if not files:
        sys.exit("no files here carry a provenance header")

    head = git(upstream, "rev-parse", "HEAD").stdout.strip()
    print(f"upstream {upstream} at {head[:12]}\n")

    drift = 0
    for f in files:
        rel, commit = f["upstream_path"], f["commit"]
        name = f["local_path"].relative_to(HERE)

        if args.log:
            if not commit:
                print(f"{name}: no commit recorded, cannot diff")
                continue
            got = git(upstream, "log", "--oneline", f"{commit}..HEAD", "--", rel)
            lines = [ln for ln in got.stdout.splitlines() if ln.strip()]
            print(f"{name}: {len(lines)} upstream commit(s) since {commit[:12]}")
            for ln in lines:
                print(f"    {ln}")
            drift += len(lines)
            continue

        pinned = upstream_content(upstream, commit, rel) if commit else None
        if pinned is None:
            print(f"{name}: cannot read {rel} at {commit}")
            continue

        # Compare bodies, not headers: the provenance block is ours by construction.
        def body(text):
            return [ln for ln in text.splitlines() if not ln.startswith("#")]

        local = f["local_path"].read_text()
        if body(local) == body(pinned):
            print(f"{name}: clean against {commit[:12]}")
        else:
            print(f"{name}: LOCAL EDITS against {commit[:12]}")
            drift += 1

        if args.update:
            current = upstream_content(upstream, "HEAD", rel)
            if current is None:
                print(f"    upstream no longer has {rel} -- left alone, decide by hand")
                continue
            header = []
            for ln in local.splitlines(keepends=True):
                header.append(ln)
                if ln.startswith("#") is False and ln.strip() == "":
                    break
            new_header = re.sub(
                r"(^#\s*commit:\s*)[0-9a-f]{7,40}",
                rf"\g<1>{head[:40]}",
                "".join(header),
                flags=re.MULTILINE,
            )
            f["local_path"].write_text(new_header + current)
            print("    rewritten from HEAD; review the diff")

    print(f"\n{drift} item(s) need attention" if drift else "\nnothing to do")
    return 1 if (drift and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
