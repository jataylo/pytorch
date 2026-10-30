# mypy: allow-untyped-defs
import argparse
import base64
import functools
import importlib
import logging
import os
import sys
from typing import TypeVar

from torch._inductor.async_compile import pre_fork_setup
from torch._inductor.codecache import torch_key
from torch._inductor.compile_worker.subproc_pool import (
    SubprocKind,
    SubprocMain,
    SubprocPickler,
)
from torch._inductor.compile_worker.utils import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path


_T = TypeVar("_T")


log = logging.getLogger(__name__)

_set_triton_ptxas_path()

try:
    import triton

    if triton is None:
        raise AssertionError("triton failed to preload in parent")
except ImportError:
    pass


def _lookup_and_create_type(base: type[_T], qname: str) -> _T:
    """
    Given a base type and qualified name: import & lookup that name, check
    that it's of the given type and then instantiate it.
    """
    pkg, name = qname.rsplit(".", 1)
    mod = importlib.import_module(pkg)
    ty = getattr(mod, name)
    if not issubclass(ty, base):
        raise TypeError(f"Type {ty} is not a subtype of {base}")
    return ty()


def _preload_flydsl(arch: str | None) -> None:
    """Import and set up FlyDSL before the workers fork, as Triton is imported above,
    so each inherits it rather than paying ~2 s on its first kernel. Only a head
    start: on failure the workers set up alone."""
    try:
        from torch._inductor.codegen.flydsl.flydsl_simd_runtime import preload_worker

        preload_worker(arch)
    except Exception:
        log.exception("compile_worker FlyDSL preload failed")


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--pickler", type=functools.partial(_lookup_and_create_type, SubprocPickler)
        )
        parser.add_argument("--kind", type=SubprocKind)
        parser.add_argument("--workers", type=int)
        parser.add_argument("--parent", type=int)
        parser.add_argument("--read-fd", type=int)
        parser.add_argument("--write-fd", type=int)
        parser.add_argument("--torch-key", type=str)
        parser.add_argument("--flydsl-preload", type=str, default=None)
        args = parser.parse_args()
        if os.getppid() != args.parent:
            sys.exit(0)
        read_fd = os.fdopen(args.read_fd, "rb")
        write_fd = os.fdopen(args.write_fd, "wb")

        pre_fork_setup()

        torch_key.set(base64.b64decode(args.torch_key.encode("utf-8")))  # type: ignore[attr-defined]

        _async_compile_initializer(args.parent)

        if args.flydsl_preload is not None:
            _preload_flydsl(args.flydsl_preload or None)

        SubprocMain(args.pickler, args.kind, args.workers, read_fd, write_fd).main()
    except Exception:
        log.exception("Uncaught exception in compile_worker subprocess")


if __name__ == "__main__":
    main()
