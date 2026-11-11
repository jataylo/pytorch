from __future__ import annotations

import json
import os

from torch._logging import trace_structured

from ... import config


def record_kernel_decision(event: dict) -> None:
    context = os.environ.get("TORCHINDUCTOR_FLYDSL_DIAGNOSTIC_CONTEXT")
    entry = {**(json.loads(context) if context else {}), **event}
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "flydsl_kernel_decision",
            "encoding": "json",
        },
        payload_fn=lambda: json.dumps(entry, default=str),
    )
    path = config.flydsl.simd_diagnostics_path
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(entry, default=str) + "\n").encode())
        finally:
            os.close(fd)
