# Supplemental static kernel inventory

Static definitions and reachable wrapper callsites in associated caches; not dynamic launch counts or proof that failed attempts executed these graphs.

Scanned 2176 unique wrapper sources from 620 cache paths and 1241 record files.

Counts: `{"definitions_flydsl": 336046, "definitions_triton": 366, "static_launch_sites_flydsl": 441176, "static_launch_sites_triton": 3224}`.

Unmatched Triton context sites: 0; uncertain dispatch sites: 0; unresolved launch references: 0.

These are static source counts. Unknown or preexisting-cache associations are not evidence of a new unsupported FlyDSL operation. Native GEMM/convolution policy calls and other native operations are separated in native_calls.jsonl.

Full provenance, limitations, missing sources and parse errors are in summary.json; per-site named fallback matches are in triton_diagnostic_matches.jsonl.
