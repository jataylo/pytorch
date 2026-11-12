# Native Triton results

- `validation.json` and `test_results.json`: fresh validation and source hashes for the cleaned branch.
- `clean_branch/`: fresh 16-case, two-policy smoke benchmark (128 rows, zero failures).
- `reference_v5/`: historical 142-case results from commit `b95d6d3c9a9`, before the upstream-only extraction. These measurements do not claim to benchmark the cleaned branch.

Historical files retain their values and recorded statuses. The original source
paths identify the measured environment. Compiler caches, model weights,
checkpoints and unrelated backend result bundles are excluded.

See [the change description](../../TRITON_CHANGE_DESCRIPTION_AND_PR_PLAN.md)
for measurement definitions, accuracy limits and reproduction commands.
