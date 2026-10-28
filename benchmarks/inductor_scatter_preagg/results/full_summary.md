Geomean speedup over regular Inductor (base) by measured contention.
'vs best' and 'slower' compare adaptive with best(base, ps) per case;
errors are the bucket's max, base / ps / adaptive.

embedding bfloat16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.00 | 1.05 | 0.86 | 0.86 | 3 | 3e-03 / 3e-03 / 3e-03 |
| 8-64 | 4 | 1.00 | 1.00 | 1.51 | 1.33 | 1.33 | 0 | 3e-03 / 3e-03 / 3e-03 |
| 64-1K | 12 | 1.00 | 1.01 | 1.59 | 0.97 | 0.97 | 1 | 3e-03 / 3e-03 / 3e-03 |
| 1K-16K | 14 | 1.00 | 1.00 | 1.47 | 0.87 | 0.87 | 3 | 3e-03 / 3e-03 / 3e-03 |
| >=16K | 34 | 1.00 | 0.99 | 7.31 | 1.10 | 1.10 | 0 | 4e-03 / 4e-03 / 4e-03 |
| all | 72 | 1.00 | 1.00 | 3.07 | 0.86 | 0.86 | 7 | 4e-03 / 4e-03 / 4e-03 |
  0.86  embedding bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.87  embedding bfloat16 rows=262144 d=128 n=1048576 fewhot (contention 4.2e+03)
  0.89  embedding bfloat16 rows=262144 d=16 n=1048576 uniform (contention 5)
  0.94  embedding bfloat16 rows=262144 d=512 n=262144 fewhot (contention 1.04e+03)
  0.95  embedding bfloat16 rows=262144 d=512 n=262144 sorted (contention 2)

embedding float16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.01 | 1.05 | 0.86 | 0.86 | 3 | 3e-04 / 3e-04 / 3e-04 |
| 8-64 | 4 | 1.00 | 1.01 | 1.52 | 1.33 | 1.33 | 0 | 4e-04 / 4e-04 / 4e-04 |
| 64-1K | 12 | 1.00 | 1.00 | 1.58 | 0.97 | 0.97 | 0 | 4e-04 / 4e-04 / 4e-04 |
| 1K-16K | 14 | 1.00 | 1.00 | 1.48 | 0.94 | 0.94 | 2 | 4e-04 / 4e-04 / 4e-04 |
| >=16K | 34 | 1.00 | 1.00 | 7.34 | 1.12 | 1.12 | 0 | 4e-04 / 4e-04 / 4e-04 |
| all | 72 | 1.00 | 1.00 | 3.07 | 0.86 | 0.86 | 5 | 4e-04 / 4e-04 / 4e-04 |
  0.86  embedding float16 rows=262144 d=512 n=262144 uniform (contention 2.01)
  0.90  embedding float16 rows=262144 d=16 n=1048576 uniform (contention 5)
  0.94  embedding float16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.94  embedding float16 rows=4096 d=512 n=262144 fewhot (contention 1.1e+03)
  0.95  embedding float16 rows=262144 d=512 n=262144 fewhot (contention 1.05e+03)

embedding float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.00 | 1.10 | 0.93 | 0.92 | 1 | 1e-07 / 1e-07 / 2e-07 |
| 8-64 | 4 | 1.00 | 1.00 | 1.39 | 1.29 | 1.29 | 0 | 4e-07 / 3e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 1.04 | 1.79 | 1.11 | 1.11 | 0 | 9e-07 / 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.00 | 0.93 | 1.24 | 0.98 | 0.98 | 0 | 7e-06 / 7e-06 / 6e-06 |
| >=16K | 24 | 1.00 | 1.02 | 6.61 | 1.46 | 1.46 | 0 | 3e-05 / 3e-05 / 4e-06 |
| all | 48 | 1.00 | 1.01 | 2.96 | 0.93 | 0.92 | 1 | 3e-05 / 3e-05 / 6e-06 |
  0.92  embedding float32 rows=262144 d=128 n=1048576 uniform (contention 5)

embedding_pad float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.02 | 1.08 | 0.82 | 0.82 | 2 | 2e-07 / 2e-07 / 2e-07 |
| 8-64 | 4 | 1.00 | 1.01 | 1.40 | 1.30 | 1.28 | 0 | 3e-07 / 4e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 1.00 | 1.70 | 0.93 | 0.93 | 1 | 1e-06 / 1e-06 / 6e-07 |
| 1K-16K | 6 | 1.00 | 0.99 | 1.24 | 0.94 | 0.94 | 1 | 5e-06 / 6e-06 / 5e-06 |
| >=16K | 24 | 1.00 | 1.00 | 6.65 | 1.45 | 1.45 | 0 | 3e-05 / 3e-05 / 4e-06 |
| all | 48 | 1.00 | 1.00 | 2.95 | 0.82 | 0.82 | 4 | 3e-05 / 3e-05 / 5e-06 |
  0.82  embedding_pad float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.93  embedding_pad float32 rows=4096 d=128 n=1048576 uniform (contention 257)
  0.94  embedding_pad float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.21e+03)
  0.95  embedding_pad float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

fused_producer bfloat16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.64 | 3.17 | 0.98 | 0.98 | 0 | 9e-03 / 7e-03 / 3e-03 |
| 8-64 | 4 | 1.00 | 1.30 | 6.24 | 1.07 | 1.07 | 0 | 3e-02 / 2e-02 / 4e-03 |
| 64-1K | 12 | 1.00 | 1.22 | 6.32 | 1.77 | 1.77 | 0 | 2e-01 / 4e-03 / 3e-03 |
| 1K-16K | 14 | 1.00 | 5.34 | 24.87 | 7.40 | 1.89 | 0 | 1e+00 / 1e+00 / 4e-03 |
| >=16K | 34 | 1.00 | 12.42 | 186.44 | 23.30 | 3.72 | 0 | 1e+00 / 1e+00 / 4e-03 |
| all | 72 | 1.00 | 4.54 | 37.74 | 0.98 | 0.98 | 0 | 1e+00 / 1e+00 / 4e-03 |

fused_producer float16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.29 | 0.98 | 0.69 | 0.69 | 4 | 1e-03 / 8e-04 / 9e-04 |
| 8-64 | 4 | 1.00 | 0.38 | 1.15 | 0.92 | 0.90 | 1 | 3e-03 / 2e-03 / 9e-04 |
| 64-1K | 12 | 1.00 | 1.04 | 1.36 | 0.65 | 0.65 | 5 | 9e-03 / 5e-04 / 7e-03 |
| 1K-16K | 14 | 1.00 | 0.88 | 1.40 | 0.81 | 0.75 | 1 | 9e-01 / 9e-01 / 5e-01 |
| >=16K | 34 | 1.00 | 5.58 | 7.55 | 1.23 | 0.45 | 13 | 1e+00 / inf / inf |
| all | 72 | 1.00 | 1.83 | 2.93 | 0.65 | 0.45 | 24 | 1e+00 / inf / inf |
  0.45  fused_producer float16 rows=64 d=16 n=1048576 fewhot (contention 1.99e+04)
  0.50  fused_producer float16 rows=64 d=16 n=1048576 zipf (contention 9.75e+04)
  0.52  fused_producer float16 rows=64 d=16 n=1048576 uniform (contention 1.64e+04)
  0.54  fused_producer float16 rows=64 d=16 n=1048576 hot50 (contention 2.76e+05)
  0.65  fused_producer float16 rows=4096 d=64 n=1048576 zipf (contention 4.01e+04)

fused_producer float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.55 | 1.35 | 1.23 | 1.23 | 0 | 2e-07 / 2e-07 / 1e-07 |
| 8-64 | 4 | 1.00 | 0.66 | 1.61 | 1.39 | 1.39 | 0 | 4e-07 / 4e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 1.40 | 2.13 | 1.61 | 1.19 | 0 | 1e-06 / 1e-07 / 8e-07 |
| 1K-16K | 6 | 1.00 | 0.86 | 1.50 | 1.28 | 1.09 | 0 | 8e-06 / 8e-06 / 9e-06 |
| >=16K | 24 | 1.00 | 3.49 | 6.84 | 1.22 | 1.03 | 0 | 4e-05 / 4e-05 / 5e-06 |
| all | 48 | 1.00 | 1.67 | 3.31 | 1.22 | 1.03 | 0 | 4e-05 / 4e-05 / 9e-06 |

gnn float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.64 | 1.14 | 0.79 | 0.79 | 2 | 1e-07 / 1e-07 / 1e-07 |
| 8-64 | 4 | 1.00 | 0.72 | 1.33 | 1.00 | 1.00 | 0 | 5e-07 / 3e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 1.25 | 1.51 | 0.99 | 0.80 | 2 | 8e-07 / 1e-07 / 7e-07 |
| 1K-16K | 6 | 1.00 | 0.86 | 1.20 | 0.95 | 0.95 | 3 | 5e-06 / 6e-06 / 5e-06 |
| >=16K | 24 | 1.00 | 3.61 | 5.11 | 0.96 | 0.72 | 12 | 4e-05 / 2e-05 / 4e-06 |
| all | 48 | 1.00 | 1.73 | 2.55 | 0.79 | 0.72 | 19 | 4e-05 / 2e-05 / 5e-06 |
  0.72  gnn float32 rows=64 d=128 n=1048576 uniform (contention 1.64e+04)
  0.72  gnn float32 rows=64 d=128 n=1048576 zipf (contention 9.75e+04)
  0.74  gnn float32 rows=64 d=128 n=1048576 fewhot (contention 1.99e+04)
  0.74  gnn float32 rows=64 d=128 n=1048576 runs32 (contention 1.64e+04)
  0.79  gnn float32 rows=262144 d=128 n=1048576 sorted (contention 5)

hist int32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.00 | 0.53 | 0.49 | 0.25 | 0.25 | 2 | 0e+00 / 0e+00 / 0e+00 |
| 64-1K | 2 | 1.00 | 0.49 | 0.41 | 0.19 | 0.19 | 2 | 0e+00 / 0e+00 / 0e+00 |
| >=16K | 8 | 1.00 | 1.62 | 1.62 | 1.04 | 0.51 | 4 | 0e+00 / 0e+00 / 0e+00 |
| all | 12 | 1.00 | 1.10 | 1.06 | 0.19 | 0.19 | 8 | 0e+00 / 0e+00 / 0e+00 |
  0.19  hist int32 rows=4096 d=1 n=1048576 sorted (contention 257)
  0.25  hist int32 rows=262144 d=1 n=1048576 sorted (contention 5.01)
  0.51  hist int32 rows=4096 d=1 n=1048576 zipf (contention 4.03e+04)
  0.82  hist int32 rows=4096 d=1 n=1048576 hot50 (contention 2.62e+05)
  0.83  hist int32 rows=64 d=1 n=1048576 uniform (contention 1.64e+04)

hist int64

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.00 | 0.99 | 5.60 | 4.19 | 4.19 | 0 | 0e+00 / 0e+00 / 0e+00 |
| 64-1K | 2 | 1.00 | 1.02 | 5.97 | 4.31 | 4.28 | 0 | 0e+00 / 0e+00 / 0e+00 |
| >=16K | 8 | 1.00 | 4.77 | 38.14 | 10.28 | 5.86 | 0 | 0e+00 / 0e+00 / 0e+00 |
| all | 12 | 1.00 | 2.84 | 20.34 | 4.19 | 4.19 | 0 | 0e+00 / 0e+00 / 0e+00 |

index_add bfloat16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.64 | 3.00 | 0.88 | 0.88 | 2 | 1e-02 / 6e-03 / 3e-03 |
| 8-64 | 4 | 1.00 | 1.37 | 6.39 | 0.95 | 0.95 | 1 | 2e-02 / 2e-02 / 4e-03 |
| 64-1K | 12 | 1.00 | 1.25 | 6.31 | 1.56 | 1.56 | 0 | 7e-02 / 5e-03 / 3e-03 |
| 1K-16K | 14 | 1.00 | 5.61 | 26.65 | 7.19 | 1.74 | 0 | 6e-01 / 6e-01 / 3e-03 |
| >=16K | 34 | 1.00 | 12.53 | 191.03 | 24.31 | 3.54 | 0 | 7e-01 / 5e-01 / 3e-03 |
| all | 72 | 1.00 | 4.64 | 38.50 | 0.88 | 0.88 | 3 | 7e-01 / 6e-01 / 4e-03 |
  0.88  index_add bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.89  index_add bfloat16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.95  index_add bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33)

index_add float16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.28 | 1.04 | 0.77 | 0.77 | 2 | 1e-03 / 9e-04 / 1e-03 |
| 8-64 | 4 | 1.00 | 0.37 | 1.28 | 1.05 | 1.05 | 0 | 4e-03 / 3e-03 / 7e-04 |
| 64-1K | 12 | 1.00 | 1.06 | 1.57 | 0.98 | 0.73 | 3 | 7e-03 / 6e-04 / 6e-03 |
| 1K-16K | 14 | 1.00 | 0.90 | 1.55 | 0.85 | 0.84 | 3 | 7e-02 / 7e-02 / 5e-02 |
| >=16K | 34 | 1.00 | 5.70 | 7.85 | 1.25 | 0.46 | 12 | 3e-01 / 8e-02 / 5e-02 |
| all | 72 | 1.00 | 1.85 | 3.16 | 0.77 | 0.46 | 20 | 3e-01 / 8e-02 / 5e-02 |
  0.46  index_add float16 rows=64 d=16 n=1048576 uniform (contention 1.64e+04)
  0.50  index_add float16 rows=64 d=16 n=1048576 fewhot (contention 1.99e+04)
  0.52  index_add float16 rows=64 d=16 n=1048576 zipf (contention 9.75e+04)
  0.56  index_add float16 rows=64 d=16 n=1048576 hot50 (contention 2.75e+05)
  0.68  index_add float16 rows=4096 d=64 n=1048576 zipf (contention 4.04e+04)

index_add float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.59 | 1.33 | 1.19 | 1.19 | 0 | 1e-07 / 1e-07 / 2e-07 |
| 8-64 | 4 | 1.00 | 0.66 | 1.71 | 1.63 | 1.62 | 0 | 4e-07 / 2e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 1.41 | 2.07 | 1.51 | 1.14 | 0 | 9e-07 / 1e-07 / 6e-07 |
| 1K-16K | 6 | 1.00 | 0.89 | 1.26 | 0.88 | 0.88 | 2 | 5e-06 / 5e-06 / 6e-06 |
| >=16K | 24 | 1.00 | 3.63 | 7.06 | 1.19 | 0.90 | 2 | 3e-05 / 2e-05 / 4e-06 |
| all | 48 | 1.00 | 1.73 | 3.28 | 0.88 | 0.88 | 4 | 3e-05 / 2e-05 / 6e-06 |
  0.88  index_add float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.2e+03)
  0.89  index_add float32 rows=4096 d=128 n=1048576 fewhot (contention 4.47e+03)
  0.90  index_add float32 rows=4096 d=32 n=1048576 zipf (contention 4.03e+04)
  0.97  index_add float32 rows=64 d=32 n=1048576 uniform (contention 1.64e+04)

index_add_dim1 float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.69 | 0.70 | 0.16 | 0.16 | 5 | 1e-07 / 1e-07 / 2e-07 |
| 8-64 | 4 | 1.00 | 0.54 | 1.18 | 1.08 | 1.08 | 0 | 1e-07 / 1e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 0.36 | 1.17 | 0.99 | 0.99 | 0 | 7e-07 / 1e-07 / 6e-07 |
| 1K-16K | 6 | 1.00 | 1.14 | 1.59 | 0.79 | 0.79 | 1 | 4e-06 / 3e-06 / 4e-06 |
| >=16K | 24 | 1.00 | 0.78 | 2.93 | 1.11 | 1.11 | 0 | 5e-06 / 5e-06 / 2e-06 |
| all | 48 | 1.00 | 0.71 | 1.77 | 0.16 | 0.16 | 6 | 5e-06 / 5e-06 / 4e-06 |
  0.16  index_add_dim1 float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.55  index_add_dim1 float32 rows=1048576 d=128 n=1048576 sorted (contention 2)
  0.56  index_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)
  0.79  index_add_dim1 float32 rows=1048576 d=128 n=1048576 fewhot (contention 4.21e+03)
  0.80  index_add_dim1 float32 rows=1048576 d=32 n=1048576 sorted (contention 2)

index_put bfloat16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.63 | 3.03 | 0.88 | 0.88 | 2 | 1e-02 / 6e-03 / 3e-03 |
| 8-64 | 4 | 1.00 | 1.37 | 6.39 | 0.94 | 0.94 | 1 | 3e-02 / 3e-02 / 3e-03 |
| 64-1K | 12 | 1.00 | 1.26 | 6.30 | 1.55 | 1.55 | 0 | 5e-02 / 4e-03 / 3e-03 |
| 1K-16K | 14 | 1.00 | 5.42 | 25.68 | 6.76 | 1.75 | 0 | 3e-01 / 3e-01 / 3e-03 |
| >=16K | 34 | 1.00 | 12.28 | 187.56 | 23.77 | 3.52 | 0 | 8e-01 / 6e-01 / 3e-03 |
| all | 72 | 1.00 | 4.56 | 37.92 | 0.88 | 0.88 | 3 | 8e-01 / 6e-01 / 3e-03 |
  0.88  index_put bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.90  index_put bfloat16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.94  index_put bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33.1)

index_put float16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 10 | 1.00 | 0.33 | 1.17 | 0.73 | 0.73 | 3 | 1e-03 / 8e-04 / 1e-03 |
| 8-64 | 4 | 1.00 | 0.38 | 1.27 | 1.00 | 1.00 | 0 | 4e-03 / 2e-03 / 7e-04 |
| 64-1K | 14 | 1.00 | 1.02 | 1.61 | 1.00 | 0.81 | 3 | 9e-03 / 6e-04 / 6e-03 |
| 1K-16K | 14 | 1.00 | 0.95 | 1.58 | 0.85 | 0.72 | 4 | 6e-02 / 6e-02 / 4e-02 |
| >=16K | 36 | 1.00 | 5.77 | 8.18 | 1.23 | 0.47 | 12 | 3e-01 / 1e-01 / 4e-02 |
| all | 78 | 1.00 | 1.84 | 3.22 | 0.73 | 0.47 | 22 | 3e-01 / 1e-01 / 4e-02 |
  0.47  index_put float16 rows=64 d=16 n=1048576 fewhot (contention 1.99e+04)
  0.47  index_put float16 rows=64 d=16 n=1048576 uniform (contention 1.64e+04)
  0.49  index_put float16 rows=64 d=16 n=1048576 zipf (contention 9.76e+04)
  0.56  index_put float16 rows=64 d=16 n=1048576 hot50 (contention 2.74e+05)
  0.72  index_put float16 rows=4096 d=64 n=1048576 hot50 (contention 2.63e+05)

index_put float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| ~1 | 77 | 1.00 | 1.01 | 1.01 | 0.72 | 0.70 | 20 | 9e-08 / 9e-08 / 9e-08 |
| 1.5-8 | 81 | 1.00 | 0.58 | 1.05 | 0.23 | 0.23 | 21 | 2e-07 / 1e-07 / 1e-07 |
| 8-64 | 112 | 1.00 | 0.60 | 1.07 | 0.16 | 0.16 | 30 | 5e-07 / 4e-07 / 2e-07 |
| 64-1K | 86 | 1.00 | 0.85 | 1.16 | 0.15 | 0.15 | 28 | 1e-06 / 2e-06 / 2e-06 |
| 1K-16K | 249 | 1.00 | 1.40 | 1.97 | 0.15 | 0.15 | 75 | 1e-05 / 8e-06 / 7e-06 |
| >=16K | 278 | 1.00 | 3.73 | 6.54 | 0.59 | 0.51 | 54 | 2e-04 / 3e-04 / 9e-05 |
| all | 883 | 1.00 | 1.46 | 2.25 | 0.15 | 0.15 | 228 | 2e-04 / 3e-04 / 9e-05 |
  0.15  index_put float32 rows=4096 d=1 n=4194304 runs32 (contention 1.05e+03)
  0.15  index_put float32 rows=65536 d=1 n=4194304 sorted (contention 65)
  0.15  index_put float32 rows=65536 d=1 n=4194304 runs32 (contention 95.7)
  0.16  index_put float32 rows=262144 d=1 n=4194304 runs32 (contention 48.1)
  0.16  index_put float32 rows=1048576 d=1 n=4194304 runs32 (contention 35.9)

index_put int32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.00 | 0.37 | 1.52 | 1.43 | 1.43 | 0 | 0e+00 / 0e+00 / 0e+00 |
| 64-1K | 2 | 1.00 | 1.50 | 2.29 | 1.70 | 1.18 | 0 | 0e+00 / 0e+00 / 0e+00 |
| >=16K | 8 | 1.00 | 8.16 | 13.43 | 1.56 | 0.97 | 0 | 0e+00 / 0e+00 / 0e+00 |
| all | 12 | 1.00 | 3.68 | 6.95 | 1.43 | 0.97 | 0 | 0e+00 / 0e+00 / 0e+00 |

index_put int64

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.00 | 0.74 | 6.28 | 4.96 | 4.96 | 0 | 0e+00 / 0e+00 / 0e+00 |
| 64-1K | 2 | 1.00 | 0.96 | 7.72 | 5.67 | 5.67 | 0 | 0e+00 / 0e+00 / 0e+00 |
| >=16K | 8 | 1.00 | 15.51 | 286.27 | 39.98 | 6.50 | 0 | 0e+00 / 0e+00 / 0e+00 |
| all | 12 | 1.00 | 5.88 | 82.94 | 4.96 | 4.96 | 0 | 0e+00 / 0e+00 / 0e+00 |

index_put_2d float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.00 | 1.24 | 1.03 | 1.03 | 0 | 1e-07 / 2e-07 / 1e-07 |
| 8-64 | 4 | 1.00 | 1.00 | 1.68 | 1.56 | 1.56 | 0 | 4e-07 / 4e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 1.00 | 1.92 | 1.07 | 1.07 | 0 | 8e-07 / 8e-07 / 9e-07 |
| 1K-16K | 6 | 1.00 | 1.01 | 1.24 | 1.07 | 1.06 | 0 | 6e-06 / 5e-06 / 4e-06 |
| >=16K | 24 | 1.00 | 1.02 | 7.11 | 1.27 | 1.27 | 0 | 2e-05 / 3e-05 / 5e-06 |
| all | 48 | 1.00 | 1.01 | 3.22 | 1.03 | 1.03 | 0 | 2e-05 / 3e-05 / 5e-06 |

masked_index_put float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.01 | 1.14 | 1.02 | 0.96 | 1 | 2e-07 / 2e-07 / 1e-07 |
| 8-64 | 4 | 1.00 | 1.00 | 1.49 | 1.37 | 1.36 | 0 | 3e-07 / 3e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 0.98 | 1.78 | 1.08 | 1.08 | 0 | 8e-07 / 8e-07 / 8e-07 |
| 1K-16K | 6 | 1.00 | 1.00 | 1.17 | 0.78 | 0.78 | 1 | 6e-06 / 7e-06 / 5e-06 |
| >=16K | 24 | 1.00 | 0.96 | 6.56 | 1.07 | 1.07 | 0 | 2e-05 / 2e-05 / 3e-06 |
| all | 48 | 1.00 | 0.98 | 2.97 | 0.78 | 0.78 | 2 | 2e-05 / 2e-05 / 5e-06 |
  0.78  masked_index_put float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.2e+03)
  0.96  masked_index_put float32 rows=262144 d=128 n=1048576 uniform (contention 5)

moe_combine float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.62 | 1.19 | 0.68 | 0.68 | 2 | 2e-07 / 1e-07 / 2e-07 |
| 8-64 | 4 | 1.00 | 0.67 | 1.24 | 0.98 | 0.98 | 0 | 3e-07 / 3e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 1.31 | 1.54 | 0.95 | 0.75 | 2 | 8e-07 / 1e-07 / 8e-07 |
| 1K-16K | 6 | 1.00 | 0.85 | 1.25 | 1.03 | 0.69 | 2 | 6e-06 / 4e-06 / 5e-06 |
| >=16K | 24 | 1.00 | 3.56 | 5.19 | 0.62 | 0.62 | 10 | 2e-05 / 2e-05 / 5e-06 |
| all | 48 | 1.00 | 1.71 | 2.59 | 0.62 | 0.62 | 16 | 2e-05 / 2e-05 / 5e-06 |
  0.62  moe_combine float32 rows=64 d=128 n=1048576 uniform (contention 1.64e+04)
  0.67  moe_combine float32 rows=64 d=128 n=1048576 runs32 (contention 1.64e+04)
  0.68  moe_combine float32 rows=262144 d=128 n=1048576 sorted (contention 5)
  0.69  moe_combine float32 rows=4096 d=128 n=1048576 fewhot (contention 4.44e+03)
  0.69  moe_combine float32 rows=64 d=128 n=1048576 fewhot (contention 1.99e+04)

scatter_add bfloat16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.01 | 1.36 | 1.03 | 1.03 | 0 | 1e-02 / 1e-02 / 3e-03 |
| 8-64 | 4 | 1.00 | 1.00 | 1.52 | 1.14 | 1.14 | 0 | 3e-02 / 2e-02 / 3e-03 |
| 64-1K | 12 | 1.00 | 1.01 | 2.22 | 1.54 | 1.54 | 0 | 7e-02 / 6e-02 / 4e-03 |
| 1K-16K | 14 | 1.00 | 1.00 | 1.38 | 0.74 | 0.74 | 3 | 4e-01 / 4e-01 / 3e-03 |
| >=16K | 34 | 1.00 | 1.00 | 6.90 | 1.63 | 1.63 | 0 | 8e-01 / 8e-01 / 3e-03 |
| all | 72 | 1.00 | 1.00 | 3.21 | 0.74 | 0.74 | 3 | 8e-01 / 8e-01 / 4e-03 |
  0.74  scatter_add bfloat16 rows=262144 d=128 n=1048576 fewhot (contention 4.22e+03)
  0.82  scatter_add bfloat16 rows=262144 d=512 n=262144 zipf (contention 6.52e+03)
  0.93  scatter_add bfloat16 rows=262144 d=512 n=262144 fewhot (contention 1.06e+03)

scatter_add float16

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.02 | 1.06 | 0.77 | 0.68 | 2 | 1e-03 / 1e-03 / 1e-03 |
| 8-64 | 4 | 1.00 | 1.01 | 1.25 | 1.06 | 1.03 | 0 | 3e-03 / 3e-03 / 8e-04 |
| 64-1K | 12 | 1.00 | 1.01 | 1.51 | 1.04 | 1.04 | 0 | 8e-03 / 7e-03 / 5e-03 |
| 1K-16K | 14 | 1.00 | 1.01 | 1.52 | 0.82 | 0.82 | 2 | 7e-02 / 5e-02 / 3e-02 |
| >=16K | 34 | 1.00 | 1.00 | 7.97 | 1.36 | 1.36 | 0 | 2e-01 / 2e-01 / 5e-02 |
| all | 72 | 1.00 | 1.00 | 3.16 | 0.77 | 0.68 | 4 | 2e-01 / 2e-01 / 5e-02 |
  0.68  scatter_add float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.80  scatter_add float16 rows=262144 d=64 n=1048576 sorted (contention 5)
  0.82  scatter_add float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)
  0.83  scatter_add float16 rows=4096 d=64 n=1048576 fewhot (contention 4.41e+03)

scatter_add float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.00 | 1.33 | 1.20 | 1.20 | 0 | 1e-07 / 2e-07 / 1e-07 |
| 8-64 | 4 | 1.00 | 0.99 | 1.70 | 1.62 | 1.62 | 0 | 3e-07 / 4e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 0.99 | 2.09 | 1.49 | 1.49 | 0 | 8e-07 / 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.00 | 1.00 | 1.44 | 1.29 | 1.28 | 0 | 5e-06 / 5e-06 / 6e-06 |
| >=16K | 24 | 1.00 | 0.98 | 7.02 | 1.19 | 1.19 | 0 | 3e-05 / 3e-05 / 4e-06 |
| all | 48 | 1.00 | 0.99 | 3.33 | 1.19 | 1.19 | 0 | 3e-05 / 3e-05 / 6e-06 |

scatter_add_dim1 float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.00 | 0.69 | 0.16 | 0.16 | 5 | 1e-07 / 1e-07 / 1e-07 |
| 8-64 | 4 | 1.00 | 1.00 | 1.19 | 1.09 | 1.09 | 0 | 1e-07 / 1e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 1.00 | 1.18 | 0.99 | 0.99 | 0 | 7e-07 / 7e-07 / 7e-07 |
| 1K-16K | 6 | 1.00 | 0.99 | 1.51 | 0.77 | 0.76 | 1 | 4e-06 / 4e-06 / 6e-06 |
| >=16K | 24 | 1.00 | 1.01 | 2.93 | 1.09 | 1.09 | 0 | 4e-06 / 6e-06 / 2e-06 |
| all | 48 | 1.00 | 1.00 | 1.75 | 0.16 | 0.16 | 6 | 4e-06 / 6e-06 / 6e-06 |
  0.16  scatter_add_dim1 float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.54  scatter_add_dim1 float32 rows=1048576 d=128 n=1048576 sorted (contention 2)
  0.57  scatter_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)
  0.76  scatter_add_dim1 float32 rows=1048576 d=128 n=1048576 fewhot (contention 4.21e+03)
  0.80  scatter_add_dim1 float32 rows=1048576 d=32 n=1048576 sorted (contention 2)

scatter_reduce float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.00 | 1.32 | 1.20 | 1.20 | 0 | 2e-07 / 2e-07 / 1e-07 |
| 8-64 | 4 | 1.00 | 1.00 | 1.70 | 1.62 | 1.62 | 0 | 3e-07 / 3e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 1.01 | 2.09 | 1.50 | 1.50 | 0 | 8e-07 / 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.00 | 1.00 | 1.31 | 0.98 | 0.98 | 0 | 5e-06 / 5e-06 / 5e-06 |
| >=16K | 24 | 1.00 | 1.00 | 7.05 | 1.19 | 1.19 | 0 | 2e-05 / 3e-05 / 4e-06 |
| all | 48 | 1.00 | 1.00 | 3.30 | 0.98 | 0.98 | 0 | 2e-05 / 3e-05 / 5e-06 |

scatter_reduce_noself float32

| contention | cases | base | ps | adaptive | worst vs base | worst vs best | slower | max err |
|---|---|---|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 1.00 | 1.26 | 1.14 | 1.14 | 0 | 2e-07 / 2e-07 / 1e-07 |
| 8-64 | 4 | 1.00 | 1.00 | 1.58 | 1.51 | 1.51 | 0 | 4e-07 / 4e-07 / 1e-07 |
| 64-1K | 6 | 1.00 | 0.98 | 1.84 | 1.34 | 1.34 | 0 | 8e-07 / 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.00 | 1.00 | 1.27 | 1.18 | 1.17 | 0 | 5e-06 / 4e-06 / 4e-06 |
| >=16K | 24 | 1.00 | 1.00 | 5.72 | 1.14 | 1.14 | 0 | 2e-05 / 3e-05 / 3e-06 |
| all | 48 | 1.00 | 1.00 | 2.87 | 1.14 | 1.14 | 0 | 2e-05 / 3e-05 / 4e-06 |

Overall, adaptive vs base and vs best(base, ps):

| dtype | cases | geomean vs base | worst vs base | geomean vs best | worst vs best | < 0.95 vs best |
|---|---|---|---|---|---|---|
| all | 2281 | 3.46 | 0.15 | 2.02 | 0.15 | 359 |
| bfloat16 | 360 | 14.02 | 0.74 | 5.36 | 0.74 | 13 |
| float16 | 366 | 3.11 | 0.65 | 1.79 | 0.45 | 73 |
| float32 | 1507 | 2.46 | 0.15 | 1.62 | 0.15 | 265 |
| int32 | 24 | 2.71 | 0.19 | 1.08 | 0.19 | 8 |
| int64 | 24 | 41.07 | 4.19 | 9.77 | 4.19 | 0 |
