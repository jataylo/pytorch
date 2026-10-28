Speedup of preagg over regular Inductor (base) by measured contention;
'slower' counts cases below 0.97, errors are the max, base / preagg.

embedding bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.05 | 0.86 | 3 | 3e-03 / 3e-03 |
| 8-64 | 4 | 1.54 | 1.33 | 0 | 3e-03 / 3e-03 |
| 64-1K | 12 | 1.62 | 0.97 | 0 | 3e-03 / 3e-03 |
| 1K-16K | 14 | 1.44 | 0.85 | 3 | 3e-03 / 3e-03 |
| >=16K | 34 | 7.34 | 1.12 | 0 | 4e-03 / 4e-03 |
| all | 72 | 3.07 | 0.85 | 6 | 4e-03 / 4e-03 |
  0.85  embedding bfloat16 rows=262144 d=128 n=1048576 fewhot (contention 4.2e+03)
  0.86  embedding bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.94  embedding bfloat16 rows=262144 d=16 n=1048576 uniform (contention 5)
  0.94  embedding bfloat16 rows=262144 d=512 n=262144 fewhot (contention 1.04e+03)
  0.95  embedding bfloat16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.96  embedding bfloat16 rows=4096 d=512 n=262144 fewhot (contention 1.12e+03)

embedding float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.03 | 0.86 | 4 | 3e-04 / 3e-04 |
| 8-64 | 4 | 1.54 | 1.34 | 0 | 4e-04 / 4e-04 |
| 64-1K | 12 | 1.61 | 0.97 | 1 | 4e-04 / 4e-04 |
| 1K-16K | 14 | 1.46 | 0.93 | 2 | 4e-04 / 4e-04 |
| >=16K | 34 | 7.35 | 1.11 | 0 | 4e-04 / 4e-04 |
| all | 72 | 3.07 | 0.86 | 7 | 4e-04 / 4e-04 |
  0.86  embedding float16 rows=262144 d=512 n=262144 uniform (contention 2.01)
  0.93  embedding float16 rows=262144 d=128 n=1048576 sorted (contention 5)
  0.93  embedding float16 rows=4096 d=512 n=262144 fewhot (contention 1.1e+03)
  0.95  embedding float16 rows=262144 d=16 n=1048576 uniform (contention 5)
  0.95  embedding float16 rows=262144 d=512 n=262144 fewhot (contention 1.05e+03)
  0.95  embedding float16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.97  embedding float16 rows=4096 d=512 n=262144 uniform (contention 64.9)

embedding float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.10 | 0.93 | 1 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.41 | 1.32 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.75 | 1.07 | 0 | 1e-06 / 7e-07 |
| 1K-16K | 6 | 1.19 | 0.99 | 0 | 7e-06 / 5e-06 |
| >=16K | 24 | 6.68 | 1.45 | 0 | 2e-05 / 4e-06 |
| all | 48 | 2.96 | 0.93 | 1 | 2e-05 / 5e-06 |
  0.93  embedding float32 rows=262144 d=128 n=1048576 uniform (contention 5)

embedding_pad float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.07 | 0.88 | 2 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.43 | 1.33 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.73 | 0.94 | 1 | 1e-06 / 7e-07 |
| 1K-16K | 6 | 1.24 | 0.98 | 0 | 5e-06 / 5e-06 |
| >=16K | 24 | 6.52 | 1.45 | 0 | 2e-05 / 4e-06 |
| all | 48 | 2.93 | 0.88 | 3 | 2e-05 / 5e-06 |
  0.88  embedding_pad float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.94  embedding_pad float32 rows=4096 d=128 n=1048576 uniform (contention 257)
  0.96  embedding_pad float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

fused_producer bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.17 | 0.98 | 0 | 9e-03 / 3e-03 |
| 8-64 | 4 | 6.30 | 1.07 | 0 | 3e-02 / 4e-03 |
| 64-1K | 12 | 6.32 | 1.78 | 0 | 2e-01 / 3e-03 |
| 1K-16K | 14 | 24.44 | 7.58 | 0 | 1e+00 / 4e-03 |
| >=16K | 34 | 183.98 | 23.42 | 0 | 1e+00 / 4e-03 |
| all | 72 | 37.41 | 0.98 | 0 | 1e+00 / 4e-03 |

fused_producer float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.69 | 4 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.15 | 0.89 | 1 | 4e-03 / 9e-04 |
| 64-1K | 12 | 1.34 | 0.65 | 2 | 9e-03 / 7e-03 |
| 1K-16K | 14 | 1.39 | 0.81 | 1 | 9e-01 / 5e-01 |
| >=16K | 34 | 7.54 | 1.23 | 0 | 1e+00 / inf |
| all | 72 | 2.93 | 0.65 | 8 | 1e+00 / inf |
  0.65  fused_producer float16 rows=4096 d=512 n=262144 sorted (contention 65)
  0.69  fused_producer float16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.79  fused_producer float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.80  fused_producer float16 rows=262144 d=64 n=1048576 sorted (contention 5)
  0.81  fused_producer float16 rows=4096 d=64 n=1048576 fewhot (contention 4.45e+03)
  0.83  fused_producer float16 rows=4096 d=512 n=262144 runs32 (contention 97)
  0.89  fused_producer float16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.89  fused_producer float16 rows=262144 d=512 n=262144 runs32 (contention 33.1)

fused_producer float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.35 | 1.23 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.70 | 1.62 | 0 | 5e-07 / 1e-07 |
| 64-1K | 6 | 2.14 | 1.60 | 0 | 1e-06 / 9e-07 |
| 1K-16K | 6 | 1.41 | 1.29 | 0 | 8e-06 / 9e-06 |
| >=16K | 24 | 6.74 | 1.22 | 0 | 3e-05 / 7e-06 |
| all | 48 | 3.27 | 1.22 | 0 | 3e-05 / 9e-06 |

gnn float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.08 | 0.98 | 0 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.36 | 1.29 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.53 | 0.96 | 1 | 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.14 | 0.95 | 1 | 4e-06 / 4e-06 |
| >=16K | 24 | 5.82 | 1.48 | 0 | 4e-05 / 4e-06 |
| all | 48 | 2.69 | 0.95 | 2 | 4e-05 / 4e-06 |
  0.95  gnn float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.24e+03)
  0.96  gnn float32 rows=4096 d=128 n=1048576 uniform (contention 257)

hist int32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 0.95 | 0.70 | 1 | 0e+00 / 0e+00 |
| 64-1K | 2 | 0.94 | 0.82 | 1 | 0e+00 / 0e+00 |
| >=16K | 8 | 2.55 | 2.24 | 0 | 0e+00 / 0e+00 |
| all | 12 | 1.83 | 0.70 | 2 | 0e+00 / 0e+00 |
  0.70  hist int32 rows=262144 d=1 n=1048576 sorted (contention 5.01)
  0.82  hist int32 rows=4096 d=1 n=1048576 sorted (contention 257)

hist int64

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 11.50 | 11.33 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 14.28 | 13.06 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 52.92 | 8.31 | 0 | 0e+00 / 0e+00 |
| all | 12 | 32.98 | 8.31 | 0 | 0e+00 / 0e+00 |

index_add bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.02 | 0.87 | 2 | 1e-02 / 3e-03 |
| 8-64 | 4 | 6.42 | 0.95 | 1 | 2e-02 / 4e-03 |
| 64-1K | 12 | 6.36 | 1.56 | 0 | 7e-02 / 3e-03 |
| 1K-16K | 14 | 25.78 | 7.09 | 0 | 6e-01 / 3e-03 |
| >=16K | 34 | 193.81 | 24.00 | 0 | 7e-01 / 3e-03 |
| all | 72 | 38.60 | 0.87 | 3 | 7e-01 / 4e-03 |
  0.87  index_add bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.89  index_add bfloat16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.95  index_add bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33)

index_add float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.02 | 0.68 | 3 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.28 | 1.01 | 0 | 3e-03 / 7e-04 |
| 64-1K | 12 | 1.51 | 0.98 | 0 | 7e-03 / 6e-03 |
| 1K-16K | 14 | 1.57 | 0.84 | 1 | 7e-02 / 5e-02 |
| >=16K | 34 | 7.88 | 1.24 | 0 | 3e-01 / 4e-02 |
| all | 72 | 3.15 | 0.68 | 4 | 3e-01 / 5e-02 |
  0.68  index_add float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.80  index_add float16 rows=262144 d=64 n=1048576 sorted (contention 5)
  0.84  index_add float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)
  0.96  index_add float16 rows=262144 d=64 n=1048576 uniform (contention 5)

index_add float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.33 | 1.19 | 0 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.72 | 1.62 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 2.12 | 1.50 | 0 | 1e-06 / 6e-07 |
| 1K-16K | 6 | 1.19 | 0.92 | 1 | 5e-06 / 6e-06 |
| >=16K | 24 | 6.96 | 1.19 | 0 | 3e-05 / 6e-06 |
| all | 48 | 3.25 | 0.92 | 1 | 3e-05 / 6e-06 |
  0.92  index_add float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.2e+03)

index_add_dim1 float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.30 | 0.87 | 1 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.39 | 1.33 | 0 | 1e-07 / 1e-07 |
| 64-1K | 6 | 1.51 | 1.44 | 0 | 7e-07 / 7e-07 |
| 1K-16K | 6 | 1.84 | 1.10 | 0 | 4e-06 / 4e-06 |
| >=16K | 24 | 3.02 | 1.14 | 0 | 5e-06 / 2e-06 |
| all | 48 | 2.12 | 0.87 | 1 | 5e-06 / 4e-06 |
  0.87  index_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

index_put bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.02 | 0.87 | 2 | 1e-02 / 3e-03 |
| 8-64 | 4 | 6.49 | 0.96 | 1 | 3e-02 / 3e-03 |
| 64-1K | 12 | 6.37 | 1.56 | 0 | 5e-02 / 3e-03 |
| 1K-16K | 14 | 25.40 | 6.87 | 0 | 3e-01 / 3e-03 |
| >=16K | 34 | 190.08 | 23.90 | 0 | 8e-01 / 3e-03 |
| all | 72 | 38.18 | 0.87 | 3 | 8e-01 / 3e-03 |
  0.87  index_put bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.90  index_put bfloat16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.96  index_put bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33.1)

index_put float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 10 | 1.17 | 0.76 | 2 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.28 | 1.03 | 0 | 4e-03 / 7e-04 |
| 64-1K | 14 | 1.62 | 0.98 | 0 | 7e-03 / 6e-03 |
| 1K-16K | 14 | 1.57 | 0.85 | 1 | 6e-02 / 4e-02 |
| >=16K | 36 | 8.13 | 1.25 | 0 | 2e-01 / 5e-02 |
| all | 78 | 3.21 | 0.76 | 3 | 2e-01 / 5e-02 |
  0.76  index_put float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.78  index_put float16 rows=262144 d=64 n=1048576 sorted (contention 5)
  0.85  index_put float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)

index_put float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| ~1 | 77 | 1.08 | 0.86 | 5 | 9e-08 / 9e-08 |
| 1.5-8 | 81 | 1.20 | 0.63 | 10 | 2e-07 / 1e-07 |
| 8-64 | 112 | 1.34 | 0.72 | 14 | 7e-07 / 2e-07 |
| 64-1K | 86 | 1.37 | 0.71 | 19 | 2e-06 / 2e-06 |
| 1K-16K | 249 | 2.14 | 0.64 | 12 | 8e-06 / 8e-06 |
| >=16K | 278 | 6.99 | 1.02 | 0 | 5e-04 / 3e-04 |
| all | 883 | 2.50 | 0.63 | 60 | 5e-04 / 3e-04 |
  0.63  index_put float32 rows=65536 d=8 n=65536 sorted (contention 1.99)
  0.63  index_put float32 rows=65536 d=32 n=65536 uniform (contention 1.99)
  0.64  index_put float32 rows=1048576 d=1 n=1048576 fewhot (contention 4.2e+03)
  0.66  index_put float32 rows=1048576 d=1 n=4194304 sorted (contention 5)
  0.71  index_put float32 rows=65536 d=1 n=4194304 runs32 (contention 95.7)
  0.71  index_put float32 rows=262144 d=1 n=1048576 sorted (contention 5)
  0.72  index_put float32 rows=65536 d=1 n=4194304 sorted (contention 65)
  0.72  index_put float32 rows=1048576 d=1 n=262144 fewhot (contention 1.05e+03)
  0.72  index_put float32 rows=262144 d=1 n=4194304 runs32 (contention 48.1)
  0.73  index_put float32 rows=4096 d=1 n=4194304 runs32 (contention 1.05e+03)

index_put int32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.54 | 1.47 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 2.37 | 1.67 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 13.36 | 1.56 | 0 | 0e+00 / 0e+00 |
| all | 12 | 6.98 | 1.47 | 0 | 0e+00 / 0e+00 |

index_put int64

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 6.36 | 5.00 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 7.88 | 5.77 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 288.83 | 41.38 | 0 | 0e+00 / 0e+00 |
| all | 12 | 83.91 | 5.00 | 0 | 0e+00 / 0e+00 |

index_put_2d float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.24 | 1.03 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.70 | 1.59 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.94 | 1.07 | 0 | 9e-07 / 7e-07 |
| 1K-16K | 6 | 1.22 | 1.06 | 0 | 6e-06 / 4e-06 |
| >=16K | 24 | 6.94 | 1.27 | 0 | 3e-05 / 4e-06 |
| all | 48 | 3.18 | 1.03 | 0 | 3e-05 / 4e-06 |

masked_index_put float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.14 | 0.96 | 1 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.52 | 1.40 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.80 | 1.08 | 0 | 9e-07 / 8e-07 |
| 1K-16K | 6 | 1.21 | 0.96 | 1 | 5e-06 / 5e-06 |
| >=16K | 24 | 6.58 | 1.07 | 0 | 2e-05 / 4e-06 |
| all | 48 | 2.99 | 0.96 | 2 | 2e-05 / 5e-06 |
  0.96  masked_index_put float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.96  masked_index_put float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.2e+03)

moe_combine float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.10 | 0.96 | 1 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.37 | 1.30 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.60 | 1.21 | 0 | 7e-07 / 8e-07 |
| 1K-16K | 6 | 1.19 | 1.01 | 0 | 5e-06 / 5e-06 |
| >=16K | 24 | 6.10 | 1.08 | 0 | 2e-05 / 6e-06 |
| all | 48 | 2.79 | 0.96 | 1 | 2e-05 / 6e-06 |
  0.96  moe_combine float32 rows=262144 d=128 n=1048576 sorted (contention 5)

scatter_add bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.36 | 1.03 | 0 | 1e-02 / 3e-03 |
| 8-64 | 4 | 1.57 | 1.15 | 0 | 3e-02 / 3e-03 |
| 64-1K | 12 | 2.20 | 1.54 | 0 | 5e-02 / 4e-03 |
| 1K-16K | 14 | 1.37 | 0.75 | 3 | 4e-01 / 3e-03 |
| >=16K | 34 | 6.93 | 1.63 | 0 | 7e-01 / 3e-03 |
| all | 72 | 3.21 | 0.75 | 3 | 7e-01 / 4e-03 |
  0.75  scatter_add bfloat16 rows=262144 d=128 n=1048576 fewhot (contention 4.22e+03)
  0.81  scatter_add bfloat16 rows=262144 d=512 n=262144 zipf (contention 6.52e+03)
  0.93  scatter_add bfloat16 rows=262144 d=512 n=262144 fewhot (contention 1.06e+03)

scatter_add float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.04 | 0.68 | 2 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.28 | 1.04 | 0 | 3e-03 / 8e-04 |
| 64-1K | 12 | 1.51 | 1.04 | 0 | 6e-03 / 5e-03 |
| 1K-16K | 14 | 1.52 | 0.82 | 2 | 5e-02 / 3e-02 |
| >=16K | 34 | 7.92 | 1.36 | 0 | 3e-01 / 4e-02 |
| all | 72 | 3.14 | 0.68 | 4 | 3e-01 / 4e-02 |
  0.68  scatter_add float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.78  scatter_add float16 rows=262144 d=64 n=1048576 sorted (contention 5)
  0.82  scatter_add float16 rows=4096 d=64 n=1048576 fewhot (contention 4.41e+03)
  0.84  scatter_add float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)

scatter_add float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.33 | 1.20 | 0 | 1e-07 / 2e-07 |
| 8-64 | 4 | 1.72 | 1.61 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 2.09 | 1.49 | 0 | 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.42 | 1.25 | 0 | 6e-06 / 5e-06 |
| >=16K | 24 | 7.03 | 1.19 | 0 | 3e-05 / 4e-06 |
| all | 48 | 3.33 | 1.19 | 0 | 3e-05 / 5e-06 |

scatter_add_dim1 float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.30 | 0.87 | 1 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.39 | 1.32 | 0 | 1e-07 / 1e-07 |
| 64-1K | 6 | 1.49 | 1.43 | 0 | 7e-07 / 7e-07 |
| 1K-16K | 6 | 1.86 | 1.16 | 0 | 4e-06 / 5e-06 |
| >=16K | 24 | 2.94 | 1.15 | 0 | 6e-06 / 2e-06 |
| all | 48 | 2.09 | 0.87 | 1 | 6e-06 / 5e-06 |
  0.87  scatter_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

scatter_reduce float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.33 | 1.20 | 0 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.72 | 1.61 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 2.10 | 1.50 | 0 | 8e-07 / 6e-07 |
| 1K-16K | 6 | 1.29 | 0.96 | 1 | 5e-06 / 5e-06 |
| >=16K | 24 | 6.95 | 1.19 | 0 | 3e-05 / 4e-06 |
| all | 48 | 3.27 | 0.96 | 1 | 3e-05 / 5e-06 |
  0.96  scatter_reduce float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.19e+03)

scatter_reduce_noself float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.26 | 1.15 | 0 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.60 | 1.50 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.88 | 1.34 | 0 | 7e-07 / 7e-07 |
| 1K-16K | 6 | 1.27 | 1.18 | 0 | 5e-06 / 4e-06 |
| >=16K | 24 | 5.84 | 1.14 | 0 | 2e-05 / 4e-06 |
| all | 48 | 2.91 | 1.14 | 0 | 2e-05 / 4e-06 |

Overall:

| dtype | cases | geomean | worst | < 0.95 |
|---|---|---|---|---|
| all | 2281 | 3.66 | 0.63 | 100 |
| bfloat16 | 360 | 14.03 | 0.75 | 12 |
| float16 | 366 | 3.10 | 0.65 | 23 |
| float32 | 1507 | 2.65 | 0.63 | 63 |
| int32 | 24 | 3.58 | 0.70 | 2 |
| int64 | 24 | 52.61 | 5.00 | 0 |
