Speedup of preagg over regular Inductor (base) by measured contention;
'slower' counts cases below 0.97, errors are the max, base / preagg.

embedding bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.21 | 0.86 | 2 | 3e-03 / 3e-03 |
| 8-64 | 4 | 1.52 | 1.33 | 0 | 3e-03 / 3e-03 |
| 64-1K | 12 | 1.65 | 1.00 | 0 | 3e-03 / 3e-03 |
| 1K-16K | 14 | 1.60 | 1.01 | 0 | 3e-03 / 3e-03 |
| >=16K | 34 | 7.34 | 1.12 | 0 | 4e-03 / 4e-03 |
| all | 72 | 3.19 | 0.86 | 2 | 4e-03 / 4e-03 |
  0.86  embedding bfloat16 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.89  embedding bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)

embedding float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.16 | 0.89 | 2 | 3e-04 / 3e-04 |
| 8-64 | 4 | 1.52 | 1.34 | 0 | 4e-04 / 4e-04 |
| 64-1K | 12 | 1.64 | 1.09 | 0 | 4e-04 / 4e-04 |
| 1K-16K | 14 | 1.57 | 1.01 | 0 | 4e-04 / 4e-04 |
| >=16K | 34 | 7.31 | 1.11 | 0 | 4e-04 / 4e-04 |
| all | 72 | 3.16 | 0.89 | 2 | 4e-04 / 4e-04 |
  0.89  embedding float16 rows=262144 d=512 n=262144 uniform (contention 2.01)
  0.90  embedding float16 rows=262144 d=128 n=1048576 uniform (contention 5)

embedding float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.14 | 0.93 | 1 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.42 | 1.30 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.79 | 1.07 | 0 | 9e-07 / 7e-07 |
| 1K-16K | 6 | 1.18 | 0.93 | 1 | 7e-06 / 6e-06 |
| >=16K | 24 | 6.66 | 1.45 | 0 | 3e-05 / 4e-06 |
| all | 48 | 2.98 | 0.93 | 2 | 3e-05 / 6e-06 |
  0.93  embedding float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.93  embedding float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.19e+03)

embedding_pad float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.10 | 0.89 | 2 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.41 | 1.29 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.74 | 0.94 | 1 | 1e-06 / 7e-07 |
| 1K-16K | 6 | 1.26 | 0.99 | 0 | 5e-06 / 4e-06 |
| >=16K | 24 | 6.56 | 1.45 | 0 | 2e-05 / 4e-06 |
| all | 48 | 2.95 | 0.89 | 3 | 2e-05 / 4e-06 |
  0.89  embedding_pad float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.94  embedding_pad float32 rows=4096 d=128 n=1048576 uniform (contention 257)
  0.96  embedding_pad float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

fused_producer bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.52 | 0.99 | 0 | 9e-03 / 3e-03 |
| 8-64 | 4 | 6.28 | 1.05 | 0 | 3e-02 / 4e-03 |
| 64-1K | 12 | 6.53 | 1.84 | 0 | 2e-01 / 3e-03 |
| 1K-16K | 14 | 27.13 | 8.20 | 0 | 1e+00 / 4e-03 |
| >=16K | 34 | 187.07 | 23.32 | 0 | 1e+00 / 4e-03 |
| all | 72 | 39.13 | 0.99 | 0 | 1e+00 / 4e-03 |

fused_producer float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.20 | 0.89 | 1 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.16 | 0.90 | 1 | 4e-03 / 9e-04 |
| 64-1K | 12 | 1.52 | 1.02 | 0 | 8e-03 / 6e-03 |
| 1K-16K | 14 | 1.58 | 0.90 | 1 | 9e-01 / 5e-01 |
| >=16K | 34 | 7.76 | 1.23 | 0 | 1e+00 / inf |
| all | 72 | 3.17 | 0.89 | 3 | 1e+00 / inf |
  0.89  fused_producer float16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.90  fused_producer float16 rows=262144 d=512 n=262144 runs32 (contention 33.1)
  0.90  fused_producer float16 rows=4096 d=64 n=1048576 fewhot (contention 4.45e+03)

fused_producer float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.45 | 1.23 | 0 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.70 | 1.61 | 0 | 5e-07 / 1e-07 |
| 64-1K | 6 | 2.14 | 1.61 | 0 | 1e-06 / 8e-07 |
| 1K-16K | 6 | 1.42 | 1.29 | 0 | 8e-06 / 8e-06 |
| >=16K | 24 | 6.72 | 1.22 | 0 | 3e-05 / 7e-06 |
| all | 48 | 3.31 | 1.22 | 0 | 3e-05 / 8e-06 |

gnn float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.11 | 0.98 | 0 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.36 | 1.29 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.54 | 0.96 | 1 | 7e-07 / 6e-07 |
| 1K-16K | 6 | 1.15 | 0.99 | 0 | 5e-06 / 5e-06 |
| >=16K | 24 | 5.85 | 1.47 | 0 | 4e-05 / 4e-06 |
| all | 48 | 2.71 | 0.96 | 1 | 4e-05 / 5e-06 |
  0.96  gnn float32 rows=4096 d=128 n=1048576 uniform (contention 257)

hist int32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.14 | 0.99 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 1.06 | 0.98 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 2.43 | 1.67 | 0 | 0e+00 / 0e+00 |
| all | 12 | 1.87 | 0.98 | 0 | 0e+00 / 0e+00 |

hist int64

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 14.00 | 11.53 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 16.05 | 12.77 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 52.75 | 8.43 | 0 | 0e+00 / 0e+00 |
| all | 12 | 34.68 | 8.43 | 0 | 0e+00 / 0e+00 |

index_add bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.46 | 0.86 | 1 | 1e-02 / 3e-03 |
| 8-64 | 4 | 6.45 | 0.95 | 1 | 2e-02 / 4e-03 |
| 64-1K | 12 | 6.56 | 1.61 | 0 | 7e-02 / 3e-03 |
| 1K-16K | 14 | 28.86 | 8.08 | 0 | 6e-01 / 3e-03 |
| >=16K | 34 | 191.58 | 24.02 | 0 | 7e-01 / 3e-03 |
| all | 72 | 40.06 | 0.86 | 2 | 7e-01 / 4e-03 |
  0.86  index_add bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.95  index_add bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33)

index_add float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.18 | 0.88 | 1 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.29 | 1.06 | 0 | 4e-03 / 7e-04 |
| 64-1K | 12 | 1.52 | 0.99 | 0 | 6e-03 / 6e-03 |
| 1K-16K | 14 | 1.64 | 0.95 | 1 | 7e-02 / 4e-02 |
| >=16K | 34 | 7.95 | 1.25 | 0 | 3e-01 / 4e-02 |
| all | 72 | 3.25 | 0.88 | 2 | 3e-01 / 4e-02 |
  0.88  index_add float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.95  index_add float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)

index_add float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.42 | 1.19 | 0 | 1e-07 / 2e-07 |
| 8-64 | 4 | 1.73 | 1.66 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 2.09 | 1.50 | 0 | 9e-07 / 6e-07 |
| 1K-16K | 6 | 1.19 | 0.90 | 1 | 6e-06 / 5e-06 |
| >=16K | 24 | 7.02 | 1.19 | 0 | 2e-05 / 5e-06 |
| all | 48 | 3.29 | 0.90 | 1 | 2e-05 / 5e-06 |
  0.90  index_add float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.2e+03)

index_add_dim1 float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.29 | 0.86 | 1 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.39 | 1.32 | 0 | 1e-07 / 1e-07 |
| 64-1K | 6 | 1.49 | 1.45 | 0 | 6e-07 / 7e-07 |
| 1K-16K | 6 | 1.91 | 1.21 | 0 | 4e-06 / 4e-06 |
| >=16K | 24 | 2.98 | 1.13 | 0 | 6e-06 / 2e-06 |
| all | 48 | 2.11 | 0.86 | 1 | 6e-06 / 4e-06 |
  0.86  index_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

index_put bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.44 | 0.86 | 1 | 1e-02 / 3e-03 |
| 8-64 | 4 | 6.46 | 0.95 | 1 | 3e-02 / 3e-03 |
| 64-1K | 12 | 6.59 | 1.61 | 0 | 5e-02 / 3e-03 |
| 1K-16K | 14 | 28.46 | 8.02 | 0 | 3e-01 / 3e-03 |
| >=16K | 34 | 189.56 | 23.95 | 0 | 8e-01 / 3e-03 |
| all | 72 | 39.75 | 0.86 | 2 | 8e-01 / 3e-03 |
  0.86  index_put bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.95  index_put bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33.1)

index_put float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 10 | 1.33 | 0.89 | 1 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.29 | 1.05 | 0 | 3e-03 / 7e-04 |
| 64-1K | 14 | 1.64 | 0.99 | 0 | 7e-03 / 5e-03 |
| 1K-16K | 14 | 1.65 | 0.96 | 1 | 5e-02 / 4e-02 |
| >=16K | 36 | 8.24 | 1.25 | 0 | 2e-01 / 5e-02 |
| all | 78 | 3.33 | 0.89 | 2 | 2e-01 / 5e-02 |
  0.89  index_put float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.96  index_put float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)

index_put float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| ~1 | 77 | 1.07 | 0.86 | 7 | 9e-08 / 9e-08 |
| 1.5-8 | 81 | 1.25 | 0.82 | 6 | 2e-07 / 2e-07 |
| 8-64 | 112 | 1.43 | 0.84 | 9 | 5e-07 / 3e-07 |
| 64-1K | 86 | 1.39 | 0.76 | 11 | 2e-06 / 1e-06 |
| 1K-16K | 249 | 2.18 | 0.85 | 9 | 9e-06 / 7e-06 |
| >=16K | 278 | 7.01 | 1.06 | 0 | 7e-04 / 1e-04 |
| all | 883 | 2.55 | 0.76 | 42 | 7e-04 / 1e-04 |
  0.76  index_put float32 rows=4096 d=8 n=65536 fewhot (contention 277)
  0.79  index_put float32 rows=262144 d=8 n=65536 fewhot (contention 263)
  0.80  index_put float32 rows=65536 d=32 n=65536 fewhot (contention 262)
  0.82  index_put float32 rows=65536 d=8 n=262144 sorted (contention 5.01)
  0.84  index_put float32 rows=65536 d=8 n=65536 sorted (contention 1.99)
  0.84  index_put float32 rows=65536 d=8 n=65536 fewhot (contention 264)
  0.84  index_put float32 rows=4096 d=8 n=65536 uniform (contention 17)
  0.85  index_put float32 rows=4194304 d=128 n=1048576 fewhot (contention 4.18e+03)
  0.85  index_put float32 rows=65536 d=8 n=65536 uniform (contention 2)
  0.86  index_put float32 rows=1048576 d=8 n=65536 uniform (contention 1.06)

index_put int32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.74 | 1.61 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 2.35 | 1.68 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 13.32 | 1.56 | 0 | 0e+00 / 0e+00 |
| all | 12 | 7.11 | 1.56 | 0 | 0e+00 / 0e+00 |

index_put int64

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 7.26 | 6.50 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 7.81 | 5.80 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 287.92 | 39.79 | 0 | 0e+00 / 0e+00 |
| all | 12 | 85.47 | 5.80 | 0 | 0e+00 / 0e+00 |

index_put_2d float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.26 | 1.01 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.70 | 1.60 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.94 | 1.05 | 0 | 9e-07 / 8e-07 |
| 1K-16K | 6 | 1.22 | 1.05 | 0 | 6e-06 / 4e-06 |
| >=16K | 24 | 6.92 | 1.28 | 0 | 2e-05 / 4e-06 |
| all | 48 | 3.18 | 1.01 | 0 | 2e-05 / 4e-06 |

masked_index_put float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.19 | 0.95 | 1 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.52 | 1.40 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.81 | 1.06 | 0 | 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.22 | 0.98 | 0 | 5e-06 / 5e-06 |
| >=16K | 24 | 6.63 | 1.07 | 0 | 3e-05 / 4e-06 |
| all | 48 | 3.03 | 0.95 | 1 | 3e-05 / 5e-06 |
  0.95  masked_index_put float32 rows=262144 d=128 n=1048576 uniform (contention 5)

moe_combine float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.17 | 1.08 | 0 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.37 | 1.29 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.62 | 1.21 | 0 | 1e-06 / 7e-07 |
| 1K-16K | 6 | 1.21 | 1.04 | 0 | 6e-06 / 5e-06 |
| >=16K | 24 | 6.10 | 1.08 | 0 | 2e-05 / 3e-06 |
| all | 48 | 2.83 | 1.04 | 0 | 2e-05 / 5e-06 |

scatter_add bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.56 | 1.02 | 0 | 1e-02 / 3e-03 |
| 8-64 | 4 | 1.58 | 1.15 | 0 | 2e-02 / 3e-03 |
| 64-1K | 12 | 2.31 | 1.56 | 0 | 7e-02 / 4e-03 |
| 1K-16K | 14 | 1.55 | 0.93 | 1 | 4e-01 / 3e-03 |
| >=16K | 34 | 6.97 | 1.64 | 0 | 8e-01 / 3e-03 |
| all | 72 | 3.37 | 0.93 | 1 | 8e-01 / 4e-03 |
  0.93  scatter_add bfloat16 rows=262144 d=512 n=262144 zipf (contention 6.52e+03)

scatter_add float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.22 | 0.99 | 0 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.27 | 1.05 | 0 | 3e-03 / 8e-04 |
| 64-1K | 12 | 1.53 | 1.06 | 0 | 6e-03 / 6e-03 |
| 1K-16K | 14 | 1.61 | 0.94 | 2 | 6e-02 / 3e-02 |
| >=16K | 34 | 8.00 | 1.36 | 0 | 2e-01 / 5e-02 |
| all | 72 | 3.25 | 0.94 | 2 | 2e-01 / 5e-02 |
  0.94  scatter_add float16 rows=4096 d=64 n=1048576 fewhot (contention 4.41e+03)
  0.95  scatter_add float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)

scatter_add float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.42 | 1.21 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.72 | 1.62 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 2.09 | 1.49 | 0 | 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.44 | 1.25 | 0 | 6e-06 / 5e-06 |
| >=16K | 24 | 7.11 | 1.19 | 0 | 4e-05 / 4e-06 |
| all | 48 | 3.39 | 1.19 | 0 | 4e-05 / 5e-06 |

scatter_add_dim1 float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.29 | 0.86 | 1 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.39 | 1.32 | 0 | 1e-07 / 1e-07 |
| 64-1K | 6 | 1.50 | 1.44 | 0 | 6e-07 / 7e-07 |
| 1K-16K | 6 | 1.81 | 1.15 | 0 | 5e-06 / 6e-06 |
| >=16K | 24 | 2.95 | 1.15 | 0 | 5e-06 / 2e-06 |
| all | 48 | 2.08 | 0.86 | 1 | 5e-06 / 6e-06 |
  0.86  scatter_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

scatter_reduce float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.41 | 1.20 | 0 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.72 | 1.62 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 2.09 | 1.49 | 0 | 9e-07 / 6e-07 |
| 1K-16K | 6 | 1.30 | 0.93 | 1 | 5e-06 / 4e-06 |
| >=16K | 24 | 7.01 | 1.19 | 0 | 2e-05 / 4e-06 |
| all | 48 | 3.33 | 0.93 | 1 | 2e-05 / 4e-06 |
  0.93  scatter_reduce float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.19e+03)

scatter_reduce_noself float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.33 | 1.15 | 0 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.59 | 1.50 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.90 | 1.34 | 0 | 9e-07 / 7e-07 |
| 1K-16K | 6 | 1.28 | 1.18 | 0 | 5e-06 / 4e-06 |
| >=16K | 24 | 5.84 | 1.14 | 0 | 2e-05 / 4e-06 |
| all | 48 | 2.94 | 1.14 | 0 | 2e-05 / 4e-06 |

Overall:

| dtype | cases | geomean | worst | < 0.95 |
|---|---|---|---|---|
| all | 2281 | 3.75 | 0.76 | 56 |
| bfloat16 | 360 | 14.63 | 0.86 | 5 |
| float16 | 366 | 3.23 | 0.88 | 10 |
| float32 | 1507 | 2.69 | 0.76 | 41 |
| int32 | 24 | 3.64 | 0.98 | 0 |
| int64 | 24 | 54.44 | 5.80 | 0 |
