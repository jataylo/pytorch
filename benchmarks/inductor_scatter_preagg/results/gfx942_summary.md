Speedup of preagg over regular Inductor (base) by measured contention;
'slower' counts cases below 0.97, errors are the max, base / preagg.

embedding bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.15 | 0.86 | 2 | 3e-03 / 3e-03 |
| 8-64 | 4 | 1.52 | 1.32 | 0 | 3e-03 / 3e-03 |
| 64-1K | 12 | 1.65 | 1.00 | 0 | 3e-03 / 3e-03 |
| 1K-16K | 14 | 1.63 | 1.01 | 0 | 3e-03 / 3e-03 |
| >=16K | 34 | 7.27 | 1.12 | 0 | 4e-03 / 4e-03 |
| all | 72 | 3.17 | 0.86 | 2 | 4e-03 / 4e-03 |
  0.86  embedding bfloat16 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.89  embedding bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)

embedding float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.16 | 0.89 | 2 | 3e-04 / 3e-04 |
| 8-64 | 4 | 1.53 | 1.35 | 0 | 4e-04 / 4e-04 |
| 64-1K | 12 | 1.65 | 1.09 | 0 | 4e-04 / 4e-04 |
| 1K-16K | 14 | 1.59 | 1.02 | 0 | 4e-04 / 4e-04 |
| >=16K | 34 | 7.28 | 1.11 | 0 | 4e-04 / 4e-04 |
| all | 72 | 3.16 | 0.89 | 2 | 4e-04 / 4e-04 |
  0.89  embedding float16 rows=262144 d=512 n=262144 uniform (contention 2.01)
  0.90  embedding float16 rows=262144 d=128 n=1048576 uniform (contention 5)

embedding float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.14 | 0.93 | 1 | 1e-07 / 2e-07 |
| 8-64 | 4 | 1.42 | 1.33 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.75 | 1.07 | 0 | 7e-07 / 6e-07 |
| 1K-16K | 6 | 1.19 | 0.98 | 0 | 7e-06 / 5e-06 |
| >=16K | 24 | 6.66 | 1.46 | 0 | 3e-05 / 3e-06 |
| all | 48 | 2.98 | 0.93 | 1 | 3e-05 / 5e-06 |
  0.93  embedding float32 rows=262144 d=128 n=1048576 uniform (contention 5)

embedding_pad float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.10 | 0.88 | 2 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.40 | 1.28 | 0 | 5e-07 / 1e-07 |
| 64-1K | 6 | 1.72 | 0.93 | 1 | 9e-07 / 6e-07 |
| 1K-16K | 6 | 1.24 | 0.98 | 0 | 5e-06 / 4e-06 |
| >=16K | 24 | 6.59 | 1.45 | 0 | 2e-05 / 3e-06 |
| all | 48 | 2.95 | 0.88 | 3 | 2e-05 / 4e-06 |
  0.88  embedding_pad float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.93  embedding_pad float32 rows=4096 d=128 n=1048576 uniform (contention 257)
  0.96  embedding_pad float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

fused_producer bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.49 | 0.97 | 0 | 9e-03 / 3e-03 |
| 8-64 | 4 | 6.32 | 1.07 | 0 | 3e-02 / 4e-03 |
| 64-1K | 12 | 6.57 | 1.83 | 0 | 2e-01 / 3e-03 |
| 1K-16K | 14 | 27.02 | 8.07 | 0 | 1e+00 / 4e-03 |
| >=16K | 34 | 185.74 | 23.17 | 0 | 1e+00 / 4e-03 |
| all | 72 | 38.98 | 0.97 | 0 | 1e+00 / 4e-03 |

fused_producer float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.20 | 0.89 | 1 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.14 | 0.91 | 1 | 4e-03 / 9e-04 |
| 64-1K | 12 | 1.52 | 1.04 | 0 | 8e-03 / 7e-03 |
| 1K-16K | 14 | 1.58 | 0.92 | 1 | 9e-01 / 5e-01 |
| >=16K | 34 | 7.79 | 1.23 | 0 | 1e+00 / inf |
| all | 72 | 3.18 | 0.89 | 3 | 1e+00 / inf |
  0.89  fused_producer float16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.91  fused_producer float16 rows=262144 d=512 n=262144 runs32 (contention 33.1)
  0.92  fused_producer float16 rows=4096 d=64 n=1048576 fewhot (contention 4.45e+03)

fused_producer float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.44 | 1.23 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.70 | 1.58 | 0 | 5e-07 / 1e-07 |
| 64-1K | 6 | 2.13 | 1.61 | 0 | 1e-06 / 8e-07 |
| 1K-16K | 6 | 1.42 | 1.28 | 0 | 8e-06 / 8e-06 |
| >=16K | 24 | 6.73 | 1.22 | 0 | 3e-05 / 4e-06 |
| all | 48 | 3.31 | 1.22 | 0 | 3e-05 / 8e-06 |

gnn float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.11 | 0.98 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.37 | 1.30 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.53 | 0.96 | 1 | 9e-07 / 6e-07 |
| 1K-16K | 6 | 1.14 | 0.95 | 1 | 5e-06 / 6e-06 |
| >=16K | 24 | 5.86 | 1.53 | 0 | 4e-05 / 4e-06 |
| all | 48 | 2.71 | 0.95 | 2 | 4e-05 / 6e-06 |
  0.95  gnn float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.24e+03)
  0.96  gnn float32 rows=4096 d=128 n=1048576 uniform (contention 257)

hist int32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.22 | 0.97 | 1 | 0e+00 / 0e+00 |
| 64-1K | 2 | 1.09 | 0.98 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 2.52 | 2.23 | 0 | 0e+00 / 0e+00 |
| all | 12 | 1.94 | 0.97 | 1 | 0e+00 / 0e+00 |
  0.97  hist int32 rows=262144 d=1 n=1048576 sorted (contention 5.01)

hist int64

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 13.87 | 11.52 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 16.20 | 12.87 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 52.77 | 8.35 | 0 | 0e+00 / 0e+00 |
| all | 12 | 34.69 | 8.35 | 0 | 0e+00 / 0e+00 |

index_add bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.42 | 0.86 | 1 | 1e-02 / 3e-03 |
| 8-64 | 4 | 6.36 | 0.95 | 1 | 2e-02 / 4e-03 |
| 64-1K | 12 | 6.61 | 1.62 | 0 | 7e-02 / 3e-03 |
| 1K-16K | 14 | 28.75 | 8.10 | 0 | 6e-01 / 3e-03 |
| >=16K | 34 | 192.55 | 24.51 | 0 | 7e-01 / 3e-03 |
| all | 72 | 40.09 | 0.86 | 2 | 7e-01 / 4e-03 |
  0.86  index_add bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.95  index_add bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33)

index_add float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.19 | 0.91 | 1 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.30 | 1.05 | 0 | 4e-03 / 7e-04 |
| 64-1K | 12 | 1.52 | 0.99 | 0 | 7e-03 / 6e-03 |
| 1K-16K | 14 | 1.61 | 0.95 | 1 | 7e-02 / 4e-02 |
| >=16K | 34 | 7.89 | 1.23 | 0 | 2e-01 / 4e-02 |
| all | 72 | 3.23 | 0.91 | 2 | 2e-01 / 4e-02 |
  0.91  index_add float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.95  index_add float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)

index_add float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.41 | 1.19 | 0 | 1e-07 / 2e-07 |
| 8-64 | 4 | 1.71 | 1.61 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 2.13 | 1.50 | 0 | 1e-06 / 6e-07 |
| 1K-16K | 6 | 1.22 | 1.00 | 0 | 6e-06 / 5e-06 |
| >=16K | 24 | 7.01 | 1.18 | 0 | 3e-05 / 3e-06 |
| all | 48 | 3.30 | 1.00 | 0 | 3e-05 / 5e-06 |

index_add_dim1 float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.30 | 0.87 | 1 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.39 | 1.32 | 0 | 1e-07 / 1e-07 |
| 64-1K | 6 | 1.51 | 1.45 | 0 | 7e-07 / 8e-07 |
| 1K-16K | 6 | 1.89 | 1.16 | 0 | 4e-06 / 4e-06 |
| >=16K | 24 | 3.01 | 1.14 | 0 | 5e-06 / 2e-06 |
| all | 48 | 2.12 | 0.87 | 1 | 5e-06 / 4e-06 |
  0.87  index_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

index_put bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.45 | 0.86 | 1 | 1e-02 / 3e-03 |
| 8-64 | 4 | 6.42 | 0.95 | 1 | 3e-02 / 3e-03 |
| 64-1K | 12 | 6.60 | 1.62 | 0 | 5e-02 / 3e-03 |
| 1K-16K | 14 | 28.28 | 8.00 | 0 | 3e-01 / 3e-03 |
| >=16K | 34 | 190.93 | 23.54 | 0 | 8e-01 / 3e-03 |
| all | 72 | 39.86 | 0.86 | 2 | 8e-01 / 3e-03 |
  0.86  index_put bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.95  index_put bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33.1)

index_put float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 10 | 1.33 | 0.92 | 1 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.28 | 1.04 | 0 | 3e-03 / 7e-04 |
| 64-1K | 14 | 1.63 | 0.99 | 0 | 6e-03 / 6e-03 |
| 1K-16K | 14 | 1.63 | 0.92 | 1 | 6e-02 / 4e-02 |
| >=16K | 36 | 8.17 | 1.23 | 0 | 2e-01 / 4e-02 |
| all | 78 | 3.30 | 0.92 | 2 | 2e-01 / 4e-02 |
  0.92  index_put float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.92  index_put float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)

index_put float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| ~1 | 77 | 1.07 | 0.88 | 8 | 8e-08 / 9e-08 |
| 1.5-8 | 81 | 1.26 | 0.82 | 7 | 2e-07 / 2e-07 |
| 8-64 | 112 | 1.42 | 0.98 | 0 | 6e-07 / 2e-07 |
| 64-1K | 86 | 1.44 | 0.80 | 11 | 2e-06 / 1e-06 |
| 1K-16K | 249 | 2.19 | 0.87 | 9 | 8e-06 / 7e-06 |
| >=16K | 278 | 7.02 | 1.03 | 0 | 1e-04 / 9e-04 |
| all | 883 | 2.57 | 0.80 | 35 | 1e-04 / 9e-04 |
  0.80  index_put float32 rows=65536 d=32 n=65536 fewhot (contention 262)
  0.82  index_put float32 rows=65536 d=8 n=262144 sorted (contention 5.01)
  0.83  index_put float32 rows=4096 d=48 n=65536 fewhot (contention 275)
  0.85  index_put float32 rows=4096 d=128 n=65536 fewhot (contention 279)
  0.87  index_put float32 rows=4194304 d=128 n=1048576 fewhot (contention 4.18e+03)
  0.88  index_put float32 rows=262144 d=128 n=65536 uniform (contention 1.25)
  0.89  index_put float32 rows=65536 d=128 n=65536 sorted (contention 2)
  0.89  index_put float32 rows=262144 d=48 n=65536 fewhot (contention 262)
  0.90  index_put float32 rows=65536 d=8 n=65536 fewhot (contention 264)
  0.91  index_put float32 rows=1048576 d=48 n=262144 fewhot (contention 1.05e+03)

index_put int32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.72 | 1.55 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 2.37 | 1.68 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 13.48 | 1.56 | 0 | 0e+00 / 0e+00 |
| all | 12 | 7.16 | 1.55 | 0 | 0e+00 / 0e+00 |

index_put int64

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 7.26 | 6.52 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 7.83 | 5.81 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 290.46 | 41.59 | 0 | 0e+00 / 0e+00 |
| all | 12 | 86.00 | 5.81 | 0 | 0e+00 / 0e+00 |

index_put_2d float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.26 | 1.02 | 0 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.70 | 1.60 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.94 | 1.07 | 0 | 9e-07 / 1e-06 |
| 1K-16K | 6 | 1.22 | 1.06 | 0 | 6e-06 / 4e-06 |
| >=16K | 24 | 6.94 | 1.27 | 0 | 3e-05 / 4e-06 |
| all | 48 | 3.19 | 1.02 | 0 | 3e-05 / 4e-06 |

masked_index_put float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.19 | 0.95 | 1 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.51 | 1.40 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.81 | 1.06 | 0 | 8e-07 / 6e-07 |
| 1K-16K | 6 | 1.19 | 0.89 | 1 | 6e-06 / 5e-06 |
| >=16K | 24 | 6.59 | 1.07 | 0 | 2e-05 / 3e-06 |
| all | 48 | 3.01 | 0.89 | 2 | 2e-05 / 5e-06 |
  0.89  masked_index_put float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.2e+03)
  0.95  masked_index_put float32 rows=262144 d=128 n=1048576 uniform (contention 5)

moe_combine float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.16 | 1.01 | 0 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.36 | 1.28 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.58 | 1.20 | 0 | 8e-07 / 8e-07 |
| 1K-16K | 6 | 1.20 | 1.03 | 0 | 6e-06 / 5e-06 |
| >=16K | 24 | 6.09 | 1.07 | 0 | 3e-05 / 3e-06 |
| all | 48 | 2.81 | 1.01 | 0 | 3e-05 / 5e-06 |

scatter_add bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.56 | 1.02 | 0 | 9e-03 / 3e-03 |
| 8-64 | 4 | 1.57 | 1.16 | 0 | 2e-02 / 3e-03 |
| 64-1K | 12 | 2.29 | 1.56 | 0 | 7e-02 / 4e-03 |
| 1K-16K | 14 | 1.55 | 0.93 | 1 | 4e-01 / 3e-03 |
| >=16K | 34 | 6.92 | 1.62 | 0 | 9e-01 / 3e-03 |
| all | 72 | 3.36 | 0.93 | 1 | 9e-01 / 4e-03 |
  0.93  scatter_add bfloat16 rows=262144 d=512 n=262144 zipf (contention 6.52e+03)

scatter_add float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.20 | 0.94 | 1 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.28 | 1.05 | 0 | 3e-03 / 8e-04 |
| 64-1K | 12 | 1.53 | 1.05 | 0 | 7e-03 / 5e-03 |
| 1K-16K | 14 | 1.60 | 0.92 | 2 | 7e-02 / 3e-02 |
| >=16K | 34 | 7.99 | 1.35 | 0 | 2e-01 / 6e-02 |
| all | 72 | 3.25 | 0.92 | 3 | 2e-01 / 6e-02 |
  0.92  scatter_add float16 rows=4096 d=64 n=1048576 fewhot (contention 4.41e+03)
  0.94  scatter_add float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.95  scatter_add float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)

scatter_add float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.36 | 1.21 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.72 | 1.62 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 2.10 | 1.49 | 0 | 8e-07 / 8e-07 |
| 1K-16K | 6 | 1.44 | 1.28 | 0 | 5e-06 / 4e-06 |
| >=16K | 24 | 7.09 | 1.18 | 0 | 3e-05 / 3e-06 |
| all | 48 | 3.37 | 1.18 | 0 | 3e-05 / 4e-06 |

scatter_add_dim1 float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.30 | 0.86 | 1 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.39 | 1.31 | 0 | 1e-07 / 1e-07 |
| 64-1K | 6 | 1.50 | 1.44 | 0 | 7e-07 / 7e-07 |
| 1K-16K | 6 | 1.81 | 1.13 | 0 | 5e-06 / 5e-06 |
| >=16K | 24 | 2.96 | 1.10 | 0 | 5e-06 / 2e-06 |
| all | 48 | 2.09 | 0.86 | 1 | 5e-06 / 5e-06 |
  0.86  scatter_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

scatter_reduce float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.40 | 1.21 | 0 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.73 | 1.64 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 2.10 | 1.50 | 0 | 7e-07 / 6e-07 |
| 1K-16K | 6 | 1.31 | 0.99 | 0 | 5e-06 / 4e-06 |
| >=16K | 24 | 7.04 | 1.18 | 0 | 2e-05 / 4e-06 |
| all | 48 | 3.33 | 0.99 | 0 | 2e-05 / 4e-06 |

scatter_reduce_noself float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.31 | 1.15 | 0 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.60 | 1.51 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.86 | 1.35 | 0 | 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.28 | 1.16 | 0 | 5e-06 / 4e-06 |
| >=16K | 24 | 5.81 | 1.14 | 0 | 3e-05 / 3e-06 |
| all | 48 | 2.92 | 1.14 | 0 | 3e-05 / 4e-06 |

Overall:

| dtype | cases | geomean | worst | < 0.95 |
|---|---|---|---|---|
| all | 2281 | 3.76 | 0.80 | 48 |
| bfloat16 | 360 | 14.60 | 0.86 | 5 |
| float16 | 366 | 3.22 | 0.89 | 12 |
| float32 | 1507 | 2.70 | 0.80 | 31 |
| int32 | 24 | 3.73 | 0.97 | 0 |
| int64 | 24 | 54.62 | 5.81 | 0 |
