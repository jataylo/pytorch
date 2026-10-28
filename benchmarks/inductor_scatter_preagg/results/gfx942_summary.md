Speedup of preagg over regular Inductor (base) by measured contention;
'slower' counts cases below 0.97, errors are the max, base / preagg.

embedding bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.06 | 0.86 | 3 | 3e-03 / 3e-03 |
| 8-64 | 4 | 1.54 | 1.34 | 0 | 3e-03 / 3e-03 |
| 64-1K | 12 | 1.62 | 0.96 | 1 | 3e-03 / 3e-03 |
| 1K-16K | 14 | 1.45 | 0.86 | 3 | 3e-03 / 3e-03 |
| >=16K | 34 | 7.30 | 1.12 | 0 | 4e-03 / 4e-03 |
| all | 72 | 3.07 | 0.86 | 7 | 4e-03 / 4e-03 |
  0.86  embedding bfloat16 rows=262144 d=128 n=1048576 fewhot (contention 4.2e+03)
  0.86  embedding bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.93  embedding bfloat16 rows=262144 d=16 n=1048576 uniform (contention 5)
  0.94  embedding bfloat16 rows=262144 d=512 n=262144 fewhot (contention 1.04e+03)
  0.95  embedding bfloat16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.96  embedding bfloat16 rows=4096 d=512 n=262144 fewhot (contention 1.12e+03)
  0.96  embedding bfloat16 rows=4096 d=512 n=262144 uniform (contention 65)

embedding float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.06 | 0.86 | 3 | 3e-04 / 3e-04 |
| 8-64 | 4 | 1.53 | 1.34 | 0 | 4e-04 / 4e-04 |
| 64-1K | 12 | 1.61 | 0.98 | 0 | 4e-04 / 4e-04 |
| 1K-16K | 14 | 1.44 | 0.92 | 2 | 4e-04 / 4e-04 |
| >=16K | 34 | 7.33 | 1.11 | 0 | 4e-04 / 4e-04 |
| all | 72 | 3.07 | 0.86 | 5 | 4e-04 / 4e-04 |
  0.86  embedding float16 rows=262144 d=512 n=262144 uniform (contention 2.01)
  0.92  embedding float16 rows=4096 d=512 n=262144 fewhot (contention 1.1e+03)
  0.95  embedding float16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.96  embedding float16 rows=262144 d=512 n=262144 fewhot (contention 1.05e+03)
  0.97  embedding float16 rows=262144 d=16 n=1048576 uniform (contention 5)

embedding float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.09 | 0.93 | 1 | 1e-07 / 2e-07 |
| 8-64 | 4 | 1.41 | 1.32 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.79 | 1.08 | 0 | 1e-06 / 8e-07 |
| 1K-16K | 6 | 1.18 | 0.96 | 1 | 7e-06 / 5e-06 |
| >=16K | 24 | 6.68 | 1.45 | 0 | 2e-05 / 3e-06 |
| all | 48 | 2.96 | 0.93 | 2 | 2e-05 / 5e-06 |
  0.93  embedding float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.96  embedding float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.19e+03)

embedding_pad float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.09 | 0.88 | 2 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.40 | 1.31 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.75 | 0.93 | 1 | 9e-07 / 7e-07 |
| 1K-16K | 6 | 1.23 | 0.96 | 1 | 5e-06 / 3e-06 |
| >=16K | 24 | 6.58 | 1.45 | 0 | 2e-05 / 5e-06 |
| all | 48 | 2.94 | 0.88 | 4 | 2e-05 / 5e-06 |
  0.88  embedding_pad float32 rows=262144 d=128 n=1048576 uniform (contention 5)
  0.93  embedding_pad float32 rows=4096 d=128 n=1048576 uniform (contention 257)
  0.96  embedding_pad float32 rows=1048576 d=128 n=1048576 uniform (contention 2)
  0.96  embedding_pad float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.21e+03)

fused_producer bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.18 | 0.97 | 0 | 9e-03 / 3e-03 |
| 8-64 | 4 | 6.26 | 1.05 | 0 | 3e-02 / 4e-03 |
| 64-1K | 12 | 6.30 | 1.77 | 0 | 2e-01 / 3e-03 |
| 1K-16K | 14 | 24.51 | 7.47 | 0 | 1e+00 / 4e-03 |
| >=16K | 34 | 187.33 | 22.96 | 0 | 1e+00 / 4e-03 |
| all | 72 | 37.72 | 0.97 | 0 | 1e+00 / 4e-03 |

fused_producer float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.00 | 0.69 | 4 | 1e-03 / 9e-04 |
| 8-64 | 4 | 1.15 | 0.91 | 1 | 3e-03 / 9e-04 |
| 64-1K | 12 | 1.52 | 1.02 | 0 | 1e-02 / 6e-03 |
| 1K-16K | 14 | 1.54 | 0.81 | 1 | 9e-01 / 5e-01 |
| >=16K | 34 | 7.78 | 1.23 | 0 | 1e+00 / inf |
| all | 72 | 3.10 | 0.69 | 6 | 1e+00 / inf |
  0.69  fused_producer float16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.81  fused_producer float16 rows=4096 d=64 n=1048576 fewhot (contention 4.45e+03)
  0.81  fused_producer float16 rows=262144 d=64 n=1048576 sorted (contention 5)
  0.84  fused_producer float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.89  fused_producer float16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.91  fused_producer float16 rows=262144 d=512 n=262144 runs32 (contention 33.1)

fused_producer float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.35 | 1.23 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.69 | 1.58 | 0 | 5e-07 / 1e-07 |
| 64-1K | 6 | 2.15 | 1.61 | 0 | 1e-06 / 8e-07 |
| 1K-16K | 6 | 1.39 | 1.26 | 0 | 8e-06 / 8e-06 |
| >=16K | 24 | 6.72 | 1.22 | 0 | 4e-05 / 8e-06 |
| all | 48 | 3.27 | 1.22 | 0 | 4e-05 / 8e-06 |

gnn float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.08 | 0.98 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.37 | 1.29 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.53 | 0.96 | 1 | 7e-07 / 7e-07 |
| 1K-16K | 6 | 1.15 | 0.98 | 0 | 5e-06 / 5e-06 |
| >=16K | 24 | 5.89 | 1.52 | 0 | 4e-05 / 4e-06 |
| all | 48 | 2.71 | 0.96 | 1 | 4e-05 / 5e-06 |
  0.96  gnn float32 rows=4096 d=128 n=1048576 uniform (contention 257)

hist int32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.22 | 0.96 | 1 | 0e+00 / 0e+00 |
| 64-1K | 2 | 1.07 | 1.01 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 2.52 | 2.23 | 0 | 0e+00 / 0e+00 |
| all | 12 | 1.93 | 0.96 | 1 | 0e+00 / 0e+00 |
  0.96  hist int32 rows=262144 d=1 n=1048576 sorted (contention 5.01)

hist int64

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 13.87 | 11.74 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 16.12 | 12.94 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 52.30 | 8.21 | 0 | 0e+00 / 0e+00 |
| all | 12 | 34.46 | 8.21 | 0 | 0e+00 / 0e+00 |

index_add bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.05 | 0.87 | 2 | 1e-02 / 3e-03 |
| 8-64 | 4 | 6.42 | 0.95 | 1 | 2e-02 / 4e-03 |
| 64-1K | 12 | 6.37 | 1.56 | 0 | 7e-02 / 3e-03 |
| 1K-16K | 14 | 26.23 | 7.10 | 0 | 6e-01 / 3e-03 |
| >=16K | 34 | 191.83 | 24.50 | 0 | 7e-01 / 3e-03 |
| all | 72 | 38.60 | 0.87 | 3 | 7e-01 / 4e-03 |
  0.87  index_add bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.90  index_add bfloat16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.95  index_add bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33)

index_add float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.05 | 0.80 | 3 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.29 | 1.05 | 0 | 3e-03 / 7e-04 |
| 64-1K | 12 | 1.52 | 0.97 | 0 | 7e-03 / 6e-03 |
| 1K-16K | 14 | 1.60 | 0.85 | 1 | 8e-02 / 4e-02 |
| >=16K | 34 | 7.88 | 1.23 | 0 | 2e-01 / 4e-02 |
| all | 72 | 3.18 | 0.80 | 4 | 2e-01 / 4e-02 |
  0.80  index_add float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.80  index_add float16 rows=262144 d=64 n=1048576 sorted (contention 5)
  0.85  index_add float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)
  0.95  index_add float16 rows=262144 d=64 n=1048576 uniform (contention 5)

index_add float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.32 | 1.19 | 0 | 1e-07 / 2e-07 |
| 8-64 | 4 | 1.72 | 1.62 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 2.12 | 1.50 | 0 | 1e-06 / 7e-07 |
| 1K-16K | 6 | 1.20 | 0.97 | 0 | 5e-06 / 5e-06 |
| >=16K | 24 | 6.96 | 1.18 | 0 | 3e-05 / 6e-06 |
| all | 48 | 3.25 | 0.97 | 0 | 3e-05 / 6e-06 |

index_add_dim1 float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.30 | 0.87 | 1 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.39 | 1.32 | 0 | 1e-07 / 1e-07 |
| 64-1K | 6 | 1.50 | 1.45 | 0 | 7e-07 / 8e-07 |
| 1K-16K | 6 | 1.94 | 1.25 | 0 | 4e-06 / 4e-06 |
| >=16K | 24 | 3.00 | 1.12 | 0 | 4e-06 / 2e-06 |
| all | 48 | 2.13 | 0.87 | 1 | 4e-06 / 4e-06 |
  0.87  index_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

index_put bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 3.02 | 0.87 | 2 | 1e-02 / 3e-03 |
| 8-64 | 4 | 6.46 | 0.95 | 1 | 3e-02 / 3e-03 |
| 64-1K | 12 | 6.36 | 1.56 | 0 | 5e-02 / 3e-03 |
| 1K-16K | 14 | 25.49 | 6.82 | 0 | 3e-01 / 3e-03 |
| >=16K | 34 | 192.54 | 23.58 | 0 | 8e-01 / 3e-03 |
| all | 72 | 38.42 | 0.87 | 3 | 8e-01 / 3e-03 |
  0.87  index_put bfloat16 rows=262144 d=512 n=262144 uniform (contention 2)
  0.89  index_put bfloat16 rows=262144 d=512 n=262144 sorted (contention 2)
  0.95  index_put bfloat16 rows=262144 d=512 n=262144 runs32 (contention 33.1)

index_put float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 10 | 1.15 | 0.68 | 2 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.29 | 1.04 | 0 | 4e-03 / 7e-04 |
| 64-1K | 14 | 1.63 | 0.97 | 1 | 7e-03 / 6e-03 |
| 1K-16K | 14 | 1.62 | 0.86 | 1 | 6e-02 / 4e-02 |
| >=16K | 36 | 8.17 | 1.23 | 0 | 2e-01 / 4e-02 |
| all | 78 | 3.23 | 0.68 | 4 | 2e-01 / 4e-02 |
  0.68  index_put float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.80  index_put float16 rows=262144 d=64 n=1048576 sorted (contention 5)
  0.86  index_put float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)
  0.97  index_put float16 rows=4096 d=64 n=1048576 uniform (contention 257)

index_put float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| ~1 | 77 | 1.10 | 0.86 | 7 | 8e-08 / 9e-08 |
| 1.5-8 | 81 | 1.25 | 0.78 | 8 | 2e-07 / 2e-07 |
| 8-64 | 112 | 1.42 | 0.84 | 5 | 7e-07 / 3e-07 |
| 64-1K | 86 | 1.40 | 0.75 | 14 | 2e-06 / 1e-06 |
| 1K-16K | 249 | 2.16 | 0.64 | 11 | 9e-06 / 6e-06 |
| >=16K | 278 | 6.95 | 1.05 | 0 | 5e-04 / 5e-04 |
| all | 883 | 2.54 | 0.64 | 45 | 5e-04 / 5e-04 |
  0.64  index_put float32 rows=1048576 d=1 n=1048576 fewhot (contention 4.2e+03)
  0.71  index_put float32 rows=1048576 d=1 n=262144 fewhot (contention 1.05e+03)
  0.75  index_put float32 rows=4096 d=8 n=65536 fewhot (contention 277)
  0.76  index_put float32 rows=262144 d=1 n=262144 fewhot (contention 1.05e+03)
  0.78  index_put float32 rows=65536 d=8 n=262144 sorted (contention 5.01)
  0.79  index_put float32 rows=262144 d=8 n=65536 fewhot (contention 263)
  0.80  index_put float32 rows=65536 d=32 n=65536 fewhot (contention 262)
  0.81  index_put float32 rows=4096 d=32 n=65536 fewhot (contention 275)
  0.83  index_put float32 rows=65536 d=1 n=65536 fewhot (contention 266)
  0.84  index_put float32 rows=4096 d=128 n=65536 fewhot (contention 279)

index_put int32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 1.55 | 1.48 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 2.39 | 1.70 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 13.31 | 1.60 | 0 | 0e+00 / 0e+00 |
| all | 12 | 6.99 | 1.48 | 0 | 0e+00 / 0e+00 |

index_put int64

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 2 | 6.39 | 5.04 | 0 | 0e+00 / 0e+00 |
| 64-1K | 2 | 7.79 | 5.80 | 0 | 0e+00 / 0e+00 |
| >=16K | 8 | 290.50 | 40.47 | 0 | 0e+00 / 0e+00 |
| all | 12 | 84.11 | 5.04 | 0 | 0e+00 / 0e+00 |

index_put_2d float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.24 | 1.03 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.70 | 1.59 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.94 | 1.07 | 0 | 1e-06 / 9e-07 |
| 1K-16K | 6 | 1.22 | 1.06 | 0 | 6e-06 / 4e-06 |
| >=16K | 24 | 6.92 | 1.27 | 0 | 2e-05 / 4e-06 |
| all | 48 | 3.17 | 1.03 | 0 | 2e-05 / 4e-06 |

masked_index_put float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.14 | 0.96 | 1 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.52 | 1.39 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.82 | 1.08 | 0 | 7e-07 / 8e-07 |
| 1K-16K | 6 | 1.19 | 0.92 | 1 | 7e-06 / 5e-06 |
| >=16K | 24 | 6.53 | 1.06 | 0 | 2e-05 / 4e-06 |
| all | 48 | 2.98 | 0.92 | 2 | 2e-05 / 5e-06 |
  0.92  masked_index_put float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.2e+03)
  0.96  masked_index_put float32 rows=262144 d=128 n=1048576 uniform (contention 5)

moe_combine float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.09 | 0.96 | 1 | 2e-07 / 2e-07 |
| 8-64 | 4 | 1.37 | 1.30 | 0 | 3e-07 / 1e-07 |
| 64-1K | 6 | 1.60 | 1.21 | 0 | 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.20 | 1.04 | 0 | 6e-06 / 5e-06 |
| >=16K | 24 | 6.08 | 1.07 | 0 | 2e-05 / 4e-06 |
| all | 48 | 2.78 | 0.96 | 1 | 2e-05 / 5e-06 |
  0.96  moe_combine float32 rows=262144 d=128 n=1048576 sorted (contention 5)

scatter_add bfloat16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.38 | 1.03 | 0 | 1e-02 / 3e-03 |
| 8-64 | 4 | 1.55 | 1.15 | 0 | 2e-02 / 3e-03 |
| 64-1K | 12 | 2.23 | 1.56 | 0 | 6e-02 / 4e-03 |
| 1K-16K | 14 | 1.37 | 0.75 | 3 | 4e-01 / 3e-03 |
| >=16K | 34 | 6.88 | 1.62 | 0 | 8e-01 / 3e-03 |
| all | 72 | 3.21 | 0.75 | 3 | 8e-01 / 4e-03 |
  0.75  scatter_add bfloat16 rows=262144 d=128 n=1048576 fewhot (contention 4.22e+03)
  0.81  scatter_add bfloat16 rows=262144 d=512 n=262144 zipf (contention 6.52e+03)
  0.93  scatter_add bfloat16 rows=262144 d=512 n=262144 fewhot (contention 1.06e+03)

scatter_add float16

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.06 | 0.71 | 2 | 1e-03 / 1e-03 |
| 8-64 | 4 | 1.29 | 1.03 | 0 | 3e-03 / 8e-04 |
| 64-1K | 12 | 1.52 | 1.03 | 0 | 6e-03 / 5e-03 |
| 1K-16K | 14 | 1.58 | 0.82 | 2 | 9e-02 / 4e-02 |
| >=16K | 34 | 7.98 | 1.36 | 0 | 2e-01 / 5e-02 |
| all | 72 | 3.19 | 0.71 | 4 | 2e-01 / 5e-02 |
  0.71  scatter_add float16 rows=262144 d=16 n=1048576 sorted (contention 5)
  0.80  scatter_add float16 rows=262144 d=64 n=1048576 sorted (contention 5)
  0.82  scatter_add float16 rows=4096 d=64 n=1048576 fewhot (contention 4.41e+03)
  0.85  scatter_add float16 rows=262144 d=64 n=1048576 fewhot (contention 4.2e+03)

scatter_add float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.33 | 1.20 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.74 | 1.61 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 2.11 | 1.50 | 0 | 8e-07 / 8e-07 |
| 1K-16K | 6 | 1.42 | 1.27 | 0 | 5e-06 / 5e-06 |
| >=16K | 24 | 7.08 | 1.18 | 0 | 3e-05 / 5e-06 |
| all | 48 | 3.35 | 1.18 | 0 | 3e-05 / 5e-06 |

scatter_add_dim1 float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.29 | 0.86 | 1 | 1e-07 / 1e-07 |
| 8-64 | 4 | 1.39 | 1.32 | 0 | 1e-07 / 1e-07 |
| 64-1K | 6 | 1.49 | 1.44 | 0 | 7e-07 / 7e-07 |
| 1K-16K | 6 | 1.85 | 1.13 | 0 | 5e-06 / 4e-06 |
| >=16K | 24 | 2.94 | 1.15 | 0 | 5e-06 / 2e-06 |
| all | 48 | 2.09 | 0.86 | 1 | 5e-06 / 4e-06 |
  0.86  scatter_add_dim1 float32 rows=1048576 d=128 n=1048576 uniform (contention 2)

scatter_reduce float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.32 | 1.21 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.71 | 1.61 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 2.07 | 1.50 | 0 | 8e-07 / 7e-07 |
| 1K-16K | 6 | 1.28 | 0.94 | 1 | 5e-06 / 4e-06 |
| >=16K | 24 | 6.96 | 1.18 | 0 | 3e-05 / 4e-06 |
| all | 48 | 3.27 | 0.94 | 1 | 3e-05 / 4e-06 |
  0.94  scatter_reduce float32 rows=1048576 d=32 n=1048576 fewhot (contention 4.19e+03)

scatter_reduce_noself float32

| contention | cases | geomean | worst | slower | max err |
|---|---|---|---|---|---|
| 1.5-8 | 8 | 1.25 | 1.15 | 0 | 2e-07 / 1e-07 |
| 8-64 | 4 | 1.62 | 1.51 | 0 | 4e-07 / 1e-07 |
| 64-1K | 6 | 1.91 | 1.35 | 0 | 7e-07 / 7e-07 |
| 1K-16K | 6 | 1.28 | 1.17 | 0 | 5e-06 / 4e-06 |
| >=16K | 24 | 5.81 | 1.14 | 0 | 2e-05 / 5e-06 |
| all | 48 | 2.91 | 1.14 | 0 | 2e-05 / 5e-06 |

Overall:

| dtype | cases | geomean | worst | < 0.95 |
|---|---|---|---|---|
| all | 2281 | 3.70 | 0.64 | 77 |
| bfloat16 | 360 | 14.07 | 0.75 | 11 |
| float16 | 366 | 3.15 | 0.68 | 20 |
| float32 | 1507 | 2.68 | 0.64 | 46 |
| int32 | 24 | 3.68 | 0.96 | 0 |
| int64 | 24 | 53.83 | 5.04 | 0 |
