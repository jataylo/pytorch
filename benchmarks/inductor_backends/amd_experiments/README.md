# Independent AMD lowering experiments

This branch supplies measurement tools only; it changes no compiler behavior. Each sibling feature branch adds one lowering concept, its workload module, and measured results. Toggle only that branch's default-off configuration; its disabled arm is upstream lowering.

The benchmark compiles full static graphs, freezes their backend callables, checks three random input seeds against eager (FP64 accumulation for scans), and records paired rotating GPU event measurements. It reports both captured kernel-sequence time and host-issued graph replay, with warm and flushed caches. Inputs, restoration, and cache preparation are outside event intervals. Raw samples, generated sources, compile time and provenance accompany each measurement. Runtime speedup percentages mean `(stock / feature - 1) * 100`; time saved is a different statistic. Microbenchmark gains are not model gains.

Use the matching ROCm PyTorch/Triton environment and make this checkout's complete `torch._inductor` package importable. Run `python benchmarks/inductor_backends/amd_experiments/benchmark.py --output <directory>`. The local run commands are recorded with each result.
