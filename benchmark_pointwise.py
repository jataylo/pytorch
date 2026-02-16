#!/usr/bin/env python3
"""
Comprehensive Pointwise Kernel Benchmark
Tests eager vs compile performance across multiple shapes and operations.
"""

import torch
import time
import os
import math
from typing import Dict, List, Tuple
from statistics import geometric_mean

# Default to new heuristics if not explicitly set
# (Can be overridden by setting env var before running)
os.environ.setdefault("TORCHINDUCTOR_POINTWISE_HEURISTICS", "1")

# Enable real bench mode to see predicted vs actual validation
os.environ.setdefault("TORCHINDUCTOR_HEURISTICS_REAL_BENCH", "1")

# Disable dynamic shapes for consistent benchmarking
os.environ.setdefault("TORCHINDUCTOR_DYNAMIC_SHAPES", "0")

# Set CUDA graph pool limit to infinity for better performance
os.environ.setdefault("TORCH_CUDA_GRAPH_POOL_LIMIT", "999999999")


class PointwiseBenchmark:
    """Benchmark pointwise kernels with various shapes"""
    
    def __init__(self, device='cuda', warmup_iters=10, bench_iters=50, clear_cache_per_shape=False):
        self.device = device
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.results = []
        self.clear_cache_per_shape = clear_cache_per_shape  # Clear cache after each shape
        
        # Define test shapes (name, shape) - including odd shapes and larger ranges
        self.shapes = [
            # 1D shapes - power of 2
            ("1D_tiny", (512,)),
            ("1D_small", (4096,)),
            ("1D_medium", (65536,)),
            ("1D_large", (1048576,)),
            ("1D_huge", (16777216,)),
            ("1D_massive", (67108864,)),
            
            # 1D shapes - odd sizes
            ("1D_odd_small", (3333,)),
            ("1D_odd_medium", (54321,)),
            ("1D_odd_large", (1234567,)),
            
            # 2D shapes - power of 2
            ("2D_square_tiny", (128, 128)),
            ("2D_square_small", (512, 512)),
            ("2D_square_medium", (2048, 2048)),
            ("2D_square_large", (4096, 4096)),
            ("2D_square_huge", (8192, 8192)),
            
            # 2D shapes - rectangular
            ("2D_wide_thin", (64, 16384)),
            ("2D_wide_med", (256, 8192)),
            ("2D_tall_thin", (16384, 64)),
            ("2D_tall_med", (8192, 256)),
            
            # 2D shapes - odd dimensions
            ("2D_odd_square", (777, 777)),
            ("2D_odd_wide", (333, 3333)),
            ("2D_odd_tall", (3333, 333)),
            ("2D_odd_mixed", (1234, 5678)),
            
            # 3D shapes - power of 2
            ("3D_tiny", (16, 16, 16)),
            ("3D_small", (32, 32, 32)),
            ("3D_medium", (64, 64, 64)),
            ("3D_large", (128, 128, 128)),
            ("3D_huge", (256, 256, 256)),
            
            # 3D shapes - batched (common in ML)
            ("3D_batch_small", (4, 512, 512)),
            ("3D_batch_medium", (16, 256, 256)),
            ("3D_batch_large", (32, 512, 512)),
            ("3D_batch_video", (8, 128, 128)),
            
            # 3D shapes - odd dimensions
            ("3D_odd_small", (17, 31, 47)),
            ("3D_odd_medium", (33, 65, 97)),
            ("3D_odd_batch", (7, 333, 333)),
        ]
    
    def benchmark_op(self, name: str, eager_fn, compile_fn, inputs: List[torch.Tensor]) -> Dict:
        """
        Benchmark a single operation.
        
        Returns:
            dict with timing results and speedup
        """
        # Warmup eager
        for _ in range(self.warmup_iters):
            _ = eager_fn(*inputs)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        
        # Time eager
        if self.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.bench_iters):
            _ = eager_fn(*inputs)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        eager_time = (time.perf_counter() - start) / self.bench_iters * 1000  # ms
        
        # Warmup compiled
        for _ in range(self.warmup_iters):
            _ = compile_fn(*inputs)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        
        # Time compiled
        if self.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.bench_iters):
            _ = compile_fn(*inputs)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) / self.bench_iters * 1000  # ms
        
        speedup = eager_time / compile_time if compile_time > 0 else 0.0
        
        return {
            'name': name,
            'eager_ms': eager_time,
            'compile_ms': compile_time,
            'speedup': speedup,
        }
    
    def bench_elementwise_add(self):
        """Benchmark: z = x + y"""
        print("\n" + "="*80)
        print("  Benchmark 1: Elementwise Add (z = x + y)")
        print("="*80)
        
        def eager_fn(x, y):
            return x + y
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            y = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"add_{shape_name}", eager_fn, compile_fn, [x, y])
            result['op'] = 'add'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")
            
            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()
    
    def bench_elementwise_mul(self):
        """Benchmark: z = x * y"""
        print("\n" + "="*80)
        print("  Benchmark 2: Elementwise Multiply (z = x * y)")
        print("="*80)
        
        def eager_fn(x, y):
            return x * y
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            y = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"mul_{shape_name}", eager_fn, compile_fn, [x, y])
            result['op'] = 'mul'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_relu_sigmoid(self):
        """Benchmark: z = sigmoid(relu(x))"""
        print("\n" + "="*80)
        print("  Benchmark 3: ReLU + Sigmoid (z = sigmoid(relu(x)))")
        print("="*80)
        
        def eager_fn(x):
            return torch.sigmoid(torch.relu(x))
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"relu_sigmoid_{shape_name}", eager_fn, compile_fn, [x])
            result['op'] = 'relu_sigmoid'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_gelu(self):
        """Benchmark: z = gelu(x)"""
        print("\n" + "="*80)
        print("  Benchmark 4: GELU Activation (z = gelu(x))")
        print("="*80)
        
        def eager_fn(x):
            return torch.nn.functional.gelu(x)
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"gelu_{shape_name}", eager_fn, compile_fn, [x])
            result['op'] = 'gelu'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_tanh(self):
        """Benchmark: Tanh activation"""
        print("\n" + "="*80)
        print("  Benchmark 5: Tanh Activation (z = tanh(x))")
        print("="*80)
        
        def eager_fn(x):
            return torch.tanh(x)
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"tanh_{shape_name}", eager_fn, compile_fn, [x])
            result['op'] = 'tanh'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_silu(self):
        """Benchmark: SiLU/Swish activation"""
        print("\n" + "="*80)
        print("  Benchmark 6: SiLU/Swish (z = x * sigmoid(x))")
        print("="*80)
        
        def eager_fn(x):
            return torch.nn.functional.silu(x)
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"silu_{shape_name}", eager_fn, compile_fn, [x])
            result['op'] = 'silu'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_squared_relu(self):
        """Benchmark: Squared ReLU"""
        print("\n" + "="*80)
        print("  Benchmark 7: Squared ReLU (z = relu(x)^2)")
        print("="*80)
        
        def eager_fn(x):
            r = torch.relu(x)
            return r * r
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"squared_relu_{shape_name}", eager_fn, compile_fn, [x])
            result['op'] = 'squared_relu'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_bias_add_relu(self):
        """Benchmark: Bias add + ReLU"""
        print("\n" + "="*80)
        print("  Benchmark 8: Bias Add + ReLU (z = relu(x + bias))")
        print("="*80)
        
        def eager_fn(x, bias):
            return torch.relu(x + bias)
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            bias = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"bias_relu_{shape_name}", eager_fn, compile_fn, [x, bias])
            result['op'] = 'bias_relu'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_leaky_relu(self):
        """Benchmark: Leaky ReLU"""
        print("\n" + "="*80)
        print("  Benchmark 9: Leaky ReLU (z = leaky_relu(x, 0.01))")
        print("="*80)
        
        def eager_fn(x):
            return torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"leaky_relu_{shape_name}", eager_fn, compile_fn, [x])
            result['op'] = 'leaky_relu'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_2d_specific(self):
        """Benchmark: 2D-specific operations that use YBLOCK"""
        print("\n" + "="*80)
        print("  Benchmark 10: 2D Pointwise (tests XBLOCK + YBLOCK)")
        print("  z = x.T + y * w - forces 2D kernel structure")
        print("="*80)
        
        def eager_fn(x, y, w):
            # Transpose creates non-contiguous access pattern
            # This tends to preserve 2D structure in kernel
            return x.transpose(-2, -1) + y * w
        
        compile_fn = torch.compile(eager_fn, mode='max-autotune')
        
        # Use only 2D shapes for this test
        # For transpose compatibility, filter out extremely wide/tall shapes
        shapes_2d = [(name, shape) for name, shape in self.shapes 
                     if len(shape) == 2 and abs(shape[0] - shape[1]) < max(shape[0], shape[1]) * 0.9]
        
        for shape_name, shape in shapes_2d:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            # Create transposed shape for y and w to match x.T
            transposed_shape = (shape[1], shape[0])
            y = torch.randn(transposed_shape, device=self.device, dtype=torch.float32)
            w = torch.randn(transposed_shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"2d_{shape_name}", eager_fn, compile_fn, [x, y, w])
            result['op'] = '2d_specific'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_3d_specific(self):
        """Benchmark: 3D-specific operations that use ZBLOCK"""
        print("\n" + "="*80)
        print("  Benchmark 11: 3D Pointwise (tests XBLOCK + YBLOCK + ZBLOCK)")
        print("  z = x.permute(2,1,0) + y * w - forces 3D kernel structure")
        print("="*80)
        
        def eager_fn(x, y, w):
            # Permutation creates non-contiguous access pattern
            # This tends to preserve 3D structure in kernel
            x_perm = x.permute(2, 1, 0)  # Reverse all dimensions
            return x_perm + y * w
        
        compile_fn = torch.compile(eager_fn, mode='max-autotune')
        
        # Use only 3D shapes for this test
        # Filter to cube-like or near-cube shapes for permutation compatibility
        shapes_3d = [(name, shape) for name, shape in self.shapes 
                     if len(shape) == 3]
        
        for shape_name, shape in shapes_3d:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            # Create reversed shape for y and w to match x.permute(2,1,0)
            reversed_shape = (shape[2], shape[1], shape[0])
            y = torch.randn(reversed_shape, device=self.device, dtype=torch.float32)
            w = torch.randn(reversed_shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"3d_{shape_name}", eager_fn, compile_fn, [x, y, w])
            result['op'] = '3d_specific'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_fused_pointwise(self):
        """Benchmark: Fused pointwise operations"""
        print("\n" + "="*80)
        print("  Benchmark 12: Fused Pointwise Model")
        print("  z = gelu((x + y) * w + b)")
        print("="*80)
        
        def eager_fn(x, y, w, b):
            # This should fuse into a single kernel with torch.compile
            t1 = x + y
            t2 = t1 * w
            t3 = t2 + b
            return torch.nn.functional.gelu(t3)
        
        compile_fn = torch.compile(eager_fn)
        
        for shape_name, shape in self.shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            y = torch.randn(shape, device=self.device, dtype=torch.float32)
            w = torch.randn(shape, device=self.device, dtype=torch.float32)
            b = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"fused_{shape_name}", eager_fn, compile_fn, [x, y, w, b])
            result['op'] = 'fused'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_heavy_fusion_mlp(self):
        """Benchmark: MLP-style fusion (feed-forward network)"""
        print("\n" + "="*80)
        print("  Benchmark 13: Heavy Fusion - MLP Style")
        print("  Simulates: Linear + GELU + Linear + Dropout pattern")
        print("="*80)
        
        def eager_fn(x, w1, b1, w2, b2):
            # First layer: weight * x + bias + gelu
            h1 = x * w1 + b1
            h1_act = torch.nn.functional.gelu(h1)
            
            # Second layer: weight * h1 + bias
            h2 = h1_act * w2 + b2
            
            # Residual connection + normalization
            residual = x + h2
            mean = torch.mean(residual)
            std = torch.std(residual)
            normalized = (residual - mean) / (std + 1e-6)
            
            return normalized
        
        compile_fn = torch.compile(eager_fn)
        
        # Use subset of shapes
        fusion_shapes = [
            ("1D_large", (1048576,)),
            ("1D_huge", (16777216,)),
            ("2D_square_medium", (2048, 2048)),
            ("2D_square_large", (4096, 4096)),
            ("2D_odd_mixed", (1234, 5678)),
            ("3D_medium", (64, 64, 64)),
            ("3D_batch_medium", (16, 256, 256)),
        ]
        
        for shape_name, shape in fusion_shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            w1 = torch.randn(shape, device=self.device, dtype=torch.float32)
            b1 = torch.randn(shape, device=self.device, dtype=torch.float32)
            w2 = torch.randn(shape, device=self.device, dtype=torch.float32)
            b2 = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"heavy_mlp_{shape_name}", eager_fn, compile_fn, [x, w1, b1, w2, b2])
            result['op'] = 'heavy_mlp'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_heavy_fusion_attention(self):
        """Benchmark: Attention-style fusion (softmax + scaling)"""
        print("\n" + "="*80)
        print("  Benchmark 14: Heavy Fusion - Attention Style")
        print("  Simulates: QK scaling + softmax + value multiply pattern")
        print("="*80)
        
        def eager_fn(q, k, v, mask):
            # Attention scores with scaling
            scores = q * k / (q.shape[-1] ** 0.5)
            
            # Apply mask
            scores_masked = scores + mask * -1e9
            
            # Softmax (eager does it in multiple ops)
            scores_exp = torch.exp(scores_masked - torch.max(scores_masked))
            scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
            attn_weights = scores_exp / (scores_sum + 1e-9)
            
            # Apply to values
            output = attn_weights * v
            
            # Layer norm
            mean = torch.mean(output)
            var = torch.var(output)
            normalized = (output - mean) / torch.sqrt(var + 1e-6)
            
            return normalized
        
        compile_fn = torch.compile(eager_fn)
        
        # Use subset of shapes
        fusion_shapes = [
            ("1D_medium", (65536,)),
            ("1D_large", (1048576,)),
            ("2D_square_small", (512, 512)),
            ("2D_square_medium", (2048, 2048)),
            ("2D_odd_square", (777, 777)),
            ("3D_small", (32, 32, 32)),
            ("3D_medium", (64, 64, 64)),
        ]
        
        for shape_name, shape in fusion_shapes:
            q = torch.randn(shape, device=self.device, dtype=torch.float32)
            k = torch.randn(shape, device=self.device, dtype=torch.float32)
            v = torch.randn(shape, device=self.device, dtype=torch.float32)
            mask = torch.randint(0, 2, shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"heavy_attn_{shape_name}", eager_fn, compile_fn, [q, k, v, mask])
            result['op'] = 'heavy_attn'
            result['shape'] = shape_name
            result['numel'] = q.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_heavy_fusion_conv_style(self):
        """Benchmark: Conv-style fusion (activations + batch norm + residual)"""
        print("\n" + "="*80)
        print("  Benchmark 15: Heavy Fusion - Conv Block Style")
        print("  Simulates: Conv output + BatchNorm + ReLU + Residual pattern")
        print("="*80)
        
        def eager_fn(x, running_mean, running_var, weight, bias, residual):
            # Batch normalization
            normalized = (x - running_mean) / torch.sqrt(running_var + 1e-5)
            scaled = normalized * weight + bias
            
            # Activation
            activated = torch.relu(scaled)
            
            # Residual connection
            with_residual = activated + residual
            
            # Optional: second activation
            output = torch.nn.functional.silu(with_residual)
            
            return output
        
        compile_fn = torch.compile(eager_fn)
        
        # Use subset of shapes
        fusion_shapes = [
            ("1D_medium", (65536,)),
            ("1D_large", (1048576,)),
            ("2D_square_medium", (2048, 2048)),
            ("2D_odd_wide", (333, 3333)),
            ("3D_batch_small", (4, 512, 512)),
            ("3D_batch_medium", (16, 256, 256)),
        ]
        
        for shape_name, shape in fusion_shapes:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            running_mean = torch.randn(1, device=self.device, dtype=torch.float32).expand_as(x)
            running_var = torch.abs(torch.randn(1, device=self.device, dtype=torch.float32)).expand_as(x) + 1e-5
            weight = torch.randn(shape, device=self.device, dtype=torch.float32)
            bias = torch.randn(shape, device=self.device, dtype=torch.float32)
            residual = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"heavy_conv_{shape_name}", eager_fn, compile_fn, 
                                      [x, running_mean, running_var, weight, bias, residual])
            result['op'] = 'heavy_conv'
            result['shape'] = shape_name
            result['numel'] = x.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def bench_heavy_fusion_branching(self):
        """Benchmark: HEAVY fusion - many operations with branching"""
        print("\n" + "="*80)
        print("  Benchmark 16: Heavy Fusion - Branching Paths")
        print("  Complex multi-op fusion with branching (15+ ops)")
        print("="*80)
        
        def eager_fn(x1, x2, x3, x4):
            # Path 1: Multiple activations
            a = torch.relu(x1)
            b = torch.sigmoid(x2)
            c = torch.tanh(x3)
            d = torch.nn.functional.gelu(x4)
            
            # Path 2: Combine paths
            e = a * b + c * d
            
            # Path 3: More operations
            f = torch.abs(e)
            g = f + torch.sqrt(torch.abs(e) + 1e-6)
            h = torch.sin(g)
            
            # Path 4: Final combination
            i = h * torch.cos(e)
            j = i + torch.exp(-torch.abs(i))
            
            # Final normalization
            mean = torch.mean(j)
            std = torch.std(j)
            result = (j - mean) / (std + 1e-6)
            
            return result
        
        compile_fn = torch.compile(eager_fn)
        
        # Use subset of shapes for heavy fusion (it's expensive)
        heavy_shapes = [
            ("1D_medium", (65536,)),
            ("1D_large", (1048576,)),
            ("2D_square_small", (512, 512)),
            ("2D_square_medium", (2048, 2048)),
            ("2D_odd_square", (777, 777)),
            ("3D_small", (32, 32, 32)),
            ("3D_medium", (64, 64, 64)),
            ("3D_odd_small", (17, 31, 47)),
        ]
        
        for shape_name, shape in heavy_shapes:
            x1 = torch.randn(shape, device=self.device, dtype=torch.float32)
            x2 = torch.randn(shape, device=self.device, dtype=torch.float32)
            x3 = torch.randn(shape, device=self.device, dtype=torch.float32)
            x4 = torch.randn(shape, device=self.device, dtype=torch.float32)
            
            result = self.benchmark_op(f"heavy_branch_{shape_name}", eager_fn, compile_fn, [x1, x2, x3, x4])
            result['op'] = 'heavy_branching'
            result['shape'] = shape_name
            result['numel'] = x1.numel()
            self.results.append(result)
            
            print(f"  {shape_name:20s} | Eager: {result['eager_ms']:8.4f}ms | "
                  f"Compile: {result['compile_ms']:8.4f}ms | "
                  f"Speedup: {result['speedup']:6.3f}x")

            # Clear cache after each shape to see fresh heuristics
            if self.clear_cache_per_shape:
                self.clear_compilation_cache()

    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("  SUMMARY")
        print("="*80)
        
        # Group by operation
        ops = {}
        for result in self.results:
            op = result['op']
            if op not in ops:
                ops[op] = []
            ops[op].append(result['speedup'])
        
        print("\nGeometric Mean Speedup by Operation:")
        print("-" * 50)
        overall_speedups = []
        for op, speedups in sorted(ops.items()):
            gmean = geometric_mean(speedups)
            overall_speedups.extend(speedups)
            print(f"  {op:20s}: {gmean:6.3f}x")
        
        overall_gmean = geometric_mean(overall_speedups)
        print("-" * 50)
        print(f"  {'OVERALL':20s}: {overall_gmean:6.3f}x")
        print("="*80)
        
        # Best and worst
        best = max(self.results, key=lambda x: x['speedup'])
        worst = min(self.results, key=lambda x: x['speedup'])
        
        print(f"\nBest speedup:  {best['name']:40s} {best['speedup']:6.3f}x")
        print(f"Worst speedup: {worst['name']:40s} {worst['speedup']:6.3f}x")
        
        # Size analysis
        print("\nSpeedup by Problem Size:")
        print("-" * 50)
        size_groups = {
            'small (<100K)': [],
            'medium (100K-1M)': [],
            'large (1M-10M)': [],
            'huge (>10M)': [],
        }
        
        for result in self.results:
            numel = result['numel']
            if numel < 100000:
                size_groups['small (<100K)'].append(result['speedup'])
            elif numel < 1000000:
                size_groups['medium (100K-1M)'].append(result['speedup'])
            elif numel < 10000000:
                size_groups['large (1M-10M)'].append(result['speedup'])
            else:
                size_groups['huge (>10M)'].append(result['speedup'])
        
        for size, speedups in size_groups.items():
            if speedups:
                gmean = geometric_mean(speedups)
                print(f"  {size:20s}: {gmean:6.3f}x ({len(speedups)} kernels)")
        
        print("="*80)
    
    def save_csv(self, filename='pointwise_benchmark_results.csv'):
        """Save results to CSV"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['op', 'shape', 'numel', 'eager_ms', 'compile_ms', 'speedup'])
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'op': result['op'],
                    'shape': result['shape'],
                    'numel': result['numel'],
                    'eager_ms': result['eager_ms'],
                    'compile_ms': result['compile_ms'],
                    'speedup': result['speedup'],
                })
        
        print(f"\n✅ Results saved to {filename}")
    
    def clear_compilation_cache(self):
        """Clear PyTorch compilation cache to force recompilation"""
        import shutil
        
        # CRITICAL: Clear PyTorch's in-memory compilation cache
        # This forces torch.compile to recompile for the next shape
        try:
            torch._dynamo.reset()
            if hasattr(torch._inductor, 'metrics'):
                torch._inductor.metrics.reset()
        except Exception as e:
            print(f"Warning: Could not reset dynamo cache: {e}")
        
        # Clear Inductor cache
        if os.path.exists('/tmp/torchinductor_root/'):
            try:
                shutil.rmtree('/tmp/torchinductor_root/')
            except:
                pass
        
        # Clear Triton cache
        triton_cache = os.path.expanduser('~/.triton/cache/')
        if os.path.exists(triton_cache):
            try:
                shutil.rmtree(triton_cache)
            except:
                pass
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_all(self, show_all_heuristics=False, clear_every=5):
        """Run all benchmarks
        
        Args:
            show_all_heuristics: Clear cache periodically to show heuristics
            clear_every: Clear cache every N benchmarks
        """
        print("\n" + "="*80)
        print("  POINTWISE KERNEL BENCHMARK SUITE")
        print("="*80)
        print(f"  Device: {self.device}")
        print(f"  Warmup iterations: {self.warmup_iters}")
        print(f"  Benchmark iterations: {self.bench_iters}")
        print(f"  Pointwise heuristics: {os.environ.get('TORCHINDUCTOR_POINTWISE_HEURISTICS', '?')}")
        
        if self.clear_cache_per_shape:
            print(f"  🔄 Cache clearing: After EACH shape (to show heuristics for every sub-problem)")
        elif show_all_heuristics:
            print(f"  🔄 Cache clearing: Every {clear_every} benchmarks (to show heuristics)")
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU: {props.name}")
            if torch.version.hip:
                print(f"  ROCm version: {torch.version.hip}")
        
        print("="*80)
        
        bench_count = 0
        def maybe_clear_cache():
            nonlocal bench_count
            bench_count += 1
            if show_all_heuristics and bench_count % clear_every == 0:
                print(f"\n🔄 Clearing cache (benchmark #{bench_count}) to show heuristics...")
                self.clear_compilation_cache()
        
        # Run all benchmarks
        print("\n" + "="*80)
        print("  PHASE 1: Basic Elementwise Operations (5 ops)")
        print("="*80)
        self.bench_elementwise_add()
        maybe_clear_cache()
        self.bench_elementwise_mul()
        maybe_clear_cache()
        self.bench_relu_sigmoid()
        maybe_clear_cache()
        self.bench_gelu()
        maybe_clear_cache()
        
        print("\n" + "="*80)
        print("  PHASE 2: Additional Elementwise Operations (5 new ops)")
        print("="*80)
        self.bench_tanh()
        maybe_clear_cache()
        self.bench_silu()
        maybe_clear_cache()
        self.bench_squared_relu()
        maybe_clear_cache()
        self.bench_bias_add_relu()
        maybe_clear_cache()
        self.bench_leaky_relu()
        maybe_clear_cache()
        
        print("\n" + "="*80)
        print("  PHASE 3: Multi-Dimensional Kernels (Tests YBLOCK/ZBLOCK)")
        print("="*80)
        self.bench_2d_specific()
        maybe_clear_cache()
        self.bench_3d_specific()
        maybe_clear_cache()
        
        print("\n" + "="*80)
        print("  PHASE 4: Basic Fusion")
        print("="*80)
        self.bench_fused_pointwise()
        maybe_clear_cache()
        
        print("\n" + "="*80)
        print("  PHASE 5: Heavy Fusion Patterns (4 variants)")
        print("="*80)
        self.bench_heavy_fusion_mlp()
        maybe_clear_cache()
        self.bench_heavy_fusion_attention()
        maybe_clear_cache()
        self.bench_heavy_fusion_conv_style()
        maybe_clear_cache()
        self.bench_heavy_fusion_branching()
        maybe_clear_cache()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_csv()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark pointwise kernels')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=50, help='Benchmark iterations')
    parser.add_argument('--no-heuristics', action='store_true', help='Disable pointwise heuristics')
    parser.add_argument('--csv', type=str, default='pointwise_benchmark_results.csv', help='Output CSV file')
    parser.add_argument('--quick', action='store_true', help='Run quick subset (10 shapes instead of 38)')
    parser.add_argument('--show-all-heuristics', action='store_true', help='Clear cache periodically to show heuristics for all benchmarks')
    parser.add_argument('--clear-cache-every', type=int, default=5, help='Clear cache every N benchmarks (with --show-all-heuristics)')
    parser.add_argument('--clear-per-shape', action='store_true', help='Clear cache after EACH shape (shows heuristics for every sub-problem)')
    
    args = parser.parse_args()
    
    # Configure heuristics
    # Respect env var if already set, otherwise use --no-heuristics flag or default to enabled
    if args.no_heuristics:
        os.environ["TORCHINDUCTOR_POINTWISE_HEURISTICS"] = "0"
        print("\n⚠️  Pointwise heuristics DISABLED (using original behavior)")
    elif "TORCHINDUCTOR_POINTWISE_HEURISTICS" not in os.environ:
        # Only set if not already set by user
        os.environ["TORCHINDUCTOR_POINTWISE_HEURISTICS"] = "1"
        print("\n✅ Pointwise heuristics ENABLED (using new optimizations)")
    else:
        # Respect user's env var setting
        if os.environ.get("TORCHINDUCTOR_POINTWISE_HEURISTICS") == "0":
            print("\n⚠️  Pointwise heuristics DISABLED (env var TORCHINDUCTOR_POINTWISE_HEURISTICS=0)")
        else:
            print("\n✅ Pointwise heuristics ENABLED (env var TORCHINDUCTOR_POINTWISE_HEURISTICS=1)")
    
    # Run benchmark
    benchmark = PointwiseBenchmark(
        device=args.device,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        clear_cache_per_shape=args.clear_per_shape
    )
    
    # If quick mode, use subset of shapes
    if args.quick:
        print("\n⚡ QUICK MODE: Using subset of shapes for faster testing")
        benchmark.shapes = [
            ("1D_small", (4096,)),
            ("1D_large", (1048576,)),
            ("2D_square_small", (512, 512)),
            ("2D_square_medium", (2048, 2048)),
            ("2D_odd_square", (777, 777)),
            ("2D_wide_thin", (64, 16384,)),
            ("3D_small", (32, 32, 32)),
            ("3D_medium", (64, 64, 64)),
            ("3D_odd_small", (17, 31, 47)),
            ("3D_batch_medium", (16, 256, 256)),
        ]
    
    benchmark.run_all(
        show_all_heuristics=args.show_all_heuristics,
        clear_every=args.clear_cache_every
    )
    
    if args.csv:
        benchmark.save_csv(args.csv)


if __name__ == "__main__":
    main()

