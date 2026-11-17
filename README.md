# Pointer SLC draft

Draft Inductor change. When `TORCHINDUCTOR_POINTER_SLC=1`, every Triton
launch asks HIP whether any pointer input is an SLC allocation. If one is,
the kernel receives `distribution_tile` set to the first tiling size
(`XBLOCK`, `BLOCK_M`, `XBLOCK_0`, ...). Otherwise that argument is `0`.

CK, CK-tile, and FlyDSL are not wired up.

## Turn it on

```bash
export TORCHINDUCTOR_POINTER_SLC=1
```

That is `config.rocm.pointer_slc`. It is off by default. Turning it on
changes Triton kernel signatures, so cached cubins from a run without the
flag do not match.

There is no real `IS_SLC` field on `hipPointerAttribute_t` yet. This draft
treats bit 31 of `allocationFlags` (`1 << 31`) as that field.
`mark_pointer_slc(ptr)` and `TORCHINDUCTOR_FAKE_SLC=1` force the bit
without a driver change. A null pointer, a missing HIP runtime, and a failed
`hipPointerGetAttributes` all count as not SLC, in the Python launcher and
in the C++ wrapper. A failed query also clears HIP's sticky error, so the
next launch does not report it.

## What gets the argument

Inductor-generated Triton kernels, combo kernels, and Triton templates
(except `use_jit` templates, which are plain `@triton.jit` and never see
this launcher). The argument is a runtime scalar, not `tl.constexpr`,
because the value depends on the allocation, not the compile config.

`inductor_meta["distribution_tile"]` is `"launcher"` when Inductor added
the argument. A user kernel opts in by declaring its own `distribution_tile:
tl.int32` (or `int64`). Inductor then sets the meta to `"caller"` and the
launcher overwrites whatever placeholder the wrapper passed. A user kernel
that does not declare it is left unchanged.

The tile is the first name matching `BLOCK(?:_|$)`, looked up in the
signature first and then in the config: `XBLOCK` before `R0_BLOCK`,
`BLOCK_M` before `BLOCK_N`. On a normal pointwise or reduction kernel that
name is a signature constexpr, so the value is the autotuned block size. On
a template, `BLOCK_M` / `BLOCK_N` / `BLOCK_K` are assigned in the body
rather than passed in, so the value comes from the template's
`inductor_meta["config_args"]`. A kernel with no tiling name passes `0`.

The argument never gets a compile-time value hint: Inductor leaves it out of
`divisible_by_16` and `equal_to_1`, because its value changes per launch.

## Pointwise kernel

Before, flag off:

```python
@triton.jit
def triton_poi_fused_add(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    ...

def launcher(in_ptr0, in_ptr1, out_ptr0, xnumel, stream):
    grid_0 = -(xnumel // -1024)
    grid_1 = 1
    grid_2 = 1
    runner(in_ptr0, in_ptr1, out_ptr0, xnumel)
```

After, flag on. `1024` is this config's `XBLOCK`. The launcher does not
take `distribution_tile`. It computes it and passes it to the kernel.

```python
@triton.jit
def triton_poi_fused_add(
    in_ptr0, in_ptr1, out_ptr0, xnumel, distribution_tile, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    ...

def launcher(in_ptr0, in_ptr1, out_ptr0, xnumel, stream):
    grid_0 = -(xnumel // -1024)
    grid_1 = 1
    grid_2 = 1
    _any_slc = (
        pointer_is_slc(in_ptr0)
        or pointer_is_slc(in_ptr1)
        or pointer_is_slc(out_ptr0)
    )
    distribution_tile = 1024 if _any_slc else 0
    runner(in_ptr0, in_ptr1, out_ptr0, xnumel, distribution_tile)
```

A reduction uses the same shape. The signature constexprs are `XBLOCK` then
`R0_BLOCK`, so `distribution_tile` is `XBLOCK`, not `R0_BLOCK`.

## Template kernel

`BLOCK_M` is set in the template body, not in the signature, so the
launcher reads it from `config_args`.

Before:

```python
def triton_tem_fused_mm(arg_A, arg_B, out_ptr0):
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    ...

def launcher(arg_A, arg_B, out_ptr0, stream):
    grid_0 = ...
    grid_1 = 1
    grid_2 = 1
    runner(arg_A, arg_B, out_ptr0)
```

After:

```python
def triton_tem_fused_mm(arg_A, arg_B, out_ptr0, distribution_tile):
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    ...

def launcher(arg_A, arg_B, out_ptr0, stream):
    grid_0 = ...
    grid_1 = 1
    grid_2 = 1
    _any_slc = pointer_is_slc(arg_A) or pointer_is_slc(arg_B) or pointer_is_slc(out_ptr0)
    distribution_tile = 128 if _any_slc else 0
    runner(arg_A, arg_B, out_ptr0, distribution_tile)
```

## C++ wrapper

The eager wrapper and the lazy launcher both declare the scalar after the
device pointers, then pass its address in the original argument slot.

```cpp
#ifdef USE_ROCM
auto inductor_ptr_is_slc = [](const void* p) -> bool {
    if (!p) return false;
    hipPointerAttribute_t a{};
    if (hipPointerGetAttributes(&a, p) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    return (a.allocationFlags & 2147483648u) != 0;
};
int32_t inductor_distribution_tile =
    inductor_ptr_is_slc(var_0) || inductor_ptr_is_slc(var_1)
        ? static_cast<int32_t>(1024) : 0;
#else
int32_t inductor_distribution_tile = 0;
#endif
```

`1024` is a literal once the config is known. On the lazy path, before a
config is chosen, an `XBLOCK` tile is `kernel_result.xblocks[0]` (and
`yblocks` / `zblocks` / `r0blocks` for the other axes). A precomputed grid
with one shared block size stays a literal. Mixed sizes become a switch on
`kernel_result.config_index`.

## User kernel

```python
@triton.jit
def my_kernel(x, y, distribution_tile):
    ...
```

The wrapper may pass a placeholder. The launcher overwrites it:

```python
def launcher(x, y, distribution_tile, stream):
    grid_0 = ...
    grid_1 = 1
    grid_2 = 1
    _any_slc = pointer_is_slc(x) or pointer_is_slc(y)
    distribution_tile = 128 if _any_slc else 0
    runner(x, y, distribution_tile)
```

## Tests

`test/inductor/test_hip_pointer_slc.py` covers the attribute bit, failed
lookups, both launcher modes, first-tile selection for kernels and
templates, and the C++ snippets. It does not need a GPU. The HIP query has
been checked against real device tensors on ROCm. An end-to-end Inductor
compile with this branch has not been run.
