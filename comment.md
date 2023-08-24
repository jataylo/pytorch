conv_configs

(BEFORE CHANGE)
(
(64, 256, 16, 2, 4), 
(256, 64, 16, 2, 4), 
(1024, 16, 16, 1, 8), 
(128, 128, 32, 2, 8), 
(64, 64, 32, 2, 4), 
(64, 256, 32, 2, 8), 
(256, 64, 32, 2, 8))

(AFTER CHANGE - None rocm)
(
(64, 256, 16, 2, 4), 
(256, 64, 16, 2, 4), 
(1024, 16, 16, 1, 8), 
(128, 128, 32, 2, 8), 
(64, 64, 32, 2, 4), 
(64, 256, 32, 2, 8), 
(256, 64, 32, 2, 8)
)

(AFTER CHANGE - ROCm)
(
(64, 256, 16, 1, 4), 
(256, 64, 16, 1, 4), 
(1024, 16, 16, 1, 8), 
(128, 128, 32, 1, 8), 
(64, 64, 32, 1, 4), 
(64, 256, 32, 1, 8), 
(256, 64, 32, 1, 8)
)



mm_configs

(BEFORE CHANGE)
mm_configs
((64, 64, 32, 2, 4), 
(64, 128, 32, 3, 4), 
(128, 64, 32, 3, 4), 
(64, 128, 32, 4, 8), 
(128, 64, 32, 4, 8), 
(64, 32, 32, 5, 8), 
(32, 64, 32, 5, 8), 
(128, 128, 32, 2, 8), 
(64, 64, 64, 3, 8), 
(32, 32, 128, 2, 4), 
(64, 64, 16, 2, 4), 
(32, 32, 16, 1, 2))

int8_mm_configs
(
(64, 64, 32, 2, 4), 
(64, 128, 32, 3, 4), 
(128, 64, 32, 3, 4), 
(64, 128, 32, 4, 8), 
(128, 64, 32, 4, 8), 
(64, 32, 32, 5, 8), 
(32, 64, 32, 5, 8), 
(128, 128, 32, 2, 8), 
(64, 64, 64, 3, 8), 
(128, 256, 128, 3, 8), 
(256, 128, 128, 3, 8)
)

(AFTER CHANGE - No ROCm)
mm_configs
(
(64, 64, 32, 2, 4), 
(64, 128, 32, 3, 4), 
(128, 64, 32, 3, 4), 
(64, 128, 32, 4, 8), 
(128, 64, 32, 4, 8), 
(64, 32, 32, 5, 8), 
(32, 64, 32, 5, 8), 
(128, 128, 32, 2, 8), 
(64, 64, 64, 3, 8), 
(32, 32, 128, 2, 4), 
(64, 64, 16, 2, 4), 
(32, 32, 16, 1, 2)
)

int8_configs
(
(64, 64, 32, 2, 4), 
(64, 128, 32, 3, 4), 
(128, 64, 32, 3, 4), 
(64, 128, 32, 4, 8), 
(128, 64, 32, 4, 8), 
(64, 32, 32, 5, 8), 
(32, 64, 32, 5, 8), 
(128, 128, 32, 2, 8), 
(64, 64, 64, 3, 8), 
(128, 256, 128, 3, 8), 
(256, 128, 128, 3, 8)
)


(AFTER CHANGE - ROCm)
mm_configs
(
(64, 64, 32, 1, 4), 
(64, 128, 32, 1, 4), 
(128, 64, 32, 1, 4), 
(64, 128, 32, 1, 8), 
(128, 64, 32, 1, 8), 
(64, 32, 32, 1, 8), 
(32, 64, 32, 1, 8), 
(128, 128, 32, 1, 8), 
(64, 64, 64, 1, 8), 
(64, 64, 16, 1, 4), 
(32, 32, 16, 1, 2)
)

(AFTER CHANGE - ROCM)
int8_mm_configs
(
(64, 64, 32, 1, 4), 
(64, 128, 32, 1, 4), 
(128, 64, 32, 1, 4), 
(64, 128, 32, 1, 8), 
(128, 64, 32, 1, 8), 
(64, 32, 32, 1, 8), 
(32, 64, 32, 1, 8), 
(128, 128, 32, 1, 8), 
(64, 64, 64, 1, 8)
)

mm_plus_mm

Original:
[<triton.runtime.autotuner.Config object at 0x7f9dcfc4e1d0>, <triton.runtime.autotuner.Config object at 0x7f9dcfc4e5f0>, <triton.runtime.autotuner.Config object at 0x7f9dcfc4d270>, <triton.runtime.autotuner.Config object at 0x7f9dcfb579d0>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57790>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57670>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57970>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57ac0>, <triton.runtime.autotuner.Config object at 0x7f9dcfb579a0>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57a60>]

After change - no ROCm
[<triton.runtime.autotuner.Config object at 0x7f9dcfb57b50>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57b80>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57be0>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57c40>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57ca0>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57d00>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57d60>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57dc0>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57e20>, <triton.runtime.autotuner.Config object at 0x7f9dcfb57e80>]

MAX_AUTOTUNE log to verify
ExternKernelCaller(extern_kernels._mm_plus_mm)
TritonTemplateCaller(/tmp/torchinductor_root/w7/cw7tibckrubr4blgzo2wxqsuszcu6kktg3m3ulbywdjdx3loekyf.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=4)
TritonTemplateCaller(/tmp/torchinductor_root/zi/cziadg6jiurq6wi5c3sml3d2bngkgyldg3bvzekwmzgdynvqccjf.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/jv/cjvicdxe77ywsfgipbipnobl7sg25fnd4luplmcxkg53jxyfsji5.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=16)
TritonTemplateCaller(/tmp/torchinductor_root/oo/cooaupuodzdjzlzsq6hk6p3q6sbfwxjxjj3il4yxot7fqeq4zyet.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/u7/cu7wncwk4rgrmz5ibsywzozc73a4x4xgg46cetvu7xohakeduaxu.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/3p/c3pmqu4wxiri3pctkeq7ugzjmta75xg522wka76akkgnqlkhtvjw.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/ff/cffv5gxhrk5mll53j2w4hvoeclaumnjruxonmto4o34dn2zgqmn6.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/bs/cbs7rekdixkkbypx7eomq3s7yntk7dqqt6klbcelxyrbntn4gvmd.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=32, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/wn/cwnc74qawqd2fpt45sqbsm2hzijhtoek6cr7o75apy5ks2eslxej.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=16, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=4)
TritonTemplateCaller(/tmp/torchinductor_root/nc/cncaszarsw2mxvumukoqtoqmedmfj6dyccev6pgyzbztbxyah5ng.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=16, BLOCK_M=32, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=2)



After change - ROCm
[<triton.runtime.autotuner.Config object at 0x7f3ad952fe50>, <triton.runtime.autotuner.Config object at 0x7f3ad952feb0>, <triton.runtime.autotuner.Config object at 0x7f3ad952ff10>, <triton.runtime.autotuner.Config object at 0x7f3ad952ff70>, <triton.runtime.autotuner.Config object at 0x7f3ad952ffd0>, <triton.runtime.autotuner.Config object at 0x7f3ad9578070>, <triton.runtime.autotuner.Config object at 0x7f3ad95780d0>, <triton.runtime.autotuner.Config object at 0x7f3ad9578130>, <triton.runtime.autotuner.Config object at 0x7f3ad9578190>]

MAX_AUTOTUNE log to verify:
ExternKernelCaller(extern_kernels._mm_plus_mm)
TritonTemplateCaller(/tmp/torchinductor_root/e3/ce3giwk6mjj4doboazushhkrlbuqg2bdsxp64fpr4bszl4vrt2tb.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=4)
TritonTemplateCaller(/tmp/torchinductor_root/hl/chlf74iccwcztxrqyeclfytc4y5iglfkap43kyuuarshh4zp3zjq.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/si/csie65r5karqtew34dxaoevjjbe3nr4zbg7sxw6tzoz5o7wmlk6d.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=16)
TritonTemplateCaller(/tmp/torchinductor_root/r7/cr7gmanqx5yda5fpjq7mqu7gipgmbsx4tth32v522oxh4q6kkodk.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/yl/cylqmexqqa4qn72tyqdg4lqxvvb6pnelked5d6tndk67daxxn4ps.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/3p/c3pmqu4wxiri3pctkeq7ugzjmta75xg522wka76akkgnqlkhtvjw.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=128, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/ff/cffv5gxhrk5mll53j2w4hvoeclaumnjruxonmto4o34dn2zgqmn6.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=8)
TritonTemplateCaller(/tmp/torchinductor_root/uw/cuwh572otqokk5n4fbjaaci3bbhb5mhfoiklphkazueslrbm6np4.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=16, BLOCK_M=64, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=4)
TritonTemplateCaller(/tmp/torchinductor_root/nc/cncaszarsw2mxvumukoqtoqmedmfj6dyccev6pgyzbztbxyah5ng.py, ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=16, BLOCK_M=32, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=1, num_warps=2)

