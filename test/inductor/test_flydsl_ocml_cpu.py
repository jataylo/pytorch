# Owner(s): ["module: inductor"]
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch._inductor.codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


RUNTIME_UNAVAILABLE_REASON = _flydsl_runtime_unavailable_reason()
if RUNTIME_UNAVAILABLE_REASON is None:
    from torch._inductor.codegen.flydsl import flydsl_simd_runtime as fdp
else:
    fdp = None

LIBRARY_NAMES = (
    "ocml",
    "oclc_finite_only_off",
    "oclc_unsafe_math_off",
    "oclc_daz_opt_off",
    "oclc_correctly_rounded_sqrt_on",
    "oclc_wavefrontsize64_on",
    "oclc_isa_version_950",
    "oclc_abi_version_500",
)
OPTIONAL_NAMES = {"oclc_daz_opt_off", "oclc_correctly_rounded_sqrt_on"}
REQUIRED_NAMES = tuple(name for name in LIBRARY_NAMES if name not in OPTIONAL_NAMES)


@unittest.skipIf(
    RUNTIME_UNAVAILABLE_REASON is not None, RUNTIME_UNAVAILABLE_REASON or ""
)
@instantiate_parametrized_tests
class TestFlyDSLOCMLResolver(TestCase):
    @parametrize(
        "missing",
        [
            (),
            ("oclc_daz_opt_off",),
            ("oclc_correctly_rounded_sqrt_on",),
            ("oclc_daz_opt_off", "oclc_correctly_rounded_sqrt_on"),
        ],
    )
    def test_legacy_and_modern_control_libraries(self, missing):
        from flydsl.runtime import device

        root = Path("/test/rocm/amdgcn/bitcode")
        existing = {
            str(root / f"{name}.bc") for name in LIBRARY_NAMES if name not in missing
        }
        with (
            patch.dict(os.environ, {"ROCM_PATH": "/test/rocm"}),
            patch.object(device, "get_rocm_arch", return_value="gfx950"),
            patch("os.path.isfile", side_effect=existing.__contains__),
        ):
            actual = fdp._device_libs()
        self.assertEqual(
            actual,
            [str(root / f"{name}.bc") for name in LIBRARY_NAMES if name not in missing],
        )

    @parametrize("missing", REQUIRED_NAMES)
    def test_missing_required_library_is_an_error(self, missing):
        from flydsl.runtime import device

        root = Path("/test/rocm/amdgcn/bitcode")
        existing = {
            str(root / f"{name}.bc") for name in LIBRARY_NAMES if name != missing
        }
        with (
            patch.dict(os.environ, {"ROCM_PATH": "/test/rocm"}),
            patch.object(device, "get_rocm_arch", return_value="gfx950"),
            patch("os.path.isfile", side_effect=existing.__contains__),
        ):
            with self.assertRaisesRegex(
                FileNotFoundError, re.escape(str(root / f"{missing}.bc"))
            ):
                fdp._device_libs()


@unittest.skipIf(
    RUNTIME_UNAVAILABLE_REASON is not None, RUNTIME_UNAVAILABLE_REASON or ""
)
class TestFlyDSLOCMLPipeline(TestCase):
    def test_precise_math_target_options(self):
        from flydsl.compiler.backends import get_backend

        fragments = get_backend("rocm", arch="gfx950").pipeline_fragments(
            compile_hints={}
        )
        target = next(
            fragment
            for fragment in fragments
            if fragment.startswith("rocdl-attach-target")
        )
        for option in [
            "chip=gfx950",
            "correct-sqrt=true",
            "daz=false",
            "finite-only=false",
            "unsafe-math=false",
            "fast=false",
            "wave64=true",
        ]:
            self.assertIn(option, target)


@unittest.skipIf(
    RUNTIME_UNAVAILABLE_REASON is not None, RUNTIME_UNAVAILABLE_REASON or ""
)
@instantiate_parametrized_tests
class TestFlyDSLOCMLCompileCPU(TestCase):
    def setUp(self):
        super().setUp()
        from flydsl.runtime import device

        root = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "amdgcn/bitcode"
        if (
            not (root / "ocml.bc").is_file()
            or not (root / "oclc_isa_version_950.bc").is_file()
        ):
            self.skipTest("requires a ROCm SDK with gfx950 device libraries")
        with patch.object(device, "get_rocm_arch", return_value="gfx950"):
            self.libraries = fdp._device_libs()

    def test_real_sdk_paths_and_required_controls(self):
        self.assertTrue(all(Path(path).is_file() for path in self.libraries))
        self.assertTrue(
            {f"{name}.bc" for name in REQUIRED_NAMES}
            <= {Path(path).name for path in self.libraries}
        )

    @parametrize("output_format", ["fatbin", "llvm", "assembly"])
    def test_compile_dynamic_ocml_math_without_device_runtime(self, output_format):
        from flydsl._mlir import ir
        from flydsl._mlir.dialects import gpu
        from flydsl.compiler.backends import get_backend, rocm
        from flydsl.compiler.jit_function import MlirCompiler
        from flydsl.runtime import device, device_runtime

        text = """module attributes {gpu.container_module} {
          gpu.module @kernels {
            llvm.func @__ocml_log1p_f32(f32) -> f32
            llvm.func @__ocml_sqrt_f32(f32) -> f32
            gpu.func @ocml_cpu_probe(%input: !llvm.ptr<1>, %out: !llvm.ptr<1>) kernel {
              %x = llvm.load %input : !llvm.ptr<1> -> f32
              %log = llvm.call @__ocml_log1p_f32(%x) : (f32) -> f32
              %sqrt = llvm.call @__ocml_sqrt_f32(%x) : (f32) -> f32
              %value = arith.addf %log, %sqrt : f32
              llvm.store %value, %out : f32, !llvm.ptr<1>
              gpu.return
            }
          }
        }"""
        backend = get_backend("rocm", arch="gfx950")
        original_pipeline = backend.pipeline_fragments

        def pipeline(*, compile_hints):
            fragments = original_pipeline(compile_hints=compile_hints)
            return [
                fragment.replace("format=fatbin", f"format={output_format}")
                for fragment in fragments
            ]

        initialized_before = torch.cuda.is_initialized()
        prohibited = RuntimeError(
            "CPU compiler test must not query or initialize a GPU"
        )
        with (
            patch.object(torch.cuda, "_lazy_init", side_effect=prohibited),
            patch.object(device, "get_rocm_arch", side_effect=prohibited),
            patch.object(rocm, "get_rocm_arch", side_effect=prohibited),
            patch.object(device_runtime, "get_device_runtime", side_effect=prohibited),
            patch.object(backend, "pipeline_fragments", side_effect=pipeline),
            patch.dict(
                os.environ,
                {"FLYDSL_COMPILE_BACKEND": "rocm", "FLYDSL_COMPILE_LLVM_DIR": ""},
            ),
        ):
            with ir.Context(), ir.Location.unknown():
                module = ir.Module.parse(text)
                compiled = MlirCompiler.compile(
                    module,
                    arch="gfx950",
                    func_name="ocml_cpu_probe",
                    link_libs=self.libraries,
                )
                binaries = [
                    operation
                    for operation in compiled.body.operations
                    if operation.operation.name == "gpu.binary"
                ]
                self.assertEqual(len(binaries), 1)
                objects = list(ir.ArrayAttr(binaries[0].attributes["objects"]))
                self.assertEqual(len(objects), 1)
                payload = gpu.ObjectAttr(objects[0]).object
                if output_format == "fatbin":
                    self.assertTrue(payload.startswith(b"\x7fELF"))
                elif output_format == "llvm":
                    self.assertTrue(payload.startswith(b"BC\xc0\xde"))
                else:
                    assembly = payload.decode("utf-8")
                    self.assertRegex(assembly, r"\.amdhsa_float_denorm_mode_32\s+3")
                    self.assertRegex(assembly, r"\.wavefront_size:\s+64")
                    self.assertIn("gfx950", assembly)
        self.assertEqual(torch.cuda.is_initialized(), initialized_before)


if __name__ == "__main__":
    run_tests()
