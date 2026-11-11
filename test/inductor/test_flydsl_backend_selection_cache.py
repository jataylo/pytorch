# Owner(s): ["module: inductor"]

import ast
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestFlyDSLBackendSelectionCache(TestCase):
    @parametrize("winner", [0, 1])
    def test_cached_candidate_is_honored(self, winner):
        source = (
            Path(__file__).resolve().parents[2]
            / "torch/_inductor/codegen/flydsl/flydsl_simd_runtime.py"
        )
        node = next(
            item
            for item in ast.parse(source.read_text()).body
            if isinstance(item, ast.FunctionDef) and item.name == "pick_fastest"
        )
        tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        calls, events = [], []

        def first(*args, stream=None):
            calls.append(0)

        def second(*args, stream=None):
            calls.append(1)

        mains = [first, second]
        torch_stub = ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(is_current_stream_capturing=lambda: False)
        torch_stub.Tensor = type("NoRealTensor", (), {})
        package = ModuleType("_flydsl_cache_fixture")
        package.__path__ = []
        diagnostics = ModuleType("_flydsl_cache_fixture.flydsl_diagnostics")
        diagnostics.record_kernel_decision = events.append
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "choice"
            path.write_text(mains[winner].__name__ + "\n")
            namespace = {
                "__package__": package.__name__,
                "_pick_path": lambda candidates: str(path),
            }
            exec(compile(tree, str(source), "exec"), namespace)
            with patch.dict(
                sys.modules,
                {
                    "torch": torch_stub,
                    package.__name__: package,
                    diagnostics.__name__: diagnostics,
                },
            ):
                actual = namespace["pick_fastest"](
                    mains,
                    (),
                    (),
                    metadata={
                        "category": "backend_selection",
                        "backends": ["flydsl", "triton"],
                    },
                )
        self.assertIs(actual, mains[winner])
        self.assertEqual(calls, [winner])
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["cache_hit"])
        self.assertEqual(events[0]["selected_backend"], ["flydsl", "triton"][winner])


if __name__ == "__main__":
    run_tests()
