"""Check hardware selection without importing Ray or loading a model."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py"


def selected_backend(major):
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_default_mm_attention_backend")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: (major, 0)))}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[function.name]()


class BackendTests(unittest.TestCase):
    def test_hopper_h800_retains_fa3(self):
        self.assertEqual(selected_backend(9), "fa3")

    def test_blackwell_b200_avoids_fa3(self):
        self.assertEqual(selected_backend(10), "triton_attn")

    def test_blackwell_rtx_pro_6000_avoids_fa3(self):
        self.assertEqual(selected_backend(12), "triton_attn")

    def test_ampere_avoids_fa3(self):
        self.assertEqual(selected_backend(8), "triton_attn")

    def test_launch_uses_hardware_choice_and_preserves_user_override(self):
        tree = ast.parse(SOURCE.read_text())
        args = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "args" for t in n.targets)
                    and isinstance(n.value, ast.Dict))
        fields = {k.value: v for k, v in zip(args.keys, args.values) if isinstance(k, ast.Constant)}
        self.assertEqual(ast.unparse(fields["mm_attention_backend"]), "_default_mm_attention_backend()")
        self.assertEqual(ast.unparse(fields["attention_backend"]), "attention_backend")
        self.assertIsNone(args.keys[-1])
        self.assertEqual(ast.unparse(args.values[-1]), "engine_kwargs")


if __name__ == "__main__":
    unittest.main()
