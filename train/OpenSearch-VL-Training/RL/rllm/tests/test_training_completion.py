"""Run the real loop's scheduling statements without loading GPU libraries."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "rllm/trainer/verl/agent_workflow_trainer.py"


def scheduling_loop():
    tree = ast.parse(SOURCE.read_text())
    fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit_agent")
    epoch_loop = next(n for n in fit.body if isinstance(n, ast.For))
    batch_loop = next(n for n in epoch_loop.body if isinstance(n, ast.For))

    def code(node):
        return ast.unparse(node)

    # Keep the production counters, resume guard, save conditions, validation
    # conditions and return ordering. Only GPU work and metric collection are
    # replaced; expected batch numbers are asserted independently below.
    initialize = next(n for n in fit.body if code(n) == "self.global_steps = 0")
    load = next(n for n in fit.body if code(n) == "self._load_checkpoint()")
    resume_guard = next((n for n in fit.body if isinstance(n, ast.If) and "self.total_training_steps" in code(n.test)), None)
    increment = next(n for n in fit.body if isinstance(n, ast.AugAssign) and code(n.target) == "self.global_steps")
    periodic_validation = next(n for n in ast.walk(batch_loop) if isinstance(n, ast.If) and "test_freq" in code(n.test))
    periodic_save = next(n for n in ast.walk(batch_loop) if isinstance(n, ast.If) and "save_freq" in code(n.test) and "%" in code(n.test))
    finish = next(n for n in batch_loop.body if isinstance(n, ast.If) and "self.total_training_steps" in code(n.test))
    tail_increment = next(n for n in batch_loop.body if isinstance(n, ast.AugAssign) and code(n.target) == "self.global_steps")
    selected = {id(finish), id(tail_increment)}
    tail = [n for n in batch_loop.body if id(n) in selected]
    batch_loop.body = ast.parse("validated_this_step = False\nself.processed.append(self.global_steps)\n").body + [
        ast.If(test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="batch_dict", ctx=ast.Load())),
               body=[periodic_validation, periodic_save], orelse=[]),
    ] + tail
    epoch_loop.body = [batch_loop]
    function = ast.FunctionDef(
        name="run", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[initialize, load] + ([resume_guard] if resume_guard is not None else []) + [increment]
             + ast.parse("last_saved_step = None\n").body + [epoch_loop],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "marked_timer": lambda *a, **k: nullcontext(), "timing_raw": {}, "metrics": {},
        "pprint": lambda *a: None,
        "logger": SimpleNamespace(log=lambda **k: None),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["run"]


class Trainer:
    def __init__(self, total, save=50, test=50, resumed=0, empty=False):
        self.total_training_steps = total
        self.config = SimpleNamespace(trainer=SimpleNamespace(total_epochs=1, save_freq=save, test_freq=test))
        self.train_dataloader = [empty] * max(total, 1)
        self.resumed = resumed
        self.processed, self.saved, self.validated, self.events = [], [], [], []
        self.val_reward_fn = object()
        self.agent_execution_engine = SimpleNamespace(set_training_step=lambda *a, **k: None)

    def _load_checkpoint(self):
        self.global_steps = self.resumed

    def _save_checkpoint(self):
        self.saved.append(self.global_steps)
        self.events.append(("save", self.global_steps))

    def _validate_agent(self):
        self.validated.append(self.global_steps)
        self.events.append(("validate", self.global_steps))
        return {}


class CompletionTests(unittest.TestCase):
    def run_trainer(self, *args, **kwargs):
        trainer = Trainer(*args, **kwargs)
        scheduling_loop()(trainer)
        return trainer

    def test_three_requested_batches_really_run_three(self):
        t = self.run_trainer(3, save=1, test=1)
        self.assertEqual(t.processed, [1, 2, 3])
        self.assertEqual(t.saved, [1, 2, 3])
        self.assertEqual(t.validated, [1, 2, 3])

    def test_formal_run_saves_final_nonperiodic_batch(self):
        t = self.run_trainer(248)
        self.assertEqual(t.processed, list(range(1, 249)))
        self.assertEqual(t.saved, [50, 100, 150, 200, 248])
        self.assertEqual(t.validated, [50, 100, 150, 200, 248])

    def test_last_periodic_checkpoint_is_not_saved_twice(self):
        t = self.run_trainer(100)
        self.assertEqual(t.saved, [50, 100])
        self.assertEqual(t.validated, [50, 100])

    def test_resuming_runs_only_remaining_batch_numbers(self):
        t = self.run_trainer(248, resumed=246)
        self.assertEqual(t.processed, [247, 248])
        self.assertEqual(t.saved, [248])

    def test_completed_checkpoint_does_not_run_again(self):
        t = self.run_trainer(3, resumed=3)
        self.assertEqual(t.processed, [])
        self.assertEqual(t.saved, [])

    def test_final_filtered_batch_still_saves_data_position(self):
        t = self.run_trainer(3, empty=True)
        self.assertEqual(t.processed, [1, 2, 3])
        self.assertEqual(t.saved, [3])

    def test_disabled_periodic_saving_stays_disabled(self):
        t = self.run_trainer(3, save=-1)
        self.assertEqual(t.saved, [])

    def test_nonperiodic_final_save_precedes_final_validation(self):
        t = self.run_trainer(3)
        self.assertEqual(t.events, [("save", 3), ("validate", 3)])

    def test_one_batch_target(self):
        t = self.run_trainer(1)
        self.assertEqual(t.processed, [1])
        self.assertEqual(t.saved, [1])


if __name__ == "__main__":
    unittest.main()
