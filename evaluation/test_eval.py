import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluation.prepare import dump, write_rows
from evaluation.run import fingerprint, parse_output
from evaluation.score import answer_list


class ActionBoundaryTests(unittest.TestCase):
    def test_tool_text_inside_reasoning_is_not_executed(self):
        text = '<think>Consider <tool_call>{"name":"vs_visit","arguments":{}}</tool_call></think><response>42</response>'
        self.assertEqual(parse_output(text), ('answer', '42'))

    def test_truncated_reasoning_is_not_an_answer(self):
        with self.assertRaises(ValueError):
            parse_output('<think>The answer might be 42')

    def test_two_actions_are_not_executed(self):
        call = '<tool_call>{"name":"vs_calculate","arguments":{"expression":"1+2"}}</tool_call>'
        with self.assertRaises(ValueError):
            parse_output(call + call)

    def test_final_answer_and_tool_call_are_ambiguous(self):
        with self.assertRaises(ValueError):
            parse_output('<tool_call>{"name":"vs_visit","arguments":{}}</tool_call><response>42</response>')

    def test_final_answers_require_tags_for_both_models(self):
        for text in ('The answer is 42.', '', '<think>42</think>',
                     '<response>42', 'prefix <response>42</response>',
                     '<response>42</response> suffix', '<response><response>42</response>',
                     '<response>42</response></response>', '<response> </response>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_output(text)

    def test_tagged_final_answers_allow_reasoning_and_whitespace(self):
        for text in ('<response>42</response>', ' \n<response> 42 </response>\n',
                     '<think>Consider the evidence.</think><response>42</response>'):
            with self.subTest(text=text):
                self.assertEqual(parse_output(text), ('answer', '42'))

    def test_lists_do_not_execute_generated_code(self):
        text = '__import__("os").getcwd()'
        self.assertEqual(answer_list(text), [text])
        self.assertEqual(answer_list('["Australia", "New Zealand"]'), ['Australia', 'New Zealand'])


class PairedComparisonTests(unittest.TestCase):
    def fixture(self, root):
        protocol_id = 'same-protocol'
        for model, score in [('base', 0), ('sft', 1)]:
            run = root / model
            scores = root / (model + '_scores')
            run.mkdir(); scores.mkdir()
            dump(run / 'run.json', {'run_id': model, 'model': model,
                                   'protocol_sha256': protocol_id,
                                   'protocol': {'selected_ids': ['task']}})
            pred = {'id': 'task', 'run_id': model, 'infrastructure_error': False}
            write_rows(run / 'predictions.jsonl', [pred])
            dump(scores / 'score_config.json', {'run_id': model, 'policy_id': 'judge',
                                              'policy': {'metric': 'test-only'},
                                              'protocol_sha256': protocol_id})
            write_rows(scores / 'scores.jsonl', [{'id': 'task', 'input_fingerprint': 'input',
                       'source_dataset': 'test', 'policy_id': 'judge',
                       'prediction_sha256': fingerprint(pred), 'score': score,
                       'status': 'answered', 'tool_calls': 1}])
        return [sys.executable, '-m', 'evaluation.compare',
                '--base-run', str(root / 'base'), '--sft-run', str(root / 'sft'),
                '--base-scores', str(root / 'base_scores'), '--sft-scores', str(root / 'sft_scores'),
                '--output', str(root / 'comparison')]

    def test_difference_is_percentage_points(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = subprocess.run(self.fixture(root), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'comparison/comparison.json').read_text())
            self.assertEqual(report['macro_delta_percentage_points'], 100)

    def test_missing_scores_cannot_silently_shrink_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); command = self.fixture(root)
            (root / 'sft_scores/scores.jsonl').write_text('')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'comparison/comparison.csv').exists())

    def test_different_protocols_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); command = self.fixture(root)
            path = root / 'sft/run.json'
            value = json.loads(path.read_text()); value['protocol_sha256'] = 'different'
            dump(path, value)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_changed_predictions_need_rescoring(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); command = self.fixture(root)
            write_rows(root / 'sft/predictions.jsonl', [{'id': 'task', 'run_id': 'sft', 'answer': 'changed'}])
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
