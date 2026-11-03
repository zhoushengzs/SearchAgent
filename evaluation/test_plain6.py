"""Check that adding the ordinary prompt does not change native tool behavior."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from evaluation.grpo_native import evaluate_task, load_runtime
from evaluation.plain6_benchmark import ALL_CONDITIONS, compare_protocol, score_all
from evaluation.plain6_run import ordinary_prompt, ordinary_system
from evaluation.test_grpo_native import FakeModel


class ComparisonChecks(unittest.TestCase):
    def test_budget_or_tool_difference_cannot_be_reported_as_prompt_only(self):
        native = {'system': 'native', 'code_sha256': {'shared.py': 'a'}, 'max_steps': 10, 'tools': ['six']}
        plain = {**native, 'system': 'ordinary', 'code_sha256': {'shared.py': 'a', 'extra.py': 'b'}}
        compare_protocol(plain, native, {'extra.py': 'b'})
        for key, value in [('max_steps', 20), ('tools', ['nine'])]:
            with self.assertRaises(ValueError):
                compare_protocol({**plain, key: value}, native, {'extra.py': 'b'})

    def test_five_condition_report_and_scoring_credential_are_scoped_to_scorer(self):
        from evaluation import unified_score as scorer
        original_conditions = scorer.CONDITIONS
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summaries = {c: {'correct': 1, 'count': 778, 'micro_accuracy_pct': 0.12, 'answered': 2}
                         for c in ALL_CONDITIONS}
            def fake_main():
                self.assertEqual(len(scorer.CONDITIONS), 5)
                self.assertEqual(os.environ['TEST_JUDGE_KEY'], 'test-only-credential')
                scorer.render_results(output, summaries)
                scorer.write_json_atomic(output / 'SERVER_SUCCESS.json', {'conditions': 4, 'count_each': 778})
            with patch.object(scorer, 'main', side_effect=fake_main), patch.dict(os.environ, {}, clear=True):
                score_all(output, {'schema': 'test'}, {'TEST_JUDGE_KEY': 'test-only-credential'}, 'TEST_JUDGE_KEY')
                self.assertNotIn('TEST_JUDGE_KEY', os.environ)
            self.assertEqual(scorer.CONDITIONS, original_conditions)
            self.assertEqual(json.loads((output / 'SERVER_SUCCESS.json').read_text())['conditions'], 5)
            self.assertEqual(len(json.loads((output / 'summary.json').read_text())['conditions']), 5)
            self.assertIn('五组', (output / '.codex/RESULTS.md').read_text())


@unittest.skipUnless(os.environ.get('NATIVE_GRPO_RUNTIME'), 'Set the real native runtime path')
class ActualPromptComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent, cls.session = load_runtime(os.environ['NATIVE_GRPO_RUNTIME'])

    def test_only_system_changes_in_actual_first_input_and_calculator_still_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'source.png'; Image.new('RGB', (8, 8), 'red').save(image)
            task = {'id': 't', 'source_dataset': 'fixture', 'input_fingerprint': 'fp',
                    'model_input': {'question': 'What is 2+2?', 'images': [{'path': str(image)}]},
                    'answer': 'SECRET_REFERENCE'}
            texts = ['<tool_call>{"name":"vs_calculate","arguments":{"expression":"2+2"}}</tool_call>',
                     '<response>4</response>']
            args = SimpleNamespace(output=root / 'native', max_steps=2, prompt_tokens=28672,
                                   max_new_tokens=4096, max_pixels=262144, temperature=0.0)
            native_prompt = self.session.system_prompt()
            a = evaluate_task(task, FakeModel(texts), self.agent, self.session, args, 'native')
            args.output = root / 'plain'
            with ordinary_prompt(self.agent, self.session):
                b = evaluate_task(task, FakeModel(texts), self.agent, self.session, args, 'plain')
            self.assertEqual(self.session.system_prompt(), native_prompt)
            for row in [a, b]:
                self.assertEqual((row['status'], row['answer'], row['tool_calls']), ('answered', '4', 1))
            inputs = []
            for condition in ['native', 'plain']:
                calls = (root / condition / 'artifacts/t/raw_calls.jsonl').read_text()
                self.assertNotIn('SECRET_REFERENCE', calls)
                messages = json.loads(calls.splitlines()[0])['messages']
                for message in messages:
                    if 'images' in message:
                        message['images'] = [image['sha256'] for image in message['images']]
                inputs.append(messages)
            self.assertEqual(inputs[0][1:], inputs[1][1:])
            self.assertNotEqual(inputs[0][0], inputs[1][0])
            plain_prompt = inputs[1][0]['content']
            self.assertEqual(plain_prompt, ordinary_system(self.session))
            schemas = plain_prompt.split('<tools>\n')[1].split('\n</tools>')[0]
            self.assertEqual([json.loads(line) for line in schemas.splitlines()], list(self.session.get_all_tools().values()))


if __name__ == '__main__':
    unittest.main()
