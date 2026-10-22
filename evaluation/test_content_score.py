import copy
import unittest
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from evaluation.score import content_answer
from evaluation.score import main
from evaluation.prepare import digest, write_rows
from evaluation.common import task_fingerprint


class ContentAnswerTests(unittest.TestCase):
    def prediction(self, text, status='invalid_output', generation_status='ok'):
        return {'status': status, 'answer': '', 'turns': [{'text': text, 'status': generation_status}]}

    def test_recovers_tagged_answer_without_changing_source(self):
        pred = self.prediction('Explanation.\n<response>42</response>\nFurther explanation.')
        before = copy.deepcopy(pred)
        self.assertEqual(content_answer(pred), '42')
        self.assertEqual(pred, before)

    def test_plain_answer_allowed_and_reasoning_not_submitted(self):
        self.assertEqual(content_answer(self.prediction('<think>Maybe 7</think>The answer is 42.')), 'The answer is 42.')
        self.assertEqual(content_answer(self.prediction('<think>42</think>')), '')

    def test_does_not_recover_actions_ambiguous_or_broken_output(self):
        for text in ('<tool_call>{"name":"web_search","arguments":{}}</tool_call>',
                     '<tool_call>{}</tool_call><response>42</response>',
                     '<response>42', '<think>unfinished',
                     '<response>42</response><response>7</response>', '<response> </response>'):
            with self.subTest(text=text):
                self.assertEqual(content_answer(self.prediction(text)), '')

    def test_truncated_or_failed_generation_not_recovered(self):
        for status in ('length', 'step_limit', 'inference_error', 'unavailable_tool'):
            self.assertEqual(content_answer(self.prediction('<response>42</response>', status)), '')
        self.assertEqual(content_answer(self.prediction('<response>42</response>', generation_status='length')), '')
        self.assertEqual(content_answer({'status': 'answered', 'answer': '42'}), '42')

    def test_partial_content_score_preserves_source_and_labels_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks = []
            for index in range(3):
                task = {'id': f't{index}', 'source_dataset': 'fixture',
                        'model_input': {'question': 'What is the answer?', 'images': []},
                        'dataset_provenance': {}, 'categories': {}}
                task['input_fingerprint'] = task_fingerprint(task)
                tasks.append(task)
            write_rows(root / 'tasks.jsonl', tasks)
            write_rows(root / 'references.jsonl', [{**t, 'answer': '42'} for t in tasks])
            (root / 'run.json').write_text(json.dumps({'run_id': 'fixture', 'protocol_sha256': 'fixture',
                'tasks_file': str(root / 'tasks.jsonl'), 'protocol': {
                    'tasks_sha256': digest(root / 'tasks.jsonl'), 'selected_ids': [t['id'] for t in tasks]}}))
            pred = {**self.prediction('Explanation <response>42</response>'),
                    'id': 't0', 'run_id': 'fixture', 'source_dataset': 'fixture',
                    'input_fingerprint': tasks[0]['input_fingerprint'], 'infrastructure_error': False,
                    'tool_calls': 0, 'elapsed_seconds': 1}
            failed = {**pred, 'id': 't1', 'input_fingerprint': tasks[1]['input_fingerprint'],
                      'infrastructure_error': True}
            write_rows(root / 'predictions.jsonl', [pred, failed])
            original = (root / 'predictions.jsonl').read_bytes()
            argv = ['score', '--run', str(root), '--references', str(root / 'references.jsonl'),
                    '--output', str(root / 'scores'), '--judge-model', 'fixture', '--key-env', 'TEST_JUDGE_KEY',
                    '--env-file', str(root / 'absent.env'), '--answer-content-only', '--allow-partial']
            response = {'choices': [{'finish_reason': 'stop', 'message': {
                'content': json.dumps({'correct': True, 'reason': 'Matches.'})}}]}
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'TEST_JUDGE_KEY': 'fake'}), \
                 patch('evaluation.score.APIClient.post', return_value=response) as post:
                main()
            self.assertEqual(json.loads(post.call_args.args[0]['messages'][1]['content'])['candidate'], '42')
            summary = json.loads((root / 'scores/summary.json').read_text())
            self.assertEqual(summary['micro_accuracy_pct'], 100)
            self.assertEqual(summary['scoring_scope']['missing_count'], 1)
            self.assertEqual(summary['scoring_scope']['excluded_infrastructure_count'], 1)
            self.assertTrue(summary['scoring_scope']['provisional'])
            self.assertEqual((root / 'predictions.jsonl').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
