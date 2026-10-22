"""Offline integration tests: no API credentials or model inference required."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import browsecomp_v3_official as official
from evaluation.common import task_fingerprint
from evaluation.prepare import digest, write_rows
from evaluation.remote import RemoteAPIError


def fixture(index=0):
    task = {'id': f'browsecomp_v3_train_{index}', 'source_dataset': 'browsecomp_v3',
            'model_input': {'question': 'Which city?', 'images': []},
            'dataset_provenance': {'revision': 'fixture'}, 'categories': {}}
    task['input_fingerprint'] = task_fingerprint(task)
    ref = {'id': task['id'], 'input_fingerprint': task['input_fingerprint'], 'answer': 'Paris',
           'annotations': {'metadata': {'domain': {'d1': 'Travel'}}, 'sub_goals': [
               {'sg_id': 1, 'description': 'Find city', 'key_info': 'Paris'},
               {'sg_id': 2, 'description': 'Find date', 'key_info': '1900'}]}}
    prediction = {'id': task['id'], 'source_dataset': 'browsecomp_v3',
                  'input_fingerprint': task['input_fingerprint'], 'run_id': 'fixture-run',
                  'status': 'answered', 'answer': 'Paris', 'infrastructure_error': False,
                  'turns': [{'text': '<SUBGOAL>The city is Paris.</SUBGOAL>'}],
                  'tool_results': [{'content': 'SECRET_TOOL_RESULT'}]}
    return task, ref, prediction


def verdict(vector=(1, 0), correct=True, anomaly=False):
    return {'answer_correct': correct, 'process_score': sum(vector) / len(vector),
            'subgoal_vector': list(vector), 'subgoal_details': [
                {'subgoal_index': i, 'status': 'ACHIEVED' if value else 'NOT_ACHIEVED',
                 'reason': 'fixture evidence'} for i, value in enumerate(vector, 1)],
            'anomaly_flag': anomaly, 'anomaly_reason': 'None', 'confidence': 'HIGH',
            'judge_reasoning': 'fixture evaluation'}


def arguments(**overrides):
    values = dict(judge_model='fixture-judge', base_url='https://example.invalid/v1',
                  judge_thinking='provider-default', judge_max_tokens=4096,
                  response_format='json_schema', workers=1)
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeClient:
    calls = []
    fail = False

    def __init__(self, *args):
        self.stop = threading.Event()

    def post(self, payload):
        type(self).calls.append(payload)
        if type(self).fail:
            raise RemoteAPIError('fixture network failure')
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(verdict())}}]}


class OfficialIntegrationTests(unittest.TestCase):
    def setUp(self):
        FakeClient.calls, FakeClient.fail = [], False

    def sample(self):
        return official.build_sample(*fixture(), 'fixture-model')

    def export_fixture(self, root):
        task, ref, prediction = fixture()
        run_dir = root / 'run'
        run_dir.mkdir()
        write_rows(root / 'tasks.jsonl', [task])
        write_rows(root / 'references.jsonl', [ref])
        write_rows(run_dir / 'predictions.jsonl', [prediction])
        (run_dir / 'run.json').write_text(json.dumps({
            'run_id': 'fixture-run', 'model': 'fixture-model',
            'protocol': {'tasks_sha256': digest(root / 'tasks.jsonl'),
                         'selected_ids': [task['id']], 'max_steps': 10}}))
        return run_dir, root / 'tasks.jsonl', root / 'references.jsonl', root / 'export'

    def test_extracts_original_assistant_text_without_teacher_or_tool_evidence(self):
        sample = self.sample()
        self.assertIn('The city is Paris.', sample['prediction']['subgoals'])
        self.assertNotIn('1900', str(sample['prediction']))
        self.assertNotIn('SECRET_TOOL_RESULT', str(sample))
        task, ref, pred = fixture()
        pred['turns'] = [{'text': 'I am searching for evidence.'}]
        sample = official.build_sample(task, ref, pred, 'fixture-model')
        self.assertEqual(sample['prediction']['subgoals'], [])

    def test_nonanswer_stays_empty_but_keeps_process(self):
        task, ref, pred = fixture()
        pred['status'] = 'step_limit'
        sample = official.build_sample(task, ref, pred, 'fixture-model')
        self.assertEqual(sample['prediction']['final_answer'], '')
        self.assertTrue(sample['prediction']['subgoals'])
        with self.assertRaises(ValueError):
            official.validate_verdict(verdict(), sample)
        official.validate_verdict(verdict(correct=False), sample)

    def test_export_is_repeatable_and_rejects_changed_samples_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self.export_fixture(Path(folder))
            samples, manifest = official.prepare_export(*inputs, expected_count=1)
            self.assertEqual(manifest['inference_calls'], 0)
            self.assertEqual(official.prepare_export(*inputs, expected_count=1), (samples, manifest))
            (inputs[-1] / 'samples/0.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'sample was changed'):
                official.prepare_export(*inputs, expected_count=1)
        task, ref, pred = fixture()
        ref['annotations']['sub_goals'] = []
        with self.assertRaisesRegex(ValueError, 'missing official'):
            official.build_sample(task, ref, pred, 'fixture-model')

    def test_missing_prediction_infrastructure_failure_and_wrong_reference_rejected(self):
        for failure in ('missing', 'infrastructure', 'reference'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                inputs = self.export_fixture(Path(folder))
                if failure == 'missing':
                    write_rows(inputs[0] / 'predictions.jsonl', [])
                elif failure == 'infrastructure':
                    pred = fixture()[2]
                    pred['infrastructure_error'] = True
                    write_rows(inputs[0] / 'predictions.jsonl', [pred])
                else:
                    ref = fixture()[1]
                    ref['input_fingerprint'] = 'wrong'
                    write_rows(inputs[2], [ref])
                with self.assertRaises(ValueError):
                    official.prepare_export(*inputs, expected_count=1)

    def test_anomaly_does_not_override_correctness_and_ps_is_per_question(self):
        sample = self.sample()
        v = official.validate_verdict(verdict(anomaly=True), sample)
        records = [dict(score={'final': 1, 'process': .5}, metadata={'verdict': v}),
                   dict(score={'final': 0, 'process': .25}, metadata={'verdict': verdict(correct=False)})]
        summary = official.summarize(records)
        self.assertEqual(summary['sr_pct'], 50)
        self.assertEqual(summary['ps_pct'], 37.5)
        self.assertEqual(summary['anomaly_count'], 1)

    def test_rejects_inconsistent_judge_verdicts(self):
        for key, value in [('process_score', .8), ('subgoal_vector', [1]),
                           ('answer_correct', 'true'), ('subgoal_vector', [True, 0]),
                           ('confidence', 'unknown')]:
            with self.subTest(key=key, value=value):
                bad = verdict()
                bad[key] = value
                with self.assertRaises(ValueError):
                    official.validate_verdict(bad, self.sample())

    def test_official_prompt_schema_and_compatible_json_mode(self):
        sample = self.sample()
        record = official.evaluate_sample(sample, FakeClient(), arguments(), 'policy')
        self.assertEqual(record['score'], {'final': 1, 'process': .5})
        request = FakeClient.calls[-1]
        self.assertEqual(request['response_format'], official.RESPONSE_FORMAT)
        self.assertNotIn('temperature', request)
        self.assertIn('SG1: Find city | key_info: Paris', request['messages'][0]['content'])
        official.evaluate_sample(sample, FakeClient(), arguments(response_format='json_object', judge_thinking='disabled'), 'policy')
        self.assertEqual(FakeClient.calls[-1]['thinking'], {'type': 'disabled'})
        self.assertEqual(FakeClient.calls[-1]['response_format'], {'type': 'json_object'})
        self.assertIn(json.dumps(official.RESPONSE_FORMAT['json_schema']['schema'], ensure_ascii=False),
                      FakeClient.calls[-1]['messages'][0]['content'])

    def test_resume_retains_scores_and_failed_calls_are_not_zero(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(official, 'APIClient', FakeClient), contextlib.redirect_stdout(io.StringIO()):
            inputs = self.export_fixture(Path(folder))
            samples, manifest = official.prepare_export(*inputs, expected_count=1)
            FakeClient.fail = True
            with self.assertRaises(SystemExit):
                official.score_export(samples, manifest, inputs[-1], arguments(), 'fake-key')
            self.assertFalse((inputs[-1] / 'summary.json').exists())
            self.assertEqual((inputs[-1] / 'eval.jsonl').read_text(), '')
            FakeClient.fail = False
            official.score_export(samples, manifest, inputs[-1], arguments(), 'fake-key')
            count = len(FakeClient.calls)
            # Recover a partial write while retaining the valid, paid judgment.
            with (inputs[-1] / 'eval.jsonl').open('ab') as stream:
                stream.write(b'{"unfinished":')
            official.score_export(samples, manifest, inputs[-1], arguments(), 'fake-key')
            self.assertEqual(len(FakeClient.calls), count)
            self.assertTrue(list(inputs[-1].glob('eval.interrupted-tail.*')))
            self.assertEqual(json.loads((inputs[-1] / 'summary.json').read_text())['count'], 1)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                official.score_export(samples, manifest, inputs[-1], arguments(judge_model='different'), 'fake-key')


if __name__ == '__main__':
    unittest.main()
