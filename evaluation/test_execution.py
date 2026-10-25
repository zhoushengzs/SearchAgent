"""No-network tests for same-model observations, content recovery and retries."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from evaluation.execution import execute_with_retries, final_content, observe_result, execution_policy
from evaluation.remote import APIClient, RemoteFatalError
from evaluation.run import run_task, tool_definitions
from evaluation.test_validation_retry import SequenceModel, call


def observation(**updates):
    return {'visual_facts': [], 'language_facts': [], 'reasoning': 'Only the supplied result is considered.',
            'hypothesis': None, 'revision_of_step': None, 'open_questions': [], **updates}


class ObservationTests(unittest.TestCase):
    def evaluate(self, replies, result, summaries=True):
        model = SequenceModel(replies)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), image_side=512, mode='agent',
                                   max_steps=3, max_pixels=262144, max_new_tokens=128,
                                   context_tokens=4096, protocol='training', tool_summaries=summaries)
            if summaries is None:
                del args.tool_summaries
            task = {'id': 'fixture', 'source_dataset': 'fixture', 'input_fingerprint': 'fixture',
                    'model_input': {'question': 'Use only this question.', 'images': []}}
            with patch('agent.tools.EvidenceTools.execute', side_effect=result), patch('evaluation.execution.time.sleep'):
                record = run_task(task, model, args, tool_definitions(), 'fixture')
        return record, model

    def test_default_reads_raw_results_without_an_extra_model_call(self):
        result = {'tool': 'web_search', 'status': 'ok',
                  'data': {'results': [{'snippet': 'The page states 42.'}]}}
        with patch('evaluation.run.observe_result') as observer:
            record, model = self.evaluate([
                call('vs_web_search', {'query': 'q'}), '<response>42</response>'],
                [result], summaries=None)
        observer.assert_not_called()
        self.assertEqual(len(model.messages), 2)
        self.assertEqual(record['answer'], '42')
        self.assertEqual(record['observations'], [])
        feedback = next(m for m in model.messages[-1] if m['role'] == 'tool')
        payload = json.loads(feedback['content'])
        self.assertEqual(payload['result'], result)
        self.assertIsNone(payload['observation'])
        self.assertFalse(execution_policy()['tool_summaries'])
        self.assertIsNone(execution_policy()['observer'])
        self.assertEqual(execution_policy()['observer_attempts'], 0)

    def test_summary_and_raw_evidence_reach_next_decision_with_matching_tool_envelope(self):
        summary = observation(language_facts=['The page states 42.'])
        result = {'tool': 'web_search', 'status': 'ok', 'data': {'results': [{'snippet': 'The page states 42.'}]}}
        record, model = self.evaluate([
            call('vs_web_search', {'query': 'question'}), json.dumps(summary), '<response>42</response>'], [result])
        self.assertEqual(record['answer'], '42')
        self.assertEqual(record['observations'][0]['observation'], summary)
        self.assertEqual(len(record['turns']), 2)  # Observer calls are a separate audit ledger.
        self.assertEqual(len(model.messages), 3)
        feedback = next(m for m in model.messages[-1] if m['role'] == 'tool')
        self.assertEqual(json.loads(feedback['content'])['result'], result)
        self.assertEqual(json.loads(feedback['content'])['observation'], summary)

    def test_invalid_summary_does_not_discard_raw_evidence_or_stop_answering(self):
        result = {'tool': 'web_search', 'status': 'ok', 'data': {'results': []}}
        record, model = self.evaluate([
            call('vs_web_search', {'query': 'q'}), 'bad', 'bad', 'bad', '<response>uncertain</response>'], [result])
        self.assertEqual(record['status'], 'answered')
        self.assertEqual(record['observations'][0]['status'], 'invalid_output')
        feedback = next(m for m in model.messages[-1] if m['role'] == 'tool')
        self.assertIsNone(json.loads(feedback['content'])['observation'])
        self.assertFalse(record['infrastructure_error'])

    def test_recovered_network_error_is_logged_without_marking_whole_task_failed(self):
        failed = {'tool': 'web_search', 'status': 'error', 'error_type': 'ConnectTimeout'}
        ok = {'tool': 'web_search', 'status': 'ok', 'data': {'results': []}}
        record, _ = self.evaluate([call('vs_web_search', {'query': 'q'}), '<response>42</response>'],
                                  [failed, ok], summaries=False)
        self.assertEqual(record['status'], 'answered')
        self.assertFalse(record['infrastructure_error'])
        self.assertEqual(len(record['turns'][0]['execution_attempts']), 2)
        self.assertEqual(record['tool_calls'], 1)

    def test_persistent_network_errors_keep_answer_but_flag_incomplete_run(self):
        failed = {'tool': 'web_search', 'status': 'error', 'error_type': 'ConnectTimeout'}
        record, _ = self.evaluate([call('vs_web_search', {'query': 'q'}), '<response>uncertain</response>'],
                                  [failed] * 3, summaries=False)
        self.assertEqual(record['status'], 'answered')
        self.assertTrue(record['infrastructure_error'])
        self.assertEqual(len(record['turns'][0]['execution_attempts']), 3)

    def test_search_credit_exhaustion_is_flagged_without_repeated_requests(self):
        failed = {'tool': 'web_search', 'status': 'error', 'error_type': 'SerperKeysExhausted'}
        record, _ = self.evaluate([call('vs_web_search', {'query': 'q'}), '<response>uncertain</response>'],
                                  [failed], summaries=False)
        self.assertTrue(record['infrastructure_error'])
        self.assertEqual(len(record['turns'][0]['execution_attempts']), 1)

    def test_observer_receives_executed_pixels_and_cannot_claim_pixels_for_text_only_result(self):
        summary = observation(visual_facts=['A red region is visible.'])
        captured = []
        def generate(messages):
            captured.append(messages)
            return {'status': 'ok', 'text': json.dumps(summary)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crop.png'
            Image.new('RGB', (32, 32), 'red').save(path)
            result = {'tool': 'crop', 'status': 'ok', 'data': {'view': {'path': str(path), 'sha256': 'fixture'}}}
            value, ledger = observe_result('q', [], {'tool': 'crop'}, result, generate, 262144)
            self.assertEqual(value, summary)
            self.assertEqual(captured[0][1]['content'][0]['image'], str(path))
            self.assertEqual(ledger['observer'], 'same_tested_model')
        value, ledger = observe_result('q', [], {'tool': 'web_search'},
                                      {'tool': 'web_search', 'status': 'ok', 'data': {}}, generate, 262144)
        self.assertIsNone(value)
        self.assertEqual(len(ledger['attempts']), 3)


class RetryTests(unittest.TestCase):
    def test_transient_http_retry_keeps_distinct_attempt_artifact_steps(self):
        seen = []
        def execute(name, args, step):
            seen.append(step)
            return {'status': 'error', 'error_type': 'HTTPError', 'http_status': 503}
        with patch('evaluation.execution.time.sleep') as sleep:
            _, attempts = execute_with_retries(SimpleNamespace(execute=execute), 'visit', {}, 2)
        self.assertEqual(seen, [2, '2_retry_2', '2_retry_3'])
        self.assertEqual(len(attempts), 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])

    def test_permanent_errors_are_not_retried(self):
        for result in [{'status': 'error', 'error_type': 'HTTPError', 'http_status': 403},
                       {'status': 'error', 'error_type': 'InvalidArguments'},
                       {'status': 'error', 'error_type': 'SerperKeysExhausted'}]:
            with self.subTest(result=result), patch('evaluation.execution.time.sleep') as sleep:
                _, attempts = execute_with_retries(SimpleNamespace(execute=lambda *args: result), 'visit', {}, 1)
                self.assertEqual(len(attempts), 1)
                sleep.assert_not_called()

    def test_model_api_connection_failure_already_retries_and_fatal_credit_error_does_not(self):
        import requests
        client = APIClient('fixture')
        body = {'choices': [{'message': {'content': 'answer'}}]}
        response = SimpleNamespace(status_code=200, json=lambda: body)
        with patch('evaluation.remote.requests.post', side_effect=[requests.ConnectionError(), response]) as post, \
                patch.object(client, '_pause'):
            self.assertEqual(client.post({}), body)
            self.assertEqual(post.call_count, 2)
        response = SimpleNamespace(status_code=402, json=lambda: {})
        with patch('evaluation.remote.requests.post', return_value=response) as post:
            with self.assertRaises(RemoteFatalError):
                client.post({})
            self.assertEqual(post.call_count, 1)


class FinalContentTests(unittest.TestCase):
    def test_final_text_is_preserved_but_actions_and_truncated_thought_are_not(self):
        self.assertEqual(final_content('Explanation <response>42</response>'), '42')
        self.assertEqual(final_content('42', allow_plain=True), '42')
        for text in ['<think>42', '<response>42', call('x', {}),
                     '{"name":"vs_web_search","arguments":{"query":"42"}}']:
            self.assertEqual(final_content(text, allow_plain=True), '')


if __name__ == '__main__':
    unittest.main()
