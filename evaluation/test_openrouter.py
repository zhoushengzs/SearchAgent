import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from evaluation.compare import cross_backend_differences
from evaluation.remote import (APIClient, OpenRouterModel, RemoteFatalError, RemoteAPIError,
                               encode_image, parallel_results)
from evaluation.run import fingerprint, parse_output, run_task, tool_definitions
from evaluation.score import recover_scores, score_all
from evaluation.test_parallel import task


class TransportTests(unittest.TestCase):
    def model(self, bodies):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = list(range(20))
        client = Mock()
        client.post.side_effect = bodies
        return OpenRouterModel('qwen/qwen3-vl-8b-instruct', tokenizer, client)

    def body(self, message, finish='stop', prompt=20):
        return {'id': 'request-1', 'model': 'qwen/qwen3-vl-8b-instruct', 'provider': 'fixture',
                'usage': {'prompt_tokens': prompt, 'completion_tokens': 8},
                'choices': [{'message': message, 'finish_reason': finish}]}

    def test_remote_tool_call_executes_real_calculator_and_returns_same_schema(self):
        first = self.body({'tool_calls': [{'function': {'name': 'vs_calculate', 'arguments': '{"expression":"1+2"}'}}]}, 'tool_calls')
        model = self.model([first, self.body({'content': '<response>3</response>'})])
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(tool_summaries=False, output=Path(folder), image_side=1536, mode='agent', max_steps=30,
                                   max_pixels=262144, max_new_tokens=4096, context_tokens=32768)
            result = run_task(task(1), model, args, tool_definitions(), 'test-run')
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['answer'], '3')
        self.assertEqual(result['tool_calls'], 1)
        self.assertEqual(result['turns'][0]['result']['status'], 'ok')
        wire = model.client.post.call_args.args[0]
        call = wire['messages'][2]['tool_calls'][0]
        self.assertEqual(wire['messages'][3]['tool_call_id'], call['id'])
        self.assertEqual(wire['max_tokens'], 4096)
        self.assertEqual(wire['temperature'], 0)
        self.assertNotIn('parallel_tool_calls', wire)
        self.assertTrue(wire['provider']['require_parameters'])
        self.assertEqual(model.client.post.call_count, 2)  # No extra observer call.

    def test_length_cannot_be_scored_as_answer(self):
        model = self.model([self.body({'content': '<response>3</response>'}, 'length')])
        self.assertEqual(model.generate([], [], 64, 8192)['status'], 'length')

    def test_context_guard_makes_no_api_call_and_actual_usage_is_checked(self):
        model = self.model([])
        self.assertEqual(model.generate([], [], 100, 110)['status'], 'context_limit')
        model.client.post.assert_not_called()
        model = self.model([self.body({'content': 'answer'}, prompt=8000)])
        self.assertEqual(model.generate([], [], 1024, 8192)['status'], 'context_limit')

    def test_multiple_native_actions_are_not_executed(self):
        call = {'function': {'name': 'vs_calculate', 'arguments': '{"expression":"1+2"}'}}
        model = self.model([self.body({'tool_calls': [call, call]}, 'tool_calls')])
        self.assertEqual(model.generate([], [], 4096, 32768)['status'], 'invalid_output')

    def test_transmitted_image_respects_budget_and_original_is_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'original.png'
            Image.new('RGB', (1536, 1024), 'red').save(path)
            original = path.read_bytes()
            encoded, size = encode_image(path, 262144)
            self.assertLessEqual(size[0] * size[1], 262144)
            self.assertEqual(Image.open(io.BytesIO(base64.b64decode(encoded.split(',', 1)[1]))).size, size)
            self.assertEqual(path.read_bytes(), original)

    def test_native_arguments_inside_thinking_remain_non_executable(self):
        model = self.model([self.body({'content': '<think>Example <tool_call>x</tool_call></think><response>3</response>'})])
        result = model.generate([], [], 4096, 32768)
        self.assertEqual(parse_output(result['text']), ('answer', '3'))


class ConcurrencyTests(unittest.TestCase):
    def test_bounded_parallelism_and_no_missing_or_duplicate_results(self):
        lock = threading.Lock()
        active = peak = 0
        def work(index):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.005)
            with lock:
                active -= 1
            return index
        result = list(parallel_results(work, range(25), 4))
        self.assertEqual(sorted(result), list(range(25)))
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 4)

    def test_429_retries_and_credit_failure_stops(self):
        limited = Mock(status_code=429, headers={'Retry-After': '2'})
        limited.json.return_value = {'error': {'code': 429}}
        success = Mock(status_code=200)
        success.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}
        client = APIClient('not-a-real-key')
        with patch('evaluation.remote.requests.post', side_effect=[limited, success]) as post, patch.object(client, '_pause') as pause:
            client.post({})
            self.assertEqual(post.call_count, 2)
            pause.assert_called_once_with(limited, 0)
        denied = Mock(status_code=402)
        denied.json.return_value = {'error': {'code': 402, 'message': 'private-provider-details'}}
        with patch('evaluation.remote.requests.post', return_value=denied) as post:
            with self.assertRaises(RemoteFatalError) as error:
                client.post({})
            self.assertNotIn('private-provider-details', str(error.exception))
            self.assertTrue(client.stop.is_set())
            self.assertEqual(post.call_count, 1)

    def test_incompatible_endpoint_prints_safe_actionable_error(self):
        import contextlib
        denied = Mock(status_code=404)
        denied.json.return_value = {'error': {'code': 404, 'message': 'private-provider-details'}}
        client = APIClient('not-a-real-key')
        output = io.StringIO()
        with patch('evaluation.remote.requests.post', return_value=denied), contextlib.redirect_stderr(output):
            with self.assertRaises(RemoteFatalError):
                client.post({})
        self.assertIn('Remote HTTP 404', output.getvalue())
        self.assertIn('No compatible endpoint', output.getvalue())
        self.assertNotIn('private-provider-details', output.getvalue())
        self.assertNotIn('not-a-real-key', output.getvalue())

    def test_retry_after_is_honored(self):
        client = APIClient('fixture')
        response = Mock(headers={'Retry-After': '123'})
        with patch('evaluation.remote.time.monotonic', return_value=100):
            client._pause(response, 0)
        self.assertGreaterEqual(client.cooldown, 223)


class ScoringTests(unittest.TestCase):
    def test_official_deepseek_judge_uses_explicit_thinking_and_output_budget(self):
        body = {'model': 'deepseek-flash', 'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"correct":true,"reason":"equivalent"}'}}]}
        with tempfile.TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            args = fixture[0]
            args.base_url = 'https://api.deepseek.com'
            args.judge_model = 'deepseek-flash'
            args.judge_thinking = 'disabled'
            args.judge_max_tokens = 1024
            with patch.object(APIClient, 'post', return_value=body) as post:
                score_all(*fixture)
            self.assertEqual(post.call_count, 4)
            for call in post.call_args_list:
                payload = call.args[0]
                self.assertEqual(payload['model'], 'deepseek-flash')
                self.assertEqual(payload['thinking'], {'type': 'disabled'})
                self.assertEqual(payload['max_tokens'], 1024)
                self.assertEqual(payload['response_format'], {'type': 'json_object'})

    def fixture(self, folder):
        args = SimpleNamespace(tool_summaries=False, output=Path(folder), base_url='https://example.invalid/v1', judge_model='fixture', workers=3)
        tasks = {f'task_{i}': task(i) for i in range(4)}
        predictions = {key: {'id': key, 'source_dataset': 'fixture', 'input_fingerprint': t['input_fingerprint'],
                            'status': 'answered', 'answer': '3', 'tool_calls': 1, 'elapsed_seconds': 1}
                       for key, t in tasks.items()}
        references = {(key, t['input_fingerprint']): {'answer': '3'} for key, t in tasks.items()}
        config = {'policy_id': 'policy', 'run_id': 'fixture'}
        run = {'run_id': 'fixture', 'protocol': {'selected_ids': list(tasks)}}
        return args, run, tasks, predictions, references, config, 'fixture-key'

    def test_parallel_scoring_resumes_without_requests_and_keeps_denominator(self):
        body = {'model': 'judge', 'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"correct":true,"reason":"equal","id":"injected","score":0}'}}]}
        with tempfile.TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            with patch.object(APIClient, 'post', return_value=body) as post:
                score_all(*fixture)
                self.assertEqual(post.call_count, 4)
                score_all(*fixture)
                self.assertEqual(post.call_count, 4)
            summary = json.loads((Path(folder) / 'summary.json').read_text())
            self.assertEqual(summary['count'], 4)
            self.assertEqual(summary['micro_accuracy_pct'], 100)
            self.assertEqual({r['id'] for r in recover_scores(Path(folder) / 'scores.jsonl')}, set(fixture[2]))

    def test_judge_failure_does_not_become_zero_score_or_final_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(APIClient, 'post', side_effect=RemoteAPIError('fixture')):
                with self.assertRaises(SystemExit):
                    score_all(*self.fixture(folder))
            self.assertFalse((Path(folder) / 'summary.json').exists())
            self.assertEqual((Path(folder) / 'scores.jsonl').read_text(), '')

    def test_interrupted_score_tail_is_recovered(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scores.jsonl'
            path.write_bytes(b'{"id":"ok"}\n{"id":')
            self.assertEqual(recover_scores(path), [{'id': 'ok'}])
            self.assertEqual(path.read_bytes(), b'{"id":"ok"}\n')


class ComparisonTests(unittest.TestCase):
    def test_cross_backend_does_not_allow_changed_budget_or_prompt(self):
        local = {'system': 'same', 'max_new_tokens': 4096, 'device': 'mps', 'selected_ids': ['1']}
        remote = {**local, 'device': 'openrouter', 'api': {'provider': 'fixture'}}
        def wrap(protocol):
            return {'protocol': protocol, 'protocol_sha256': fingerprint(protocol)}
        self.assertIn('device', cross_backend_differences(wrap(remote), wrap(local)))
        for key, value in [('max_new_tokens', 1024), ('system', 'different'), ('selected_ids', ['2'])]:
            changed = {**remote, key: value}
            with self.assertRaises(ValueError):
                cross_backend_differences(wrap(changed), wrap(local))


class RunnerTests(unittest.TestCase):
    def test_resume_retries_only_infrastructure_failures_and_workers_do_not_change_run(self):
        from evaluation.run_openrouter import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'tokenizer').mkdir()
            (root / 'tokenizer/config.json').write_text('{"model_type":"qwen3_vl"}')
            (root / 'tasks.jsonl').write_text(''.join(json.dumps(task(i)) + '\n' for i in range(3)))
            argv = ['remote', '--tokenizer', str(root / 'tokenizer'), '--tasks', str(root / 'tasks.jsonl'),
                    '--expected-count', '3', '--output', str(root / 'run'), '--workers', '2', '--mode', 'direct',
                    '--env-file', str(root / 'absent.env')]
            seen = []
            def evaluate(t, model, args, schemas, run_id):
                failed = t['id'] == 'task_1' and t['id'] not in seen
                seen.append(t['id'])
                return {'id': t['id'], 'input_fingerprint': t['input_fingerprint'], 'source_dataset': 'fixture',
                        'run_id': run_id, 'infrastructure_error': failed, 'status': 'inference_error' if failed else 'answered',
                        'answer': '3', 'tool_calls': 0, 'elapsed_seconds': 1}
            fake_transformers = SimpleNamespace(AutoTokenizer=Mock())
            with patch.dict(sys.modules, {'transformers': fake_transformers}), patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}), \
                 patch('evaluation.run_openrouter.importlib.metadata.version', return_value='fixture'), \
                 patch('evaluation.run_openrouter.run_task', side_effect=evaluate), patch.object(APIClient, 'post') as post:
                with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                    main()
                self.assertEqual(len(seen), 3)
                with patch.object(sys, 'argv', argv + ['--workers', '3', '--retry-infrastructure-errors']):
                    main()
                self.assertEqual(seen.count('task_1'), 2)
                self.assertEqual(len(seen), 4)
                with patch.object(sys, 'argv', argv):
                    main()
                self.assertEqual(len(seen), 4)
                post.assert_not_called()
            completion = json.loads((root / 'run/completion.json').read_text())
            self.assertTrue(completion['complete'])
            self.assertEqual(completion['recorded'], 3)


if __name__ == '__main__':
    unittest.main()
