import importlib.util
import ast
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'vision_deepresearch_async_workflow/tools/serper_keys.py'
spec = importlib.util.spec_from_file_location('serper_keys_tested', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def worker_exhaust(path):
    m.SerperKeyPool(['dummy-a', 'dummy-b'], path).exhausted(0)


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'state.json'
        self.env = patch.dict(os.environ, {'SERPER_API_KEYS': '["dummy-a","dummy-b"]',
                                          'SERPER_KEY_STATE_FILE': str(self.path)}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def request(self, responses, url='https://google.serper.dev/search', **kw):
        used = []
        replies = iter(responses)
        def post(url, headers, **kwargs):
            used.append(headers.copy())
            return next(replies)
        fake = types.SimpleNamespace(post=post)
        with patch.dict('sys.modules', {'requests': fake}), patch.object(m.time, 'sleep'):
            response = m.serper_post(url, **kw)
        return response, used

    def test_order_and_deduplication(self):
        self.assertEqual(m.configured_keys({'SERPER_API_KEYS': '["a","b","a"]'}), ('a','b'))
        self.assertEqual(m.configured_keys({'SERP_API_KEY':'a', 'SERPER_API_KEY_10':'c',
                                           'SERPER_API_KEY_2':'b'}), ('a','b','c'))

    def test_explicit_list_does_not_add_legacy_credentials(self):
        self.assertEqual(m.configured_keys({'SERPER_API_KEYS':'["chosen"]',
                                           'SERPER_API_KEY':'old'}), ('chosen',))

    def test_quota_rotates_and_persists(self):
        ok = types.SimpleNamespace(status_code=200, text='{}')
        quota = types.SimpleNamespace(status_code=403, text='Not enough credits')
        out, used = self.request([quota, ok])
        self.assertIs(out, ok)
        self.assertEqual([h['X-API-KEY'] for h in used], ['dummy-a','dummy-b'])
        self.assertEqual(m.SerperKeyPool(['dummy-a','dummy-b'], self.path).current()[0], 1)

    def test_rate_limit_retries_same_key(self):
        rate = types.SimpleNamespace(status_code=429, text='rate limit quota exceeded per minute')
        ok = types.SimpleNamespace(status_code=200, text='{}')
        _, used = self.request([rate, ok])
        self.assertEqual([h['X-API-KEY'] for h in used], ['dummy-a','dummy-a'])
        self.assertEqual(m.key_pool().current()[0], 0)

    def test_authentication_failure_does_not_consume_another_key(self):
        auth = types.SimpleNamespace(status_code=401, text='Invalid API key')
        _, used = self.request([auth])
        self.assertEqual(len(used), 1)
        self.assertEqual(m.key_pool().current()[0], 0)

    def test_all_exhausted_bypasses_normal_exception_handlers(self):
        quota = types.SimpleNamespace(status_code=403, text='Not enough credits')
        with self.assertRaises(m.SerperKeysExhausted):
            self.request([quota, quota])
        self.assertFalse(issubclass(m.SerperKeysExhausted, Exception))
        self.assertNotIn('dummy-', self.path.read_text())

    def test_workers_do_not_skip_unspent_key_on_stale_failure(self):
        ctx = multiprocessing.get_context('fork')
        processes = [ctx.Process(target=worker_exhaust, args=(str(self.path),)) for _ in range(4)]
        for p in processes:p.start()
        for p in processes:
            p.join(10)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(m.key_pool().current()[0], 1)

    def test_direct_keys_never_go_to_other_hosts(self):
        with self.assertRaises(ValueError):self.request([], url='https://example.com/search')

    def test_gateway_headers_are_preserved(self):
        ok = types.SimpleNamespace(status_code=200, text='{}')
        _, used = self.request([ok], url='https://example.com/search',
                               headers={'Authorization': 'Bearer dummy-gateway'})
        self.assertNotIn('X-API-KEY', used[0])

    def test_corrupted_state_does_not_silently_restart_spent_keys(self):
        self.path.write_text('{')
        with self.assertRaises(json.JSONDecodeError):m.key_pool().current()

    def test_judge_configuration_does_not_enable_crop_search_proxy(self):
        source = MODULE.with_name('crop_and_search_tool.py').read_text()
        original = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.ClassDef) and n.name == 'CropAndSearchTool')
        constructor = next(n for n in original.body
                           if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        cls = ast.ClassDef(name='CropAndSearchTool', bases=[ast.Name(id='Base', ctx=ast.Load())],
                           keywords=[], body=[constructor], decorator_list=[])
        tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        class Base:
            def __init__(self, **kwargs):pass
        namespace = {'Base':Base, 'os':os, 'PIL_AVAILABLE':True,
                     'configured_keys':m.configured_keys}
        with patch.dict(os.environ, {'JUDGE_API_KEY':'dummy-judge',
                                    'API_GATEWAY_USER':'dummy-user',
                                    'API_GATEWAY_KEY':'dummy-proxy'}):
            exec(compile(tree, '<crop-constructor>', 'exec'), namespace)
            tool = namespace['CropAndSearchTool']()
        self.assertFalse(tool._use_gateway)
        self.assertEqual(tool.serp_api_key, 'dummy-a')


if __name__ == '__main__':
    unittest.main(verbosity=2)
