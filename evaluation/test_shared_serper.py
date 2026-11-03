"""No-network checks for opting evaluation into the training Serper pool.

Set TRAINING_SERPER_MODULE_FILE to also exercise the real training pool against
synthetic credentials and temporary state; never use the production state here.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from agent import tools


def response(status=200, body=None):
    value = Mock()
    value.status_code = status
    value.text = json.dumps(body or {})
    value.json.return_value = body or {'organic': [], 'images': []}
    return value


class LegacyPoolTests(unittest.TestCase):
    def test_default_uses_original_single_key_request(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'SERPER_API_KEY': 'synthetic-legacy'}, clear=True), patch.object(tools, '_SERPER_POOL', None), patch('requests.post', return_value=response()) as post:
            self.assertTrue(tools.serper_configured())
            instance = tools.EvidenceTools(Path(directory))
            instance._search('test query', False)
            self.assertEqual(post.call_args.kwargs['headers'], {'X-API-KEY': 'synthetic-legacy'})
            self.assertIsNone(instance.shared_serper)

    def test_json_list_does_not_silently_change_default_backend(self):
        with patch.dict(os.environ, {'SERPER_API_KEYS': '["synthetic"]'}, clear=True):
            self.assertFalse(tools.serper_configured())

    def test_shared_backend_requires_explicit_absolute_state(self):
        for state in ['', 'relative/state.json']:
            with self.subTest(state=state), patch.dict(os.environ, {'VS_SERPER_BACKEND': 'training', 'SERPER_KEY_STATE_FILE': state}, clear=True):
                with self.assertRaisesRegex(ValueError, 'absolute SERPER_KEY_STATE_FILE'):
                    tools.serper_configured()


@unittest.skipUnless(os.environ.get('TRAINING_SERPER_MODULE_FILE'), 'Set TRAINING_SERPER_MODULE_FILE for real-pool simulation')
class SharedTrainingPoolTests(unittest.TestCase):
    def setUp(self):
        source = Path(os.environ['TRAINING_SERPER_MODULE_FILE'])
        spec = importlib.util.spec_from_file_location('simulated_training_serper', source)
        self.backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.backend)
        package = types.ModuleType('project_runtime')
        package.serper_keys = self.backend
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state = self.root / 'state.json'
        self.keys = [f'synthetic-key-{index}' for index in range(212)]
        self.state.write_text(json.dumps({'pool': hashlib.sha256(json.dumps(self.keys[:25]).encode()).hexdigest(), 'index': 22, 'key_count': 25}))
        self.environment = {'VS_SERPER_BACKEND': 'training', 'SERPER_API_KEYS': json.dumps(self.keys), 'SERPER_KEY_STATE_FILE': str(self.state)}
        self.env_patch = patch.dict(os.environ, self.environment, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.module_patch = patch.dict(sys.modules, {'project_runtime': package})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.network = patch('requests.post', side_effect=AssertionError('Unexpected real request'))
        self.post = self.network.start()
        self.addCleanup(self.network.stop)

    def test_resumes_shared_index_rotates_quota_and_observes_training_advance(self):
        self.assertTrue(tools.serper_configured())
        instance = tools.EvidenceTools(self.root / 'eval')
        self.post.side_effect = [response(403, {'message': 'Not enough credits'}), response()]
        instance._search('test query', False)
        self.assertEqual([call.kwargs['headers']['X-API-KEY'] for call in self.post.call_args_list], self.keys[22:24])
        state = json.loads(self.state.read_text())
        self.assertEqual((state['index'], state['key_count']), (23, 212))
        # Simulate a separate training worker moving the same state forward.
        training_worker = self.backend.SerperKeyPool(self.keys, self.state)
        training_worker.exhausted(23)
        self.post.reset_mock()
        self.post.side_effect = None
        self.post.return_value = response()
        instance._search('image test', True)
        self.assertEqual(self.post.call_args.args[0], 'https://google.serper.dev/images')
        self.assertEqual(self.post.call_args.kwargs['headers']['X-API-KEY'], self.keys[24])
        self.assertFalse(any(key in self.state.read_text() for key in self.keys))

    def test_rate_limit_retries_without_advancing_key(self):
        instance = tools.EvidenceTools(self.root / 'eval')
        self.post.side_effect = [response(429, {'message': 'Rate limit per second'}), response()]
        with patch.object(self.backend.time, 'sleep'):
            instance._search('test query', False)
        self.assertEqual([call.kwargs['headers']['X-API-KEY'] for call in self.post.call_args_list], [self.keys[22]] * 2)
        self.assertEqual(json.loads(self.state.read_text())['index'], 22)

    def test_exhaustion_uses_evaluator_exception_without_request(self):
        self.state.write_text(json.dumps({'pool': hashlib.sha256(json.dumps(self.keys).encode()).hexdigest(), 'index': 212, 'key_count': 212}))
        instance = tools.EvidenceTools(self.root / 'eval')
        with self.assertRaises(tools.SerperKeysExhausted):
            instance._search('test query', False)
        self.post.assert_not_called()


if __name__ == '__main__':
    unittest.main()
