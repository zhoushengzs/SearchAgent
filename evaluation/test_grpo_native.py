"""No-network tests of native action handling, image inputs and reference separation."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from evaluation.grpo_native import NativeEngine, evaluate_task, load_runtime


class FakeModel:
    def __init__(self, texts):
        self.texts, self.calls = iter(texts), []

    def generate(self, messages, tools, cap, context, **kwargs):
        self.calls.append((messages, tools, cap, context, kwargs))
        return {'text': next(self.texts), 'status': 'ok', 'input_tokens': 32, 'new_tokens': 12}


class EngineTests(unittest.TestCase):
    def test_images_are_saved_and_native_tools_are_not_injected_twice(self):
        import asyncio
        with tempfile.TemporaryDirectory() as folder:
            model = FakeModel(['<response>4</response>'])
            engine = NativeEngine(model, Path(folder)/'images', 128, 64, 262144)
            messages = [{'role': 'system', 'content': 'native prompt'},
                        {'role': 'user', 'content': 'question', 'images': [Image.new('RGB', (4, 4), 'red')]}]
            asyncio.run(engine.get_model_response(messages, 64))
            sent, tools, cap, context, options = model.calls[0]
            self.assertEqual(sent[0]['content'][0]['text'], 'native prompt')
            self.assertEqual(tools, [])
            self.assertEqual((cap, context), (64, 192))
            self.assertFalse(options['do_sample'])
            self.assertTrue(Path(sent[1]['content'][0]['image']).exists())
            self.assertNotIn('images', sent[1])

    def test_native_worker_command_never_receives_detached_references(self):
        from evaluation.native_benchmark import native_commands
        args = SimpleNamespace(max_steps=10, prompt_tokens=28672, temperature=0.0,
                               retry_infrastructure_errors=False)
        commands = native_commands(args, Path('/native-runtime'), Path('/new-results'), ['0', '1'])
        self.assertEqual(len(commands), 2)
        for device, command in commands:
            self.assertNotIn('--references', command)
            self.assertEqual(command[command.index('--max-steps')+1], '10')
            self.assertEqual(command[command.index('--prompt-tokens')+1], '28672')
            self.assertEqual(command[command.index('--num-shards')+1], '2')
            self.assertNotIn('evaluation.run', command)


@unittest.skipUnless(os.environ.get('NATIVE_GRPO_RUNTIME'), 'Set NATIVE_GRPO_RUNTIME to test the actual training agent')
class ActualNativeAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent, cls.session = load_runtime(os.environ['NATIVE_GRPO_RUNTIME'])

    def run_case(self, texts, steps):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        task = {'id': 'test0', 'source_dataset': 'fixture', 'input_fingerprint': 'fixture',
                'model_input': {'question': 'What is 2+2?', 'images': []},
                'answer': 'SECRET_REFERENCE_MUST_NOT_ENTER_MODEL'}
        args = SimpleNamespace(output=root, prompt_tokens=1024, max_new_tokens=64,
                               max_pixels=262144, temperature=0.0, max_steps=steps)
        model = FakeModel(texts)
        with patch.dict(os.environ, {}, clear=True):
            row = evaluate_task(task, model, self.agent, self.session, args, 'fixture-run')
        return root, row, model

    def test_real_agent_calculates_then_answers_without_reference(self):
        root, row, model = self.run_case([
            '<think>Calculate.</think><tool_call>{"name":"vs_calculate","arguments":{"expression":"2+2"}}</tool_call>',
            '<think>Done.</think><response>4</response>'], 2)
        self.assertEqual((row['status'], row['answer'], row['tool_calls']), ('answered', '4', 1))
        self.assertEqual(len(model.calls), 2)
        self.assertFalse(row['infrastructure_error'])
        audit = ''.join(p.read_text() for p in root.rglob('*.json*'))
        self.assertNotIn('SECRET_REFERENCE_MUST_NOT_ENTER_MODEL', audit)
        self.assertTrue((root/'artifacts/test0/evidence/step_1_result.json').exists())
        self.assertIn('Only the six tools', model.calls[0][0][0]['content'][0]['text'])

    def test_no_legacy_forced_answer_or_three_extra_format_attempts(self):
        _, row, model = self.run_case(['Plain answer without native tags.'], 1)
        self.assertEqual(row['status'], 'step_limit')
        self.assertEqual(row['answer'], '')
        self.assertEqual(len(model.calls), 1)

    def test_successful_repeated_action_uses_the_native_rejection(self):
        call = '<tool_call>{"name":"vs_calculate","arguments":{"expression":"2+2"}}</tool_call>'
        _, row, model = self.run_case([call, call, '<response>4</response>'], 3)
        self.assertEqual(row['duplicate_calls'], 1)
        self.assertEqual(row['tool_calls'], 2)
        self.assertEqual(row['status'], 'answered')
        self.assertIn('RepeatedAction', json.dumps(model.calls[2][0]))

    def test_nine_name_evaluator_alias_is_not_a_native_tool(self):
        with self.assertRaises(ValueError):
            self.session.parse_action('<tool_call>{"name":"text_search","arguments":{"q":"x"}}</tool_call>')


if __name__ == '__main__':
    unittest.main()
