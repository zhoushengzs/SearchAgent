"""Unit tests for the training-format protocol and repetition controls."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import protocol
from evaluation.run import run_task, tool_definitions
from evaluation.test_parallel import task


def fake_task():
    row = task(0)
    row['model_input']['images'] = []
    return row


class ProtocolTests(unittest.TestCase):
    def test_training_assets_match_the_saved_sft_prompt_and_tools(self):
        self.assertIn('Tool action supervision does not imply', protocol.training_system())
        names = [tool['function']['name'] for tool in protocol.training_tools()]
        self.assertEqual(names, ['image_search', 'text_search', 'web_search', 'layout_parsing', 'crop',
                                 'vs_web_search', 'vs_image_search', 'vs_visit', 'vs_inspect_image',
                                 'vs_crop', 'vs_calculate'])

    def test_state_message_has_refreshed_training_fields(self):
        tools = SimpleNamespace(configuration=lambda: {'backend': 'serper', 'available_tools': ['web_search']},
                                inventory=lambda: [{'asset_id': 'image_1', 'origin': 'initial_input'}])
        initial_views = [{'asset_id': 'image_1', 'width': 100, 'height': 50}]
        attached = [{'asset_id': 'image_1', 'path': '/tmp/a.png', 'width': 100, 'height': 50,
                     'image_index': 1, 'observed_at_step': 0}]
        message = protocol.state_message(task(0) | {'model_input': {'question': 'q', 'images': []}},
                                         tools, initial_views, attached, [1, 2], 3, 10, 262144)
        state = json.loads(message['content'][-1]['text'].split('\n', 1)[1])
        self.assertEqual(state['executed_prefix_step_ids'], [1, 2])
        self.assertEqual(state['remaining_decisions'], 8)
        self.assertEqual(state['attached_images'][0]['image_index'], 1)
        self.assertEqual(state['tool_configuration']['available_tools'],
                         protocol.STATE_AVAILABLE_TOOLS)
        self.assertEqual([block.get('type') for block in message['content']], ['text', 'image', 'text', 'text'])

    def test_native_tool_names_map_onto_project_tools(self):
        attached = [{'asset_id': 'image_1', 'width': 100, 'height': 50}]
        self.assertEqual(protocol.normalize_call('web_search', {'q': 'x'}, attached),
                         ('vs_web_search', {'query': 'x'}))
        self.assertEqual(protocol.normalize_call('inspect_image', {'asset_id': 'image_1'}, attached),
                         ('vs_inspect_image', {'asset_id': 'image_1'}))
        self.assertEqual(protocol.normalize_call('calculate', {'expression': '1+1'}, attached),
                         ('vs_calculate', {'expression': '1+1'}))
        self.assertEqual(protocol.normalize_call('crop', {'image': 'img_1', 'x': 0, 'y': 0,
                                                          'width': 50, 'height': 25}, attached),
                         ('vs_crop', {'asset_id': 'image_1', 'bbox': [0.0, 0.0, 0.5, 0.5]}))
        with self.assertRaises(ValueError):
            protocol.normalize_call('image_search', {'url': 'https://example.com/x.png'}, attached)


class GuardTests(unittest.TestCase):
    def test_rejected_and_duplicate_actions_keep_matching_remote_tool_history(self):
        from evaluation.remote import OpenRouterModel

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]

        class Model:
            calls = 0
            wire = OpenRouterModel('fixture', Tokenizer(), None)
            def generate(self, messages, schemas, max_tokens, context_tokens, **kwargs):
                self.wire.prepare(messages, schemas)  # Raises on an orphan tool result.
                self.calls += 1
                if self.calls == 1:
                    text = '<tool_call>{"name":"layout_parsing","arguments":{"image":"img_1"}}</tool_call>'
                elif schemas:
                    text = '<tool_call>{"name":"vs_web_search","arguments":{"query":"same"}}</tool_call>'
                else:
                    text = '<response>done</response>'
                return {'text': text, 'status': 'ok'}

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), image_side=512, mode='agent', max_steps=6,
                                   max_pixels=262144, max_new_tokens=64, context_tokens=1024,
                                   protocol='training', repetition_penalty=1.1, repeat_limit=2)
            with patch('agent.tools.EvidenceTools.execute',
                       return_value={'tool': 'web_search', 'status': 'ok', 'data': {}}):
                record = run_task(fake_task(), Model(), args, tool_definitions(), 'fixture')
        self.assertEqual(record['status'], 'answered')
        self.assertFalse(record['infrastructure_error'])
        self.assertEqual(record['duplicate_calls'], 2)
        self.assertEqual(record['tool_calls'], 1)

    def test_duplicate_calls_are_not_executed_and_force_a_final_answer(self):
        class Model:
            calls = 0
            def generate(self, messages, schemas, max_tokens, context_tokens, **kwargs):
                self.calls += 1
                if schemas:
                    return {'text': '<tool_call>{"name":"vs_web_search","arguments":{"query":"same"}}</tool_call>',
                            'status': 'ok'}
                assert 'FINAL DECISION' in messages[-1]['content']
                return {'text': '<response>final answer</response>', 'status': 'ok'}
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), image_side=512, mode='agent', max_steps=5,
                                   max_pixels=262144, max_new_tokens=64, context_tokens=1024,
                                   protocol='eval', repetition_penalty=1.0, repeat_limit=2)
            with patch('agent.tools.EvidenceTools.execute',
                       return_value={'tool': 'web_search', 'status': 'ok', 'data': {}}), \
                 patch('agent.tools.EvidenceTools.configuration',
                       return_value={'available_tools': ['web_search']}):
                record = run_task(fake_task(), Model(), args, tool_definitions(), 'fixture')
        self.assertEqual(record['status'], 'answered')
        self.assertEqual(record['answer'], 'final answer')
        self.assertEqual(record['duplicate_calls'], 2)
        self.assertTrue(record['turns'][-1].get('forced_answer'))
        self.assertEqual(record['tool_calls'], 1)

    def test_repetition_penalty_is_forwarded_to_local_generation(self):
        class Model:
            seen = None
            def generate(self, messages, schemas, max_tokens, context_tokens, **kwargs):
                self.seen = kwargs
                return {'text': '<response>ok</response>', 'status': 'ok'}
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), image_side=512, mode='agent', max_steps=1,
                                   max_pixels=262144, max_new_tokens=64, context_tokens=1024,
                                   protocol='eval', repetition_penalty=1.1, repeat_limit=0)
            model = Model()
            with patch('agent.tools.EvidenceTools.configuration',
                       return_value={'available_tools': ['web_search']}):
                run_task(fake_task(), model, args, tool_definitions(), 'fixture')
        self.assertEqual(model.seen, {'repetition_penalty': 1.1})

    def test_training_protocol_uses_training_prompt_and_refreshed_state(self):
        class Model:
            seen = []
            def generate(self, messages, schemas, max_tokens, context_tokens, **kwargs):
                self.seen.append(messages)
                if len(self.seen) == 1:
                    return {'text': '<tool_call>{"name":"vs_web_search","arguments":{"query":"q"}}</tool_call>',
                            'status': 'ok'}
                return {'text': '<response>done</response>', 'status': 'ok'}
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), image_side=512, mode='agent', max_steps=2,
                                   max_pixels=262144, max_new_tokens=64, context_tokens=1024,
                                   protocol='training', repetition_penalty=1.0, repeat_limit=0)
            model = Model()
            with patch('agent.tools.EvidenceTools.execute',
                       return_value={'tool': 'web_search', 'status': 'ok', 'data': {}}):
                record = run_task(fake_task(), model, args, tool_definitions(), 'fixture')
        self.assertEqual(record['status'], 'answered')
        first = model.seen[0]
        self.assertEqual(first[0]['content'], protocol.training_system())
        state_text = first[1]['content'][-1]['text']
        self.assertIn('"attached_images"', state_text)
        self.assertIn('"executed_prefix_step_ids"', state_text)
        # The second decision is rebuilt with the executed prefix, not a stale step-1 state.
        second = model.seen[1]
        self.assertIn('"executed_prefix_step_ids": [1]', second[1]['content'][-1]['text'])


class RemotePayloadTests(unittest.TestCase):
    def test_openrouter_payload_carries_repetition_penalty(self):
        from evaluation.remote import OpenRouterModel

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]

        class Client:
            payload = None
            def post(self, payload):
                self.payload = payload
                return {'model': 'qwen/qwen3-vl-8b-instruct',
                        'choices': [{'message': {'content': '<response>ok</response>'},
                                     'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 2},
                        'id': 'request-1', 'provider': 'fixture'}

        client = Client()
        model = OpenRouterModel('qwen/qwen3-vl-8b-instruct', Tokenizer(), client)
        out = model.generate([{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'}],
                             [], 64, 1024, repetition_penalty=1.1)
        self.assertEqual(out['status'], 'ok')
        self.assertEqual(client.payload['repetition_penalty'], 1.1)


if __name__ == '__main__':
    unittest.main()
