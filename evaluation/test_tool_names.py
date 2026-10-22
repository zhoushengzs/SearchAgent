"""Offline regression checks for the public tool names seen by both models."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from evaluation.run import available_tool_definitions, initial_message, run_task, tool_definitions


class ToolNameTests(unittest.TestCase):
    def test_plain_final_answer_is_a_model_format_error(self):
        with tempfile.TemporaryDirectory() as folder:
            model = SimpleNamespace(generate=lambda *args: {'text': '42', 'status': 'ok'})
            args = SimpleNamespace(output=Path(folder), image_side=512, mode='agent',
                                   max_steps=3, max_pixels=512, max_new_tokens=64, context_tokens=1024)
            task = {'id': 'fixture', 'input_fingerprint': 'fixture', 'source_dataset': 'fixture',
                    'model_input': {'question': 'test', 'images': []}}
            record = run_task(task, model, args, tool_definitions(), 'fixture')
            self.assertEqual(record['status'], 'invalid_output')
            self.assertEqual(record['answer'], '')
            self.assertFalse(record['infrastructure_error'])

    def test_initial_list_matches_definitions_and_preserves_disabled_tools(self):
        for disabled in (set(), {'image_search'}, {'image_search', 'web_search'}):
            internal = {'web_search', 'image_search', 'visit', 'inspect_image', 'crop',
                        'calculate', 'finish', 'abstain'} - disabled
            config = {'available_tools': sorted(internal), 'backend': 'fixture'}
            tools = SimpleNamespace(configuration=lambda: config, inventory=lambda: [])
            schemas = available_tool_definitions(tools, tool_definitions())
            message = initial_message({'model_input': {'question': 'test'}}, tools, [],
                                      10, 'agent', 512, schemas)
            state = json.loads(message['content'][-1]['text'].split('\n', 1)[1])
            names = set(state['tool_configuration']['available_tools'])
            self.assertEqual(names, {schema['function']['name'] for schema in schemas})
            self.assertEqual(names, {'vs_' + name for name in internal - {'finish', 'abstain'}})
            self.assertEqual(config['available_tools'], sorted(internal))

    def test_old_and_terminal_names_are_rejected_without_execution(self):
        for name in ('web_search', 'image_search', 'visit', 'inspect_image', 'crop',
                     'calculate', 'finish', 'abstain'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                model = SimpleNamespace(generate=lambda *args: {
                    'text': '<tool_call>' + json.dumps({'name': name, 'arguments': {}}) + '</tool_call>',
                    'status': 'ok'})
                args = SimpleNamespace(output=Path(folder), image_side=512, mode='agent',
                                       max_steps=3, max_pixels=512, max_new_tokens=64, context_tokens=1024)
                task = {'id': 'fixture', 'input_fingerprint': 'fixture', 'source_dataset': 'fixture',
                        'model_input': {'question': 'test', 'images': []}}
                with patch('agent.tools.EvidenceTools.execute') as execute:
                    record = run_task(task, model, args, tool_definitions(), 'fixture')
                    execute.assert_not_called()
                self.assertEqual(record['status'], 'unavailable_tool')
                self.assertFalse(record['infrastructure_error'])


if __name__ == '__main__':
    unittest.main()
