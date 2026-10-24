"""Offline checks of correction attempts shared by local and remote evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.protocol import training_tools
from evaluation.remote import OpenRouterModel
from evaluation.run import run_task, tool_definitions


def call(name, arguments):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


class SequenceModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1])
        self.transport = OpenRouterModel('offline', tokenizer, None)

    def generate(self, messages, schemas, *args, **kwargs):
        self.transport.prepare(messages, schemas)
        self.messages.append(messages)
        value = next(self.replies)
        return value if isinstance(value, dict) else {'text': value, 'status': 'ok'}


class ValidationRetryTests(unittest.TestCase):
    def run_sequence(self, replies, max_steps=3):
        model = SequenceModel(replies)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(tool_summaries=False, output=Path(directory), image_side=512, mode='agent',
                                   max_steps=max_steps, max_pixels=262144, max_new_tokens=64,
                                   context_tokens=1024, protocol='training', repeat_limit=2)
            task = {'id': 'fixture', 'source_dataset': 'fixture', 'input_fingerprint': 'fixture',
                    'model_input': {'question': 'question', 'images': []}}
            with patch('agent.tools.EvidenceTools.execute', return_value={
                    'tool': 'web_search', 'status': 'ok', 'data': {}}) as execute:
                result = run_task(task, model, args, tool_definitions(), 'fixture')
                executions = execute.call_count
        return result, model, executions

    def test_unknown_name_and_bad_arguments_correct_without_executing_rejected_proposals(self):
        result, model, executions = self.run_sequence([
            call('vs_text_search', {'q': 'q'}), call('vs_web_search', {'q': 'q'}),
            call('vs_web_search', {'query': 'q'}), '<response>answer</response>'])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(executions, 1)
        self.assertEqual([t['step'] for t in result['turns']], [1, 1, 1, 2])
        self.assertIn('vs_text_search', model.messages[1][-1]['content'])
        self.assertEqual(len([m for m in model.messages[-1] if m['role'] == 'tool']), 1)

    def test_three_failed_attempts_request_a_plain_final_answer_without_execution(self):
        result, model, executions = self.run_sequence(['bad'] * 3 + ['The answer is 42.'])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['answer'], 'The answer is 42.')
        self.assertTrue(result['validation_exhausted'])
        self.assertFalse(result['infrastructure_error'])
        self.assertEqual(len(model.messages), 4)
        self.assertIn('FINAL DECISION', model.messages[-1][-1]['content'])
        self.assertEqual(executions, 0)

    def test_final_decision_also_retries_format_and_forbidden_tool(self):
        result, _, executions = self.run_sequence([
            '<response>unclosed', call('vs_web_search', {'query': 'q'}),
            '<response>answer</response>'], max_steps=1)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(executions, 0)
        self.assertTrue(all(t['forced_answer'] for t in result['turns']))
        self.assertEqual(len(result['turns']), 3)

    def test_extra_explanation_does_not_discard_a_stated_answer(self):
        result, model, _ = self.run_sequence(['Explanation. <response>42</response>'])
        self.assertEqual(result['answer'], '42')
        self.assertEqual(len(model.messages), 1)

    def test_tool_proposals_are_never_scored_as_final_answers(self):
        result, model, executions = self.run_sequence([call('vs_text_search', {'q': 'q'})] * 6)
        self.assertEqual(result['status'], 'step_limit')
        self.assertTrue(result['final_answer_unavailable'])
        self.assertEqual(result['answer'], '')
        self.assertEqual(len(model.messages), 6)
        self.assertEqual(executions, 0)

    def test_context_failure_is_not_retried_as_a_format_error(self):
        result, model, _ = self.run_sequence([{'text': '', 'status': 'context_limit'}])
        self.assertEqual(result['status'], 'context_limit')
        self.assertEqual(len(model.messages), 1)

    def test_unsupported_tools_removed_from_schema_and_state(self):
        _, model, _ = self.run_sequence(['<response>answer</response>'])
        state = json.loads(model.messages[0][1]['content'][-1]['text'].split('\n', 1)[1])
        names = [tool['function']['name'] for tool in training_tools()]
        self.assertEqual(state['tool_configuration']['available_tools'], names)
        self.assertNotIn('image_search', names)
        self.assertNotIn('layout_parsing', names)
        self.assertNotIn('finish', names)
        self.assertIn('vs_image_search', names)


if __name__ == '__main__':
    unittest.main()
