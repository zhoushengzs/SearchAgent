"""Checks that diagnostic guards preserve evidence and do not accept truncations."""
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.generation_guard import repeated_token_span
from evaluation.local_api import LocalCompletionClient
from evaluation.audit_training_repetition import audit


class GuardTests(unittest.TestCase):
    def test_long_phrase_repeated_with_gaps_is_detected(self):
        phrase = list(range(32))
        hit = repeated_token_span(phrase + [100, 101] + phrase + [102] + phrase)
        self.assertEqual(hit['positions'], [0, 34, 67])

    def test_short_or_overlapping_repetition_is_not_enough(self):
        self.assertIsNone(repeated_token_span(list(range(150))))
        self.assertIsNone(repeated_token_span([1] * 70))
        self.assertIsNone(repeated_token_span(list(range(32)) * 2))

    def test_adapter_saves_raw_text_before_final_answer_extraction(self):
        raw = '<think>Check units.</think><response>42</response>'
        class Model:
            def generate(self, *args, **kwargs):
                return {'text': raw, 'status': 'ok', 'new_tokens': 16, 'stop_reason': 'eos'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.jsonl'
            client = LocalCompletionClient('/tmp/model', model=Model(), audit_path=path, answer_tag='answer')
            result = client.create(messages=[{'role': 'user', 'content': 'q'}])
            self.assertEqual(result.choices[0].message.content, '<answer>42</answer>')
            self.assertEqual(json.loads(path.read_text())['generation']['text'], raw)

    def test_training_audit_separates_failed_retries_from_successful_duplicates(self):
        call = '<tool_call>{"name":"vs_web_search","arguments":{"query":"q"}}</tool_call>'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'id': status, 'trajectory_id': status, 'conversations': [
                {'from':'human','value':'q'}, {'from':'gpt','value':call},
                {'from':'observation','value':json.dumps({'result':{'status':status}})},
                {'from':'gpt','value':call}]} for status in ['ok','error']]
            source = root/'train.jsonl'
            source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            original = source.read_bytes()
            summary = audit(source, root)
            self.assertEqual(summary['counts']['same_tool_and_arguments'], 2)
            self.assertEqual(summary['counts']['review_candidates_after_success'], 1)
            self.assertEqual(json.loads((root/'review_candidates.jsonl').read_text())['id'], 'ok')
            self.assertEqual(source.read_bytes(), original)

    def test_guarded_partial_answer_is_audited_but_not_submitted(self):
        class Model:
            def generate(self, *args, **kwargs):
                assert kwargs['repetition_guard'] and kwargs['max_time'] == 5
                return {'text':'<answer>42</answer> repeated fragment', 'status':'length',
                        'stop_reason':'repetition_guard','new_tokens':96}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'raw.jsonl'
            client = LocalCompletionClient('/tmp/model', model=Model(), audit_path=path,
                generation_limits={'repetition_guard':True,'max_time':5})
            result = client.create(messages=[{'role':'user','content':'q'}])
            self.assertEqual(result.choices[0].finish_reason, 'length')
            self.assertNotIn('<answer>', result.choices[0].message.content)
            self.assertIn('<answer>42</answer>',json.loads(path.read_text())['generation']['text'])
