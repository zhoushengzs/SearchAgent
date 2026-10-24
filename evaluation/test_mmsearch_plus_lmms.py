import json
from pathlib import Path
import tempfile
import unittest

from evaluation.mmsearch_plus_lmms import run_evaluation, score_answer
from evaluation.prepare import digest, write_rows
from evaluation.common import task_fingerprint


class MetricsTests(unittest.TestCase):
    def test_alternatives_accept_one_normalized_match(self):
        self.assertEqual(score_answer('  UNITED States!! ', ['USA', 'United States']),
                         {'f1_score': 1., 'exact_match': 1.})

    def test_partial_word_overlap_is_not_correctness(self):
        result = score_answer('New York', ['New York City'])
        self.assertEqual(result['exact_match'], 0)
        self.assertAlmostEqual(result['f1_score'], .8)

    def test_empty_predictions_and_missing_gold_retain_upstream_behavior(self):
        for prediction, gold in [('', ['Paris']), ('Paris', [])]:
            self.assertEqual(score_answer(prediction, gold), {'f1_score': 0., 'exact_match': 0.})

    def test_duplicate_token_behavior_is_preserved_from_upstream(self):
        result = score_answer('red red', ['red red'])
        self.assertEqual(result, {'f1_score': .5, 'exact_match': 1.})

    def test_export_identity_missing_references_and_infrastructure_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);run_dir = root/'run';run_dir.mkdir()
            tasks=[];refs=[];predictions=[]
            for i in range(3):
                task={'id': f'mmsearch_plus_train_{i:04}', 'source_dataset':'mmsearch_plus',
                      'model_input':{'question':f'Question {i}', 'images':[]},
                      'dataset_provenance':{'revision':'fixture'},'categories':{}}
                task['input_fingerprint']=task_fingerprint(task);tasks.append(task)
                refs.append({'id':task['id'],'input_fingerprint':task['input_fingerprint'],
                             'answer':'Paris' if i==0 else '', 'answer_candidates':['Paris'] if i==0 else [],
                             'answer_format':'alternatives'})
                predictions.append({'id':task['id'],'input_fingerprint':task['input_fingerprint'],
                                    'source_dataset':'mmsearch_plus','run_id':'fixture','status':'invalid_output',
                                    'answer':'','turns':[{'status':'ok','text':'Explanation. <response>PARIS!</response>'}],
                                    'infrastructure_error':i==2})
            write_rows(root/'tasks.jsonl',tasks);write_rows(root/'references.jsonl',refs)
            write_rows(run_dir/'predictions.jsonl',predictions)
            (run_dir/'run.json').write_text(json.dumps({'run_id':'fixture','protocol':{
                'tasks_sha256':digest(root/'tasks.jsonl'),'selected_ids':[t['id'] for t in tasks]}}))
            args=(run_dir,root/'tasks.jsonl',root/'references.jsonl',root/'scores')
            summary=run_evaluation(*args,expected_count=3)
            self.assertEqual(summary['raw_upstream_metrics']['exact_match_pct'],50)
            self.assertEqual(summary['valid_reference_metrics']['exact_match_pct'],100)
            self.assertEqual(summary['missing_reference_ids'],[tasks[1]['id']])
            self.assertEqual(len(summary['excluded']),1)
            self.assertEqual(run_evaluation(*args,expected_count=3),summary)
            # Altering a source prediction invalidates the existing score snapshot.
            predictions[0]['turns'][0]['text']='<response>London</response>'
            write_rows(run_dir/'predictions.jsonl',predictions)
            with self.assertRaisesRegex(ValueError,'different inputs'):
                run_evaluation(*args,expected_count=3)


if __name__ == '__main__':
    unittest.main()
