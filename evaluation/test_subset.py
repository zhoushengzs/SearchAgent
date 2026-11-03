import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.common import read_predictions
from evaluation.package import export_bundle
from evaluation.prepare import digest, dump, write_rows
from evaluation.run import fingerprint
from evaluation.subset import derive_subset_run
from evaluation.test_parallel import task


class SubsetTests(unittest.TestCase):
    def fixture(self, root):
        values = [task(i) for i in range(4)]
        write_rows(root / 'tasks.jsonl', values)
        write_rows(root / 'refs.jsonl', [dict(id=t['id'], input_fingerprint=t['input_fingerprint'], answer='reference') for t in values])
        protocol = {'tasks_sha256': digest(root / 'tasks.jsonl'), 'selected_ids': [t['id'] for t in values], 'max_steps': 10}
        run = {'model': 'base', 'model_identity': {'model': 'base'}, 'protocol': protocol,
               'protocol_sha256': fingerprint(protocol), 'requested_device': 'openrouter'}
        run['run_id'] = fingerprint(run)
        run.update(tasks_file=str(root / 'tasks.jsonl'), shard={'index': 0, 'count': 1, 'selected_ids': protocol['selected_ids']})
        source = root / 'source'
        source.mkdir()
        dump(source / 'run.json', run)
        records = [dict(id=t['id'], input_fingerprint=t['input_fingerprint'], source_dataset=t['source_dataset'],
                        run_id=run['run_id'], status=['answered', 'step_limit', 'unavailable_tool', 'answered'][i],
                        answer='original answer' if i in (0, 3) else '', tool_calls=i,
                        infrastructure_error=i == 3, turns=[]) for i, t in enumerate(values)]
        write_rows(source / 'predictions.jsonl', records)
        return values, records, source

    def test_exclusion_and_derived_run_preserve_all_retained_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            values, records, source = self.fixture(root)
            before = (source / 'predictions.jsonl').read_bytes()
            manifest = export_bundle(root / 'tasks.jsonl', root / 'refs.jsonl', root / 'bundle', 4,
                                     exclude_ids=['task_3'], exclusion_reason='user requested')
            self.assertEqual(manifest['task_count'], 3)
            self.assertEqual(manifest['exclusions']['excluded_task_ids'], ['task_3'])
            self.assertEqual(len((root / 'bundle/references.jsonl').read_text().splitlines()), 3)
            result = derive_subset_run(source, root / 'bundle/tasks.jsonl', root / 'derived')
            self.assertEqual(result['inference_calls'], 0)
            run = json.loads((root / 'derived/run.json').read_text())
            derived = read_predictions(root / 'derived/predictions.jsonl', run['run_id'], {t['id']: t for t in values[:3]})
            for original in records[:3]:
                record = copy.deepcopy(derived[original['id']])
                provenance = record.pop('derived_from_prediction')
                self.assertEqual(provenance['sha256'], fingerprint(original))
                record['run_id'] = original['run_id']
                self.assertEqual(record, original)
            self.assertEqual((source / 'predictions.jsonl').read_bytes(), before)
            self.assertTrue(json.loads((root / 'derived/completion.json').read_text())['complete'])
            self.assertIn('post_hoc', run['derivation']['type'])

    def test_unknown_exclusions_and_empty_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            for excluded in [['missing'], [f'task_{i}' for i in range(4)]]:
                with self.assertRaises(ValueError):
                    export_bundle(root / 'tasks.jsonl', root / 'refs.jsonl', root / 'bundle', 4, exclude_ids=excluded)
                self.assertFalse((root / 'bundle').exists())

    def test_retained_infrastructure_failure_is_not_cleared_by_subsetting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, source = self.fixture(root)
            export_bundle(root / 'tasks.jsonl', root / 'refs.jsonl', root / 'bundle', 4, exclude_ids=['task_0'])
            with self.assertRaisesRegex(ValueError, 'infrastructure errors'):
                derive_subset_run(source, root / 'bundle/tasks.jsonl', root / 'derived')
            self.assertFalse((root / 'derived').exists())


if __name__ == '__main__':
    unittest.main()
