import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.provenance import file_sha
from evaluation.common import load_tasks, materialize_task_paths, read_predictions, shard_tasks, task_fingerprint
from evaluation.merge import merge_shards
from evaluation.package import export_bundle
from evaluation.prepare import dump, write_rows
from evaluation.run import fingerprint, run_task, tool_definitions


def task(index, images=None):
    value = {'id': f'task_{index}', 'source_dataset': 'fixture',
             'model_input': {'question': f'Original question {index}', 'images': images or []},
             'dataset_provenance': {'revision': 'test'}, 'categories': {}}
    value['input_fingerprint'] = task_fingerprint(value)
    return value


class PortableInputTests(unittest.TestCase):
    def test_bundle_is_self_contained_after_move_and_reference_is_detached(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'original.png'
            Image.new('RGB', (32, 32), 'red').save(image)
            tasks = [task(0, [{'index': 1, 'path': str(image), 'sha256': file_sha(image)}])]
            write_rows(root / 'tasks.jsonl', tasks)
            write_rows(root / 'refs.jsonl', [{'id': tasks[0]['id'], 'input_fingerprint': tasks[0]['input_fingerprint'], 'answer': 'SECRET_REFERENCE'}])
            export_bundle(root / 'tasks.jsonl', root / 'refs.jsonl', root / 'bundle', 1)
            shutil.move(str(root / 'bundle'), root / 'server')
            image.unlink()
            exported = load_tasks(root / 'server/tasks.jsonl', 1)[0]
            resolved = materialize_task_paths(exported, root / 'server/tasks.jsonl')
            self.assertTrue(Path(resolved['model_input']['images'][0]['path']).exists())
            self.assertEqual(exported['input_fingerprint'], tasks[0]['input_fingerprint'])
            self.assertNotIn('SECRET_REFERENCE', (root / 'server/tasks.jsonl').read_text())

    def test_teacher_steps_and_changed_questions_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.jsonl'
            value = task(0)
            value['steps'] = [{'answer': 'leak'}]
            write_rows(path, [value])
            with self.assertRaises(ValueError):
                load_tasks(path)
            value = task(0)
            value['model_input']['question'] = 'different question'
            write_rows(path, [value])
            with self.assertRaises(ValueError):
                load_tasks(path)

    def test_wrong_full_count_and_unsafe_task_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.jsonl'
            write_rows(path, [task(0)])
            with self.assertRaises(ValueError):
                load_tasks(path, 3103, limit=1)
            value = task(0)
            value['id'] = '../escape'
            write_rows(path, [value])
            with self.assertRaises(ValueError):
                load_tasks(path)

    def test_changed_image_content_is_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'image.png'
            Image.new('RGB', (8, 8), 'red').save(image)
            value = task(0, [{'index': 1, 'path': str(image), 'sha256': file_sha(image)}])
            Image.new('RGB', (8, 8), 'blue').save(image)
            with self.assertRaises(ValueError):
                materialize_task_paths(value, Path(folder) / 'tasks.jsonl')

    def test_rebase_legacy_mac_path(self):
        from evaluation.common import resolve_image
        actual = resolve_image('/old/mac/project/data/image.png', '/server/tasks.jsonl',
                               '/server/project', '/old/mac/project')
        self.assertEqual(actual, Path('/server/project/data/image.png'))

    def test_eight_way_split_has_no_repetition_or_omission(self):
        values = [task(index) for index in range(3103)]
        parts = [shard_tasks(values, rank, 8) for rank in range(8)]
        self.assertEqual([len(part) for part in parts], [388] * 7 + [387])
        self.assertEqual({t['id'] for part in parts for t in part}, {t['id'] for t in values})
        self.assertEqual(sum(map(len, parts)), 3103)


class MergeTests(unittest.TestCase):
    def fixture(self, root, count=8, total=17):
        values = [task(index) for index in range(total)]
        tasks_file = root / 'tasks.jsonl'
        write_rows(tasks_file, values)
        from evaluation.prepare import digest
        protocol = {'tasks_sha256': digest(tasks_file), 'selected_ids': [t['id'] for t in values]}
        run = {'model': '/model', 'model_identity': {'weights': 'same'}, 'protocol': protocol,
               'protocol_sha256': fingerprint(protocol), 'requested_device': 'cuda'}
        run['run_id'] = fingerprint(run)
        for rank in range(count):
            directory = root / 'shards' / f'{rank:03d}'
            directory.mkdir(parents=True)
            assigned = shard_tasks(values, rank, count)
            dump(directory / 'run.json', {**run, 'shard': {'index': rank, 'count': count,
                                                          'selected_ids': [t['id'] for t in assigned]}})
            write_rows(directory / 'predictions.jsonl', [
                {'id': t['id'], 'source_dataset': t['source_dataset'], 'input_fingerprint': t['input_fingerprint'],
                 'run_id': run['run_id'], 'infrastructure_error': False, 'answer': 'fresh answer', 'status': 'answered'}
                for t in assigned])
        return tasks_file, values, run

    def test_complete_eight_worker_results_merge_in_original_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks, values, _ = self.fixture(root)
            self.assertEqual(merge_shards(root / 'shards', root / 'merged', tasks), 17)
            records = [json.loads(line) for line in (root / 'merged/predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['id'] for r in records], [t['id'] for t in values])
            self.assertNotIn('shard', json.loads((root / 'merged/run.json').read_text()))

    def test_missing_worker_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks, _, _ = self.fixture(root)
            shutil.rmtree(root / 'shards/007')
            with self.assertRaises(ValueError):
                merge_shards(root / 'shards', root / 'merged', tasks)

    def test_missing_prediction_and_infrastructure_failure_are_rejected(self):
        for failure in ['missing', 'infrastructure', 'wrong_fingerprint', 'duplicate']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                tasks, _, _ = self.fixture(root)
                path = root / 'shards/000/predictions.jsonl'
                records = [json.loads(line) for line in path.read_text().splitlines()]
                if failure == 'missing':
                    records.pop()
                elif failure == 'infrastructure':
                    records[0]['infrastructure_error'] = True
                elif failure == 'wrong_fingerprint':
                    records[0]['input_fingerprint'] = 'wrong'
                else:
                    records.append(records[0])
                write_rows(path, records)
                with self.assertRaises(ValueError):
                    merge_shards(root / 'shards', root / 'merged', tasks)

    def test_mixed_model_or_rank_assignment_is_rejected(self):
        for failure in ['model', 'rank', 'assignment', 'protocol']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                tasks, _, _ = self.fixture(root)
                path = root / 'shards/000/run.json'
                value = json.loads(path.read_text())
                if failure == 'model':
                    value['model'] = '/different'
                elif failure == 'rank':
                    value['shard']['index'] = 1
                elif failure == 'assignment':
                    value['shard']['selected_ids'].reverse()
                else:
                    value['protocol']['different_budget'] = 1
                dump(path, value)
                with self.assertRaises(ValueError):
                    merge_shards(root / 'shards', root / 'merged', tasks)

    def test_latest_infrastructure_retry_is_merged_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks, _, _ = self.fixture(root)
            path = root / 'shards/000/predictions.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            previous = {**records[0], 'infrastructure_error': True}
            write_rows(path, [previous, *records])
            self.assertEqual(merge_shards(root / 'shards', root / 'merged', tasks), 17)

    def test_launcher_starts_eight_isolated_processes_and_merges(self):
        from evaluation.launch import launch_workers
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks, _, _ = self.fixture(root)
            commands = [(str(rank), [sys.executable, '-c',
                         f'import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == "{rank}"'])
                        for rank in range(8)]
            launch_workers(SimpleNamespace(output=root, tasks=tasks), commands)
            self.assertTrue((root / 'merged/predictions.jsonl').is_file())
            self.assertEqual(len(list((root / 'logs').glob('*.log'))), 8)

    def test_launcher_worker_failure_does_not_publish_merged_results(self):
        from evaluation.launch import launch_workers
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks, _, _ = self.fixture(root, count=1)
            with self.assertRaises(ValueError):
                launch_workers(SimpleNamespace(output=root, tasks=tasks),
                               [('0', [sys.executable, '-c', 'raise SystemExit(7)'])])
            self.assertFalse((root / 'merged').exists())

    def test_score_uses_merged_task_manifest_and_rejects_a_single_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks, values, _ = self.fixture(root)
            references = [{'id': value['id'], 'input_fingerprint': value['input_fingerprint'], 'answer': 'answer'} for value in values]
            write_rows(root / 'references.jsonl', references)
            merge_shards(root / 'shards', root / 'merged', tasks)
            command = [sys.executable, '-m', 'evaluation.score', '--run', str(root / 'merged'),
                       '--output', str(root / 'score'), '--judge-model', 'unused', '--export-only']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            command[command.index('--run') + 1] = str(root / 'shards/000')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


class ResumeAndFreshExecutionTests(unittest.TestCase):
    def test_runner_shards_share_one_run_identity_and_resume_without_model_loading(self):
        from evaluation.run import main

        class FixtureModel:
            def __init__(self, *args):
                pass

            def generate(self, *args):
                return {'text': '<response>fresh answer</response>', 'status': 'ok',
                        'input_tokens': 10, 'new_tokens': 4}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            values = [task(index) for index in range(17)]
            write_rows(root / 'tasks.jsonl', values)
            model = root / 'model'
            model.mkdir()
            dump(model / 'config.json', {})
            (model / 'model.safetensors').write_bytes(b'fixture-not-real-weights')
            for rank in range(8):
                command = ['run', '--model', str(model), '--tasks', str(root / 'tasks.jsonl'),
                           '--expected-count', '17', '--num-shards', '8', '--shard-index', str(rank),
                           '--mode', 'direct', '--device', 'cuda', '--output', str(root / 'shards' / f'{rank:03d}')]
                with patch.object(sys, 'argv', command), patch('evaluation.run.LocalModel', FixtureModel):
                    main()
            self.assertEqual(merge_shards(root / 'shards', root / 'merged', root / 'tasks.jsonl'), 17)
            with patch.object(sys, 'argv', command), patch('evaluation.run.LocalModel') as model_constructor:
                main()
                model_constructor.assert_not_called()

    def test_corruption_in_the_middle_is_not_silently_recovered(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'predictions.jsonl'
            path.write_bytes(b'{bad json}\n{}\n')
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                read_predictions(path, 'run', {}, recover_tail=True)
            self.assertEqual(path.read_bytes(), before)

    def test_interrupted_final_line_is_backed_up_and_completed_work_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            value = task(0)
            path = Path(folder) / 'predictions.jsonl'
            record = {'id': value['id'], 'input_fingerprint': value['input_fingerprint'],
                      'source_dataset': value['source_dataset'], 'run_id': 'run'}
            path.write_bytes((json.dumps(record) + '\n{"interrupted":').encode())
            with self.assertRaises(ValueError):
                read_predictions(path, 'run', {value['id']: value})
            latest = read_predictions(path, 'run', {value['id']: value}, recover_tail=True)
            self.assertEqual(latest[value['id']], record)
            self.assertTrue(path.with_name('predictions.jsonl.interrupted-tail').is_file())
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_model_runs_real_calculator_from_a_fresh_original_input(self):
        class FixtureModel:
            def __init__(self):
                self.calls = 0

            def generate(self, messages, schemas, max_tokens, context_tokens):
                self.calls += 1
                if self.calls == 1:
                    text = json.dumps(messages)
                    assert 'SECRET_REFERENCE' not in text
                    state = json.loads(messages[1]['content'][-1]['text'].split('\n', 1)[1])
                    assert state['executed_prefix_step_ids'] == []
                    assert state['question'] == 'Original question 0'
                    result = '<tool_call>{"name":"vs_calculate","arguments":{"expression":"2+3"}}</tool_call>'
                else:
                    assert messages[-1]['role'] == 'tool'
                    actual = json.loads(messages[-1]['content'])['result']
                    assert actual['status'] == 'ok'
                    assert actual['data']['value'] == 5
                    result = '<response>5</response>'
                return {'text': result, 'status': 'ok', 'input_tokens': 10, 'new_tokens': 10}

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), image_side=512, mode='agent', max_steps=2,
                                   max_pixels=262144, max_new_tokens=64, context_tokens=1024)
            record = run_task(task(0), FixtureModel(), args, tool_definitions(), 'fixture')
            self.assertEqual(record['status'], 'answered')
            self.assertEqual(record['answer'], '5')
            self.assertEqual(record['tool_calls'], 1)


if __name__ == '__main__':
    unittest.main()
