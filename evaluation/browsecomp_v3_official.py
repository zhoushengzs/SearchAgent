"""Export BrowseComp-V3 predictions and evaluate SR/PS with pinned official rules."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from evaluation.common import load_tasks, read_predictions, write_json_atomic
from evaluation.prepare import digest
from evaluation.remote import APIClient, RemoteAPIError, output_lock, parallel_results
from evaluation.run import fingerprint, load_env
from evaluation.vendor.browsecomp_v3_official import (
    OFFICIAL_SOURCE, OfficialSubgoalExtractor, PROMPT_TEMPLATE, RESPONSE_FORMAT,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_VERSION = 'browsecomp-v3-recorded-trace-adapter-v1'


def build_sample(task, reference, prediction, model_name):
    """Use official assistant-text extraction, without generating new reasoning."""
    annotations = reference.get('annotations', {})
    metadata = annotations.get('metadata', {})
    goals = annotations.get('sub_goals')
    if not isinstance(goals, list) or not goals:
        raise ValueError(f'{task["id"]}: missing official sub_goals')
    for goal in goals:
        if (not isinstance(goal, dict) or not isinstance(goal.get('description'), str)
                or not isinstance(goal.get('key_info'), str) or not goal['key_info'].strip()):
            raise ValueError(f'{task["id"]}: invalid official sub-goal')
    if not isinstance(reference.get('answer'), str) or not reference['answer'].strip():
        raise ValueError(f'{task["id"]}: missing gold final answer')
    messages = [{'role': 'assistant', 'content': turn['text']}
                for turn in prediction.get('turns', [])
                if isinstance(turn.get('text'), str) and turn['text'].strip()]
    subgoals = OfficialSubgoalExtractor()._extract_subgoals_from_messages(messages, goals)
    # Recorded final answers are used verbatim. No answer is invented from tool
    # arguments, failed formatting or a truncated generation.
    answer = prediction.get('answer', '') if prediction.get('status') == 'answered' else ''
    if not isinstance(answer, str):
        raise ValueError('Recorded final answer must be a string')
    qid = task['id'].removeprefix('browsecomp_v3_train_')
    if qid in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+', qid):
        raise ValueError('Unsafe official question ID')
    return {
        'execution': {'q_id': qid, 'model_name': model_name, 'status': prediction['status']},
        'item': {'question': task['model_input']['question'],
                 'images': [image['path'] for image in task['model_input']['images']],
                 'tags': {key: metadata.get(key) for key in ['level', 'difficulty', 'domain', 'fc_num']}},
        'ground_truth': {'final_answer': reference['answer'], 'sub_goals': copy.deepcopy(goals)},
        'prediction': {'final_answer': answer, 'subgoals': subgoals},
        'process': {'messages': messages},
        'adapter_provenance': {
            'task_id': task['id'], 'input_fingerprint': task['input_fingerprint'],
            'prediction_sha256': fingerprint(prediction),
            'original_status': prediction['status'],
            'answer_policy': 'recorded_final_answer_only; no complete answer means empty string',
            'subgoal_policy': 'verbatim official extraction from original assistant text; no model rewriting',
        },
    }


def prepare_export(run_dir, tasks_file, references_file, output, expected_count=59):
    run_dir, tasks_file, references_file, output = map(Path, [run_dir, tasks_file, references_file, output])
    run = json.loads((run_dir / 'run.json').read_text())
    if run.get('shard', {}).get('count', 1) > 1:
        raise ValueError('Use the merged SFT directory, not an individual GPU shard')
    if digest(tasks_file) != run['protocol']['tasks_sha256']:
        raise ValueError('Task file differs from the recorded inference run')
    all_tasks = load_tasks(tasks_file)
    by_id = {task['id']: task for task in all_tasks}
    selected = run['protocol']['selected_ids']
    if len(set(selected)) != len(selected) or not set(selected) <= set(by_id):
        raise ValueError('Invalid source run task selection')
    tasks = [task for task in all_tasks if task['source_dataset'] == 'browsecomp_v3']
    if len(tasks) != expected_count:
        raise ValueError(f'Expected {expected_count} BrowseComp-V3 tasks, found {len(tasks)}')
    if not {task['id'] for task in tasks} <= set(selected):
        raise ValueError('Inference run does not cover the complete requested BrowseComp-V3 subset')
    source_sha = digest(run_dir / 'predictions.jsonl')
    predictions = read_predictions(run_dir / 'predictions.jsonl', run['run_id'], {key: by_id[key] for key in selected})
    if digest(run_dir / 'predictions.jsonl') != source_sha:
        raise ValueError('Source predictions changed during export; retry after they are stable')
    refs = {}
    for line in references_file.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        ref = json.loads(line)
        key = (ref['id'], ref['input_fingerprint'])
        if key in refs:
            raise ValueError('Duplicate reference identity')
        refs[key] = ref
    samples = []
    for task in tasks:
        key = task['id']
        if key not in predictions:
            raise ValueError(f'{key}: prediction is missing')
        if predictions[key].get('infrastructure_error'):
            raise ValueError(f'{key}: unresolved infrastructure error')
        ref = refs.get((key, task['input_fingerprint']))
        if ref is None:
            raise ValueError(f'{key}: reference is missing or mismatched')
        samples.append(build_sample(task, ref, predictions[key], run['model']))
    if len({s['execution']['q_id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate official question IDs')
    manifest = {
        'adapter': ADAPTER_VERSION, 'official_source': OFFICIAL_SOURCE,
        'vendor_sha256': digest(Path(__file__).parent / 'vendor/browsecomp_v3_official.py'),
        'adapter_sha256': digest(__file__), 'source_run_id': run['run_id'],
        'source_run_sha256': digest(run_dir / 'run.json'),
        'source_predictions_sha256': source_sha, 'tasks_sha256': digest(tasks_file),
        'references_sha256': digest(references_file), 'count': len(samples),
        'task_ids': [task['id'] for task in tasks],
        'sample_sha256': {s['execution']['q_id']: fingerprint(s) for s in samples},
        'complete_final_answers': sum(bool(s['prediction']['final_answer'].strip()) for s in samples),
        'samples_with_extracted_subgoals': sum(bool(s['prediction']['subgoals']) for s in samples),
        'inference_calls': 0,
        'scope': f"Project {len(samples)}-task post-hoc holdout subset; original {run['protocol'].get('max_steps', 'unknown')}-decision agent trajectories, not the full 300-task/20-turn benchmark.",
    }
    if output.exists():
        if not (output / 'manifest.json').is_file() or json.loads((output / 'manifest.json').read_text()) != manifest:
            raise ValueError('Output belongs to different inputs/code; choose a new directory')
        for sample in samples:
            path = output / 'samples' / (sample['execution']['q_id'] + '.json')
            if not path.is_file() or fingerprint(json.loads(path.read_text())) != fingerprint(sample):
                raise ValueError('Exported sample was changed; use an unmodified export')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.bcv3-export-', dir=output.parent))
        try:
            (temporary / 'samples').mkdir()
            for sample in samples:
                write_json_atomic(temporary / 'samples' / (sample['execution']['q_id'] + '.json'), sample)
            write_json_atomic(temporary / 'manifest.json', manifest)
            temporary.rename(output)
        except BaseException:
            shutil.rmtree(temporary)
            raise
    return samples, manifest


def official_prompt(sample):
    goals = '\n'.join(f'- SG{g.get("sg_id")}: {g.get("description", "")} | key_info: {g.get("key_info", "")}'
                      for g in sample['ground_truth']['sub_goals']) or '无'
    predicted = '\n'.join(f'- {s}' for s in sample['prediction']['subgoals']) or '无'
    return PROMPT_TEMPLATE.format(question=sample['item']['question'],
                                  gt_answer=sample['ground_truth']['final_answer'],
                                  formatted_sub_goals=goals,
                                  pred_answer=sample['prediction']['final_answer'],
                                  formatted_pred_subgoals=predicted)


def validate_verdict(verdict, sample):
    required = set(RESPONSE_FORMAT['json_schema']['schema']['required'])
    if not isinstance(verdict, dict) or set(verdict) != required:
        raise ValueError('Judge verdict has missing or unexpected fields')
    count = len(sample['ground_truth']['sub_goals'])
    vector, details = verdict['subgoal_vector'], verdict['subgoal_details']
    if type(verdict['answer_correct']) is not bool or type(verdict['anomaly_flag']) is not bool:
        raise ValueError('Judge correctness and anomaly values must be booleans')
    if not sample['prediction']['final_answer'].strip() and verdict['answer_correct']:
        raise ValueError('A missing final answer cannot be correct')
    if (not isinstance(vector, list) or len(vector) != count
            or any(type(value) is not int or value not in (0, 1) for value in vector)):
        raise ValueError('Subgoal vector differs from the gold sub-goal count')
    score = verdict['process_score']
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Process score must be in [0,1]')
    if not math.isclose(score, sum(vector) / count, rel_tol=0, abs_tol=0.001):
        raise ValueError('Process score is inconsistent with the subgoal vector')
    if not isinstance(details, list) or len(details) != count:
        raise ValueError('Missing per-subgoal details')
    for index, detail in enumerate(details, 1):
        if (not isinstance(detail, dict) or set(detail) != {'subgoal_index', 'status', 'reason'}
                or type(detail['subgoal_index']) is not int or detail['subgoal_index'] != index
                or detail['status'] != ('ACHIEVED' if vector[index - 1] else 'NOT_ACHIEVED')
                or not isinstance(detail['reason'], str)):
            raise ValueError('Invalid per-subgoal detail')
    if verdict['confidence'] not in ('HIGH', 'MEDIUM', 'LOW'):
        raise ValueError('Invalid judge confidence')
    if any(not isinstance(verdict[key], str) for key in ['anomaly_reason', 'judge_reasoning']):
        raise ValueError('Invalid judge explanation')
    return verdict


def evaluate_sample(sample, client, args, policy_id):
    prompt = official_prompt(sample)
    if args.response_format == 'json_object':
        # Compatible APIs enforce JSON syntax only. Supply the same official
        # schema in the prompt so their field names also match the contract.
        prompt += '\n\nReturn a JSON object conforming exactly to this official output schema:\n' + json.dumps(
            RESPONSE_FORMAT['json_schema']['schema'], ensure_ascii=False)
    request = {'model': args.judge_model, 'messages': [{'role': 'user', 'content': prompt}],
               'max_tokens': args.judge_max_tokens,
               'response_format': copy.deepcopy(RESPONSE_FORMAT) if args.response_format == 'json_schema' else {'type': 'json_object'}}
    if args.judge_thinking != 'provider-default':
        request['thinking'] = {'type': args.judge_thinking}
    # The official request leaves sampling settings unspecified; do the same.
    for attempt in range(3):
        try:
            body = client.post(request)
            choice = body['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Judge output was truncated or unfinished')
            content = choice['message']['content']
            verdict = validate_verdict(json.loads(content), sample)
            return {'q_id': sample['execution']['q_id'], 'model_name': sample['execution']['model_name'],
                    'judge_model_name': args.judge_model, 'tags': sample['item']['tags'],
                    'score': {'final': float(verdict['answer_correct']), 'process': float(verdict['process_score'])},
                    'metadata': {'verdict': verdict, 'judge_output': content, 'returned_model': body.get('model'),
                                 'usage': body.get('usage'), 'request_id': body.get('id'), 'provider': body.get('provider')},
                    'sample_sha256': fingerprint(sample), 'policy_id': policy_id}
        except RemoteAPIError as exc:
            return {'q_id': sample['execution']['q_id'], 'error_type': type(exc).__name__}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            if attempt == 2:
                return {'q_id': sample['execution']['q_id'], 'error_type': 'InvalidJudgeVerdict'}


def summarize(records):
    if not records:
        raise ValueError('Cannot summarize an empty evaluation')
    def metrics(values):
        return {'count': len(values), 'sr_pct': 100 * sum(r['score']['final'] for r in values) / len(values),
                'ps_pct': 100 * sum(r['score']['process'] for r in values) / len(values)}
    domains = {}
    for record in records:
        domain = (record.get('tags', {}).get('domain') or {}).get('d1') or 'unknown'
        domains.setdefault(str(domain), []).append(record)
    return {**metrics(records), 'by_domain': {key: metrics(value) for key, value in sorted(domains.items())},
            'anomaly_count': sum(r['metadata']['verdict']['anomaly_flag'] for r in records),
            'metric': 'BrowseComp-V3 official SR/PS rules on the project subset',
            'anomaly_policy': 'Flags do not change final answer correctness.'}


def read_judgments(path):
    """Preserve and recover only a malformed final line after interruption."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    records, offset = [], 0
    for index, line in enumerate(lines):
        try:
            if line.strip():
                records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError) as exc:
            if index != len(lines) - 1:
                raise ValueError('Malformed evaluation before the final line') from exc
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='eval.interrupted-tail.', delete=False) as backup:
                backup.write(line)
            with path.open('r+b') as stream:
                stream.truncate(offset)
            return records
        offset += len(line)
    if lines and not lines[-1].endswith(b'\n'):
        with path.open('ab') as stream:
            stream.write(b'\n')
    return records


def score_export(samples, manifest, output, args, key):
    output = Path(output)
    policy = {'official_revision': OFFICIAL_SOURCE['revision'], 'adapter': ADAPTER_VERSION,
              'adapter_sha256': digest(__file__), 'vendor_sha256': manifest['vendor_sha256'],
              'judge_model': args.judge_model, 'base_url': args.base_url,
              'judge_thinking': args.judge_thinking, 'judge_max_tokens': args.judge_max_tokens,
              'response_format': args.response_format, 'prompt_sha256': fingerprint(PROMPT_TEMPLATE),
              'schema_sha256': fingerprint(RESPONSE_FORMAT),
              'failure_policy': 'Judge failures remain unscored and resumable, never zero.'}
    config = {'source_run_id': manifest['source_run_id'], 'manifest_sha256': fingerprint(manifest),
              'policy': policy, 'policy_id': fingerprint(policy)}
    path = output / 'eval.jsonl'
    expected = {s['execution']['q_id']: s for s in samples}
    with output_lock(output):
        config_path = output / 'judge_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Judge configuration changed; use a new export/output directory')
        write_json_atomic(config_path, config)
        previous = {}
        if path.exists():
            for record in read_judgments(path):
                qid = record['q_id']
                if (qid not in expected or qid in previous or record.get('policy_id') != config['policy_id']
                        or record.get('sample_sha256') != fingerprint(expected[qid])):
                    raise ValueError('Saved evaluation contains conflicting records')
                verdict = validate_verdict(record['metadata']['verdict'], expected[qid])
                if record['score'] != {'final': float(verdict['answer_correct']), 'process': float(verdict['process_score'])}:
                    raise ValueError('Saved scores differ from the judge verdict')
                previous[qid] = record
        pending = [s for qid, s in expected.items() if qid not in previous]
        client = APIClient(key, args.base_url)
        failures = []
        with path.open('a', encoding='utf-8') as stream:
            for record in parallel_results(lambda s: evaluate_sample(s, client, args, config['policy_id']), pending, args.workers, client.stop):
                if record.get('error_type'):
                    failures.append(record)
                    print(json.dumps(record), flush=True)
                    continue
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                previous[record['q_id']] = record
                print(json.dumps({'evaluated': len(previous), 'total': len(samples), 'q_id': record['q_id'],
                                  'final': record['score']['final'], 'process': record['score']['process']}), flush=True)
        missing = sorted(set(expected) - set(previous))
        write_json_atomic(output / 'completion.json', {'expected': len(samples), 'recorded': len(previous),
                                                      'missing_ids': missing, 'complete': not missing})
        if missing:
            (output / 'summary.json').unlink(missing_ok=True)
            write_json_atomic(output / 'judge_errors.json', {'missing_ids': missing, 'failures': failures})
            raise SystemExit('Official-rule judging incomplete. Fix API/response errors and rerun the same command; saved scores are retained.')
        (output / 'judge_errors.json').unlink(missing_ok=True)
        summary = summarize([previous[qid] for qid in expected])
        summary.update(policy_id=config['policy_id'], official_revision=OFFICIAL_SOURCE['revision'],
                       source_run_id=manifest['source_run_id'], scope=manifest['scope'])
        write_json_atomic(output / 'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=59)
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--judge-model', default='gpt-4o')
    parser.add_argument('--base-url', default='https://api.openai.com/v1')
    parser.add_argument('--key-env', default='OPENAI_API_KEY')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--judge-thinking', choices=['provider-default', 'enabled', 'disabled'], default='provider-default')
    parser.add_argument('--judge-max-tokens', type=int, default=4096)
    parser.add_argument('--response-format', choices=['json_schema', 'json_object'], default='json_schema')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.expected_count < 1 or args.judge_max_tokens < 1 or not 1 <= args.workers <= 64:
        parser.error('Counts and output budget must be positive; workers must be 1..64')
    try:
        run = json.loads((args.run / 'run.json').read_text())
        tasks = args.tasks or Path(run['tasks_file'])
        references = args.references or tasks.resolve().parent / 'references.jsonl'
        if not args.export_only:
            load_env(args.env_file)
            if not os.environ.get(args.key_env):
                raise ValueError(f'Missing judge API key in {args.key_env}')
        samples, manifest = prepare_export(args.run, tasks, references, args.output, args.expected_count)
        if args.export_only:
            print(json.dumps({'output': str(args.output.resolve()), 'count': len(samples),
                              'final_answers': manifest['complete_final_answers'],
                              'subgoal_traces': manifest['samples_with_extracted_subgoals'],
                              'official_revision': OFFICIAL_SOURCE['revision'], 'api_requests': 0}, ensure_ascii=False, indent=2))
            return
        score_export(samples, manifest, args.output, args, os.environ[args.key_env])
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
