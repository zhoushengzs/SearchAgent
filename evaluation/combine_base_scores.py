"""Audit and collect the six Base benchmark results, preserving their protocols."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.browsecomp_v3_official import validate_verdict
from evaluation.common import load_tasks, write_json_atomic
from evaluation.prepare import digest
from evaluation.run import fingerprint


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def unique(rows):
    by_id = {row['id']: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError('Duplicate task IDs')
    return by_id


def combine(tasks_path, references_path, generic, browsecomp, mm, output):
    tasks = unique(load_tasks(tasks_path))
    refs = unique(read_rows(references_path))
    generic, browsecomp, mm, output = map(Path, [generic, browsecomp, mm, output])
    scores = unique(read_rows(generic / 'scores.jsonl'))
    generic_config = read_json(generic / 'score_config.json')
    if set(scores) != set(tasks):
        raise ValueError('Prior Base scores do not cover the fixed task bundle')
    if generic_config['policy']['references_sha256'] != digest(references_path):
        raise ValueError('Prior reference file differs')
    for key, row in scores.items():
        if (row['input_fingerprint'] != tasks[key]['input_fingerprint']
                or row['source_dataset'] != tasks[key]['source_dataset']
                or row['policy_id'] != generic_config['policy_id']
                or row['score'] not in (0, 1) or bool(row['score']) != row['correct']):
            raise ValueError('Prior score identity or correctness differs: ' + key)

    bc_manifest = read_json(browsecomp / 'manifest.json')
    bc_config = read_json(browsecomp / 'judge_config.json')
    bc_completion = read_json(browsecomp / 'completion.json')
    bc_rows = read_rows(browsecomp / 'eval.jsonl')
    bc_expected = {k for k, task in tasks.items() if task['source_dataset'] == 'browsecomp_v3'}
    if (bc_manifest['source_run_id'] != generic_config['run_id'] or not bc_completion['complete']
            or bc_manifest['tasks_sha256'] != digest(tasks_path)
            or bc_manifest['references_sha256'] != digest(references_path)):
        raise ValueError('BrowseComp-V3 source or completion differs')
    bc_by_id = {}
    for row in bc_rows:
        sample = read_json(browsecomp / 'samples' / (row['q_id'] + '.json'))
        key = sample['adapter_provenance']['task_id']
        if key not in bc_expected or key in bc_by_id:
            raise ValueError('Unexpected or repeated BrowseComp-V3 task')
        if (row['sample_sha256'] != fingerprint(sample)
                or row['sample_sha256'] != bc_manifest['sample_sha256'][row['q_id']]
                or row['policy_id'] != bc_config['policy_id']
                or sample['adapter_provenance']['input_fingerprint'] != tasks[key]['input_fingerprint']):
            raise ValueError('BrowseComp-V3 sample identity differs')
        v = validate_verdict(row['metadata']['verdict'], sample)
        if row['score'] != {'final': float(v['answer_correct']), 'process': float(v['process_score'])}:
            raise ValueError('BrowseComp-V3 score differs from verdict')
        bc_by_id[key] = row
    if set(bc_by_id) != bc_expected:
        raise ValueError('BrowseComp-V3 judgments incomplete')

    mm_config = read_json(mm / 'trial_config.json')
    mm_answers = unique(read_rows(mm / 'answers.jsonl'))
    mm_expected = {k for k, task in tasks.items() if task['source_dataset'] == 'mmbrowsecomp'}
    if (set(mm_config['selected_ids']) != mm_expected or set(mm_answers) != mm_expected
            or mm_config['tasks_sha256'] != digest(tasks_path)
            or mm_config['references_sha256'] != digest(references_path)):
        raise ValueError('MM-BrowseComp inference is incomplete or uses different tasks')
    mm_results = {}
    for key, answer in mm_answers.items():
        row = read_json(mm / 'scores' / (key + '_eval.json'))
        details = row['evaluation_details']
        checklist = refs[key]['annotations']['checklist']
        if (row['id'] != key or row['question'] != tasks[key]['model_input']['question'].split('Question: ')[-1]
                or answer['question'] != tasks[key]['model_input']['question']
                or row['generated_answer_text'] != answer['generated_answer_text']
                or row['reference_answer_text'] != refs[key]['answer']
                or row['reference_checklist_items'] != checklist
                or details.get('parsing_error')
                or details['parsed_overall_correctness'] not in ('YES', 'NO')):
            raise ValueError('MM-BrowseComp answer/judgment differs: ' + key)
        n, c = details['parsed_checklist_total'], details['parsed_checklist_correct']
        vector = details['checklist_result_vector']
        if (type(n) is not int or n != len(checklist) or n < 1 or type(c) is not int or not 0 <= c <= n
                or not isinstance(vector, list) or len(vector) != n
                or any(type(v) is not int or v not in (0, 1) for v in vector)
                or sum(vector) != c or abs(details['calculated_checklist_score'] - c/n) > 1e-8):
            raise ValueError('Invalid MM-BrowseComp checklist: ' + key)
        mm_results[key] = row

    table, combined = [], []
    for name in sorted({task['source_dataset'] for task in tasks.values()}):
        ids = [key for key, task in tasks.items() if task['source_dataset'] == name]
        if name == 'browsecomp_v3':
            correct = sum(bc_by_id[key]['score']['final'] for key in ids)
            extra = {'ps_pct': 100 * sum(bc_by_id[key]['score']['process'] for key in ids) / len(ids)}
            metric, source = 'SR', browsecomp / 'eval.jsonl'
            judge = bc_config['policy']['judge_model']
            inference = 'Original project agent; 10 decision steps'
            values = {key: bc_by_id[key]['score']['final'] for key in ids}
        elif name == 'mmbrowsecomp':
            correct = sum(mm_results[key]['evaluation_details']['parsed_overall_correctness'] == 'YES' for key in ids)
            strict = sum(mm_results[key]['evaluation_details']['parsed_overall_correctness'] == 'YES'
                         and mm_results[key]['evaluation_details']['parsed_checklist_correct'] ==
                         mm_results[key]['evaluation_details']['parsed_checklist_total'] for key in ids)
            extra = {'sa_pct': 100 * strict / len(ids),
                     'checklist_pct': 100 * sum(mm_results[key]['evaluation_details']['calculated_checklist_score'] for key in ids) / len(ids),
                     'truncated_count': sum(mm_answers[key].get('generation_status') == 'output_truncated' for key in ids)}
            metric, source = 'OA', mm / 'scores'
            judge = sorted({mm_results[key]['judge_model_name'] for key in ids})
            inference = f"Reconstructed Agent-R1 workflow; {mm_config['max_turns']} decision steps"
            values = {key: float(mm_results[key]['evaluation_details']['parsed_overall_correctness'] == 'YES') for key in ids}
        else:
            correct = sum(scores[key]['score'] for key in ids)
            extra = {}
            metric, source = 'Project reference-judge accuracy', generic / 'scores.jsonl'
            judge = generic_config['policy']['judge_model']
            inference = 'Original project agent; 10 decision steps'
            values = {key: scores[key]['score'] for key in ids}
        table.append({'benchmark': name, 'count': len(ids), 'correct': int(correct),
                      'metric': metric, 'score_pct': 100 * correct / len(ids),
                      **extra, 'judge': judge, 'inference_protocol': inference, 'source': str(source.resolve())})
        combined.extend({'id': key, 'source_dataset': name, 'input_fingerprint': tasks[key]['input_fingerprint'],
                         'primary_correct': bool(values[key]), 'metric': metric, 'source': str(source.resolve())} for key in ids)
    report = {'model': mm_config['model'], 'count': len(tasks), 'complete': True,
              'datasets': table, 'tasks_sha256': digest(tasks_path), 'references_sha256': digest(references_path),
              'source_files_sha256': {str(p.resolve()): digest(p) for p in [generic/'scores.jsonl',
                  browsecomp/'eval.jsonl', mm/'answers.jsonl', *sorted((mm/'scores').glob('*_eval.json'))]},
              'scope': 'Per-benchmark Base scores. MM-BrowseComp uses a different agent/tool budget. No unified comparable overall accuracy is asserted.',
              'notes': ['Other four benchmarks retain their prior project reference-judge scores, not official benchmark evaluator scores.',
                        'BrowseComp-V3 uses official SR/PS rules with substituted DeepSeek judge.',
                        'MM-BrowseComp uses official OA/SA/checklist rules with substituted DeepSeek judge.',
                        'Use identical inference and judging protocols per benchmark for subsequent SFT comparison.']}
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists() and read_json(output/'summary.json') != report:
        raise ValueError('Existing combined report differs; choose another output')
    write_json_atomic(output/'summary.json', report)
    with (output/'scores.jsonl').open('w', encoding='utf-8') as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    fields = ['benchmark','count','correct','metric','score_pct','ps_pct','sa_pct','checklist_pct','truncated_count','inference_protocol','source']
    with (output/'benchmark_scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(table)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--generic-scores', type=Path, required=True)
    parser.add_argument('--browsecomp-scores', type=Path, required=True)
    parser.add_argument('--mmbrowsecomp', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    report=combine(args.tasks,args.references,args.generic_scores,args.browsecomp_scores,args.mmbrowsecomp,args.output)
    print(json.dumps({'output':str(args.output.resolve()),'count':report['count'],'datasets':report['datasets']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
