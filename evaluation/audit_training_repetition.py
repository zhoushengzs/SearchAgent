"""Quarantine candidates for review without rewriting the training corpus."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from evaluation.common import write_json_atomic
from evaluation.prepare import digest


def action(text):
    match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.S)
    if match:
        try:
            return json.dumps(json.loads(match.group(1)), sort_keys=True, ensure_ascii=False)
        except ValueError:
            pass
    return None


def audit(source, output):
    counts, types, trajectories = Counter(), Counter(), set()
    with source.open() as stream, (output/'review_candidates.jsonl').open('w') as rejected:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts['rows'] += 1
            messages = row['conversations']
            answers = [m['value'] for m in messages if m['from'] == 'gpt']
            if len(answers) < 2:
                continue
            counts['rows_with_history'] += 1
            exact = answers[-1] == answers[-2]
            final, prior = action(answers[-1]), action(answers[-2])
            repeated = final is not None and final == prior
            counts['exact_answer_repeated'] += int(exact)
            counts['same_tool_and_arguments'] += int(repeated)
            if not repeated and not exact:
                continue
            observation = next((m for m in reversed(messages[:-1]) if m['from']=='observation'), None)
            try:
                result = json.loads(observation['value']).get('result', {}) if observation else {}
            except ValueError:
                result = {}
            status = result.get('status', 'missing')
            counts['previous_result_' + status] += 1
            if status != 'ok':
                continue
            counts['review_candidates_after_success'] += 1
            types[row.get('sample_type', 'unknown')] += 1
            trajectory = row.get('trajectory_id')
            if trajectory:
                trajectories.add(trajectory)
            rejected.write(json.dumps({'id':row['id'], 'line':line_number,
                'trajectory_id':trajectory, 'step':row.get('step'),
                'sample_type':row.get('sample_type'), 'previous_tool_status':status,
                'exact_answer_repeated':exact, 'same_tool_and_arguments':repeated,
                'tool_call':json.loads(final) if final else None,
                'decision':'review_required_not_automatically_bad'}, ensure_ascii=False)+'\n')
    summary = {'source':str(source.resolve()), 'source_sha256':digest(source),
               'counts':dict(counts), 'candidate_sample_types':dict(types),
               'candidate_trajectories':len(trajectories),
               'limitations':['Tool status ok does not prove sufficient or correct evidence.',
                              'Candidates are not automatically removed or repaired.',
                              'Review full trajectories before using a filtered corpus.'],
               'supervision':{'mask_history':True,'train_on_prompt':False,'target':'last assistant turn'}}
    write_json_atomic(output/'summary.json', summary)
    write_json_atomic(output/'candidate_trajectory_ids.json', sorted(trajectories))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    print(json.dumps(audit(a.input,a.output),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
