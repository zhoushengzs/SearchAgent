"""Resume MM-BrowseComp official prompt/parser judging on saved model outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

from evaluation.common import write_json_atomic
from evaluation.prepare import digest
from evaluation.remote import APIClient, RemoteAPIError, output_lock, parallel_results
from evaluation.run import load_env

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.codex/official_eval/MM-BrowseComp/src/eval_safe.py'


def official():
    spec=importlib.util.spec_from_file_location('mm_official_judge',SOURCE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(row, answer, reference, judge):
    d=row['evaluation_details']
    if (row['id']!=answer['id'] or row['generated_answer_text']!=answer['generated_answer_text']
            or row['reference_answer_text']!=reference['answer']
            or row['reference_checklist_items']!=reference['checklist']
            or row['judge_model_name']!=judge or d.get('parsing_error')
            or d['parsed_overall_correctness'] not in ('YES','NO')):
        raise ValueError('MM judge answer/reference/response identity differs')
    n,c,v=d['parsed_checklist_total'],d['parsed_checklist_correct'],d['checklist_result_vector']
    if (type(n) is not int or n!=len(reference['checklist']) or n<1 or type(c) is not int
            or not 0<=c<=n or not isinstance(v,list) or len(v)!=n
            or any(type(x) is not int or x not in (0,1) for x in v) or sum(v)!=c
            or abs(d['calculated_checklist_score']-c/n)>1e-8):
        raise ValueError('MM judge checklist is inconsistent')
    return row


def judge_one(answer, reference, args, client, evaluator):
    question=reference['question'].split('Question: ')[-1]
    prompt=evaluator.construct_judge_prompt(question,answer['generated_answer_text'],reference['answer'],reference['checklist'])
    for attempt in range(3):
        try:
            body=client.post({'model':args.judge,'messages':[{'role':'user','content':prompt}],'max_tokens':5120})
            choice=body['choices'][0]
            if choice.get('finish_reason')!='stop':
                raise ValueError('Judge response was truncated')
            text=choice['message']['content']
            parsed=evaluator.parse_judge_response(text)
            c,n=parsed.get('checklist_correct_count'),parsed.get('checklist_total_count')
            row={'id':answer['id'],'question':question,'generated_answer_text':answer['generated_answer_text'],
                 'reference_answer_text':reference['answer'],'reference_checklist_items':reference['checklist'],
                 'judge_model_name':args.judge,'evaluation_details':{
                     'raw_judge_response':text,'parsed_overall_correctness':parsed.get('overall_correctness'),
                     'parsing_error':parsed.get('error'),'parsed_checklist_correct':c,'parsed_checklist_total':n,
                     'checklist_result_vector':parsed.get('checklist_result_vector'),
                     'calculated_checklist_score':c/n if type(c) is int and type(n) is int and n>0 else None},
                 'timestamp':datetime.now(timezone.utc).isoformat(),'judge_api_metadata':{
                     'returned_model':body.get('model'),'usage':body.get('usage'),'request_id':body.get('id'),
                     'base_url':args.base_url,'official_prompt_parser_sha256':digest(SOURCE)}}
            return validate(row,answer,reference,args.judge)
        except RemoteAPIError as exc:
            return {'id':answer['id'],'error_type':type(exc).__name__}
        except (ValueError,KeyError,TypeError,IndexError):
            if attempt==2:
                return {'id':answer['id'],'error_type':'InvalidJudgeResponse'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--judge',default='deepseek/deepseek-v4.1-flash-1017')
    p.add_argument('--base-url',default='https://openrouter.ai/api/v1')
    p.add_argument('--key-env',default='OPENROUTER_API_KEY')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    load_env(ROOT/'.env')
    if not os.environ.get(args.key_env):raise SystemExit('Missing judge credential: '+args.key_env)
    output=args.output.resolve()
    config=json.loads((output/'trial_config.json').read_text())
    answers=[json.loads(s) for s in (output/'answers.jsonl').read_text().splitlines() if s.strip()]
    references={r['id']:r for r in map(json.loads,(output/'judge_dataset.jsonl').read_text().splitlines())}
    if len(answers)!=len({r['id'] for r in answers}) or {r['id'] for r in answers}!=set(config['selected_ids']):
        raise SystemExit('Model outputs are incomplete; finish inference first')
    if config['judge']!=args.judge or config['judge_base_url']!=args.base_url:
        raise SystemExit('Judge differs from the original paused batch')
    evaluator=official()
    scores=output/'scores';scores.mkdir(exist_ok=True)
    with output_lock(output):
        policy={'judge':args.judge,'base_url':args.base_url,'max_tokens':5120,
                'answers_sha256':digest(output/'answers.jsonl'),'references_sha256':digest(output/'judge_dataset.jsonl'),
                'official_prompt_parser_sha256':digest(SOURCE),'adapter_sha256':digest(__file__)}
        policy_path=output/'judging_resume_config.json'
        if policy_path.exists() and json.loads(policy_path.read_text())!=policy:
            raise SystemExit('Saved judge configuration differs')
        write_json_atomic(policy_path,policy)
        pending=[];completed={};errors=[]
        for answer in answers:
            path=scores/(answer['id']+'_eval.json')
            if path.exists():
                completed[answer['id']]=validate(json.loads(path.read_text()),answer,references[answer['id']],args.judge)
            else:pending.append(answer)
        print(json.dumps({'retained_scores':len(completed),'to_judge':len(pending)}),flush=True)
        client=APIClient(os.environ[args.key_env],args.base_url)
        for row in parallel_results(lambda a:judge_one(a,references[a['id']],args,client,evaluator),pending,args.workers,client.stop):
            if row.get('error_type'):
                errors.append(row);print(json.dumps(row),flush=True);continue
            write_json_atomic(scores/(row['id']+'_eval.json'),row)
            completed[row['id']]=row
            print(json.dumps({'id':row['id'],'judged':len(completed),'total':len(answers)}),flush=True)
        missing=sorted(set(config['selected_ids'])-set(completed))
        write_json_atomic(output/'judge_completion.json',{'complete':not missing,'recorded':len(completed),
                          'expected':len(answers),'missing_ids':missing,'failures':errors})
        if missing:raise SystemExit('Judging incomplete; rerun the same command after fixing the reported errors')
        values=list(completed.values());n=len(values)
        oa=sum(r['evaluation_details']['parsed_overall_correctness']=='YES' for r in values)
        sa=sum(r['evaluation_details']['parsed_overall_correctness']=='YES' and
               r['evaluation_details']['parsed_checklist_correct']==r['evaluation_details']['parsed_checklist_total'] for r in values)
        summary={'total_items_evaluated':n,'overall_correct_count':oa,'overall_accuracy_percent':100*oa/n,
                 'strict_correct_count':sa,'strict_accuracy_percent':100*sa/n,
                 'average_checklist_score_percent':100*sum(r['evaluation_details']['calculated_checklist_score'] for r in values)/n,
                 'items_with_parsing_errors':0,'failed_task_count':0,
                 'scope':'All saved judgments, including those completed before the pause.'}
        write_json_atomic(scores/'aggregate_metrics.json',summary)
        write_json_atomic(scores/'all_results_summary.json',values)
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
