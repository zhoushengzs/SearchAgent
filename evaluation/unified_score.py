"""Score four evaluation conditions with a fixed judge and shared verdict cache."""
import argparse
import json
import os
from pathlib import Path
import time

from evaluation.common import load_tasks, read_predictions, write_json_atomic
from evaluation.prepare import digest
from evaluation.remote import APIClient, RemoteAPIError, output_lock
from evaluation.run import fingerprint, load_env
from evaluation.score import JUDGE_PROMPT, recover_scores, score_summary
from evaluation.unified_cache import memo
from evaluation.unified_common import CONDITIONS,LABELS


def validate_verdict(body):
    choice=body['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Unfinished judge response')
    raw=json.loads(choice['message']['content'])
    if type(raw.get('correct')) is not bool or not isinstance(raw.get('reason'),str):
        raise ValueError('Invalid judge verdict')
    return {'correct':raw['correct'],'reason':raw['reason'],'returned_judge_model':body.get('model'),
            'judge_usage':body.get('usage'),'judge_request_id':body.get('id')}


def render_results(output,summaries):
    lines=['# 统一6工具：四组全新评测结果','',
           '所有条件使用同一题单、共享文字和图片处理配置、生成设置及评分规则。工具组使用同一原生Agent循环；直接回答组没有工具，只进行一次回答。','',
           '| 条件 | 正确 / 总题数 | 准确率 | 提交最终答案 |','|---|---:|---:|---:|']
    for name in CONDITIONS:
        s=summaries[name]
        lines.append(f"| {LABELS[name]} | {s['correct']}/{s['count']} | {s['micro_accuracy_pct']:.2f}% | {s['answered']} |")
    lines+=['','同问题、相同参考答案、相同模型最终答案只调用一次裁判并共享判断。未完成作答记错；评分接口错误不记为模型答错。',
            '','','这些是项目参考答案评分，不是各公开题库的官方成绩。']
    (output/'.codex').mkdir(exist_ok=True)
    (output/'.codex/RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_json_atomic(output/'summary.json',{'schema':'unified-six-tool-four-way-v1','conditions':summaries})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['output','tasks','references','env-file']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--key-env',default='JUDGE_API_KEY')
    a=p.parse_args();load_env(a.env_file)
    plan=json.loads((a.output/'plan.json').read_text());judge=plan['scoring']
    if judge['base_url']!='https://api.deepseek.com/v1':raise ValueError('Unexpected judge recipient')
    key=os.environ.get(a.key_env)
    if not key:raise ValueError('Existing judge credential unavailable')
    tasks={t['id']:t for t in load_tasks(a.tasks,778)}
    refs={r['id']:r for r in (json.loads(l) for l in a.references.read_text().splitlines() if l.strip())}
    if set(refs)!=set(tasks):raise ValueError('Detached references do not match the task set')
    policy={'metric':'project_reference_judge_accuracy_not_official_benchmark_score',
            'judge_model':judge['judge_model'],'base_url':judge['base_url'],'temperature':0,
            'judge_thinking':'disabled','judge_max_tokens':4096,'prompt':JUDGE_PROMPT,
            'answer_extraction':'native_answered_status','references_sha256':digest(a.references),
            'tasks_sha256':digest(a.tasks),'judge_cache':'reuse identical complete request across conditions',
            'code_sha256':{n:digest(Path(__file__).parent/n) for n in ['unified_score.py','unified_cache.py','score.py','remote.py']}}
    policy_id=fingerprint(policy);client=APIClient(key,judge['base_url']);summaries={}
    for condition in CONDITIONS:
        root=a.output/condition;run=json.loads((root/'full/merged/run.json').read_text())
        if set(run['protocol']['selected_ids'])!=set(tasks) or run['protocol']['tasks_sha256']!=digest(a.tasks):
            raise ValueError('A condition has a different task set')
        predictions=read_predictions(root/'full/merged/predictions.jsonl',run['run_id'],tasks)
        if set(predictions)!=set(tasks) or any(v.get('infrastructure_error') for v in predictions.values()):
            raise ValueError('Inference incomplete; do not score a partial condition')
        directory=root/'scores';directory.mkdir(exist_ok=True)
        with output_lock(directory):
            config={'run_id':run['run_id'],'policy_id':policy_id,'policy':policy}
            path=directory/'score_config.json'
            if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Judge policy changed')
            write_json_atomic(path,config)
            saved={r['id']:r for r in recover_scores(directory/'scores.jsonl')}
            for task_id,row in saved.items():
                if task_id not in tasks or row['policy_id']!=policy_id or row['prediction_sha256']!=fingerprint(predictions[task_id]):
                    raise ValueError('Saved score identity changed')
            for task_id in run['protocol']['selected_ids']:
                if task_id in saved:continue
                pred=predictions[task_id];ref=refs[task_id]
                if ref['input_fingerprint']!=pred['input_fingerprint']:raise ValueError('Reference identity differs')
                candidate=pred.get('answer','').strip() if pred['status']=='answered' else ''
                if not candidate:
                    verdict={'correct':False,'reason':'Native agent did not submit a final answer.',
                             'judge_cache_hit':False,'returned_judge_model':None}
                else:
                    payload={'question':tasks[task_id]['model_input']['question'],'candidate':candidate,'reference':ref}
                    request={'model':judge['judge_model'],'temperature':0,'max_tokens':4096,
                             'thinking':{'type':'disabled'},'response_format':{'type':'json_object'},
                             'messages':[{'role':'system','content':JUDGE_PROMPT},
                                         {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]}
                    def fetch():
                        for attempt in range(3):
                            body=client.post(request)
                            try:return validate_verdict(body)
                            except (ValueError,KeyError,TypeError,IndexError):
                                write_json_atomic(directory/(task_id+f'.judge-invalid-{attempt}.json'),
                                                  {k:body.get(k) for k in ['id','model','choices','usage']})
                        raise ValueError('Judge did not return a complete valid verdict')
                    try:
                        verdict,hit,cache_key=memo(a.output/'judge_cache','verdict',{'policy_id':policy_id,'request':request},fetch)
                        verdict={**verdict,'judge_cache_hit':hit,'judge_cache_key':cache_key}
                    except (ValueError,RemoteAPIError) as exc:
                        write_json_atomic(directory/'judge_errors.json',{'id':task_id,'error_type':type(exc).__name__})
                        raise SystemExit('Judge incomplete; preserve saved scores and resume the same command') from None
                row={'id':task_id,'source_dataset':pred['source_dataset'],'input_fingerprint':pred['input_fingerprint'],
                     'prediction_sha256':fingerprint(pred),'policy_id':policy_id,'score':float(verdict['correct']),
                     'status':pred['status'],'tool_calls':pred['tool_calls'],**verdict}
                with (directory/'scores.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                saved[task_id]=row
                print(json.dumps({'condition':condition,'scored':len(saved),'total':778,'id':task_id,'score':row['score']}),flush=True)
            summary=score_summary(list(saved.values()))
            summary.update(correct=int(sum(r['score'] for r in saved.values())),run_id=run['run_id'],policy_id=policy_id,
                           answered=sum(p['status']=='answered' for p in predictions.values()))
            write_json_atomic(directory/'summary.json',summary)
            (directory/'judge_errors.json').unlink(missing_ok=True)
            summaries[condition]=summary
    render_results(a.output,summaries)
    write_json_atomic(a.output/'SERVER_SUCCESS.json',{'count_each':778,'conditions':4,'policy_id':policy_id,
                                                    'completed_at':time.time()})
    print('UNIFIED_FOUR_CONDITION_SCORING_SUCCESS',flush=True)


if __name__=='__main__':main()
