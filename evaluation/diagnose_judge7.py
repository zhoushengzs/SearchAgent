import json, time, concurrent.futures, fcntl
from pathlib import Path
from evaluation.server_benchmark import runtime_environment, ROOT, BUNDLE
from evaluation.remote import APIClient
out=ROOT/'outputs/judge7-diagnosis'
out.mkdir(exist_ok=True)
lock=(out/'.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
config=json.loads((ROOT/'outputs/base165-778-score/score_config.json').read_text())
p=config['policy']
ids=json.loads((ROOT/'outputs/base165-778-score/judge_errors.json').read_text())['missing_ids']
def rows(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
tasks={r['id']:r for r in rows(BUNDLE/'tasks.jsonl')}
refs={r['id']:r for r in rows(BUNDLE/'references.jsonl')}
preds={r['id']:r for r in rows(ROOT/'outputs/base165-778/merged/predictions.jsonl')}
env,key=runtime_environment()
client=APIClient(env[key],p['base_url'],attempts=1,timeout=120)
def check(i):
 payload={'question':tasks[i]['model_input']['question'],'candidate':preds[i]['answer'],'reference':refs[i]}
 request={'model':p['judge_model'],'temperature':p['temperature'],'max_tokens':p['judge_max_tokens'],'response_format':{'type':'json_object'},'messages':[{'role':'system','content':p['prompt']},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]}
 if p['judge_thinking']!='provider-default': request['thinking']={'type':p['judge_thinking']}
 r={'id':i,'policy_id':config['policy_id'],'time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 try:
  body=client.post(request); choice=body.get('choices',[{}])[0]; msg=choice.get('message',{}); content=msg.get('content')
  r.update(finish_reason=choice.get('finish_reason'),content_chars=len(content or ''),reasoning_chars=len(msg.get('reasoning_content') or ''),usage=body.get('usage'),returned_model=body.get('model'))
  try:
   v=json.loads(content); r.update(json_valid=True,correct_is_bool=isinstance(v,dict) and type(v.get('correct')) is bool,reason_is_text=isinstance(v,dict) and isinstance(v.get('reason'),str))
  except (ValueError,TypeError): r['json_valid']=False
  r['valid']=r['finish_reason']=='stop' and r.get('correct_is_bool',False) and r.get('reason_is_text',False)
 except Exception as e: r['error_type']=type(e).__name__
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for r in pool.map(check,ids):
  (out/(r['id']+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2))
  print(json.dumps(r),flush=True)
