from pathlib import Path
import json,re,sys,os
P=Path('/cluster/home/shenqunlun/zhousheng');S=P/'SearchAgent-recoil-20261104';R=S/'datasets/grpo/runs/priority_sft8_smoke/smoke8_20261104T074007Z'
sys.path.insert(0,str(S/'train/OpenSearch-VL-Training/RL/rllm'))
from transformers import AutoProcessor
from rllm.parser.chat_template_parser import QwenChatTemplateParser
from rllm.parser.mm_cumulative import tokenize_cumulative_mm
proc=AutoProcessor.from_pretrained(str(P/'qwen3-8b-sft-transformers4'),trust_remote_code=True)
parser=QwenChatTemplateParser(proc.tokenizer,processor=proc)
sys.path.insert(0,str(S/'datasets/grpo/integration'))
from project_runtime.views import materialize_view
V=P/'recoil_sft8_20261104/audit_views';V.mkdir(exist_ok=True)
rows=[]
for f in sorted(R.glob('episodes/episodes/train_step_*/episode_*.json')):
 e=json.loads(f.read_text())
 for t in e['trajectories']:
  if not t['steps']:continue
  ms=t['steps'][-1]['chat_completions'];r={'episode':e['episode_id'],'trajectory':t['name'],'step':e['training_step']}
  ims=[m for m in ms if m.get('images')]
  if not ims:r['status']='missing_images';rows.append(r);continue
  ims[0]['images']=e['task']['images']
  failed=False
  for j,m in enumerate(ims[1:]):
   try:
    data=json.loads(re.search(r'<tool_response>\s*(.*?)\s*</tool_response>',m['content'],re.S).group(1))['data']
    assert data['asset_id']=='image_1' and data['origin']=='initial_input'
    dest=V/(str(e['training_step'])+'_'+e['task_hash']+'_'+str(j)+'_'+t['name']+'.png')
    view=materialize_view(e['task']['images'][0],dest,data['view']['bbox'],max_side=1536)
    assert view['sha256']==data['view']['sha256'], 'tool image hash mismatch'
    m['images']=[str(dest)]
   except Exception as ex:r.update(status='image_reconstruction_error',error=str(ex));failed=True;break
  if failed:rows.append(r);continue
  try:
   tok=tokenize_cumulative_mm(parser,ms,max_prompt_length=4096,max_response_length=8192,cache={})
   mask=tok.response_mask.clone();ri=t.get('info',{}).get('recoil',{});mask[tok.turn_ids<int(ri.get('trainable_from') or 0)]=0
   if ri.get('mask_from') is not None:mask[tok.turn_ids>=int(ri['mask_from'])]=0
   txt=proc.tokenizer.decode(tok.response_ids);learn=proc.tokenizer.decode(tok.response_ids[mask.bool()])
   finals=[a for m in ms if m['role']=='assistant' for a in re.findall(r'<response>(.*?)</response>',m['content'],re.S) if a.strip()]
   r.update(status='reconstructed',image_present=bool(tok.multi_modal_inputs.get('pixel_values') is not None),dropped_messages=tok.dropped_messages,final_count=len(finals),final_retained=all(x in txt for x in finals),final_in_assistant_mask=all(x in learn for x in finals),trajectory_info=t.get('info'))
  except Exception as ex:r.update(status='error',error=str(ex))
  rows.append(r)
out={'run':str(R),'scope':'Independent saved trajectory reconstruction; including RECOIL branch-specific mask; not sufficient alone for final learning approval','rows':rows}
(P/'recoil_sft8_20261104/saved_input_audit_complete.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({'rows':len(rows),'status_counts':{s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)}}))
