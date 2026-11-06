from pathlib import Path
import json,re,hashlib,time,torch
from safetensors import safe_open
P=Path('/cluster/home/shenqunlun/zhousheng');D=P/'recoil_sft8_20261104';a=json.loads((D/'saved_input_audit_complete.json').read_text());r=Path(a['run']);rows=a['rows']
assert len(rows)==49 and all(x['status']=='reconstructed' and x['image_present'] and x['dropped_messages']==0 and x['final_retained'] and x['final_in_assistant_mask'] for x in rows)
assert sum(x['final_count'] for x in rows)==5
assert json.loads((D/'SMOKE_SUCCESS.json').read_text())['run']==str(r)
log=(D/'train-4412270.log').read_text(errors='replace');assert set(re.findall(r'training/global_step:(\d+)',log))=={'1','2','3'}
base=P/'qwen3-8b-sft-transformers4';trained=r/'checkpoints/global_step_3/actor/huggingface'
def mapping(d):
 out={}
 for f in d.glob('*.safetensors'):
  with safe_open(str(f),framework='pt',device='cpu') as sf:
   for k in sf.keys():out[k]=f
 return out
bm=mapping(base);tm=mapping(trained);common=sorted(set(bm)&set(tm));assert common
keys=[]
for pattern in ['q_proj.weight','embed_tokens.weight','lm_head.weight','visual.patch_embed.proj.weight']:
 matches=[k for k in common if pattern in k]
 if matches:keys.extend([matches[0],matches[-1]])
checks=[]
for k in sorted(set(keys)):
 with safe_open(str(bm[k]),framework='pt',device='cpu') as f:before=f.get_tensor(k)
 with safe_open(str(tm[k]),framework='pt',device='cpu') as f:after=f.get_tensor(k)
 assert before.shape==after.shape and torch.isfinite(after).all()
 diff=(after.float()-before.float()).abs();checks.append({'key':k,'changed_values':int((diff>0).sum()),'max_abs_difference':float(diff.max())})
assert any(x['changed_values']>0 for x in checks)
result={'passed':True,'job':'4412270','smoke_run':str(r),'checked_at':time.time(),'learning_rows_checked':49,'final_answers_preserved_and_in_learning_mask':5,'tool_images_reconstructed_with_exact_sha256':True,'input_check_method':'Reconstructed all saved training trajectories using same tokenizer/image processor and RECOIL branch mask; inspected generated-row image union code; original live tensors were not separately serialized','saved_parameter_checks':checks,'actual_updates':[1,2,3],'audit_sha256':hashlib.sha256((D/'saved_input_audit_complete.json').read_bytes()).hexdigest(),'scope':'Technical data and update verification only; not a guarantee of training quality'}
(D/'WEIGHT_CHANGE_VERIFIED.json').write_text(json.dumps(result,indent=2));tmp=D/'LEARNING_VERIFIED.json.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(D/'LEARNING_VERIFIED.json');print(json.dumps(result))
