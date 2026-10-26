#!/usr/bin/env python3
"""Independent streaming verification of source/clean/review/removed partitions."""
import argparse
import collections as C
import hashlib
import json
import re
from pathlib import Path

CALL=re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>',re.S)
THINK=re.compile(r'<think>(.*?)</think>',re.S)
RESPONSE=re.compile(r'<response>(.*?)</response>',re.S)
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def calls(text): return [json.loads(x) for x in CALL.findall(text)]
def hist(row):
    result=[]; pending=None
    for m in row['conversations'][1:-1]:
        if m['from']=='gpt':
            a=calls(m['value']); assert len(a)==1; pending=a[0]
        elif m['from']=='observation':
            assert pending is not None
            o=json.loads(m['value']); result.append((o['step'],pending,o)); pending=None
        else: raise AssertionError('unexpected role')
    assert pending is None
    return result

def state(row):
    text=row['conversations'][0]['value']; marker=text.find('Recorded task state follows.')
    if marker<0: return None
    return json.loads(text[text.find('\n{',marker)+1:])

def refs(obj,allowed):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k in ('depends_on','evidence_steps'):
                assert isinstance(v,list) and all(isinstance(n,int) and not isinstance(n,bool) and n in allowed for n in v)
            elif k in ('revision_of_step','observed_at_step'):
                assert v is None or (isinstance(v,int) and not isinstance(v,bool) and v in allowed)
            else: refs(v,allowed)
    elif isinstance(obj,list):
        for v in obj: refs(v,allowed)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--cleaned',type=Path,required=True); p.add_argument('--audit',type=Path,required=True)
    args=p.parse_args(); summary=json.loads((args.audit/'summary.json').read_text()); reports={}; retained_signatures={}
    with (args.audit/'row_manifest.jsonl').open() as mf:
        for split in ('train','val'):
            handles={'clean':(args.cleaned/(split+'.jsonl')).open('rb'),'review':(args.cleaned/'review'/(split+'.jsonl')).open('rb'),
                     'removed':(args.cleaned/'removed_targets'/(split+'.jsonl')).open('rb')}
            count=C.Counter(); source_hash=hashlib.sha256()
            with (args.source/(split+'.jsonl')).open('rb') as f:
                for ln,raw in enumerate(f,1):
                    source_hash.update(raw); m=json.loads(next(mf)); assert m['file']==split+'.jsonl' and m['line']==ln
                    assert hashlib.sha256(raw).hexdigest()==m['source_sha256']
                    original=json.loads(raw); routed=handles[m['route']].readline(); assert routed
                    count[m['route']]+=1
                    if m['route']!='clean':
                        assert routed==raw
                    else:
                        cleaned=json.loads(routed)
                        assert cleaned['id']==original['id'] and cleaned.get('step')==original.get('step')
                        assert cleaned.get('images')==original.get('images')
                        assert calls(cleaned['conversations'][-1]['value'])==calls(original['conversations'][-1]['value'])
                        before=hist(original); after=hist(cleaned); bystep={step:(a,o) for step,a,o in before}
                        retained={step for step,a,o in after}; ost=state(original); nst=state(cleaned)
                        if ost is not None:
                            assert ost['question']==nst['question']
                            assert ost['remaining_decisions']==nst['remaining_decisions']
                            assert ost['tool_configuration']==nst['tool_configuration']
                            assert nst['executed_prefix_step_ids']==[step for step,a,o in after]
                            placeholders=[int(x) for x in re.findall(r'img_(\d+): <image>',cleaned['conversations'][0]['value'].split('Recorded task state follows.')[0])]
                            assert placeholders==list(range(1,len(cleaned.get('images',[]))+1))
                            assert [im['image_index'] for im in nst.get('attached_images',[])]==placeholders
                        for i,(step,a,o) in enumerate(before):
                            if step not in retained:
                                assert i>0 and canon(before[i-1][1])==canon(a) and before[i-1][0]+1==step
                                count['deleted_contiguous_only']+=1
                        allowed={0}
                        for step,a,o in after:
                            oa,oo=bystep[step]
                            assert a==oa and o['result']==oo['result']
                            refs(o.get('observation'),allowed)
                            # The assistant directly preceding this observation.
                            am=next(x for j,x in enumerate(cleaned['conversations']) if j+1<len(cleaned['conversations']) and cleaned['conversations'][j+1]['from']=='observation' and json.loads(cleaned['conversations'][j+1]['value'])['step']==step)
                            for t in THINK.findall(am['value']):
                                if t.strip(): refs(json.loads(t),allowed)
                            allowed.add(step)
                            tid=cleaned.get('trajectory_id') or cleaned['id'].split(':')[0]; key=(tid,step)
                            sig=canon([a,o])
                            assert key not in retained_signatures or retained_signatures[key]==sig
                            retained_signatures[key]=sig
                        if nst is not None: refs(nst.get('attached_images',[]),allowed)
                        for t in THINK.findall(cleaned['conversations'][-1]['value']):
                            if t.strip(): refs(json.loads(t),allowed)
                        if cleaned.get('teacher_decision_json'): refs(json.loads(cleaned['teacher_decision_json']),allowed)
                        if RESPONSE.findall(cleaned['conversations'][-1]['value'])!=RESPONSE.findall(original['conversations'][-1]['value']): count['final_response_reference_edits']+=1
                        if m['allowed_retry_target']: count['allowed_retries']+=1
                        if after and calls(cleaned['conversations'][-1]['value']) and canon(after[-1][1])==canon(calls(cleaned['conversations'][-1]['value'])[0]):
                            assert m['allowed_retry_target']
                    final=not calls(original['conversations'][-1]['value'])
                    if final:
                        assert m['route']!='removed'; count['final_answers_preserved']+=1
            for h in handles.values(): assert not h.readline(); h.close()
            assert source_hash.hexdigest()==summary['source_integrity'][split]['sha256_before']
            assert count['final_answers_preserved']==summary['counts'][split]['input_final_answer_rows']
            reports[split]=dict(count); print(split,json.dumps(reports[split]),flush=True)
        assert not mf.readline()
    result={'independent_verification_passed':True,'checks':['partition_conservation','review_and_removed_byte_identical','source_hash_unchanged','only_original_contiguous_actions_removed',
        'nonconsecutive_actions_preserved','retained_tool_arguments_and_results_immutable','structured_references_valid_at_each_step','image_order_and_state_consistent',
        'budget_and_task_question_preserved','cross_prefix_observation_consistency','final_answers_preserved','no_unsupported_repeated_target'],
        'counts':reports}
    (args.audit/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('INDEPENDENT_VERIFICATION_PASSED',flush=True)
if __name__=='__main__': main()
