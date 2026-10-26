#!/usr/bin/env python3
"""Independent read-only comparison; writes only a verification report."""
import argparse
import collections as C
import hashlib
import json
import re
from pathlib import Path

THINK=re.compile(r'<think>(.*?)</think>',re.S)
CALL=re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>',re.S)
ANSWER=re.compile(r'<response>(.*?)</response>',re.S)
SEQUENCE=r'\d+(?:\s*(?:[-–—]|,|/|&|\band(?=\s*\d)|\bto(?=\s*\d)|\bthrough(?=\s*\d)|、|和|至|到)\s*\d+)*'
MENTION=re.compile(r'\b(?P<label>steps?|turns?)\s*#?\s*(?P<ns>'+SEQUENCE+r')|第\s*(?P<cn>\d+)\s*步|步骤\s*[#：:]?\s*(?P<cn2>'+SEQUENCE+r')',re.I)
PROSE={'rationale','reasoning','information_need','hypothesis','open_questions','visual_facts','language_facts','answers','reason','missing_evidence'}
MULTI={'depends_on','evidence_steps','executed_prefix_step_ids'}
SINGLE={'revision_of_step','observed_at_step'}
DOMAIN=re.compile(r'(?:manual|assembly|installation|recipe|instructions?|building guide|printed diagram|build box|说明书|安装|菜谱|食谱|装配|搭建)[^.!?\n]{0,35}$',re.I)

def fields(v): return [json.loads(x) for x in CALL.findall(v)]
def hist(row):
    out=[]; previous=None
    for m in row['conversations'][1:-1]:
        if m['from']=='gpt': previous=m
        else:
            assert m['from']=='observation' and previous is not None
            o=json.loads(m['value']); out.append((previous,o)); previous=None
    assert previous is None
    return out

def state(row):
    value=row['conversations'][0]['value']; start=value.find('Recorded task state follows.')
    if start<0: return None
    return json.loads(value[value.find('\n{',start)+1:])

def expand(seq):
    ns=[]
    for m in re.finditer(r'(\d+)\s*(?:[-–—]|\bto\b|\bthrough\b|至|到)\s*(\d+)|(\d+)',seq,re.I):
        if m[3] is not None: ns.append(int(m[3]))
        else: ns.extend(range(int(m[1]),int(m[2])+1))
    return ns

def expected_prose(text,mapping):
    def replace(m):
        nums=expand(m['ns'] or m['cn'] or m['cn2'])
        if not any(n in mapping and n!=mapping[n] for n in nums): return m.group(0)
        before=text[max(0,m.start()-80):m.start()]; after=text[m.end():m.end()+90]
        domain=DOMAIN.search(before) or re.match(r'^\s*(?:of|in)\s+(?:the\s+)?(?:manual|assembly|installation|recipe|instructions?|building guide|说明书)',after,re.I)
        explicit=re.search(r'(?:agent|tool|execution|executed)\s*$',before,re.I)
        if domain and not explicit: return m.group(0)
        assert all(n in mapping for n in nums)
        new=list(dict.fromkeys(mapping[n] for n in nums))
        if m['cn']: return '第'+str(new[0])+'步'
        if m['cn2']: return '步骤 '+', '.join(map(str,new))
        label=m['label']
        if len(new)>1 and not label.lower().endswith('s'): label+='s'
        return label+' '+', '.join(map(str,new))
    return MENTION.sub(replace,text)

def compare_meta(old,new,mapping,prose=False):
    assert type(old)==type(new),(type(old),type(new))
    if isinstance(old,dict):
        assert set(old)==set(new)
        for k,v in old.items():
            if k in MULTI: assert new[k]==list(dict.fromkeys(mapping[n] for n in v)),(k,v,new[k])
            elif k in SINGLE: assert new[k]==(None if v is None else mapping[v]),(k,v,new[k])
            else: compare_meta(v,new[k],mapping,prose or k in PROSE)
    elif isinstance(old,list):
        assert len(old)==len(new)
        for a,b in zip(old,new): compare_meta(a,b,mapping,prose)
    elif prose and isinstance(old,str): assert new==expected_prose(old,mapping),(old,new)
    else: assert old==new,(old,new)

def check_refs(obj,allowed):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k in MULTI: assert isinstance(v,list) and all(type(n) is int and n in allowed for n in v)
            elif k in SINGLE: assert v is None or (type(v) is int and v in allowed)
            else: check_refs(v,allowed)
    elif isinstance(obj,list):
        for v in obj: check_refs(v,allowed)

def compare_assistant(old,new,mapping):
    assert fields(old)==fields(new)
    ot=THINK.findall(old); nt=THINK.findall(new); assert len(ot)==len(nt)
    for a,b in zip(ot,nt):
        if a.strip(): compare_meta(json.loads(a),json.loads(b),mapping)
        else: assert a==b
    oa=ANSWER.findall(old); na=ANSWER.findall(new); assert len(oa)==len(na)
    for a,b in zip(oa,na): assert b==expected_prose(a,mapping)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--audit-dir',type=Path,required=True); ap.add_argument('--stage1-audit',type=Path,required=True); args=ap.parse_args()
    summary=json.loads((args.audit_dir/'summary.json').read_text()); aliases={}
    with (args.stage1_audit/'trajectory_plans.jsonl').open() as f:
        for line in f:
            x=json.loads(line); aliases[x['trajectory_id']]={int(k):v for k,v in x.get('aliases',{}).items()}
    counts={}; ids=set(); consistency={}
    with (args.audit_dir/'step_id_mapping.jsonl').open() as manifest:
        for split in ['train','val']:
            counter=C.Counter(); h=hashlib.sha256()
            with (args.input/(split+'.jsonl')).open('rb') as inp,(args.output/(split+'.jsonl')).open('rb') as out,(args.output/'review_renumbering'/(split+'.jsonl')).open('rb') as review:
                for ln,raw in enumerate(inp,1):
                    h.update(raw); old=json.loads(raw); record=json.loads(next(manifest))
                    assert record['file']==split+'.jsonl' and record['line']==ln
                    assert record['source_line_sha256']==hashlib.sha256(raw).hexdigest()
                    if record['route']=='review':
                        assert review.readline()==raw; counter['review_rows']+=1; continue
                    encoded=out.readline(); assert encoded; new=json.loads(encoded); counter['rows_verified']+=1
                    assert new['id'] not in ids; ids.add(new['id'])
                    if 'step' not in old:
                        assert encoded==raw; counter['auxiliary_unchanged']+=1; counter['final_answers_preserved']+=not bool(fields(old['conversations'][-1]['value'])); continue
                    before=hist(old); after=hist(new); assert len(before)==len(after)
                    oldsteps=[o['step'] for m,o in before]
                    direct={0:0,**{n:i for i,n in enumerate(oldsteps,1)}}; mapping=dict(direct)
                    tid=old['trajectory_id']; als=aliases.get(tid,{})
                    for deleted in als:
                        representative=deleted; visited=set()
                        while representative in als:
                            assert representative not in visited; visited.add(representative); representative=als[representative]
                        if deleted<old['step'] and deleted not in direct and representative in direct: mapping[deleted]=direct[representative]
                    target=len(after)+1
                    assert new['step']==target and [o['step'] for m,o in after]==list(range(1,target))
                    if old['step']==target: assert new['id']==old['id']
                    else:
                        expected=re.sub(r':teacher-step:'+str(old['step'])+r'$',':teacher-step:'+str(target)+':source-step:'+str(old['step']),old['id'])
                        assert expected==new['id']
                    for key in set(old)-{'id','step','conversations','teacher_decision_json'}: assert new[key]==old[key],key
                    assert len(old['conversations'])==len(new['conversations'])
                    os=state(old); ns=state(new)
                    for k in set(os)-{'executed_prefix_step_ids','attached_images','image_inventory','initial_views'}: assert os[k]==ns[k],k
                    assert ns['executed_prefix_step_ids']==list(range(1,target))
                    for k in ('attached_images','image_inventory','initial_views'):
                        if k in os: compare_meta(os[k],ns[k],mapping)
                    allowed={0}
                    for (om,oo),(nm,no) in zip(before,after):
                        compare_assistant(om['value'],nm['value'],mapping)
                        assert oo['result']==no['result']
                        assert no['step']==direct[oo['step']]
                        compare_meta(oo.get('observation'),no.get('observation'),mapping)
                        for t in THINK.findall(nm['value']):
                            if t.strip(): check_refs(json.loads(t),allowed)
                        check_refs(no.get('observation'),allowed); allowed.add(no['step'])
                        key=(tid,oo['step'])
                        assert key not in consistency or consistency[key]==no['step']; consistency[key]=no['step']
                    tm={**mapping,old['step']:target}
                    compare_assistant(old['conversations'][-1]['value'],new['conversations'][-1]['value'],tm)
                    for t in THINK.findall(new['conversations'][-1]['value']):
                        if t.strip(): check_refs(json.loads(t),allowed)
                    if old.get('teacher_decision_json'):
                        compare_meta(json.loads(old['teacher_decision_json']),json.loads(new['teacher_decision_json']),tm)
                        check_refs(json.loads(new['teacher_decision_json']),allowed)
                    check_refs(ns['attached_images'],allowed)
                    assert old['images']==new['images']
                    counter['final_answers_preserved']+=not bool(fields(old['conversations'][-1]['value']))
                    counter['renumbered_rows']+=record['route']=='renumbered'
                assert not out.readline() and not review.readline()
            assert h.hexdigest()==summary['sources'][split]['sha256']
            assert counter['rows_verified']+counter['review_rows']==summary['counts'][split]['input_rows']
            counts[split]=dict(counter); print(split,json.dumps(dict(counter)),flush=True)
        assert not manifest.readline()
    result={'independent_verification_passed':True,'counts':counts,'sample_ids_unique':True,'source_hashes_match':True,
        'all_steps_contiguous':True,'references_checked_against_direct_maps':True,'results_and_tool_arguments_unchanged':True,
        'only_step_references_changed_in_generated_text':True,'question_images_and_budget_unchanged':True,'cross_prefix_history_numbering_consistent':True}
    (args.audit_dir/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('INDEPENDENT_VERIFICATION_PASSED',flush=True)

if __name__=='__main__': main()
