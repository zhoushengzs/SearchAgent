#!/usr/bin/env python3
"""Renumber a cleaned ShareGPT dataset into a NEW, non-overwritten version.

Original step 0 remains the initial evidence. Kept historical events become 1..H;
current decision becomes H+1. References map once, never transitively through a
renumbering dictionary. Existing deleted-event aliases compose BEFORE numbering.
No network, model calls, or changes to source data/teacher jobs.
"""
import argparse
import collections as C
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

VERSION='1.0.0'
CALL=re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>',re.S)
THINK=re.compile(r'<think>(.*?)</think>',re.S)
RESPONSE=re.compile(r'<response>(.*?)</response>',re.S)
REF_LIST={'depends_on','evidence_steps','executed_prefix_step_ids'}
REF_SINGLE={'revision_of_step','observed_at_step'}
PROSE_KEYS={'rationale','reasoning','information_need','hypothesis','open_questions','visual_facts','language_facts','answers','reason','missing_evidence'}
NUM_SEQUENCE=r'\d+(?:\s*(?:[-–—]|,|/|&|\band(?=\s*\d)|\bto(?=\s*\d)|\bthrough(?=\s*\d)|、|和|至|到)\s*\d+)*'
MENTION=re.compile(r'\b(?P<label>steps?|turns?)\s*#?\s*(?P<nums>'+NUM_SEQUENCE+r')|第\s*(?P<cn>\d+)\s*步|步骤\s*[#：:]?\s*(?P<cn2>'+NUM_SEQUENCE+r')',re.I)
DOMAIN_BEFORE=re.compile(r'(?:manual|assembly|installation|recipe|instructions?|building guide|printed diagram|build box|说明书|安装|菜谱|食谱|装配|搭建)[^.!?\n]{0,35}$',re.I)
DOMAIN_AFTER=re.compile(r'^\s*(?:of|in)\s+(?:the\s+)?(?:manual|assembly|installation|recipe|instructions?|building guide|说明书)',re.I)
AGENT_BEFORE=re.compile(r'(?:agent|tool|execution|executed|previous|prior|earlier|preceding|visual facts|visual evidence|visual descriptions|language facts|findings|hypothesis|hypotheses|conclusion|search snippet|search result|tool result|web search|search query|prior notes|hypothesized|identified|identification|image from|date from|timeout in|described in|之前|此前|工具|搜索|观察|假设)[^.!?\n]{0,95}$',re.I)
AGENT_AFTER=re.compile(r"^(?:'s)?\s*(?:(?:explicitly|previously|visually|textually)\s+)?(?:provided|attempted|initiated|established|hypothesized|identified|confirmed|showed|reported|indicated|describe|described|failed|returned|had|did have|which|suggests|shows|reasoning|search result|visual description|visual facts|hypothesis|的搜索|的观察|的结果)",re.I)

class Review(Exception):
    def __init__(self,code,details=None): self.code=code; self.details=details; super().__init__(code)

def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def dump(path,x): Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def get_calls(value): return [json.loads(m) for m in CALL.findall(value)]

def state_of(row):
    text=row['conversations'][0]['value']; marker=text.find('Recorded task state follows.')
    if marker<0: return None,None
    pos=text.find('\n{',marker)
    if pos<0: raise Review('unparseable_state')
    return text[:pos+1],json.loads(text[pos+1:])

def history(row):
    messages=row['conversations']; out=[]; pending=None
    if messages[-1]['from']!='gpt': raise Review('target_role')
    for i,m in enumerate(messages[1:-1],1):
        if m['from']=='gpt':
            cs=get_calls(m['value'])
            if pending is not None or len(cs)!=1: raise Review('history_pairing')
            pending=(i,cs[0])
        elif m['from']=='observation':
            if pending is None: raise Review('history_pairing')
            o=json.loads(m['value']); step=o['step']
            if not isinstance(step,int) or isinstance(step,bool) or step<=0: raise Review('invalid_step')
            out.append({'step':step,'assistant_index':pending[0],'observation_index':i,'call':pending[1],'observation':o})
            pending=None
        else: raise Review('unexpected_role')
    if pending is not None: raise Review('history_pairing')
    ids=[a['step'] for a in out]
    if ids!=sorted(set(ids)): raise Review('history_order')
    return out

def resolve_alias(old,aliases):
    seen=set()
    while old in aliases:
        if old in seen: raise Review('alias_cycle')
        seen.add(old); old=aliases[old]
    return old

def maps_for(row,hist,aliases):
    ids=[a['step'] for a in hist]; direct={0:0,**{s:i for i,s in enumerate(ids,1)}}
    old_target=row['step']
    if not isinstance(old_target,int) or isinstance(old_target,bool) or (ids and old_target<=ids[-1]): raise Review('current_step_order')
    composed=dict(direct)
    for deleted in aliases:
        representative=resolve_alias(deleted,aliases)
        if deleted<old_target and deleted not in direct and representative in direct:
            composed[deleted]=direct[representative]
    return direct,composed,old_target,len(hist)+1

def numbers(sequence):
    result=[]
    for part in re.finditer(r'(\d+)\s*(?:[-–—]|\bto\b|\bthrough\b|至|到)\s*(\d+)|(\d+)',sequence,re.I):
        if part.group(3) is not None: result.append(int(part.group(3)))
        else:
            lo,hi=int(part.group(1)),int(part.group(2))
            if lo>hi or hi-lo>1000: raise Review('unsupported_reference_range',sequence)
            result.extend(range(lo,hi+1))
    return result

def rewrite_text(text,mapping,path,edits):
    def replace(match):
        sequence=match.group('nums') or match.group('cn') or match.group('cn2')
        ns=numbers(sequence)
        if not any(n in mapping and mapping[n]!=n for n in ns): return match.group(0)
        before=text[max(0,match.start()-120):match.start()]
        after=text[match.end():match.end()+90]
        domain=bool(DOMAIN_BEFORE.search(before[-80:]) or DOMAIN_AFTER.search(after))
        explicit=bool(re.search(r'(?:agent|tool|execution|executed)\s*$',before,re.I))
        if domain and not explicit:
            edits.append({'field':path,'kind':'domain_step_unchanged','before':match.group(0),'after':match.group(0)})
            return match.group(0)
        clear=explicit or bool(AGENT_BEFORE.search(before) or AGENT_AFTER.search(after))
        if not clear: raise Review('ambiguous_prose_step_reference',{'field':path,'mention':match.group(0),'context':before[-90:]+match.group(0)+after[:90]})
        if any(n not in mapping for n in ns): raise Review('unresolved_prose_step_reference',{'field':path,'mention':match.group(0)})
        vals=list(dict.fromkeys(mapping[n] for n in ns))
        if match.group('cn'):
            new='第'+str(vals[0])+'步'
        elif match.group('cn2'):
            new='步骤 '+', '.join(map(str,vals))
        else:
            label=match.group('label')
            if len(vals)>1 and not label.lower().endswith('s'): label+='s'
            new=label+' '+', '.join(map(str,vals))
        edits.append({'field':path,'kind':'agent_step_reference','before':match.group(0),'after':new})
        return new
    return MENTION.sub(replace,text)

def ref_number(n,mapping,path):
    if not isinstance(n,int) or isinstance(n,bool) or n not in mapping: raise Review('unresolved_structured_reference',{'field':path,'value':n})
    return mapping[n]  # Direct lookup: 5->4 must not then become 4->3.

def rewrite_fields(value,mapping,path,edits,prose=False):
    if isinstance(value,dict):
        out={}
        for k,v in value.items():
            field=path+'.'+k
            if k in REF_LIST:
                if not isinstance(v,list): raise Review('invalid_reference_list',field)
                out[k]=list(dict.fromkeys(ref_number(n,mapping,field) for n in v))
            elif k in REF_SINGLE:
                out[k]=None if v is None else ref_number(v,mapping,field)
            else: out[k]=rewrite_fields(v,mapping,field,edits,prose or k in PROSE_KEYS)
        return out
    if isinstance(value,list): return [rewrite_fields(v,mapping,path+'[]',edits,prose) for v in value]
    if prose and isinstance(value,str): return rewrite_text(value,mapping,path,edits)
    return value

def rewrite_assistant(value,mapping,path,edits):
    def think(match):
        content=match.group(1)
        if not content.strip(): return match.group(0)
        payload=rewrite_fields(json.loads(content),mapping,path+'.think',edits)
        return '<think>'+json.dumps(payload,ensure_ascii=False)+'</think>'
    def response(match):
        return '<response>'+rewrite_text(match.group(1),mapping,path+'.response',edits)+'</response>'
    return RESPONSE.sub(response,THINK.sub(think,value))

def new_id(original,old,new):
    if old==new: return original
    pattern=r':teacher-step:'+str(old)+r'$'
    if re.search(pattern,original): return re.sub(pattern,':teacher-step:'+str(new)+':source-step:'+str(old),original)
    return original+':renumbered-step:'+str(new)+':source-step:'+str(old)

def transform(row,aliases=None):
    aliases=aliases or {}; hist=history(row); prefix,state=state_of(row)
    if 'step' not in row:
        if hist or state is not None: raise Review('unexpected_unstepped_row')
        return row,{'changed':False,'auxiliary':True,'old_to_new_history':{},'source_history_to_new':{},'text_edits':[]}
    if state is None: raise Review('missing_recorded_state')
    ids=[a['step'] for a in hist]
    if state.get('executed_prefix_step_ids')!=ids: raise Review('source_state_history_mismatch')
    direct,composed,old_target,target=maps_for(row,hist,aliases)
    changed=old_target!=target or any(k!=v for k,v in direct.items())
    if not changed:
        return row,{'changed':False,'auxiliary':False,'old_to_new_history':direct,'source_history_to_new':composed,'old_target_step':old_target,'new_target_step':target,'text_edits':[]}
    out=copy.deepcopy(row); edits=[]
    # Reindex metadata in historical Agent-generated notes only. The user's
    # question, raw tool results, tool arguments and physical file names stay exact.
    for a in hist:
        ai=a['assistant_index']; oi=a['observation_index']
        out['conversations'][ai]['value']=rewrite_assistant(row['conversations'][ai]['value'],composed,'messages['+str(ai)+']',edits)
        obs=copy.deepcopy(a['observation']); obs['step']=direct[a['step']]
        if 'observation' in obs: obs['observation']=rewrite_fields(obs['observation'],composed,'messages['+str(oi)+'].observer',edits)
        out['conversations'][oi]['value']=json.dumps(obs,ensure_ascii=False)
    target_mapping={**composed,old_target:target}
    out['conversations'][-1]['value']=rewrite_assistant(row['conversations'][-1]['value'],target_mapping,'target',edits)
    if row.get('teacher_decision_json'):
        out['teacher_decision_json']=json.dumps(rewrite_fields(json.loads(row['teacher_decision_json']),target_mapping,'teacher_decision',edits),ensure_ascii=False)
    ns=copy.deepcopy(state); ns['executed_prefix_step_ids']=list(range(1,len(hist)+1))
    for k in ('attached_images','image_inventory','initial_views'):
        if k in ns: ns[k]=rewrite_fields(ns[k],composed,'state.'+k,edits)
    out['conversations'][0]['value']=prefix+json.dumps(ns,ensure_ascii=False)
    out['step']=target; out['id']=new_id(row['id'],old_target,target)
    validate(out)
    assert out.get('images')==row.get('images')
    assert state['question']==ns['question']
    assert state.get('remaining_decisions')==ns.get('remaining_decisions')
    for before,after in zip(hist,history(out)):
        assert before['call']==after['call'] and before['observation']['result']==after['observation']['result']
    assert get_calls(out['conversations'][-1]['value'])==get_calls(row['conversations'][-1]['value'])
    return out,{'changed':True,'auxiliary':False,'old_to_new_history':direct,'source_history_to_new':composed,
                'old_target_step':old_target,'new_target_step':target,'old_id':row['id'],'new_id':out['id'],'text_edits':edits}

def check_refs(value,allowed):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in REF_LIST:
                if not isinstance(v,list) or any(not isinstance(n,int) or isinstance(n,bool) or n not in allowed for n in v): raise Review('dangling_reference',{'field':k,'value':v})
            elif k in REF_SINGLE:
                if v is not None and (not isinstance(v,int) or isinstance(v,bool) or v not in allowed): raise Review('dangling_reference',{'field':k,'value':v})
            else: check_refs(v,allowed)
    elif isinstance(value,list):
        for v in value: check_refs(v,allowed)

def validate(row):
    hist=history(row); prefix,state=state_of(row)
    if 'step' not in row:
        if hist: raise Review('auxiliary_with_history')
        return
    if [a['step'] for a in hist]!=list(range(1,len(hist)+1)) or row['step']!=len(hist)+1: raise Review('noncontiguous_numbering')
    if state['executed_prefix_step_ids']!=list(range(1,len(hist)+1)): raise Review('state_numbering')
    allowed={0}
    for a in hist:
        for text in THINK.findall(row['conversations'][a['assistant_index']]['value']):
            if text.strip(): check_refs(json.loads(text),allowed)
        check_refs(a['observation'].get('observation'),allowed)
        allowed.add(a['step'])
    for text in THINK.findall(row['conversations'][-1]['value']):
        if text.strip(): check_refs(json.loads(text),allowed)
    if row.get('teacher_decision_json'):
        decision=json.loads(row['teacher_decision_json']); check_refs(decision,allowed)
        calls=get_calls(row['conversations'][-1]['value'])
        if calls:
            expected='vs_'+decision['tool'] if decision['tool'] in ('web_search','image_search','visit','inspect_image','crop','calculate') else decision['tool']
            if len(calls)!=1 or expected!=calls[0]['name'] or decision.get('arguments',{})!=calls[0]['arguments']: raise Review('teacher_target_mismatch')
    for key in ('initial_views','image_inventory','attached_images'): check_refs(state.get(key,[]),allowed)
    indices=[int(x) for x in re.findall(r'img_(\d+): <image>',prefix)]
    if indices!=list(range(1,len(row.get('images',[]))+1)): raise Review('image_placeholder_order')
    if [x['image_index'] for x in state.get('attached_images',[])]!=indices: raise Review('image_index_order')

def load_aliases(path):
    result={}
    with Path(path).open() as f:
        for line in f:
            obj=json.loads(line); result[obj['trajectory_id']]={int(k):v for k,v in obj.get('aliases',{}).items()}
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-dir',type=Path,required=True); p.add_argument('--stage1-audit',type=Path,required=True)
    p.add_argument('--replace-generated-output',action='store_true',help='Only replace an output with this program matching source/audit manifest')
    args=p.parse_args(argv); source=args.input.resolve(); output=args.output.resolve(); audit=args.audit_dir.resolve()
    if not source.is_dir() or output==source or output in source.parents: p.error('invalid input/output path')
    if output.exists() or audit.exists():
        if not args.replace_generated_output: p.error('output/audit already exist; default refuses overwrite')
        previous=json.loads((audit/'summary.json').read_text())
        if previous.get('version')!=VERSION or previous.get('input')!=str(source) or previous.get('output')!=str(output) or previous.get('audit_dir')!=str(audit):
            p.error('refuse replacement of unrecognized data')
        if not (output/'renumbering_manifest.json').is_file(): p.error('missing generated output marker')
    if output in audit.parents or audit in output.parents or output==audit: p.error('data and audit directories must be separate')
    output.mkdir(parents=True,exist_ok=args.replace_generated_output); audit.mkdir(parents=True,exist_ok=args.replace_generated_output); (output/'review_renumbering').mkdir(exist_ok=args.replace_generated_output); (output/'configs').mkdir(exist_ok=args.replace_generated_output)
    stage1=json.loads((args.stage1_audit/'summary.json').read_text()); aliases=load_aliases(args.stage1_audit/'trajectory_plans.jsonl')
    summaries={}; sources={}; all_ids=set(); step_meanings={}; modified_examples=[]; review_reasons=C.Counter(); row_count=0
    with (audit/'step_id_mapping.jsonl').open('w') as mf:
        for split in ('train','val'):
            counts=C.Counter(); oldhash=hashlib.sha256(); sh=source/(split+'.jsonl'); before=sh.stat()
            with sh.open('rb') as f,(output/(split+'.jsonl')).open('wb') as out,(output/'review_renumbering'/(split+'.jsonl')).open('wb') as review:
                for ln,raw in enumerate(f,1):
                    counts['input_rows']+=1; oldhash.update(raw); row=json.loads(raw); final=not get_calls(row['conversations'][-1]['value'])
                    tid=row.get('trajectory_id') or row['id'].split(':')[0]; record={'file':split+'.jsonl','line':ln,'old_id':row['id'],'trajectory_id':tid,'source_line_sha256':hashlib.sha256(raw).hexdigest()}
                    try:
                        changed,details=transform(row,aliases.get(tid,{})); validate(changed)
                        if changed['id'] in all_ids: raise Review('duplicate_new_sample_id')
                        all_ids.add(changed['id'])
                        for old,new in details.get('old_to_new_history',{}).items():
                            key=(tid,old)
                            if key in step_meanings and step_meanings[key]!=new: raise Review('cross_prefix_numbering_disagreement')
                            step_meanings[key]=new
                        encoded=(json.dumps(changed,ensure_ascii=False)+'\n').encode() if details['changed'] else raw
                        out.write(encoded); counts['output_rows']+=1; counts['renumbered_rows']+=details['changed']; counts['auxiliary_rows_unchanged']+=details['auxiliary']
                        counts['final_answer_rows']+=final; counts['text_reference_edits']+=sum(e['kind']=='agent_step_reference' for e in details['text_edits'])
                        record.update(route='renumbered' if details['changed'] else 'unchanged',new_id=changed['id'],**{k:v for k,v in details.items() if k not in ('new_id','old_id')})
                        if details['changed'] and len(modified_examples)<10:
                            modified_examples.append({'file':split+'.jsonl','line':ln,'old_id':row['id'],'new_id':changed['id'],
                                'old_history':[a['step'] for a in history(row)],'new_history':[a['step'] for a in history(changed)],'old_target':row['step'],'new_target':changed['step']})
                    except Review as e:
                        review.write(raw); counts['review_rows']+=1; counts['review_final_answer_rows']+=final; review_reasons[e.code]+=1
                        record.update(route='review',reason=e.code,details=e.details)
                    mf.write(json.dumps(record,ensure_ascii=False)+'\n'); row_count+=1
            after=sh.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            checksum=oldhash.hexdigest(); assert hash_file(sh)==checksum
            assert counts['input_rows']==counts['output_rows']+counts['review_rows']
            summaries[split]=dict(counts); sources[split]={'path':str(sh),'sha256':checksum,'unchanged':True,'bytes':before.st_size}
            print(split,json.dumps(dict(counts),ensure_ascii=False),flush=True)
    shutil.copyfile(source/'dataset_info.json',output/'dataset_info.json')
    for config in (source/'configs').glob('*.json'):
        dest=output/'configs'/config.name
        if config.name=='export_notes.json': dest=output/'configs'/'stage1_export_notes.json'
        if config.name in ('full.json','lora.json'):
            obj=json.loads(config.read_text()); obj['dataset_dir']=str(output)
            obj['output_dir']=str(output/'runs'/('agent_full' if config.name=='full.json' else 'agent_lora'))
            if 'deepspeed' in obj: obj['deepspeed']=str(output/'configs'/'zero3.json')
            dump(dest,obj)
        else: shutil.copyfile(config,dest)
    summary={'version':VERSION,'input':str(source),'output':str(output),'audit_dir':str(audit),'counts':summaries,'sources':sources,
        'review_reasons':dict(review_reasons),'examples':modified_examples,'step0':'initial evidence, unchanged','historical_ids':'contiguous 1..H','current_decision_id':'H+1',
        'original_event_aliases':'composed from stage-1 plan before direct new-index lookup',
        'sample_ids':'changed decision gets teacher-step:<new>:source-step:<old>; no colliding IDs',
        'remaining_decisions':'source real execution budget unchanged; renumbering does not refund calls',
        'tool_arguments_and_results':'unchanged','images_and_asset_paths':'unchanged; observed_at_step remapped',
        'prior_review_dir':str(source/'review'),'prior_removed_targets_dir':str(source/'removed_targets'),
        'prior_teacher_configs_and_jobs_modified':False,'model_calls':0,'all_rows_accounted_for':True,'completed':True}
    dump(audit/'summary.json',summary)
    dump(output/'renumbering_manifest.json',{'source':str(source),'audit_dir':str(audit),'counts':summaries,
        'previous_cleaning_manifest':str(source/'cleaning_manifest.json'),'previous_review_and_removed_partitions':{'review':str(source/'review'),'removed_targets':str(source/'removed_targets')},'completed':True})
    dump(output/'configs'/'export_notes.json',{'purpose':'stage-1 cleaning + contiguous step renumbering','source':str(source),
        'audit_dir':str(audit),'counts':summaries,'mask_history':True,'history_steps':'1..H','current_step':'H+1',
        'initial_step':0,'remaining_decisions':'unchanged original execution budget','new_tool_execution':False,'teacher_model_called':False})
    print('COMPLETED',output,flush=True); return summary

if __name__=='__main__': main()
