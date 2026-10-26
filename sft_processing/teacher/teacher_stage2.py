#!/usr/bin/env python3
"""OpenRouter next-decision optimization: prepare / doctor / run / export.

Standard library only. Does not execute investigation tools or synthesize their
results. Candidate positives require structural checks and a separate reviewer.
API key is read only from OPENROUTER_API_KEY, never from CLI/config/logs.
"""
import argparse
import base64
import collections as C
import copy
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

VERSION='2.0.0-numbered'
import renumber_sft as numbering
BEHAVIORS=['forward','reassess_existing','revisit_resource','refine_query','retry','finish','abstain']
COMMON_REQUIRED=['tool','arguments','rationale','information_need','gap_type','depends_on']

class Invalid(Exception): pass
class APIError(Exception):
    def __init__(self,status,code):
        self.status=status; self.code=code
        super().__init__(f'OpenRouter status={status} code={code}')

def dumps(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x): return hashlib.sha256(dumps(x).encode()).hexdigest()
def read_json(p): return json.loads(Path(p).read_text())
def write_json(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def config_at(path):
    path=Path(path).resolve(); cfg=read_json(path); cfg['_config_dir']=str(path.parent)
    return cfg

def cleaner(cfg):
    path=Path(cfg['cleaner_script'])
    spec=importlib.util.spec_from_file_location('stage1_cleaner',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def prompt(cfg,role): return (Path(cfg.get('prompts_dir',str(Path(cfg['_config_dir'])/'prompts')))/(role+'.txt')).read_text()

def history_from_base(base,c):
    dummy=copy.deepcopy(base)
    dummy['conversations'].append({'from':'gpt','value':'<think></think>\n<response></response>'})
    return c.history(dummy)

def base_from_row(row,c,kind=None,cfg=None):
    if cfg and cfg.get('numbering_version')=='v2_contiguous':
        if kind=='control':
            rebuilt=copy.deepcopy(row)
            numbering.validate(rebuilt)
        else:
            plan=c.make_plan(row)
            rebuilt,_=c.rebuild_row(row,plan)
            rebuilt,_=numbering.transform(rebuilt,plan['aliases'])
            numbering.validate(rebuilt)
    else:
        plan=c.make_plan(row)
        rebuilt,_=c.rebuild_row(row,plan)
    # No original target text/teacher_decision_json ever reaches the generator.
    base=copy.deepcopy(rebuilt)
    base['conversations']=base['conversations'][:-1]
    base.pop('teacher_decision_json',None)
    base['sample_type']='teacher_optimization_pending'
    return base

def jsonlines(path,with_source_hash=False):
    with Path(path).open('rb') as f:
        for ln,line in enumerate(f,1):
            if with_source_hash: yield ln,json.loads(line),hashlib.sha256(line).hexdigest()
            else: yield ln,json.loads(line)

def job_for(row,source,ln,kind,c,cfg=None):
    base=base_from_row(row,c,kind,cfg)
    visible={'conversations':base['conversations'],'images':base.get('images',[]),'tools':base.get('tools'),'system':base.get('system')}
    return {'job_id':digest(visible),'base_row':base,'provenance':{'file':str(source),'line':ln,'original_id':row['id'],
        'trajectory_id':row.get('trajectory_id'),'kind':kind,'numbering_version':(cfg or {}).get('numbering_version','legacy_fixture'),'source_step':row.get('step'),'prepared_step':base.get('step')},'prefix_hash':digest(visible),'status':'prepared'}

def prepare(cfg,limit=100):
    c=cleaner(cfg); path=Path(cfg['jobs_file']); report=path.with_suffix('.prepare.json')
    if path.exists() or report.exists(): raise Invalid('jobs output already exists; use a new jobs_file')
    path.parent.mkdir(parents=True,exist_ok=True)
    root=Path(cfg['cleaned_dir']); recovered=[]; controls=[]; seen=set(); skips=C.Counter()
    recovery_limit=round(limit*cfg.get('recovery_fraction',0.6))
    eligible=None
    if cfg.get('recovery_manifest'):
        eligible={item['line']:item for _,item in jsonlines(cfg['recovery_manifest'])
                  if item['file']=='train.jsonl' and item['route']=='removed'}
    recovery_source=Path(cfg.get('recovery_source',str(root/'removed_targets'/'train.jsonl')))
    # Select earliest eligible removed target per trajectory. Its history is
    # cleaned and verified before target removal; we never truncate future state.
    for kind,source,collection,desired in [
        ('recovery',recovery_source,recovered,recovery_limit),
        ('control',root/'train.jsonl',controls,limit-recovery_limit)]:
        for ln,row,raw_sha in jsonlines(source,True):
            if kind=='recovery' and eligible is not None:
                if ln not in eligible: continue
                expected=eligible[ln]
                if row['id']!=expected['id'] or raw_sha!=expected['source_sha256']:
                    raise Invalid('original_source_changed_since_cleaning')
            tid=row.get('trajectory_id')
            if not tid or tid in seen: continue
            if len(collection)>=desired: break
            try:
                _,state=c.parse_state(row)
                if state is None: raise Invalid('no_recorded_state')
                job=job_for(row,source,ln,kind,c,cfg)
                context_job(job,c)
                preflight_images(job,cfg,load_data=False)
                collection.append(job); seen.add(tid)
            except (c.NeedsReview,numbering.Review,Invalid,ValueError,KeyError,TypeError) as e:
                skips[getattr(e,'code',str(e))]+=1
    jobs=[]
    # Interleave sources so a three-job pilot includes a normal-control case.
    while recovered or controls:
        if recovered: jobs.append(recovered.pop(0))
        if controls: jobs.append(controls.pop(0))
        if recovered: jobs.append(recovered.pop(0))
    with path.open('w') as f:
        for job in jobs: f.write(json.dumps(job,ensure_ascii=False)+'\n')
    info={'jobs':len(jobs),'by_source':dict(C.Counter(j['provenance']['kind'] for j in jobs)),
          'unique_trajectories':len(seen),'skips':dict(skips),'train_only':True,'original_target_in_teacher_input':False,
          'jobs_sha256':sha_file(path),'teacher_called':False,'numbering_version':cfg.get('numbering_version'),'recovery_source':str(recovery_source)}
    write_json(report,info); print(json.dumps(info,ensure_ascii=False,indent=2)); return info

def native_tools(base,c):
    _,state=c.parse_state(base)
    available=state.get('tool_configuration',{}).get('available_tools',[])
    functions={t['function']['name']:t['function'] for t in json.loads(base.get('tools','[]'))}
    tools={}
    for name in available:
        if name in ('finish','abstain'): continue
        exported='vs_'+name
        fn=functions.get(exported)
        if fn: tools[name]={'exported_name':exported,'description':fn.get('description',''),'parameters':fn['parameters']}
    if 'finish' in available:
        tools['finish']={'parameters':{'type':'object','properties':{'answers':{'type':'array','items':{'type':'string'},'minItems':1},
            'reasoning':{'type':'string'},'evidence_steps':{'type':'array','items':{'type':'integer'}}},
            'required':['answers','reasoning','evidence_steps'],'additionalProperties':False},'description':'Evidence-supported final answer.'}
    if 'abstain' in available:
        tools['abstain']={'parameters':{'type':'object','properties':{'reason':{'type':'string'},'missing_evidence':{'type':'array','items':{'type':'string'}}},
            'required':['reason','missing_evidence'],'additionalProperties':False},'description':'Stop with unresolved evidence; do not fabricate an answer.'}
    if not tools: raise Invalid('no_supported_project_tools')
    return tools

def context_job(job,c):
    base=job['base_row']
    if 'teacher_decision_json' in base: raise Invalid('original_target_leak')
    _,state=c.parse_state(base)
    return {'current_state_and_question':state,'visible_prefix':base['conversations'],
            'image_order':[{'image_index':i,'reference':'img_'+str(i)} for i in range(1,len(base.get('images',[]))+1)],
            'decision_schema':{'required':COMMON_REQUIRED,'gap_type':['visual','language','mixed','none'],'tool_arguments':native_tools(base,c)},
            'retry_policy':{'max_extra_transient_retries':1,'must_be_supported_by_existing_observation':True},
            'execution_status':'Choose one NEXT decision. Its tool has NOT executed. No future result or answer is provided.'}

def preflight_images(job,cfg,load_data=False):
    images=job['base_row'].get('images',[])
    if len(images)>cfg.get('max_images',12): raise Invalid('image_count_exceeds_limit_no_silent_drop')
    result=[]; total=0
    for i,name in enumerate(images,1):
        p=Path(name) if Path(name).is_absolute() else Path(cfg['media_dir'])/name
        if not p.is_file(): raise Invalid('missing_image:'+name)
        size=p.stat().st_size; total+=size
        if size>cfg.get('max_single_image_bytes',15000000): raise Invalid('image_too_large:'+name)
        mime=mimetypes.guess_type(str(p))[0]
        if mime not in ('image/png','image/jpeg','image/webp','image/gif'): raise Invalid('unsupported_image_type:'+name)
        rec={'image_index':i,'path':str(p),'bytes':size,'sha256':sha_file(p)}
        if load_data:
            rec['part']={'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode('ascii')}}
        result.append(rec)
    if total>cfg.get('max_total_image_bytes',35000000): raise Invalid('total_images_exceed_limit_no_silent_drop')
    return result

def load_jobs(cfg):
    path=Path(cfg['jobs_file']); jobs=[j for _,j in jsonlines(path)]
    if len({j['job_id'] for j in jobs})!=len(jobs): raise Invalid('duplicate_job_ids')
    if any(j['provenance']['file'].endswith('val.jsonl') for j in jobs): raise Invalid('validation_job_must_not_enter_training')
    for job in jobs:
        if cfg.get('numbering_version')=='v2_contiguous':
            if job['provenance'].get('numbering_version')!='v2_contiguous': raise Invalid('old_numbered_jobs_forbidden')
            validate_numbered_base(job['base_row'])
        b=job['base_row']; visible={'conversations':b['conversations'],'images':b.get('images',[]),'tools':b.get('tools'),'system':b.get('system')}
        if digest(visible)!=job['prefix_hash'] or job['job_id']!=job['prefix_hash']: raise Invalid('job_prefix_modified')
    return jobs

def validate_numbered_base(base):
    dummy=copy.deepcopy(base)
    dummy['conversations'].append({'from':'gpt','value':'<think></think>\n<response></response>'})
    numbering.validate(dummy)

def strictify(schema):
    s=copy.deepcopy(schema)
    if s.get('type')=='object':
        props=s.get('properties',{}); originally_required=set(s.get('required',[]))
        s['properties']={k:(strictify(v) if k in originally_required else {'anyOf':[strictify(v),{'type':'null'}]}) for k,v in props.items()}
        s['required']=list(props); s['additionalProperties']=False
    if 'items' in s: s['items']=strictify(s['items'])
    for key in ('anyOf','oneOf'):
        if key in s: s[key]=[strictify(v) for v in s[key]]
    return s

def teacher_schema(tools):
    branches=[]
    for name,tool in tools.items():
        props={'tool':{'type':'string','enum':[name]},'arguments':tool['parameters'],'rationale':{'type':'string'},
            'information_need':{'type':'string'},'gap_type':{'type':'string','enum':['visual','language','mixed','none']},
            'depends_on':{'type':'array','items':{'type':'integer'}}}
        branches.append({'type':'object','properties':props,'required':list(props),'additionalProperties':False})
    annotation={'type':'object','properties':{
        'behavior':{'type':'string','enum':BEHAVIORS},'revisit_steps':{'type':'array','items':{'type':'integer'}},
        'revised_claim_step':{'type':['integer','null']},
        'trigger_evidence':{'type':'array','items':{'type':'object','properties':{'step':{'type':'integer'},'location':{'type':'string'},'support':{'type':'string'}},'required':['step','location','support'],'additionalProperties':False}},
        'prior_claim':{'type':['string','null']},'claim_update':{'type':'string','enum':['maintain','weaken','reject','unresolved','not_applicable']},
        'why_existing_evidence_is_insufficient':{'type':['string','null']},'expected_decision_effect':{'type':'string'},'retry_observation_step':{'type':['integer','null']}},
        'additionalProperties':False}
    annotation['required']=list(annotation['properties'])
    return {'type':'object','properties':{'decision':{'anyOf':branches},'annotation':annotation},'required':['decision','annotation'],'additionalProperties':False}

def check_schema(value,schema,path='$'):
    if 'anyOf' in schema:
        errors=[]
        for branch in schema['anyOf']:
            try: check_schema(value,branch,path); return
            except Invalid as e: errors.append(str(e))
        raise Invalid(path+': no allowed schema branch')
    types=schema.get('type'); types=[types] if isinstance(types,str) else types
    tests={'object':lambda x:isinstance(x,dict),'array':lambda x:isinstance(x,list),'string':lambda x:isinstance(x,str),
           'integer':lambda x:isinstance(x,int) and not isinstance(x,bool),'number':lambda x:isinstance(x,(int,float)) and not isinstance(x,bool),
           'boolean':lambda x:isinstance(x,bool),'null':lambda x:x is None}
    if types and not any(tests[t](value) for t in types): raise Invalid(path+': invalid type')
    if 'enum' in schema and value not in schema['enum']: raise Invalid(path+': enum mismatch')
    if isinstance(value,dict):
        if any(k not in value for k in schema.get('required',[])): raise Invalid(path+': required field missing')
        props=schema.get('properties',{})
        if schema.get('additionalProperties') is False and set(value)-set(props): raise Invalid(path+': unexpected properties')
        for k,v in value.items():
            if k in props: check_schema(v,props[k],path+'.'+k)
    if isinstance(value,list):
        if len(value)<schema.get('minItems',0) or ('maxItems' in schema and len(value)>schema['maxItems']): raise Invalid(path+': array length')
        for i,v in enumerate(value): check_schema(v,schema.get('items',{}),path+'['+str(i)+']')
    if isinstance(value,(int,float)) and not isinstance(value,bool):
        if ('minimum' in schema and value<schema['minimum']) or ('maximum' in schema and value>schema['maximum']): raise Invalid(path+': out of bounds')

def all_urls(value):
    out=set()
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('url','image_url','final_url') and isinstance(v,str): out.add(v)
            out.update(all_urls(v))
    elif isinstance(value,list):
        for v in value: out.update(all_urls(v))
    return out

def validate_candidate(candidate,job,c):
    tools=native_tools(job['base_row'],c)
    decision=candidate.get('decision',{})
    if isinstance(decision,dict) and decision.get('tool') in tools and isinstance(decision.get('arguments'),dict):
        schema=tools[decision['tool']]['parameters']; required=set(schema.get('required',[]))
        decision['arguments']={k:v for k,v in decision['arguments'].items() if v is not None or k in required}
    check_schema(candidate,teacher_schema(tools))
    d=candidate['decision']; ann=candidate['annotation']; args=d['arguments']; tool=d['tool']
    if not d['rationale'].strip() or len(d['rationale'])>4000: raise Invalid('empty_or_excessive_rationale')
    hist=history_from_base(job['base_row'],c); allowed={0}|{a['step'] for a in hist}
    c.validate_references(d,allowed)
    if any(x not in allowed for x in ann['revisit_steps']): raise Invalid('invalid_revisit_step')
    if any(x not in allowed for x in ann['revisit_steps']): raise Invalid('invalid_revisit_step')
    if ann['revised_claim_step'] is not None and ann['revised_claim_step'] not in allowed: raise Invalid('invalid_revised_claim_step')
    if ann['retry_observation_step'] is not None and ann['retry_observation_step'] not in allowed: raise Invalid('invalid_retry_observation_step')
    if any(x['step'] not in allowed or not x['location'].strip() or not x['support'].strip() for x in ann['trigger_evidence']): raise Invalid('invalid_trigger_reference')
    revisit=ann['behavior'] in ('reassess_existing','revisit_resource')
    if revisit and (not ann['revisit_steps'] or not ann['trigger_evidence']): raise Invalid('revisit_without_locatable_trigger')
    _,state=c.parse_state(job['base_row'])
    assets={x.get('asset_id') for field in ('initial_views','image_inventory','attached_images') for x in state.get(field,[]) if isinstance(x,dict)}
    known_urls=set()
    for a in hist: known_urls.update(all_urls(a['result']))
    if tool in ('crop','inspect_image') and args.get('asset_id') and args['asset_id'] not in assets: raise Invalid('unknown_asset_id')
    if tool=='visit' and args['url'] not in known_urls: raise Invalid('url_not_obtained_from_visible_tool_results')
    if tool=='inspect_image' and args.get('image_url') and args['image_url'] not in known_urls: raise Invalid('image_url_not_observed')
    if tool=='inspect_image' and not (args.get('image_url') or args.get('asset_id')): raise Invalid('empty_inspection_target')
    if tool=='crop':
        box=args.get('bbox',[])
        if len(box)!=4 or not (box[0]<box[2] and box[1]<box[3]): raise Invalid('invalid_crop_bbox')
    if tool=='calculate' and re.search(r'[A-Za-z_]\w*\s*\(',args.get('expression','')): raise Invalid('unsupported_calculator_function')
    if tool not in ('finish','abstain'):
        call={'name':tools[tool]['exported_name'],'arguments':args}
        if hist and c.action_key(call)==hist[-1]['key']:
            retry,why=c.classify_retry(hist,call)
            if not retry: raise Invalid('unsupported_adjacent_repeat:'+why)
        if ann['behavior']=='retry' and (not hist or not c.classify_retry(hist,call)[0]): raise Invalid('retry_label_without_observation_support')
    elif ann['behavior']!=tool:
        raise Invalid('terminal_behavior_mismatch')
    return {'valid':True,'scope':'next_decision_only','reference_steps':sorted(allowed),'tools_executed':False}

def parse_reply(text):
    if not isinstance(text,str): raise Invalid('non_text_model_reply')
    s=text.strip()
    if s.startswith('```') and s.endswith('```'):
        s=re.sub(r'^```(?:json)?\s*','',s,flags=re.I); s=s[:-3].strip()
    try: value=json.loads(s)
    except ValueError: raise Invalid('invalid_model_json')
    if not isinstance(value,dict): raise Invalid('reply_not_json_object')
    return value

class OpenRouter:
    def __init__(self,key,cfg,max_calls):
        if not key: raise Invalid('OPENROUTER_API_KEY is missing')
        self.key=key; self.cfg=cfg; self.calls=0; self.max_calls=max_calls
    def models(self):
        request=urllib.request.Request('https://openrouter.ai/api/v1/models',headers={'Authorization':'Bearer '+self.key})
        try:
            with urllib.request.urlopen(request,timeout=30) as response: data=json.load(response)
            return {x['id']:x for x in data['data']}
        except Exception as e: raise APIError(getattr(e,'code',0),'model_preflight_failed') from None
    def chat(self,payload):
        for attempt in range(self.cfg.get('http_retries',2)+1):
            if self.calls>=self.max_calls: raise Invalid('api_call_limit_reached')
            self.calls+=1
            request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode(),
                headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','X-OpenRouter-Title':'Agent SFT data optimization'},method='POST')
            try:
                with urllib.request.urlopen(request,timeout=self.cfg.get('timeout_seconds',180)) as response: data=json.load(response)
                if 'error' in data: raise APIError(data['error'].get('code',0),'api_response_error')
                choices=data.get('choices',[])
                if not choices or choices[0].get('finish_reason') not in ('stop','end_turn'):
                    raise APIError(0,'incomplete_or_missing_completion')
                message=choices[0].get('message',{})
                if message.get('tool_calls'): raise APIError(0,'unexpected_live_tool_call')
                content=message.get('content')
                return content,{'id':data.get('id'),'model':data.get('model'),'usage':data.get('usage',{}),'finish_reason':choices[0].get('finish_reason')}
            except urllib.error.HTTPError as e:
                # Never print response body/headers: a provider may echo a request.
                code=e.code
                if code in (429,502,503,504) and attempt<self.cfg.get('http_retries',2):
                    time.sleep(min(2**(attempt+1),8)); continue
                raise APIError(code,'http_error') from None
            except APIError: raise
            except Exception:
                # Unknown transport completion status is not automatically retried:
                # the server may already have billed a completed request.
                raise APIError(0,'transport_status_unknown_no_automatic_retry') from None

def payload_for(job,cfg,c,model,role,candidate=None,validation=None):
    context=context_job(job,c)
    text=json.dumps(context,ensure_ascii=False)
    if len(text)>cfg.get('max_context_characters',180000): raise Invalid('context_exceeds_limit_no_silent_truncation')
    if role=='reviewer':
        text+='\nCANDIDATE (untrusted until reviewed):\n'+json.dumps(candidate,ensure_ascii=False)
        text+='\nLOCAL STRUCTURAL CHECKS:\n'+json.dumps(validation,ensure_ascii=False)+'\nACTUAL NEW TOOL EXECUTION: not executed.'
    content=[{'type':'text','text':text}]
    images=preflight_images(job,cfg,load_data=True)
    for im in images:
        content.append({'type':'text','text':'Actual image img_'+str(im['image_index'])})
        content.append(im['part'])
    payload={'model':model,'messages':[{'role':'system','content':prompt(cfg,role)},{'role':'user','content':content}],
        'stream':False,'max_tokens':cfg.get('max_output_tokens',6000),'temperature':cfg.get('temperature',0.2),
        'provider':{'require_parameters':True},'response_format':{'type':'json_object'}}
    if role=='teacher' and cfg.get('response_format')=='json_schema':
        payload['response_format']={'type':'json_schema','json_schema':{'name':'agent_next_decision','strict':True,'schema':strictify(teacher_schema(native_tools(job['base_row'],c)))}}
    return payload

def model_names(cfg,args):
    teacher=getattr(args,'model',None) or os.environ.get('OPENROUTER_MODEL') or cfg.get('teacher_model')
    reviewer=getattr(args,'reviewer_model',None) or os.environ.get('OPENROUTER_REVIEWER_MODEL') or cfg.get('reviewer_model') or teacher
    if not teacher: raise Invalid('Set teacher_model in config or OPENROUTER_MODEL to a multimodal model ID')
    return teacher,reviewer

def doctor(cfg,online=False,args=None):
    c=cleaner(cfg); jobs=load_jobs(cfg); stats=C.Counter(); invalid=[]
    prompt(cfg,'teacher'); prompt(cfg,'reviewer')
    for job in jobs:
        try:
            text=json.dumps(context_job(job,c),ensure_ascii=False)
            if len(text)>cfg.get('max_context_characters',180000): raise Invalid('context_limit')
            images=preflight_images(job,cfg,False); stats['jobs_ready']+=1; stats['images']+=len(images)
        except Exception as e: invalid.append({'job_id':job['job_id'],'error':str(e)})
    if online:
        teacher,reviewer=model_names(cfg,args)
        client=OpenRouter(os.environ.get('OPENROUTER_API_KEY'),cfg,0)
        verify_models(client.models(),[teacher,reviewer],require_images=stats['images']>0)
    result={'offline':not online,'jobs':len(jobs),'ready':dict(stats),'invalid':invalid,'key_present':bool(os.environ.get('OPENROUTER_API_KEY')),
            'model_calls_made':0,'images_verified_locally':True}
    print(json.dumps(result,ensure_ascii=False,indent=2)); return result

def verify_models(models,names,require_images=True):
    for name in names:
        if name not in models: raise Invalid('model_not_in_OpenRouter_catalog:'+name)
        m=models[name]
        if require_images and 'image' not in m.get('architecture',{}).get('input_modalities',[]): raise Invalid('model_has_no_advertised_image_input:'+name)
        if 'response_format' not in m.get('supported_parameters',[]): raise Invalid('model_has_no_advertised_response_format:'+name)

def run_fingerprint(cfg,teacher,reviewer):
    cfg_clean={k:v for k,v in cfg.items() if not k.startswith('_')}
    image_hashes=sorted({(im['path'],im['sha256']) for job in load_jobs(cfg) for im in preflight_images(job,cfg,False)})
    return digest({'image_hashes':image_hashes,'config':cfg_clean,'teacher':teacher,'reviewer':reviewer,'jobs_sha256':sha_file(cfg['jobs_file']),
        'teacher_prompt':prompt(cfg,'teacher'),'reviewer_prompt':prompt(cfg,'reviewer'),
        'program':sha_file(Path(__file__)),'numbering_program':sha_file(Path(numbering.__file__)),'cleaner':sha_file(cfg['cleaner_script'])})

def review_accepted(review,candidate):
    required={'verdict','future_information_used','tool_result_fabricated','references_valid','action_necessary_or_reasonable','image_access_sufficient','validated_scope'}
    if not required.issubset(review): raise Invalid('reviewer_missing_required_checks')
    if review['validated_scope']!='next_decision_only': raise Invalid('reviewer_overclaims_execution_scope')
    if review['verdict']!='accept': return False
    if review['future_information_used'] is not False or review['tool_result_fabricated'] is not False: return False
    if any(review[x] is not True for x in ('references_valid','action_necessary_or_reasonable','image_access_sufficient')): return False
    if candidate['annotation']['behavior'] in ('reassess_existing','revisit_resource') and review.get('grounded_trigger') is not True: return False
    return True

def run(cfg,args,client_override=None):
    c=cleaner(cfg); jobs=load_jobs(cfg); teacher,reviewer=model_names(cfg,args)
    fingerprint=run_fingerprint(cfg,teacher,reviewer); run_dir=Path(cfg['runs_dir'])/fingerprint[:16]
    if args.dry_run:
        first=jobs[0] if jobs else None
        if first:
            p=payload_for(first,cfg,c,teacher,'teacher')
            print(json.dumps({'dry_run':True,'network_calls':0,'run_dir':str(run_dir),'model':teacher,'jobs':len(jobs),
                'first_job':first['job_id'],'payload_bytes':len(json.dumps(p).encode()),'original_target_supplied':False},ensure_ascii=False,indent=2))
        return run_dir
    client=client_override or OpenRouter(os.environ.get('OPENROUTER_API_KEY'),cfg,args.max_api_calls)
    if client_override is None:
        verify_models(client.models(),[teacher,reviewer],require_images=any(j['base_row'].get('images') for j in jobs))
    run_dir.mkdir(parents=True,exist_ok=True)
    write_json(run_dir/'run_metadata.json',{'fingerprint':fingerprint,'teacher_model':teacher,'reviewer_model':reviewer,'program_version':VERSION,
        'jobs_file':cfg['jobs_file'],'source_cleaned_dir':cfg['cleaned_dir'],'teacher_prompt_hash':digest(prompt(cfg,'teacher')),
        'reviewer_prompt_hash':digest(prompt(cfg,'reviewer')),'candidate_scope':'next_decision_only','tools_executed':False})
    db=sqlite3.connect(run_dir/'results.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,status TEXT,teacher TEXT,reviewer TEXT,error TEXT,usage TEXT)')
    processed=0; statuses=C.Counter(); fatal=None
    for job in jobs:
        old=db.execute('SELECT status,teacher,reviewer,usage FROM jobs WHERE id=?',(job['job_id'],)).fetchone()
        if old and old[0]!='generated' and not (args.retry_errors and old[0]=='error'): continue
        if processed>=args.limit: break
        if client.calls>=args.max_api_calls: break
        processed+=1; candidate=None; review=None; usage=[]; status='error'; err=None
        try:
            if old and old[0]=='generated':
                candidate=json.loads(old[1]); usage=json.loads(old[3] or '[]')
                validation=validate_candidate(candidate,job,c)
            else:
                body=payload_for(job,cfg,c,teacher,'teacher')
                raw,meta=client.chat(body); usage.append({'role':'teacher',**meta})
                candidate=parse_reply(raw); validation=validate_candidate(candidate,job,c)
                db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?)',(job['job_id'],'generated',json.dumps(candidate,ensure_ascii=False),None,None,json.dumps(usage)))
                db.commit()
            body=payload_for(job,cfg,c,reviewer,'reviewer',candidate,validation)
            raw,meta=client.chat(body); usage.append({'role':'reviewer',**meta}); review=parse_reply(raw)
            status='accepted' if review_accepted(review,candidate) else ('needs_review' if review.get('verdict')=='needs_review' else 'rejected')
        except Invalid as e:
            err=str(e)
            if err=='api_call_limit_reached' and candidate is not None: status='generated'
        except APIError as e:
            err=str(e)
            if candidate is not None: status='generated'
            if e.status in (400,401,402,403,404): fatal=e
        except Exception as e:
            err='local_exception:'+type(e).__name__
            if candidate is not None: status='generated'
        db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?)',(job['job_id'],status,
            json.dumps(candidate,ensure_ascii=False) if candidate is not None else None,
            json.dumps(review,ensure_ascii=False) if review is not None else None,err,json.dumps(usage,ensure_ascii=False)))
        db.commit(); statuses[status]+=1
        print(json.dumps({'job':job['job_id'][:12],'status':status,'error':err,'api_calls_this_run':client.calls},ensure_ascii=False),flush=True)
        if fatal: break
    overall=dict(db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status').fetchall())
    summary={'run_dir':str(run_dir),'processed_this_run':processed,'api_calls_this_run':client.calls,'statuses':overall,'tools_executed':False,
        'scope':'model-reviewed next-decision candidates; no full new tool trajectory generated'}
    write_json(run_dir/'summary.json',summary); db.close()
    if args.export: export_candidates(cfg,run_dir)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if fatal: raise fatal
    return run_dir

def render_candidate(job,candidate,c):
    d=candidate['decision']; row=copy.deepcopy(job['base_row']); tool=d['tool']; args=d['arguments']
    think={k:d[k] for k in ('rationale','information_need','gap_type','depends_on')}
    if tool=='finish':
        think['reasoning']=args['reasoning']; think['evidence_steps']=args['evidence_steps']; response='\n'.join(args['answers'])
        tail='<response>'+response+'</response>'
    elif tool=='abstain':
        response=args['reason']
        if args['missing_evidence']: response+='\nMissing evidence: '+'; '.join(args['missing_evidence'])
        tail='<response>'+response+'</response>'
    else:
        fn=native_tools(row,c)[tool]['exported_name']; tail='<tool_call>'+json.dumps({'name':fn,'arguments':args},ensure_ascii=False)+'</tool_call>'
    row['conversations'].append({'from':'gpt','value':'<think>'+json.dumps(think,ensure_ascii=False)+'</think>\n'+tail})
    row['teacher_decision_json']=json.dumps(d,ensure_ascii=False)
    row['id']=row['id']+':teacher-v1:'+job['job_id'][:12]
    row['sample_type']='openrouter_reviewed_next_decision'
    c.validate_row(row)
    return row

def export_candidates(cfg,run_dir):
    run_dir=Path(run_dir).resolve(); metadata=read_json(run_dir/'run_metadata.json')
    if Path(metadata['jobs_file']).resolve()!=Path(cfg['jobs_file']).resolve(): raise Invalid('export_jobs_mismatch')
    expected=run_fingerprint(cfg,metadata['teacher_model'],metadata['reviewer_model'])
    if metadata['fingerprint']!=expected: raise Invalid('export_configuration_or_prompt_changed')
    jobs={j['job_id']:j for j in load_jobs(cfg)}; c=cleaner(cfg)
    db=sqlite3.connect(run_dir/'results.sqlite'); rows=[]; provenance=[]; statuses=C.Counter()
    for jid,status,teacher,reviewer,error,usage in db.execute('SELECT id,status,teacher,reviewer,error,usage FROM jobs ORDER BY id'):
        statuses[status]+=1
        if status!='accepted': continue
        candidate=json.loads(teacher); review=json.loads(reviewer); job=jobs[jid]
        validate_candidate(candidate,job,c)
        if not review_accepted(review,candidate): raise Invalid('accepted_record_failed_revalidation')
        row=render_candidate(job,candidate,c); rows.append(row)
        provenance.append({'id':row['id'],'job_id':jid,'original':job['provenance'],'annotation':candidate['annotation'],'review':review,'usage':json.loads(usage or '[]'),
            'validation_scope':'next_decision_only','tools_executed':False})
    db.close()
    for filename,items in [('teacher_candidates.jsonl',rows),('candidate_provenance.jsonl',provenance)]:
        destination=run_dir/filename; temporary=destination.with_suffix(destination.suffix+'.tmp')
        with temporary.open('w') as f:
            for item in items: f.write(json.dumps(item,ensure_ascii=False)+'\n')
        temporary.replace(destination)
    info={'exported_candidates':len(rows),'statuses':dict(statuses),'output':str(run_dir/'teacher_candidates.jsonl'),
          'merged_into_clean_dataset':False,'full_new_trajectories_generated':False}
    write_json(run_dir/'export_summary.json',info); print(json.dumps(info,ensure_ascii=False,indent=2)); return rows

def cli(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','doctor','run','export'):
        p=sub.add_parser(name); p.add_argument('--config',type=Path,required=True)
        if name=='prepare': p.add_argument('--limit',type=int,default=100)
        if name in ('doctor','run'):
            p.add_argument('--model'); p.add_argument('--reviewer-model')
        if name=='doctor': p.add_argument('--online',action='store_true')
        if name=='run':
            p.add_argument('--limit',type=int,default=3); p.add_argument('--max-api-calls',type=int,default=8)
            p.add_argument('--dry-run',action='store_true'); p.add_argument('--retry-errors',action='store_true'); p.add_argument('--export',action='store_true')
        if name=='export': p.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(argv); cfg=config_at(args.config)
    if getattr(args,'limit',1)<1: parser.error('limit must be >=1')
    if args.command=='prepare': return prepare(cfg,args.limit)
    if args.command=='doctor': return doctor(cfg,args.online,args)
    if args.command=='run': return run(cfg,args)
    return export_candidates(cfg,args.run_dir)

if __name__=='__main__':
    try: cli()
    except (Invalid,APIError) as e:
        print('STOP:',str(e),file=sys.stderr); sys.exit(2)
