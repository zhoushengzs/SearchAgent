#!/usr/bin/env python3
"""Conservative, provenance-preserving ShareGPT Agent SFT cleaning. Python stdlib only.

Never overwrites inputs or an existing output directory. No model/network calls.
Original step IDs and original remaining budget are retained; deleted IDs alias an
older equivalent observation. Clean/review/removed partitions account for every row.
"""
import argparse
import collections as C
import copy
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

VERSION = '1.0.0'
CALL = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)
THINK = re.compile(r'<think>(.*?)</think>', re.S)
RESPONSE = re.compile(r'<response>(.*?)</response>', re.S)
STATE_MARKER = 'Recorded task state follows.'
REF_LIST_KEYS = {'depends_on', 'evidence_steps'}
REF_ONE_KEYS = {'revision_of_step', 'observed_at_step'}
PROSE_KEYS = {'rationale', 'information_need', 'reasoning', 'hypothesis', 'open_questions', 'visual_facts', 'language_facts'}
STEP_MENTION = re.compile(r'\b(?:steps?|turns?)\s*#?\s*(\d+)\b|第\s*(\d+)\s*步|步骤\s*[#：:]?\s*(\d+)', re.I)
STEP_GROUP = re.compile(r'\b(?:steps?|turns?)\s*#?\s*\d+(?:\s*(?:,|and|&|to|through|[-–—])\s*\d+)+', re.I)
EXPLICIT_AGENT_REF = re.compile(r'\b(?:agent|tool|execution|executed)\s+steps?\s*#?\s*(\d+)\b', re.I)
TRANSIENT_ERRORS = {'ConnectTimeout', 'ReadTimeout', 'Timeout', 'TimeoutError', 'ConnectionError', 'ConnectionResetError', 'RemoteDisconnected', 'SSLError', 'URLError', 'RequestException', 'ConnectionReset', 'HTTPTimeout'}
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}
DETERMINISTIC_ERRORS = {'ValueError', 'TypeError', 'KeyError', 'SyntaxError', 'ValidationError', 'JSONDecodeError', 'NotImplementedError', 'AssertionError'}
TIME_KEYS = {'observed_at', 'elapsed_ms', 'duration_ms'}

class NeedsReview(Exception):
    def __init__(self, code, details=None):
        self.code = code
        self.details = details
        super().__init__(code)

def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def sha(x):
    return hashlib.sha256(canon(x).encode('utf-8')).hexdigest()

def file_sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def parse_calls(text):
    out = []
    for match in CALL.finditer(text):
        obj = json.loads(match.group(1))
        if not isinstance(obj, dict) or not isinstance(obj.get('name'), str) or not isinstance(obj.get('arguments'), dict):
            raise NeedsReview('invalid_tool_schema')
        out.append(obj)
    return out

def action_key(call):
    return canon([call['name'], call['arguments']])

def parse_state(row):
    first = row['conversations'][0]
    value = first['value']
    if STATE_MARKER not in value:
        return None, None
    marker = value.index(STATE_MARKER)
    pos = value.find('\n{', marker)
    if pos < 0:
        raise NeedsReview('unparseable_state')
    return value[:pos+1], json.loads(value[pos+1:])

def history(row):
    messages = row['conversations']
    if not messages or messages[-1]['from'] != 'gpt':
        raise NeedsReview('non_assistant_target')
    result = []
    pending = None
    for idx, msg in enumerate(messages[1:-1], 1):
        if msg['from'] == 'gpt':
            calls = parse_calls(msg['value'])
            if len(calls) != 1 or pending is not None:
                raise NeedsReview('invalid_history_action_pair')
            pending = (idx, calls[0])
        elif msg['from'] == 'observation':
            if pending is None:
                raise NeedsReview('unpaired_observation')
            observation = json.loads(msg['value'])
            step = observation.get('step')
            if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
                raise NeedsReview('invalid_history_step')
            mi, call = pending
            result.append({'step': step, 'call': call, 'key': action_key(call), 'observation': observation, 'message_index': mi,
                           'result': observation.get('result', {}), 'assistant': messages[mi]})
            pending = None
        else:
            raise NeedsReview('unsupported_history_role', msg['from'])
    if pending is not None:
        raise NeedsReview('unpaired_history_action')
    steps = [a['step'] for a in result]
    if steps != sorted(set(steps)):
        raise NeedsReview('non_monotonic_history_steps')
    return result

def canonical_result(result):
    # Only discard top-level timing metadata. All content, IDs, snippets and URLs remain.
    return {k:v for k,v in result.items() if k not in TIME_KEYS}

def canonical_observer(observation):
    obs = copy.deepcopy(observation.get('observation'))
    if isinstance(obs, dict):
        # revision_of_step is remapped as a dependency; not itself content evidence.
        obs.pop('revision_of_step', None)
    return obs

def temporary_failure(result):
    if result.get('status') != 'error':
        return False
    code = result.get('http_status')
    err = result.get('error_type')
    # An explicit deterministic class or non-retryable HTTP status takes precedence.
    if err in DETERMINISTIC_ERRORS:
        return False
    if isinstance(code, int):
        return code in TRANSIENT_HTTP
    return err in TRANSIENT_ERRORS

def deterministic_failure(result):
    if result.get('status') != 'error':
        return False
    code = result.get('http_status')
    return result.get('error_type') in DETERMINISTIC_ERRORS or (isinstance(code, int) and 400 <= code < 500 and code not in TRANSIENT_HTTP)

def classify_retry(hist, current_call, max_extra_retries=1):
    """Current action has no outcome: use only previous real observations."""
    if not hist or action_key(current_call) != hist[-1]['key']:
        return False, 'not_adjacent_repeat'
    previous = hist[-1]
    if previous['result'].get('status') == 'ok':
        return False, 'repeated_successful_action'
    if deterministic_failure(previous['result']):
        return False, 'repeated_deterministic_failure'
    if not temporary_failure(previous['result']):
        return False, 'unclassified_failure_requires_review'
    count = 0
    for a in reversed(hist):
        if a['key'] != previous['key'] or not temporary_failure(a['result']):
            break
        count += 1
    if count <= max_extra_retries:
        return True, 'observation_grounded_transient_retry'
    return False, 'transient_retry_budget_exceeded'

def attachment_signatures(state):
    by_step = C.defaultdict(set)
    if not state:
        return by_step
    for image in state.get('attached_images', []):
        sig = {k:v for k,v in image.items() if k not in ('image_index', 'observed_at_step')}
        by_step[image.get('observed_at_step', 0)].add(canon(sig))
    return by_step

def equivalent_step(candidate, representative, attachments):
    if candidate['result'].get('status') not in ('ok', 'error') or representative['result'].get('status') not in ('ok', 'error'):
        return False, 'unknown_result_status'
    if canon(canonical_result(candidate['result'])) != canon(canonical_result(representative['result'])):
        return False, 'different_tool_result'
    if canon(canonical_observer(candidate['observation'])) != canon(canonical_observer(representative['observation'])):
        return False, 'observer_equivalence_unconfirmed'
    if not attachments[candidate['step']].issubset(attachments[representative['step']]):
        return False, 'unique_or_unconfirmed_image_attachment'
    return True, 'equivalent_result_observer_and_images'

def make_plan(row, max_extra_retries=1):
    _, state = parse_state(row)
    hist = history(row)
    attachments = attachment_signatures(state)
    aliases = {}
    uncertain = {}
    retained_retries = []
    step_entries = []
    run_anchor = None
    by_step = {a['step']: a for a in hist}
    for idx, current in enumerate(hist):
        previous = hist[idx-1] if idx else None
        contiguous = previous is not None and current['step'] == previous['step']+1 and current['key'] == previous['key']
        if not contiguous:
            run_anchor = current
            continue
        retry, retry_reason = classify_retry(hist[:idx], current['call'], max_extra_retries)
        if retry:
            retained_retries.append(current['step'])
            if current['result'].get('status') == 'ok':
                run_anchor = current
            step_entries.append({'step':current['step'], 'route':'keep_observation_grounded_retry', 'previous_step':previous['step'], 'reason':retry_reason})
            continue
        # Preserve an actually successful recovery after a failure even if it exceeded
        # the training-target retry budget; do not erase uniquely successful evidence.
        if previous['result'].get('status') == 'error' and current['result'].get('status') == 'ok':
            retained_retries.append(current['step'])
            run_anchor = current
            step_entries.append({'step':current['step'], 'route':'keep_actual_failure_recovery_history', 'previous_step':previous['step'], 'reason':retry_reason})
            continue
        anchor = run_anchor
        eq, why = equivalent_step(current, anchor, attachments)
        if eq and retry_reason != 'unclassified_failure_requires_review':
            aliases[current['step']] = anchor['step']
            step_entries.append({'step':current['step'], 'route':'merge_equivalent_contiguous_duplicate', 'representative_step':anchor['step'], 'reason':why})
        else:
            uncertain[current['step']] = retry_reason if retry_reason == 'unclassified_failure_requires_review' else why
            step_entries.append({'step':current['step'], 'route':'review_unique_or_unconfirmed_evidence', 'representative_step':anchor['step'], 'reason':uncertain[current['step']]})
    return {'aliases':aliases, 'uncertain':uncertain, 'retained_retries':retained_retries, 'steps':step_entries,
            'history_steps':[a['step'] for a in hist]}

def map_reference(n, aliases):
    if not isinstance(n, int) or isinstance(n, bool):
        raise NeedsReview('non_integer_step_reference', n)
    visited = set()
    while n in aliases:
        if n in visited:
            raise NeedsReview('cyclic_alias')
        visited.add(n)
        n = aliases[n]
    return n

def rewrite_prose(text, aliases):
    if not aliases or not isinstance(text, str):
        return text
    # Map explicitly labelled agent/tool execution references. Bare "step 7" may
    # mean a manual/build/recipe step, so do not guess its meaning.
    def explicit(match):
        n = int(match.group(1))
        return match.group(0)[:match.start(1)-match.start()] + str(map_reference(n,aliases)) + match.group(0)[match.end(1)-match.start():]
    updated = EXPLICIT_AGENT_REF.sub(explicit, text)
    for group in STEP_GROUP.finditer(updated):
        if any(int(n) in aliases for n in re.findall(r'\d+', group.group(0))):
            raise NeedsReview('ambiguous_natural_language_step_group', group.group(0))
    for match in STEP_MENTION.finditer(updated):
        n = int(next(v for v in match.groups() if v is not None))
        if n in aliases:
            raise NeedsReview('ambiguous_natural_language_step_reference', match.group(0))
    return updated

def rewrite_fields(value, aliases, prose=False):
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if key in REF_LIST_KEYS:
                if not isinstance(val, list):
                    raise NeedsReview('invalid_reference_list', key)
                out[key] = list(dict.fromkeys(map_reference(x,aliases) for x in val))
            elif key in REF_ONE_KEYS:
                out[key] = None if val is None else map_reference(val,aliases)
            else:
                out[key] = rewrite_fields(val,aliases, prose or key in PROSE_KEYS)
        return out
    if isinstance(value,list):
        return [rewrite_fields(v,aliases,prose) for v in value]
    return rewrite_prose(value,aliases) if prose and isinstance(value,str) else value

def validate_references(value, allowed):
    if isinstance(value,dict):
        for key,val in value.items():
            if key in REF_LIST_KEYS:
                if not isinstance(val,list) or any(not isinstance(v,int) or isinstance(v,bool) or v not in allowed for v in val):
                    raise NeedsReview('dangling_structured_reference', {'field':key,'value':val})
            elif key in REF_ONE_KEYS:
                if val is not None and (not isinstance(val,int) or isinstance(val,bool) or val not in allowed):
                    raise NeedsReview('dangling_structured_reference', {'field':key,'value':val})
            else:
                validate_references(val,allowed)
    elif isinstance(value,list):
        for val in value:
            validate_references(val,allowed)

def rewrite_assistant(message, aliases, allowed):
    value = message['value']
    def replace_think(match):
        payload = match.group(1)
        if not payload.strip():
            return match.group(0)
        obj = rewrite_fields(json.loads(payload),aliases)
        validate_references(obj,allowed)
        return '<think>'+json.dumps(obj,ensure_ascii=False)+'</think>'
    value = THINK.sub(replace_think,value)
    # Tool arguments are immutable, except explicit finish/evidence control fields.
    # Actual tool calls in this package are retrieval/inspection/calculation calls.
    for call in parse_calls(value):
        if call['name'] in ('finish','abstain'):
            raise NeedsReview('unexpected_terminal_tool_serialization')
    def replace_response(match):
        return '<response>'+rewrite_prose(match.group(1),aliases)+'</response>'
    value = RESPONSE.sub(replace_response,value)
    out = dict(message)
    out['value'] = value
    return out

def rebuild_row(row, plan):
    prefix, state = parse_state(row)
    hist = history(row)
    if state is None:
        if hist:
            raise NeedsReview('history_without_state')
        return row, {'merged_history_actions':0,'mapped_structured_references':0}
    original_steps = [a['step'] for a in hist]
    if state.get('executed_prefix_step_ids') != original_steps:
        raise NeedsReview('original_state_history_mismatch')
    for a in hist:
        if a['step'] in plan['uncertain']:
            raise NeedsReview('history_contains_unconfirmed_duplicate', {'step':a['step'],'reason':plan['uncertain'][a['step']]})
    aliases = {k:v for k,v in plan['aliases'].items() if k in original_steps}
    # Validate equivalence again against this exact row's image inventory.
    attachments = attachment_signatures(state)
    by_step = {a['step']:a for a in hist}
    for step, representative in aliases.items():
        if representative not in by_step:
            raise NeedsReview('missing_representative_step', step)
        eq, why = equivalent_step(by_step[step],by_step[representative],attachments)
        if not eq:
            raise NeedsReview('row_local_equivalence_unconfirmed', {'step':step,'reason':why})
    result = copy.deepcopy(row)
    new_messages = [copy.deepcopy(row['conversations'][0])]
    allowed = {0}
    for a in hist:
        if a['step'] in aliases:
            continue
        new_messages.append(rewrite_assistant(a['assistant'], aliases,allowed))
        observation = copy.deepcopy(a['observation'])
        # Preserve all original tool result bytes/values. Rewrite observer control
        # references and explicit prose only, never retrieved content.
        if 'observation' in observation:
            observation['observation'] = rewrite_fields(observation['observation'],aliases)
            validate_references(observation['observation'],allowed)
        new_messages.append({'from':'observation','value':json.dumps(observation,ensure_ascii=False)})
        allowed.add(a['step'])
    new_messages.append(rewrite_assistant(row['conversations'][-1],aliases,allowed))
    if row.get('teacher_decision_json'):
        decision = rewrite_fields(json.loads(row['teacher_decision_json']),aliases)
        validate_references(decision,allowed)
        result['teacher_decision_json'] = json.dumps(decision,ensure_ascii=False)
    new_state = copy.deepcopy(state)
    new_state['executed_prefix_step_ids'] = [a['step'] for a in hist if a['step'] not in aliases]
    for field in ('attached_images','image_inventory','initial_views'):
        if field in new_state:
            new_state[field] = rewrite_fields(new_state[field],aliases)
            validate_references(new_state[field],allowed)
    # ID, step, remaining_decisions, images order and raw tool arguments remain
    # stable. Gaps explicitly represent removed redundant events, not new time.
    new_messages[0]['value'] = prefix + json.dumps(new_state,ensure_ascii=False)
    result['conversations'] = new_messages
    validate_row(result)
    if result.get('images') != row.get('images'):
        raise AssertionError('image list mutated')
    if result.get('step') != row.get('step') or new_state.get('remaining_decisions') != state.get('remaining_decisions'):
        raise AssertionError('original step/budget mutated')
    return result, {'merged_history_actions':len(aliases)}

def validate_row(row):
    prefix,state = parse_state(row)
    hist = history(row)
    if state is None:
        return
    assert [a['step'] for a in hist] == state.get('executed_prefix_step_ids')
    image_indices = [int(n) for n in re.findall(r'img_(\d+): <image>',prefix)]
    if image_indices != list(range(1,len(row.get('images',[]))+1)):
        raise NeedsReview('image_placeholder_mismatch')
    attached = state.get('attached_images',[])
    if [im.get('image_index') for im in attached] != image_indices:
        raise NeedsReview('attached_image_index_mismatch')
    allowed={0}
    for a in hist:
        for match in THINK.finditer(a['assistant']['value']):
            if match.group(1).strip():
                validate_references(json.loads(match.group(1)),allowed)
        validate_references(a['observation'].get('observation'),allowed)
        allowed.add(a['step'])
    validate_references(attached,allowed)
    for match in THINK.finditer(row['conversations'][-1]['value']):
        if match.group(1).strip():
            validate_references(json.loads(match.group(1)),allowed)
    if row.get('teacher_decision_json'):
        decision = json.loads(row['teacher_decision_json'])
        validate_references(decision,allowed)
        calls=parse_calls(row['conversations'][-1]['value'])
        if calls:
            if len(calls)!=1:
                raise NeedsReview('multi_action_target')
            expected='vs_'+decision['tool'] if decision['tool'] in ('web_search','image_search','visit','inspect_image','crop','calculate') else decision['tool']
            if calls[0]['name']!=expected or canon(calls[0]['arguments'])!=canon(decision.get('arguments',{})):
                raise NeedsReview('teacher_target_mismatch')


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--max-extra-retries',type=int,default=1)
    args=parser.parse_args(argv)
    source=args.input.resolve(); output=args.output.resolve(); audit=args.audit_dir.resolve()
    if not source.is_dir() or any(not (source/(s+'.jsonl')).is_file() for s in ('train','val')):
        parser.error('input must contain train.jsonl and val.jsonl')
    if output==source or output in source.parents or output.exists() or audit.exists():
        parser.error('use new output and audit directories; no overwrite allowed')
    if audit==output or audit in output.parents or output in audit.parents:
        parser.error('output and audit must be separate directories')
    if args.max_extra_retries<0:
        parser.error('max-extra-retries must be >=0')
    output.mkdir(parents=True); audit.mkdir(parents=True)
    for directory in ('review','removed_targets','configs'):
        (output/directory).mkdir()
    started=time.time()
    stat_before={s:(source/(s+'.jsonl')).stat() for s in ('train','val')}
    input_hashes={}; input_counts={}; parse_errors={}; signature_conflicts=set(); trajectory_splits={}
    tmp=tempfile.TemporaryDirectory(prefix='sft_cleaner_')
    db=sqlite3.connect(str(Path(tmp.name)/'longest.sqlite'))
    db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF')
    db.execute('CREATE TABLE longest (tid TEXT PRIMARY KEY, split TEXT, line INTEGER, histlen INTEGER, payload TEXT)')
    maxlens={}; signatures={}; image_paths=set(); input_final=C.Counter(); rationale_stats=C.Counter()
    print('PASS 1: index longest visible histories and verify source consistency',flush=True)
    for split in ('train','val'):
        h=hashlib.sha256(); count=0
        with (source/(split+'.jsonl')).open('rb') as f:
            for line_no,raw in enumerate(f,1):
                count+=1; h.update(raw)
                try:
                    row=json.loads(raw); tid=row.get('trajectory_id') or row['id'].split(':')[0]
                    if tid in trajectory_splits and trajectory_splits[tid]!=split:
                        signature_conflicts.add(tid)
                    trajectory_splits[tid]=split
                    hist=history(row)
                    calls=parse_calls(row['conversations'][-1]['value'])
                    if len(calls)>1: raise NeedsReview('multi_action_target')
                    is_final=not calls
                    input_final[split]+=is_final
                    for image in row.get('images',[]): image_paths.add(image)
                    for a in hist:
                        key=(tid,a['step'])
                        signature=sha([a['call'],a['observation'],a['assistant']['value']])
                        if key in signatures and signatures[key]!=signature:
                            signature_conflicts.add(tid)
                        signatures[key]=signature
                    # Identical rationale alone is measured, never a deletion rule.
                    if hist:
                        last_think=THINK.search(row['conversations'][-1]['value'])
                        prev_think=THINK.search(hist[-1]['assistant']['value'])
                        if last_think and prev_think and last_think.group(1).strip() and prev_think.group(1).strip():
                            lt=json.loads(last_think.group(1)); pt=json.loads(prev_think.group(1))
                            if lt.get('rationale') and lt.get('rationale')==pt.get('rationale'):
                                rationale_stats[split+'_identical_rationale_targets']+=1
                                if calls and action_key(calls[0])!=hist[-1]['key']:
                                    rationale_stats[split+'_same_rationale_different_action_preserved_rule']+=1
                    if tid not in maxlens or len(hist)>maxlens[tid]:
                        maxlens[tid]=len(hist)
                        db.execute('INSERT OR REPLACE INTO longest VALUES (?,?,?,?,?)',(tid,split,line_no,len(hist),raw.decode('utf-8')))
                except (NeedsReview,ValueError,KeyError,TypeError) as e:
                    code=e.code if isinstance(e,NeedsReview) else 'input_parse_error'
                    parse_errors[(split,line_no)]={'code':code,'details':str(e)}
        input_hashes[split]=h.hexdigest(); input_counts[split]=count
        db.commit(); print(' indexed',split,count,'rows',flush=True)
    del signatures
    plans={}; plan_counts=C.Counter(); recovery_counts=C.Counter()
    with (audit/'trajectory_plans.jsonl').open('w') as pf:
        for tid,split,line_no,payload in db.execute('SELECT tid,split,line,payload FROM longest ORDER BY tid'):
            row=json.loads(payload)
            try:
                plan=make_plan(row,args.max_extra_retries)
            except (NeedsReview,ValueError,KeyError,TypeError) as e:
                plan={'aliases':{},'uncertain':{},'retained_retries':[],'steps':[],'history_steps':[], 'plan_error':str(e)}
            if tid in signature_conflicts:
                plan['plan_error']='conflicting_history_or_cross_split_trajectory'
            plans[tid]=plan
            plan_counts['trajectories']+=1; plan_counts['mergeable_unique_executed_steps']+=len(plan['aliases'])
            plan_counts['unconfirmed_unique_executed_steps']+=len(plan['uncertain'])
            for entry in plan['steps']:
                recovery_counts[entry['route']]+=1
            pf.write(json.dumps({'trajectory_id':tid,'split':split,'longest_source_line':line_no,**plan},ensure_ascii=False)+'\n')
    db.close(); tmp.cleanup()
    split_counts={}; review_reasons={}; route_final=C.Counter(); modifications=C.Counter(); clean_unique_pairs=set(); missing_images=[]
    with (audit/'row_manifest.jsonl').open('w') as manifest:
        for split in ('train','val'):
            counts=C.Counter(); reasons=C.Counter(); hashes={r:hashlib.sha256() for r in ('clean','review','removed')}
            paths={'clean':output/(split+'.jsonl'),'review':output/'review'/(split+'.jsonl'),'removed':output/'removed_targets'/(split+'.jsonl')}
            handles={r:p.open('wb') for r,p in paths.items()}
            print('PASS 2:',split,'rewrite and partition',flush=True)
            with (source/(split+'.jsonl')).open('rb') as f:
                for ln,raw in enumerate(f,1):
                    counts['input_rows']+=1; route='clean'; reason='unchanged_or_safe_rebuild'; detail=None; merged=0; target_retry=False
                    row=None; is_final=False; original_ids=[]; tid=None
                    try:
                        if (split,ln) in parse_errors:
                            err=parse_errors[(split,ln)]; raise NeedsReview(err['code'],err['details'])
                        row=json.loads(raw); tid=row.get('trajectory_id') or row['id'].split(':')[0]
                        plan=plans[tid]
                        calls=parse_calls(row['conversations'][-1]['value']); hist=history(row); is_final=not calls
                        original_ids=[a['step'] for a in hist]
                        if calls and hist and action_key(calls[0])==hist[-1]['key']:
                            allowed,why=classify_retry(hist,calls[0],args.max_extra_retries)
                            if allowed:
                                target_retry=True
                            elif why=='unclassified_failure_requires_review':
                                raise NeedsReview(why)
                            else:
                                route='removed'; reason=why
                        if route=='clean':
                            if plan.get('plan_error'): raise NeedsReview('trajectory_plan_error',plan['plan_error'])
                            result,changes=rebuild_row(row,plan)
                            merged=changes['merged_history_actions']
                            validate_row(result)
                            # Ensure tool parameter strings and all retained raw results
                            # remain exact, and original nonconsecutive repeats survive.
                            cleaned_hist=history(result)
                            original_by_step={a['step']:a for a in hist}
                            for a in cleaned_hist:
                                before=original_by_step[a['step']]
                                assert a['key']==before['key']
                                assert a['result']==before['result']
                            assert all(step in {a['step'] for a in cleaned_hist} for step in original_ids if step not in plan['aliases'])
                            target_calls=parse_calls(result['conversations'][-1]['value'])
                            assert [action_key(c) for c in target_calls]==[action_key(c) for c in calls]
                            if target_calls and cleaned_hist and action_key(target_calls[0])==cleaned_hist[-1]['key'] and not target_retry:
                                raise NeedsReview('new_unsupported_adjacent_repeat')
                            for a in hist:
                                if a['step'] in plan['aliases']:
                                    clean_unique_pairs.add((tid,a['step']))
                            modifications['history_action_occurrences_removed']+=merged
                            modifications['rows_with_history_merges']+=bool(merged)
                            modifications['observation_grounded_retry_targets_kept']+=target_retry
                            # byte-preserve unchanged no-history auxiliaries, serialize
                            # all structured decisions after reference validation.
                            encoded=(json.dumps(result,ensure_ascii=False)+'\n').encode('utf-8') if hist or row.get('trajectory_id') else raw
                            if result!=row: modifications['modified_rows']+=1
                        else:
                            encoded=raw
                    except NeedsReview as e:
                        route='review'; reason=e.code; detail=e.details; encoded=raw
                    except (ValueError,KeyError,TypeError,AssertionError) as e:
                        route='review'; reason='rebuild_or_validation_error'; detail=repr(e); encoded=raw
                    if route!='clean': encoded=raw
                    handles[route].write(encoded); hashes[route].update(encoded)
                    counts[route+'_rows']+=1; reasons[route+':'+reason]+=1
                    if is_final:
                        counts[route+'_final_answer_rows']+=1; route_final[(split,route)]+=1
                        assert route!='removed','final answer must never be discarded'
                    record={'file':split+'.jsonl','line':ln,'id':row.get('id') if row else None,'trajectory_id':tid,'step':row.get('step') if row else None,
                            'route':route,'reason':reason,'details':detail,'is_final_answer':is_final,'history_merges_applied':merged if route=='clean' else 0,
                            'allowed_retry_target':target_retry if route=='clean' else False,
                            'source_sha256':hashlib.sha256(raw).hexdigest()}
                    manifest.write(json.dumps(record,ensure_ascii=False)+'\n')
            for h in handles.values(): h.close()
            assert sum(counts[r+'_rows'] for r in ('clean','review','removed'))==input_counts[split]
            assert counts['clean_final_answer_rows']+counts['review_final_answer_rows']==input_final[split]
            counts['input_final_answer_rows']=input_final[split]
            split_counts[split]=dict(counts); review_reasons[split]=dict(reasons)
            for route,p in paths.items():
                assert file_sha(p)==hashes[route].hexdigest()
            print(' partition',split,dict(counts),flush=True)
    # Preserve the input manifest and original export metadata, but prepare clean
    # dataset configs that resolve images against the unmodified source media root.
    shutil.copyfile(source/'dataset_info.json',output/'dataset_info.json')
    for path in (source/'configs').glob('*.json'):
        destination=output/'configs'/path.name
        if path.name=='export_notes.json':
            destination=output/'configs'/'original_export_notes.json'
        if path.name in ('full.json','lora.json'):
            config=json.loads(path.read_text())
            config['dataset_dir']=str(output); config['media_dir']=str(source)
            config['output_dir']=str(output/'runs'/('agent_full' if path.name=='full.json' else 'agent_lora'))
            if 'deepspeed' in config: config['deepspeed']=str(output/'configs'/'zero3.json')
            write_json(destination,config)
        else:
            shutil.copyfile(path,destination)
    write_json(output/'configs'/'export_notes.json',{'cleaner_version':VERSION,'source':str(source),'audit_dir':str(audit),
        'counts':split_counts,'mask_history':True,'original_step_ids_preserved':True,'remaining_decisions':'original execution budget, unchanged',
        'review_and_removed_rows':'original rows preserved outside the clean train/val files',
        'images':'media_dir points to source; no image files fabricated or copied','teacher_model_called':False})
    for image in sorted(image_paths):
        path=Path(image) if Path(image).is_absolute() else source/image
        if not path.is_file(): missing_images.append(image)
    write_json(audit/'missing_image_paths.json',{'media_dir':str(source),'unique_referenced_paths':len(image_paths),'missing_count':len(missing_images),'missing_paths':missing_images})
    source_unchanged={}
    for split in ('train','val'):
        path=source/(split+'.jsonl'); after=path.stat()
        assert (after.st_size,after.st_mtime_ns)==(stat_before[split].st_size,stat_before[split].st_mtime_ns)
        end_sha=file_sha(path); assert end_sha==input_hashes[split]
        source_unchanged[split]={'path':str(path),'sha256_before':input_hashes[split],'sha256_after':end_sha,'unchanged':True}
    summary={'cleaner_version':VERSION,'input':str(source),'output':str(output),'audit_dir':str(audit),
        'policy':{'strict_action_equality':True,'first_equivalent_observation_retained':True,'max_extra_transient_retries':args.max_extra_retries,
                  'observer_equivalence':'exact content except revision_of_step provenance','tool_result_equivalence':'exact except top-level timing',
                  'rationale_only_deletion':False,'nonconsecutive_repeats_preserved':True,'original_step_ids_and_budget_preserved':True,
                  'unconfirmed_evidence':'original row in review partition','final_answers':'all kept in clean or review',
                  'natural_language_references':'explicit agent/tool/execution step references mapped; ambiguous deleted-step mentions queued'},
        'counts':split_counts,'plan_counts':dict(plan_counts),'plan_routes':dict(recovery_counts),'modifications':dict(modifications),
        'unique_steps_removed_from_released_clean_rows':len(clean_unique_pairs),'routes_and_reasons':review_reasons,'rationale_audit':dict(rationale_stats),
        'source_integrity':source_unchanged,'structural_validation':'every released clean row passed rebuilding, reference, image index, tool-target and immutable-result checks',
        'image_availability':{'unique_paths':len(image_paths),'missing_paths':len(missing_images),'details':str(audit/'missing_image_paths.json')},
        'teacher_model_called':False,'training_run':False,'completed':True,'elapsed_seconds':round(time.time()-started,2)}
    write_json(audit/'summary.json',summary)
    write_json(output/'cleaning_manifest.json',{'completed':True,'source':str(source),'audit_dir':str(audit),'counts':split_counts,'source_integrity':source_unchanged})
    print('COMPLETED',json.dumps(summary['counts'],ensure_ascii=False),flush=True)
    print('AUDIT',audit,flush=True)
    return summary

if __name__=='__main__':
    main()
