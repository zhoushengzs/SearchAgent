"""Read-only verification of every prepared task against its source."""
import collections, hashlib, json
from pathlib import Path
import teacher_stage2 as s

def verify():
    here=Path(__file__).resolve().parent
    cfg=s.config_at(here/'config.json'); c=s.cleaner(cfg); jobs=s.load_jobs(cfg)
    source_lines=collections.defaultdict(set)
    for j in jobs: source_lines[j['provenance']['file']].add(j['provenance']['line'])
    originals={}
    for file,lines in source_lines.items():
        for ln,r in s.jsonlines(file):
            if ln in lines: originals[(file,ln)]=r
            if ln>=max(lines): break
    seen=set(); kinds=collections.Counter(); image_count=0; changed=0
    for j in jobs:
        pr=j['provenance']; b=j['base_row']; r=originals[(pr['file'],pr['line'])]
        assert pr['trajectory_id'] not in seen; seen.add(pr['trajectory_id'])
        assert pr['numbering_version']=='v2_contiguous'
        assert 'teacher_decision_json' not in b
        assert b['conversations'][-1]['from']=='observation' or len(b['conversations'])==1
        before=c.history(r); after=s.history_from_base(b,c)
        assert [x['step'] for x in after]==list(range(1,len(after)+1))
        assert b['step']==len(after)+1
        oldstate=c.parse_state(r)[1]; state=c.parse_state(b)[1]
        assert state['executed_prefix_step_ids']==list(range(1,len(after)+1))
        assert oldstate['question']==state['question']
        assert oldstate['remaining_decisions']==state['remaining_decisions']
        assert b.get('images')==r.get('images')
        assert b.get('tools')==r.get('tools') and b.get('system')==r.get('system')
        if pr['kind']=='control':
            assert b['conversations']==r['conversations'][:-1]
            retained=before
        else:
            plan=c.make_plan(r)
            retained=[a for a in before if a['step'] not in plan['aliases']]
        assert len(retained)==len(after)
        for x,y in zip(retained,after):
            assert x['call']==y['call'] and x['result']==y['result']
        assert set(b)==(set(r)-{'teacher_decision_json'})
        allowed={0}
        for a in after:
            for text in s.numbering.THINK.findall(a['assistant']['value']):
                if text.strip(): c.validate_references(json.loads(text),allowed)
            c.validate_references(a['observation'].get('observation'),allowed)
            allowed.add(a['step'])
        for field in ('initial_views','image_inventory','attached_images'):
            c.validate_references(state.get(field,[]),allowed)
        kinds[pr['kind']]+=1
        image_count+=len(s.preflight_images(j,cfg))
        changed+=int(pr['source_step']!=pr['prepared_step'])
    assert len(jobs)==100 and kinds=={'recovery':60,'control':40}
    return {'passed':True,'jobs':len(jobs),'unique_trajectories':len(seen),'by_source':dict(kinds),'images_checked':image_count,'current_step_changed':changed,'numbering_version':'v2_contiguous','old_jobs_reusable':False,'model_calls':0}
if __name__=='__main__': print(json.dumps(verify(),ensure_ascii=False,indent=2))
