import argparse
import base64
import copy
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'cleaning'))
from test_clean_sft import row
import teacher_stage2 as s

CLEANER=Path(__file__).resolve().parent.parent/'cleaning'/'clean_sft.py'
FUNCTIONS=[{'type':'function','function':{'name':'vs_web_search','description':'search','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}},
{'type':'function','function':{'name':'vs_inspect_image','description':'inspect','parameters':{'type':'object','properties':{'asset_id':{'type':'string'},'image_url':{'type':'string'}},'required':[]}}},
{'type':'function','function':{'name':'vs_crop','description':'crop','parameters':{'type':'object','properties':{'asset_id':{'type':'string'},'bbox':{'type':'array','items':{'type':'number','minimum':0,'maximum':1},'minItems':4,'maxItems':4}},'required':['asset_id','bbox']}}}]

def configured_row():
    r=row(['A'],target='TARGET_SHOULD_NOT_LEAK')
    prefix,state=s.cleaner({'cleaner_script':str(CLEANER)}).parse_state(r)
    state['tool_configuration']['available_tools']=['web_search','inspect_image','crop','finish','abstain']
    state['image_inventory']=[{'asset_id':'image_1'}]
    r['conversations'][0]['value']=prefix+json.dumps(state); r['tools']=json.dumps(FUNCTIONS)
    return r

def candidate():
    return {'decision':{'tool':'web_search','arguments':{'query':'B refined'},'rationale':'Need a source matching the unresolved identity.',
        'information_need':'source identity','gap_type':'language','depends_on':[1]},
        'annotation':{'behavior':'refine_query','revisit_steps':[],'revised_claim_step':None,'trigger_evidence':[],
        'prior_claim':None,'claim_update':'not_applicable','why_existing_evidence_is_insufficient':'No matched primary source.',
        'expected_decision_effect':'Locate a matching source.','retry_observation_step':None}}

def review():
    return {'verdict':'accept','validated_scope':'next_decision_only','future_information_used':False,'tool_result_fabricated':False,
            'references_valid':True,'action_necessary_or_reasonable':True,'image_access_sufficient':True,'grounded_trigger':True}

class FakeClient:
    def __init__(self,fail_review=False): self.calls=0; self.fail_review=fail_review
    def chat(self,payload):
        self.calls+=1
        if payload['messages'][0]['content']=='ROLE:reviewer':
            if self.fail_review: raise s.APIError(0,'simulated_unknown_transport')
            result=review()
        else:
            assert 'TARGET_SHOULD_NOT_LEAK' not in json.dumps(payload)
            result=candidate()
        return json.dumps(result),{'id':'mock'+str(self.calls),'model':'fake/vision','usage':{'prompt_tokens':100,'completion_tokens':20}}

class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.p=Path(self.tmp.name)
        (self.p/'prompts').mkdir()
        for role in ('teacher','reviewer'): (self.p/'prompts'/(role+'.txt')).write_text('ROLE:'+role)
        self.cfg={'_config_dir':str(self.p),'cleaner_script':str(CLEANER),'jobs_file':str(self.p/'jobs.jsonl'),
                  'cleaned_dir':str(self.p/'cleaned'),'media_dir':str(self.p),'runs_dir':str(self.p/'runs'),
                  'teacher_model':'fake/vision','reviewer_model':'fake/vision','max_context_characters':180000}
        self.c=s.cleaner(self.cfg)
        self.job=s.job_for(configured_row(),self.p/'train.jsonl',1,'recovery',self.c)
        Path(self.cfg['jobs_file']).write_text(json.dumps(self.job)+'\n')
        self.args=argparse.Namespace(model=None,reviewer_model=None,dry_run=False,limit=1,max_api_calls=4,retry_errors=False,export=True)
    def tearDown(self): self.tmp.cleanup()
    def test_original_target_removed(self):
        self.assertNotIn('teacher_decision_json',self.job['base_row'])
        self.assertNotIn('TARGET_SHOULD_NOT_LEAK',json.dumps(s.context_job(self.job,self.c)))
    def test_valid_refined_search(self): self.assertTrue(s.validate_candidate(candidate(),self.job,self.c)['valid'])
    def test_adjacent_repeat_rejected(self):
        x=candidate(); x['decision']['arguments']['query']='A'
        with self.assertRaises(s.Invalid): s.validate_candidate(x,self.job,self.c)
    def test_nonconsecutive_revisit_allowed(self):
        r=configured_row(); r2=row(['A','B'],target='Z'); r2['tools']=r['tools']
        prefix,st=self.c.parse_state(r2); st['tool_configuration']['available_tools']=['web_search','finish']; r2['conversations'][0]['value']=prefix+json.dumps(st)
        j=s.job_for(r2,self.p/'train.jsonl',1,'control',self.c)
        x=candidate(); x['decision']['arguments']['query']='A'
        self.assertTrue(s.validate_candidate(x,j,self.c)['valid'])
    def test_missing_evidence_step_rejected(self):
        x=candidate(); x['decision']['depends_on']=[99]
        with self.assertRaises(self.c.NeedsReview): s.validate_candidate(x,self.job,self.c)
    def test_revisit_needs_trigger(self):
        x=candidate(); x['annotation']['behavior']='revisit_resource'; x['annotation']['revisit_steps']=[1]
        with self.assertRaises(s.Invalid): s.validate_candidate(x,self.job,self.c)
    def test_unknown_url_rejected(self):
        # Supplied wrapper tools do not include visit, so schema itself rejects it.
        x=candidate(); x['decision']['tool']='visit'; x['decision']['arguments']={'url':'https://unknown.test'}
        with self.assertRaises(s.Invalid): s.validate_candidate(x,self.job,self.c)
    def test_optional_null_removed_and_asset_checked(self):
        x=candidate(); x['decision']['tool']='inspect_image'; x['decision']['arguments']={'asset_id':'image_1','image_url':None}
        s.validate_candidate(x,self.job,self.c); self.assertEqual(x['decision']['arguments'],{'asset_id':'image_1'})
        x['decision']['arguments']['asset_id']='made_up'
        with self.assertRaises(s.Invalid): s.validate_candidate(x,self.job,self.c)
    def test_bbox_bounds_and_order(self):
        x=candidate(); x['decision']['tool']='crop'; x['decision']['arguments']={'asset_id':'image_1','bbox':[0,0,2,1]}
        with self.assertRaises(s.Invalid): s.validate_candidate(x,self.job,self.c)
        x['decision']['arguments']['bbox']=[0.5,0.5,0.1,0.1]
        with self.assertRaises(s.Invalid): s.validate_candidate(x,self.job,self.c)
    def test_actual_image_sent_and_missing_image_blocks(self):
        data=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6kS8AAAAASUVORK5CYII=')
        (self.p/'image.png').write_bytes(data)
        j=copy.deepcopy(self.job); j['base_row']['images']=['image.png']
        p=s.payload_for(j,self.cfg,self.c,'fake/vision','teacher')
        self.assertTrue(p['messages'][1]['content'][-1]['image_url']['url'].startswith('data:image/png;base64,'))
        (self.p/'image.png').unlink()
        with self.assertRaises(s.Invalid): s.payload_for(j,self.cfg,self.c,'fake/vision','teacher')
    def test_no_silent_context_truncation(self):
        cfg={**self.cfg,'max_context_characters':1}
        with self.assertRaises(s.Invalid): s.payload_for(self.job,cfg,self.c,'fake/vision','teacher')
    def test_reviewer_overclaim_rejected(self):
        r=review(); r['validated_scope']='executed_continuation'
        with self.assertRaises(s.Invalid): s.review_accepted(r,candidate())
        r=review(); r['future_information_used']=True
        self.assertFalse(s.review_accepted(r,candidate()))
    def test_prepare_only_train_unique_tasks(self):
        root=self.p/'cleaned'; (root/'removed_targets').mkdir(parents=True)
        r1=configured_row(); r2=configured_row(); r2['trajectory_id']='T2'; r2['id']='T2:2'
        (root/'removed_targets'/'train.jsonl').write_text(json.dumps(r1)+'\n')
        (root/'train.jsonl').write_text(json.dumps(r2)+'\n')
        Path(self.cfg['jobs_file']).unlink()
        with redirect_stdout(io.StringIO()): info=s.prepare(self.cfg,2)
        self.assertEqual(info['jobs'],2); self.assertEqual(info['unique_trajectories'],2)
        self.assertFalse(info['original_target_in_teacher_input'])
    def test_end_to_end_mock_review_export_resume(self):
        fake=FakeClient()
        with redirect_stdout(io.StringIO()): run_dir=s.run(self.cfg,self.args,fake)
        self.assertEqual(fake.calls,2)
        values=[json.loads(x) for x in (run_dir/'teacher_candidates.jsonl').read_text().splitlines()]
        self.assertEqual(len(values),1); self.assertEqual(len(self.c.history(values[0])),1)
        self.assertEqual(json.loads(values[0]['teacher_decision_json'])['arguments'],{'query':'B refined'})
        self.assertNotIn('TARGET_SHOULD_NOT_LEAK',json.dumps(values))
        again=FakeClient()
        with redirect_stdout(io.StringIO()): s.run(self.cfg,self.args,again)
        self.assertEqual(again.calls,0)
    def test_resume_reviewer_without_repeating_teacher(self):
        first=FakeClient(fail_review=True)
        with redirect_stdout(io.StringIO()): run_dir=s.run(self.cfg,self.args,first)
        db=sqlite3.connect(run_dir/'results.sqlite'); self.assertEqual(db.execute('SELECT status FROM jobs').fetchone()[0],'generated'); db.close()
        second=FakeClient()
        with redirect_stdout(io.StringIO()): s.run(self.cfg,self.args,second)
        self.assertEqual(second.calls,1)
    def test_changed_prompt_invalidates_export(self):
        with redirect_stdout(io.StringIO()): run_dir=s.run(self.cfg,self.args,FakeClient())
        (self.p/'prompts'/'teacher.txt').write_text('CHANGED')
        with self.assertRaises(s.Invalid): s.export_candidates(self.cfg,run_dir)
    def test_catalog_checks_vision_and_structured_output(self):
        with self.assertRaises(s.Invalid): s.verify_models({'fake/vision':{'architecture':{'input_modalities':['text']},'supported_parameters':['response_format']}},['fake/vision'])
    def test_bad_json_and_extra_fields_rejected(self):
        with self.assertRaises(s.Invalid): s.parse_reply('prefix {} suffix')
        x=candidate(); x['invented']='field'
        with self.assertRaises(s.Invalid): s.validate_candidate(x,self.job,self.c)
    def test_schema_optional_fields_are_nullable(self):
        schema=s.strictify({'type':'object','properties':{'x':{'type':'string'}},'required':[]})
        s.check_schema({'x':None},schema)

if __name__=='__main__': unittest.main(verbosity=2)
