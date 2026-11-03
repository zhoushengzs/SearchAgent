import argparse
import base64
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import simulate_backtracking as s

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/Users/ss/Downloads/OpenSearch-VL-SFT-Agent-NoImages')
PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6kS8AAAAASUVORK5CYII=')

def make_fixture(p):
    cfg={'existing_teacher_script':str(ROOT/'sft_processing/teacher/teacher_stage2.py'),
        'cleaner_script':str(SOURCE/'.codex/sft_cleaning_20261025/clean_sft.py'),'media_dir':str(p),'max_simulated_actions':8,
        'max_context_characters':200000,'max_output_tokens':4000,'runs_dir':str(p/'runs'),'jobs_file':str(p/'jobs.jsonl'),'prompts_dir':str(p/'prompts')}
    (p/'prompts').mkdir(); (p/'prompts'/'teacher.txt').write_text('TEACHER'); (p/'prompts'/'reviewer.txt').write_text('REVIEWER')
    for name in ('initial.png','detail.png'): (p/name).write_bytes(PNG)
    api=s.load_api(cfg)
    images=[{'image_index':1,'asset_id':'image_1','observed_at_step':0,'sha256':'initial'},
            {'image_index':2,'asset_id':'image_1','observed_at_step':3,'sha256':'detail'}]
    st={'question':'Which object is shown?','initial_views':[{'asset_id':'image_1','sha256':'initial'}],
        'image_inventory':[{'asset_id':'image_1','origin':'initial_input'}],'attached_images':images,
        'tool_configuration':{'available_tools':['web_search','inspect_image','finish']},'remaining_decisions':27,'executed_prefix_step_ids':[1,2,3]}
    functions=[]
    for name,params in [('vs_web_search',{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}),
                        ('vs_inspect_image',{'type':'object','properties':{'asset_id':{'type':'string'}},'required':['asset_id']})]:
        functions.append({'type':'function','function':{'name':name,'description':'tool','parameters':params}})
    calls=[{'name':'vs_web_search','arguments':{'query':'candidate A'}}, {'name':'vs_web_search','arguments':{'query':'compare candidate B'}}, {'name':'vs_inspect_image','arguments':{'asset_id':'image_1'}}]
    bank=[]; conv=[{'from':'human','value':'img_1: <image>\nimg_2: <image>\nRecorded task state follows.\n'+json.dumps(st)}]
    for i,call in enumerate(calls,1):
        res={'tool':'web_search' if i<3 else 'inspect_image','status':'ok','data':{'clue':'A uncertain' if i==1 else 'B clue','asset_id':'image_1'},'observed_at':'original time '+str(i)}
        bank.append({'bank_step':i,'call':call,'result':res,'original_decision_note':{}})
        conv.extend([{'from':'gpt','value':'<think>{"rationale":"old hypothesis","depends_on":[]}</think>\n<tool_call>'+json.dumps(call)+'</tool_call>'},
                     {'from':'observation','value':json.dumps({'step':i,'result':res,'observation':{'reasoning':'FUTURE_OBSERVER_SECRET'}})}])
    answer={'answers':['GOLD_SECRET_B'],'reasoning':'known final answer','evidence_steps':[3]}
    conv.append({'from':'gpt','value':'<think></think>\n<response>GOLD_SECRET_B</response>'})
    row={'id':'task:4','trajectory_id':'task','step':4,'source':'fixture','split':'train','images':['initial.png','detail.png'],'conversations':conv,
        'system':'Use observed evidence','tools':json.dumps(functions),'teacher_decision_json':json.dumps({'tool':'finish','arguments':answer}),
        'reference_answer':'GOLD_SECRET_B'}
    job={'source_row':row,'bank':bank,'known_answer':answer,'answer_source':{'sample_type':'reviewed_repaired_teacher_decision','independently_proven_ground_truth':False},
         'provenance':{'file':'source/train.jsonl','line':1,'trajectory_id':'task','original_id':'task:4'}}
    job['job_id']=api.digest(job)
    job['image_hashes']=[{'path':r['path'],'sha256':r['sha256']} for r in api.preflight_images({'base_row':row},cfg,False)]
    Path(cfg['jobs_file']).write_text(json.dumps(job)+'\n')
    steps=[]
    for i in (1,2,3):
        steps.append({'bank_step':i,'rationale':'A is a tentative possibility.' if i==1 else ('The later comparison raises doubt; recheck the input image.' if i==3 else 'Check another candidate.'),
            'information_need':'identity','gap_type':'visual' if i==3 else 'language','depends_on':[] if i==1 else [i-1],
            'behavior':'revisit' if i==3 else 'explore','trigger_steps':[2] if i==3 else [],'revisit_steps':[0] if i==3 else [],
            'hypothesis_before':'candidate A' if i==3 else None,'hypothesis_after':'A is not yet established' if i==3 else None,
            'error_context':i==1,'supervise':i!=1})
    plan={'status':'ready','steps':steps,'final':{'answers':['GOLD_SECRET_B'],'reasoning':'Image and comparison support B.','evidence_steps':[0,2,3]}}
    return cfg,job,plan

def good_review():
    return {'verdict':'accept',**{k:True for k in ('prefix_visibility_valid','answer_supported','backtrack_has_real_trigger','no_observation_fabrication','no_unnecessary_backtrack','wrong_steps_not_supervised')}}

class Fake:
    def __init__(self,plan): self.plan=plan; self.calls=0
    def chat(self,payload):
        self.calls+=1
        assert 'known_answer_for_teacher_only' in payload['messages'][1]['content'][0]['text']
        return json.dumps(good_review() if payload['messages'][0]['content']=='REVIEWER' else self.plan),{'id':'mock','usage':{}}

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.p=Path(self.tmp.name); self.cfg,self.job,self.plan=make_fixture(self.p)
    def tearDown(self): self.tmp.cleanup()
    def test_teacher_knows_answer_and_full_process(self):
        ctx=s.teacher_context(self.job,self.cfg)
        self.assertIn('GOLD_SECRET_B',json.dumps(ctx)); self.assertEqual(len(ctx['complete_recorded_process']),3)
    def test_student_prefix_has_no_gold_or_future_images(self):
        rows,_=s.assemble(self.job,self.plan,self.cfg)
        self.assertEqual(len(rows),3)
        self.assertNotIn('GOLD_SECRET_B',json.dumps(rows[0]))
        self.assertEqual(rows[0]['images'],['initial.png'])
        self.assertEqual(rows[1]['images'],['initial.png'])
        self.assertEqual(rows[-1]['images'],['initial.png','detail.png'])
    def test_wrong_target_not_exported_but_in_history(self):
        rows,_=s.assemble(self.job,self.plan,self.cfg)
        self.assertEqual([r['step'] for r in rows],[2,3,4]); self.assertIn('tentative',rows[0]['conversations'][1]['value'])
    def test_results_immutable_and_old_observer_not_reused(self):
        rows,_=s.assemble(self.job,self.plan,self.cfg)
        obs=[json.loads(m['value']) for m in rows[-1]['conversations'] if m['from']=='observation']
        self.assertEqual([x['result'] for x in obs],[x['result'] for x in self.job['bank']]); self.assertNotIn('FUTURE_OBSERVER_SECRET',json.dumps(rows))
    def test_no_fabricated_result_allowed(self):
        self.plan['steps'][2]['result']={'made_up':True}
        with self.assertRaises(s.Invalid): s.assemble(self.job,self.plan,self.cfg)
    def test_wrong_supervision_rejected(self):
        self.plan['steps'][0]['supervise']=True
        with self.assertRaises(s.Invalid): s.assemble(self.job,self.plan,self.cfg)
    def test_unknown_bank_event_rejected(self):
        self.plan['steps'][1]['bank_step']=999
        with self.assertRaises(s.Invalid): s.assemble(self.job,self.plan,self.cfg)
    def test_future_reference_rejected(self):
        self.plan['steps'][1]['depends_on']=[3]
        with self.assertRaises(s.Invalid): s.assemble(self.job,self.plan,self.cfg)
    def test_consecutive_same_call_rejected(self):
        self.plan['steps'][1]['bank_step']=1
        with self.assertRaises(s.Invalid): s.assemble(self.job,self.plan,self.cfg)
    def test_first_use_order_rejected(self):
        self.plan['steps'][0]['bank_step']=2; self.plan['steps'][1]['bank_step']=1
        with self.assertRaises(s.Invalid): s.assemble(self.job,self.plan,self.cfg)
    def test_return_to_old_resource_does_not_need_old_action_repeat(self):
        rows,info=s.assemble(self.job,self.plan,self.cfg)
        self.assertEqual(info['backtracking_steps'],1); self.assertEqual(info['new_live_tool_calls'],0)
    def test_review_needs_every_criterion(self):
        self.assertTrue(s.accepted(good_review())); r=good_review(); r['prefix_visibility_valid']=False; self.assertFalse(s.accepted(r))
    def test_end_to_end_and_resume_no_duplicate_calls(self):
        a=argparse.Namespace(model='fake/vision',reviewer_model=None,dry_run=False,limit=1,max_api_calls=4)
        fake=Fake(self.plan)
        with redirect_stdout(io.StringIO()): out=s.run(self.cfg,a,fake)
        self.assertEqual(fake.calls,2); self.assertEqual(len((out/'student_sft_candidates.jsonl').read_text().splitlines()),3)
        again=Fake(self.plan)
        with redirect_stdout(io.StringIO()): s.run(self.cfg,a,again)
        self.assertEqual(again.calls,0)
    def test_needs_execution_not_published(self):
        a=argparse.Namespace(model='fake/vision',reviewer_model=None,dry_run=False,limit=1,max_api_calls=4)
        fake=Fake({'status':'needs_execution','requested_actions':[{'tool':'crop','arguments':{}}]})
        with redirect_stdout(io.StringIO()): out=s.run(self.cfg,a,fake)
        self.assertEqual(fake.calls,1); self.assertEqual((out/'student_sft_candidates.jsonl').read_text(),'')
    def test_doctor_offline(self):
        with redirect_stdout(io.StringIO()): r=s.doctor(self.cfg)
        self.assertEqual(r['model_calls'],0); self.assertTrue(r['teacher_receives_complete_process_and_answer'])

if __name__=='__main__': unittest.main(verbosity=2)
