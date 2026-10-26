import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import renumber_sft as r

def sample(ids=(1,3,4),target=6,refs=(3,4),reason='Use the observations.',tid='T'):
    state={'question':'Follow manual step 3.','executed_prefix_step_ids':list(ids),'remaining_decisions':24,
        'initial_views':[],'image_inventory':[{'asset_id':'image_1'}],
        'attached_images':[{'image_index':1,'asset_id':'image_1','sha256':'same','observed_at_step':ids[-1] if ids else 0}],
        'tool_configuration':{'version':'original'}}
    messages=[{'from':'human','value':'img_1: <image>\nRecorded task state follows.\n'+json.dumps(state)}]
    for i,old in enumerate(ids):
        think={'rationale':'Review evidence.','depends_on':[] if i==0 else [ids[i-1]]}
        call={'name':'vs_web_search','arguments':{'query':'manual step '+str(old)}}
        obs={'step':old,'result':{'status':'ok','data':{'step':old,'text':'manual step '+str(old)}},
            'observation':{'reasoning':'Recorded facts.','revision_of_step':None if i==0 else ids[i-1]}}
        messages.extend([{'from':'gpt','value':'<think>'+json.dumps(think)+'</think>\n<tool_call>'+json.dumps(call)+'</tool_call>'},
                         {'from':'observation','value':json.dumps(obs)}])
    think={'rationale':reason,'depends_on':list(refs),'evidence_steps':list(refs)}
    decision={'tool':'finish','arguments':{'answers':['Answer.'],'reasoning':reason,'evidence_steps':list(refs)},**think}
    messages.append({'from':'gpt','value':'<think>'+json.dumps(think)+'</think>\n<response>Answer.</response>'})
    return {'id':tid+':teacher-step:'+str(target),'trajectory_id':tid,'step':target,'conversations':messages,'images':['original_image.png'],
            'teacher_decision_json':json.dumps(decision),'sample_type':'original','system':'system','tools':'[]'}

class Tests(unittest.TestCase):
    def test_direct_not_transitive_lookup(self):
        x,d=r.transform(sample())
        self.assertEqual([a['step'] for a in r.history(x)],[1,2,3])
        dec=json.loads(x['teacher_decision_json']); self.assertEqual(dec['depends_on'],[2,3]); self.assertEqual(dec['arguments']['evidence_steps'],[2,3])
        self.assertEqual(x['step'],4)
    def test_source_alias_composition(self):
        src=sample(ids=(1,3),target=4,refs=(1,3))
        x,d=r.transform(src,{2:1})
        self.assertEqual(d['source_history_to_new'],{0:0,1:1,3:2,2:1})
        self.assertEqual(json.loads(x['teacher_decision_json'])['arguments']['evidence_steps'],[1,2])
    def test_zero_initial_evidence(self):
        x,d=r.transform(sample(refs=(0,3)))
        self.assertEqual(json.loads(x['teacher_decision_json'])['depends_on'],[0,2])
    def test_image_observation_mapped_without_file_rename(self):
        src=sample(); x,d=r.transform(src)
        self.assertEqual(x['images'],src['images'])
        self.assertEqual(r.state_of(x)[1]['attached_images'][0]['observed_at_step'],3)
    def test_budget_and_user_question_preserved(self):
        src=sample(); x,d=r.transform(src)
        self.assertEqual(r.state_of(x)[1]['remaining_decisions'],24)
        self.assertEqual(r.state_of(x)[1]['question'],r.state_of(src)[1]['question'])
    def test_tool_args_and_results_not_rewritten(self):
        src=sample(); x,d=r.transform(src)
        for a,b in zip(r.history(src),r.history(x)):
            self.assertEqual(a['call'],b['call']); self.assertEqual(a['observation']['result'],b['observation']['result'])
    def test_observer_revision(self):
        x,d=r.transform(sample())
        self.assertEqual(r.history(x)[-1]['observation']['observation']['revision_of_step'],2)
    def test_clear_prose_single(self):
        x,d=r.transform(sample(reason='Visual evidence from Step 3 requires checking.'))
        self.assertIn('Step 2',x['conversations'][-1]['value'])
    def test_prose_replaces_numbers_simultaneously(self):
        x,d=r.transform(sample(reason='Previous findings from Steps3 and4 support the answer.'))
        self.assertIn('Steps 2, 3',x['conversations'][-1]['value'])
    def test_ranges_compose_alias_and_deduplicate(self):
        edits=[]
        self.assertEqual(r.rewrite_text('Previous steps (Steps 1-4) are known.',{1:1,2:1,3:2,4:3},'reasoning',edits),'Previous steps (Steps 1, 2, 3) are known.')
    def test_domain_reference_unchanged(self):
        x,d=r.transform(sample(reason='Follow manual step 3 carefully.'))
        self.assertIn('manual step 3',x['conversations'][-1]['value'])
        self.assertIn('domain_step_unchanged',[e['kind'] for e in d['text_edits']])
    def test_ambiguous_prose_is_queued_not_guessed(self):
        with self.assertRaises(r.Review): r.transform(sample(reason='Follow Step 3 carefully.'))
    def test_target_self_reference_in_prose(self):
        x,d=r.transform(sample(reason='Agent step 6 completes the task.'))
        self.assertIn('Agent step 4',x['conversations'][-1]['value'])
    def test_no_sample_id_collision_after_branch_compression(self):
        a,_=r.transform(sample(ids=(1,3),target=4,refs=(3,)))
        b,_=r.transform(sample(ids=(1,3),target=5,refs=(3,)))
        self.assertEqual(a['step'],b['step']); self.assertNotEqual(a['id'],b['id'])
    def test_already_contiguous_exactly_unchanged(self):
        src=sample(ids=(1,2,3),target=4,refs=(3,))
        out,d=r.transform(src); self.assertIs(out,src); self.assertFalse(d['changed'])
    def test_final_answers_remain(self):
        src=sample(); out,d=r.transform(src)
        self.assertEqual(r.RESPONSE.findall(out['conversations'][-1]['value']),['Answer.'])
        self.assertEqual(json.loads(out['teacher_decision_json'])['arguments']['answers'],['Answer.'])
    def test_finish_answer_text_reference_stays_in_sync(self):
        src=sample(); text='Agent step 3 provides this answer.'
        src['conversations'][-1]['value']=src['conversations'][-1]['value'].replace('<response>Answer.</response>','<response>'+text+'</response>')
        dec=json.loads(src['teacher_decision_json']); dec['arguments']['answers']=[text]; src['teacher_decision_json']=json.dumps(dec)
        out,_=r.transform(src)
        self.assertEqual(r.RESPONSE.findall(out['conversations'][-1]['value']),['Agent step 2 provides this answer.'])
        self.assertEqual(json.loads(out['teacher_decision_json'])['arguments']['answers'],['Agent step 2 provides this answer.'])
    def test_future_structured_reference_rejected(self):
        with self.assertRaises(r.Review): r.transform(sample(refs=(99,)))
    def test_existing_initial_answer_auxiliary(self):
        src={'id':'aux','conversations':[{'from':'human','value':'Question.'},{'from':'gpt','value':'<think></think>\n<response>A</response>'}],'images':[]}
        out,d=r.transform(src); self.assertIs(src,out); self.assertTrue(d['auxiliary'])
    def test_alias_cycle_rejected(self):
        with self.assertRaises(r.Review): r.transform(sample(),{2:5,5:2})
    def test_actual_preflight_phrases(self):
        phrases=['as hypothesized in Step 3','the image from Step 3 is visible',"Step 3's visual description",'as identified in Step 3','failed due to timeout in step 3','date from Step 3','the collage described in Step 3','Steps3 and4 explicitly describe these subjects','Step 3 visually identified the people','Step 3 attempted to verify the identity','Step 3 initiated a search',"Step 3's hypothesis"]
        for phrase in phrases:
            e=[]; out=r.rewrite_text(phrase,{3:2,4:3},'observer',e); self.assertNotIn('Step 3',out); self.assertNotIn('step 3',out)

if __name__=='__main__': unittest.main(verbosity=2)
