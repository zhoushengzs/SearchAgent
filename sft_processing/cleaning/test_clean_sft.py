import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import clean_sft as c

def call(q):
    return {'name':'vs_web_search','arguments':{'query':q}}

def result(q='A',status='ok',error='ConnectTimeout'):
    if status=='error': return {'tool':'web_search','status':'error','error_type':error,'observed_at':'timestamp'}
    return {'tool':'web_search','status':'ok','data':{'query':q,'results':[{'url':'https://example.test/'+q,'snippet':'fact'}]},'observed_at':'timestamp'}

def row(keys, target='FINISH', results=None, refs=None, rationale='next action', attachments=None):
    steps=list(range(1,len(keys)+1)); refs=refs if refs is not None else ([steps[-1]] if steps else [0])
    attached=attachments or []
    prefix=''.join('img_%d: <image>\n'%i for i in range(1,len(attached)+1))+c.STATE_MARKER+'\n'
    state={'question':'manual step 202','initial_views':[],'image_inventory':[], 'attached_images':attached,
           'executed_prefix_step_ids':steps,'remaining_decisions':30-len(keys),'tool_configuration':{}}
    messages=[{'from':'human','value':prefix+json.dumps(state)}]
    for i,q in enumerate(keys,1):
        think={'rationale':'inspect available evidence','information_need':'facts','gap_type':'language','depends_on':[] if i==1 else [i-1]}
        messages.append({'from':'gpt','value':'<think>'+json.dumps(think)+'</think>\n<tool_call>'+json.dumps(call(q))+'</tool_call>'})
        observation={'step':i,'result':copy.deepcopy(results[i-1]) if results else result(q),
                     'observation':{'visual_facts':[],'language_facts':['fact'],'reasoning':'supported','hypothesis':'candidate','revision_of_step':i-1 if i>1 else None,'open_questions':[]}}
        messages.append({'from':'observation','value':json.dumps(observation)})
    think={'rationale':rationale,'information_need':'none' if target=='FINISH' else 'facts','gap_type':'none' if target=='FINISH' else 'language','depends_on':refs}
    if target=='FINISH':
        think['evidence_steps']=refs
        decision={'tool':'finish','arguments':{'answers':['answer'],'reasoning':rationale,'evidence_steps':refs},**think}
        tail='<response>answer</response>'
    else:
        decision={'tool':'web_search','arguments':{'query':target},**think}
        tail='<tool_call>'+json.dumps(call(target))+'</tool_call>'
    messages.append({'from':'gpt','value':'<think>'+json.dumps(think)+'</think>\n'+tail})
    return {'id':'T:teacher-step:'+str(len(keys)+1),'trajectory_id':'T','step':len(keys)+1,
            'conversations':messages,'images':['image%d.png'%i for i in range(len(attached))],
            'teacher_decision_json':json.dumps(decision),'sample_type':'original','system':'test','tools':'[]'}

class CleanerTests(unittest.TestCase):
    def test_key_order_and_strict_string_equality(self):
        a={'name':'x','arguments':{'b':1,'a':2}}; b={'arguments':{'a':2,'b':1},'name':'x'}
        self.assertEqual(c.action_key(a),c.action_key(b))
        self.assertNotEqual(c.action_key(call('A')),c.action_key(call('A ')))
        self.assertNotEqual(c.action_key(call('A')),c.action_key(call('a')))
    def test_aa_merge_and_reference_mapping(self):
        r=row(['A','A','A','B'],refs=[1,2,3,4]); p=c.make_plan(r)
        self.assertEqual(p['aliases'],{2:1,3:1})
        cleaned,_=c.rebuild_row(r,p)
        self.assertEqual([a['step'] for a in c.history(cleaned)],[1,4])
        d=json.loads(cleaned['teacher_decision_json'])
        self.assertEqual(d['depends_on'],[1,4]); self.assertEqual(d['arguments']['evidence_steps'],[1,4])
        self.assertEqual(c.parse_state(cleaned)[1]['remaining_decisions'],26)
        self.assertEqual(cleaned['step'],5)
    def test_nonconsecutive_revisit_survives(self):
        r=row(['A','B','A']); p=c.make_plan(r)
        self.assertFalse(p['aliases']); self.assertFalse(p['uncertain'])
        self.assertEqual(len(c.history(c.rebuild_row(r,p)[0])),3)
    def test_query_refinement_survives(self):
        r=row(['A','A year 2020','A year 2020 official']); self.assertFalse(c.make_plan(r)['aliases'])
    def test_same_rationale_different_action_survives(self):
        r=row(['A'],target='B',rationale='inspect available evidence')
        cleaned,_=c.rebuild_row(r,c.make_plan(r)); self.assertEqual(c.parse_calls(cleaned['conversations'][-1]['value'])[0]['arguments']['query'],'B')
    def test_transient_error_retry_allowed(self):
        r=row(['A'],target='A',results=[result(status='error')])
        self.assertTrue(c.classify_retry(c.history(r),call('A'))[0])
    def test_deterministic_error_retry_rejected(self):
        r=row(['A'],target='A',results=[result(status='error',error='ValueError')])
        self.assertFalse(c.classify_retry(c.history(r),call('A'))[0])
    def test_http_status_precedes_generic_error(self):
        r=result(status='error',error='ConnectionError'); r['http_status']=403
        self.assertFalse(c.temporary_failure(r))
        r['http_status']=429; self.assertTrue(c.temporary_failure(r))
    def test_unknown_failure_requires_review(self):
        r=row(['A'],target='A',results=[result(status='error',error='UnexpectedFault')])
        self.assertEqual(c.classify_retry(c.history(r),call('A'))[1],'unclassified_failure_requires_review')
    def test_retry_budget(self):
        r=row(['A','A'],target='A',results=[result(status='error'),result(status='error')])
        self.assertEqual(c.classify_retry(c.history(r),call('A'))[1],'transient_retry_budget_exceeded')
    def test_success_recovery_and_following_duplicate(self):
        r=row(['A','A','A'],results=[result(status='error'),result(),result()])
        p=c.make_plan(r)
        self.assertIn(2,p['retained_retries']); self.assertEqual(p['aliases'],{3:2})
        self.assertEqual([a['step'] for a in c.history(c.rebuild_row(r,p)[0])],[1,2])
    def test_unique_result_goes_to_review(self):
        different=result(); different['data']['results'].append({'url':'https://other.test/new','snippet':'new fact'})
        r=row(['A','A'],results=[result(),different]); p=c.make_plan(r)
        self.assertEqual(p['uncertain'],{2:'different_tool_result'})
        with self.assertRaises(c.NeedsReview): c.rebuild_row(r,p)
    def test_new_observer_content_not_silently_deleted(self):
        r=row(['A','A']); o=json.loads(r['conversations'][4]['value']); o['observation']['language_facts'].append('new fact')
        r['conversations'][4]['value']=json.dumps(o)
        self.assertEqual(c.make_plan(r)['uncertain'][2],'observer_equivalence_unconfirmed')
    def test_timing_not_unique_evidence(self):
        a=result(); b=result(); b['observed_at']='later'; b['elapsed_ms']=500
        r=row(['A','A'],results=[a,b]); self.assertEqual(c.make_plan(r)['aliases'],{2:1})
    def test_new_image_attachment_requires_review(self):
        images=[{'image_index':1,'asset_id':'image_1','observed_at_step':0,'sha256':'initial'},
                {'image_index':2,'asset_id':'new_asset','observed_at_step':2,'sha256':'newpixels'}]
        r=row(['A','A'],attachments=images)
        self.assertEqual(c.make_plan(r)['uncertain'][2],'unique_or_unconfirmed_image_attachment')
    def test_equivalent_images_preserved_and_observation_step_mapped(self):
        images=[{'image_index':1,'asset_id':'same','observed_at_step':1,'sha256':'pixels'},
                {'image_index':2,'asset_id':'same','observed_at_step':2,'sha256':'pixels'}]
        r=row(['A','A'],attachments=images); cleaned,_=c.rebuild_row(r,c.make_plan(r))
        self.assertEqual(cleaned['images'],r['images'])
        self.assertEqual([x['observed_at_step'] for x in c.parse_state(cleaned)[1]['attached_images']],[1,1])
    def test_raw_result_unchanged(self):
        r=row(['A','A','B']); cleaned,_=c.rebuild_row(r,c.make_plan(r))
        self.assertEqual(c.history(cleaned)[-1]['result'],c.history(r)[-1]['result'])
    def test_explicit_agent_reference_maps(self):
        r=row(['A','A'],rationale='Use evidence from tool step 2.')
        cleaned,_=c.rebuild_row(r,c.make_plan(r))
        self.assertIn('tool step 1',cleaned['conversations'][-1]['value'])
    def test_ambiguous_manual_step_not_rewritten(self):
        r=row(['A','A'],rationale='Check manual step 2 again.')
        with self.assertRaises(c.NeedsReview): c.rebuild_row(r,c.make_plan(r))
        self.assertIn('manual step 2',r['conversations'][-1]['value'])
    def test_group_reference_detected(self):
        with self.assertRaises(c.NeedsReview): c.rewrite_prose('Steps 1 and 2 support this.',{2:1})
    def test_dangling_reference_rejected(self):
        r=row(['A'],refs=[99])
        with self.assertRaises(c.NeedsReview): c.rebuild_row(r,c.make_plan(r))
    def test_final_answer_never_is_tool_repeat(self):
        r=row(['A','A']); self.assertEqual(c.parse_calls(r['conversations'][-1]['value']),[])
    def test_full_pipeline_partitions_hashes_and_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); source=p/'source'; source.mkdir(); (source/'configs').mkdir()
            (source/'dataset_info.json').write_text('{}'); (source/'configs'/'export_notes.json').write_text('{}')
            first=row([],target='A'); duplicate=row(['A'],target='A'); final=row(['A','A'])
            testval=row([],target='FINISH'); testval['id']='V:1'; testval['trajectory_id']='V'
            for split,values in [('train',[first,duplicate,final]),('val',[testval])]:
                (source/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in values))
            before=c.file_sha(source/'train.jsonl')
            with redirect_stdout(io.StringIO()):
                summary=c.main(['--input',str(source),'--output',str(p/'out'),'--audit-dir',str(p/'audit')])
            self.assertEqual(summary['counts']['train']['clean_rows'],2)
            self.assertEqual(summary['counts']['train']['removed_rows'],1)
            self.assertEqual(summary['counts']['train']['input_final_answer_rows'],1)
            self.assertEqual(c.file_sha(source/'train.jsonl'),before)
            outputrows=[json.loads(s) for s in (p/'out'/'train.jsonl').read_text().splitlines()]
            self.assertEqual(len(c.history(outputrows[-1])),1)
            self.assertEqual(json.loads(outputrows[-1]['teacher_decision_json'])['arguments']['evidence_steps'],[1])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                c.main(['--input',str(source),'--output',str(p/'out'),'--audit-dir',str(p/'audit2')])

if __name__=='__main__': unittest.main(verbosity=2)
