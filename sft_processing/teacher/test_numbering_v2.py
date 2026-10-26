import copy, json, unittest
from test_teacher_stage2 import Stage2Tests as _Fixtures, configured_row, candidate, row
import teacher_stage2 as s

class NumberingTests(unittest.TestCase):
    setUp= _Fixtures.setUp
    tearDown= _Fixtures.tearDown
    def test_recovery_renumbers_and_strips_target(self):
        cfg={**self.cfg,'numbering_version':'v2_contiguous'}
        r=row(['A','A','B'],target='TARGET_SHOULD_NOT_LEAK',refs=[2,3])
        j=s.job_for(r,self.p/'train.jsonl',1,'recovery',self.c,cfg)
        b=j['base_row']; s.validate_numbered_base(b)
        self.assertEqual(b['step'],3)
        self.assertEqual([a['step'] for a in s.history_from_base(b,self.c)],[1,2])
        st=self.c.parse_state(b)[1]
        self.assertEqual(st['executed_prefix_step_ids'],[1,2])
        self.assertEqual(st['remaining_decisions'],27)
        self.assertNotIn('TARGET_SHOULD_NOT_LEAK',json.dumps(b))
        self.assertNotIn('teacher_decision_json',b)
    def test_old_jobs_refused(self):
        with self.assertRaisesRegex(s.Invalid,'old_numbered_jobs_forbidden'):
            s.load_jobs({**self.cfg,'numbering_version':'v2_contiguous'})
    def test_gapped_control_refused(self):
        r=row(['A','B']); r['step']=9
        with self.assertRaises(s.numbering.Review):
            s.job_for(r,self.p/'train.jsonl',1,'control',self.c,{**self.cfg,'numbering_version':'v2_contiguous'})
    def test_control_unchanged_history(self):
        r=configured_row()
        j=s.job_for(r,self.p/'train.jsonl',1,'control',self.c,{**self.cfg,'numbering_version':'v2_contiguous'})
        self.assertEqual(j['base_row']['conversations'],r['conversations'][:-1])
    def test_invalid_revisit_reference_refused(self):
        x=candidate(); x['annotation']['revisit_steps']=[99]
        with self.assertRaisesRegex(s.Invalid,'invalid_revisit_step'):
            s.validate_candidate(x,self.job,self.c)

if __name__=='__main__': unittest.main()
