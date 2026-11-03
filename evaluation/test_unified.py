"""No-network checks for changing only weights in the three agent conditions."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from evaluation.grpo_native import load_runtime
from evaluation.unified_cache import memo,evidence_snapshots
from evaluation.unified_common import SharedLocalModel,weight_identity
from evaluation.unified_score import validate_verdict
from evaluation.unified_run import direct_task


class CacheTests(unittest.TestCase):
    def test_identical_request_reuses_real_result_and_normalizes_tuple_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            loader=Mock(return_value={'text':'real public evidence'})
            a,hit,key=memo(d,'page',{'url':'https://example.org','timeout':(10,30)},loader)
            b,hit2,key2=memo(d,'page',{'url':'https://example.org','timeout':(10,30)},loader)
            self.assertEqual(a,b);self.assertFalse(hit);self.assertTrue(hit2);self.assertEqual(key,key2)
            loader.assert_called_once()

    def test_errors_are_not_cached_as_success_and_corruption_is_not_silently_refetched(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):memo(d,'search',{'q':'x'},Mock(side_effect=RuntimeError('offline')))
            value,hit,key=memo(d,'search',{'q':'x'},lambda:{'answer':'real'})
            self.assertFalse(hit)
            p=Path(d)/'search'/(key+'.json');r=json.loads(p.read_text());r['value']['answer']='changed';p.write_text(json.dumps(r))
            never=Mock()
            with self.assertRaises(ValueError):memo(d,'search',{'q':'x'},never)
            never.assert_not_called()

    def test_judge_requires_complete_boolean_verdict(self):
        for body in [{'choices':[{'finish_reason':'length','message':{'content':'{}'}}]},
                     {'choices':[{'finish_reason':'stop','message':{'content':'{"correct":"true","reason":"x"}'}}]}]:
            with self.assertRaises(ValueError):validate_verdict(body)
        r=validate_verdict({'choices':[{'finish_reason':'stop','message':{'content':'{"correct":true,"reason":"x","id":"bad"}'}}]})
        self.assertTrue(r['correct']);self.assertNotIn('id',r)


class WeightAndAssetsTests(unittest.TestCase):
    def test_weight_shapes_are_read_without_loading_tensor_contents(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.safetensors';header=json.dumps({'weight':{'dtype':'BF16','shape':[2,3],'data_offsets':[0,12]}}).encode()
            p.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*12)
            identity,shapes=weight_identity(d)
            self.assertEqual(shapes,{'weight':[2,3]});self.assertIn('model.safetensors',identity)

    def test_all_models_load_processor_and_configuration_from_common_assets(self):
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,set_device=Mock()),
                              backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                                       cudnn=SimpleNamespace(allow_tf32=True)),bfloat16='BF16')
        config=Mock();processor=Mock();generation=Mock();weights=Mock()
        transformers=SimpleNamespace(AutoConfig=config,AutoProcessor=processor,
                                     AutoModelForImageTextToText=weights,GenerationConfig=generation)
        with patch.dict(sys.modules,{'torch':torch,'transformers':transformers}):
            SharedLocalModel('/base','/common');SharedLocalModel('/sft','/common')
        self.assertEqual([c.args[0] for c in processor.from_pretrained.call_args_list],['/common','/common'])
        self.assertEqual([c.args[0] for c in config.from_pretrained.call_args_list],['/common','/common'])
        self.assertEqual([c.args[0] for c in generation.from_pretrained.call_args_list],['/common','/common'])
        self.assertEqual([c.args[0] for c in weights.from_pretrained.call_args_list],['/base','/sft'])


@unittest.skipUnless(os.environ.get('NATIVE_GRPO_RUNTIME'),'Set the actual native runtime for integration checks')
class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.agent,cls.session=load_runtime(os.environ['NATIVE_GRPO_RUNTIME'])

    def test_direct_condition_has_zero_available_tools_and_cannot_execute_search(self):
        class Model:
            calls=[]
            def generate(self,messages,tools,cap,context,**kwargs):
                self.calls.append(messages)
                return {'text':'<response>4</response>','status':'ok','input_tokens':32,'new_tokens':8}
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True):
            model=Model();task={'id':'t','source_dataset':'fixture','input_fingerprint':'fp',
                               'model_input':{'question':'2+2?','images':[]},'answer':'SECRET_REFERENCE'}
            with patch('project_runtime.evidence.EvidenceTools.execute',side_effect=AssertionError('direct must never execute tools')):
                r=direct_task(task,model,self.session,SimpleNamespace(output=Path(d)),'run')
            self.assertEqual(r['answer'],'4');self.assertEqual(len(model.calls),1)
            text=json.dumps(model.calls)
            self.assertNotIn('SECRET_REFERENCE',text);self.assertNotIn('# Tools',text)
            self.assertIn('No tools are available',text)
            self.assertIn('available_tools\\\": []',text)

    def test_shared_search_restores_url_permissions_for_the_second_model(self):
        import importlib
        evidence=importlib.import_module('project_runtime.evidence')
        response=Mock();response.json.return_value={'organic':[{'link':'https://example.org/a','title':'Evidence'}]}
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True),patch.object(evidence,'serper_post',return_value=response) as post:
            a=evidence.EvidenceTools(Path(d)/'a');b=evidence.EvidenceTools(Path(d)/'b');a.serper_key=b.serper_key=True
            with evidence_snapshots(Path(d)/'cache'):
                x=a._search('same query',False);y=b._search('same query',False)
            self.assertEqual(x,y);post.assert_called_once()
            self.assertIn('https://example.org/a',b.urls)


if __name__=='__main__':unittest.main()
