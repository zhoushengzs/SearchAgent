"""Behavior checks for SFT tools, image evidence and the actual agent loop."""
import asyncio
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'integration'))
from project_runtime import evidence
from project_runtime.session import ProjectSession, get_all_tools, parse_action, system_prompt
from merged_agent import DeepResearchAgent


def call(name, args):
    return '<think>Get evidence.</think><tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {'SERPER_API_KEYS': '["test-only"]',
                                          'SERPER_KEY_STATE_FILE': str(Path(self.tmp.name) / 'keys.json')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.session = ProjectSession('Question without an answer', [Image.new('RGB', (1000, 400), 'red')], self.tmp.name, 8)

    def test_only_six_actual_tools_are_advertised(self):
        names = set(get_all_tools())
        self.assertEqual(names, {'vs_web_search','vs_image_search','vs_visit','vs_inspect_image','vs_crop','vs_calculate'})
        self.assertNotIn('"name": "image_search"', system_prompt())
        self.assertNotIn('COS', system_prompt())

    def test_crop_uses_original_not_preview_and_attaches_pixels(self):
        self.assertEqual(self.session.initial_message()['images'][0].size, (512, 205))
        result, message = self.session.execute('vs_crop', {'asset_id':'image_1','bbox':[0.5,0,1,1]}, 1)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['data']['view']['pixel_bbox'], [500,0,1000,400])
        self.assertEqual(message['images'][0].size, (500,400))
        self.assertIn('img_2', message['content'])
        self.assertIn('"remaining_decisions": 7', message['content'])

    def test_text_image_search_sends_query_never_uploads(self):
        response = Mock(status_code=200)
        response.json.return_value = {'images':[{'link':'https://example.org/page','title':'A cat','imageUrl':'https://example.org/cat.png'}]}
        with patch.object(evidence, 'serper_post', return_value=response) as post:
            result, _ = self.session.execute('vs_image_search', {'query':'red cat'}, 1)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(post.call_args.args, ('https://google.serper.dev/images',))
        self.assertEqual(post.call_args.kwargs['json'], {'q':'red cat','num':5})
        self.assertNotIn('files', post.call_args.kwargs)

    def test_read_page_and_acquire_image_from_real_result_shape(self):
        response = Mock(status_code=200)
        response.json.return_value = {'organic':[{'link':'https://example.org/page','title':'Page'}]}
        with patch.object(evidence, 'serper_post', return_value=response):
            self.session.execute('vs_web_search', {'query':'a topic'}, 1)
        with patch.object(evidence, 'fetch_public', return_value=(b'<p>Actual evidence</p><img src="/photo.png">','https://example.org/page','text/html',200)):
            result, _ = self.session.execute('vs_visit', {'url':'https://example.org/page'}, 2)
        self.assertIn('Actual evidence', result['data']['text'])
        import io
        data = io.BytesIO()
        Image.new('RGB',(20,30),'blue').save(data,format='PNG')
        with patch.object(evidence, 'fetch_public', return_value=(data.getvalue(),'https://example.org/photo.png','image/png',200)):
            result, message = self.session.execute('vs_inspect_image', {'image_url':'https://example.org/photo.png'}, 3)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(message['images'][0].getpixel((0,0)), (0,0,255))

    def test_unknown_url_and_bad_crop_return_failure(self):
        for name, args in [('vs_visit',{'url':'https://example.org/not-acquired'}),
                           ('vs_crop',{'asset_id':'image_1','bbox':[0.8,0,0.2,1]})]:
            result, message = self.session.execute(name,args,1)
            self.assertEqual(result['status'],'error')
            self.assertNotIn('images',message)

    def test_calculation_rejects_python_code(self):
        result, _ = self.session.execute('vs_calculate',{'expression':'(12+8)*3'},1)
        self.assertEqual(result['data']['value'],60)
        result, _ = self.session.execute('vs_calculate',{'expression':'__import__("os").system("id")'},2)
        self.assertEqual(result['status'],'error')

    def test_image_mapping_and_sessions_are_isolated(self):
        with tempfile.TemporaryDirectory() as d:
            other = ProjectSession('Other',[Image.new('RGB',(30,40),'blue'),Image.new('RGB',(50,60),'green')],d,3)
            self.assertEqual([a['asset_id'] for a in other.attached], ['image_1','image_2'])
            self.assertEqual(len(other.initial_message()['images']),2)
            self.assertEqual(len(self.session.attached),1)

    def test_action_parser_rejects_missing_and_multiple_actions(self):
        good = call('vs_calculate',{'expression':'1+1'})
        for bad in [good+good,good[:-4],call('image_search',{'url':'img_1'}),'<think>x</think><response></response>']:
            with self.assertRaises((ValueError,TypeError)):
                parse_action(bad)
        self.assertEqual(parse_action('<think>x</think><response>2</response>'), ('answer','2'))

    def test_private_network_addresses_are_rejected(self):
        with patch.object(evidence.socket, 'getaddrinfo', return_value=[(2,1,6,'',('127.0.0.1',443))]):
            with self.assertRaises(ValueError):
                evidence.check_public_url('https://example.org')

    def test_actual_agent_loop_receives_crop_then_answers(self):
        captured = []
        class Engine:
            async def get_model_response(self, messages, **kwargs):
                captured.append(copy.deepcopy(messages))
                text = call('vs_crop',{'asset_id':'image_1','bbox':[0,0,0.5,1]}) if len(captured)==1 else '<think>Read the crop.</think><response>red</response>'
                return SimpleNamespace(text=text,prompt_length=100,completion_length=30)
        agent = DeepResearchAgent(Engine(), tools=get_all_tools())
        agent.max_llm_calls=3
        result = asyncio.run(agent.run('What color?',answer='SECRET_REFERENCE',images=[Image.new('RGB',(1000,400),'red')]))
        self.assertEqual(result['termination'],'answer')
        self.assertEqual(result['prediction'],'red')
        self.assertEqual(captured[1][-1]['images'][0].size,(500,400))
        self.assertNotIn('SECRET_REFERENCE',str(captured))
        self.assertEqual(captured[1][-1]['images'][0].getpixel((0,0)),(255,0,0))

    def test_three_failed_tools_stop_and_mark_all_steps(self):
        class Engine:
            async def get_model_response(self, **kwargs):
                return SimpleNamespace(text=call('vs_inspect_image',{'asset_id':'missing'}))
        agent = DeepResearchAgent(Engine(),tools=get_all_tools())
        result = asyncio.run(agent.run('Question',images=[]))
        self.assertEqual(result['termination'],'consecutive_step_errors')
        self.assertEqual(len([m for m in result['messages'] if m['role']=='assistant' and m['step_error']]),3)
        self.assertEqual(result['prediction'],'')

    def test_workflow_recognizes_structured_tool_failure(self):
        path=ROOT/'integration/merged_workflow.py'
        tree=ast.parse(path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_has_tool_error_observation')
        module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        namespace={'json':json,'Any':object}
        exec(compile(module,str(path),'exec'),namespace)
        check=namespace[fn.name]
        self.assertTrue(check('<tool_response>{"status":"error"}</tool_response>'))
        self.assertFalse(check('<tool_response>{"status":"ok"}</tool_response>'))

    def test_legacy_builder_cannot_overwrite_integrated_code(self):
        path=ROOT/'integration/merged_agent.py'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        subprocess.run([sys.executable,str(ROOT/'scripts/build_integration.py')],check=True,capture_output=True)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)


    def test_legacy_jina_keys_are_removed_from_the_training_environment(self):
        import importlib.util
        path=ROOT/'scripts/training_control.py'
        spec=importlib.util.spec_from_file_location('grpo_control_no_jina_test',path)
        control=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(control)
        config_root=Path(self.tmp.name)/'control'
        (config_root/'integration').mkdir(parents=True)
        (config_root/'integration/local_training.env').write_text(
            'JINA_API_KEY=unused-file-value\nSERPER_API_KEY=test-only\n')
        with patch.object(control,'ROOT',config_root), patch.dict(os.environ,{
                'JINA_API_KEY':'unused-shell-value','JINA_API_URL':'unused-url'}):
            env=control.environment(config_root/'rllm')
        self.assertFalse(any(k.startswith('JINA_') for k in env))
        self.assertEqual(env['SERPER_API_KEY'],'test-only')


    def test_child_process_finds_tools_in_the_selected_python_environment(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('grpo_control_path_test',ROOT/'scripts/training_control.py')
        control=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(control)
        config_root=Path(self.tmp.name)/'path-control'
        (config_root/'integration').mkdir(parents=True)
        with patch.object(control,'ROOT',config_root), patch.dict(os.environ,{'PATH':'/usr/bin:/bin'}):
            env=control.environment(config_root/'rllm')
        self.assertEqual(env['PATH'].split(os.pathsep)[0],str(Path(sys.executable).parent))
        self.assertEqual(Path(__import__('shutil').which('python',path=env['PATH'])).parent,Path(sys.executable).parent)


    def test_webpage_is_bounded_and_can_be_read_with_an_offset(self):
        self.session.tools.urls.add('https://example.org/page')
        text='A'*4000+'B'*4000+'C'*2000
        with patch.object(evidence,'fetch_public',return_value=(text.encode(),'https://example.org/page','text/plain',200)):
            first,_=self.session.execute('vs_visit',{'url':'https://example.org/page'},1)
            second,_=self.session.execute('vs_visit',{'url':'https://example.org/page','offset':4000},2)
        self.assertEqual(first['data']['text'],'A'*4000)
        self.assertEqual(second['data']['text'],'B'*4000)
        self.assertEqual(first['data']['total_chars'],10000)

    def test_last_decision_requests_an_answer_instead_of_more_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            session=ProjectSession('Question',[],directory,1)
            message=session.initial_message()
            self.assertIn('Do not call another tool',message['content'])
            self.assertIn('"remaining_decisions": 1',message['content'])


    def test_duplicate_successful_search_does_not_spend_another_request(self):
        response=Mock(status_code=200)
        response.json.return_value={'organic':[{'link':'https://example.org/page','title':'Evidence'}]}
        with patch.object(evidence,'serper_post',return_value=response) as post:
            first,_=self.session.execute('vs_web_search',{'query':'a topic'},1)
            repeated,message=self.session.execute('vs_web_search',{'query':'a topic'},2)
        self.assertEqual(first['status'],'ok')
        self.assertEqual(repeated['error_type'],'RepeatedAction')
        self.assertEqual(post.call_count,1)
        self.assertIn('No new request was sent',message['content'])

    def test_failed_search_can_be_retried(self):
        import requests
        failed=Mock(status_code=503)
        failed.raise_for_status.side_effect=requests.HTTPError(response=failed)
        good=Mock(status_code=200)
        good.json.return_value={'organic':[{'link':'https://example.org/page','title':'Evidence'}]}
        with patch.object(evidence,'serper_post',side_effect=[failed,good]) as post:
            first,_=self.session.execute('vs_web_search',{'query':'a topic'},1)
            second,_=self.session.execute('vs_web_search',{'query':'a topic'},2)
        self.assertEqual(first['status'],'error')
        self.assertEqual(second['status'],'ok')
        self.assertEqual(post.call_count,2)


if __name__=='__main__':
    unittest.main()
