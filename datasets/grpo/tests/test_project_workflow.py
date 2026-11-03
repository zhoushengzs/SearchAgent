"""Exercise the real rLLM episode/reward conversion with scripted model outputs.

Requires the installed training environment. This is not a GPU training test.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'integration'))
from merged_workflow import DeepResearchWorkflow
from project_runtime.session import get_all_tools


class WorkflowTests(unittest.TestCase):
    def run_case(self, tool, arguments, query_score):
        class Engine:
            def __init__(self):
                self.calls = 0
            async def get_model_response(self, messages, **kwargs):
                self.calls += 1
                content = ('<tool_call>' + json.dumps({'name':tool,'arguments':arguments}) + '</tool_call>'
                           if self.calls == 1 else '<response>red</response>')
                return SimpleNamespace(text='<think>Check evidence.</think>'+content,
                                       prompt_length=100,completion_length=30)
        async def reward(task, prediction):
            return SimpleNamespace(is_correct=prediction=='red',metadata={},reward=1.0)
        with ThreadPoolExecutor(1) as executor, patch('merged_workflow._judge_query_utility', new=AsyncMock(return_value=query_score)):
            workflow = DeepResearchWorkflow(Engine(), executor, tools=get_all_tools(), reward_function=reward)
            return asyncio.run(workflow.run({'question':'What color?','answer':'red',
                                            'images':[Image.new('RGB',(1000,400),'red')]},'offline-workflow-test'))

    def test_crop_reaches_training_step_with_correct_reward(self):
        episode = self.run_case('vs_crop',{'asset_id':'image_1','bbox':[0,0,0.5,1]},0.5)
        self.assertTrue(episode.is_correct)
        self.assertEqual(len(episode.trajectories[0].steps),2)
        self.assertAlmostEqual(episode.trajectories[0].reward,0.95)
        last_context=episode.trajectories[0].steps[-1].chat_completions
        self.assertEqual(last_context[-2]['images'][0].size,(500,400))
        self.assertEqual(episode.metrics['reward/format'],1.0)
        self.assertEqual(episode.metrics['tools/calls'],1)
        self.assertEqual(episode.metrics['tools/images_returned'],1)

    def test_tool_failure_is_recorded_and_reduces_reward(self):
        episode = self.run_case('vs_inspect_image',{'asset_id':'nonexistent'},0.0)
        self.assertTrue(episode.trajectories[0].steps[0].info['step_error'])
        self.assertEqual(episode.metrics['reward/format'],0.5)
        self.assertAlmostEqual(episode.trajectories[0].reward,0.85)
        self.assertEqual(episode.metrics['tools/errors'],1)


if __name__=='__main__':
    unittest.main()
