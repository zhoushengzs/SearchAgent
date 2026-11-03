"""Small real-CUDA checks for host-copy policy; does not load the model."""
from types import SimpleNamespace
from pathlib import Path
import ast
import asyncio
import os
import unittest
from unittest.mock import patch, Mock, AsyncMock
import torch

from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
import verl.utils.megatron_utils as memory_helpers


@unittest.skipUnless(torch.cuda.is_available(), 'Requires a visible NVIDIA GPU')
class HostCopyTests(unittest.TestCase):
    def optimizer(self, pinned):
        parameter=torch.nn.Parameter(torch.ones(8,device='cuda',dtype=torch.float32))
        optimizer=HybridDeviceOptimizer([parameter],offload_fraction=1.0,
            cpu_optimizer_cls=torch.optim.AdamW,gpu_optimizer_cls=torch.optim.AdamW,
            param_update_in_fp32=True,pin_cpu_params=pinned,pin_cpu_grads=pinned,
            overlap_cpu_optimizer_d2h_h2d=False,lr=0.1,weight_decay=0)
        return parameter,optimizer

    def test_unpinned_optimizer_copy_and_real_update(self):
        parameter,optimizer=self.optimizer(False)
        self.assertTrue(optimizer.gpu_params_map_cpu_copy)
        self.assertTrue(all(not p.is_pinned() for p in optimizer.gpu_params_map_cpu_copy.values()))
        parameter.grad=torch.ones_like(parameter)
        optimizer.step()
        torch.cuda.synchronize()
        torch.testing.assert_close(parameter,torch.full_like(parameter,0.9),atol=1e-6,rtol=1e-6)

    def test_pinned_policy_still_works_when_requested(self):
        _,optimizer=self.optimizer(True)
        self.assertTrue(all(p.is_pinned() for p in optimizer.gpu_params_map_cpu_copy.values()))

    def test_model_round_trip_releases_unpinned_host_mirror(self):
        values=torch.arange(8,device='cuda',dtype=torch.float32)
        buffer=SimpleNamespace(param_data=values.clone(),grad_data=torch.zeros_like(values))
        model=SimpleNamespace(buffers=[buffer],expert_parallel_buffers=[])
        with patch.object(memory_helpers,'DDP',SimpleNamespace):
            for _ in range(2):
                memory_helpers.offload_megatron_model_to_cpu([model],pin_memory=False)
                self.assertFalse(buffer.param_data.cpu_data.is_pinned())
                self.assertEqual(buffer.param_data.untyped_storage().nbytes(),0)
                memory_helpers.load_megatron_model_to_gpu([model])
                torch.cuda.synchronize()
                torch.testing.assert_close(buffer.param_data,values)
                self.assertFalse(hasattr(buffer.param_data,'cpu_data'))

    def test_unpinned_cpu_gradients_release_and_next_update_still_works(self):
        parameter,optimizer=self.optimizer(False)
        for expected in [0.9,0.8]:
            optimizer.zero_grad(set_to_none=True)
            parameter.grad=torch.ones_like(parameter)
            optimizer.step()
            torch.cuda.synchronize()
            torch.testing.assert_close(parameter,torch.full_like(parameter,expected),atol=1e-6,rtol=1e-6)
            self.assertFalse(optimizer.cpu_copy_map_grad)
            self.assertTrue(all(p.grad is None for sub in optimizer.cpu_optimizers for group in sub.param_groups for p in group['params']))

    def test_pinned_gradient_cache_reattaches_after_zero_grad(self):
        parameter,optimizer=self.optimizer(True)
        for expected in [0.9,0.8]:
            optimizer.zero_grad(set_to_none=True)
            parameter.grad=torch.ones_like(parameter)
            optimizer.step()
            torch.cuda.synchronize()
            torch.testing.assert_close(parameter,torch.full_like(parameter,expected),atol=1e-6,rtol=1e-6)

    def test_actual_rollout_transition_forwards_host_copy_policy(self):
        current=Path(__file__).resolve()
        default=current.parents[2]/'RL/rllm'
        if not default.is_dir():default=current.parents[3]/'train/OpenSearch-VL-Training/RL/rllm'
        source=Path(os.environ.get('GRPO_RLLM_ROOT',str(default)))/'verl/verl/workers/megatron_workers.py'
        tree=ast.parse(source.read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='rollout_mode')
        method.decorator_list=[]
        for pinned in [False,True]:
            with self.subTest(pinned=pinned):
                offload=Mock()
                cuda=SimpleNamespace(get_rng_state=lambda:'current-state',set_rng_state=Mock())
                namespace={'set_expandable_segments':Mock(),'aggressive_empty_cache':Mock(),
                    'load_megatron_model_to_gpu':Mock(),'offload_megatron_model_to_cpu':offload,
                    'log_gpu_memory_usage':Mock(),'logger':None,'get_torch_device':lambda:cuda}
                module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
                exec(compile(module,str(source),'exec'),namespace)
                actor=SimpleNamespace(_is_offload_param=True,actor=SimpleNamespace(actor_module=[object()]),
                    bridge=SimpleNamespace(export_weights=lambda models:()),
                    config=SimpleNamespace(actor=SimpleNamespace(optim={'override_optimizer_config':{'pin_cpu_params':pinned}}),
                                           rollout=SimpleNamespace(free_cache_engine=False)),
                    rollout=SimpleNamespace(update_weights=AsyncMock()),gen_random_states='generation-state')
                asyncio.run(namespace['rollout_mode'](actor))
                self.assertIs(offload.call_args.kwargs['pin_memory'],pinned)


if __name__=='__main__':
    unittest.main()
