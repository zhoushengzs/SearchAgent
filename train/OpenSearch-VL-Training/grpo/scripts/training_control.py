"""Prepare/register data and launch 1/2-GPU checks or 8-GPU GRPO runs."""
from __future__ import annotations
import argparse
import datetime
import importlib.util
import json
import os
import platform
import random
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / '.codex/server_settings.json'


def setup():
    default = ROOT.parents[1] / 'train/OpenSearch-VL-Training/RL/rllm'
    code = Path(input(f'官方训练代码 RL/rllm 的完整地址 [默认 {default}]：').strip().strip('"\'') or default).expanduser().resolve()
    model = Path(input('已经训练好的 SFT 模型完整地址：').strip().strip('"\'')).expanduser().resolve()
    if not (code / 'rllm/data/dataset.py').is_file():
        raise SystemExit('这个地址下没有官方 rllm/data/dataset.py，请重新检查。')
    if not (model / 'config.json').is_file():
        raise SystemExit('模型地址下没有 config.json；请填写完整模型文件夹，不是单独的权重文件。')
    SETTINGS.parent.mkdir(exist_ok=True)
    SETTINGS.write_text(json.dumps({'rllm_root': str(code), 'model_path': str(model)}, ensure_ascii=False, indent=2) + '\n')
    env = ROOT / 'integration/local_training.env'
    if not env.exists():
        env.write_text((ROOT / 'integration/training.env.example').read_text())
        env.chmod(0o600)
    print('已保存本机地址：', SETTINGS)
    print('请在这个文件中填写搜索和评分服务：', env)
    print('接下来运行：python scripts/training_control.py check')


def settings(args):
    saved = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    code = args.rllm_root or saved.get('rllm_root') or ROOT.parents[1] / 'train/OpenSearch-VL-Training/RL/rllm'
    model = args.model_path or saved.get('model_path')
    return Path(code).expanduser().resolve(), Path(model).expanduser().resolve() if model else None


def environment(code):
    env = os.environ.copy()
    # Build helpers (for example ninja) must come from the selected Python environment.
    env['PATH'] = os.pathsep.join(dict.fromkeys([str(Path(sys.executable).parent)] +
        [part for part in env.get('PATH', '').split(os.pathsep) if part]))
    local = ROOT / 'integration/local_training.env'
    source = local if local.exists() else code / '.env'
    if source.exists():
        try:
            from dotenv import dotenv_values
        except ImportError:
            raise SystemExit('当前环境缺少 python-dotenv，请先安装官方训练依赖。')
        for key, value in dotenv_values(source).items():
            if value:
                env[key] = value
    # These six SFT tools read webpages directly; discard legacy service keys.
    env = {key: value for key, value in env.items() if not key.startswith('JINA_')}
    for a, b in [('SERP_API_KEY', 'SERPER_API_KEY'), ('JUDGE_BASE_URL', 'JUDGE_API_BASE_URL')]:
        if env.get(a) and not env.get(b): env[b] = env[a]
        if env.get(b) and not env.get(a): env[a] = env[b]
    env['GRPO_SKIP_FRAMEWORK_ENV'] = '1'
    env['GRPO_DATA_ROOT'] = str(ROOT)
    env['GRPO_RLLM_ROOT'] = str(code)
    env['GRPO_PYTHON'] = sys.executable
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'integration'), str(code), str(code / 'verl'), env.get('PYTHONPATH', '')])
    # Fresh SGLang subprocesses need the CUDA libraries installed in this
    # Python environment, even if the host has a different CUDA version.
    site_packages = Path(sysconfig.get_paths()['purelib'])
    library_dirs = sorted((site_packages / 'nvidia').glob('*/lib'))
    torch_lib = site_packages / 'torch/lib'
    if torch_lib.is_dir():
        library_dirs.append(torch_lib)
    # The migrated AutoDL base environment ships CUDA 13 runtime libraries
    # (nvidia/cu13/lib); the CUDA-13 build of Transformer Engine 2.6 needs them
    # even though torch itself runs on cu128. Only used when that directory exists.
    base_prefix = Path(sys.executable).resolve().parents[3]
    library_dirs.extend(sorted(base_prefix.glob('lib/python*/site-packages/nvidia/cu13/lib')))
    if library_dirs:
        previous = [p for p in env.get('LD_LIBRARY_PATH', '').split(os.pathsep) if p]
        env['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys(
            [str(p) for p in library_dirs if p.is_dir()] + previous
        ))
    return env


def inspect_server(code, model, env):
    modules = ['torch', 'pandas', 'polars', 'pyarrow', 'PIL', 'ray', 'hydra', 'sglang', 'transformer_engine', 'megatron', 'mbridge', 'dotenv']
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    print('Python 地址：', sys.executable)
    print('系统：', platform.system())
    print('官方代码：', code, '存在' if (code / 'rllm/data/dataset.py').is_file() else '不存在')
    print('SFT 模型：', model or '尚未设置')
    print('缺少的依赖：', ', '.join(missing) if missing else '未发现')
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], text=True, timeout=15)
        print('服务器显卡：\n' + gpu)
    except (OSError, subprocess.SubprocessError):
        print('没有读取到 NVIDIA 显卡；本机可整理数据，但不能执行这里的正式 GPU 训练。')
    from project_runtime.serper_keys import configured_keys
    print('工具：六种 SFT 工具；文字找图片，无需 COS；网页内容直接读取。')
    for label, value in [('网页与图片搜索', bool(configured_keys(env))),
                         ('答案评分', bool(env.get('JUDGE_API_KEY') or env.get('OPENAI_API_KEY'))),
                         ('评分服务地址', bool(env.get('JUDGE_BASE_URL'))), ('评分模型', bool(env.get('JUDGE_MODEL')))]:
        print(label + '：' + ('已设置' if value else '未设置'))
    print('本命令不调用搜索或评分接口，不启动训练。')
    return missing


def profile(name, run):
    short = name in ['debug1', 'debug2', 'smoke8', 'smoke4']
    single = name == 'debug1'
    small = name in ['debug1', 'debug2']
    gpus = {'debug1': 1, 'debug2': 2, 'smoke4': 4, 'smoke8': 8, 'train4': 4, 'train8': 8}[name]
    model_parts = 1 if single else 2
    four = name in ['smoke4', 'train4']
    values = {
        'algorithm.adv_estimator': 'grpo', 'algorithm.use_kl_in_reward': False,
        'trainer.n_gpus_per_node': gpus, 'trainer.nnodes': 1,
        'actor_rollout_ref.actor.megatron.tensor_model_parallel_size': model_parts,
        'actor_rollout_ref.actor.megatron.pipeline_model_parallel_size': 1,
        'actor_rollout_ref.actor.megatron.context_parallel_size': 1,
        'actor_rollout_ref.ref.megatron.tensor_model_parallel_size': model_parts,
        'actor_rollout_ref.ref.megatron.pipeline_model_parallel_size': 1,
        'actor_rollout_ref.ref.megatron.context_parallel_size': 1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': model_parts if small else (2 if four else 4),
        'actor_rollout_ref.actor.use_kl_loss': False,
        'data.train_batch_size': gpus if small else (4 if short else (16 if four else 32)),
        'data.val_batch_size': gpus if small else 4,
        'actor_rollout_ref.actor.ppo_mini_batch_size': gpus if small else (4 if short else (4 if four else 8)),
        'actor_rollout_ref.rollout.n': 2 if short else 4,
        'rllm.workflow.n_parallel_tasks': 1 if small else (2 if short else (4 if four else 8)),
        'rllm.workflow.n_parallel_tools': 1 if single else (2 if small else (4 if short else (8 if four else 16))),
        'data.max_prompt_length': 8192,
        'data.max_response_length': 1024 if single else (2048 if short else 4096),
        'actor_rollout_ref.rollout.max_num_seqs': 2 if single else (4 if short else 32),
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.35 if single else (0.45 if short else 0.65),
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': 12288 if short else 16384,
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': 12288 if short else 16384,
        'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu': 12288 if short else 16384,
        'trainer.total_epochs': 1, 'trainer.total_training_steps': 3 if short else None,
        'trainer.save_freq': 1 if short else 50, 'trainer.test_freq': 1 if short else 50,
        'trainer.val_before_train': False, 'trainer.resume_mode': 'disable',
        'trainer.experiment_name': run.name, 'trainer.default_local_dir': str(run / 'checkpoints'),
        'trainer.log_episodes': True, 'trainer.episode_log_dir': str(run / 'episodes'),
        'trainer.logger': '["console"]',
    }
    if name == 'train4':
        # 4-GPU 正式训练：每个保存间隔产生完整模型+优化器，磁盘只有 ~378G，
        # 只保留最近一次 checkpoint。
        values['trainer.max_actor_ckpt_to_keep'] = 1
    if name == 'debug2':
        # Keep optimizer state on the two GPUs throughout phase changes.
        # CPU offload exceeded this server's 220 GB host-memory limit.
        values['actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction'] = 0.0
        values['actor_rollout_ref.actor.megatron.optimizer_offload'] = False
        # Leave room for the resident optimizer while generating answers.
        values['actor_rollout_ref.rollout.gpu_memory_utilization'] = 0.25
    if single:
        # One H20 has 96 GB VRAM and this instance has 150 GB host memory.
        # Balance the 150 GB host limit against 96 GB VRAM; avoid copy/step overlap.
        values['actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction'] = 0.75
        values['actor_rollout_ref.actor.megatron.optimizer_offload'] = False
        values['actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d'] = False
        values['+actor_rollout_ref.actor.optim.override_optimizer_config.pin_cpu_grads'] = False
        values['+actor_rollout_ref.actor.optim.override_optimizer_config.pin_cpu_params'] = False
        values['actor_rollout_ref.actor.megatron.grad_offload'] = False
        values['actor_rollout_ref.rollout.gpu_memory_utilization'] = 0.25
        # Megatron sequence parallelism is unavailable with only one model part.
        values['actor_rollout_ref.actor.megatron.sequence_parallel'] = False
        values['actor_rollout_ref.ref.megatron.sequence_parallel'] = False
    def literal(value):
        if value is None:return 'null'
        if isinstance(value, bool):return str(value)
        return str(value)
    return [f'{key}={literal(value)}' for key, value in values.items()]


def choose_debug_rows(data, limit):
    data = list(data)
    random.Random(42).shuffle(data)
    first = []
    for predicate in [lambda r: len(r['images']) > 1,
                      lambda r: r['id'].startswith('sft_val:'),
                      lambda r: r['id'].startswith('rl8k:')]:
        match = next((r for r in data if predicate(r) and r['id'] not in {x['id'] for x in first}), None)
        if match is not None:first.append(match)
    used = {r['id'] for r in first}
    return (first + [r for r in data if r['id'] not in used])[:limit]


def prepare_registry(code, env, run, short):
    subprocess.run([sys.executable, str(ROOT / 'scripts/register_merged.py'), '--rllm-root', str(code)], env=env, check=True)
    if not short:
        return ROOT
    sys.path[:0] = [str(ROOT / 'scripts'), str(code), str(code / 'verl')]
    from register_merged import load_registry, bind_registry
    cls = load_registry(ROOT, code)
    bind_registry(cls, ROOT)
    chosen = {}
    for split, limit in [('train', 16), ('test', 4)]:
        data = cls.load_dataset('Vision-DeepResearch-QA', split).get_data()
        chosen[split] = choose_debug_rows(data, limit)
    registry_root = run / 'debug_data'
    bind_registry(cls, registry_root)
    for split, data in chosen.items():cls.register_dataset('Vision-DeepResearch-QA', data, split)
    print('独立调试题单：16 道训练题、4 道训练验证题；不使用 benchmark 测试题。')
    return registry_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['setup', 'check', 'register', 'debug1', 'debug2', 'smoke4', 'smoke8', 'train4', 'train8'])
    parser.add_argument('--rllm-root', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--gpus', help='例如单卡 0、双卡 0,1；指定时仅使用这些显卡')
    parser.add_argument('--dry-run', action='store_true', help='只显示启动参数，不登记数据、不调用模型')
    parser.add_argument('--config-only', action='store_true', help='在完整训练环境中检查配置，不开始训练')
    parser.add_argument('--no-save', action='store_true', help='仅限短程调试：不写模型文件，结果不能用于继续训练')
    args = parser.parse_args()
    if args.mode == 'setup':return setup()
    sys.path.insert(0, str(ROOT / 'integration'))
    code, model = settings(args)
    env = environment(code)
    if args.mode == 'check':return inspect_server(code, model, env)
    if args.mode == 'register':
        subprocess.run([sys.executable, str(ROOT / 'scripts/register_merged.py'), '--rllm-root', str(code)], env=env, check=True)
        return
    short = args.mode not in ('train8', 'train4')
    expected = {'debug1': 1, 'debug2': 2, 'smoke4': 4, 'smoke8': 8, 'train4': 4, 'train8': 8}[args.mode]
    if args.gpus:
        ids = args.gpus.split(',')
        if len(ids) != expected or len(set(ids)) != expected:
            raise SystemExit(f'{args.mode} 需要指定 {expected} 张不同的卡。')
        env['CUDA_VISIBLE_DEVICES'] = args.gpus
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = ROOT / 'runs' / (args.mode + '_' + stamp)
    overrides = profile(args.mode, run)
    if args.no_save:
        if not short:
            raise SystemExit('--no-save 只用于短程调试；正式训练保留完整模型保存。')
        overrides.append('trainer.save_freq=0')
    command = ['bash', str(ROOT / 'integration/official_8gpu_grpo.sh'), *overrides]
    if args.dry_run:
        print('模式：', args.mode, '；需要显卡：', expected)
        print('调试题单：16/4' if short else '正式题单：7940/882')
        print('输出目录：', run)
        print('SFT 模型：', model or '尚未设置；正式启动前必须设置')
        print(shlex.join(command))
        print('仅展示参数，没有启动训练。单卡/双卡显存是否足够不能由这个检查确定。')
        return
    if model is None or not (model / 'config.json').is_file():raise SystemExit('请先执行 setup，设置有效的 SFT 模型文件夹。')
    if args.config_only:
        env['GRPO_MODEL_PATH'] = str(model)
        subprocess.run(command + ['--cfg', 'job'], env=env, check=True)
        print('配置展开完成；没有开始训练。')
        return
    missing = inspect_server(code, model, env)
    if missing:raise SystemExit('训练依赖尚不完整，请在已安装官方训练环境的 Conda 环境中运行。')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < expected:
        raise SystemExit(f'没有找到 {expected} 张可用 NVIDIA 显卡。')
    if torch.cuda.get_device_capability(0)[0] >= 12:
        # This Blackwell setup uses FlashAttention 2 for packed training inputs.
        # The installed cuDNN path mixes CUDA runtimes; keep Hopper defaults.
        env.setdefault('NVTE_FUSED_ATTN', '0')
        print('当前 Blackwell 显卡使用已检查的 FlashAttention 2 训练计算方式。')
    if not env.get('JUDGE_BASE_URL') or not env.get('JUDGE_MODEL') or not (env.get('JUDGE_API_KEY') or env.get('OPENAI_API_KEY')):
        raise SystemExit('先配置答案评分服务，避免所有参考长答案仅按文字比较。')
    from project_runtime.serper_keys import configured_keys
    if not configured_keys(env):
        raise SystemExit('先配置 SERPER_API_KEYS 或 SERPER_API_KEY，再调试真实搜索任务。')
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_project_tools.py'), '--live'], env=env, check=True)
    run.mkdir(parents=True, exist_ok=False)
    env['GRPO_MODEL_PATH'] = str(model)
    env['GRPO_REGISTRY_ROOT'] = str(prepare_registry(code, env, run, short))
    env['GRPO_MAX_PROMPT_LENGTH'] = '8192'
    env['GRPO_MAX_RESPONSE_LENGTH'] = '1024' if args.mode == 'debug1' else ('2048' if short else '4096')
    env['GRPO_AGENT_MAX_STEPS'] = '6' if short else '20'
    metadata = {'mode': args.mode, 'model_path': str(model), 'rllm_root': str(code), 'registry_root': env['GRPO_REGISTRY_ROOT'],
                'command': command, 'output_directory': str(run), 'benchmark_used': False, 'tool_profile': 'sft_six_tools', 'requires_cos': False}
    (run / 'launch.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print('现在开始调用真实模型、搜索和评分服务。日志：', run / 'training.log', flush=True)
    with (run / 'training.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line);log.flush()
            status = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            raise
    if status:raise SystemExit(status)
    if args.no_save:
        print('短程检查已经结束；本次未保存模型。评分、模型更新和验证结果见上述日志。')
    else:
        print('程序已经结束。请继续确认日志里有有效评分、实际模型更新、保存文件，以及重新读取检查。')


if __name__ == '__main__':main()
