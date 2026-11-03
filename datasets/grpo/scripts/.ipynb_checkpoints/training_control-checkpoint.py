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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / '.codex/server_settings.json'


def setup():
    default = ROOT.parents[1] / 'train/OpenSearch-VL/RL/rllm'
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
    code = args.rllm_root or saved.get('rllm_root') or ROOT.parents[1] / 'train/OpenSearch-VL/RL/rllm'
    model = args.model_path or saved.get('model_path')
    return Path(code).expanduser().resolve(), Path(model).expanduser().resolve() if model else None


def environment(code):
    env = os.environ.copy()
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
    for a, b in [('SERP_API_KEY', 'SERPER_API_KEY'), ('JUDGE_BASE_URL', 'JUDGE_API_BASE_URL')]:
        if env.get(a) and not env.get(b): env[b] = env[a]
        if env.get(b) and not env.get(a): env[a] = env[b]
    env['GRPO_SKIP_FRAMEWORK_ENV'] = '1'
    env['GRPO_DATA_ROOT'] = str(ROOT)
    env['GRPO_RLLM_ROOT'] = str(code)
    env['GRPO_PYTHON'] = sys.executable
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'integration'), str(code), str(code / 'verl'), env.get('PYTHONPATH', '')])
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
    except (FileNotFoundError, subprocess.SubprocessError):
        print('没有读取到 NVIDIA 显卡；本机可整理数据，但不能执行这里的正式 GPU 训练。')
    for label, value in [('网页搜索', bool(env.get('SERP_API_KEY') or env.get('API_GATEWAY_KEY'))),
                         ('网页访问', bool(env.get('JINA_API_KEY') or env.get('API_GATEWAY_KEY'))),
                         ('答案评分', bool(env.get('JUDGE_API_KEY') or env.get('OPENAI_API_KEY'))),
                         ('评分服务地址', bool(env.get('JUDGE_BASE_URL'))), ('评分模型', bool(env.get('JUDGE_MODEL')))]:
        print(label + '：' + ('已设置' if value else '未设置'))
    print('本命令不调用搜索或评分接口，不启动训练。')
    return missing


def profile(name, run):
    short = name in ['debug1', 'debug2', 'smoke8']
    single = name == 'debug1'
    small = name in ['debug1', 'debug2']
    gpus = 1 if single else (2 if name == 'debug2' else 8)
    model_parts = 1 if single else 2
    values = {
        'algorithm.adv_estimator': 'grpo', 'algorithm.use_kl_in_reward': False,
        'trainer.n_gpus_per_node': gpus, 'trainer.nnodes': 1,
        'actor_rollout_ref.actor.megatron.tensor_model_parallel_size': model_parts,
        'actor_rollout_ref.actor.megatron.pipeline_model_parallel_size': 1,
        'actor_rollout_ref.actor.megatron.context_parallel_size': 1,
        'actor_rollout_ref.ref.megatron.tensor_model_parallel_size': model_parts,
        'actor_rollout_ref.ref.megatron.pipeline_model_parallel_size': 1,
        'actor_rollout_ref.ref.megatron.context_parallel_size': 1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': model_parts if small else 4,
        'actor_rollout_ref.actor.use_kl_loss': False,
        'data.train_batch_size': gpus if small else (4 if short else 32),
        'data.val_batch_size': gpus if small else 4,
        'actor_rollout_ref.actor.ppo_mini_batch_size': gpus if small else (4 if short else 8),
        'actor_rollout_ref.rollout.n': 2 if short else 4,
        'rllm.workflow.n_parallel_tasks': 1 if small else (2 if short else 8),
        'rllm.workflow.n_parallel_tools': 1 if single else (2 if small else (4 if short else 16)),
        'data.max_prompt_length': 4096 if single else 8192,
        'data.max_response_length': 1024 if single else (2048 if short else 4096),
        'actor_rollout_ref.rollout.max_num_seqs': 2 if single else (4 if short else 32),
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.35 if single else (0.45 if short else 0.65),
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': 8192 if single else (12288 if short else 16384),
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': 8192 if single else (12288 if short else 16384),
        'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu': 8192 if single else (12288 if short else 16384),
        'trainer.total_epochs': 1, 'trainer.total_training_steps': 3 if short else None,
        'trainer.save_freq': 1 if short else 50, 'trainer.test_freq': 1 if short else 50,
        'trainer.val_before_train': False, 'trainer.resume_mode': 'disable',
        'trainer.experiment_name': run.name, 'trainer.default_local_dir': str(run / 'checkpoints'),
        'trainer.log_episodes': True, 'trainer.episode_log_dir': str(run / 'episodes'),
        'trainer.logger': '["console"]',
    }
    if single:
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
    parser.add_argument('mode', choices=['setup', 'check', 'register', 'debug1', 'debug2', 'smoke8', 'train8'])
    parser.add_argument('--rllm-root', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--gpus', help='例如单卡 0、双卡 0,1；指定时仅使用这些显卡')
    parser.add_argument('--dry-run', action='store_true', help='只显示启动参数，不登记数据、不调用模型')
    parser.add_argument('--config-only', action='store_true', help='在完整训练环境中检查配置，不开始训练')
    args = parser.parse_args()
    if args.mode == 'setup':return setup()
    code, model = settings(args)
    env = environment(code)
    if args.mode == 'check':return inspect_server(code, model, env)
    if args.mode == 'register':
        subprocess.run([sys.executable, str(ROOT / 'scripts/register_merged.py'), '--rllm-root', str(code)], env=env, check=True)
        return
    short = args.mode != 'train8'
    expected = 1 if args.mode == 'debug1' else (2 if args.mode == 'debug2' else 8)
    if args.gpus:
        ids = args.gpus.split(',')
        if len(ids) != expected or len(set(ids)) != expected:
            raise SystemExit(f'{args.mode} 需要指定 {expected} 张不同的卡。')
        env['CUDA_VISIBLE_DEVICES'] = args.gpus
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = ROOT / 'runs' / (args.mode + '_' + stamp)
    overrides = profile(args.mode, run)
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
    if not env.get('JUDGE_BASE_URL') or not env.get('JUDGE_MODEL') or not (env.get('JUDGE_API_KEY') or env.get('OPENAI_API_KEY')):
        raise SystemExit('先配置答案评分服务，避免所有参考长答案仅按文字比较。')
    if not (env.get('SERP_API_KEY') or env.get('API_GATEWAY_KEY')):
        raise SystemExit('先配置搜索服务，再调试真实搜索任务。')
    run.mkdir(parents=True, exist_ok=False)
    env['GRPO_MODEL_PATH'] = str(model)
    env['GRPO_REGISTRY_ROOT'] = str(prepare_registry(code, env, run, short))
    env['GRPO_MAX_PROMPT_LENGTH'] = '4096' if args.mode == 'debug1' else '8192'
    env['GRPO_MAX_RESPONSE_LENGTH'] = '1024' if args.mode == 'debug1' else ('2048' if short else '4096')
    env['GRPO_AGENT_MAX_STEPS'] = '6' if short else '20'
    metadata = {'mode': args.mode, 'model_path': str(model), 'rllm_root': str(code), 'registry_root': env['GRPO_REGISTRY_ROOT'],
                'command': command, 'output_directory': str(run), 'benchmark_used': False}
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
    print('程序已经结束。请继续确认日志里有有效评分、实际模型更新、保存文件，以及重新读取检查。')


if __name__ == '__main__':main()
