"""Create an isolated entrypoint for the pinned OpenSearch-VL training code."""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Pinned upstream source no longer has the expected adapter anchor')
    return text.replace(before, after, 1)



def adapt_training_control(train, workflow, launch):
    train = replace_once(train, "from rllm.trainer.agent_trainer import AgentTrainer\n",
        "from rllm.trainer.agent_trainer import AgentTrainer\n"
        "from runtime_setup import configure_runtime\nconfigure_runtime(DatasetRegistry, _bundle_root)\n")
    anchor = '        self.agent = DeepResearchAgent(\n            rollout_engine=rollout_engine,\n            tools=self.tools,\n            system_prompt=self.system_prompt,\n        )\n'
    workflow = replace_once(workflow, anchor, anchor +
        '        self.agent.max_prompt_tokens = int(os.environ.get("GRPO_MAX_PROMPT_LENGTH", self.agent.max_prompt_tokens))\n'
        '        self.agent.max_response_tokens = int(os.environ.get("GRPO_MAX_RESPONSE_LENGTH", self.agent.max_response_tokens))\n'
        '        self.agent.max_llm_calls = int(os.environ.get("GRPO_AGENT_MAX_STEPS", self.agent.max_llm_calls))\n')
    launch = replace_once(launch, 'if [ -f .env ]; then', 'if [ "${GRPO_SKIP_FRAMEWORK_ENV:-0}" != "1" ] && [ -f .env ]; then')
    launch = launch.rstrip() + ' \\\n    "$@"\n'
    return train, workflow, launch


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-output', type=Path,
                        help='Optional new directory for an unadapted upstream reference copy')
    args = parser.parse_args()
    if args.upstream_output is None:
        import py_compile
        integration = root / 'integration'
        for path in sorted(integration.rglob('*.py')):
            py_compile.compile(str(path), doraise=True)
        import sys
        sys.path.insert(0, str(integration))
        from project_runtime.session import get_all_tools
        print('已检查当前项目接入代码；保留六种 SFT 工具：' + ', '.join(sorted(get_all_tools())))
        print('当前 integration 是随项目保存的正式代码，不再从旧官方副本重新覆盖。')
        return
    source = root / 'upstream/OpenSearch-VL/RL/rllm/vision_deepresearch_async_workflow'
    dest = args.upstream_output.expanduser().resolve()
    if dest == (root / 'integration').resolve() or (root / 'integration').resolve() in dest.parents:
        raise SystemExit('参考代码不能写入正在使用的 integration 目录。')
    dest.mkdir(parents=True, exist_ok=False)
    agent = (source / 'deepresearch_agent.py').read_text()
    agent = replace_once(agent, '        if image_path:\n            self._image_paths["img_1"] = image_path\n',
                         '        initial_image_paths = kwargs.pop("initial_image_paths", None)\n'
                         '        if initial_image_paths is None:\n'
                         '            initial_image_paths = [image_path] if image_path else []\n'
                         '        for image_index, initial_path in enumerate(initial_image_paths, 1):\n'
                         '            self._image_paths[f"img_{image_index}"] = initial_path\n')
    workflow = (source / 'deepresearch_workflow.py').read_text()
    workflow = replace_once(workflow, 'from vision_deepresearch_async_workflow.deepresearch_agent import DeepResearchAgent',
                            'from merged_agent import DeepResearchAgent')
    workflow = replace_once(workflow, '                    image_path=raw_images[0],\n',
                            '                    image_path=raw_images[0],\n                    initial_image_paths=raw_images,\n')
    train = (source / 'train_deepresearch_workflow_megatron.py').read_text()
    train = replace_once(train, 'from vision_deepresearch_async_workflow.deepresearch_workflow import (', 'from merged_workflow import (')
    train = replace_once(train, 'from rllm.data.dataset import DatasetRegistry\n',
                          'from rllm.data.dataset import DatasetRegistry\n'
                          'import os\nfrom pathlib import Path\n'
                          '_bundle_root = Path(os.environ["GRPO_DATA_ROOT"]).resolve()\n'
                          'DatasetRegistry._REGISTRY_DIR = str(_bundle_root / "registry")\n'
                          'DatasetRegistry._REGISTRY_FILE = str(_bundle_root / "registry/dataset_registry.json")\n'
                          'DatasetRegistry._DATASET_DIR = str(_bundle_root / "registered")\n')
    launch = (source / 'run/qwen3-vl-8b-single-node.sh').read_text()
    launch = replace_once(launch, 'set -xeuo pipefail', 'set -euo pipefail')
    launch = replace_once(launch, 'cd "${SCRIPT_DIR}/../.."', 'cd "${GRPO_RLLM_ROOT:?Set the OpenSearch-VL RL/rllm directory}"')
    launch = replace_once(launch, 'adv_estimator="rloo"', 'adv_estimator="grpo"')
    launch = replace_once(launch, 'MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct', 'MODEL_PATH="${GRPO_MODEL_PATH:?Set the trained SFT checkpoint path}"')
    launch = replace_once(launch, 'python3 -m vision_deepresearch_async_workflow.train_deepresearch_workflow_megatron', '"${GRPO_PYTHON:-python}" -m merged_train')
    launch = launch.replace('python -c ', '"${GRPO_PYTHON:-python}" -c ')
    launch = replace_once(launch, "exp_name='open_mm_searcher_8b_single_node'", "exp_name='search_vl_rl8k_sftval_grpo_8b'")
    train, workflow, launch = adapt_training_control(train, workflow, launch)
    outputs = {'merged_agent.py': agent, 'merged_workflow.py': workflow, 'merged_train.py': train, 'official_8gpu_grpo.sh': launch}
    for name, text in outputs.items():
        (dest / name).write_text(text)
    license_path = root / 'upstream/OpenSearch-VL/LICENSE'
    (dest / 'LICENSE.OpenSearch-VL').write_bytes(license_path.read_bytes())
    wrapper = '''#!/usr/bin/env bash
set -euo pipefail
GRPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GRPO_DATA_ROOT="$(cd "${GRPO_SCRIPT_DIR}/.." && pwd)"
export GRPO_RLLM_ROOT="${1:-${GRPO_DATA_ROOT}/../../train/OpenSearch-VL/RL/rllm}"
export GRPO_MODEL_PATH="${2:-}"
export GRPO_PYTHON="${GRPO_PYTHON:-python}"
if [ -z "${GRPO_MODEL_PATH}" ]; then
    echo '用法：bash integration/start_grpo.sh "OpenSearch-VL/RL/rllm 的地址" "SFT 模型地址"'
    echo '请先在服务器激活已经安装 OpenSearch-VL 训练依赖的 Conda 环境。'
    exit 2
fi
if [ ! -f "${GRPO_RLLM_ROOT}/rllm/data/dataset.py" ]; then
    echo "找不到官方训练程序：${GRPO_RLLM_ROOT}"
    exit 2
fi
export GRPO_RLLM_ROOT="$(cd "${GRPO_RLLM_ROOT}" && pwd)"
export PYTHONPATH="${GRPO_SCRIPT_DIR}:${GRPO_RLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${GRPO_PYTHON}" "${GRPO_DATA_ROOT}/scripts/register_merged.py" --rllm-root "${GRPO_RLLM_ROOT}"
exec bash "${GRPO_SCRIPT_DIR}/official_8gpu_grpo.sh"
'''
    (dest / 'start_grpo.sh').write_text(wrapper)
    metadata = {'base_repository': 'shawn0728/OpenSearch-VL', 'base_commit': json.loads((root / 'upstream/OpenSearch-VL/revision.json').read_text())['commit'],
                'dataset_registry_name': 'Vision-DeepResearch-QA', 'algorithm': 'grpo', 'upstream_gpu_preset': '8B single-node 8 GPUs',
                'changes': ['Use this bundle\'s isolated dataset registry.', 'Use provided SFT checkpoint instead of the base model.',
                            'Label all initial images and make all of them available to image tools.', 'Use GRPO instead of the upstream RLOO default.',
                            'Disable shell command tracing to avoid printing environment credentials.'],
                'output_sha256': {name: hashlib.sha256((dest / name).read_bytes()).hexdigest() for name in outputs}}
    (dest / 'upstream_reference_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
