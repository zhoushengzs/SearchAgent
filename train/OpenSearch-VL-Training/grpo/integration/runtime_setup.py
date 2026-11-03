"""Keep dataset selection and tool settings consistent inside Ray workers."""
import os
from pathlib import Path

def configure_runtime(registry, bundle_root):
    selected = Path(os.environ.get("GRPO_REGISTRY_ROOT", str(bundle_root))).resolve()
    registry._REGISTRY_DIR = str(selected / "registry")
    registry._REGISTRY_FILE = str(selected / "registry/dataset_registry.json")
    registry._DATASET_DIR = str(selected / "registered")
    import rllm.trainer.verl.ray_runtime_env as runtime_env
    for prefix in ["GRPO_", "PYTHONPATH", "SERP_", "SERPER_", "JUDGE_", "OPENAI_", "TOGETHER_", "API_GATEWAY_", "LAYOUT_", "COS_", "QWEN_"]:
        if prefix not in runtime_env.FORWARD_PREFIXES:
            runtime_env.FORWARD_PREFIXES.append(prefix)
