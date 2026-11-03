import hydra

from project_runtime.session import get_all_tools
from merged_workflow import (
    DeepResearchWorkflow,
)

from rllm.data.dataset import DatasetRegistry
import os
from pathlib import Path
_bundle_root = Path(os.environ["GRPO_DATA_ROOT"]).resolve()
DatasetRegistry._REGISTRY_DIR = str(_bundle_root / "registry")
DatasetRegistry._REGISTRY_FILE = str(_bundle_root / "registry/dataset_registry.json")
DatasetRegistry._DATASET_DIR = str(_bundle_root / "registered")
from rllm.rewards.reward_fn import deepresearch_reward_fn_async
from rllm.trainer.agent_trainer import AgentTrainer
from runtime_setup import configure_runtime
configure_runtime(DatasetRegistry, _bundle_root)


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer_megatron",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("Vision-DeepResearch-QA", "train")
    test_dataset = DatasetRegistry.load_dataset("Vision-DeepResearch-QA", "test")

    trainer = AgentTrainer(
        workflow_class=DeepResearchWorkflow,
        workflow_args={
            "reward_function": deepresearch_reward_fn_async,
            "tools": get_all_tools(),
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
