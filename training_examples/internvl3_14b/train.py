import json
from pathlib import Path
import torch
from transformers import TrainerCallback
from protocol import register

register()
from llamafactory.train.tuner import run_exp


class RecordMemory(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            p = Path(args.output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / f"memory-rank-{args.process_index}.json").write_text(
                json.dumps(
                    {
                        "device": torch.cuda.get_device_name(),
                        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "step": state.global_step,
                        "logs": state.log_history,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    run_exp(callbacks=[RecordMemory()])
