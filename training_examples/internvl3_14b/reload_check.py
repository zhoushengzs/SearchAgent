"""Run after the 3-step training. Requires one allocated GPU with enough memory."""

import os, json, math
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel
from safetensors.torch import load_file
from protocol import register

root = Path(
    os.environ.get("SFT_WORK", str(Path.home() / "internvl3-sft-work"))
).resolve()
assert torch.cuda.is_available(), "需要已分配的 NVIDIA GPU"
adapter = root / "outputs/smoke-lora"
state_path = adapter / "trainer_state.json"
if not state_path.exists():
    state_path = adapter / "checkpoint-3/trainer_state.json"
state = json.loads(state_path.read_text())
assert state["global_step"] >= 3, "未完成3步训练"
for entry in state.get("log_history", []):
    for name in ["loss", "grad_norm", "train_loss"]:
        if name in entry:
            assert math.isfinite(float(entry[name])), f"训练出现非有限值: {name}"
weights = load_file(str(adapter / "adapter_model.safetensors"))
assert all(torch.isfinite(t).all().item() for t in weights.values()), (
    "保存的参数中出现NaN或Inf"
)
assert any(
    "lora_B" in n and torch.count_nonzero(t).item() > 0 for n, t in weights.items()
), "未检测到 LoRA 参数更新"
row = json.loads(next((root / "prepared/smoke.jsonl").open()))
processor = AutoProcessor.from_pretrained(root / "model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
processor.image_max_pixels = 262144
processor.image_min_pixels = 1024
processor.crop_to_patches = False
template = register()
template.fix_special_tokens(tokenizer)
roles = {
    "human": "user",
    "gpt": "assistant",
    "observation": "observation",
    "function_call": "function",
}
# Exclude the last training target. Never send the reference answer to generation.
messages = [
    {"role": roles[m["from"]], "content": m["value"]} for m in row["conversations"][:-1]
] + [{"role": "assistant", "content": ""}]
images = [str(root / "data" / x) for x in row["images"]]
messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
ids, _ = template.encode_oneturn(tokenizer, messages, row["system"], row["tools"])
mm = template.mm_plugin.get_mm_inputs(
    images, [], [], [len(images)], [0], [0], [ids], processor
)
base = AutoModelForImageTextToText.from_pretrained(
    root / "model",
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    attn_implementation="sdpa",
    local_files_only=True,
)
model = PeftModel.from_pretrained(base, adapter).eval()
inputs = {
    "input_ids": torch.tensor([ids], device="cuda"),
    "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device="cuda"),
}
inputs.update(
    {
        k: v.to(device="cuda", dtype=torch.bfloat16)
        if v.is_floating_point()
        else v.to("cuda")
        for k, v in mm.items()
    }
)
with torch.inference_mode():
    result = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=256,
        eos_token_id=template.get_stop_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
text = tokenizer.decode(result[0, len(ids) :], skip_special_tokens=True)
assert text.strip(), "重载后输出为空"
report = {
    "status": "RELOAD_AND_GENERATION_PASS_NOT_ACCURACY_TEST",
    "example_id": row["id"],
    "output": text,
    "new_tokens": len(result[0]) - len(ids),
}
(root / "reload-check.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
print(json.dumps(report, ensure_ascii=False, indent=2))
