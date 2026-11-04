import argparse, json
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoProcessor, AutoTokenizer
from llamafactory.data.template import TEMPLATES
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from protocol import register

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.data)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
processor.image_max_pixels = 262144
processor.image_min_pixels = 1024
processor.crop_to_patches = False
base = TEMPLATES["intern_vl"]
template = register()
template.fix_special_tokens(tokenizer)
options = SimpleNamespace(mask_history=True, train_on_prompt=False, cutoff_len=1000000)
encoder = SupervisedDatasetProcessor(
    template=template, tokenizer=tokenizer, processor=processor, data_args=options
)
roles = {
    "human": "user",
    "gpt": "assistant",
    "observation": "observation",
    "function_call": "function",
}
selected = []
reports = []
observed = False
for line in (root / "train.jsonl").open():
    row = json.loads(line)
    has_observation = any(m["from"] == "observation" for m in row["conversations"])
    if len(selected) >= 8 and (observed or not has_observation):
        if observed:
            break
        continue
    messages = [
        {"role": roles[m["from"]], "content": m["value"]} for m in row["conversations"]
    ]
    images = [str(root / x) for x in row["images"]]
    ids, labels = encoder._encode_data_example(
        messages[:-1], messages[-1:], row["system"], row["tools"], images, [], []
    )
    if len(ids) > 8192:
        continue
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    supervised = tokenizer.decode(
        [x for x in labels if x != -100], skip_special_tokens=False
    )
    assert row["conversations"][-1]["value"] in supervised
    assert "<IMG_CONTEXT>" in decoded and ids.count(
        tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    ) == 256 * len(images)
    assert "Action Input:" not in template.format_tools.apply(content=row["tools"])[0]
    assert "<tool_call>" in template.format_tools.apply(content=row["tools"])[0]
    if has_observation:
        assert "<tool_response>" in decoded
    selected.append(row)
    observed |= has_observation
    reports.append(
        {
            "id": row["id"],
            "tokens": len(ids),
            "label_tokens": sum(x != -100 for x in labels),
            "images": len(images),
            "observation": has_observation,
        }
    )
    if has_observation:
        (out / "rendered-training-example.txt").write_text(decoded)
assert observed and len(selected) >= 8
with (out / "smoke.jsonl").open("w") as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
info = json.loads((root / "dataset_info.json").read_text())["train"]
info["file_name"] = "smoke.jsonl"
(out / "dataset_info.json").write_text(
    json.dumps({"smoke": info}, ensure_ascii=False, indent=2)
)
result = {
    "status": "CPU_PASS_NOT_GPU_TRAINED",
    "transformers": __import__("transformers").__version__,
    "torch": __import__("torch").__version__,
    "examples": reports,
    "default_intern_vl_has_conflicting_action_instruction": "Action Input:"
    in base.format_tools.apply(content=selected[0]["tools"])[0],
}
(out / "cpu-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
