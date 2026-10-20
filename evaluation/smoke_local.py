"""A small real image-generation check; does not score a benchmark."""
import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if not torch.backends.mps.is_available():
        raise SystemExit('Apple GPU is unavailable to this process. Run from the local terminal.')
    start = time.monotonic()
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map='mps',
        attn_implementation='sdpa', local_files_only=True,
    ).eval()
    load_seconds = time.monotonic() - start
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': str(Path(a.image).resolve()),
         'max_pixels': 262144},
        {'type': 'text', 'text': 'Briefly describe the visible image in one sentence.'},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt',
    ).to('mps')
    start = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=48, do_sample=False, use_cache=True)
    torch.mps.synchronize()
    answer = processor.batch_decode(
        generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
    )[0]
    result = {'model': a.model, 'image': a.image, 'device': 'mps',
              'dtype': 'bfloat16', 'load_seconds': load_seconds,
              'generation_seconds': time.monotonic() - start,
              'new_tokens': generated.shape[1] - inputs['input_ids'].shape[1],
              'output': answer, 'benchmark_score': None}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
