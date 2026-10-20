'Preserve complete benchmark inputs without inventing agent observations.'
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.provenance import file_sha, fingerprint
from sources.huggingface import _extract_answer, _first_field, save_image_from_field

TASK_SCHEMA = "visual-agent-task-v1"


def task_fingerprint(task: dict) -> str:
    return fingerprint({"question": task["model_input"]["question"],
                        "images": [{k: v for k, v in im.items() if k != "path"}
                                   for im in task["model_input"]["images"]],
                        "dataset_provenance": task["dataset_provenance"]})


def normalize_benchmark_task(example: dict[str, Any], config: dict[str, Any], index: int,
                             image_cache_dir: Path, source_name: str,
                             skip_missing_answer: bool = True, image_base_dir: Path | None = None,
                             used_split: str | None = None) -> tuple[dict | None, str | None]:
    if config.get("usage") != "evaluation" or not config.get("environment"):
        return None, "source has no registered benchmark environment"
    _, question = _first_field(example, config["question_fields"])
    if not isinstance(question, str) or not question.strip():
        return None, "missing benchmark question"
    _, answer_value = _first_field(example, config["answer_fields"])
    answer = _extract_answer(answer_value, config.get("answer_selection", "first"))
    if skip_missing_answer and not answer.strip():
        return None, "missing benchmark reference"
    answers = [str(x) for x in answer_value] if isinstance(answer_value, (list, tuple)) else ([answer] if answer else [])
    fields = config.get("benchmark_image_fields")
    if fields:
        values = [example[f] for f in fields if example.get(f) is not None]
    else:
        _, images = _first_field(example, config["image_fields"])
        values = list(images) if isinstance(images, (list, tuple)) else ([images] if images is not None else [])
    expected = example.get("num_images")
    if expected is not None and (type(expected) is not int or expected != len(values)):
        return None, "image count differs from declared benchmark input"
    # A text-only MMSearch query is valid for the full benchmark runner.
    if not values and source_name != "mmsearch" and not config.get("allow_text_only"):
        return None, "benchmark images missing"
    split = used_split or config["split"]
    cache = fingerprint({"config": config.get("config_name"), "split": split,
                         "revision": config.get("revision")})[:12]
    image_inputs = []
    for j, value in enumerate(values):
        path = image_cache_dir / source_name / ("tasks_" + cache) / f"{index:06d}_{j + 1}.png"
        saved = save_image_from_field(value, path, image_base_dir)
        if not saved:
            return None, f"failed to save benchmark image {j + 1}"
        image_inputs.append({"index": j + 1, "path": saved, "sha256": file_sha(saved)})
    _, raw_id = _first_field(example, config["id_fields"])
    sample_id = f"{source_name}_{re.sub(r'[^A-Za-z0-9_-]', '_', split)}_{raw_id if raw_id is not None else index}"
    provenance = {"hf_id": config["hf_id"], "config": config.get("config_name"),
                  "split": split, "revision": config.get("revision")}
    # Keep the original image placeholders/order, especially multi-image prompts.
    model_input = {"question": question.strip(), "images": image_inputs}
    identity = task_fingerprint({"model_input": model_input, "dataset_provenance": provenance})
    reference = {"id": sample_id, "input_fingerprint": identity,
                 "answer": answer, "answer_candidates": answers,
                 "answer_format": "set" if config.get("answer_selection") == "all" else "alternatives",
                 "annotations": {k: example[k] for k in config.get("reference_fields", []) if k in example}}
    if reference["answer_format"] == "set":
        reference.update(required_answers=list(answer_value), answer_evals=example.get("answer_evals"))
        reference["answer_candidates"] = [answer]
    return {"schema": TASK_SCHEMA, "id": sample_id, "source_dataset": source_name,
            "data_usage": "evaluation", "model_input": model_input,
            "input_fingerprint": identity, "dataset_provenance": provenance,
            "environment": config["environment"], "trajectory_executed": False,
            "categories": {k: example[k] for k in ("domain", "subdomain", "category", "difficulty", "subtask",
                                                     "question_hop", "entity_num", "question_type")
                           if isinstance(example.get(k), (str, int))},
            "reference": reference}, None


def separate_references(tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Only the evaluator should load the second output file."""
    return ([{k: v for k, v in task.items() if k != "reference"} for task in tasks],
            [task["reference"] for task in tasks])
