'Load and normalize local JSON / JSONL samples into the standard schema.'
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.utils import get_logger, json_safe, make_stable_id
from sources.partition import derived_partition
from sources.registry import is_evaluation_source

logger = get_logger(__name__)

Sample = dict[str, Any]

# Field-name aliases (checked in order).
IMAGE_FIELDS = ["image_path", "image", "img_path", "image_file", "img"]
QUESTION_FIELDS = ["question", "query", "prompt", "instruction", "text"]
ANSWER_FIELDS = [
    "answer",
    "label",
    "gt_answer",
    "ground_truth",
    "target",
    "final_answer",
]
# "existing_reasoning" first: it is the key Stage 1 writes (MMSearch stores
# its reference query there), so a normalized pool must round-trip it.
REASONING_FIELDS = ["existing_reasoning", "reasoning", "rationale", "explanation", "cot"]
ID_FIELDS = ["id", "sample_id", "question_id", "uid"]


def _first_present(example: dict[str, Any], fields: list[str]) -> Any:
    """Return the first non-empty value among ``fields`` in ``example``."""
    for f in fields:
        if f in example and example[f] not in (None, "", [], {}):
            return example[f]
    return None


def _coerce_answer(value: Any) -> str:
    """Turn a possibly-list answer into a single string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            if item not in (None, ""):
                return str(item)
        return ""
    return str(value)


def _coerce_text(value: Any) -> str:
    """Best-effort text extraction (handles chat-style message lists)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Single message dict.
        for key in ("content", "text", "value"):
            if key in value:
                return _coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        # Chat-style: extract the last user turn if possible.
        user_turns = [
            m for m in value
            if isinstance(m, dict) and str(m.get("role", "")).lower() == "user"
        ]
        if user_turns:
            return _coerce_text(user_turns[-1])
        return " ".join(_coerce_text(v) for v in value).strip()
    return str(value)


def normalize_sample(
    example: dict[str, Any],
    index: int,
    source_dataset: str = "local",
    skip_missing_answer: bool = True,
) -> tuple[Sample | None, str | None]:
    'Normalize one local example into the standard schema.'
    if example.get("schema") == "visual-agent-task-v1":
        raise ValueError("Use agent_pipeline.py for benchmark task manifests; main.py only executes the single-image visual module")
    image_value = _first_present(example, IMAGE_FIELDS)
    if not image_value:
        return None, "missing image field"
    image_path = str(image_value)

    question = _coerce_text(_first_present(example, QUESTION_FIELDS))
    if not question:
        return None, "missing question field"

    answer = _coerce_answer(_first_present(example, ANSWER_FIELDS))
    needs_review = bool(example.get("needs_human_review", False))
    if not answer:
        if skip_missing_answer:
            return None, "missing answer field"
        needs_review = True

    reasoning = _coerce_text(_first_present(example, REASONING_FIELDS)) or ""

    raw_id = _first_present(example, ID_FIELDS)
    sample_id = (
        str(raw_id)
        if raw_id is not None
        else make_stable_id(source_dataset, index, question, answer, image_path)
    )

    source = str(example.get("source_dataset") or source_dataset)

    sample: Sample = {
        "id": sample_id,
        "image_path": image_path,
        "question": question,
        "answer": answer,
        "source_dataset": source,
        "existing_reasoning": reasoning,
        "needs_human_review": needs_review,
        "answer_candidates": example.get("answer_candidates", [answer]),
        "reference_answer": example.get("reference_answer"),
        "data_usage": ("train" if derived_partition(example, check_files=False) == "train" else "evaluation")
                      if is_evaluation_source(source, example.get("dataset_provenance"))
                      else example.get("data_usage", "train"),
        "dataset_provenance": example.get("dataset_provenance", {}),
        "trajectory_derivation": example.get("trajectory_derivation"),
        "leakage_group": example.get("leakage_group"),
        "image_sha256": example.get("image_sha256"),
        "raw": json_safe(example),
    }
    return sample, None


def load_local_file(path: str | Path) -> list[dict[str, Any]]:
    """Load a local .json (list or single object) or .jsonl file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # JSONL vs JSON auto-detection.
    if p.suffix.lower() == ".jsonl":
        return _parse_jsonl_text(text, p)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to line-delimited parsing.
        return _parse_jsonl_text(text, p)

    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        # Could be {"data": [...]} or a single sample.
        for key in ("data", "samples", "examples"):
            if isinstance(data.get(key), list):
                return [d for d in data[key] if isinstance(d, dict)]
        return [data]
    raise ValueError(f"Unsupported JSON structure in {p}")


def _parse_jsonl_text(text: str, path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)
    return records


def load_and_normalize_local(
    path: str | Path,
    source_dataset: str = "local",
    skip_missing_answer: bool = True,
) -> tuple[list[Sample], list[dict[str, Any]]]:
    'Load a local file and normalize every record.'
    raw_records = load_local_file(path)
    samples: list[Sample] = []
    errors: list[dict[str, Any]] = []
    for i, rec in enumerate(raw_records):
        sample, err = normalize_sample(
            rec, i, source_dataset=source_dataset,
            skip_missing_answer=skip_missing_answer,
        )
        if sample is not None:
            samples.append(sample)
        else:
            errors.append({"index": i, "error": err})
    logger.info(
        "Loaded %d record(s) from %s -> %d valid, %d skipped.",
        len(raw_records), path, len(samples), len(errors),
    )
    return samples, errors
