"""Content identities shared by generation, verification and export."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PIPELINE_VERSION = "causal-policy-v2"
VERIFICATION_VERSION = "visual-language-v3"


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_family(model: str) -> str | None:
    """Conservative family recognition; changing providers is not independence."""
    name = model.lower().split("/")[-1]
    for family, prefixes in {
        "qwen": ("qwen",), "gemini": ("gemini",), "kimi": ("kimi",),
        "deepseek": ("deepseek",), "mimo": ("mimo",),
        "claude": ("claude",), "openai": ("gpt-", "o1", "o3", "o4"),
        "llama": ("llama", "meta-llama"), "pixtral": ("pixtral",),
        "glm": ("glm",), "internvl": ("internvl",),
    }.items():
        if name.startswith(prefixes):
            return family
    return None


def independent_models(generator: str, judge: str) -> bool:
    a, b = model_family(generator), model_family(judge)
    return a is not None and b is not None and a != b


def record_fingerprint(record: dict[str, Any]) -> str:
    """Bind a verdict to its exact annotation, input and execution ledger."""
    return fingerprint({k: record.get(k) for k in (
        "id", "source_dataset", "question", "answer", "image_path",
        "image_size", "image_sha256", "trace_variant", "model", "provider",
        "prompt_sha", "generation_fingerprint", "annotation", "execution",
        "needs_human_review", "truncated",
        "data_usage", "dataset_provenance", "answer_candidates",
        "trajectory_derivation", "leakage_group",
        "reference_answer",
    )})
