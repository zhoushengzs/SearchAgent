'MMSearch-Plus loader.'
from __future__ import annotations

import base64
import binascii
import hashlib
import itertools
from pathlib import Path
from typing import Any

from core.utils import get_logger
from sources.huggingface import load_hf_dataset, normalize_hf_sample
from sources.registry import DATASET_REGISTRY

logger = get_logger(__name__)

Sample = dict[str, Any]

_SOURCE = "mmsearch_plus"


def _decrypt(value: str, canary: str) -> str:
    """Inverse of the upstream encryption; returns ``value`` unchanged if it
    is not ciphertext (e.g. a future plaintext release)."""
    if not value:
        return value
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return value
    digest = hashlib.sha256(canary.encode()).digest()
    key = digest * (len(data) // len(digest) + 1)
    try:
        return bytes(a ^ b for a, b in zip(data, key)).decode("utf-8")
    except UnicodeDecodeError:
        return value


def load_mmsearch_plus(
    image_cache_dir: Path,
    skip_missing_answer: bool = True,
    max_samples: int | None = None,
    split_override: str | None = None,
    config_override: str | None = None,
    streaming: bool = False,
    task_mode: bool = False,
) -> tuple[list[Sample], dict[str, int]]:
    config = {**DATASET_REGISTRY[_SOURCE]}
    if config_override is not None:
        config["config_name"] = config_override
    canary = config["canary"]
    dataset, split = load_hf_dataset(config, split_override=split_override, streaming=streaming)
    if task_mode:
        from sources.benchmarks import normalize_benchmark_task
        normalize = normalize_benchmark_task
    else:
        normalize = normalize_hf_sample

    samples: list[Sample] = []
    skipped = multi = raw_count = 0
    iterable = iter(dataset)
    if max_samples is not None:
        iterable = itertools.islice(iterable, max_samples)
    for i, row in enumerate(iterable):
        raw_count += 1
        example = dict(row)
        images = [example[f] for f in config["benchmark_image_fields"] if example.get(f) is not None]
        if example.get("num_images") != len(images) or not images:
            skipped += 1
            continue
        if len(images) > 1 and not task_mode:
            multi += 1
            continue
        example["question"] = _decrypt(example.get("question") or "", canary)
        answers = [_decrypt(a, canary) for a in (example.get("answer") or []) if a]
        example["answer"] = answers
        example["id"] = f"{i:04d}"
        sample, err = normalize(
            example, config, i, image_cache_dir, _SOURCE,
            skip_missing_answer=skip_missing_answer,
            used_split=split,
        )
        if sample is None:
            skipped += 1
            logger.debug("%s[%d] skipped: %s", _SOURCE, i, err)
            continue
        samples.append(sample)

    logger.info("Source '%s': raw=%d valid=%d skipped=%d (multi-image=%d)",
                _SOURCE, raw_count, len(samples), skipped + multi, multi)
    return samples, {"raw": raw_count, "valid": len(samples),
                     "skipped": skipped + multi}
