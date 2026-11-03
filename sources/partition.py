'Explicit benchmark-derived seed splits, made before generation/filtering.'
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from core.provenance import file_sha, fingerprint
from sources.registry import is_evaluation_source

PARTITION_VERSION = "benchmark-derived-v1"


def _descriptor(sample: dict, image_root: Path | None = None, *, check_files: bool = True) -> dict:
    task = sample.get("schema") == "visual-agent-task-v1"
    question = sample["model_input"]["question"] if task else sample["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A seed needs its original question")
    images = sample["model_input"]["images"] if task else [
        {"path": sample["image_path"], "sha256": sample.get("image_sha256")}]
    hashes = []
    for image in images:
        expected = image.get("sha256")
        if check_files:
            path = Path(image["path"])
            if image_root is not None and not path.is_absolute():
                path = image_root / path
            digest = file_sha(path)
            if expected is not None and expected != digest:
                raise ValueError("Initial image hash changed")
        else:
            digest = expected
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Missing valid initial image hash")
        hashes.append(digest)
    provenance = sample.get("dataset_provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("revision"), str) or not provenance["revision"]:
        raise ValueError("Pin dataset_provenance.revision before partitioning")
    return {"source": sample["source_dataset"], "dataset_provenance": provenance,
            "question_sha256": fingerprint(question), "images": hashes,
            "leakage_group": sample.get("leakage_group")}


def _tokens(descriptor: dict) -> set[str]:
    tokens = {"image:" + sha for sha in descriptor["images"]}
    if not tokens:
        # Text-only benchmark tasks have no visual byte identity.
        tokens.add("question:" + descriptor["question_sha256"])
    if descriptor.get("leakage_group") is not None:
        tokens.add("group:" + fingerprint(descriptor["leakage_group"]))
    return tokens


def _role(group_key: str, seed: int, fraction: float) -> str:
    value = int(fingerprint({"group": group_key, "seed": seed})[:16], 16) / 2**64
    return "holdout" if value < fraction else "train"


def partition_benchmark_seeds(samples: list[dict], *, split_name: str, seed: int = 42,
                              holdout_fraction: float = 0.2, image_root: Path | None = None) -> list[dict]:
    """Return copies; never turn an official benchmark split into an official train split."""
    if not isinstance(split_name, str) or not split_name.strip() or type(seed) is not int or \
            not math.isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
        raise ValueError("Use a split name, integer seed and 0 < holdout_fraction < 1")
    if not samples:
        raise ValueError("Cannot partition an empty pool")
    ids = [str(sample["id"]) for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Original seed inputs must have globally unique ids")
    descriptors = []
    for sample in samples:
        if any(sample.get(k) is not None for k in
               ("trajectory_derivation", "execution", "annotation", "generation_fingerprint")):
            raise ValueError("Partition original inputs before generation; do not reassign derived records")
        if sample.get("schema") == "visual-agent-task-v1" and "reference" in sample:
            raise ValueError("Detach benchmark references before partitioning")
        if not is_evaluation_source(sample.get("source_dataset", ""), sample.get("dataset_provenance")):
            raise ValueError("Only recognized benchmark sources use benchmark-derived partitioning")
        descriptors.append(_descriptor(sample, image_root))
    # Union-find links all tasks sharing at least one image, transitively.
    parent = list(range(len(samples)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    token_sets = [_tokens(d) for d in descriptors]
    for i, tokens in enumerate(token_sets):
        for token in tokens:
            if token in owners:
                parent[find(i)] = find(owners[token])
            else:
                owners[token] = i
    components: dict[int, set[str]] = {}
    for i, tokens in enumerate(token_sets):
        components.setdefault(find(i), set()).update(tokens)
    corpus = fingerprint(sorted(fingerprint(d) for d in descriptors))
    output = []
    for i, (sample, descriptor) in enumerate(zip(samples, descriptors)):
        tokens = sorted(components[find(i)])
        group = fingerprint(tokens)
        role = _role(group, seed, holdout_fraction)
        derivation = {"version": PARTITION_VERSION, "purpose": "benchmark_derived",
                      "split_name": split_name.strip(), "seed": seed, "holdout_fraction": holdout_fraction,
                      "partition": role, "corpus_fingerprint": corpus, "group_key": group,
                      "group_tokens": tokens, "input_fingerprint": fingerprint(descriptor)}
        row = copy.deepcopy(sample)
        row.update(trajectory_derivation=derivation, data_usage="train" if role == "train" else "evaluation")
        if row.get("schema") != "visual-agent-task-v1":
            row["image_sha256"] = descriptor["images"][0]
        output.append(row)
    return output


def derived_partition(sample: dict[str, Any], *, check_files: bool = True) -> str | None:
    """Validate input binding and deterministic assignment; None means invalid/absent."""
    try:
        d = sample["trajectory_derivation"]
        if not isinstance(d, dict) or d.get("version") != PARTITION_VERSION or d.get("purpose") != "benchmark_derived":
            return None
        if not is_evaluation_source(sample.get("source_dataset", ""), sample.get("dataset_provenance")):
            return None
        fraction, seed, tokens = d["holdout_fraction"], d["seed"], d["group_tokens"]
        if type(seed) is not int or type(fraction) not in (float, int) or not math.isfinite(fraction) or not 0 < fraction < 1:
            return None
        if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens) or tokens != sorted(set(tokens)):
            return None
        descriptor = _descriptor(sample, check_files=check_files)
        if not _tokens(descriptor) <= set(tokens) or fingerprint(tokens) != d["group_key"] or \
                fingerprint(descriptor) != d["input_fingerprint"]:
            return None
        if not isinstance(d.get("split_name"), str) or not d["split_name"].strip() or \
                not isinstance(d.get("corpus_fingerprint"), str) or len(d["corpus_fingerprint"]) != 64:
            return None
        role = _role(d["group_key"], seed, fraction)
        expected_usage = "train" if role == "train" else "evaluation"
        return role if d["partition"] == role and sample.get("data_usage") == expected_usage else None
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return None
