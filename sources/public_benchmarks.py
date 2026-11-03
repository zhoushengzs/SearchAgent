'Pinned author-release adapters for retrieval and image-reasoning benchmarks.'
from __future__ import annotations

import base64
import hashlib
import itertools
import json
import zipfile
from pathlib import Path, PurePosixPath

from core.provenance import file_sha
from core.utils import get_logger
from sources.benchmarks import normalize_benchmark_task
from sources.download import RangeFile, cached_download
from sources.huggingface import normalize_hf_sample
from sources.registry import DATASET_REGISTRY

logger = get_logger(__name__)
V3_PASSPHRASE = "A_Visual_Vertical_Verifiable_Benchmark_for_Multimodal_Browsing_Agents"
M3_ARCHIVE_SIZE = 1862320666
TIR_ARCHIVE_SIZE = 1244816332


def decode_xor(value: str, password: str) -> str:
    data = base64.b64decode(value, validate=True)
    key = hashlib.sha256(password.encode("utf-8")).digest()
    decoded = bytes(byte ^ key[i % len(key)] for i, byte in enumerate(data)).decode("utf-8")
    if not decoded.strip() or any(ord(c) < 32 and c not in "\n\r\t" for c in decoded):
        raise ValueError("Invalid decoded benchmark text")
    return decoded


def decode_v3(value: dict | str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    value = json.loads(value) if isinstance(value, str) else value
    key = hashlib.sha256(V3_PASSPHRASE.encode("utf-8")).digest()
    fields = [base64.b64decode(value[name], validate=True) for name in ("iv", "ciphertext", "tag")]
    return AESGCM(key).decrypt(fields[0], fields[1] + fields[2], None).decode("utf-8")


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Unsafe dataset image path")
    return str(path)


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        if stream.read(1) == "[":
            stream.seek(0)
            yield from json.load(stream)
            return
        stream.seek(0)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Benchmark row must be an object")
                yield row


def _metadata(source, config, cache, root):
    if root is not None:
        folder = Path(root) / source
        for name in ("metadata.decrypted.jsonl", "metadata.jsonl"):
            path = folder / name
            if path.is_file():
                return path, folder, name == "metadata.decrypted.jsonl"
        raise FileNotFoundError(f"Missing {folder}/metadata.jsonl")
    revision = config["revision"]
    if source == "mmbrowsecomp":
        url = f"https://raw.githubusercontent.com/MMBrowseComp/MM-BrowseComp/{revision}/data/MMBrowseComp_400.jsonl"
    elif source == "dynvqa":
        url = f"https://raw.githubusercontent.com/Alibaba-NLP/OmniSearch/{revision}/dataset/DynVQA_en/DynVQA_en.202502.jsonl"
    else:
        filename = {"browsecomp_v3": "data/train.jsonl", "m3vqa": "quesions.jsonl", "tirbench": "TIR-Bench.json"}[source]
        url = f"https://huggingface.co/datasets/{config['hf_id']}/resolve/{revision}/{filename}"
    return cached_download(url, cache / "downloads"), None, False


def _local_image(folder: Path, value: str) -> Path:
    relative = _relative(value)
    candidates = [folder / relative, folder / "images" / relative,
                  folder / "images" / PurePosixPath(relative).name]
    if relative.startswith("data/"):
        candidates.append(folder / relative[5:])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing local initial image: {relative}")


def _prepare_row(source: str, row: dict, *, decoded: bool = False) -> dict:
    example = dict(row)
    if source == "mmbrowsecomp":
        if not decoded:
            for name in ("question", "answer"):
                example[name] = decode_xor(row[name], row["canary"])
            example["checklist"] = [decode_xor(value, row["canary"]) for value in row.get("checklist", [])]
        if not isinstance(example.get("images"), list):
            raise ValueError("MM-BrowseComp images must be a list")
    elif source == "browsecomp_v3":
        if not decoded:
            encrypted = row.get("encrypted_data")
            example["question"] = decode_v3(encrypted["question"] if encrypted else row["encrypted_question"])
            example["answer"] = decode_v3(encrypted["answer"] if encrypted else row["encrypted_answer"])
        metadata = _json(row.get("metadata", {}))
        paths = _json(row.get("image_paths", []))
        if not paths:
            paths = metadata.get("images") or ([row["image"]] if row.get("image") else [])
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            raise ValueError("Invalid BrowseComp-V3 image_paths")
        example["images"] = paths
        if metadata.get("vis_inputs") is not None and metadata["vis_inputs"] != len(paths):
            raise ValueError("BrowseComp-V3 image count mismatch")
        example["metadata"] = metadata
        example["sub_goals"] = _json(row.get("sub_goals", []))
        # Difficulty is a stratum, never a planner hint. Gold metadata stays detached.
        example["difficulty"] = metadata.get("difficulty")
    elif source == "dynvqa":
        example["images"] = [row["image_url"]]
    elif source == "m3vqa":
        answers, alternatives = row["answers"], row["answer_evals"]
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError("M3-VQA requires all nonempty answers")
        if not isinstance(alternatives, list) or len(alternatives) != len(answers) or any(
                not isinstance(group, list) or not group or any(not isinstance(a, str) for a in group)
                for group in alternatives):
            raise ValueError("M3-VQA answer alternatives must be grouped per required answer")
        example["images"] = [row["image_id"]]
    elif source == "tirbench":
        example["question"] = row["prompt"]
        example["images"] = [row[key] for key in ("image_1", "image_2") if row.get(key)]
        example["category"] = row["task"]
    else:
        raise ValueError("Unknown public benchmark")
    return example


def _load(source: str, image_cache_dir: Path, skip_missing_answer=True, max_samples=None,
          split_override=None, config_override=None, streaming=False, task_mode=False,
          source_data_root: Path | None = None):
    config = dict(DATASET_REGISTRY[source])
    if split_override is not None and split_override != config["split"]:
        raise ValueError(f"{source} adapter supports exact pinned split {config['split']!r}")
    if config_override not in (None, "default"):
        raise ValueError(f"{source} has no config {config_override!r}")
    cache = Path(image_cache_dir) / source
    metadata, folder, decoded = _metadata(source, config, cache, source_data_root)
    source_hash = file_sha(metadata)
    rows = _rows(metadata)
    if max_samples is not None:
        rows = itertools.islice(rows, max_samples)
    archive = remote = None
    archive_names = {}
    archive_failed = False
    samples = []
    raw = skipped = 0
    try:
        for i, row in enumerate(rows):
            raw += 1
            try:
                example = _prepare_row(source, row, decoded=decoded)
                if not task_mode and len(example["images"]) != 1:
                    raise ValueError("Visual-module inputs require exactly one initial image")
                values = []
                for value in example["images"]:
                    if source in {"m3vqa", "tirbench"}:
                        try:
                            if folder is None:
                                raise FileNotFoundError
                            values.append(str(_local_image(folder, value)))
                            continue
                        except FileNotFoundError:
                            if archive is None:
                                if archive_failed:
                                    raise ValueError("Source archive unavailable; provide local assets")
                                archive_name = "images.zip" if source == "m3vqa" else "data.zip"
                                try:
                                    if folder is not None:
                                        archive = zipfile.ZipFile(folder / archive_name)
                                    else:
                                        url = f"https://huggingface.co/datasets/{config['hf_id']}/resolve/{config['revision']}/{archive_name}"
                                        size = M3_ARCHIVE_SIZE if source == "m3vqa" else TIR_ARCHIVE_SIZE
                                        remote = RangeFile(url, size, cache / "ranges")
                                        archive = zipfile.ZipFile(remote)
                                    for name in archive.namelist():
                                        key = PurePosixPath(name).name if source == "m3vqa" else name
                                        archive_names.setdefault(key, []).append(name)
                                except Exception:
                                    archive_failed = True
                                    raise
                            relative = _relative(value)
                            names = archive_names.get(relative, [])
                            if len(names) != 1:
                                raise ValueError("Image archive member missing or ambiguous")
                            info = archive.getinfo(names[0])
                            if info.file_size > 32 * 1024 * 1024:
                                raise ValueError("Image archive member too large")
                            values.append(archive.read(info))
                    elif folder is not None:
                        # Locally materialized assets can use basename of author URLs.
                        name = value.rsplit("/", 1)[-1] if value.startswith(("http://", "https://")) else value
                        values.append(str(_local_image(folder, name)))
                    else:
                        if source == "browsecomp_v3":
                            value = f"https://huggingface.co/datasets/{config['hf_id']}/resolve/{config['revision']}/{_relative(value)}"
                        elif source == "mmbrowsecomp":
                            prefix = "https://raw.githubusercontent.com/MMBrowseComp/MM-BrowseComp/"
                            if value.startswith(prefix):
                                tail = value[len(prefix):]
                                tail = tail.removeprefix("refs/heads/main/").removeprefix("main/")
                                value = prefix + config["revision"] + "/" + _relative(tail)
                        values.append(str(cached_download(value, cache / "assets", max_bytes=32 * 1024 * 1024)))
                example["images"] = values
                normalizer = normalize_benchmark_task if task_mode else normalize_hf_sample
                sample, error = normalizer(example, config, i, Path(image_cache_dir), source,
                    skip_missing_answer=skip_missing_answer, used_split=config["split"])
                if sample is None:
                    raise ValueError(error)
                sample["dataset_provenance"].update(metadata_sha256=source_hash,
                    transport="local_author_assets" if folder else "pinned_author_release")
                if source == "dynvqa":
                    sample["dataset_provenance"]["repository"] = "Alibaba-NLP/OmniSearch"
                    sample["dataset_provenance"]["data_file"] = "dataset/DynVQA_en/DynVQA_en.202502.jsonl"
                if source == "mmbrowsecomp":
                    sample["dataset_provenance"].update(hf_id=None, repository="MMBrowseComp/MM-BrowseComp",
                                                       data_file="data/MMBrowseComp_400.jsonl")
                if task_mode:
                    # Metadata provenance is part of the input/reference binding.
                    from sources.benchmarks import task_fingerprint
                    identity = task_fingerprint(sample)
                    sample["input_fingerprint"] = sample["reference"]["input_fingerprint"] = identity
                samples.append(sample)
            except Exception as exc:
                skipped += 1
                logger.warning("%s row %d skipped: %s", source, i, type(exc).__name__)
                logger.debug("%s row %d detail: %s", source, i, exc)
                if archive_failed:
                    break  # Do not retry a failed gigabyte archive for every row.
    finally:
        if archive is not None:
            archive.close()
        if remote is not None:
            remote.close()
    logger.info("%s: raw=%d valid=%d skipped=%d", source, raw, len(samples), skipped)
    return samples, {"raw": raw, "valid": len(samples), "skipped": skipped}


def load_mmbrowsecomp(**kwargs):
    return _load("mmbrowsecomp", **kwargs)


def load_browsecomp_v3(**kwargs):
    return _load("browsecomp_v3", **kwargs)


def load_dynvqa(**kwargs):
    return _load("dynvqa", **kwargs)


def load_m3vqa(**kwargs):
    return _load("m3vqa", **kwargs)


def load_tirbench(**kwargs):
    return _load("tirbench", **kwargs)
