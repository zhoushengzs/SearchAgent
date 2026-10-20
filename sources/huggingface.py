'HuggingFace dataset loading + normalization into the standard sample schema.'
from __future__ import annotations

import base64
import binascii
import io
import itertools
import json
import os
from collections import Counter
import re
from pathlib import Path
from typing import Any

from sources.registry import dataset_usage
from core.provenance import fingerprint
from core.utils import get_logger, json_safe, make_stable_id

logger = get_logger(__name__)

Sample = dict[str, Any]


# --------------------------------------------------------------------------- #
# Loading (new-`datasets`-compatible)
# --------------------------------------------------------------------------- #
def load_hf_dataset(config: dict[str, Any], split_override: str | None = None,
                    config_override: str | None = None, streaming: bool = False):
    """Load exactly the requested split; never silently substitute train/test."""
    from datasets import get_dataset_split_names, load_dataset

    name = config_override if config_override is not None else config.get("config_name")
    split = split_override or config.get("split", "train")
    kwargs = {"revision": config["revision"]} if config.get("revision") else {}
    available = get_dataset_split_names(config["hf_id"], config_name=name, **kwargs)
    if split.split("[", 1)[0] not in available:
        raise ValueError(f"Requested split {split!r} unavailable for {config['hf_id']}; available: {available}")
    ds = load_dataset(config["hf_id"], name, split=split, streaming=streaming, **kwargs)
    logger.info("Loaded %s config=%s split=%s streaming=%s", config["hf_id"], name, split, streaming)
    return ds, split


def print_dataset_debug_info(dataset, dataset_name: str, n_preview: int = 1) -> None:
    """Print features and the first sample's field names + types (dry-run)."""
    print(f"\n{'=' * 70}")
    print(f"[DRY-RUN] Source: {dataset_name}")
    print(f"{'=' * 70}")
    try:
        print(f"Num rows: {len(dataset)}")
    except Exception:  # noqa: BLE001
        print("Num rows: <unknown>")

    features = getattr(dataset, "features", None)
    if features is not None:
        print("Features:")
        for key, feat in features.items():
            print(f"  - {key}: {feat}")

    for i, example in enumerate(itertools.islice(iter(dataset), n_preview)):
        print(f"\nSample #{i} field names and types:")
        for key, value in example.items():
            type_name = type(value).__name__
            preview = ""
            if isinstance(value, str):
                preview = f" -> {value[:80]!r}"
            elif isinstance(value, (int, float, bool)):
                preview = f" -> {value!r}"
            elif isinstance(value, list):
                preview = f" -> list[{len(value)}]"
            print(f"  - {key}: {type_name}{preview}")


# --------------------------------------------------------------------------- #
# Field extraction
# --------------------------------------------------------------------------- #
def _first_field(example: dict[str, Any], fields: list[str]) -> tuple[str | None, Any]:
    for f in fields:
        if f in example and example[f] not in (None, "", [], {}):
            return f, example[f]
    return None, None


_IMAGE_PLACEHOLDER_RE = re.compile(r"<image[^>]*>", re.IGNORECASE)


def _extract_question(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _IMAGE_PLACEHOLDER_RE.sub("", value).strip()
    if isinstance(value, dict):
        for key in ("content", "text", "value", "question", "problem"):
            if key in value:
                return _extract_question(value[key])
        return str(value)
    if isinstance(value, (list, tuple)):
        user_turns = [
            m for m in value
            if isinstance(m, dict) and str(m.get("role", "")).lower() == "user"
        ]
        if user_turns:
            return _extract_question(user_turns[-1])
        texts = [_extract_question(v) for v in value]
        return " ".join(t for t in texts if t).strip()
    return str(value)


def _extract_answer(value: Any, selection: str = "first") -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        answers = [str(item).strip() for item in value if item is not None and str(item).strip()]
        if not answers:
            return ""
        if selection == "all":
            return json.dumps(answers, ensure_ascii=False)
        return Counter(answers).most_common(1)[0][0] if selection == "majority" else answers[0]
    return str(value)


def expand_hf_example(example: dict[str, Any], config: dict[str, Any], index: int):
    """Flatten independent QA pairs sharing ONE image; no dialogue-answer context."""
    field = config.get("qa_field")
    if not field:
        yield example
        return
    pairs = example.get(field)
    if not isinstance(pairs, list):
        raise ValueError(f"Expected QA array {field!r}")
    for j, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError("Malformed QA pair")
        yield {**example, "question": pair.get(config["qa_question_field"]),
               "answer": pair.get(config["qa_answer_field"]), "question_id": f"{index}_{j}"}


# --------------------------------------------------------------------------- #
# Image saving
# --------------------------------------------------------------------------- #
def _is_pil_image(value: Any) -> bool:
    from PIL import Image
    return isinstance(value, Image.Image)


def _looks_base64_image(value: str) -> bool:
    """Heuristic: a long base64 blob that decodes to image bytes."""
    stripped = value.strip()
    if len(stripped) < 100:
        return False
    # base64 charset only (+ url-safe), allow trailing '='.
    if not re.fullmatch(r"[A-Za-z0-9+/=_\-]+", stripped):
        return False
    try:
        raw = base64.b64decode(stripped[:64] + "=" * (-len(stripped[:64]) % 4))
    except (binascii.Error, ValueError):
        return False
    # JPEG/PNG/GIF magic bytes.
    return raw[:2] == b"\xff\xd8" or raw[:8].startswith(b"\x89PNG") or raw[:3] == b"GIF"


def save_image_from_field(
    value: Any, output_path: Path, base_dir: Path | None = None
) -> str | None:
    """Persist an image value to ``output_path``; return the saved path or None."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # list -> unwrap the single element (multi-image samples are rejected
    # upstream in normalize_hf_sample, where the skip reason is recorded).
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
        if value is None:
            return None

    if _is_pil_image(value):
        return _save_pil(value, output_path)

    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        raw_path = value.get("path")
        if raw_bytes:
            return _save_bytes_as_image(raw_bytes, output_path)
        if raw_path:
            resolved = _resolve_path(raw_path, base_dir)
            if resolved:
                return _reencode(resolved, output_path)
        return None

    if isinstance(value, (bytes, bytearray)):
        return _save_bytes_as_image(bytes(value), output_path)

    if isinstance(value, str):
        # 1) local / base_dir-relative path.
        resolved = _resolve_path(value, base_dir)
        if resolved:
            return _reencode(resolved, output_path)
        # 2) base64-encoded image bytes.
        if _looks_base64_image(value):
            try:
                raw = base64.b64decode(value)
                return _save_bytes_as_image(raw, output_path)
            except (binascii.Error, ValueError):
                return None
        return None

    return None


def _safe_exists(value: str | Path) -> bool:
    'Path.exists() that never raises.'
    try:
        return Path(value).exists()
    except (OSError, ValueError):
        return False


def _resolve_path(value: str, base_dir: Path | None) -> Path | None:
    if _safe_exists(value):
        return Path(value)
    if base_dir is None:
        return None
    base_dir = Path(base_dir)

    # 1) Direct join: base_dir / value.
    cand = base_dir / value
    if _safe_exists(cand):
        return cand

    # 2) Some datasets prefix the path with the repo name (e.g.
    #    "<repo>/data/x.jpg") while the snapshot only contains
    #    "data/x.jpg". Retry after stripping leading path segments.
    parts = Path(value).parts
    for start in range(1, len(parts)):
        cand = base_dir.joinpath(*parts[start:])
        if _safe_exists(cand):
            return cand

    # 3) Last resort: match by basename anywhere under the snapshot.
    name = Path(value).name
    if name:
        try:
            match = next(base_dir.rglob(name), None)
        except OSError:
            match = None
        if match is not None:
            return match
    return None


def _save_pil(img: Any, output_path: Path) -> str | None:
    try:
        if img.mode not in ("RGB", "L", "RGBA"):
            img = img.convert("RGB")
        img.save(output_path)
        return str(output_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save PIL image to %s: %s", output_path, exc)
        return None


def _reencode(src: Path, output_path: Path) -> str | None:
    """Open an existing image file and save it losslessly in the PNG cache."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(src) as img:
            return _save_pil(img, output_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Invalid source image %s: %s", src, exc)
        return None


def _save_bytes_as_image(raw: bytes, output_path: Path) -> str | None:
    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(raw)) as img:
            return _save_pil(img, output_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Invalid image bytes for %s: %s", output_path, exc)
        return None


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize_hf_sample(
    example: dict[str, Any],
    dataset_config: dict[str, Any],
    index: int,
    image_cache_dir: Path,
    source_name: str,
    skip_missing_answer: bool = True,
    image_base_dir: Path | None = None,
    used_split: str | None = None,
) -> tuple[Sample | None, str | None]:
    """Normalize one HF example into the standard schema."""
    img_field, img_value = _first_field(example, dataset_config["image_fields"])
    if img_value is None:
        return None, "missing image field"
    # Labeling a multi-image question from its first image alone would produce
    # answers unsupported by the attached evidence — skip, with the real reason.
    if isinstance(img_value, (list, tuple)) and len(img_value) > 1:
        return None, f"multi-image sample ({len(img_value)} images)"

    split = used_split or dataset_config.get("split", "train")
    cache_split = re.sub(r"[^A-Za-z0-9_-]", "_", split)
    cache_identity = fingerprint({"split": split, "config": dataset_config.get("config_name"),
                                  "revision": dataset_config.get("revision")})[:12]
    image_out = image_cache_dir / source_name / (cache_split + "_" + cache_identity) / f"{source_name}_{index:06d}.png"
    image_path = save_image_from_field(img_value, image_out, base_dir=image_base_dir)
    if not image_path:
        return None, f"failed to save image (field={img_field})"

    _, q_value = _first_field(example, dataset_config["question_fields"])
    question = _extract_question(q_value)
    if not question.strip():
        return None, "missing question field"

    _, a_value = _first_field(example, dataset_config["answer_fields"])
    answer = _extract_answer(a_value, dataset_config.get("answer_selection", "first"))
    needs_review = False
    if not answer.strip():
        if skip_missing_answer:
            return None, "missing answer field"
        needs_review = True

    _, r_value = _first_field(example, dataset_config["reasoning_fields"])
    existing_reasoning = _extract_question(r_value) if r_value is not None else ""

    _, id_value = _first_field(example, dataset_config["id_fields"])
    sample_id = (
        f"{source_name}_{cache_split}_{id_value}"
        if id_value is not None
        else make_stable_id(source_name, index, question, answer, image_path)
    )

    sample: Sample = {
        "id": sample_id,
        "image_path": image_path,
        "question": question,
        "answer": answer,
        "source_dataset": source_name,
        "existing_reasoning": existing_reasoning,
        "needs_human_review": needs_review,
        "domain": dataset_config.get("domain", ""),
        "answer_candidates": [str(x) for x in a_value] if isinstance(a_value, (list, tuple)) else [answer],
        "data_usage": dataset_usage(dataset_config, split),
        "dataset_provenance": {"hf_id": dataset_config["hf_id"], "config": dataset_config.get("config_name"),
                               "split": split, "revision": dataset_config.get("revision")},
        "raw": json_safe(dict(example)),
    }
    if dataset_config.get("answer_selection") == "all":
        sample["answer_candidates"] = [answer]
        sample["reference_answer"] = {"format": "set", "required_answers": list(a_value),
                                      "answer_evals": example.get("answer_evals")}
    return sample, None


# --------------------------------------------------------------------------- #
# Repo image snapshot (for path-string image fields)
# --------------------------------------------------------------------------- #
def _maybe_snapshot_repo(config: dict[str, Any], dataset) -> Path | None:
    'If images are repo-relative path strings, download the repo once.'
    try:
        first = next(iter(dataset))
    except Exception:  # noqa: BLE001
        return None
    _, iv = _first_field(first, config["image_fields"])
    if isinstance(iv, (list, tuple)):
        iv = iv[0] if iv else None
    if not isinstance(iv, str):
        return None
    if _safe_exists(iv) or _looks_base64_image(iv):
        return None  # self-contained already

    # Looks like a repo-relative path -> snapshot the dataset repo.
    try:
        from huggingface_hub import snapshot_download

        logger.info("Images for %s are repo paths; downloading snapshot...",
                    config["hf_id"])
        base = snapshot_download(config["hf_id"], repo_type="dataset", revision=config.get("revision"))
        return Path(base)
    except Exception as exc:  # noqa: BLE001
        logger.warning("snapshot_download failed for %s: %s", config["hf_id"], exc)
        return None


# --------------------------------------------------------------------------- #
# Multi-source loading
# --------------------------------------------------------------------------- #
def load_sources(
    source_names: list[str],
    registry: dict[str, dict[str, Any]],
    image_cache_dir: Path,
    split_override: str | None = None,
    config_override: str | None = None,
    streaming: bool = False,
    skip_missing_answer: bool = True,
    max_per_source: int | None = None,
    dry_run: bool = False,
    task_mode: bool = False,
    source_data_root: Path | None = None,
) -> tuple[list[Sample], dict[str, dict[str, int]]]:
    'Load and normalize multiple sources.'
    image_cache_dir = Path(image_cache_dir)
    os.environ.setdefault("HF_HOME", str(image_cache_dir.resolve().parent / ".hf_cache"))
    all_samples: list[Sample] = []
    stats: dict[str, dict[str, int]] = {}

    for source_name in source_names:
        config = registry.get(source_name)
        if config is None:
            logger.warning("Unknown source '%s' — skipping.", source_name)
            continue
        if task_mode and (config.get("usage") != "evaluation" or not config.get("environment")):
            raise ValueError(f"{source_name} has no complete benchmark task adapter")

        # Sources with a registry-declared custom loader (e.g. zip-packed
        # images) dispatch generically; failure modes all flow through the
        # same zero-stats bail-out as regular sources.
        custom = config.get("custom_loader")
        if custom:
            if dry_run:
                logger.info("Source: %s (custom loader %s, skipping dry-run)",
                            source_name, custom)
                stats[source_name] = {"raw": 0, "valid": 0, "skipped": 0}
                continue
            try:
                module_name, func_name = custom.split(":")
                import importlib

                loader_fn = getattr(importlib.import_module(module_name), func_name)
            except (ImportError, AttributeError, ValueError) as exc:
                logger.error(
                    "Source %s declares custom_loader %r but it cannot be "
                    "imported (%s) — the source contributes ZERO samples. "
                    "Add the module or drop the source from --sources.",
                    source_name, custom, exc,
                )
                stats[source_name] = {"raw": 0, "valid": 0, "skipped": 0}
                continue
            try:
                local_kwargs = {"source_data_root": source_data_root} if config.get("supports_local_data") else {}
                samples, source_stats = loader_fn(
                    image_cache_dir=image_cache_dir,
                    skip_missing_answer=skip_missing_answer,
                    max_samples=max_per_source,
                    split_override=split_override, config_override=config_override,
                    streaming=streaming, task_mode=task_mode,
                    **local_kwargs,
                )
                all_samples.extend(samples)
                stats[source_name] = source_stats
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load %s: %s", source_name, exc)
                stats[source_name] = {"raw": 0, "valid": 0, "skipped": 0}
            continue

        logger.info("Loading source: %s (%s)", source_name, config.get("hf_id"))
        try:
            dataset, used_split = load_hf_dataset(
                config,
                split_override=split_override,
                config_override=config_override,
                streaming=streaming,
            )
        except Exception as exc:  # noqa: BLE001 - non-fatal
            logger.warning("Failed to load source '%s': %s — continuing.",
                           source_name, exc)
            stats[source_name] = {"raw": 0, "valid": 0, "skipped": 0}
            continue

        if dry_run:
            print_dataset_debug_info(dataset, f"{source_name} (split={used_split})")
            stats[source_name] = {"raw": 0 if streaming else len(dataset), "valid": 0, "skipped": 0}
            continue

        image_base_dir = _maybe_snapshot_repo(config, dataset)

        valid = skipped = raw_count = 0
        iterable = iter(dataset)
        if max_per_source is not None:
            iterable = itertools.islice(iterable, max_per_source)
        effective_config = {**config, "config_name": config_override if config_override is not None else config.get("config_name")}
        if task_mode:
            from sources.benchmarks import normalize_benchmark_task
            normalize = normalize_benchmark_task
        else:
            normalize = normalize_hf_sample
        try:
            for i, example in enumerate(iterable):
                raw_count += 1
                try:
                    for expanded in expand_hf_example(example, effective_config, i):
                        sample, err = normalize(
                            expanded, effective_config, i, image_cache_dir, source_name,
                            skip_missing_answer=skip_missing_answer, image_base_dir=image_base_dir, used_split=used_split)
                        if sample is not None:
                            all_samples.append(sample)
                            valid += 1
                        else:
                            skipped += 1
                            if skipped <= 3:
                                logger.debug("%s[%d] skipped: %s", source_name, i, err)
                except (TypeError, ValueError, OSError) as exc:
                    skipped += 1
                    logger.warning("%s[%d] normalization failed: %s", source_name, i, exc)
        except Exception as exc:  # streaming/network decoding can fail between rows
            logger.warning("%s stream stopped after %d rows: %s", source_name, raw_count, exc)
        stats[source_name] = {"raw": raw_count, "valid": valid, "skipped": skipped}
        logger.info("Source '%s': raw=%d valid=%d skipped=%d",
                    source_name, raw_count, valid, skipped)

    return all_samples, stats
