"""Stage 2: answer-blind sequential visual search with persistent executed views."""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.config import (DEFAULT_MAX_TOKENS, PROVIDER_CHOICES, DEFAULT_RETRY,
                         DEFAULT_SEED, DEFAULT_SLEEP, DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT)
from core.jsonl import append_jsonl, iter_jsonl, write_jsonl
from core.provenance import PIPELINE_VERSION, file_sha, fingerprint
from core.utils import get_logger, read_image_size
from labeling.client import VLMClient
from labeling.policy import run_policy
from labeling.prompt import DEFAULT_PROMPT_PATH, Instruction
from records.quality import execution_errors
from records.store import is_successful_record, latest_records, resume_key
from records.validator import (ALLOWED_TRACE_VARIANTS, DEFAULT_TRACE_VARIANT,
                               MAX_TRACE_STEPS, check_ground_truth, validate_annotation)
from sources.local import load_and_normalize_local
from sources.partition import derived_partition
from sources.sampler import random_sample

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute answer-blind visual search annotations.")
    p.add_argument("--input", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--views-dir", help="Persistent view cache; defaults beside output.")
    p.add_argument("--sample-size", type=int)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--provider", default="auto", choices=PROVIDER_CHOICES)
    p.add_argument("--model")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--retry", type=int, default=DEFAULT_RETRY)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p.add_argument("--skip-missing-answer", action="store_true", default=False)
    p.add_argument("--limit", type=int)
    p.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH))
    p.add_argument("--trace-variant", default=DEFAULT_TRACE_VARIANT,
                   choices=sorted(ALLOWED_TRACE_VARIANTS))
    p.add_argument("--max-steps", type=int, default=MAX_TRACE_STEPS)
    p.add_argument("--global-side", type=int, default=512,
                   help="Longest side of the initial coarse view.")
    p.add_argument("--crop-side", type=int, default=1536,
                   help="Longest side of each local view; never upsample missing detail.")
    args = p.parse_args()
    if not 3 <= args.max_steps <= MAX_TRACE_STEPS:
        p.error(f"--max-steps must be between 3 and {MAX_TRACE_STEPS}")
    if (min(args.concurrency, args.retry, args.max_tokens, args.timeout) < 1
            or not 64 <= args.global_side <= args.crop_side <= 2048 or args.sleep < 0):
        p.error("Invalid request budget or view dimensions (64 <= global <= crop <= 2048)")
    if not math.isfinite(args.temperature) or args.temperature < 0 or any(
            x is not None and x < 1 for x in (args.limit, args.sample_size)):
        p.error("Temperature must be finite/nonnegative and sample limits positive")
    if Path(args.input).resolve() == Path(args.output).resolve():
        p.error("Input and output must be different files")
    return args


def _resolve_image_path(image_path: str, image_root: Path) -> str:
    path = Path(image_path)
    if not path.is_absolute() and not path.exists():
        path = image_root / path
    return str(path.resolve())


def generation_identity(sample, client, args, image_root, instruction):
    path = _resolve_image_path(sample["image_path"], image_root)
    identity = {"version": PIPELINE_VERSION, "id": str(sample["id"]),
                "source_dataset": sample.get("source_dataset", ""), "image_path": path,
                "image_sha256": file_sha(path), "question": sample.get("question", ""),
                "answer": sample.get("answer", ""), "input_review": sample.get("needs_human_review", False),
                "data_usage": sample.get("data_usage", "train"),
                "dataset_provenance": sample.get("dataset_provenance", {}),
                "trajectory_derivation": sample.get("trajectory_derivation"),
                "leakage_group": sample.get("leakage_group"),
                "answer_candidates": sample.get("answer_candidates", []),
                "reference_answer": sample.get("reference_answer"),
                "provider": client.provider, "model": client.model,
                "prompt_sha": instruction.sha_for(str(sample.get("source_dataset", ""))),
                "variant": args.trace_variant, "temperature": args.temperature,
                "max_tokens": args.max_tokens, "max_steps": args.max_steps,
                "global_side": args.global_side, "crop_side": args.crop_side}
    return fingerprint(identity), identity


def _process_one(sample, client, args, image_root: Path, instruction: Instruction) -> dict:
    """Capture all failures as records; workers never corrupt the output file."""
    record = {"id": str(sample["id"]), "source_dataset": sample.get("source_dataset", ""),
              "image_path": _resolve_image_path(sample["image_path"], image_root),
              "question": sample.get("question", ""), "answer": sample.get("answer", ""),
              "data_usage": sample.get("data_usage", "train"),
              "dataset_provenance": sample.get("dataset_provenance", {}),
              "trajectory_derivation": sample.get("trajectory_derivation"),
              "leakage_group": sample.get("leakage_group"),
              "answer_candidates": sample.get("answer_candidates", []),
              "reference_answer": sample.get("reference_answer"),
              "trace_variant": args.trace_variant, "model": client.model, "provider": client.provider,
              "prompt_file": instruction.name,
              "prompt_sha": instruction.sha_for(str(sample.get("source_dataset", ""))),
              "annotation": None, "raw_model_output": "", "parse_success": False,
              "validation_errors": [], "validation_warnings": [], "api_error": None,
              "truncated": False, "needs_human_review": bool(sample.get("needs_human_review", False))}
    try:
        run_sha, identity = generation_identity(sample, client, args, image_root, instruction)
        record.update(generation_fingerprint=run_sha, image_sha256=identity["image_sha256"],
                      generation_config=identity)
        if record.get("trajectory_derivation") is not None and derived_partition(record) is None:
            raise ValueError("Benchmark-derived input or partition metadata changed; use the frozen seed pool")
        size = read_image_size(record["image_path"])
        if size is None:
            raise ValueError("Cannot read source image size")
        record["image_size"] = size
        prepared = {**sample, "image_path": record["image_path"], "image_size": size,
                    "trace_variant": args.trace_variant}
        output = Path(args.output)
        cache = Path(args.views_dir) if args.views_dir else output.parent / (output.stem + "_views")
        annotation, execution, raw = run_policy(prepared, client, instruction, cache / run_sha,
            max_steps=args.max_steps, global_side=args.global_side, crop_side=args.crop_side)
        record.update(annotation=annotation, execution=execution, raw_model_output=raw, parse_success=True)
        errors, warnings = validate_annotation(annotation)
        errors += check_ground_truth(annotation, record["answer"])
        errors += execution_errors(record)
        record.update(validation_errors=errors, validation_warnings=warnings)
        if annotation["quality_flags"].get("needs_human_review"):
            record["needs_human_review"] = True
    except Exception as exc:
        record["validation_errors"].append(f"{type(exc).__name__}: {exc}")
        record["needs_human_review"] = True
        logger.warning("id=%s annotation failed: %s", record["id"], exc)
    return record


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    samples, load_errors = load_and_normalize_local(args.input,
                                                   skip_missing_answer=args.skip_missing_answer)
    if load_errors:
        logger.info("Skipped %d invalid input records", len(load_errors))
    if args.sample_size:
        samples = random_sample(samples, args.sample_size, args.seed)
    if args.limit:
        samples = samples[:args.limit]
    keys = [(str(s["id"]), args.trace_variant) for s in samples]
    if len(set(keys)) != len(keys):
        raise SystemExit("Duplicate input ids for this trace variant")
    instruction = Instruction(args.prompt)
    client = VLMClient(model=args.model, provider=args.provider, temperature=args.temperature,
                       max_tokens=args.max_tokens, timeout=args.timeout, retry=args.retry)
    previous = latest_records(list(iter_jsonl(args.output))) if args.resume else []
    if not args.resume:
        write_jsonl(args.output, [])
    by_key = {resume_key(r): r for r in previous}
    todo = []
    for sample in samples:
        old = by_key.get((str(sample["id"]), args.trace_variant))
        try:
            current, _ = generation_identity(sample, client, args, image_root, instruction)
            done = (old and is_successful_record(old)
                    and old.get("generation_fingerprint") == current
                    and not execution_errors(old))
        except (OSError, ValueError):
            done = False
        if not done:
            todo.append(sample)
    logger.info("total=%d to_process=%d skipped=%d", len(samples), len(todo), len(samples) - len(todo))
    new = []
    def handle(record):
        append_jsonl(args.output, record)
        new.append(record)
        if len(new) == 1 or len(new) % 10 == 0:
            logger.info("processed=%d/%d successful=%d", len(new), len(todo),
                        sum(is_successful_record(r) for r in new))
    if args.concurrency == 1:
        for sample in todo:
            handle(_process_one(sample, client, args, image_root, instruction))
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_process_one, s, client, args, image_root, instruction) for s in todo]
            for future in as_completed(futures):
                handle(future.result())
    write_jsonl(args.output, latest_records(previous + new))
    logger.info("Finished: processed=%d successful=%d", len(new), sum(is_successful_record(r) for r in new))


if __name__ == "__main__":
    main()
