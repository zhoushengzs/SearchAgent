'Stage 1 CLI: prepare a normalized JSONL from HuggingFace sources.'
from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.config import PROJECT_ROOT, DEFAULT_IMAGE_CACHE, DEFAULT_SAMPLE_SIZE, DEFAULT_SEED
from sources.registry import DATASET_REGISTRY, resolve_source_names
from sources.huggingface import load_sources
from sources.sampler import (parse_quota, per_source_sample, random_sample,
                             source_distribution)
from core.utils import get_logger
from core.jsonl import write_jsonl

logger = get_logger("prepare_hf_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare normalized QA from registered training sources or held-out benchmarks."
    )
    p.add_argument("--sources", type=str, default=None,
                   help="Comma-separated source names. Default: all enabled sources.")
    p.add_argument("--output", type=str, default=None,
                   help="Output normalized JSONL path.")
    p.add_argument("--image-cache", type=str, default=str(DEFAULT_IMAGE_CACHE),
                   help="Directory to save extracted images.")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                   help="Global random sample size across the merged pool "
                        "(default: keep the whole pool).")
    p.add_argument("--per-source-size", type=int, default=None,
                   help="Cap every source to this many samples before global "
                        "sampling, so one large source cannot dominate the pool.")
    p.add_argument("--source-quota", type=str, default=None,
                   help="Per-source caps: 'name=N' pairs, optional 'default=N' "
                        "for the rest. Example: "
                        "'mmsearch_plus=150,default=100'. Takes precedence "
                        "over --per-source-size.")
    p.add_argument("--max-per-source", type=int, default=None,
                   help="DEBUG: max rows to READ per source (before normalization).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--split", type=str, default=None, help="Global split override.")
    p.add_argument("--config-name", type=str, default=None,
                   help="Global config_name override.")
    p.add_argument("--skip-missing-answer", action="store_true", default=False,
                   help="Skip samples without an answer (else keep + flag review).")
    p.add_argument("--keep-all", action="store_true", default=False,
                   help="Keep all valid samples (overrides --sample-size).")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Only print splits/features/sample fields; do not save.")
    p.add_argument("--streaming", action="store_true", help="Read only selected rows without downloading full parquet pools.")
    p.add_argument("--format", choices=("visual", "benchmark"), default="visual",
                   help="visual: single-image module inputs; benchmark: complete tasks for agent_pipeline.py or official runners.")
    p.add_argument("--references-output", help="Benchmark reference file (default: <output-stem>_references.jsonl).")
    p.add_argument("--source-data-root", help="Local author assets: <root>/<source>/metadata.jsonl and images or images.zip/data.zip.")
    p.add_argument("--allow-partial-sources", action="store_true", help="Allow a requested source to contribute zero tasks; default fails the build.")
    args = p.parse_args()
    if any(x is not None and x < 1 for x in (args.max_per_source, args.per_source_size, args.sample_size)):
        p.error("Sample limits must be positive")
    if args.references_output and args.format != "benchmark":
        p.error("--references-output requires --format benchmark")
    if args.format == "benchmark" and args.output:
        args.references_output = args.references_output or str(Path(args.output).with_name(Path(args.output).stem + "_references.jsonl"))
        if Path(args.output).resolve() == Path(args.references_output).resolve():
            p.error("Benchmark tasks and references must be different files")
    return args


def main() -> None:
    args = parse_args()
    source_names = resolve_source_names(
        [s.strip() for s in args.sources.split(",")] if args.sources else None
    )
    logger.info("Sources: %s", source_names)

    all_samples, per_source_raw_stats = load_sources(
        source_names,
        DATASET_REGISTRY,
        image_cache_dir=Path(args.image_cache),
        split_override=args.split,
        config_override=args.config_name,
        streaming=args.streaming,
        skip_missing_answer=args.skip_missing_answer,
        max_per_source=args.max_per_source,
        dry_run=args.dry_run,
        task_mode=args.format == "benchmark",
        source_data_root=Path(args.source_data_root) if args.source_data_root else None,
    )

    if args.dry_run:
        logger.info("Dry-run complete. Per-source raw counts: %s", per_source_raw_stats)
        return

    empty = [name for name in source_names if not per_source_raw_stats.get(name, {}).get("valid")]
    if empty and not args.allow_partial_sources:
        raise SystemExit(f"Requested sources produced no usable inputs: {empty}. Fix access/assets or explicitly use --allow-partial-sources.")

    # --- Report per-source valid counts -----------------------------------
    logger.info("=" * 60)
    for src, s in per_source_raw_stats.items():
        logger.info("  %-22s raw=%-6d valid=%-6d skipped=%-6d",
                    src, s["raw"], s["valid"], s["skipped"])
    logger.info("Merged all_valid_samples total: %d", len(all_samples))

    # --- Optional per-source cap (keeps one big source from dominating) ---
    pool = all_samples
    quota = parse_quota(args.source_quota)
    if quota or args.per_source_size:
        before = source_distribution(pool)
        pool = per_source_sample(pool, args.per_source_size, args.seed, quota=quota)
        after = source_distribution(pool)
        logger.info("Per-source cap applied (%s): %d -> %d samples",
                    args.source_quota or f"per-source-size={args.per_source_size}",
                    len(all_samples), len(pool))
        for src_name in sorted(before):
            if after.get(src_name, 0) != before[src_name]:
                logger.info("  %-22s %d -> %d", src_name, before[src_name],
                            after.get(src_name, 0))

    # --- Global random sampling (the formal build path) -------------------
    if args.keep_all:
        final = pool
        logger.info("--keep-all set: keeping all %d valid samples.", len(final))
    else:
        final = random_sample(pool, args.sample_size, args.seed)
        logger.info("Global random sample: requested=%s -> got %d (seed=%d)",
                    args.sample_size, len(final), args.seed)

    dist = source_distribution(final)
    total = len(final) or 1
    logger.info("Final source_dataset distribution:")
    for src_name, n in dist.items():
        logger.info("  %-22s %5d  (%4.1f%%)", src_name, n, 100 * n / total)
    top, top_n = max(dist.items(), key=lambda kv: kv[1], default=("-", 0))
    if args.format != "benchmark" and len(dist) > 1 and top_n / total > 0.40:
        logger.warning(
            "%s alone is %.0f%% of the pool — its characteristics will dominate "
            "the dataset, and the small sources may be too thin for a "
            "per-benchmark breakdown. Consider --source-quota.",
            top, 100 * top_n / total)

    if not args.output:
        logger.warning("No --output given; not writing. (Use --dry-run to inspect.)")
        return

    # Store image paths relative to the project root, so the JSONL carries no
    # machine-specific prefix and works on any checkout (run the later stages
    # from the project root).
    if args.format == "benchmark":
        from sources.benchmarks import separate_references
        for sample in final:
            for image in sample["model_input"]["images"]:
                image["path"] = os.path.relpath(Path(image["path"]).resolve(), PROJECT_ROOT)
        tasks, references = separate_references(final)
        write_jsonl(args.references_output, references)
        write_jsonl(args.output, tasks)
        logger.info("Wrote %d held-out references to %s", len(references), args.references_output)
    else:
        for sample in final:
            sample["image_path"] = os.path.relpath(
                Path(sample["image_path"]).resolve(), PROJECT_ROOT)
        write_jsonl(args.output, final)
    logger.info("Wrote %d normalized samples to %s", len(final), args.output)


if __name__ == "__main__":
    main()
