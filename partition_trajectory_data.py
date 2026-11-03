"""Freeze explicit benchmark-derived trajectory seeds and independent holdout inputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from core.jsonl import write_jsonl
from sources.partition import partition_benchmark_seeds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Original normalized visual inputs or detached benchmark tasks JSONL.")
    p.add_argument("--output-prefix", required=True, help="Writes <prefix>_train.jsonl, _holdout.jsonl and _partition.json.")
    p.add_argument("--split-name", required=True, help="Named benchmark-derived protocol, e.g. cvpr-seeds-v1.")
    p.add_argument("--image-root", default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    args = p.parse_args()
    prefix = Path(args.output_prefix)
    paths = {role: prefix.with_name(prefix.name + suffix) for role, suffix in
             (("train", "_train.jsonl"), ("holdout", "_holdout.jsonl"), ("report", "_partition.json"))}
    if any(path.exists() or path.resolve() == Path(args.input).resolve() for path in paths.values()):
        p.error("Partition outputs must be new files; freeze a new named protocol for a changed pool")
    try:
        # Fail on malformed input rather than silently change the frozen pool.
        rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Every input row must be an object")
        schemas = {row.get("schema", "visual") for row in rows}
        if len(schemas) != 1 or not schemas <= {"visual", "visual-agent-task-v1"}:
            raise ValueError("Do not mix task manifests and visual-module inputs")
        split = partition_benchmark_seeds(rows, split_name=args.split_name, seed=args.seed,
                    holdout_fraction=args.holdout_fraction, image_root=Path(args.image_root))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    report = {"split_name": args.split_name, "seed": args.seed, "holdout_fraction": args.holdout_fraction,
              "corpus_fingerprint": split[0]["trajectory_derivation"]["corpus_fingerprint"],
              "groups": len({row["trajectory_derivation"]["group_key"] for row in split}),
              "counts": {}, "source_counts": {}}
    for role in ("train", "holdout"):
        subset = [row for row in split if row["trajectory_derivation"]["partition"] == role]
        report["counts"][role] = len(subset)
        report["source_counts"][role] = dict(Counter(row["source_dataset"] for row in subset))
        write_jsonl(paths[role], subset)
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(report["counts"].values()):
        print("Warning: one partition is empty. This pool does not provide an independent derived holdout.")


if __name__ == "__main__":
    main()
