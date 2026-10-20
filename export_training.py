"""Export verified causal policy/observation SFT calls, or records for analysis."""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path
from typing import Any

from core.jsonl import iter_jsonl, write_jsonl
from core.provenance import fingerprint
from core.views import view_identity
from core.utils import get_logger
from records.store import latest_records
from records.quality import training_rejection_reason
from records.validator import STEP_FIELDS

logger = get_logger("export_training")
ECHOED_FIELDS = ("id", "image", "image_size", "question", "source_dataset", "final_answer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export exact executed calls for active visual/language SFT.")
    p.add_argument("--input", required=True, help="Verified JSONL from verify.py.")
    p.add_argument("--output", required=True)
    p.add_argument("--format", choices=("calls", "records"), default="calls",
                   help="calls: policy/observation SFT; records: analysis and evaluation only")
    p.add_argument("--tasks", choices=("all", "both", "policy", "observation", "resolution"), default="all",
                   help="all includes policy, observation and language resolution; both selects the first two")
    p.add_argument("--image-root", help="Make executed view references relative to this root.")
    p.add_argument("--max-warnings", type=int)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--split", choices=("all", "train", "validation"), default="all")
    p.add_argument("--stats", action="store_true")
    args = p.parse_args()
    if args.max_warnings is not None and args.max_warnings < 0:
        p.error("--max-warnings must be nonnegative")
    if not 0 <= args.validation_fraction < 1:
        p.error("--validation-fraction must be in [0,1)")
    if Path(args.input).resolve() == Path(args.output).resolve():
        p.error("Input and output must be different files")
    return args


def clean_step(step: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(step[k]) for k in STEP_FIELDS if k in step}


def build_trace(annotation: dict[str, Any]) -> dict[str, Any]:
    """An analysis trace. The inserted dataset answer is never a policy target."""
    trace = {k: copy.deepcopy(v) for k, v in annotation.items() if k not in ECHOED_FIELDS}
    trace["search_trace"] = [clean_step(s) for s in annotation["search_trace"]]
    return trace


def behavior_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    """Measured execution descriptors; none is a prescribed positive reward."""
    entries = rec["execution"]["steps"]
    views = [e["view"] for e in entries if e["view"]]
    distinct = {view_identity(v) for v in views}
    return {"n_steps": len(entries), "n_policy_decisions": sum(e["planner"] is not None for e in entries),
            "n_image_operations": len(views), "n_unique_views": len(distinct),
            "n_repeated_views": len(views) - len(distinct),
            "executed_view_pixels": sum(v["width"] * v["height"] for v in views),
            "n_recover": sum(e["step"]["action"] == "recover" for e in entries),
            "n_hypothesis_revisions": sum(e["step"]["belief_update"]["revision_of_step"] is not None for e in entries),
            "terminal_action": entries[-1]["step"]["action"],
            "gap_types": dict(collections.Counter(e["request"]["gap_type"] for e in entries if e["planner"]))}


def split_for(rec: dict[str, Any], fraction: float = 0.1) -> str:
    """Preserve pre-generation seed groups; otherwise group by image bytes."""
    derivation = rec.get("trajectory_derivation")
    identity = ({"seed_group": derivation["group_key"]} if derivation else
                {"image_sha256": rec["image_sha256"]})
    value = int(fingerprint(identity)[:16], 16) / 2**64
    return "validation" if value < fraction else "train"


def _image_path(path: str, root: Path | None) -> str:
    if root is not None:
        try:
            return str(Path(path).resolve().relative_to(root.resolve()))
        except ValueError:
            pass
    return path


def _meta(rec: dict[str, Any]) -> dict[str, Any]:
    return {"trace_origin": "executed_policy", "final_answer_origin": "held_out_dataset_label",
            "external_hops_executed": False, "image_sha256": rec["image_sha256"],
            "data_usage": rec.get("data_usage", "train"), "dataset_provenance": rec.get("dataset_provenance", {}),
            "trajectory_derivation": rec.get("trajectory_derivation"),
            "generation_fingerprint": rec["generation_fingerprint"],
            "model": rec["model"], "provider": rec["provider"], "trace_variant": rec["trace_variant"],
            "judge_model": rec["verification"]["judge_model"],
            "verification_version": rec["verification"]["version"],
            "verification_input_fingerprint": rec["verification"]["input_fingerprint"],
            "n_verified_claims": len(rec["verification"]["grounding"]["claims"])}


def to_training_record(rec: dict[str, Any], image_root: Path | None) -> dict[str, Any]:
    """Whole-record format for inspection/evaluation, not agent policy SFT."""
    return {"id": rec["id"], "source_dataset": rec["source_dataset"],
            "image": _image_path(rec["image_path"], image_root), "image_size": rec["image_size"],
            "question": rec["question"], "answer": rec["answer"], "trace": build_trace(rec["annotation"]),
            "answer_candidates": rec.get("answer_candidates", [rec["answer"]]),
            "behavior_metrics": behavior_metrics(rec),
            "diagnostics": {"sufficient_evidence_step": rec["annotation"]["sufficient_evidence_step"],
                            "costs_and_sufficiency_are_estimates": True, "causal_necessity_proven": False,
                            "answer_is_student_prediction": False},
            "meta": {**_meta(rec), "intended_use": "analysis_and_evaluation"}}


def to_call_records(rec: dict[str, Any], image_root: Path | None = None,
                    tasks: str = "all", validation_fraction: float = 0.1) -> list[dict]:
    """Supervise the exact request/response boundary, with no hindsight inputs."""
    rows = []
    trajectory = str(rec["id"]) + ":" + rec["trace_variant"]
    split = split_for(rec, validation_fraction)
    calls = [(entry, task, call) for entry in rec["execution"]["steps"]
             for task, call in (("policy", entry["planner"]), ("observation", entry["observer"]))]
    calls.append((rec["execution"]["steps"][-1], "resolution", rec["execution"]["resolution"]))
    for entry, task, call in calls:
        selected = tasks in ("all", task) or (tasks == "both" and task in {"policy", "observation"})
        if call is None or not selected:
            continue
        content = [{"type": "image", "image": _image_path(p, image_root)} for p in call["images"]]
        content.append({"type": "text", "text": call["prompt"]})
        rows.append({"id": f"{trajectory}:{entry['step']['step']}:{task}",
                     "trajectory_id": trajectory, "source_dataset": rec["source_dataset"],
                     "task": task, "split": split,
                     "messages": [{"role": "user", "content": content},
                                  {"role": "assistant", "content": json.dumps(call["response"], ensure_ascii=False)}],
                     "meta": {**_meta(rec), "step": entry["step"]["step"],
                              "context_limit": call["context_limit"], "loss_scope": "assistant_only",
                              "forced_initialization": entry["forced_initialization"],
                              "behavior_state": entry["step"]["belief_update"]["status"],
                              "gap_type": entry["request"]["gap_type"],
                              "trajectory_outcome": "verified_abstention" if
                                  rec["verification"]["grounding"]["mode"] == "abstention" else "resolved",
                              "answer_is_policy_target": False}})
    for row in rows:
        row["sample_weight"] = 1 / len(rows)
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.image_root) if args.image_root else None
    kept, dropped = [], collections.Counter()
    n_trajectories = 0
    for rec in latest_records(list(iter_jsonl(args.input))):
        reason = training_rejection_reason(rec, allow_abstention=args.format == "calls",
                                           allow_evaluation=args.format == "records")
        if reason:
            dropped[reason] += 1
            continue
        if args.max_warnings is not None and len(rec.get("validation_warnings") or []) > args.max_warnings:
            dropped["too many validation warnings"] += 1
            continue
        split = split_for(rec, args.validation_fraction)
        if args.split != "all" and args.split != split:
            continue
        if args.format == "calls":
            rows = to_call_records(rec, root, args.tasks, args.validation_fraction)
        else:
            rows = [{**to_training_record(rec, root), "split": split}]
        kept.extend(rows)
        n_trajectories += 1
    write_jsonl(args.output, kept)
    logger.info("Exported %d rows from %d accepted trajectories to %s", len(kept), n_trajectories, args.output)
    for reason, count in dropped.most_common():
        logger.info("  dropped %d (%s)", count, reason)
    if args.stats:
        print(f"Accepted trajectories: {n_trajectories}; rows: {len(kept)}")
        for field in ("task", "split", "source_dataset"):
            print(f"{field}: {dict(collections.Counter(r.get(field, 'record') for r in kept))}")


if __name__ == "__main__":
    main()
