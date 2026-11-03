"""Run, audit and export real multimodal retrieval trajectories from task manifests."""
from __future__ import annotations

import argparse
import collections
import json
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from agent.audit import execution_errors, training_rejection_reason, verify_rollout, reusable_verification
from agent.metrics import latest_rollouts
from agent.runner import RUN_VERSION, retry_prompt, rollout_fingerprint, run_task
from agent.tools import EvidenceTools
from core.config import PROVIDER_CHOICES
from core.jsonl import append_jsonl, write_jsonl
from core.provenance import fingerprint
from labeling.client import VLMClient


def read_rows(path):
    # JSONL records are separated by LF. str.splitlines() also splits on
    # Unicode separators (for example U+2028) that may legitimately occur
    # inside JSON string values, corrupting otherwise valid records.
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every JSONL row must be an object")
    return rows


def select_tasks(rows, *, limit=None, per_source_limit=None, seed=42):
    if per_source_limit is None:
        return rows[:limit] if limit else rows
    grouped = collections.defaultdict(list)
    for task in rows:
        grouped[task["source_dataset"]].append(task)
    return [task for source in sorted(grouped) for task in sorted(grouped[source],
        key=lambda task: fingerprint({"seed": seed, "input": task["input_fingerprint"]}))[:per_source_limit]]


def training_prompt(call):
    """Remove validation feedback from the SFT input, preserving the audited ledger."""
    prompt = call["prompt"]
    rejected = call.get("rejected_attempts") or []
    if not rejected:
        return prompt
    suffix = retry_prompt("", rejected)
    if not prompt.endswith(suffix):
        raise ValueError("Validation retry prompt does not match the recorded rejected attempts")
    return prompt[:-len(suffix)]


def export_calls(record):
    """Export verified calls with clean prompts and no failed tool observations."""
    reason = training_rejection_reason(record)
    if reason:
        raise ValueError(reason)
    if record.get("failed_attempt") or any(
            entry.get("result") and entry["result"].get("status") != "ok"
            for entry in record["steps"]):
        raise ValueError("Tool errors are not admitted to training export")
    calls = [(entry[role], role, entry["step"]) for entry in record["steps"]
             for role in ("planner", "observer") if entry[role] is not None]
    task = record["task"]
    derivation = task["trajectory_derivation"]
    validation = int(fingerprint({"group": derivation["group_key"]})[:16], 16) / 2**64 < 0.1
    rows = []
    for call, role, step in calls:
        rows.append({"id": f"{task['id']}_{record.get('run_identity', '')[:10]}_{step}_{role}",
            "task": "policy" if role == "planner" else "observation", "split": "validation" if validation else "train",
            "sample_weight": 1 / len(calls), "loss_mask": "assistant_only",
            "messages": [{"role": "user", "content": [
                *[{"type": "image", "path": path} for path in call["images"]],
                {"type": "text", "text": training_prompt(call)}]},
                {"role": "assistant", "content": [{"type": "text", "text": json.dumps(call["response"], ensure_ascii=False)}]}],
            "meta": {"source_dataset": task["source_dataset"], "dataset_provenance": task["dataset_provenance"],
                     "trajectory_derivation": derivation, "input_fingerprint": task["input_fingerprint"],
                     "protocol": record["protocol"], "external_tools_executed": True,
                     "outcome": record["termination"], "model": record["model"], "provider": record["provider"],
                     "judge_model": record["verification"]["judge_model"],
                     "reference_evaluation_receipt": fingerprint(record.get("reference_evaluation")),
                     "actual_steps": len(record["steps"]),
                     "validation_feedback_removed": bool(call.get("rejected_attempts"))}})
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Execute detached task inputs with actual evidence tools")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--artifacts-dir")
    run.add_argument("--model")
    run.add_argument("--provider", choices=PROVIDER_CHOICES, default="auto")
    run.add_argument("--max-steps", type=int, default=30)
    run.add_argument("--workers", type=int, default=1, help="Maximum tasks collected at once")
    run.add_argument("--max-context-chars", type=int, default=200000)
    run.add_argument("--max-context-images", type=int, default=32)
    run.add_argument("--image-side", type=int, default=1536)
    run.add_argument("--kb", help="Full independent KB JSONL: unique id, text, optional title/url; never a task's gold evidence subset")
    run.add_argument("--inspection-only", action="store_true", help="Explicitly run without a search backend for static visual tasks")
    sampling = run.add_mutually_exclusive_group()
    sampling.add_argument("--limit", type=int)
    sampling.add_argument("--per-source-limit", type=int, help="Balanced pilot from an already frozen partition")
    run.add_argument("--source-dataset", help="Collect only tasks from this dataset")
    run.add_argument("--seed", type=int, default=42, help="Deterministic pilot selection seed")
    run.add_argument("--no-resume", action="store_true")
    verify = sub.add_parser("verify", help="Independently audit evidence and separately evaluate detached answers")
    verify.add_argument("--input", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--references", help="Detached reference JSONL; required to admit successful answers to SFT")
    verify.add_argument("--judge-model", required=True)
    verify.add_argument("--provider", choices=PROVIDER_CHOICES, default="auto")
    verify.add_argument("--workers", type=int, default=1, help="Concurrent trajectory audits (1-32)")
    export = sub.add_parser("export", help="Export exact calls from audited derived-training trajectories")
    export.add_argument("--input", required=True)
    export.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        p.error("Input and output must be different files")
    if args.command == "run" and (not 1 <= args.max_steps <= 80 or not 1 <= args.workers <= 32 or
            not 64 <= args.image_side <= 2048 or
            args.max_context_chars < 5000 or args.max_context_images < 1 or
            any(n is not None and n < 1 for n in (args.limit, args.per_source_limit))):
        p.error("Invalid agent execution budget")
    if args.command == "verify" and not 1 <= args.workers <= 32:
        p.error("Verify workers must be between 1 and 32")
    if args.command == "verify" and args.references and Path(args.references).resolve() == Path(args.output).resolve():
        p.error("Do not overwrite detached references")
    return args


def main():
    args = parse_args()
    rows = read_rows(args.input)
    if args.command == "run":
        if args.source_dataset:
            rows = sorted((row for row in rows if row["source_dataset"] == args.source_dataset),
                          key=lambda row: fingerprint({"seed": args.seed, "input": row["input_fingerprint"]}))
            if not rows:
                raise SystemExit(f"No tasks found for source_dataset={args.source_dataset!r}")
        rows = select_tasks(rows, limit=args.limit, per_source_limit=args.per_source_limit, seed=args.seed)
        print(json.dumps({"selected_tasks": len(rows), "sources": dict(collections.Counter(r["source_dataset"] for r in rows))}))
        ids = [task["id"] for task in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit("Task IDs must be unique")
        client = VLMClient(model=args.model, provider=args.provider)
        root = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.output).with_name(Path(args.output).stem + "_artifacts")
        existing = read_rows(args.output) if Path(args.output).exists() and not args.no_resume else []
        previous = {row.get("run_identity"): row for row in existing}
        if args.no_resume:
            write_jsonl(args.output, [])
        def collect(task):
            tools = EvidenceTools(root / (fingerprint(task)[:16] + "_" + uuid.uuid4().hex[:8]),
                                  kb_path=Path(args.kb) if args.kb else None, image_side=args.image_side)
            try:
                if not tools.serper_key and not tools.kb and not args.inspection_only:
                    raise SystemExit("Set SERPER_API_KEY or provide --kb for retrieval; static visual diagnostics can use --inspection-only")
                identity = fingerprint({"version": RUN_VERSION, "task": task, "model": client.model,
                    "provider": client.provider, "tool_config": tools.configuration(), "max_steps": args.max_steps,
                    "max_context_chars": args.max_context_chars, "max_context_images": args.max_context_images})
                old = previous.get(identity)
                if old and old.get("termination") in {"finish", "abstain"} and not execution_errors(old):
                    return None
                record = run_task(task, client, tools, max_steps=args.max_steps,
                                  max_context_chars=args.max_context_chars, max_context_images=args.max_context_images)
                record["run_identity"] = identity
                return record
            finally:
                tools.close()

        def save(record):
            if record is not None:
                append_jsonl(args.output, record)
                print(json.dumps({"id": record["task"]["id"], "termination": record["termination"],
                                  "steps": len(record["steps"])}), flush=True)

        if args.workers == 1:
            for task in rows:
                save(collect(task))
        else:
            tasks = iter(rows)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(collect, task) for _, task in zip(range(args.workers), tasks)}
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        save(future.result())
                        task = next(tasks, None)
                        if task is not None:
                            pending.add(pool.submit(collect, task))
    elif args.command == "verify":
        references = read_rows(args.references) if args.references else []
        by_key = {(r["id"], r["input_fingerprint"]): r for r in references}
        if len(by_key) != len(references):
            raise SystemExit("Duplicate detached reference keys")
        rows = latest_rollouts(rows)
        if args.references and any((r["task"]["id"], r["task"]["input_fingerprint"]) not in by_key for r in rows):
            raise SystemExit("A rollout has no matching detached reference; check IDs and input fingerprints")
        judge = VLMClient(model=args.judge_model, provider=args.provider)
        previous_verified = read_rows(args.output) if Path(args.output).exists() else []
        previous_by_key = {
            (item.get("verification", {}).get("input_fingerprint"),
             item.get("verification", {}).get("judge_model")): item
            for item in previous_verified
        }
        def audit_one(row):
            task = row["task"]
            reference = by_key.get((task["id"], task["input_fingerprint"]))
            key = (rollout_fingerprint(row), judge.model)
            previous = previous_by_key.get(key)
            if previous is not None and reusable_verification(previous, row, judge.model, reference):
                return previous, False
            return verify_rollout(row, judge, reference, previous=previous), True

        verified = []
        def save_audit(result):
            out, changed = result
            if changed:
                append_jsonl(args.output, out)
            verified.append(out)
            print(json.dumps({"id": out["task"]["id"], "grounding": out["verification"]["pass"],
                              "training_rejection": training_rejection_reason(out)}), flush=True)

        if args.workers == 1:
            for row in rows:
                save_audit(audit_one(row))
        else:
            tasks = iter(rows)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(audit_one, row) for _, row in zip(range(args.workers), tasks)}
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        save_audit(future.result())
                        row = next(tasks, None)
                        if row is not None:
                            pending.add(pool.submit(audit_one, row))
        write_jsonl(args.output, verified)
    else:
        final, rejections = [], collections.Counter()
        for row in latest_rollouts(rows):
            try:
                final.extend(export_calls(row))
            except ValueError as exc:
                rejections[str(exc)] += 1
        write_jsonl(args.output, final)
        print(json.dumps({"exported_calls": len(final), "rejections": dict(rejections)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
