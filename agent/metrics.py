"""Descriptive metrics for actual interactions; declared dependencies are not causal proofs."""
from __future__ import annotations

from collections import Counter

from agent.audit import training_rejection_reason
from core.provenance import fingerprint


def latest_rollouts(records):
    return list({r.get("run_identity") or fingerprint(r["task"]): r for r in records}.values())


def rollout_statistics(records):
    rows = latest_rollouts(records)
    tools, gaps, horizons, lengths, depth_dist, admissions = [Counter() for _ in range(6)]
    errors = revisions = merges = rewritten_queries = 0
    for row in rows:
        steps = row["steps"]
        executed = [e for e in steps if e["result"] is not None]
        pending = row.get("failed_attempt")
        tool_count = len(executed) + int(bool(pending and pending.get("result")))
        lengths[tool_count] += 1
        horizons["short_0_4" if tool_count < 5 else "medium_5_14" if tool_count < 15 else "long_15_plus"] += 1
        depths = {}
        for e in steps:
            request = e["request"]
            gaps[request["gap_type"]] += 1
            dependencies = request["depends_on"]
            depths[e["step"]] = 1 + max((depths[d] for d in dependencies), default=0)
            merges += len(dependencies) >= 2
            revisions += bool(e["observation"] and e["observation"]["revision_of_step"] is not None)
            if e["result"] is not None:
                tools[request["tool"]] += 1
                errors += e["result"]["status"] == "error"
            if request["tool"] in {"web_search", "image_search"} and e["step"] > 1:
                previous = steps[e["step"] - 2]
                rewritten_queries += (previous["request"]["tool"] == request["tool"] and
                    previous["request"]["arguments"]["query"] != request["arguments"]["query"])
        if pending and pending.get("result"):
            tools[pending["result"]["tool"]] += 1
            errors += pending["result"]["status"] == "error"
        depth_dist[max(depths.values(), default=0)] += 1
        admissions[training_rejection_reason(row) or "accepted"] += 1
    return {"attempted_records": len(records), "latest_runs": len(rows),
        "unique_seed_inputs": len({r["task"]["input_fingerprint"] for r in rows}),
        "sources": dict(Counter(r["task"]["source_dataset"] for r in rows)),
        "termination": dict(Counter(r["termination"] for r in rows)), "tool_calls": dict(tools),
        "executed_tool_calls_per_run": dict(sorted(lengths.items())), "horizon_bands": dict(horizons),
        "pre_action_gaps": dict(gaps), "real_tool_errors": errors, "linked_hypothesis_revisions": revisions,
        "consecutive_query_rewrites": rewritten_queries, "declared_dependency_depth": dict(sorted(depth_dist.items())),
        "declared_branch_merges": merges, "training_admission": dict(admissions)}
