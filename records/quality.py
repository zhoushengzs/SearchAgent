"""The single fail-closed admission policy used by verification and export."""
from __future__ import annotations

import json
from typing import Any

from core.provenance import (
    PIPELINE_VERSION, VERIFICATION_VERSION, file_sha, independent_models, record_fingerprint,
    fingerprint,
)
from labeling.interaction import ACTION_FIELDS, observer_prompt, planner_prompt, resolution_prompt, visible_inputs
from records.store import is_successful_record
from records.validator import MAX_TRACE_STEPS, QUALITY_FLAG_FIELDS, check_ground_truth, validate_annotation
from sources.registry import is_evaluation_source
from sources.partition import derived_partition


def audit_fields(annotation: dict[str, Any], execution: dict | None = None) -> tuple[dict[str, str], dict[str, int]]:
    """Every visual assertion-bearing field, with its latest available step."""
    fields, limits = {}, {}
    for step in annotation["search_trace"]:
        number = step["step"]
        for name in ("observation", "reason", "decision"):
            key = f"step_{number}.{name}"
            fields[key], limits[key] = step[name], number
        key = f"step_{number}.belief_update"
        fields[key], limits[key] = json.dumps(step["belief_update"], ensure_ascii=False), number
        lb = step.get("lookback")
        if isinstance(lb, dict):
            key = f"step_{number}.lookback.rationale"
            fields[key], limits[key] = lb["rationale"], number
    fields["final_reasoning_summary"] = annotation["final_reasoning_summary"]
    limits["final_reasoning_summary"] = len(annotation["search_trace"])
    fields["answer_resolution"] = json.dumps(annotation["answer_resolution"], ensure_ascii=False)
    limits["answer_resolution"] = len(annotation["search_trace"])
    if execution:
        for entry in execution["steps"]:
            if entry.get("forced_initialization"):
                continue
            number = entry["step"]["step"]
            for name in ("rationale", "information_need"):
                key = f"step_{number}.pre_action.{name}"
                fields[key], limits[key] = entry["request"][name], number - 1
    return fields, limits


def execution_errors(record: dict[str, Any]) -> list[str]:
    execution = record.get("execution")
    if not isinstance(execution, dict) or execution.get("version") != PIPELINE_VERSION or \
            execution.get("blind_to_answer") is not True:
        return ["missing current answer-blind execution provenance; relabel this record"]
    ann = record.get("annotation")
    trace = ann.get("search_trace") if isinstance(ann, dict) else None
    ledger = execution.get("steps")
    if not isinstance(trace, list) or not isinstance(ledger, list) or len(ledger) != len(trace):
        return ["execution ledger does not cover the annotation"]
    errors = []
    try:
        from labeling.policy import validate_request
        base = execution.get("base_prompt")
        max_steps = execution.get("max_steps")
        if not isinstance(base, str) or type(max_steps) is not int or not 3 <= max_steps <= MAX_TRACE_STEPS:
            return ["missing exact interaction contract"]
        if len(trace) > max_steps:
            errors.append("trace exceeds its configured execution budget")
        if file_sha(record["image_path"]) != record.get("image_sha256"):
            errors.append("source image changed")
        for index, (step, entry) in enumerate(zip(trace, ledger)):
            if not isinstance(entry, dict) or entry.get("step") != step:
                errors.append(f"step {index + 1} differs from the execution log")
                continue
            view = entry.get("view")
            visual = step.get("action") == "global_scan" or step.get("focus_bbox") is not None
            if visual:
                if not isinstance(view, dict) or view.get("bbox") != step.get("focus_bbox"):
                    errors.append(f"step {index + 1} has no matching executed view")
                elif file_sha(view["path"]) != view.get("sha256"):
                    errors.append(f"step {index + 1} view changed")
            elif view is not None:
                errors.append(f"text-only step {index + 1} unexpectedly has a view")
            request, plan, observer = entry.get("request"), entry.get("planner"), entry.get("observer")
            prefix = trace[:index]
            validate_request(request, prefix)
            if any(request[k] != step[k] for k in ACTION_FIELDS):
                errors.append(f"step {index + 1} request was changed after execution")
            paths, context = visible_inputs(ledger[:index])
            forced = index < 2
            if entry.get("forced_initialization") is not forced or (forced and plan is not None):
                errors.append(f"step {index + 1} initialization provenance is invalid")
            if not forced and (not isinstance(plan, dict)
                    or plan.get("prompt") != planner_prompt(base, prefix, context, max_steps - index)
                    or plan.get("images") != paths or plan.get("response") != request
                    or plan.get("context_limit") != index or plan.get("prefix_fingerprint") != fingerprint(prefix)):
                errors.append(f"step {index + 1} planner input has future/mismatched context")
            metadata = {k: v for k, v in (view or {}).items() if k != "path"}
            if (not isinstance(observer, dict)
                    or observer.get("prompt") != observer_prompt(base, prefix, request, metadata, index + 1)
                    or observer.get("images") != ([view["path"]] if view else [])
                    or observer.get("response") != step or observer.get("context_limit") != index
                    or observer.get("prefix_fingerprint") != fingerprint(prefix)):
                errors.append(f"step {index + 1} observer input/target does not match execution")
        handoff = execution.get("resolution")
        if (not isinstance(handoff, dict) or handoff.get("prompt") != resolution_prompt(base, trace)
                or handoff.get("images") != [] or handoff.get("response") != ann["answer_resolution"]
                or handoff.get("context_limit") != len(trace)
                or handoff.get("prefix_fingerprint") != fingerprint(trace)):
            errors.append("language handoff does not match its executed prefix")
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        errors.append(f"cannot validate execution images: {exc}")
    return errors


def training_rejection_reason(record: dict[str, Any], *, allow_abstention: bool = False,
                              allow_evaluation: bool = False) -> str | None:
    if not is_successful_record(record) or not isinstance(record.get("annotation"), dict):
        return "labeling failed / no annotation"
    partition = derived_partition(record) if record.get("trajectory_derivation") is not None else None
    if record.get("trajectory_derivation") is not None and partition is None:
        return "benchmark-derived input or partition metadata invalid"
    if not allow_evaluation and (record.get("data_usage", "train") != "train" or
            (is_evaluation_source(record.get("source_dataset", ""), record.get("dataset_provenance"))
             and partition != "train")):
        return "held-out dataset: evaluation only"
    ann = record["annotation"]
    flags = ann.get("quality_flags")
    trace = ann.get("search_trace")
    abstention = (allow_abstention and isinstance(trace, list) and bool(trace)
                  and isinstance(trace[-1], dict) and trace[-1].get("action") == "flag_uncertainty")
    if record.get("truncated") or (record.get("needs_human_review") and not abstention):
        return "flagged for review / truncated"
    if not isinstance(flags, dict) or any(type(flags.get(k)) is not bool for k in QUALITY_FLAG_FIELDS):
        return "quality flag present or missing"
    if (abstention and (flags["possible_hallucination"] or flags["needs_human_review"] is not True)) or (
            not abstention and any(flags[k] for k in QUALITY_FLAG_FIELDS)):
        return "quality flag present or missing"
    try:
        if (ann.get("sample_meta") or {}).get("trace_variant") != record.get("trace_variant"):
            return "annotation trace variant differs from generation configuration"
        errors, _ = validate_annotation(ann)
        errors += check_ground_truth(ann, record.get("answer", ""))
        if errors or not str(record.get("answer", "")).strip():
            return "annotation invalid / missing or inconsistent dataset answer"
        if execution_errors(record):
            return "execution provenance missing or changed"
        v = record.get("verification")
        if not isinstance(v, dict) or v.get("version") != VERIFICATION_VERSION:
            return "current visual verification required"
        if v.get("input_fingerprint") != record_fingerprint(record):
            return "verification is stale"
        if not independent_models(str(record.get("model", "")), str(v.get("judge_model", ""))):
            return "a judge from a different recognized model family is required"
        g = v.get("grounding")
        if not isinstance(g, dict) or g.get("pass") is not True or g.get("complete") is not True:
            return "visual verification failed / incomplete / unknown"
        if g.get("mode") != ("abstention" if abstention else "resolved"):
            return "verification mode differs from the supervised behavior"
        claims = g.get("claims")
        if not isinstance(claims, list) or not claims or any(
                not isinstance(c, dict) or c.get("status") != "supported" for c in claims):
            return "not all visual facts were supported"
        expected_fields, _ = audit_fields(ann, record["execution"])
        if any(not isinstance(c.get("text"), str) or c["text"] not in expected_fields.get(c.get("field_id"), "")
               for c in claims if c.get("role") == "visual_fact"):
            return "visual claims differ from original fields"
        covered = g.get("covered_fields")
        if not isinstance(covered, list) or len(covered) != len(expected_fields) or set(covered) != set(expected_fields):
            return "visual field coverage incomplete"
        observed_fields = {c.get("field_id") for c in claims if c.get("role") == "visual_fact"}
        required_observations = {f"step_{entry['step']['step']}.observation"
                                 for entry in record["execution"]["steps"] if entry["view"]}
        if not required_observations <= observed_fields:
            return "an observation was omitted from the claim audit"
        language_fields = [c.get("field_id") for c in claims if c.get("role") == "language_reasoning"]
        if len(language_fields) != len(expected_fields) or set(language_fields) != set(expected_fields):
            return "language reasoning coverage incomplete"
        roles = {c.get("role") for c in claims}
        required_roles = {"abstention"} if abstention else {"visual_anchor", "resolution_plan"}
        if not required_roles <= roles or (not abstention and ann["answer_resolution"]["resolved_in_image"]
                                            and "in_image_answer" not in roles):
            return "mandatory anchor / in-image answer check missing"
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        return f"malformed record: {type(exc).__name__}"
    return None


def is_clean(record: dict[str, Any]) -> bool:
    return training_rejection_reason(record) is None
