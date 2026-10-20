"""An answer-blind controller with real, sequential image observations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from core.provenance import PIPELINE_VERSION, fingerprint
from core.structured import json_object
from core.views import materialize_view, valid_bbox
from labeling.prompt import build_prompt
from labeling.interaction import (
    ACTION_FIELDS, GAP_TYPES, REQUEST_FIELDS, TERMINAL_ACTIONS, observer_prompt, planner_prompt,
    resolution_prompt, visible_inputs,
)
from records.validator import (
    ALLOWED_ACTIONS, ALLOWED_RECOVERY_PATTERNS, ALLOWED_RECOVERY_SCOPES,
    ALLOWED_UNCERTAINTY_TYPES, BBOX_REQUIRED_ACTIONS, MAX_RECOVER_STEPS, MAX_TRACE_STEPS,
    DEFAULT_TRACE_VARIANT, RECOVERY_SCOPES_BY_PATTERN, STEP_FIELDS, _is_valid_zoom,
)

METADATA_FIELDS = ("sample_meta", "target_constraints", "sufficient_evidence_step",
                   "final_reasoning_summary", "answer_grounding",
                   "answer_resolution", "quality_flags")


def _request(action: str, region: str) -> dict[str, Any]:
    return {**dict(zip(ACTION_FIELDS, (action, region, None, None, None, None, None))),
            "rationale": "Initialize the question and available visual evidence.",
            "information_need": "Identify the question constraints." if action == "parse_question"
                                else "Locate visible candidate regions.",
            "gap_type": "language" if action == "parse_question" else "visual"}


def validate_request(request: dict[str, Any], trace: list[dict[str, Any]]) -> None:
    if set(request) != set(REQUEST_FIELDS):
        raise ValueError("Operation must contain exactly the request fields")
    if any(not isinstance(request[k], str) or not request[k].strip()
           for k in ("rationale", "information_need", "focus_region")):
        raise ValueError("A pre-action rationale and information need are required")
    action, bbox = request["action"], request["focus_bbox"]
    if not isinstance(request["gap_type"], str) or request["gap_type"] not in GAP_TYPES:
        raise ValueError("Unknown evidence-gap type")
    if action not in ALLOWED_ACTIONS or not isinstance(request["focus_region"], str):
        raise ValueError("Invalid operation/region")
    if trace and len(trace) >= 2 and action in {"parse_question", "global_scan"}:
        raise ValueError("Question parsing and coarse scanning occur only at the start")
    if action in BBOX_REQUIRED_ACTIONS and not valid_bbox(bbox):
        raise ValueError("Visual operations need a valid normalized bbox")
    if action in BBOX_REQUIRED_ACTIONS and request["gap_type"] not in {"visual", "mixed"}:
        raise ValueError("A language-only gap cannot justify another image operation")
    if action == "finish" and request["gap_type"] not in {"language", "none"}:
        raise ValueError("An unresolved visual gap needs inspection or abstention")
    if bbox is not None:
        if not valid_bbox(bbox):
            raise ValueError("Invalid normalized bbox")
    if action in {"parse_question", "global_scan", "finish"} and bbox is not None:
        raise ValueError("Question parsing/global scan must not have a bbox")
    if action == "zoom_in" and not _is_valid_zoom(bbox, trace[-1].get("focus_bbox") if trace else None):
        raise ValueError("Zoom must be a strict subregion of the previous view")
    previous = (trace[-1].get("lookback") or {}) if trace else {}
    if previous.get("verdict") == "recover" and action != "recover":
        raise ValueError("The previous decision authorized an immediate recovery")
    if action == "recover":
        if previous.get("verdict") != "recover" or \
                request["recover_to_step"] != previous.get("candidate_step"):
            raise ValueError("Recovery must follow the recorded authorization")
        if sum(s["action"] == "recover" for s in trace) >= MAX_RECOVER_STEPS:
            raise ValueError("Recovery budget exhausted")
        if request["recovery_pattern"] not in ALLOWED_RECOVERY_PATTERNS or \
                request["recover_scope"] not in ALLOWED_RECOVERY_SCOPES:
            raise ValueError("Invalid recovery pattern/scope")
        if request["recover_scope"] not in RECOVERY_SCOPES_BY_PATTERN[request["recovery_pattern"]]:
            raise ValueError("Recovery scope does not match its pattern")
    elif any(request[k] is not None for k in ("recover_to_step", "recovery_pattern", "recover_scope")):
        raise ValueError("Only recovery operations have recovery fields")
    if action == "flag_uncertainty":
        if request["gap_type"] not in {"visual", "mixed"}:
            raise ValueError("Visual abstention needs a visual evidence gap")
        if request["uncertainty_type"] not in ALLOWED_UNCERTAINTY_TYPES:
            raise ValueError("Stopping with uncertainty needs its type")
    elif request["uncertainty_type"] is not None:
        raise ValueError("uncertainty_type is only for terminal uncertainty")


def run_policy(sample: dict[str, Any], client: Any, instruction: Any,
               views_dir: Path, *, max_steps: int = MAX_TRACE_STEPS,
               global_side: int = 512, crop_side: int = 1536) -> tuple[dict, dict, str]:
    'Every next request depends only on earlier observations, never on GT.'
    base = build_prompt(sample, instruction)
    trace: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for index in range(max_steps):
        plan = None
        if index == 0:
            request = _request("parse_question", "question")
        elif index == 1:
            request = _request("global_scan", "coarse whole-image view")
        else:
            paths, context = visible_inputs(ledger)
            prompt = planner_prompt(base, trace, context, max_steps - index)
            if paths:
                result = client.generate_images(prompt, paths)
                if result.truncated:
                    raise ValueError("Planning response was truncated")
                text = result.text
            else:
                text = client.generate_text(prompt, max_tokens=1024)
            request = json_object(text)
            plan = {"prompt": prompt, "images": paths, "response": copy.deepcopy(request),
                    "prefix_fingerprint": fingerprint(trace), "context_limit": index}
            raw.append({"operation": "plan", "step": index + 1, "text": text})
        validate_request(request, trace)
        if index == max_steps - 1 and request["action"] not in TERMINAL_ACTIONS:
            raise ValueError("Policy reached its budget without a terminal operation")
        locked = {"step": index + 1, **{k: request[k] for k in ACTION_FIELDS}}
        view = None
        if request["action"] == "global_scan" or request["focus_bbox"] is not None:
            side = global_side if request["action"] == "global_scan" else crop_side
            view = materialize_view(sample["image_path"], views_dir / f"step_{index + 1:02d}.png",
                                    request["focus_bbox"], side)
        metadata = {k: v for k, v in (view or {}).items() if k != "path"}
        prompt = observer_prompt(base, trace, request, metadata, index + 1)
        if view:
            result = client.generate(prompt, view["path"])
            if result.truncated:
                raise ValueError("Observation response was truncated")
            text = result.text
        else:
            text = client.generate_text(prompt, max_tokens=2048)
        step = json_object(text)
        if set(step) != set(STEP_FIELDS) or any(step.get(k) != v for k, v in locked.items()):
            raise ValueError("Observer changed locked fields or violated the step schema")
        for field in ("observation", "reason", "decision"):
            if not isinstance(step[field], str) or not step[field].strip():
                raise ValueError(f"Empty {field} in observed step")
        observed = {"prompt": prompt, "images": [view["path"]] if view else [],
                    "response": copy.deepcopy(step), "prefix_fingerprint": fingerprint(trace),
                    "context_limit": index}
        trace.append(step)
        ledger.append({"step": copy.deepcopy(step), "view": view,
                       "request": copy.deepcopy(request), "planner": plan, "observer": observed,
                       "forced_initialization": index < 2})
        raw.append({"operation": "observe", "step": index + 1, "text": text})
        if request["action"] in TERMINAL_ACTIONS:
            break
    handoff_prompt = resolution_prompt(base, trace)
    text = client.generate_text(handoff_prompt)
    resolution = json_object(text)
    if set(resolution) != {"visual_anchor", "resolved_in_image", "external_hops"}:
        raise ValueError("Resolution planner must return exactly the handoff fields")
    handoff = {"prompt": handoff_prompt, "images": [], "response": copy.deepcopy(resolution),
               "prefix_fingerprint": fingerprint(trace), "context_limit": len(trace)}
    raw.append({"operation": "resolution", "text": text})
    prompt = (base + "\nCURRENT OPERATION: ASSEMBLE METADATA (text only).\n"
              "Immutable executed observations: " + json.dumps(trace, ensure_ascii=False) + "\n"
              f"Return exactly these metadata keys: {list(METADATA_FIELDS)}. "
              "Locked answer_resolution: " + json.dumps(resolution, ensure_ascii=False) + "\n"
              "Copy the locked answer_resolution exactly. "
              "No search_trace, final_answer, input echoes or new visual facts. "
              "External hops are plans, never completed searches.")
    text = client.generate_text(prompt)
    metadata = json_object(text)
    if set(metadata) != set(METADATA_FIELDS):
        raise ValueError("Assembler must return exactly the metadata fields")
    if metadata["answer_resolution"] != resolution:
        raise ValueError("Assembler changed the executed language handoff")
    sample_meta = metadata.get("sample_meta")
    if not isinstance(sample_meta, dict) or sample_meta.get("trace_variant") != sample.get(
            "trace_variant", DEFAULT_TRACE_VARIANT):
        raise ValueError("Assembler changed the requested trace variant")
    raw.append({"operation": "assemble", "text": text})
    annotation = {**metadata, "id": sample["id"], "image": sample["image_path"],
                  "image_size": sample["image_size"], "question": sample["question"],
                  "source_dataset": sample.get("source_dataset", ""),
                  "search_trace": trace, "final_answer": sample.get("answer", "").strip()}
    execution = {"version": PIPELINE_VERSION, "blind_to_answer": True,
                 "trace_origin": "executed_policy", "external_hops_executed": False,
                 "final_answer_origin": "held_out_dataset_label",
                 "base_prompt": base, "max_steps": max_steps,
                 "resolution": handoff,
                 "global_side": global_side, "crop_side": crop_side, "steps": ledger}
    return annotation, execution, json.dumps(raw, ensure_ascii=False)
