"""The action contract and exact contexts shared by collection and training."""
from __future__ import annotations

import json
from typing import Any
from core.views import view_identity

ACTION_FIELDS = ("action", "focus_region", "focus_bbox", "recover_to_step",
                 "recovery_pattern", "recover_scope", "uncertainty_type")
REQUEST_FIELDS = (*ACTION_FIELDS, "rationale", "information_need", "gap_type")
GAP_TYPES = frozenset({"visual", "language", "mixed", "none"})
TERMINAL_ACTIONS = frozenset({"finish", "answer_confirmation", "flag_uncertainty"})


def visible_inputs(ledger: list[dict]) -> tuple[list[str], list[dict]]:
    """Only already executed views; repeated identical views share one input."""
    paths, context, seen = [], [], set()
    for entry in ledger:
        view = entry.get("view")
        if not view:
            continue
        key = view_identity(view)
        if key in seen:
            continue
        seen.add(key)
        paths.append(view["path"])
        context.append({"image_index": len(paths), "observed_at_step": entry["step"]["step"],
                        "original_bbox": view["bbox"], "pixel_bbox": view["pixel_bbox"],
                        "width": view["width"], "height": view["height"]})
    return paths, context


def planner_prompt(base: str, trace: list[dict], context: list[dict], remaining: int) -> str:
    return (base + "\nCURRENT OPERATION: PLAN NEXT VIEW.\n"
            "Earlier immutable observations: " + json.dumps(trace, ensure_ascii=False) + "\n"
            "Previously observed image inputs: " + json.dumps(context) + "\n"
            f"{remaining} steps remain. Return exactly a JSON object with keys {list(REQUEST_FIELDS)}. "
            "Use the question, prior views and notes together. Rationale explains the choice "
            "from evidence available NOW; information_need names the unresolved question the "
            "action should address. These fields precede execution: never assert the chosen "
            "view's unseen result. Question constraints and valid deductions are language "
            "evidence. gap_type is visual/language/mixed/none. A language-only gap calls for "
            "a lookup/computation handoff, not more pixels. Compare hypotheses, recheck "
            "conflicting cues, expand context for relationships, and reconcile incomplete "
            "counts when justified. Re-reading the same pixels is not new visual information. "
            "Notes and tentative hypotheses are fallible. No future view or external result is available. "
            "Choose finish with null bbox when the anchor is established; no extra inspection "
            "is mandatory. Choose flag_uncertainty if the available evidence cannot resolve it. "
            "Before early visual abstention, inspect the relevant detail at native pixel resolution; "
            "blur in a downsampled preview does not establish missing detail in the source. "
            "The last operation must be terminal.")


def observer_prompt(base: str, trace: list[dict], request: dict[str, Any],
                    metadata: dict[str, Any], number: int) -> str:
    locked = {"step": number, **{k: request[k] for k in ACTION_FIELDS}}
    return (base + "\nCURRENT OPERATION: OBSERVE ONE STEP.\n"
            "Earlier immutable observations: " + json.dumps(trace, ensure_ascii=False) + "\n"
            "Locked fields: " + json.dumps(locked) + "\n"
            "Pre-action intent: " + json.dumps({k: request[k] for k in ("rationale", "information_need", "gap_type")}) + "\n"
            "Actual view metadata: " + json.dumps(metadata) + "\n"
            "Return this single complete step with exactly the Section 3 fields. "
            "Separate visible evidence, question constraints, deductions and remaining gaps. "
            "Do not describe an unseen future view. "
            + ("Use only the attached view for NEW visual facts; use prior notes for language reasoning."
               if metadata else "No image is attached. Use the question and prior notes only; add no new visual facts."))


def resolution_prompt(base: str, trace: list[dict]) -> str:
    return (base + "\nCURRENT OPERATION: PLAN LANGUAGE HANDOFF (text only).\n"
            "Immutable executed observations: " + json.dumps(trace, ensure_ascii=False) + "\n"
            "Use the question, recorded visual cues and valid language deductions together. "
            "Return exactly {\"visual_anchor\":string_or_null,\"resolved_in_image\":boolean,"
            "\"external_hops\":[{\"type\":\"lookup\",\"query\":string}|"
            "{\"type\":\"compute\",\"operation\":string}]}. This is an ordered PLAN, "
            "not executed external evidence. Queries use established entities; do not guess "
            "the withheld answer or a missing identity. Abstention permits a null anchor "
            "and empty hops. No search result or new image is available.")
