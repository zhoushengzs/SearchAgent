'Validate immutable executed annotation JSON; no post-hoc repairs.'
from __future__ import annotations

import re
from typing import Any

from core.utils import get_logger
from core.views import valid_bbox
from labeling.interaction import TERMINAL_ACTIONS

logger = get_logger(__name__)

STEP_FIELDS = (
    "step", "action", "focus_region", "focus_bbox", "observation", "reason",
    "decision", "recover_to_step", "recovery_pattern", "recover_scope",
    "uncertainty_type", "lookback", "belief_update",
)

# --------------------------------------------------------------------------- #
# Schema definition
# --------------------------------------------------------------------------- #
REQUIRED_FIELDS: tuple[str, ...] = (
    "sample_meta",
    "target_constraints",
    "search_trace",
    "sufficient_evidence_step",
    "final_reasoning_summary",
    "answer_grounding",
    "answer_resolution",
    "final_answer",
    "quality_flags",
)

QUALITY_FLAG_FIELDS: tuple[str, ...] = (
    "possible_hallucination",
    "anchor_not_visually_verifiable",
    "needs_human_review",
)

# answer_resolution.external_hops (instruction 6).
ALLOWED_HOP_TYPES: frozenset[str] = frozenset({"lookup", "compute"})
MAX_LOOKUP_QUERY_WORDS = 12
# Answers shorter than this are too generic ("2", "no") to call a leak.
MIN_LEAK_ANSWER_CHARS = 3

ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {
        "parse_question",
        "global_scan",
        "inspect_region",
        "zoom_in",
        "inspect_candidate",
        "compare_candidates",
        "reject_candidate",
        "confirm_candidate",
        "recover",
        "answer_confirmation",
        "finish",
        "flag_uncertainty",
    }
)

ALLOWED_UNCERTAINTY_TYPES: frozenset[str] = frozenset(
    {
        "similar_candidate_conflict",
        "missed_region_risk",
        "spatial_constraint_conflict",
        "relation_constraint_unclear",
        "occlusion_or_blur",
        "insufficient_context",
        "counting_uncertainty",
        "answer_evidence_gap",
    }
)

ALLOWED_RECOVERY_PATTERNS: frozenset[str] = frozenset(
    {
        "reject_reverification",
        "missed_region_sweep",
        "constraint_conflict_recheck",
        "attribute_reread",
        "count_reconciliation",
        "context_expansion",
    }
)

ALLOWED_RECOVERY_SCOPES: frozenset[str] = frozenset(
    {
        "single_look",
        "reinspect",
        "resweep",
    }
)

RECOVERY_SCOPES_BY_PATTERN = {
    "reject_reverification": frozenset({"single_look", "reinspect"}),
    "missed_region_sweep": frozenset({"resweep"}),
    "constraint_conflict_recheck": frozenset({"single_look", "reinspect"}),
    "attribute_reread": frozenset({"single_look"}),
    "count_reconciliation": frozenset({"reinspect"}),
    "context_expansion": frozenset({"single_look"}),
}

ALLOWED_ANSWER_FLIP_RISKS: frozenset[str] = frozenset({"high", "medium", "low"})

ALLOWED_VERDICTS: frozenset[str] = frozenset({"recover", "skip"})

ALLOWED_TRACE_VARIANTS: frozenset[str] = frozenset({"minimal", "default", "thorough"})
DEFAULT_TRACE_VARIANT = "default"

# Actions on which the instruction (Section 4.1) REQUIRES a lookback object.
LOOKBACK_REQUIRED_ACTIONS: frozenset[str] = frozenset(
    {"reject_candidate", "confirm_candidate", "answer_confirmation", "finish"}
)

# Per-action focus_bbox requirements (instruction 5.1). flag_uncertainty is
# deliberately in neither set (bbox optional).
BBOX_FORBIDDEN_ACTIONS: frozenset[str] = frozenset(
    {"parse_question", "global_scan", "finish"}
)
BBOX_REQUIRED_ACTIONS: frozenset[str] = frozenset(
    {
        "inspect_region",
        "zoom_in",
        "inspect_candidate",
        "compare_candidates",
        "reject_candidate",
        "confirm_candidate",
        "recover",
        "answer_confirmation",
    }
)

# Valid targets for recover_to_step (instruction 4.3/4.4): real decision
# points only — never parse_question. `compare_candidates` counts: a side-by-side
# discrimination is a decision point a later step can legitimately return to.
BACKTRACK_TARGET_ACTIONS: frozenset[str] = frozenset(
    {"global_scan", "inspect_region", "zoom_in", "inspect_candidate",
     "reject_candidate", "compare_candidates"}
)

MAX_RECOVER_STEPS = 3   # hard cap on recover actions in a trace
MAX_TRACE_STEPS = 12    # hard cap on total steps (instruction 8, Length)

# Step-count band per declared difficulty (instruction 8, Length). Outside the
# band is a warning, not an error: it usually means the difficulty label is
# wrong rather than the trace.
DIFFICULTY_STEP_BANDS: dict[str, tuple[int, int]] = {
    "easy": (3, 5),
    "medium": (5, 8),
    "hard": (6, 12),
}


def expected_verdict(risk: Any, residual: Any, variant: str = "default") -> str:
    'Instruction 4.2 decision rule — single source of truth (also used by'
    if risk == "high":
        return "recover"
    if variant == "thorough" and residual:
        return "recover"
    if variant == "minimal":
        return "skip"
    if risk == "medium" and residual:
        return "recover"
    return "skip"


def _mentions(text: Any, answer: str) -> bool:
    """Whole-token, case-insensitive containment of ``answer`` in ``text``."""
    a = answer.strip().rstrip(".").lower()
    if len(a) < MIN_LEAK_ANSWER_CHARS or not isinstance(text, str):
        return False
    return re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text.lower()) is not None


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _valid_bbox(bbox: Any) -> bool:
    return valid_bbox(bbox)


def _is_valid_zoom(bbox: list[float], prev_bbox: list[float] | None) -> bool:
    """Instruction 5.1: a zoom box is contained in the previous step's box
    with at most half its area."""
    if not _valid_bbox(bbox) or not _valid_bbox(prev_bbox):
        return False
    x0, y0, x1, y1 = (float(v) for v in bbox)
    px0, py0, px1, py1 = prev_bbox
    eps = 1e-6
    inside = (x0 >= px0 - eps and y0 >= py0 - eps
              and x1 <= px1 + eps and y1 <= py1 + eps)
    half = (x1 - x0) * (y1 - y0) <= 0.5 * (px1 - px0) * (py1 - py0) + eps
    return inside and half


def check_ground_truth(obj: Any, expected_answer: str) -> list[str]:
    """The instruction requires ``final_answer`` to exactly equal the given
    ground truth. Lives here (not in validate_annotation) because only the
    caller knows the expected answer."""
    gt = str(expected_answer or "").strip()
    got = str(obj.get("final_answer", "")).strip() if isinstance(obj, dict) else ""
    if gt and got != gt:
        return [f"final_answer {got!r} does not exactly match ground-truth {gt!r}"]
    return []


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate_recover_step(
    i: int, step: dict[str, Any], step_num: Any, trace: list[Any]
) -> list[str]:
    """All recover-step checks: authorisation, backtrack target, pattern, scope."""
    errors: list[str] = []

    # A recover step must be authorised by the immediately preceding step's
    # lookback (instruction 8, Look-back).
    prev = trace[i - 1] if i > 0 else None
    prev_lb = prev.get("lookback") if isinstance(prev, dict) else None
    if not (isinstance(prev_lb, dict) and prev_lb.get("verdict") == "recover"):
        errors.append(
            f"search_trace[{i}]: recover step not authorised by a "
            "preceding lookback with verdict='recover'"
        )

    recover_to = step.get("recover_to_step")
    if not isinstance(recover_to, int) or (
        isinstance(step_num, int) and recover_to >= step_num
    ):
        errors.append(f"search_trace[{i}]: recover_to_step must be int < step")
    else:
        # Backtrack target must be a real decision point, never
        # parse_question (instruction 4.3).
        target = trace[recover_to - 1] if 1 <= recover_to <= len(trace) else None
        target_action = target.get("action") if isinstance(target, dict) else None
        if target_action not in BACKTRACK_TARGET_ACTIONS:
            errors.append(
                f"search_trace[{i}]: recover_to_step={recover_to} points "
                f"at action '{target_action}', not a valid backtrack "
                f"target {sorted(BACKTRACK_TARGET_ACTIONS)}"
            )

    pattern = step.get("recovery_pattern")
    if pattern not in ALLOWED_RECOVERY_PATTERNS:
        errors.append(
            f"search_trace[{i}].recovery_pattern '{pattern}' invalid "
            "(required on recover steps)"
        )
    scope = step.get("recover_scope")
    if scope not in ALLOWED_RECOVERY_SCOPES:
        errors.append(
            f"search_trace[{i}].recover_scope '{scope}' invalid "
            "(required on recover steps)"
        )
    elif pattern in RECOVERY_SCOPES_BY_PATTERN and scope not in RECOVERY_SCOPES_BY_PATTERN[pattern]:
        errors.append(f"search_trace[{i}]: recovery scope does not match its pattern")
    return errors


def _validate_lookback(
    i: int, step_num: Any, lookback: Any, trace: list[Any], variant: str = "default"
) -> tuple[list[str], list[str]]:
    """Lookback-object checks: enums, decision rule, cost bookkeeping, wiring."""
    if not isinstance(lookback, dict):
        return [f"search_trace[{i}].lookback must be an object or null"], []

    errors: list[str] = []
    warnings: list[str] = []
    risk = lookback.get("answer_flip_risk")
    if risk not in ALLOWED_ANSWER_FLIP_RISKS:
        errors.append(f"search_trace[{i}].lookback.answer_flip_risk '{risk}' not allowed")
    verdict = lookback.get("verdict")
    if verdict not in ALLOWED_VERDICTS:
        errors.append(f"search_trace[{i}].lookback.verdict '{verdict}' not allowed")

    # Only exceptions established by the preceding log allow a skip warning.
    residual = lookback.get("residual_constraints")
    if verdict in ALLOWED_VERDICTS and risk in ALLOWED_ANSWER_FLIP_RISKS \
            and isinstance(residual, list) \
            and verdict != expected_verdict(risk, residual, variant):
        prior_recovers = [s for s in trace[:i] if s.get("action") == "recover"]
        exception = (len(prior_recovers) >= MAX_RECOVER_STEPS or
                     any(s.get("recover_to_step") == lookback.get("candidate_step") for s in prior_recovers))
        target = warnings if exception and verdict == "skip" else errors
        target.append(
            f"search_trace[{i}].lookback: verdict '{verdict}' contradicts "
            f"decision rule (risk={risk}, residual={len(residual)}) — check "
            "rationale for a stated exception"
        )

    # 'medium' is defined by an unchecked separating constraint, so medium with
    # an empty residual list is self-contradictory (instruction 4.2).
    if risk == "medium" and isinstance(residual, list) and not residual:
        errors.append(
            f"search_trace[{i}].lookback: answer_flip_risk='medium' with an "
            "empty residual_constraints is self-contradictory — name the "
            "unchecked constraint or downgrade the risk to 'low'"
        )

    # backtrack_distance bookkeeping.
    cand = lookback.get("candidate_step")
    cost = lookback.get("cost_estimate")
    if not isinstance(residual, list) or any(not _nonempty_str(v) for v in residual):
        errors.append(f"search_trace[{i}].lookback.residual_constraints must be strings")
    if not _nonempty_str(lookback.get("rationale")):
        errors.append(f"search_trace[{i}].lookback.rationale is required")
    if cand is None:
        if trace[i].get("action") not in {"answer_confirmation", "finish"}:
            errors.append(f"search_trace[{i}]: null candidate_step only allowed on successful termination")
    elif type(cand) is not int or type(step_num) is not int or not (1 <= cand < step_num and cand <= i) or \
            trace[cand - 1].get("action") not in BACKTRACK_TARGET_ACTIONS:
        errors.append(f"search_trace[{i}]: candidate_step must reference an earlier decision")
    if not isinstance(cost, dict) or any(type(cost.get(k)) is not int or cost[k] < 0
            for k in ("extra_steps", "attention_units", "backtrack_distance")) or \
            type(cost.get("requires_rezoom")) is not bool:
        errors.append(f"search_trace[{i}]: invalid lookback cost_estimate")
    elif cand is None and (any(cost[k] for k in ("extra_steps", "attention_units", "backtrack_distance"))
                           or cost["requires_rezoom"]):
        errors.append(f"search_trace[{i}]: null candidate_step must have zero cost")
    if isinstance(cand, int) and isinstance(step_num, int) and isinstance(cost, dict):
        bd = cost.get("backtrack_distance")
        if isinstance(bd, int) and bd != step_num - cand:
            errors.append(
                f"search_trace[{i}].lookback.cost_estimate.backtrack_distance="
                f"{bd} != step-candidate_step={step_num - cand}"
            )

    # verdict=recover must be immediately followed by a recover step pointing
    # at the authorising candidate_step.
    if verdict == "recover":
        nxt = trace[i + 1] if i + 1 < len(trace) else None
        if not (isinstance(nxt, dict) and nxt.get("action") == "recover"):
            errors.append(
                f"search_trace[{i}]: lookback.verdict='recover' "
                "not followed by an action='recover' step"
            )
        elif isinstance(cand, int) and nxt.get("recover_to_step") != cand:
            errors.append(
                f"search_trace[{i + 1}].recover_to_step="
                f"{nxt.get('recover_to_step')} != authorising "
                f"candidate_step={cand}"
            )
    return errors, warnings


def _validate_step(
    i: int, step: dict[str, Any], trace: list[Any], variant: str = "default"
) -> tuple[list[str], list[str]]:
    """All checks for one trace step (assumes ``step`` is a dict)."""
    errors: list[str] = []
    warnings: list[str] = []
    for field in STEP_FIELDS:
        if field not in step:
            errors.append(f"search_trace[{i}]: missing {field}")
    for field in ("focus_region", "observation", "reason", "decision"):
        if not _nonempty_str(step.get(field)):
            errors.append(f"search_trace[{i}].{field} must be a non-empty string")
    step_num = step.get("step")
    if type(step_num) is not int:
        errors.append(f"search_trace[{i}].step must be an integer, got {step_num!r}")

    action = step.get("action")
    if action is None:
        errors.append(f"search_trace[{i}].action is missing")
    elif action not in ALLOWED_ACTIONS:
        errors.append(f"search_trace[{i}].action '{action}' not allowed")

    if action == "flag_uncertainty" \
            and step.get("uncertainty_type") not in ALLOWED_UNCERTAINTY_TYPES:
        errors.append(
            f"search_trace[{i}]: uncertainty_type "
            f"'{step.get('uncertainty_type')}' invalid (required on "
            "flag_uncertainty steps)"
        )

    bbox = step.get("focus_bbox")
    if bbox is not None:
        if not _valid_bbox(bbox):
            errors.append(f"search_trace[{i}].focus_bbox invalid: {bbox!r}")

    # Per-action bbox requirements (instruction 5.1): text-only steps must not
    # carry a box; visual steps must.
    if action in BBOX_FORBIDDEN_ACTIONS and bbox is not None:
        errors.append(f"search_trace[{i}] ({action}): focus_bbox must be null")
    elif action in BBOX_REQUIRED_ACTIONS and bbox is None:
        errors.append(f"search_trace[{i}] ({action}): focus_bbox is required")

    if action == "recover":
        errors += _validate_recover_step(i, step, step_num, trace)
    elif any(step.get(k) is not None for k in ("recover_to_step", "recovery_pattern", "recover_scope")):
        errors.append(f"search_trace[{i}]: recovery fields only allowed on recover")
    if action == "zoom_in" and not _is_valid_zoom(bbox, trace[i - 1].get("focus_bbox") if i else None):
        errors.append(f"search_trace[{i}]: zoom must be contained in the previous box and at most half its area")
    if action != "flag_uncertainty" and step.get("uncertainty_type") is not None:
        errors.append(f"search_trace[{i}]: uncertainty_type only allowed on flag_uncertainty")

    # lookback object: REQUIRED on the decision-point actions.
    lookback = step.get("lookback")
    if action in LOOKBACK_REQUIRED_ACTIONS and not isinstance(lookback, dict):
        errors.append(
            f"search_trace[{i}] ({action}): lookback object is required "
            "on reject/confirm/answer_confirmation steps"
        )
    if lookback is not None:
        lb_errors, lb_warnings = _validate_lookback(i, step_num, lookback, trace, variant)
        errors += lb_errors
        warnings += lb_warnings
    update = step.get("belief_update")
    fields = {"status", "hypothesis", "revision_of_step", "evidence_steps",
              "open_visual_questions", "open_language_questions"}
    if not isinstance(update, dict) or set(update) != fields:
        errors.append(f"search_trace[{i}]: belief_update must follow the evidence-state schema")
    else:
        if not isinstance(update["status"], str) or update["status"] not in {
                "progress", "unchanged", "conflict", "resolved", "insufficient"}:
            errors.append(f"search_trace[{i}]: invalid evidence-state status")
        if update["hypothesis"] is not None and not _nonempty_str(update["hypothesis"]):
            errors.append(f"search_trace[{i}]: hypothesis must be a string or null")
        refs = update["evidence_steps"]
        if not isinstance(refs, list) or any(type(n) is not int or not 1 <= n <= i + 1 for n in refs):
            errors.append(f"search_trace[{i}]: evidence-state references unavailable steps")
        for name in ("open_visual_questions", "open_language_questions"):
            if not isinstance(update[name], list) or any(not _nonempty_str(x) for x in update[name]):
                errors.append(f"search_trace[{i}]: {name} must be a string array")
        revision = update["revision_of_step"]
        if revision is not None and (type(revision) is not int or not 1 <= revision <= i
                or not isinstance(trace[revision - 1].get("belief_update"), dict)
                or not trace[revision - 1]["belief_update"].get("hypothesis")):
            errors.append(f"search_trace[{i}]: revision must link an earlier recorded hypothesis")
        if action in {"finish", "answer_confirmation"} and (update["status"] != "resolved"
                or update["open_visual_questions"]):
            errors.append(f"search_trace[{i}]: successful termination requires resolved visual evidence")
        if action == "flag_uncertainty" and (update["status"] != "insufficient"
                or not update["open_visual_questions"]):
            errors.append(f"search_trace[{i}]: abstention requires explicit insufficient visual evidence")
    return errors, warnings


def validate_annotation(obj: Any) -> tuple[list[str], list[str]]:
    """Validate a parsed annotation object (NEW schema). Returns ``(errors, warnings)``."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(obj, dict):
        return ["annotation is not an object"], warnings

    # 1) Required fields.
    for field in REQUIRED_FIELDS:
        if field not in obj:
            errors.append(f"missing required field: {field}")

    # Reject unsafe shapes before deeper indexed/enum checks. Validation must
    # return errors, never crash on malformed model output.
    trace_shape = obj.get("search_trace")
    if not isinstance(trace_shape, list) or not trace_shape or any(
            not isinstance(s, dict) or type(s.get("step")) is not int or
            not isinstance(s.get("action"), str) for s in trace_shape):
        return errors + ["search_trace must contain objects with integer steps and string actions"], warnings
    meta_shape = obj.get("sample_meta")
    if not isinstance(meta_shape, dict):
        return errors + ["sample_meta must be an object"], warnings
    for field, choices in {
        "trace_variant": ALLOWED_TRACE_VARIANTS,
        "difficulty": set(DIFFICULTY_STEP_BANDS),
        "question_type": {"existence", "attribute", "relation", "counting", "ocr", "spatial", "other"},
    }.items():
        if not isinstance(meta_shape.get(field), str) or meta_shape[field] not in choices:
            return errors + [f"sample_meta.{field} is missing or invalid"], warnings
    if type(meta_shape.get("requires_zoom")) is not bool or \
            type(meta_shape.get("n_competing_candidates")) is not int or meta_shape["n_competing_candidates"] < 0:
        errors.append("sample_meta requires boolean requires_zoom and nonnegative candidate count")
    for s in trace_shape:
        lb = s.get("lookback")
        if isinstance(lb, dict) and any(lb.get(k) is not None and not isinstance(lb[k], str)
                                      for k in ("verdict", "answer_flip_risk")):
            return errors + ["lookback verdict/risk must be strings"], warnings
        for key in ("uncertainty_type", "recovery_pattern", "recover_scope"):
            if s.get(key) is not None and not isinstance(s[key], str):
                return errors + [f"{key} must be a string or null"], warnings
    if not _nonempty_str(obj.get("final_reasoning_summary")):
        errors.append("final_reasoning_summary must be a non-empty string")

    # 2) sample_meta shape.
    meta = obj.get("sample_meta")
    if meta is not None:
        if not isinstance(meta, dict):
            errors.append("sample_meta must be an object")
        elif "trace_variant" in meta and meta["trace_variant"] not in ALLOWED_TRACE_VARIANTS:
            warnings.append(f"sample_meta.trace_variant '{meta['trace_variant']}' not allowed")

    # 3) image_size.
    img_size = obj.get("image_size")
    if img_size is not None:
        if not isinstance(img_size, dict):
            errors.append("image_size must be an object")
        elif any(type(img_size.get(k)) is not int or img_size[k] < 1 for k in ("width", "height")):
            errors.append("image_size must have positive integer width and height")
    else:
        errors.append("image_size is required")

    # 4) target_constraints shape.
    tc = obj.get("target_constraints")
    if tc is not None and not isinstance(tc, dict):
        errors.append("target_constraints must be an object")
    elif isinstance(tc, dict):
        if not _nonempty_str(tc.get("target_object")) or any(
                not isinstance(tc.get(k), list) or any(not _nonempty_str(x) for x in tc[k])
                for k in ("target_attributes", "spatial_constraints", "relation_constraints")):
            errors.append("target_constraints requires target_object and string arrays")
        else:
            constraints = {tc["target_object"], *tc["target_attributes"], *tc["spatial_constraints"], *tc["relation_constraints"]}
            for step in trace_shape:
                residual = (step.get("lookback") or {}).get("residual_constraints", []) \
                    if isinstance(step.get("lookback"), dict) else []
                if isinstance(residual, list) and any(not isinstance(v, str) or v not in constraints for v in residual):
                    errors.append(f"step {step['step']}: residual constraints must occur in target_constraints")

    # 5) search_trace: per-step checks (see _validate_step), plus trace-level
    #    invariants: consecutive numbering, recover budget, uncertain ending.
    trace = obj.get("search_trace")
    n_steps = 0
    if not isinstance(trace, list) or not trace:
        errors.append("search_trace must be a non-empty list")
    else:
        n_steps = len(trace)
        steps: list[int] = []
        recover_count = 0
        for i, step in enumerate(trace):
            if not isinstance(step, dict):
                errors.append(f"search_trace[{i}] is not an object")
                continue
            if isinstance(step.get("step"), int):
                steps.append(step["step"])
            if step.get("action") == "recover":
                recover_count += 1
            step_errors, step_warnings = _validate_step(i, step, trace,
                meta.get("trace_variant", "default") if isinstance(meta, dict) else "default")
            errors += step_errors
            warnings += step_warnings

        expected = list(range(1, len(steps) + 1))
        if steps and steps != expected:
            errors.append(f"search_trace steps not consecutive from 1: {steps}")

        if recover_count > MAX_RECOVER_STEPS:
            errors.append(f"trace has {recover_count} recover actions, exceeds max {MAX_RECOVER_STEPS}")

        # Length (instruction 8): the hard cap is absolute; the per-difficulty
        # band is advisory because a mismatch usually means the difficulty label
        # is wrong rather than the trace.
        if n_steps > MAX_TRACE_STEPS:
            errors.append(
                f"trace has {n_steps} steps, exceeds the hard cap of {MAX_TRACE_STEPS}"
            )
        declared = (meta or {}).get("difficulty") if isinstance(meta, dict) else None
        band = DIFFICULTY_STEP_BANDS.get(declared)
        if band and not (band[0] <= n_steps <= band[1]):
            warnings.append(
                f"trace has {n_steps} steps but difficulty='{declared}' expects "
                f"{band[0]}-{band[1]} — check whether the difficulty label is right"
            )

        # A trace that gives up must route to human review.
        last = trace[-1] if isinstance(trace[-1], dict) else {}
        if len(trace) < 3 or trace[0].get("action") != "parse_question" or trace[1].get("action") != "global_scan":
            errors.append("trace must start with parse_question and global_scan")
        if last.get("action") not in TERMINAL_ACTIONS:
            errors.append("trace must end with finish, answer_confirmation or flag_uncertainty")
        if any(s["action"] in TERMINAL_ACTIONS for s in trace[:-1]):
            errors.append("terminal actions may only occur at the end")
        if last.get("action") == "flag_uncertainty":
            flags_obj = obj.get("quality_flags")
            if isinstance(flags_obj, dict) and not flags_obj.get("needs_human_review"):
                errors.append(
                    "trace ends with flag_uncertainty but "
                    "quality_flags.needs_human_review is false"
                )

    # 6) sufficient_evidence_step valid and consistent with evidence_step.
    suff = obj.get("sufficient_evidence_step")
    abstention = trace_shape[-1]["action"] == "flag_uncertainty"
    if abstention and suff is not None:
        errors.append("an unresolved trace cannot declare a sufficient evidence step")
    if not (abstention and suff is None) and (type(suff) is not int or not 1 <= suff <= n_steps):
        errors.append(f"sufficient_evidence_step must be 1 <= int <= n_steps, got {suff}")

    # 7) answer_grounding: object with final_target_bbox (array of bboxes) and evidence_step.
    grounding = obj.get("answer_grounding")
    if not isinstance(grounding, dict):
        errors.append("answer_grounding must be an object")
    else:
        bbox_list = grounding.get("final_target_bbox")
        if not isinstance(bbox_list, list) or not bbox_list:
            errors.append("answer_grounding.final_target_bbox must be non-empty")
        if bbox_list is not None:
            if not isinstance(bbox_list, list):
                errors.append(f"answer_grounding.final_target_bbox must be a list, got {type(bbox_list).__name__}")
            else:
                for i, bbox in enumerate(bbox_list):
                    if not _valid_bbox(bbox):
                        errors.append(f"answer_grounding.final_target_bbox[{i}] invalid: {bbox!r}")
                    elif bbox not in [([0, 0, 1, 1] if s.get("action") == "global_scan" else s.get("focus_bbox"))
                                      for s in trace if isinstance(s, dict)]:
                        errors.append(f"answer_grounding.final_target_bbox[{i}] was never observed")
                if len({tuple(b) for b in bbox_list if _valid_bbox(b)}) != len(bbox_list):
                    errors.append("answer_grounding contains invalid or duplicate boxes")

        ev_step = grounding.get("evidence_step")
        if ev_step is None:
            errors.append("answer_grounding.evidence_step is required")
        if ev_step is not None:
            if type(ev_step) is not int or ev_step < 1 or ev_step > n_steps:
                errors.append(f"answer_grounding.evidence_step must be 1 <= int <= n_steps, got {ev_step}")
            elif type(suff) is int and suff > ev_step:
                errors.append(
                    f"sufficient_evidence_step={suff} > evidence_step={ev_step} "
                    "(the answer cannot be fixed after the step that grounds it)"
                )
            elif isinstance(trace, list) and isinstance(trace[ev_step - 1], dict) and \
                    trace[ev_step - 1].get("action") != "global_scan" and \
                    not _valid_bbox(trace[ev_step - 1].get("focus_bbox")):
                errors.append("evidence_step must refer to an observed image view")
            if type(ev_step) is int and 1 <= ev_step <= n_steps and isinstance(bbox_list, list):
                prior_boxes = [([0, 0, 1, 1] if s.get("action") == "global_scan" else s.get("focus_bbox"))
                               for s in trace[:ev_step]]
                if any(b not in prior_boxes for b in bbox_list):
                    errors.append("final evidence includes a box not observed by evidence_step")

    # 7b) answer_resolution (instruction 6, 8 Resolution).
    errors_r, warnings_r = _validate_resolution(obj, trace if isinstance(trace, list) else [])
    errors += errors_r
    warnings += warnings_r

    # 8) final_answer non-empty.
    if not _nonempty_str(obj.get("final_answer")):
        errors.append("final_answer must be a non-empty string")

    # 9) quality_flags: object with the 3 boolean fields.
    flags = obj.get("quality_flags")
    if not isinstance(flags, dict):
        errors.append("quality_flags must be an object")
    else:
        for f in QUALITY_FLAG_FIELDS:
            if f not in flags:
                errors.append(f"quality_flags missing '{f}'")
            elif not isinstance(flags[f], bool):
                errors.append(f"quality_flags.{f} is not a bool: {flags[f]!r}")

    return errors, warnings


def _validate_resolution(
    obj: dict[str, Any], trace: list[Any]
) -> tuple[list[str], list[str]]:
    """answer_resolution shape, hop wiring, and the two answer-leak checks."""
    errors: list[str] = []
    warnings: list[str] = []
    res = obj.get("answer_resolution")
    if res is None:
        return errors, warnings  # already reported as a missing required field
    if not isinstance(res, dict):
        return ["answer_resolution must be an object"], warnings

    abstention = bool(trace) and trace[-1].get("action") == "flag_uncertainty"
    if not (abstention and res.get("visual_anchor") is None) and not _nonempty_str(res.get("visual_anchor")):
        errors.append("answer_resolution.visual_anchor must be a non-empty string")
    in_image = res.get("resolved_in_image")
    if not isinstance(in_image, bool):
        errors.append(f"answer_resolution.resolved_in_image must be a bool, got {in_image!r}")
    if abstention and in_image is not False:
        errors.append("visual abstention cannot claim an image-resolved answer")
    hops = res.get("external_hops")
    if not isinstance(hops, list):
        errors.append("answer_resolution.external_hops must be a list")
        hops = []

    answer = str(obj.get("final_answer", "") or "")
    for j, hop in enumerate(hops):
        where = f"answer_resolution.external_hops[{j}]"
        if not isinstance(hop, dict) or not isinstance(hop.get("type"), str) or hop["type"] not in ALLOWED_HOP_TYPES:
            errors.append(f"{where}: type must be one of {sorted(ALLOWED_HOP_TYPES)}")
            continue
        if hop["type"] == "lookup":
            query = hop.get("query")
            if not _nonempty_str(query):
                errors.append(f"{where}: lookup hop needs a non-empty 'query'")
                continue
            if len(query.split()) > MAX_LOOKUP_QUERY_WORDS:
                warnings.append(f"{where}: query has {len(query.split())} words "
                                f"(> {MAX_LOOKUP_QUERY_WORDS})")
            # A query that already contains the answer was written with
            # hindsight (instruction 2): it could not come from the image.
            if _mentions(query, answer):
                errors.append(f"{where}: lookup query contains the answer "
                              f"{answer!r} (hindsight leak)")
        elif not _nonempty_str(hop.get("operation")):
            errors.append(f"{where}: compute hop needs a non-empty 'operation'")

    if in_image is False and not hops and not abstention:
        errors.append("answer_resolution: resolved_in_image is false but "
                      "external_hops is empty — nothing links the image to the answer")
    if in_image is True and hops:
        errors.append("answer_resolution: resolved_in_image is true but "
                        "external_hops is not empty")

    # An out-of-image answer must not be stated as seen (instruction 2).
    if in_image is False:
        for i, step in enumerate(trace):
            if isinstance(step, dict) and _mentions(step.get("observation"), answer):
                errors.append(
                    f"search_trace[{i}].observation states the answer {answer!r} "
                    "although resolved_in_image is false"
                )
    return errors, warnings


def get_quality_flag(obj: dict[str, Any] | None, flag: str) -> bool:
    """Safely read a boolean quality flag from an annotation object."""
    if not isinstance(obj, dict):
        return False
    flags = obj.get("quality_flags")
    if isinstance(flags, dict):
        return bool(flags.get(flag, False))
    return False
