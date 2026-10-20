"""Atomic claim audit with neutral visual reads and an independent judge."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.structured import json_object
from core.views import view_identity
from records.quality import audit_fields, execution_errors

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "verification.txt"
KINDS = {"ocr", "object", "attribute", "relation", "count", "absence", "identity", "visual", "uncertainty"}
PROBES = {
    "ocr": "Transcribe all legible text exactly; explicitly mark unreadable text.",
    "object": "List visible objects and distinguish them from ambiguous shapes.",
    "attribute": "Describe visible colors, shapes, markings and states without guessing identities.",
    "relation": "Describe visible spatial and ownership relationships; separate observation from inference.",
    "count": "Enumerate distinct visible instances and their positions. State if enumeration is incomplete.",
    "absence": "Describe the visible contents and any areas whose contents cannot be established.",
    "identity": "Describe distinctive text and visual identifying cues. Flag identities that require guessing.",
    "visual": "Describe visible evidence and ambiguous details without guessing missing information.",
    "uncertainty": "Describe visible blur, occlusion, unreadable text and genuinely ambiguous cues.",
}


def extract_claims(record: dict[str, Any], client: Any, instruction: str) -> tuple[list[dict], list[str]]:
    fields, limits = audit_fields(record["annotation"], record["execution"])
    visual_steps = {entry["step"]["step"] for entry in record["execution"]["steps"] if entry["view"]}
    prompt = (instruction + "\nOPERATION: EXTRACT ALL ATOMIC VISUAL CLAIMS.\n"
              "Split every asserted visual fact, including facts embedded in reasons, "
              "decisions and summaries. Separate object, attribute, OCR, identity, count "
              "and relation claims. Questions, future plans and explicit hypotheses are "
              "not asserted observations. Do not judge correctness or add facts. "
              "Each claim text must be a verbatim quote from its original field. "
              "Never repair entities, numbers or qualifiers. For visual observations, "
              "quotes must cover all words and numbers of the original field.\n"
              "Return {\"fields\":[{\"field_id\":exact_input_key,\"claims\":["
              "{\"text\":verbatim_candidate_quote,\"kind\":ocr/object/attribute/relation/count/absence/identity/visual/uncertainty,"
              "\"evidence_steps\":[step_numbers]}],\"nonvisual_reason\":string}]}. "
              "Include EVERY input field exactly once, even if it has no claims. "
              "Evidence steps must be actual visual steps no later than the source field. "
              "For an observation with an actual view, claims must include its own step. "
              "Text-only observations and pre-action reasons can cite only earlier visual steps. "
              "Use kind uncertainty for explicit visible ambiguity/unreadability, never for a "
              "guess presented as fact. Tentative hypotheses are not asserted facts.\n"
              "Question: " + json.dumps(record["question"]) + "\n"
              "Available visual steps: " + json.dumps(sorted(visual_steps)) + "\n"
              "Field context limits: " + json.dumps(limits) + "\n"
              "Input fields: " + json.dumps(fields, ensure_ascii=False))
    obj = json_object(client.generate_text(prompt))
    rows = obj.get("fields")
    if not isinstance(rows, list) or len(rows) != len(fields):
        raise ValueError("Claim extraction did not cover every field")
    seen, claims, coverage = set(), [], []
    for row in rows:
        key = row.get("field_id") if isinstance(row, dict) else None
        if key not in fields or key in seen or not isinstance(row.get("claims"), list):
            raise ValueError("Missing, duplicate or malformed extraction field")
        seen.add(key)
        atoms = row["claims"]
        actual_observation = key.endswith(".observation") and limits[key] in visual_steps
        if not atoms and (actual_observation or not str(row.get("nonvisual_reason", "")).strip()):
            raise ValueError("A visual observation cannot silently disappear from the audit")
        coverage.append(key)
        covered = set()
        for atom in atoms:
            refs = atom.get("evidence_steps") if isinstance(atom, dict) else None
            if not isinstance(atom, dict) or atom.get("kind") not in KINDS or \
                    not isinstance(atom.get("text"), str) or not atom["text"].strip() or \
                    not isinstance(refs, list) or not refs or any(
                        type(n) is not int or n not in visual_steps or n > limits[key] for n in refs):
                raise ValueError("Malformed atomic claim or unavailable/future evidence")
            if atom["text"] not in fields[key]:
                raise ValueError("Extractor rewrote a candidate assertion")
            start = 0
            while True:
                at = fields[key].find(atom["text"], start)
                if at < 0:
                    break
                covered.update(range(at, at + len(atom["text"])))
                start = at + 1
            if actual_observation and limits[key] not in refs:
                raise ValueError("Observation claim was not tied to its actual view")
            claims.append({"claim_id": f"claim_{len(claims) + 1}", "role": "visual_fact",
                           "field_id": key, "text": atom["text"], "kind": atom["kind"],
                           "evidence_steps": sorted(set(refs))})
        if actual_observation and any(c.isalnum() and i not in covered for i, c in enumerate(fields[key])):
            raise ValueError("Extractor omitted observation words/numbers")
    return claims, coverage


def check_language_reasoning(record: dict[str, Any], client: Any, instruction: str) -> list[dict]:
    """Judge deductions with only the prefix available when each field was written."""
    fields, limits = audit_fields(record["annotation"], record["execution"])
    results = []
    for limit in sorted(set(limits.values())):
        subset = {k: text for k, text in fields.items() if limits[k] == limit}
        prefix = []
        for entry in record["execution"]["steps"][:limit]:
            step = entry["step"]
            if step["step"] < limit:
                prefix.append(step)
            else:
                # Current deductions/decisions cannot serve as their own proof.
                # NEW visual observations have a separate independent image audit.
                evidence = {k: step[k] for k in ("step", "action", "focus_region", "focus_bbox")}
                if entry["view"]:
                    evidence["observation"] = step["observation"]
                prefix.append(evidence)
        try:
            verdict = json_object(client.generate_text(instruction + "\nOPERATION: CHECK LANGUAGE REASONING.\n"
                "Use the question and the recorded prefix only. Check deductions, constraint "
                "interpretation, calculations, uncertainty statements, and action rationale. "
                "Visual facts undergo a separate image audit; do not certify outside facts. "
                "The current reasoning fields are claims to test, not additional evidence. "
                "Hypotheses/plans must remain conditional. Do not require a future inspection "
                "result to agree with an earlier reasonable exploratory choice. A supported "
                "action rationale means justified by current evidence, NOT proven optimal or "
                "causally necessary. Return {\"fields\":[{\"field_id\":exact_input_key,"
                "\"status\":\"supported\"|\"contradicted\"|\"unverifiable\",\"reason\":string}]}, "
                "each supplied field exactly once.\nQuestion: " + json.dumps(record["question"]) + "\n"
                "Recorded prefix: " + json.dumps(prefix, ensure_ascii=False) + "\n"
                "Fields to check: " + json.dumps(subset, ensure_ascii=False)))
            rows = verdict.get("fields")
            if not isinstance(rows, list) or len(rows) != len(subset):
                raise ValueError("Language audit omitted fields")
            seen = set()
            for row in rows:
                if (not isinstance(row, dict) or not isinstance(row.get("field_id"), str)
                        or row["field_id"] not in subset or row["field_id"] in seen
                        or row.get("status") not in {"supported", "contradicted", "unverifiable"}
                        or not isinstance(row.get("reason"), str) or not row["reason"].strip()):
                    raise ValueError("Malformed language verdict")
                seen.add(row["field_id"])
            results.extend({"claim_id": "language_" + row["field_id"], "role": "language_reasoning",
                            "context_limit": limit, "evidence": "question and recorded prefix",
                            **row} for row in rows)
        except Exception as exc:
            results.extend({"claim_id": "language_" + key, "role": "language_reasoning",
                            "field_id": key, "context_limit": limit, "status": "unverifiable",
                            "reason": str(exc), "error": True} for key in subset)
    return results


def _images(client: Any, prompt: str, paths: list[str]) -> dict[str, Any]:
    result = client.generate_images(prompt, paths)
    if result.truncated:
        raise ValueError("Verification response was truncated")
    return json_object(result.text)


def _view_inputs(record: dict[str, Any], refs: list[int]) -> tuple[list[str], list[dict]]:
    ledger = {e["step"]["step"]: e for e in record["execution"]["steps"]}
    seen, paths, context = set(), [], []
    for n in refs:
        view = ledger[n]["view"]
        # Identical pixels at different original locations are distinct evidence
        # (e.g. two identical screws). Deduplicate repeated views, not instances.
        key = view_identity(view) if view else None
        if view and key not in seen:
            paths.append(view["path"])
            context.append({"image_index": len(paths), "observed_at_step": n,
                            "original_bbox": view["bbox"], "pixel_bbox": view["pixel_bbox"],
                            "source_image_size": record["image_size"],
                            "width": view["width"], "height": view["height"]})
            seen.add(key)
    return paths, context


def check_grounding(record: dict[str, Any], client: Any, *, max_claims: int = 128,
                    instruction: str | None = None) -> dict[str, Any]:
    instruction = instruction if instruction is not None else PROMPT_PATH.read_text()
    errors = execution_errors(record)
    if errors:
        return {"pass": None, "complete": False, "claims": [], "errors": errors}
    try:
        claims, coverage = extract_claims(record, client, instruction)
        ann = record["annotation"]
        abstention = ann["search_trace"][-1]["action"] == "flag_uncertainty"
        evidence = ann["answer_grounding"]["evidence_step"]
        if abstention and len(ann["search_trace"]) < record["execution"]["max_steps"]:
            view = record["execution"]["steps"][evidence - 1]["view"]
            pixels = view["pixel_bbox"] if view else None
            if (not pixels or view["width"] != pixels[2] - pixels[0]
                    or view["height"] != pixels[3] - pixels[1]):
                raise ValueError("Early visual abstention needs native-resolution evidence; a downsampled preview cannot establish irrecoverable source detail")
        refs = [e["step"]["step"] for e in record["execution"]["steps"]
                if e["view"] and e["step"]["step"] <= evidence]
        if abstention:
            refs = [e["step"]["step"] for e in record["execution"]["steps"] if e["view"]]
            claims.append({"claim_id": "abstention", "role": "abstention",
                           "text": "The observed views leave the declared visual question unresolved: "
                                   + json.dumps(ann["search_trace"][-1]["belief_update"]["open_visual_questions"])
                                   + ". The stop is justified by exhausted budget or observed irrecoverable detail; "
                                     "remaining external knowledge alone cannot justify visual abstention. "
                                   + f"Executed steps: {len(ann['search_trace'])}; budget: {record['execution']['max_steps']}.",
                           "kind": "uncertainty", "evidence_steps": refs})
        else:
            claims.append({"claim_id": "visual_anchor", "role": "visual_anchor",
                           "text": "The visible evidence identifies the question's target as: "
                                   + ann["answer_resolution"]["visual_anchor"],
                           "kind": "identity", "evidence_steps": refs})
        if not abstention and ann["answer_resolution"]["resolved_in_image"]:
            claims.append({"claim_id": "in_image_answer", "role": "in_image_answer",
                           "text": "The visible evidence alone establishes this answer to the question: "
                                   + record["answer"], "kind": "visual", "evidence_steps": refs})
        # Never sample a subset and label the whole trace verified.
        n_language = len(audit_fields(ann, record["execution"])[0])
        if len(claims) + (0 if abstention else 1) + n_language > max_claims:
            raise ValueError(f"{len(claims) + (0 if abstention else 1) + n_language} checks exceed audit budget {max_claims}; increase --max-claims")
    except Exception as exc:
        return {"pass": None, "complete": False, "claims": [], "errors": [str(exc)]}

    neutral_cache = {}
    for claim in claims:
        try:
            paths, context = _view_inputs(record, claim["evidence_steps"])
            key = (tuple(paths), claim["kind"])
            if key not in neutral_cache:
                # No candidate reading, answer or caption appears in this call.
                neutral_cache[key] = _images(client, instruction + "\nOPERATION: NEUTRAL VISUAL READ.\n"
                    + PROBES[claim["kind"]] + "\nView locations in the same original image: "
                    + json.dumps(context) + "\nOverlapping crops may show the same instance; do not double count. "
                    "Return {\"observed\":string,\"uncertain\":boolean}.", paths)
            read = neutral_cache[key]
            if not isinstance(read.get("observed"), str) or type(read.get("uncertain")) is not bool:
                raise ValueError("Malformed neutral visual read")
            result = _images(client, instruction + "\nOPERATION: JUDGE ONE ATOMIC CLAIM.\n"
                "Question: " + json.dumps(record["question"]) + "\n"
                "Claim: " + json.dumps(claim["text"]) + "\n"
                "Actual view locations: " + json.dumps(context) + "\n"
                "Independent preliminary reading: " + json.dumps(read) + "\n"
                "Check against the attached views. A preliminary reading is fallible. "
                "An identity claim must also establish that the cue belongs to the target. "
                "Return {\"status\":\"supported\"|\"contradicted\"|\"unverifiable\","
                "\"evidence\":specific_visible_evidence,\"reason\":string}.", paths)
            if result.get("status") not in {"supported", "contradicted", "unverifiable"} or \
                    not isinstance(result.get("evidence"), str) or not result["evidence"].strip() or \
                    not isinstance(result.get("reason"), str) or not result["reason"].strip():
                raise ValueError("Malformed visual verdict")
            # A reader that could not establish the evidence cannot certify it.
            if read["uncertain"] and result["status"] == "supported" and claim["kind"] != "uncertainty":
                result["status"] = "unverifiable"
                result["reason"] = "Neutral visual reading was uncertain; " + result["reason"]
            claim.update(result, neutral_read=read)
        except Exception as exc:
            claim.update(status="unverifiable", evidence="", reason=str(exc), error=True)

    claims.extend(check_language_reasoning(record, client, instruction))

    # Plans must be coherent with observed evidence; this is not a web fact check.
    if not abstention:
        try:
            plan = json_object(client.generate_text(instruction + "\nOPERATION: CHECK RESOLUTION PLAN.\n"
                "Check that the ordered planned lookups/computations address the question and "
                "use only entities/values supported by the logged observations. Do not execute "
                "searches or assume the unknown answer. Return {\"consistent\":boolean,\"reason\":string}.\n"
                "Data: " + json.dumps({"question": record["question"],
                                        "observations": ann["search_trace"],
                                        "resolution": ann["answer_resolution"]}, ensure_ascii=False)))
            if type(plan.get("consistent")) is not bool or not isinstance(plan.get("reason"), str) or not plan["reason"].strip():
                raise ValueError("Malformed resolution-plan judgment")
            claims.append({"claim_id": "resolution_plan", "role": "resolution_plan",
                           "status": "supported" if plan["consistent"] else "contradicted",
                           "reason": plan["reason"], "evidence": "plan consistency only; external facts unverified"})
        except Exception as exc:
            claims.append({"claim_id": "resolution_plan", "role": "resolution_plan",
                           "status": "unverifiable", "reason": str(exc), "error": True})
    statuses = [c["status"] for c in claims]
    passed = False if "contradicted" in statuses else (None if "unverifiable" in statuses else True)
    return {"pass": passed, "complete": "unverifiable" not in statuses,
            "mode": "abstention" if abstention else "resolved",
            "claims": claims, "covered_fields": coverage, "errors": []}
