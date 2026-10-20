"""Prefix-only visual/language audit and separate reference answer evaluation."""
from __future__ import annotations

import json
import copy
import logging
from pathlib import Path

from agent.runner import (OBSERVATION_FIELDS, RUN_VERSION, call, observation_prompt, policy_prompt,
                          rollout_fingerprint, retry_prompt, validate_observation, validate_request, visible_images)
from core.provenance import file_sha, fingerprint, independent_models
from core.views import matches_materialized_view
from agent.schema import EVALUATION_SCHEMA, READER_SCHEMA, audit_schema
from sources.benchmarks import task_fingerprint
from sources.partition import derived_partition

AUDIT_VERSION = "agent-evidence-v4"
logger = logging.getLogger(__name__)



def _validated_judge_call(judge, prompt, images, schema, validator, attempts=None):
    rejected = []
    for number in range(3):
        feedback = "" if not rejected else "\nCorrect the previous response using this validation feedback. Keep the evidence and truth judgments unchanged; do not turn unsupported assertions into supported ones. complete=true means all required items were evaluated, not that the answer is correct. visual_facts must use kind=visual; language_facts must use kind=language, including when contradicted or unverifiable.\n" + json.dumps(rejected, ensure_ascii=False)
        trace = {}
        try:
            verdict = call(judge, prompt + feedback, images, schema, trace)
            return validator(verdict)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            item = {"error": str(exc)[:1000], "raw_response": trace.get("raw_response", "")[:12000]}
            key = getattr(judge, "api_key", None)
            if key:
                item = {k: v.replace(key, "[REDACTED]") for k, v in item.items()}
            rejected.append(item)
            if attempts is not None:
                attempts.append(item)
            if number == 2:
                raise ValueError("Judge response invalid after 3 attempts: " + item["error"]) from exc
            logger.warning("Judge output validation failed (%d/3): %s; requesting corrected JSON", number + 1, item["error"])


def audit_receipt(record):
    verification = {k: v for k, v in record.get("verification", {}).items() if k != "receipt_sha256"}
    return fingerprint({"verification": verification, "reference_evaluation": record.get("reference_evaluation")})


def execution_errors(record: dict) -> list[str]:
    errors = []
    try:
        if record["version"] != RUN_VERSION or record["schema"] != "visual-agent-rollout-v1" or not record["trajectory_executed"]:
            return ["Wrong execution schema/version"]
        task, steps, initial = record["task"], record["steps"], record["initial_views"]
        if "reference" in task or task_fingerprint(task) != task["input_fingerprint"]:
            errors.append("Task/reference boundary changed")
        if task.get("trajectory_derivation") is not None and derived_partition(task) is None:
            errors.append("Seed partition changed")
        if not 1 <= record["max_steps"] <= 80 or len(steps) > record["max_steps"]:
            errors.append("Invalid execution budget")
        if len(initial) != len(task["model_input"]["images"]):
            errors.append("Initial views omitted")
        for image, view in zip(task["model_input"]["images"], initial):
            if file_sha(image["path"]) != image["sha256"] or file_sha(view["path"]) != view["sha256"]:
                errors.append("Initial image/view bytes changed")
            if view.get("asset_id") != f"image_{image['index']}" or not matches_materialized_view(image["path"], view, None, 512):
                errors.append("Initial view geometry/pixels changed")
        inventory = [{"asset_id": f"image_{image['index']}", "origin": "initial_input"}
                     for image in task["model_input"]["images"]]
        for i, entry in enumerate(steps):
            prefix = steps[:i]
            request, planner, result = entry["request"], entry["planner"], entry["result"]
            validate_request(request, prefix)
            if entry["step"] != i + 1 or planner["response"] != request or \
                    planner["prompt"] != retry_prompt(policy_prompt(task["model_input"]["question"], initial, prefix,
                                                       inventory, record["max_steps"] - i, record["tool_config"]), planner.get("rejected_attempts", [])) or \
                    planner["images"] != visible_images(initial, prefix) or \
                    planner["prefix_fingerprint"] != fingerprint(prefix) or planner["context_limit"] != i:
                errors.append(f"Step {i + 1} planner context/response changed")
            if len(planner["prompt"]) > record["max_context_chars"] or len(planner["images"]) > record["max_context_images"]:
                errors.append("Planner exceeds its declared context budget")
            if request["tool"] in {"finish", "abstain"}:
                if i != len(steps) - 1 or result is not None or entry["observation"] is not None or entry["observer"] is not None:
                    errors.append("Terminal action has invented result/observation")
            else:
                if result["tool"] != request["tool"] or result["status"] not in {"ok", "error"}:
                    errors.append("Mismatched tool result")
                artifact = result["artifact"]
                if file_sha(artifact["path"]) != artifact["sha256"] or json.loads(Path(artifact["path"]).read_text()) != {
                        k: v for k, v in result.items() if k != "artifact"}:
                    errors.append("Tool output artifact changed")
                data = result.get("data", {})
                view = data.get("view")
                if view:
                    if file_sha(view["path"]) != view["sha256"] or file_sha(data["source_path"]) != data["source_sha256"]:
                        errors.append("Acquired source/view bytes changed")
                    if not matches_materialized_view(data["source_path"], view, request["arguments"].get("bbox"),
                                                     record["tool_config"]["image_side"]):
                        errors.append("Acquired view geometry/pixels changed")
                    key = data["asset_id"]
                    if key not in {asset["asset_id"] for asset in inventory}:
                        inventory.append({"asset_id": key, "origin": data["origin"]})
                raw = data.get("raw_artifact")
                if raw and file_sha(raw["path"]) != raw["sha256"]:
                    errors.append("Raw page changed")
                observer = entry["observer"]
                validate_observation(entry["observation"], prefix, result)
                if observer["response"] != entry["observation"] or observer["prompt"] != retry_prompt(observation_prompt(
                        task["model_input"]["question"], prefix, request, result), observer.get("rejected_attempts", [])) or \
                        observer["images"] != ([view["path"]] if view else []) or \
                        observer["prefix_fingerprint"] != fingerprint(prefix) or observer["context_limit"] != i:
                    errors.append(f"Step {i + 1} observer context/response changed")
                if len(observer["prompt"]) > record["max_context_chars"] or len(observer["images"]) > record["max_context_images"]:
                    errors.append("Observer exceeds its declared context budget")
            if entry["inventory"] != inventory:
                errors.append("Image inventory contains unavailable/mismatched assets")
        if record["termination"] in {"finish", "abstain"} and (not steps or steps[-1]["request"]["tool"] != record["termination"]):
            errors.append("Termination differs from the executed action")
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        errors.append("Invalid execution ledger: " + type(exc).__name__)
    return errors


def _fields(entry, role):
    if role == "observer":
        return {key: entry["observation"][key] for key in sorted(OBSERVATION_FIELDS)}
    request = entry["request"]
    fields = {key: request[key] for key in ("rationale", "information_need")}
    if request["tool"] in {"finish", "abstain"}:
        fields.update({"arguments." + k: v for k, v in request["arguments"].items()})
    return fields


def _neutral(judge, images, cache, attempts=None):
    notes = []
    for path in images:
        sha = file_sha(path)
        if sha not in cache:
            def validate_reader(verdict):
                if set(verdict) != {"facts", "uncertain_details"} or any(
                        not isinstance(verdict[k], list) or any(not isinstance(x, str) for x in verdict[k]) for k in verdict):
                    raise ValueError("Invalid neutral reader result")
                return verdict
            cache[sha] = _validated_judge_call(judge, """Read this image independently BEFORE seeing any candidate caption/answer.
Return JSON {facts:[str],uncertain_details:[str]}. Report only clearly visible pixels/text;
do not infer identities, history or external knowledge. Unreadable detail stays uncertain.""", [path], READER_SCHEMA, validate_reader, attempts)
        notes.append({"image_index": len(notes) + 1, "sha256": sha, **cache[sha]})
    return notes


def _field_units(fields):
    units = {}
    for field_id, value in fields.items():
        values = value if isinstance(value, list) else [value]
        for index, item in enumerate(values):
            if item is None or item == "":
                continue
            units[f"{field_id}[{index}]"] = {"field_id": field_id, "unit_index": index,
                "text": item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)}
    return units


def _aggregate_status(statuses):
    return "contradicted" if "contradicted" in statuses else "unverifiable" if "unverifiable" in statuses else "supported"


def _audit_call(entry, role, judge, cache, attempts=None):
    current = entry[role]
    fields = _fields(entry, role)
    units = _field_units(fields)
    limit = entry["step"] - (1 if role == "planner" else 0)
    neutral = _neutral(judge, current["images"], cache, attempts)
    prompt = """Audit ONLY the supplied candidate_units against the exact original call context.
Each unit is an original field or list item. Preserve EVERY assertion and qualifier.
Do not rewrite a wrong candidate into a correct statement. In particular, never replace its entity, script, number or uncertainty.
Neutral reading is comparison evidence, not candidate text. Empty fields are handled by code; do not output them.
For each unit, decompose ALL assertions into claims with candidate_quote copied VERBATIM from that unit.
Quotes must collectively cover all words/numbers of the unit. A whole-unit quote is allowed when it is one assertion.
Judge the quoted assertion as written, not a paraphrase. Conditional questions/plans can pass without claiming future results.
Requested information and intended inspections have kind=conditional, not visual facts. Supported actual facts/deductions must cite available evidence_steps.
A positive assertion about a detail marked uncertain by neutral reading is unverifiable.
Use attached pixels for visual facts and actual retrieved text/premises for language facts/deductions.
Text captions/alt text cannot prove pixels. Current deductions/hypotheses cannot prove themselves.
Earlier notes are provisional premises, accepted only if every prior field also passes.
An assertion that no pixels were supplied is contradicted when original_call_images is nonempty.
visual_facts claims must have kind=visual; language_facts claims must have kind=language.
Finished answers must be visual/language/derived facts, not conditional plans or uncertain placeholders.
Return JSON {complete:bool,units:[{unit_id:str,status:'supported'|'contradicted'|'unverifiable',
claims:[{candidate_quote:str,kind:'visual'|'language'|'derived'|'conditional'|'uncertainty',
status:'supported'|'contradicted'|'unverifiable',evidence_steps:[int]}]}]}.
Return every expected_unit_id exactly once. complete means coverage even when assertions are wrong.
Unit status is contradicted if any claim is contradicted, else unverifiable if any claim is unverifiable, else supported.
Step 0 denotes original question/initial pixels; other cited steps must be <= context_limit.
No future evidence or reference answer is available.
""" + json.dumps({"original_prompt": current["prompt"], "candidate_units": units,
                  "expected_unit_ids": sorted(units),
                  "original_call_images": [{"image_index": i, "sha256": file_sha(path)}
                                           for i, path in enumerate(current["images"], 1)],
                  "neutral_reading": neutral, "context_limit": limit}, ensure_ascii=False)
    def validate(verdict):
        rows = verdict.get("units")
        if verdict.get("complete") is not True or not isinstance(rows, list) or len(rows) != len(units) or \
                {row.get("unit_id") for row in rows} != set(units):
            raise ValueError("Incomplete/duplicate original-unit audit")
        grouped = {field_id: [] for field_id in fields}
        for row in rows:
            unit = units[row["unit_id"]]
            source = unit["text"]
            claims = row.get("claims")
            if row.get("status") not in {"supported", "contradicted", "unverifiable"} or not isinstance(claims, list) or not claims:
                raise ValueError("Malformed original-unit verdict")
            covered = set()
            preserved = []
            for claim in claims:
                quote = claim.get("candidate_quote")
                if not isinstance(quote, str) or not quote.strip() or quote not in source:
                    raise ValueError("Judge rewrote/invented a candidate assertion")
                start = 0
                while True:
                    at = source.find(quote, start)
                    if at < 0:
                        break
                    covered.update(range(at, at + len(quote)))
                    start = at + 1
                if claim.get("kind") not in {"visual", "language", "derived", "conditional", "uncertainty"} or \
                        claim.get("status") not in {"supported", "contradicted", "unverifiable"}:
                    raise ValueError("Malformed atomic claim")
                field_id = unit["field_id"]
                if field_id == "arguments.answers" and claim["kind"] not in {"visual", "language", "derived"}:
                    raise ValueError("Finished answers must be supported facts/deductions")
                if field_id in {"visual_facts", "language_facts"} and claim["kind"] != (
                        "visual" if field_id == "visual_facts" else "language"):
                    raise ValueError("Direct fact fields must be audited as direct facts")
                refs = claim.get("evidence_steps")
                if not isinstance(refs, list) or any(type(s) is not int or not 0 <= s <= limit for s in refs):
                    raise ValueError("Claim uses unavailable future evidence")
                if claim["status"] == "supported" and claim["kind"] != "conditional" and not refs:
                    raise ValueError("Supported fact/deduction requires available evidence")
                if claim["kind"] == "visual" and not current["images"]:
                    raise ValueError("Visual audit lacks actual pixels")
                preserved.append({"claim": quote, **claim, "unit_id": row["unit_id"]})
            if any(character.isalnum() and i not in covered for i, character in enumerate(source)):
                raise ValueError("Audit omitted candidate words/numbers")
            if row["status"] != _aggregate_status([claim["status"] for claim in claims]):
                raise ValueError("Atomic/unit verdicts are inconsistent")
            grouped[unit["field_id"]].extend(preserved)
        checked = [{"field_id": field_id, "status": _aggregate_status([c["status"] for c in claims]),
                    "claims": claims, "schema_empty": not bool(claims)} for field_id, claims in grouped.items()]
        return {"step": entry["step"], "role": role, "context_limit": limit,
                "verdict": {"complete": True, "fields": checked}, "judge_unit_verdict": verdict}
    return _validated_judge_call(judge, prompt, current["images"], audit_schema(sorted(units), limit), validate, attempts)


def evaluate_reference(record: dict, reference: dict, judge, attempts=None):
    task = record["task"]
    if reference.get("id") != task["id"] or reference.get("input_fingerprint") != task["input_fingerprint"]:
        raise ValueError("Reference does not match the original task")
    answers = record["steps"][-1]["request"]["arguments"]["answers"]
    prompt = """Evaluate these predicted answers against the detached reference.
This is answer evaluation AFTER execution, not evidence or trajectory generation.
For answer_format=set ALL required answers must be present; each answer_evals group is alternatives for ONE required answer.
For alternatives, listed answer_candidates are accepted alternative forms of the same answer.
Do not supply missing answers on behalf of the prediction; reject contradictory extra answers.
Return JSON {correct:bool,complete:bool,explanation:str}.""" + json.dumps({
        "question": task["model_input"]["question"], "predicted_answers": answers,
        "reference": {k: reference[k] for k in ("answer", "answer_candidates", "answer_format", "required_answers", "answer_evals")
                      if k in reference}}, ensure_ascii=False)
    def validate(verdict):
        if type(verdict.get("correct")) is not bool or verdict.get("complete") is not True or not isinstance(verdict.get("explanation"), str):
            raise ValueError("Invalid reference evaluation verdict")
        return {**verdict, "reference_fingerprint": fingerprint(reference), "judge_model": judge.model,
                "rollout_fingerprint": rollout_fingerprint(record)}
    return _validated_judge_call(judge, prompt, [], EVALUATION_SCHEMA, validate, attempts)


def _matching_audit(previous, record, judge_model, reference):
    audit = (previous or {}).get("verification", {})
    return bool(audit and audit.get("version") == AUDIT_VERSION and
                audit.get("judge_model") == judge_model and
                audit.get("input_fingerprint") == rollout_fingerprint(record) and
                audit.get("reference_fingerprint") == (fingerprint(reference) if reference is not None else None) and
                audit.get("receipt_sha256") == audit_receipt(previous))


def reusable_verification(previous, record, judge_model, reference):
    return _matching_audit(previous, record, judge_model, reference) and previous["verification"].get("retryable") is False


def verify_rollout(record: dict, judge, reference: dict | None = None, previous=None) -> dict:
    output = {**record}
    output.pop("verification", None)
    output.pop("reference_evaluation", None)
    verification = {"version": AUDIT_VERSION, "judge_model": judge.model,
                    "input_fingerprint": rollout_fingerprint(record), "pass": False, "complete": False,
                    "reference_fingerprint": fingerprint(reference) if reference is not None else None,
                    "calls": [], "errors": execution_errors(record), "neutral_reads": {},
                    "retryable": False, "rejected_attempts": [], "reference_errors": []}
    output["verification"] = verification
    if not independent_models(record["model"], judge.model):
        verification["errors"].append("A different recognized model family is required")
    if record["termination"] not in {"finish", "abstain"}:
        verification["errors"].append("No executed terminal behavior")
    if reference is not None and (reference.get("id") != record["task"]["id"] or
                                  reference.get("input_fingerprint") != record["task"]["input_fingerprint"]):
        verification["errors"].append("Detached reference does not match the original task")
    if record["termination"] == "abstain" and len(record["steps"]) < record["max_steps"]:
        verification["errors"].append("Abstention before decision-budget exhaustion requires further evidence acquisition")
    if verification["errors"]:
        verification["receipt_sha256"] = audit_receipt(output)
        return output
    if _matching_audit(previous, record, judge.model, reference):
        old = previous["verification"]
        for field in ("calls", "neutral_reads", "rejected_attempts"):
            verification[field] = copy.deepcopy(old.get(field, verification[field]))
    try:
        index = 0
        for entry in record["steps"]:
            for role in ("planner", "observer"):
                if entry[role] is not None:
                    if index >= len(verification["calls"]):
                        verification["calls"].append(_audit_call(entry, role, judge, verification["neutral_reads"], verification["rejected_attempts"]))
                    index += 1
        verification["complete"] = True
        verification["pass"] = all(row["status"] == "supported" and all(c["status"] == "supported" for c in row["claims"])
            for item in verification["calls"] for row in item["verdict"]["fields"])
    except Exception as exc:
        verification["errors"].append(type(exc).__name__ + ": " + str(exc))
        verification["retryable"] = True
    if verification["pass"] and reference is not None and record["termination"] == "finish":
        try:
            output["reference_evaluation"] = evaluate_reference(record, reference, judge, verification["rejected_attempts"])
        except Exception as exc:
            verification["reference_errors"].append(type(exc).__name__ + ": " + str(exc))
            verification["retryable"] = True
    verification["receipt_sha256"] = audit_receipt(output)
    return output


def training_rejection_reason(record: dict) -> str | None:
    errors = execution_errors(record)
    if errors:
        return "execution invalid: " + errors[0]
    if derived_partition(record["task"]) != "train":
        return "not an explicit benchmark-derived training seed"
    audit = record.get("verification", {})
    if not audit:
        return "independent audit missing"
    if audit.get("version") != AUDIT_VERSION or audit.get("input_fingerprint") != rollout_fingerprint(record):
        return "independent audit stale: version or trajectory changed"
    if not independent_models(record["model"], str(audit.get("judge_model", ""))):
        return "independent audit requires a different model family"
    if audit.get("errors"):
        return "independent audit error: " + audit["errors"][0]
    if audit.get("complete") is not True:
        return "independent audit incomplete"
    if audit.get("pass") is not True:
        return "independent audit rejected: unsupported or contradicted claims"
    if audit.get("reference_errors"):
        return "reference evaluation error: " + audit["reference_errors"][0]
    if audit.get("receipt_sha256") != audit_receipt(record):
        return "audit/reference verdict changed"
    expected = [(e["step"], role, set(_fields(e, role))) for e in record["steps"]
                for role in ("planner", "observer") if e[role] is not None]
    calls = audit.get("calls")
    if not isinstance(calls, list) or len(calls) != len(expected):
        return "agent audit coverage incomplete"
    for item, (step, role, fields) in zip(calls, expected):
        verdict = item.get("verdict", {})
        rows = verdict.get("fields", [])
        if item.get("step") != step or item.get("role") != role or verdict.get("complete") is not True or \
                len(rows) != len(fields) or {row.get("field_id") for row in rows} != fields or any(
                    row.get("status") != "supported" or any(c.get("status") != "supported" for c in row.get("claims", [])) for row in rows):
            return "agent audit coverage/verdict changed"
    if record["termination"] == "finish":
        grade = record.get("reference_evaluation", {})
        if grade.get("correct") is not True or grade.get("complete") is not True or \
                grade.get("rollout_fingerprint") != rollout_fingerprint(record) or grade.get("judge_model") != audit["judge_model"]:
            return "correct detached-reference evaluation required"
    elif record["termination"] != "abstain" or len(record["steps"]) != record["max_steps"]:
        return "terminal behavior not admitted"
    return None
