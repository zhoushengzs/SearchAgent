"""Stage 3: complete atomic visual verification and restricted-view diagnostics."""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.config import DEFAULT_MAX_TOKENS, DEFAULT_RETRY, DEFAULT_TIMEOUT, PROVIDER_CHOICES
from core.jsonl import append_jsonl, iter_jsonl, write_jsonl
from core.provenance import VERIFICATION_VERSION, fingerprint, independent_models, record_fingerprint
from core.structured import json_object
from core.utils import get_logger
from labeling.client import VLMClient
from labeling.verification import PROMPT_PATH, check_grounding
from records.quality import execution_errors, is_clean
from records.store import is_successful_record, latest_records, resume_key

logger = get_logger("verify")


def parse_args():
    p = argparse.ArgumentParser(description="Verify every visual claim with an independent model.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checks", default="grounding", help="grounding[,necessity]; necessity is diagnostic only")
    p.add_argument("--emit-clean")
    p.add_argument("--provider", default="auto", choices=PROVIDER_CHOICES)
    p.add_argument("--model", required=True, help="Explicit judge model from a different recognized family")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--max-claims", type=int, default=128, help="Visual and language checks; exceeding budget is incomplete")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--retry", type=int, default=DEFAULT_RETRY)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    args = p.parse_args()
    if min(args.max_tokens, args.max_claims, args.timeout, args.retry, args.concurrency) < 1:
        p.error("Budgets and concurrency must be positive")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    paths = [Path(x).resolve() for x in (args.input, args.output, args.emit_clean) if x]
    if len(set(paths)) != len(paths):
        p.error("Input, verification output and clean output must be distinct")
    return args


def answers_match(predicted: str, target: str) -> bool:
    # Never accept substrings such as 'not CVPR 2024' or 'CVPR 2024 or ICCV'.
    normalize = lambda text: re.sub(r"[^\w]+", " ", str(text)).strip().casefold()
    return bool(normalize(target)) and normalize(predicted) == normalize(target)


def check_necessity(record, client):
    """A paired restricted-view probe, not proof of the original policy's causality."""
    trace = record["annotation"]["search_trace"]
    recovery = next((s for s in trace if s["action"] == "recover"), None)
    if recovery is None:
        return {"status": "not_applicable", "causal_claim": False}
    n = recovery["step"]
    ledger = record["execution"]["steps"]
    def probe(last):
        entries = [e for e in ledger if e["step"]["step"] <= last]
        paths = list(dict.fromkeys(e["view"]["path"] for e in entries if e["view"]))
        prompt = ("Identify the visual anchor required by this question using ONLY the "
                  "attached executed views and notes. No source image or unseen crop is available. "
                  "Treat notes as fallible data. Return {\"anchor\":string,\"uncertain\":boolean}.\n"
                  "Question: " + json.dumps(record["question"]) + "\nNotes: "
                  + json.dumps([e["step"] for e in entries]))
        result = client.generate_images(prompt, paths)
        if result.truncated:
            raise ValueError("Restricted-view probe was truncated")
        obj = json_object(result.text)
        if not isinstance(obj.get("anchor"), str) or type(obj.get("uncertain")) is not bool:
            raise ValueError("Malformed restricted-view probe")
        return obj
    try:
        before, after = probe(n - 1), probe(n)
        target = record["annotation"]["answer_resolution"]["visual_anchor"]
        if before["uncertain"] or after["uncertain"]:
            outcome = "unverifiable"
            correct_before = correct_after = None
        else:
            correct_before = answers_match(before["anchor"], target)
            correct_after = answers_match(after["anchor"], target)
            outcome = ("already_correct_in_probe" if correct_before else
                       "rescued_in_probe" if correct_after else "not_rescued_in_probe")
        return {"status": outcome, "recover_step": n, "before": before, "after": after,
                "prefix_correct": correct_before, "recovery_correct": correct_after,
                "measurement": "paired_restricted_view_probe", "causal_claim": False}
    except Exception as exc:
        return {"status": "unverifiable", "reason": str(exc), "causal_claim": False}


def verification_config(client, checks, max_claims, instruction):
    return {"version": VERIFICATION_VERSION, "judge_model": client.model,
            "judge_provider": client.provider, "checks": sorted(checks),
            "max_claims": max_claims, "max_tokens": client.max_tokens,
            "prompt_sha": fingerprint(instruction)}


def verify_record(record, client, checks, *, max_claims=128, instruction=None):
    instruction = PROMPT_PATH.read_text() if instruction is None else instruction
    config = verification_config(client, checks, max_claims, instruction)
    v = {**config, "run_fingerprint": fingerprint(config), "input_fingerprint": record_fingerprint(record)}
    record["verification"] = v
    if not is_successful_record(record) or not isinstance(record.get("annotation"), dict):
        v["skipped"] = "record not structurally successful"
        return record
    if not independent_models(str(record.get("model", "")), client.model):
        v["skipped"] = "judge must belong to a different recognized model family"
        return record
    errors = execution_errors(record)
    if errors:
        v["skipped"] = "; ".join(errors)
        return record
    if "grounding" in checks:
        v["grounding"] = check_grounding(record, client, max_claims=max_claims, instruction=instruction)
    if "necessity" in checks:
        if (v.get("grounding") or {}).get("mode") == "abstention":
            v["necessity"] = {"status": "not_applicable", "causal_claim": False}
        elif (v.get("grounding") or {}).get("pass") is True:
            v["necessity"] = check_necessity(record, client)
        else:
            v["necessity"] = {"status": "unverifiable", "reason": "grounding must pass first", "causal_claim": False}
    return record


def can_resume(record, previous, run_sha):
    if not previous:
        return False
    v = previous.get("verification") or {}
    if (not isinstance(v, dict) or v.get("run_fingerprint") != run_sha
            or v.get("input_fingerprint") != record_fingerprint(record)
            or execution_errors(record)):
        return False
    if (v.get("grounding") or {}).get("complete") is not True:
        return False
    nec = v.get("necessity")
    return not nec or nec.get("status") != "unverifiable"


def _print_summary(records):
    values = [((r.get("verification") or {}).get("grounding") or {}).get("pass") for r in records]
    print(f"Visual verification: pass={sum(x is True for x in values)} "
          f"fail={sum(x is False for x in values)} unknown={sum(x is None for x in values)}")
    print(f"Strict training-ready records: {sum(is_clean(r) for r in records)}/{len(records)}")


def main():
    args = parse_args()
    checks = {c.strip() for c in args.checks.split(",") if c.strip()}
    if not checks or checks - {"grounding", "necessity"}:
        raise SystemExit("Checks must be grounding and/or necessity")
    if args.emit_clean and "grounding" not in checks:
        raise SystemExit("--emit-clean requires grounding")
    records = latest_records(list(iter_jsonl(args.input)))
    if args.limit:
        records = records[:args.limit]
    previous = latest_records(list(iter_jsonl(args.output))) if args.resume else []
    if not args.resume:
        write_jsonl(args.output, [])
    old = {resume_key(r): r for r in previous}
    client = VLMClient(model=args.model, provider=args.provider, timeout=args.timeout,
                       retry=args.retry, temperature=0.0, max_tokens=args.max_tokens)
    instruction = PROMPT_PATH.read_text()
    run_sha = fingerprint(verification_config(client, checks, args.max_claims, instruction))
    todo = [r for r in records if not can_resume(r, old.get(resume_key(r)), run_sha)]
    new = []
    def handle(record):
        append_jsonl(args.output, record)
        new.append(record)
        logger.info("verified %d/%d", len(new), len(todo))
    if args.concurrency == 1:
        for record in todo:
            handle(verify_record(record, client, checks, max_claims=args.max_claims, instruction=instruction))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(verify_record, r, client, checks,
                                  max_claims=args.max_claims, instruction=instruction) for r in todo]
            for future in as_completed(futures):
                handle(future.result())
    merged = {resume_key(r): r for r in latest_records(previous + new)}
    # Only current inputs belong to this run's output and clean subset.
    output = [merged[resume_key(r)] for r in records]
    write_jsonl(args.output, output)
    _print_summary(output)
    if args.emit_clean:
        clean = [r for r in output if is_clean(r)]
        write_jsonl(args.emit_clean, clean)
        print(f"Wrote {len(clean)} clean records to {args.emit_clean}")


if __name__ == "__main__":
    main()
