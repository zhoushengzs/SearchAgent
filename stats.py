'Statistics for normalized or labeled JSONL files.'
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from core.utils import get_logger
from records.validator import ALLOWED_VERDICTS, expected_verdict, get_quality_flag
from records.store import latest_records
from records.quality import training_rejection_reason

logger = get_logger("stats")

# Optional source-balance guidance; recovery/step frequencies are descriptive.
HEALTHY = {
    "max_source_share": 0.40,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute stats over normalized/labeled JSONL.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--image-root", type=str, default=".")
    return p.parse_args()


def _resolve(image_path: str, root: Path) -> Path:
    p = Path(image_path)
    return p if (p.is_absolute() or p.exists()) else root / p


def main() -> None:
    args = parse_args()
    raw = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(r.get("schema") == "visual-agent-rollout-v1" for r in raw):
        if not all(r.get("schema") == "visual-agent-rollout-v1" for r in raw):
            raise SystemExit("Agent rollouts and visual module records must be inspected separately")
        from agent.metrics import rollout_statistics
        print(json.dumps(rollout_statistics(raw), indent=2))
        print("Horizon bands count executed tool calls; dependency links are model declarations, not causal proof.")
        return
    records = latest_records(raw)
    root = Path(args.image_root)
    total = len(records)
    if any(r.get("schema") == "visual-agent-task-v1" for r in records):
        if not all(r.get("schema") == "visual-agent-task-v1" for r in records):
            raise SystemExit("Task manifests and visual annotations must be inspected separately")
        images = [im for r in records for im in r["model_input"]["images"]]
        counts = Counter(len(r["model_input"]["images"]) for r in records)
        print(f"Benchmark tasks: {total} (unexecuted inputs; references are separate)")
        print(f"Sources: {dict(Counter(r['source_dataset'] for r in records))}")
        print(f"Initial image count per task: {dict(sorted(counts.items()))}")
        print(f"Images present: {sum(_resolve(im['path'], root).is_file() for im in images)}/{len(images)}")
        print("Use agent_pipeline.py for adapted retrieval rollouts; official scores require official environments/evaluators.")
        return

    is_labeled = any("annotation" in r or "parse_success" in r for r in records)

    source_counts: Counter[str] = Counter()
    usage_counts: Counter[str] = Counter()
    with_answer = 0
    without_answer = 0
    image_exists = 0
    image_missing = 0
    question_len_sum = 0

    parse_success = 0
    parse_failed = 0
    needs_review = 0
    anchor_not_verifiable = 0
    possible_hallucination = 0
    validation_error_records = 0

    # Descriptive distributions of executed actions and declared decisions.
    n_ann = 0
    recover_count_dist: Counter[int] = Counter()
    pattern_dist: Counter[str] = Counter()
    verdict_dist: Counter[str] = Counter()
    qtype_dist: Counter[str] = Counter()
    variant_dist: Counter[str] = Counter()
    prompt_dist: Counter[str] = Counter()
    rule_violations = 0
    lookbacks_per_trace_sum = 0
    hindsight_hits = 0
    hindsight_checked = 0
    # Resolution half (answer_resolution): image-only vs. hop plans.
    in_image = 0
    hop_type_dist: Counter[str] = Counter()
    terminal_dist: Counter[str] = Counter()
    gap_dist: Counter[str] = Counter()
    belief_dist: Counter[str] = Counter()
    revision_count = 0
    policy_decisions = 0
    trace_lengths: Counter[int] = Counter()
    hops_per_trace: Counter[int] = Counter()
    lookup_leaks = 0

    for r in records:
        source_counts[str(r.get("source_dataset", "unknown"))] += 1
        usage_counts[str(r.get("data_usage", "train"))] += 1

        answer = str(r.get("answer", "") or "")
        if answer.strip():
            with_answer += 1
        else:
            without_answer += 1

        img = str(r.get("image_path", "") or "")
        if img and _resolve(img, root).exists():
            image_exists += 1
        else:
            image_missing += 1

        question_len_sum += len(str(r.get("question", "") or ""))

        if is_labeled:
            if r.get("parse_success"):
                parse_success += 1
            else:
                parse_failed += 1
            if r.get("validation_errors"):
                validation_error_records += 1
            if r.get("needs_human_review"):
                needs_review += 1
            if r.get("prompt_sha"):
                prompt_dist[f"{r.get('prompt_file','?')}@{r['prompt_sha']}"] += 1

            ann = r.get("annotation")
            if get_quality_flag(ann, "anchor_not_visually_verifiable"):
                anchor_not_verifiable += 1
            if any("hindsight leak" in str(e) for e in r.get("validation_errors") or []):
                lookup_leaks += 1
            if get_quality_flag(ann, "possible_hallucination"):
                possible_hallucination += 1

            if isinstance(ann, dict):
                n_ann += 1
                meta = ann.get("sample_meta") or {}
                qtype_dist[str(meta.get("question_type"))] += 1
                # The record-level copy is what resume keys on; prefer it.
                variant_dist[str(r.get("trace_variant")
                                 or meta.get("trace_variant"))] += 1

                trace = ann.get("search_trace") or []
                trace_lengths[len(trace)] += 1
                if trace and isinstance(trace[-1], dict):
                    terminal_dist[str(trace[-1].get("action"))] += 1
                for entry in (r.get("execution") or {}).get("steps", []):
                    if entry.get("planner"):
                        policy_decisions += 1
                        gap_dist[str((entry.get("request") or {}).get("gap_type"))] += 1
                n_recover = 0
                n_lookbacks = 0
                for st in trace:
                    if not isinstance(st, dict):
                        continue
                    update = st.get("belief_update") or {}
                    belief_dist[str(update.get("status"))] += 1
                    revision_count += update.get("revision_of_step") is not None
                    if st.get("action") == "recover":
                        n_recover += 1
                        pattern_dist[str(st.get("recovery_pattern"))] += 1
                    lb = st.get("lookback")
                    if isinstance(lb, dict):
                        n_lookbacks += 1
                        verdict = str(lb.get("verdict"))
                        verdict_dist[verdict] += 1
                        risk = lb.get("answer_flip_risk")
                        residual = lb.get("residual_constraints") or []
                        if verdict in ALLOWED_VERDICTS \
                                and verdict != expected_verdict(risk, residual,
                                    (ann.get("sample_meta") or {}).get("trace_variant", "default")):
                            rule_violations += 1
                recover_count_dist[n_recover] += 1
                lookbacks_per_trace_sum += n_lookbacks

                res = ann.get("answer_resolution") or {}
                if res.get("resolved_in_image") is True:
                    in_image += 1
                hops = res.get("external_hops") or []
                hops_per_trace[len(hops)] += 1
                for h in hops:
                    if isinstance(h, dict):
                        hop_type_dist[str(h.get("type"))] += 1

                # Descriptive localization overlap; it does not establish leakage.
                final = (ann.get("answer_grounding") or {}).get("final_target_bbox")
                first = next(
                    (st.get("focus_bbox") for st in trace
                     if isinstance(st, dict)
                     and st.get("action") in ("inspect_candidate", "inspect_region")
                     and st.get("focus_bbox")),
                    None,
                )
                if first and isinstance(final, list) and final \
                        and isinstance(final[0], list) and len(final[0]) == 4:
                    fx = (final[0][0] + final[0][2]) / 2
                    fy = (final[0][1] + final[0][3]) / 2
                    hindsight_checked += 1
                    if first[0] <= fx <= first[2] and first[1] <= fy <= first[3]:
                        hindsight_hits += 1

    avg_q_len = question_len_sum / total if total else 0.0

    print(f"\n{'=' * 60}")
    print(f"File: {args.input}")
    print(f"Type: {'LABELED' if is_labeled else 'NORMALIZED'}")
    print(f"{'=' * 60}")
    print(f"Total samples:                 {total}")
    print(f"Data usage:                    {dict(usage_counts)}")
    print("\nPer source_dataset:")
    for src, n in sorted(source_counts.items()):
        print(f"  {src:<28} {n:>6}  ({100 * n / (total or 1):>5.1f}%)")
    if source_counts:
        top, top_n = source_counts.most_common(1)[0]
        share = top_n / (total or 1)
        if len(source_counts) > 1 and share > HEALTHY["max_source_share"]:
            print(f"  WARNING: {top} is {share:.0%} of this file (> "
                  f"{HEALTHY['max_source_share']:.0%}) — it will dominate the "
                  "dataset's character, and the thin sources may not support a "
                  "per-benchmark breakdown. Cap it with Stage 1 --source-quota.")
    print(f"\nWith answer:                   {with_answer}")
    print(f"Without answer:                {without_answer}")
    print(f"Image path exists:             {image_exists}")
    print(f"Image path missing:            {image_missing}")
    print(f"Avg question length (chars):   {avg_q_len:.1f}")

    if is_labeled:
        print(f"\n--- Labeling ---")
        print(f"parse_success = true:          {parse_success}")
        print(f"parse_success = false:         {parse_failed}")
        print(f"records with validation errs:  {validation_error_records}")
        print(f"needs_human_review = true:     {needs_review}")
        print(f"anchor_not_visually_verifiable:{anchor_not_verifiable}")
        print(f"possible_hallucination = true: {possible_hallucination}")

        if prompt_dist:
            print(f"prompt(s) used:                {dict(prompt_dist)}")
            if len(prompt_dist) > 1:
                print("  WARNING: this file mixes multiple prompts — A/B "
                      "comparisons over it are not valid")

        print(f"\n--- Distribution health ---")
        print(f"recover count per trace:       "
              f"{dict(sorted(recover_count_dist.items()))}")
        if n_ann:
            zero = recover_count_dist.get(0, 0)
            print(f"  0-recovery share:            {zero / n_ann:.0%}")
        print(f"recovery_pattern:              {dict(pattern_dist)}")
        print(f"lookback verdicts:             {dict(verdict_dist)}")
        n_verd = sum(verdict_dist.values())
        if n_verd:
            print(f"  skip share:                  {verdict_dist.get('skip', 0) / n_verd:.0%}")
        if n_ann:
            print(f"lookbacks per trace (avg):     {lookbacks_per_trace_sum / n_ann:.1f}")
        print(f"decision-rule violations:      {rule_violations}")
        print(f"question_type:                 {dict(qtype_dist)}")

        print(f"trace_variant:                 {dict(variant_dist)}")
        print(f"\n--- Behavior coverage (descriptive) ---")
        print(f"trace lengths:                 {dict(sorted(trace_lengths.items()))}")
        print(f"autonomous policy decisions:   {policy_decisions}")
        print(f"pre-action evidence gaps:      {dict(gap_dist)}")
        print(f"belief updates:                {dict(belief_dist)}")
        print(f"linked hypothesis revisions:   {revision_count}")
        print(f"terminal actions:              {dict(terminal_dist)}")
        print(f"\n--- Answer resolution ---")
        if n_ann:
            print(f"resolved in image:             {in_image}/{n_ann} "
                  f"({in_image / n_ann:.0%})")
        print(f"external hops per trace:       {dict(sorted(hops_per_trace.items()))}")
        print(f"hop types:                     {dict(hop_type_dist)}")
        print(f"lookup queries leaking answer: {lookup_leaks} "
              "(should be 0; these records fail validation and are retried)")
        if hindsight_checked:
            print(f"localization statistic (first inspected region contains final "
                  f"target): {hindsight_hits}/{hindsight_checked} = "
                  f"{hindsight_hits / hindsight_checked:.0%}")
            print("  (descriptive only; this does not prove hindsight or causal necessity)")
        admission = Counter(training_rejection_reason(r, allow_abstention=True) or "accepted" for r in records)
        analysis = Counter(training_rejection_reason(r, allow_evaluation=True) or "accepted" for r in records)
        print(f"\nCausal-call training admission: {dict(admission)}")
        print(f"Resolved-record analysis admission: {dict(analysis)}")
    print()


if __name__ == "__main__":
    main()
