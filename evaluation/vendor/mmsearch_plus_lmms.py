"""Pinned lmms-eval MMSearch-Plus scoring functions, copied verbatim.
Adopted from https://github.com/EvolvingLMMs-Lab/lmms-eval.
Copyright (c) 2025 LMMs-Lab. Licensed under Apache License 2.0.
See mmsearch_plus_lmms.LICENSE for the upstream license notice.
Only imports and the already-decrypted input adapter are supplied locally.
"""
import logging
import re
from typing import Any, Dict, List

eval_logger = logging.getLogger(__name__)
CANARY = 'MMSearch-Plus'


def decrypt_sample(doc, canary):
    """Frozen project references are already decrypted; no gold data is changed."""
    if canary != CANARY:
        raise ValueError('Unexpected dataset canary')
    return doc

UPSTREAM = {'repository': 'https://github.com/EvolvingLMMs-Lab/lmms-eval', 'revision': '593851645597652776b39c812f99064b35aa29f4', 'files': {'lmms_eval/tasks/mmsearch_plus/utils.py': {'url': 'https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/593851645597652776b39c812f99064b35aa29f4/lmms_eval/tasks/mmsearch_plus/utils.py', 'sha256': 'dd84816645a8e90058bbb221db60a07cf575e4dbceb373b1e8b572d4584cfc11'}, 'lmms_eval/tasks/mmsearch_plus/mmsearch_plus_vqa.yaml': {'url': 'https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/593851645597652776b39c812f99064b35aa29f4/lmms_eval/tasks/mmsearch_plus/mmsearch_plus_vqa.yaml', 'sha256': 'afb02619954f1040a0e87d4e4080a64a0cce83f5a5094cdff2da25e7b1b5d542'}, 'LICENSE': {'url': 'https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/593851645597652776b39c812f99064b35aa29f4/LICENSE', 'sha256': '17358b5caadacb8108f14a6a319779cc8a20feed699f79c59e396e7e9dd5c33b'}}}


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    # Convert to lowercase
    answer = answer.lower().strip()

    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)

    # Remove extra whitespace
    answer = " ".join(answer.split())

    return answer


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score (0-1)
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    # Calculate precision and recall
    common = set(pred_tokens) & set(gt_tokens)

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    # Calculate F1
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score (1 or 0).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    return 1.0 if pred_norm == gt_norm else 0.0


def mmsearch_plus_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, Any]:
    """
    Process model results and compute metrics.

    Args:
        doc: Document containing ground truth
        result: Model predictions

    Returns:
        Dictionary containing metrics
    """
    # Decrypt the document
    doc = decrypt_sample(doc, CANARY)

    if not result or len(result) == 0:
        eval_logger.warning("Empty result received")
        return {
            "f1_score": 0.0,
            "exact_match": 0.0,
        }

    prediction = result[0].strip()

    # Get ground truth answers
    gt_answers = doc.get("answer", [])
    if isinstance(gt_answers, str):
        gt_answers = [gt_answers]

    if not gt_answers:
        eval_logger.warning("No ground truth answers found")
        return {
            "f1_score": 0.0,
            "exact_match": 0.0,
        }

    # Compute max score across all valid answers
    max_f1 = 0.0
    max_em = 0.0

    for gt_answer in gt_answers:
        f1 = compute_f1_score(prediction, gt_answer)
        em = compute_exact_match(prediction, gt_answer)

        max_f1 = max(max_f1, f1)
        max_em = max(max_em, em)

    return {
        "f1_score": max_f1,
        "exact_match": max_em,
    }
