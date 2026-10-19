"""
AI-COS - Evaluation Metrics Module (evaluation/metrics.py)
Computes Precision, Recall, F1, Precision@K, Recall@K, and
confusion matrix for SIPA and ASCD ground-truth evaluations.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import math


# ---------------------------------------------------------------------------
# Core binary classification metrics
# ---------------------------------------------------------------------------

def precision(tp: int, fp: int) -> float:
    return round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    return round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0


def f1_score(prec: float, rec: float) -> float:
    return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0


def precision_at_k(predicted: List[str], expected: Set[str], k: int) -> float:
    top_k = predicted[:k]
    hits = sum(1 for doc in top_k if doc in expected)
    return round(hits / k, 4) if k > 0 else 0.0


def recall_at_k(predicted: List[str], expected: Set[str], k: int) -> float:
    top_k = predicted[:k]
    hits = sum(1 for doc in top_k if doc in expected)
    return round(hits / len(expected), 4) if expected else 0.0


def average_precision(predicted: List[str], expected: Set[str]) -> float:
    """Compute Average Precision (AP) for a ranked list."""
    if not expected:
        return 0.0
    score = 0.0
    hits = 0
    for i, doc in enumerate(predicted, 1):
        if doc in expected:
            hits += 1
            score += hits / i
    return round(score / len(expected), 4)


# ---------------------------------------------------------------------------
# SIPA evaluation
# ---------------------------------------------------------------------------

def evaluate_sipa_scenario(
    predicted_affected: List[str],
    expected_affected: List[str],
    priority_cutoff: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate SIPA predictions for one scenario.

    Parameters
    ----------
    predicted_affected : ranked list of doc_series_codes (high → low impact)
    expected_affected  : ground-truth affected documents
    priority_cutoff    : if not None, only consider top-K = len(expected) predictions

    Returns
    -------
    Dict with tp, fp, fn, precision, recall, f1, ap, p@k, r@k
    """
    expected_set = set(expected_affected)
    pred_set = set(predicted_affected)

    tp = len(expected_set & pred_set)
    fp = len(pred_set - expected_set)
    fn = len(expected_set - pred_set)

    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    ap = average_precision(predicted_affected, expected_set)

    k = len(expected_set)
    pk = precision_at_k(predicted_affected, expected_set, k)
    rk = recall_at_k(predicted_affected, expected_set, k)

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'average_precision': ap,
        'precision_at_k': pk,
        'recall_at_k': rk,
        'k': k,
    }


def aggregate_sipa_results(scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate SIPA metrics across multiple scenarios."""
    if not scenario_results:
        return {'mean_ap': 0.0, 'mean_f1': 0.0, 'mean_precision': 0.0, 'mean_recall': 0.0}

    n = len(scenario_results)
    return {
        'num_scenarios': n,
        'mean_precision': round(sum(r['precision'] for r in scenario_results) / n, 4),
        'mean_recall': round(sum(r['recall'] for r in scenario_results) / n, 4),
        'mean_f1': round(sum(r['f1'] for r in scenario_results) / n, 4),
        'mean_ap': round(sum(r['average_precision'] for r in scenario_results) / n, 4),
        'total_tp': sum(r['true_positives'] for r in scenario_results),
        'total_fp': sum(r['false_positives'] for r in scenario_results),
        'total_fn': sum(r['false_negatives'] for r in scenario_results),
    }


# ---------------------------------------------------------------------------
# ASCD evaluation
# ---------------------------------------------------------------------------

def evaluate_ascd_case(
    detected: bool,
    expected: bool,
) -> str:
    """Map predicted/expected booleans to TP/FP/TN/FN label."""
    if expected and detected:
        return 'TP'
    if expected and not detected:
        return 'FN'
    if not expected and detected:
        return 'FP'
    return 'TN'


def compute_confusion_matrix(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute confusion matrix from a list of ASCD evaluation cases.
    Each case dict must have 'expected' (bool) and 'detected' (bool).
    """
    tp = fp = tn = fn = 0
    for c in cases:
        label = evaluate_ascd_case(c['detected'], c['expected'])
        if label == 'TP': tp += 1
        elif label == 'FP': fp += 1
        elif label == 'TN': tn += 1
        else: fn += 1

    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    fp_rate = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0

    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'false_positive_rate': fp_rate,
        'total_cases': len(cases),
    }


def aggregate_ascd_results(category_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate ASCD metrics across conflict categories."""
    overall = []
    per_category: Dict[str, Dict[str, Any]] = {}
    for cat, cases in category_results.items():
        cm = compute_confusion_matrix(cases)
        per_category[cat] = cm
        overall.extend(cases)
    overall_cm = compute_confusion_matrix(overall)
    return {'overall': overall_cm, 'per_category': per_category}
