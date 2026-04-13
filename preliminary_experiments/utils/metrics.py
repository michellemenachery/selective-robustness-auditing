"""
Metrics Computation Module
Implements all 7 primary metrics from the proposal with exact formulas.
"""
import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class JudgeOutput:
    """Output from a single judge evaluation."""
    convo_id: str
    is_original: bool                  # True = original, False = perturbed
    perturbation_family: str           # "W", "V", "R", "N"
    perturbation_type: str
    model: str
    prompt_condition: str              # "no_policy", "compressed_policy", "full_policy"

    # Pointwise scores (1-5)
    workflow_correctness: float = 0.0
    value_accuracy: float = 0.0
    rule_compliance: float = 0.0
    overall_quality: float = 0.0

    # Rationale
    rationale: str = ""

    # Pairwise (if applicable)
    pairwise_preferred: str = ""       # "original" or "perturbed"
    pairwise_confidence: str = ""


def parse_judge_response(response_text: str) -> dict:
    """
    Parse JSON response from a judge LLM.
    Handles common failure modes (markdown fences, partial JSON).
    """
    text = response_text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from the response
    import re
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return empty
    return {}


# ═══════════════════════════════════════════
# M1: PAIRWISE PREFERENCE ACCURACY
# ═══════════════════════════════════════════

def compute_pairwise_accuracy(
    pairwise_results: List[dict],
    group_by: str = None,
) -> Dict[str, dict]:
    """
    Compute pairwise preference accuracy.

    Args:
        pairwise_results: List of dicts with keys:
            - convo_id, family, perturbation_type, model, prompt_condition
            - judge_preferred: "original" or "perturbed" or "tie"
        group_by: Optional grouping key ("family", "model", "prompt_condition")

    Returns:
        Dict with accuracy, n, and confidence interval per group.
    """
    def _compute(items):
        n = len(items)
        if n == 0:
            return {"accuracy": 0.0, "n": 0, "ci_lower": 0.0, "ci_upper": 0.0}

        correct = sum(
            1 if item["judge_preferred"] == "original"
            else 0.5 if item["judge_preferred"] == "tie"
            else 0
            for item in items
        )
        acc = correct / n

        # Wilson confidence interval
        z = 1.96
        denom = 1 + z**2 / n
        center = (acc + z**2 / (2 * n)) / denom
        spread = z * math.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n) / denom

        return {
            "accuracy": round(acc, 4),
            "n": n,
            "ci_lower": round(max(0, center - spread), 4),
            "ci_upper": round(min(1, center + spread), 4),
        }

    if group_by is None:
        return {"overall": _compute(pairwise_results)}

    groups = {}
    for item in pairwise_results:
        key = item.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    return {k: _compute(v) for k, v in groups.items()}


# ═══════════════════════════════════════════
# M2: POINTWISE DETECTION RATE
# ═══════════════════════════════════════════

def compute_detection_rate(
    original_scores: List[JudgeOutput],
    perturbed_scores: List[JudgeOutput],
    group_by: str = None,
) -> Dict[str, dict]:
    """
    Compute pointwise detection rate.
    Detection = score(original) > score(perturbed) on the RELEVANT dimension.
    Ties count as non-detection (conservative).

    Args:
        original_scores: JudgeOutput for original dialogues
        perturbed_scores: JudgeOutput for corresponding perturbed dialogues
    """
    # Match by convo_id, model, and prompt_condition
    pairs = []
    pert_lookup = {}
    for p in perturbed_scores:
        key = (p.convo_id, p.model, p.prompt_condition)
        pert_lookup[key] = p

    for o in original_scores:
        key = (o.convo_id, o.model, o.prompt_condition)
        if key in pert_lookup:
            p = pert_lookup[key]
            # Select the RELEVANT dimension based on perturbation family
            if p.perturbation_family == "W":
                orig_score = o.workflow_correctness
                pert_score = p.workflow_correctness
            elif p.perturbation_family == "V":
                orig_score = o.value_accuracy
                pert_score = p.value_accuracy
            elif p.perturbation_family == "R":
                orig_score = o.rule_compliance
                pert_score = p.rule_compliance
            else:  # N (nuisance)
                orig_score = o.overall_quality
                pert_score = p.overall_quality

            detected = 1 if orig_score > pert_score else 0  # Ties = non-detection

            pairs.append({
                "convo_id": o.convo_id,
                "family": p.perturbation_family,
                "perturbation_type": p.perturbation_type,
                "model": o.model,
                "prompt_condition": o.prompt_condition,
                "detected": detected,
                "orig_score": orig_score,
                "pert_score": pert_score,
                "delta": orig_score - pert_score,
            })

    def _rate(items):
        n = len(items)
        if n == 0:
            return {"detection_rate": 0.0, "n": 0}
        det = sum(i["detected"] for i in items)
        rate = det / n
        z = 1.96
        denom = 1 + z**2 / n
        center = (rate + z**2 / (2 * n)) / denom
        spread = z * math.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denom
        return {
            "detection_rate": round(rate, 4),
            "n": n,
            "detected": det,
            "ci_lower": round(max(0, center - spread), 4),
            "ci_upper": round(min(1, center + spread), 4),
        }

    if group_by is None:
        return {"overall": _rate(pairs)}, pairs

    groups = {}
    for item in pairs:
        key = item.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    return {k: _rate(v) for k, v in groups.items()}, pairs


# ═══════════════════════════════════════════
# M3: SCORE DELTA DISTRIBUTION
# ═══════════════════════════════════════════

def compute_delta_distribution(
    pairs: List[dict],
    group_by: str = "family",
) -> Dict[str, dict]:
    """
    Compute score delta distribution statistics.
    Uses the pairs output from compute_detection_rate.
    """
    groups = {}
    for item in pairs:
        key = item.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(item["delta"])

    results = {}
    for key, deltas in groups.items():
        arr = np.array(deltas)
        results[key] = {
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "q25": round(float(np.percentile(arr, 25)), 4),
            "q75": round(float(np.percentile(arr, 75)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "n": len(deltas),
        }

    return results


# ═══════════════════════════════════════════
# M4: FALSE POSITIVE RATE
# ═══════════════════════════════════════════

def compute_false_positive_rate(
    control_pairs: List[dict],
    group_by: str = "perturbation_type",
) -> Dict[str, dict]:
    """
    Compute FP rate on nuisance and matched controls.
    FP = judge assigns higher score to original than to control (penalizes the control).
    """
    groups = {}
    for item in control_pairs:
        key = item.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    results = {}
    for key, items in groups.items():
        n = len(items)
        fp = sum(1 for i in items if i["detected"] == 1)  # "detected" a non-existent violation
        results[key] = {
            "fp_rate": round(fp / n, 4) if n > 0 else 0.0,
            "n": n,
            "false_positives": fp,
        }

    return results


# ═══════════════════════════════════════════
# M7: GWET'S AC1
# ═══════════════════════════════════════════

def compute_gwet_ac1(
    rater1: List[int],
    rater2: List[int],
) -> dict:
    """
    Compute Gwet's AC1 for two binary raters.
    rater1, rater2: lists of 0s and 1s (same length).

    Returns AC1 value and the agreement table.
    """
    assert len(rater1) == len(rater2), "Rater lists must have same length"
    n = len(rater1)
    if n == 0:
        return {"ac1": 0.0, "n": 0}

    # Build agreement table
    a = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 1 and r2 == 1)
    b = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 1 and r2 == 0)
    c_val = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 0 and r2 == 1)
    d = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 0 and r2 == 0)

    # Observed agreement
    p_a = (a + d) / n

    # Gwet's chance agreement
    p1 = ((a + b) / n + (a + c_val) / n) / 2
    p_e = 2 * p1 * (1 - p1)

    # AC1
    if p_e == 1.0:
        ac1 = 1.0
    else:
        ac1 = (p_a - p_e) / (1 - p_e)

    return {
        "ac1": round(ac1, 4),
        "observed_agreement": round(p_a, 4),
        "chance_agreement": round(p_e, 4),
        "n": n,
        "agreement_table": {"a": a, "b": b, "c": c_val, "d": d},
    }


# ═══════════════════════════════════════════
# VIABILITY CHECK METRICS
# ═══════════════════════════════════════════

def check_score_variance(scores: List[float]) -> dict:
    """
    Viability check V2: Is there enough variance in scores?
    If SD < 0.8, the rubric is too coarse.
    """
    arr = np.array(scores)
    return {
        "mean": round(float(np.mean(arr)), 3),
        "std": round(float(np.std(arr)), 3),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "pass": float(np.std(arr)) > 0.8,
        "verdict": "PASS — sufficient variance" if float(np.std(arr)) > 0.8
                   else "FAIL — scores too clustered; revise rubric",
    }


def check_prompt_sensitivity(
    results_by_condition: Dict[str, List[dict]],
) -> dict:
    """
    Viability check V3: Do prompt conditions produce different results?
    At least 20% of pairs should show different detection outcomes across conditions.
    """
    # Compare no_policy vs full_policy on the same pairs
    no_policy = {r["convo_id"]: r["detected"] for r in results_by_condition.get("no_policy", [])}
    full_policy = {r["convo_id"]: r["detected"] for r in results_by_condition.get("full_policy", [])}

    common_ids = set(no_policy.keys()) & set(full_policy.keys())
    if not common_ids:
        return {"pass": False, "verdict": "No overlapping pairs to compare"}

    different = sum(1 for cid in common_ids if no_policy[cid] != full_policy[cid])
    rate = different / len(common_ids)

    return {
        "pairs_compared": len(common_ids),
        "different_outcomes": different,
        "sensitivity_rate": round(rate, 3),
        "pass": rate >= 0.2,
        "verdict": f"PASS — {rate:.0%} sensitivity" if rate >= 0.2
                   else f"FAIL — only {rate:.0%} sensitivity; prompts not differentiated enough",
    }


def check_position_bias(
    results_a_first: List[dict],
    results_b_first: List[dict],
) -> dict:
    """
    Viability check V4: Is there position bias in pairwise evaluation?
    Preference should not differ by more than 15pp across positions.
    """
    pref_a_when_first = sum(1 for r in results_a_first if r["judge_preferred"] == "original") / max(len(results_a_first), 1)
    pref_a_when_second = sum(1 for r in results_b_first if r["judge_preferred"] == "original") / max(len(results_b_first), 1)
    gap = abs(pref_a_when_first - pref_a_when_second)

    return {
        "pref_when_position_a": round(pref_a_when_first, 3),
        "pref_when_position_b": round(pref_a_when_second, 3),
        "position_gap": round(gap, 3),
        "pass": gap < 0.15,
        "verdict": f"PASS — gap is {gap:.0%}" if gap < 0.15
                   else f"FAIL — {gap:.0%} position bias; average both orderings",
    }
