#!/usr/bin/env python3
"""
Preliminary Viability Experiment Runner
========================================
Run this BEFORE committing to the full experiment.
Tests whether each component of the pipeline works.

Usage:
    python run_viability.py                    # Run all checks
    python run_viability.py --check data       # Just check data loading
    python run_viability.py --check perturb    # Just check perturbation generation
    python run_viability.py --check judge      # Just check judge scoring (requires API)
    python run_viability.py --check metrics    # Just check metric computation

Checks run in order:
    V0: Data Loading — Can we load and parse ABCD correctly?
    V1: Perturbation Generation — Can we generate each perturbation type?
    V2: Perturbation Quality — Are perturbations well-formed?
    V3: Judge Score Variance — Do scores have enough variance? (requires API)
    V4: Prompt Sensitivity — Do prompt conditions produce different results? (requires API)
    V5: Position Bias — Is there ordering bias in pairwise? (requires API)
    V6: Metric Computation — Can we compute all metrics on synthetic data?
"""
import argparse
import json
import os
import sys
import random
from datetime import datetime
from typing import Dict, List

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ABCD_DIR, OUTPUT_DIR, VIABILITY_SAMPLE_SIZE, PERTURBATIONS_PER_FAMILY,
    MIN_TURNS, RANDOM_SEED, JUDGE_MODELS, PROMPT_CONDITIONS,
)
from utils.abcd_loader import ABCDDataset
from utils.perturbation_engine import PerturbationEngine
from utils.metrics import (
    compute_detection_rate, compute_delta_distribution,
    compute_false_positive_rate, compute_gwet_ac1,
    check_score_variance, check_prompt_sensitivity, check_position_bias,
    JudgeOutput,
)


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_result(name: str, passed: bool, detail: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")


# ═══════════════════════════════════════════
# V0: DATA LOADING
# ═══════════════════════════════════════════

def check_data_loading() -> dict:
    """
    V0: Verify we can load all ABCD files and parse them correctly.
    This is the foundation — if this fails, nothing else works.
    """
    print_header("V0: DATA LOADING CHECK")
    results = {"pass": True, "details": {}}

    # Check files exist
    required_files = {
        "abcd_v1.1.json or .gz": [
            os.path.join(ABCD_DIR, "abcd_v1.1.json"),
            os.path.join(ABCD_DIR, "abcd_v1.1.json.gz"),
        ],
        "kb.json": [os.path.join(ABCD_DIR, "kb.json")],
        "guidelines.json": [os.path.join(ABCD_DIR, "guidelines.json")],
        "ontology.json": [os.path.join(ABCD_DIR, "ontology.json")],
    }

    for name, paths in required_files.items():
        found = any(os.path.exists(p) for p in paths)
        print_result(f"File exists: {name}", found,
                     paths[0] if not found else "")
        results["details"][f"file_{name}"] = found
        if not found:
            results["pass"] = False

    if not results["pass"]:
        print("\n  Cannot proceed without required files.")
        return results

    # Load dataset
    try:
        dataset = ABCDDataset(ABCD_DIR)
        results["dataset"] = dataset
    except Exception as e:
        print_result("Dataset loading", False, str(e))
        results["pass"] = False
        return results

    # Check basic counts
    total = len(dataset.conversations)
    print_result(f"Total conversations loaded: {total}", total > 0)
    results["details"]["total_conversations"] = total

    # Check kb.json has subflow mappings
    kb_count = len(dataset.kb)
    print_result(f"kb.json subflow mappings: {kb_count}", kb_count > 0)
    results["details"]["kb_subflows"] = kb_count

    # Check guidelines.json loaded
    guide_count = len(dataset.guidelines)
    print_result(f"guidelines.json entries: {guide_count}", guide_count > 0)
    results["details"]["guidelines_entries"] = guide_count

    # Check compliant conversations
    compliant = dataset.get_compliant_conversations(min_turns=MIN_TURNS)
    print_result(
        f"Policy-compliant conversations (>={MIN_TURNS} turns): {len(compliant)}",
        len(compliant) >= VIABILITY_SAMPLE_SIZE,
        f"Need at least {VIABILITY_SAMPLE_SIZE} for viability testing"
    )
    results["details"]["compliant_conversations"] = len(compliant)

    # Check subflow distribution
    dist = dataset.get_subflow_distribution()
    print(f"\n  Top 5 subflows by compliant conversation count:")
    for sf, count in list(dist.items())[:5]:
        seq = dataset.get_action_sequence(sf)
        print(f"    {sf}: {count} convos, sequence: {seq}")

    # Check that kb.json sequences are parseable
    parseable_seqs = sum(1 for sf in dataset.kb if len(dataset.kb[sf]) >= 2)
    print_result(
        f"kb.json sequences with 2+ actions: {parseable_seqs}",
        parseable_seqs >= 10,
    )

    # Check slot value coverage
    if compliant:
        sample = compliant[0]
        has_slots = bool(sample.slot_values)
        print_result(
            f"Sample conversation has slot values: {len(sample.slot_values)} slots",
            has_slots,
        )
        if has_slots:
            for k, v in list(sample.slot_values.items())[:5]:
                print(f"    {k}: {v}")

    # Check policy retrieval
    if compliant:
        sample = compliant[0]
        policy = dataset.get_policy_for_subflow(sample.subflow)
        print_result(
            f"Policy text retrievable for '{sample.subflow}'",
            policy is not None,
            f"Length: {len(policy)} chars" if policy else "No policy found"
        )
        if policy:
            print(f"    Preview: {policy[:150]}...")

    results["pass"] = results["pass"] and len(compliant) >= VIABILITY_SAMPLE_SIZE
    return results


# ═══════════════════════════════════════════
# V1: PERTURBATION GENERATION
# ═══════════════════════════════════════════

def check_perturbation_generation(dataset: ABCDDataset) -> dict:
    """
    V1: Verify we can generate perturbations of each type.
    """
    print_header("V1: PERTURBATION GENERATION CHECK")
    results = {"pass": True, "details": {}}

    engine = PerturbationEngine(dataset, seed=RANDOM_SEED)
    compliant = dataset.get_compliant_conversations(min_turns=MIN_TURNS)

    if not compliant:
        print_result("Compliant conversations available", False)
        results["pass"] = False
        return results

    sample = compliant[:VIABILITY_SAMPLE_SIZE]

    # Generate viability set
    viability = engine.generate_viability_set(sample, per_family=PERTURBATIONS_PER_FAMILY)

    # Report per family
    for family, perts in viability.items():
        target = PERTURBATIONS_PER_FAMILY
        count = len(perts)
        passed = count >= max(1, target // 2)  # At least half the target
        print_result(
            f"Family {family}: {count}/{target} perturbations generated", passed
        )
        results["details"][f"family_{family}_count"] = count

        if not passed:
            results["pass"] = False

        # Show examples
        for p in perts[:2]:
            print(f"    [{p.perturbation_type}] {p.description}")
            print(f"      Deterministic: {p.is_deterministic}")
            if p.overlap_tags:
                print(f"      Overlaps: {p.overlap_tags}")

    # Check deterministic ratio
    all_perts = [p for perts in viability.values() for p in perts]
    deterministic = sum(1 for p in all_perts if p.is_deterministic)
    total = len(all_perts)
    det_ratio = deterministic / total if total > 0 else 0

    print_result(
        f"Deterministic perturbation ratio: {deterministic}/{total} ({det_ratio:.0%})",
        det_ratio >= 0.5,
        "Target: >= 50% deterministic"
    )
    results["details"]["deterministic_ratio"] = det_ratio

    # Save viability perturbations
    output_path = os.path.join(OUTPUT_DIR, "viability_perturbations.json")
    serializable = []
    for family, perts in viability.items():
        for p in perts:
            serializable.append({
                "convo_id": p.original_convo_id,
                "family": p.family,
                "type": p.perturbation_type,
                "description": p.description,
                "deterministic": p.is_deterministic,
                "overlap_tags": p.overlap_tags,
                "ground_truth_check": p.ground_truth_check,
                "violated_rule": p.violated_rule,
                "severity": p.severity,
                "evidence_distance": p.evidence_distance,
                "changed_slot": p.changed_slot,
                "original_value": p.original_value,
                "perturbed_value": p.perturbed_value,
            })

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved viability perturbations to: {output_path}")

    results["viability_set"] = viability
    results["engine"] = engine
    return results


# ═══════════════════════════════════════════
# V2: PERTURBATION QUALITY
# ═══════════════════════════════════════════

def check_perturbation_quality(viability_set: dict, dataset: ABCDDataset) -> dict:
    """
    V2: Check perturbation quality (automated checks).
    Manual quality checks should be done separately.
    """
    print_header("V2: PERTURBATION QUALITY CHECK (Automated)")
    results = {"pass": True, "details": {}}

    all_perts = [p for perts in viability_set.values() for p in perts]

    # Check 1: Every perturbation has a non-empty description
    has_desc = sum(1 for p in all_perts if p.description)
    print_result(
        f"All perturbations have descriptions: {has_desc}/{len(all_perts)}",
        has_desc == len(all_perts),
    )

    # Check 2: Every perturbation has a ground truth check
    has_gt = sum(1 for p in all_perts if p.ground_truth_check)
    print_result(
        f"All perturbations have ground truth checks: {has_gt}/{len(all_perts)}",
        has_gt == len(all_perts),
    )

    # Check 3: Family W perturbations have perturbed action sequences
    w_perts = viability_set.get("W", [])
    w_with_seq = sum(1 for p in w_perts if p.perturbed_action_sequence)
    print_result(
        f"Family W perturbations have action sequences: {w_with_seq}/{len(w_perts)}",
        w_with_seq == len(w_perts) or len(w_perts) == 0,
    )

    # Check 4: Family W sequences actually differ from ground truth
    w_different = 0
    for p in w_perts:
        conv = dataset.conversations.get(p.original_convo_id)
        if conv and p.perturbed_action_sequence != conv.ground_truth_sequence:
            w_different += 1
    print_result(
        f"Family W sequences differ from kb.json: {w_different}/{len(w_perts)}",
        w_different == len(w_perts) or len(w_perts) == 0,
    )

    # Check 5: Family V perturbations have changed values
    v_perts = viability_set.get("V", [])
    v_with_change = sum(1 for p in v_perts if p.original_value and p.perturbed_value
                        and p.original_value != p.perturbed_value)
    print_result(
        f"Family V values actually changed: {v_with_change}/{len(v_perts)}",
        v_with_change == len(v_perts) or len(v_perts) == 0,
    )

    # Check 6: Overlap tags are present where expected
    overlapping = sum(1 for p in all_perts if p.overlap_tags)
    print(f"\n  Perturbations with overlap tags: {overlapping}/{len(all_perts)}")
    for p in all_perts:
        if p.overlap_tags:
            print(f"    [{p.family}:{p.perturbation_type}] overlaps with: {p.overlap_tags}")

    # Manual check reminder
    print(f"\n  ⚠ MANUAL CHECKS NEEDED:")
    print(f"    - Review {len(all_perts)} perturbations for naturalness")
    print(f"    - Verify Family R rule violations against guidelines.json")
    print(f"    - Check that perturbation descriptions match actual changes")
    print(f"    - Export: {os.path.join(OUTPUT_DIR, 'viability_perturbations.json')}")

    return results


# ═══════════════════════════════════════════
# V6: METRIC COMPUTATION (on synthetic data)
# ═══════════════════════════════════════════

def check_metric_computation() -> dict:
    """
    V6: Verify all metric functions work correctly on synthetic data.
    No API calls needed — tests the math.
    """
    print_header("V6: METRIC COMPUTATION CHECK (Synthetic Data)")
    results = {"pass": True, "details": {}}

    # Create synthetic judge outputs
    rng = random.Random(RANDOM_SEED)

    # Simulate: judge catches W well (80%), V poorly (50%), R moderately (65%)
    synthetic_originals = []
    synthetic_perturbed = []
    synthetic_pairwise = []

    families = {"W": 0.80, "V": 0.50, "R": 0.65, "N": 0.05}

    for family, detect_prob in families.items():
        for i in range(20):
            detected = rng.random() < detect_prob
            orig_score = rng.uniform(3.5, 5.0)
            pert_score = orig_score - rng.uniform(0.5, 2.0) if detected else orig_score + rng.uniform(-0.3, 0.3)
            pert_score = max(1.0, min(5.0, pert_score))

            cid = f"synth_{family}_{i}"

            orig = JudgeOutput(
                convo_id=cid, is_original=True, perturbation_family=family,
                perturbation_type="synthetic", model="test_model",
                prompt_condition="no_policy",
                workflow_correctness=orig_score if family == "W" else rng.uniform(3, 5),
                value_accuracy=orig_score if family == "V" else rng.uniform(3, 5),
                rule_compliance=orig_score if family == "R" else rng.uniform(3, 5),
                overall_quality=orig_score,
            )

            pert = JudgeOutput(
                convo_id=cid, is_original=False, perturbation_family=family,
                perturbation_type="synthetic", model="test_model",
                prompt_condition="no_policy",
                workflow_correctness=pert_score if family == "W" else rng.uniform(3, 5),
                value_accuracy=pert_score if family == "V" else rng.uniform(3, 5),
                rule_compliance=pert_score if family == "R" else rng.uniform(3, 5),
                overall_quality=pert_score,
            )

            synthetic_originals.append(orig)
            synthetic_perturbed.append(pert)

            synthetic_pairwise.append({
                "convo_id": cid,
                "family": family,
                "perturbation_type": "synthetic",
                "model": "test_model",
                "prompt_condition": "no_policy",
                "judge_preferred": "original" if detected else "perturbed",
            })

    # Test M2: Detection Rate
    try:
        det_rates, pairs = compute_detection_rate(
            synthetic_originals, synthetic_perturbed, group_by="family"
        )
        print_result("M2: Detection rate computation", True)
        for family, stats in det_rates.items():
            expected = families.get(family, 0)
            print(f"    Family {family}: {stats['detection_rate']:.0%} "
                  f"(expected ~{expected:.0%}, n={stats['n']})")
    except Exception as e:
        print_result("M2: Detection rate computation", False, str(e))
        results["pass"] = False

    # Test M3: Delta Distribution
    try:
        deltas = compute_delta_distribution(pairs, group_by="family")
        print_result("M3: Delta distribution computation", True)
        for family, stats in deltas.items():
            print(f"    Family {family}: mean={stats['mean']:.2f}, "
                  f"std={stats['std']:.2f}, n={stats['n']}")
    except Exception as e:
        print_result("M3: Delta distribution computation", False, str(e))
        results["pass"] = False

    # Test M4: False Positive Rate
    try:
        control_pairs = [p for p in pairs if p["family"] == "N"]
        fp = compute_false_positive_rate(control_pairs)
        print_result("M4: False positive rate computation", True)
        for ptype, stats in fp.items():
            print(f"    {ptype}: FP rate={stats['fp_rate']:.0%}, n={stats['n']}")
    except Exception as e:
        print_result("M4: False positive rate computation", False, str(e))
        results["pass"] = False

    # Test M7: Gwet's AC1
    try:
        rater1 = [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
        rater2 = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0]
        ac1 = compute_gwet_ac1(rater1, rater2)
        print_result("M7: Gwet's AC1 computation", True,
                     f"AC1={ac1['ac1']:.3f}, agreement={ac1['observed_agreement']:.3f}")
    except Exception as e:
        print_result("M7: Gwet's AC1 computation", False, str(e))
        results["pass"] = False

    # Test Score Variance Check
    try:
        all_scores = [o.overall_quality for o in synthetic_originals + synthetic_perturbed]
        var_check = check_score_variance(all_scores)
        print_result(f"Score variance check: {var_check['verdict']}", var_check["pass"],
                     f"mean={var_check['mean']}, std={var_check['std']}")
    except Exception as e:
        print_result("Score variance check", False, str(e))

    return results


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run viability checks")
    parser.add_argument("--check", type=str, default="all",
                        choices=["all", "data", "perturb", "judge", "metrics"],
                        help="Which check to run")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  PRELIMINARY VIABILITY EXPERIMENTS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
    }

    # V0: Data Loading
    if args.check in ["all", "data"]:
        v0 = check_data_loading()
        report["checks"]["V0_data_loading"] = {
            "pass": v0["pass"],
            "details": v0.get("details", {}),
        }
        if not v0["pass"] and args.check == "all":
            print("\n\n  ✗ Data loading failed. Fix data paths before continuing.")
            save_report(report)
            return

    # V1: Perturbation Generation
    if args.check in ["all", "perturb"]:
        if "dataset" not in v0:
            print("\n  Skipping V1 — dataset not loaded")
        else:
            v1 = check_perturbation_generation(v0["dataset"])
            report["checks"]["V1_perturbation_generation"] = {
                "pass": v1["pass"],
                "details": v1.get("details", {}),
            }

            # V2: Perturbation Quality
            if "viability_set" in v1:
                v2 = check_perturbation_quality(v1["viability_set"], v0["dataset"])
                report["checks"]["V2_perturbation_quality"] = {
                    "pass": v2["pass"],
                }

    # V6: Metric Computation
    if args.check in ["all", "metrics"]:
        v6 = check_metric_computation()
        report["checks"]["V6_metric_computation"] = {"pass": v6["pass"]}

    # V3-V5: Judge checks (require API)
    if args.check in ["all", "judge"]:
        print_header("V3-V5: JUDGE CHECKS")
        print("  These checks require a configured LLM API.")
        print("  Configure call_judge_api() in judge_runner.py first.")
        print("  Then run: python run_judge_viability.py")
        report["checks"]["V3_V5_judge_checks"] = {"pass": "NOT_RUN", "details": "Requires API"}

    # Save report
    save_report(report)

    # Print summary
    print_header("VIABILITY SUMMARY")
    all_pass = True
    for check_name, check_result in report["checks"].items():
        status = check_result.get("pass", "UNKNOWN")
        if status == True:
            print(f"  ✓ {check_name}")
        elif status == False:
            print(f"  ✗ {check_name}")
            all_pass = False
        else:
            print(f"  ? {check_name} ({status})")

    if all_pass:
        print(f"\n  All automated checks passed.")
        print(f"  Next steps:")
        print(f"    1. Review perturbations manually (see {OUTPUT_DIR}/viability_perturbations.json)")
        print(f"    2. Configure judge API in judge_runner.py")
        print(f"    3. Run judge viability checks: python run_judge_viability.py")
    else:
        print(f"\n  Some checks failed. Review and fix before proceeding.")


def save_report(report: dict):
    output_path = os.path.join(OUTPUT_DIR, "viability_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to: {output_path}")


if __name__ == "__main__":
    main()
