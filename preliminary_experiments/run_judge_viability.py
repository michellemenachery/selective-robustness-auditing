#!/usr/bin/env python3
"""
Judge Viability Checks (V3-V5)
Run these AFTER configuring your LLM API in judge_runner.py.

V3: Score Variance Check — Do judge scores have enough variance?
V4: Prompt Sensitivity Check — Do different prompts produce different results?
V5: Position Bias Check — Is there ordering bias in pairwise?

Usage:
    python run_judge_viability.py
    python run_judge_viability.py --model llama-3.3-70b  # Test one model only
"""
import argparse
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ABCD_DIR, OUTPUT_DIR, RANDOM_SEED, JUDGE_MODELS, PROMPT_CONDITIONS, MIN_TURNS,
)
from utils.abcd_loader import ABCDDataset
from utils.perturbation_engine import PerturbationEngine
from utils.metrics import (
    compute_detection_rate, check_score_variance,
    check_prompt_sensitivity, check_position_bias, JudgeOutput,
)
from judge_runner import run_pointwise_evaluation, run_pairwise_evaluation


def print_header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")


def print_result(name, passed, detail=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Test only this model (default: first available)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of pairs per check (default: 10)")
    args = parser.parse_args()

    # Select model
    if args.model and args.model in JUDGE_MODELS:
        model_name = args.model
    else:
        model_name = list(JUDGE_MODELS.keys())[0]
    model_config = JUDGE_MODELS[model_name]
    print(f"\nUsing model: {model_name}")

    # Load data
    print("Loading ABCD dataset...")
    dataset = ABCDDataset(ABCD_DIR)
    engine = PerturbationEngine(dataset, seed=RANDOM_SEED)
    compliant = dataset.get_compliant_conversations(min_turns=MIN_TURNS)

    if len(compliant) < args.n:
        print(f"ERROR: Need at least {args.n} compliant conversations, found {len(compliant)}")
        return

    sample = compliant[:args.n]

    # Generate perturbations (one per conversation, mix of families)
    print(f"Generating {args.n} perturbations...")
    pairs = []
    for i, conv in enumerate(sample):
        perts = engine.generate_all_for_conversation(conv)
        if perts:
            # Pick one perturbation, cycling through families
            families = ["W", "V", "R"]
            target_family = families[i % len(families)]
            matching = [p for p in perts if p.family == target_family]
            if not matching:
                matching = perts
            pert = matching[0]

            dialogue_text = dataset.format_dialogue_text(conv)
            policy_text = dataset.get_policy_for_subflow(conv.subflow) or ""
            intent = conv.subflow

            pairs.append({
                "conv": conv,
                "pert": pert,
                "dialogue_text": dialogue_text,
                "policy_text": policy_text,
                "intent": intent,
            })

    print(f"Generated {len(pairs)} perturbation pairs")
    if not pairs:
        print("ERROR: No perturbation pairs generated")
        return

    # ═══ V3: Score Variance Check ═══
    print_header("V3: SCORE VARIANCE CHECK")
    print(f"  Running {len(pairs)} pointwise evaluations (originals only)...")

    all_scores = []
    for pair_data in pairs:
        conv = pair_data["conv"]
        try:
            result = run_pointwise_evaluation(
                dialogue_text=pair_data["dialogue_text"],
                convo_id=conv.convo_id,
                is_original=True,
                perturbation_family="N/A",
                perturbation_type="N/A",
                customer_intent=pair_data["intent"],
                model_name=model_name,
                model_config=model_config,
                prompt_condition="no_policy",
            )
            all_scores.extend([
                result.workflow_correctness,
                result.value_accuracy,
                result.rule_compliance,
                result.overall_quality,
            ])
            print(f"    {conv.convo_id}: W={result.workflow_correctness} V={result.value_accuracy} "
                  f"R={result.rule_compliance} O={result.overall_quality}")
        except Exception as e:
            print(f"    {conv.convo_id}: ERROR — {e}")

    if all_scores:
        nonzero_scores = [s for s in all_scores if s > 0]
        if nonzero_scores:
            var_result = check_score_variance(nonzero_scores)
            print_result(
                f"Score variance: {var_result['verdict']}",
                var_result["pass"],
                f"mean={var_result['mean']}, std={var_result['std']}, range={var_result['range']}"
            )
        else:
            print_result("Score variance", False, "All scores are 0 — API may not be returning valid responses")

    # ═══ V4: Prompt Sensitivity Check ═══
    print_header("V4: PROMPT SENSITIVITY CHECK")
    print(f"  Running {len(pairs[:5])} pairs × 3 prompt conditions...")

    results_by_condition = {cond: [] for cond in PROMPT_CONDITIONS}

    for pair_data in pairs[:5]:  # 5 pairs × 3 conditions = 30 calls
        conv = pair_data["conv"]
        pert = pair_data["pert"]

        for condition in PROMPT_CONDITIONS:
            try:
                # Score original
                orig_result = run_pointwise_evaluation(
                    dialogue_text=pair_data["dialogue_text"],
                    convo_id=conv.convo_id,
                    is_original=True,
                    perturbation_family=pert.family,
                    perturbation_type=pert.perturbation_type,
                    customer_intent=pair_data["intent"],
                    model_name=model_name,
                    model_config=model_config,
                    prompt_condition=condition,
                    policy_text=pair_data["policy_text"],
                    action_sequence=conv.ground_truth_sequence,
                )

                # Score perturbed
                pert_result = run_pointwise_evaluation(
                    dialogue_text=pair_data["dialogue_text"],  # Would use perturbed text in full run
                    convo_id=conv.convo_id,
                    is_original=False,
                    perturbation_family=pert.family,
                    perturbation_type=pert.perturbation_type,
                    customer_intent=pair_data["intent"],
                    model_name=model_name,
                    model_config=model_config,
                    prompt_condition=condition,
                    policy_text=pair_data["policy_text"],
                    action_sequence=conv.ground_truth_sequence,
                )

                detected = 1 if orig_result.overall_quality > pert_result.overall_quality else 0
                results_by_condition[condition].append({
                    "convo_id": conv.convo_id,
                    "detected": detected,
                    "family": pert.family,
                })
                print(f"    {conv.convo_id} [{condition}]: orig={orig_result.overall_quality} "
                      f"pert={pert_result.overall_quality} detected={detected}")

            except Exception as e:
                print(f"    {conv.convo_id} [{condition}]: ERROR — {e}")

    if results_by_condition["no_policy"] and results_by_condition["full_policy"]:
        sens_result = check_prompt_sensitivity(results_by_condition)
        print_result(
            f"Prompt sensitivity: {sens_result['verdict']}",
            sens_result["pass"],
        )

    # ═══ V5: Position Bias Check ═══
    print_header("V5: POSITION BIAS CHECK")
    print(f"  Running {min(len(pairs), 5)} pairwise evaluations × 2 orderings...")

    results_a_first = []
    results_b_first = []

    for pair_data in pairs[:5]:
        conv = pair_data["conv"]
        pert = pair_data["pert"]

        for original_first in [True, False]:
            try:
                result = run_pairwise_evaluation(
                    original_text=pair_data["dialogue_text"],
                    perturbed_text=pair_data["dialogue_text"],  # Would use perturbed text
                    convo_id=conv.convo_id,
                    perturbation_family=pert.family,
                    perturbation_type=pert.perturbation_type,
                    customer_intent=pair_data["intent"],
                    model_name=model_name,
                    model_config=model_config,
                    prompt_condition="no_policy",
                    original_first=original_first,
                )

                if original_first:
                    results_a_first.append(result)
                else:
                    results_b_first.append(result)

                print(f"    {conv.convo_id} [orig_first={original_first}]: "
                      f"preferred={result['judge_preferred']}")

            except Exception as e:
                print(f"    {conv.convo_id}: ERROR — {e}")

    if results_a_first and results_b_first:
        bias_result = check_position_bias(results_a_first, results_b_first)
        print_result(
            f"Position bias: {bias_result['verdict']}",
            bias_result["pass"],
            f"gap={bias_result['position_gap']:.0%}"
        )

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "judge_viability_results.json")
    report = {
        "model": model_name,
        "n_pairs": len(pairs),
        "v3_scores": all_scores[:40] if all_scores else [],
        "v4_sensitivity": {
            cond: [r["detected"] for r in results]
            for cond, results in results_by_condition.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
