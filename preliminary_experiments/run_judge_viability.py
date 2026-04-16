#!/usr/bin/env python3
"""
Judge Viability Checks (V3-V5) — FIXED VERSION
Now actually sends perturbed dialogue text to the judge (not the original twice).

Usage:
    python run_judge_viability.py
    python run_judge_viability.py --model llama-3.3-70b --n 10
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


def format_perturbed_dialogue(pert, dataset):
    """
    Format the perturbed turns into readable dialogue text.
    This is the key fix — we must send the PERTURBED text to the judge.
    """
    conv = dataset.conversations.get(pert.original_convo_id)
    if not conv:
        return ""

    lines = []
    for turn in pert.perturbed_turns:
        if isinstance(turn, dict):
            speaker = turn.get("Speaker", turn.get("speaker", "unknown"))
            text = turn.get("Text", turn.get("text", ""))
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            speaker = str(turn[0])
            text = str(turn[1])
        else:
            continue

        # Re-lexicalize with actual values
        for slot_key, slot_val in conv.slot_values.items():
            field_name = slot_key.split(".")[-1]
            token = f"<{field_name}>"
            text = text.replace(token, slot_val)

        lines.append(f"[{speaker.upper()}]: {text}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n", type=int, default=10)
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

    # Generate perturbations
    print(f"Generating {args.n} perturbations...")
    pairs = []
    for i, conv in enumerate(sample):
        perts = engine.generate_all_for_conversation(conv)
        if perts:
            families = ["W", "V", "R"]
            target_family = families[i % len(families)]
            matching = [p for p in perts if p.family == target_family]
            if not matching:
                matching = perts
            pert = matching[0]

            original_text = dataset.format_dialogue_text(conv)
            perturbed_text = format_perturbed_dialogue(pert, dataset)

            # Verify the texts are actually different
            texts_differ = original_text != perturbed_text
            if not texts_differ:
                print(f"  WARNING: {conv.convo_id} [{pert.perturbation_type}] — "
                      f"original and perturbed text are identical! "
                      f"This perturbation type may need LLM rewriting.")

            policy_text = dataset.get_policy_for_subflow(conv.subflow) or ""
            intent = conv.subflow

            pairs.append({
                "conv": conv,
                "pert": pert,
                "original_text": original_text,
                "perturbed_text": perturbed_text,
                "texts_differ": texts_differ,
                "policy_text": policy_text,
                "intent": intent,
            })

    print(f"Generated {len(pairs)} perturbation pairs")
    differ_count = sum(1 for p in pairs if p["texts_differ"])
    print(f"  Pairs with actually different text: {differ_count}/{len(pairs)}")

    if not pairs:
        print("ERROR: No perturbation pairs generated")
        return

    # ═══ V3: Score Variance Check ═══
    print_header("V3: SCORE VARIANCE CHECK")
    print(f"  Scoring {len(pairs)} originals AND {len(pairs)} perturbed...")

    all_orig_scores = []
    all_pert_scores = []

    for pair_data in pairs:
        conv = pair_data["conv"]
        pert = pair_data["pert"]
        try:
            # Score ORIGINAL
            orig_result = run_pointwise_evaluation(
                dialogue_text=pair_data["original_text"],
                convo_id=conv.convo_id,
                is_original=True,
                perturbation_family=pert.family,
                perturbation_type=pert.perturbation_type,
                customer_intent=pair_data["intent"],
                model_name=model_name,
                model_config=model_config,
                prompt_condition="no_policy",
            )

            # Score PERTURBED — THIS IS THE FIX
            pert_result = run_pointwise_evaluation(
                dialogue_text=pair_data["perturbed_text"],  # PERTURBED TEXT
                convo_id=conv.convo_id,
                is_original=False,
                perturbation_family=pert.family,
                perturbation_type=pert.perturbation_type,
                customer_intent=pair_data["intent"],
                model_name=model_name,
                model_config=model_config,
                prompt_condition="no_policy",
            )

            all_orig_scores.extend([
                orig_result.workflow_correctness,
                orig_result.value_accuracy,
                orig_result.rule_compliance,
                orig_result.overall_quality,
            ])
            all_pert_scores.extend([
                pert_result.workflow_correctness,
                pert_result.value_accuracy,
                pert_result.rule_compliance,
                pert_result.overall_quality,
            ])

            # Show comparison
            detected = "YES" if orig_result.overall_quality > pert_result.overall_quality else "NO"
            print(f"    {conv.convo_id} [{pert.family}:{pert.perturbation_type}]:")
            print(f"      Original: W={orig_result.workflow_correctness} V={orig_result.value_accuracy} "
                  f"R={orig_result.rule_compliance} O={orig_result.overall_quality}")
            print(f"      Perturbed: W={pert_result.workflow_correctness} V={pert_result.value_accuracy} "
                  f"R={pert_result.rule_compliance} O={pert_result.overall_quality}")
            print(f"      Detected: {detected}  TextsDiffer: {pair_data['texts_differ']}")
            if pert_result.rationale:
                print(f"      Pert rationale: {pert_result.rationale[:120]}...")

        except Exception as e:
            print(f"    {conv.convo_id}: ERROR — {e}")

    # Variance check on ALL scores combined
    all_scores = all_orig_scores + all_pert_scores
    if all_scores:
        nonzero = [s for s in all_scores if s > 0]
        if nonzero:
            var_result = check_score_variance(nonzero)
            print_result(
                f"Score variance (all): {var_result['verdict']}",
                var_result["pass"],
                f"mean={var_result['mean']}, std={var_result['std']}, range={var_result['range']}"
            )
        # Also check orig vs pert separately
        if all_orig_scores and all_pert_scores:
            import numpy as np
            orig_mean = np.mean([s for s in all_orig_scores if s > 0]) if any(s > 0 for s in all_orig_scores) else 0
            pert_mean = np.mean([s for s in all_pert_scores if s > 0]) if any(s > 0 for s in all_pert_scores) else 0
            print(f"    Original mean: {orig_mean:.2f}, Perturbed mean: {pert_mean:.2f}, "
                  f"Gap: {orig_mean - pert_mean:.2f}")

    # ═══ V4: Prompt Sensitivity Check ═══
    print_header("V4: PROMPT SENSITIVITY CHECK")
    n_sens = min(5, len(pairs))
    print(f"  Running {n_sens} pairs × 3 prompt conditions...")

    results_by_condition = {cond: [] for cond in PROMPT_CONDITIONS}

    for pair_data in pairs[:n_sens]:
        conv = pair_data["conv"]
        pert = pair_data["pert"]

        for condition in PROMPT_CONDITIONS:
            try:
                orig_result = run_pointwise_evaluation(
                    dialogue_text=pair_data["original_text"],
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

                pert_result = run_pointwise_evaluation(
                    dialogue_text=pair_data["perturbed_text"],  # PERTURBED TEXT
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

    if results_by_condition.get("no_policy") and results_by_condition.get("full_policy"):
        sens_result = check_prompt_sensitivity(results_by_condition)
        print_result(
            f"Prompt sensitivity: {sens_result['verdict']}",
            sens_result["pass"],
        )

    # ═══ V5: Position Bias Check ═══
    print_header("V5: POSITION BIAS CHECK")
    n_bias = min(5, len(pairs))
    print(f"  Running {n_bias} pairwise evaluations × 2 orderings...")

    results_a_first = []
    results_b_first = []

    for pair_data in pairs[:n_bias]:
        conv = pair_data["conv"]
        pert = pair_data["pert"]

        for original_first in [True, False]:
            try:
                result = run_pairwise_evaluation(
                    original_text=pair_data["original_text"],
                    perturbed_text=pair_data["perturbed_text"],  # PERTURBED TEXT
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
                      f"preferred={result['judge_preferred']} "
                      f"reason={result.get('reason', '')[:80]}")

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
        "pairs_with_different_text": differ_count,
        "v3_orig_scores": [s for s in all_orig_scores if s > 0][:40],
        "v3_pert_scores": [s for s in all_pert_scores if s > 0][:40],
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
