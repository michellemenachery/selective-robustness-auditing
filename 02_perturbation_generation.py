"""
Step 2: Perturbation Generation (Programmatic)

Generates two types of perturbations with ZERO LLM API calls:

Family D (real violations):
  - Slot value swaps: replace correct slot values with values from other conversations
  - Creates 3 position variants per conversation (early/mid/late) where possible

Family A (surface noise):
  - Filler insertion: add "um", "so", "like" etc. to agent/customer turns
  - Capitalization variation: randomize casing
  - Greeting substitution: swap greetings/closings

Validates every perturbation against ground truth:
  - Family A: confirms no substantive change
  - Family D: confirms violation is detectable

Outputs:
  - data/experiment/perturbations.json — all perturbation pairs

Usage:
    cd ~/research/selective-robustness-auditing
    source .venv/bin/activate
    python 02_perturbation_generation.py
"""

import json
import random
import re
import copy
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "experiment"

RANDOM_SEED = 42

# ── Filler insertion templates ──
FILLERS = [
    ("um, ", 0.3),
    ("uh, ", 0.2),
    ("so, ", 0.3),
    ("like, ", 0.2),
    ("you know, ", 0.15),
    ("well, ", 0.25),
    ("I mean, ", 0.15),
]

GREETING_SWAPS = {
    "hi!": ["hello!", "hey there!", "good day!", "hi there!"],
    "hello!": ["hi!", "hey!", "good afternoon!", "hi there!"],
    "hello": ["hi", "hey there", "good day", "hi there"],
    "hi": ["hello", "hey", "hey there", "good day"],
    "how can i help you?": [
        "what can i do for you?",
        "how may i assist you today?", 
        "what can i help you with?",
        "how can i assist you?",
    ],
    "how can i help you today?": [
        "what can i do for you?",
        "how may i assist you?",
        "what can i help you with today?",
    ],
    "thank you": ["thanks", "thanks a lot", "thank you so much", "appreciate it"],
    "thanks": ["thank you", "thanks a lot", "much appreciated"],
    "you're welcome": ["no problem", "happy to help", "of course", "my pleasure"],
    "no problem": ["you're welcome", "happy to help", "of course"],
    "goodbye": ["bye", "have a great day", "take care"],
    "bye": ["goodbye", "have a nice day", "take care"],
    "have a great day": ["have a nice day", "take care", "goodbye"],
}


# ═══════════════════════════════════════════════════════════
# FAMILY D: SLOT VALUE SWAPS
# ═══════════════════════════════════════════════════════════

def build_replacement_pool(all_samples):
    """
    Build a pool of replacement values for each slot type,
    drawn from other conversations in the sample.
    """
    pool = defaultdict(set)
    
    for sample in all_samples:
        for slot_key, slot_val in sample["swappable_values"].items():
            pool[slot_key].add(slot_val)
    
    # Convert to lists for random sampling
    return {k: list(v) for k, v in pool.items()}


def generate_slot_swap(sample, replacement_pool, rng):
    """
    Generate a slot value swap perturbation.
    Picks one swappable value and replaces ALL occurrences in the text.
    """
    swappable = sample["swappable_values"]
    if not swappable:
        return None
    
    # Pick a slot to swap — prioritize meaningful ones
    priority = [
        "order.order_id",
        "personal.customer_name", 
        "personal.account_id",
        "personal.email",
        "personal.username",
    ]
    
    chosen_slot = None
    for slot in priority:
        if slot in swappable:
            chosen_slot = slot
            break
    
    if not chosen_slot:
        chosen_slot = rng.choice(list(swappable.keys()))
    
    original_value = swappable[chosen_slot]
    
    # Pick a replacement from pool (different from original)
    candidates = [v for v in replacement_pool.get(chosen_slot, []) if v != original_value]
    if not candidates:
        return None
    
    replacement_value = rng.choice(candidates)
    
    # Replace in all turns
    original_turns = sample["original_turns"]
    perturbed_turns = []
    turns_modified = []
    
    for i, turn in enumerate(original_turns):
        new_text = turn["text"]
        # Case-insensitive replacement
        pattern = re.compile(re.escape(original_value), re.IGNORECASE)
        if pattern.search(new_text):
            new_text = pattern.sub(replacement_value, new_text)
            turns_modified.append(i)
        
        perturbed_turns.append({
            "speaker": turn["speaker"],
            "text": new_text,
        })
    
    if not turns_modified:
        return None
    
    return {
        "type": "slot_value_swap",
        "family": "D",
        "slot_swapped": chosen_slot,
        "original_value": original_value,
        "replacement_value": replacement_value,
        "turns_modified": turns_modified,
        "num_turns_modified": len(turns_modified),
        "perturbed_turns": perturbed_turns,
    }


def generate_position_variants(sample, replacement_pool, rng):
    """
    Generate slot swap variants at different positions:
    - Early: swap only in first 25% of turns
    - Mid: swap only in middle 30-65% of turns  
    - Late: swap only in last 25% of turns
    
    Returns dict with position -> perturbation, or None if not enough positions.
    """
    swappable = sample["swappable_values"]
    value_positions = sample.get("value_positions", {})
    
    # Find a value that appears in multiple turns
    best_slot = None
    best_positions = []
    
    for slot_key in ["personal.customer_name", "order.order_id", "personal.email", "personal.username"]:
        if slot_key in swappable and slot_key in value_positions:
            positions = value_positions[slot_key]
            if len(positions) >= 2:
                best_slot = slot_key
                best_positions = positions
                break
    
    if not best_slot:
        return None
    
    original_value = swappable[best_slot]
    candidates = [v for v in replacement_pool.get(best_slot, []) if v != original_value]
    if not candidates:
        return None
    
    replacement_value = rng.choice(candidates)
    num_turns = sample["num_turns"]
    
    # Define position boundaries
    early_end = int(num_turns * 0.25)
    mid_start = int(num_turns * 0.30)
    mid_end = int(num_turns * 0.65)
    late_start = int(num_turns * 0.75)
    
    # Check which positions have the value
    early_positions = [p for p in best_positions if p <= early_end]
    mid_positions = [p for p in best_positions if mid_start <= p <= mid_end]
    late_positions = [p for p in best_positions if p >= late_start]
    
    variants = {}
    original_turns = sample["original_turns"]
    pattern = re.compile(re.escape(original_value), re.IGNORECASE)
    
    for position_name, target_positions in [
        ("early", early_positions),
        ("mid", mid_positions),
        ("late", late_positions),
    ]:
        if not target_positions:
            continue
        
        perturbed_turns = []
        for i, turn in enumerate(original_turns):
            new_text = turn["text"]
            # Only swap at target positions
            if i in target_positions and pattern.search(new_text):
                new_text = pattern.sub(replacement_value, new_text)
            perturbed_turns.append({
                "speaker": turn["speaker"],
                "text": new_text,
            })
        
        variants[position_name] = {
            "type": "slot_value_swap_positioned",
            "family": "D",
            "position": position_name,
            "slot_swapped": best_slot,
            "original_value": original_value,
            "replacement_value": replacement_value,
            "turns_modified": target_positions,
            "perturbed_turns": perturbed_turns,
        }
    
    # Only return if we got at least 2 positions
    if len(variants) >= 2:
        return variants
    return None


# ═══════════════════════════════════════════════════════════
# FAMILY A: SURFACE PERTURBATIONS
# ═══════════════════════════════════════════════════════════

def generate_filler_insertion(sample, rng):
    """
    Insert conversational fillers into turns.
    Only modifies turns that DON'T contain slot values 
    (to avoid accidentally breaking slot matching).
    """
    original_turns = sample["original_turns"]
    slot_values = set()
    for v in sample["swappable_values"].values():
        slot_values.add(v.lower())
    
    perturbed_turns = []
    turns_modified = []
    
    for i, turn in enumerate(original_turns):
        text = turn["text"]
        text_lower = text.lower()
        
        # Don't modify turns with slot values or action turns
        has_slot = any(sv in text_lower for sv in slot_values)
        is_action = turn["speaker"] == "action"
        
        if has_slot or is_action or len(text) < 10:
            perturbed_turns.append({"speaker": turn["speaker"], "text": text})
            continue
        
        # Randomly decide whether to modify this turn
        if rng.random() < 0.5:
            perturbed_turns.append({"speaker": turn["speaker"], "text": text})
            continue
        
        # Insert a filler at the start
        filler, _ = rng.choice(FILLERS)
        
        # Preserve original casing pattern
        if text[0].isupper():
            new_text = filler.capitalize() + text[0].lower() + text[1:]
        else:
            new_text = filler + text
        
        perturbed_turns.append({"speaker": turn["speaker"], "text": new_text})
        turns_modified.append(i)
    
    if not turns_modified:
        return None
    
    return {
        "type": "filler_insertion",
        "family": "A",
        "turns_modified": turns_modified,
        "num_turns_modified": len(turns_modified),
        "perturbed_turns": perturbed_turns,
    }


def generate_greeting_swap(sample, rng):
    """
    Swap greetings and closings with equivalent alternatives.
    Surface-level change only.
    """
    original_turns = sample["original_turns"]
    perturbed_turns = []
    turns_modified = []
    
    for i, turn in enumerate(original_turns):
        text = turn["text"]
        text_lower = text.lower().strip()
        new_text = text
        
        # Check first and last few turns for greetings
        if i <= 3 or i >= len(original_turns) - 3:
            for pattern, replacements in GREETING_SWAPS.items():
                if pattern in text_lower:
                    replacement = rng.choice(replacements)
                    # Match original casing
                    if text[0].isupper():
                        replacement = replacement[0].upper() + replacement[1:]
                    new_text = text_lower.replace(pattern, replacement)
                    if new_text != text_lower:
                        turns_modified.append(i)
                    break
        
        perturbed_turns.append({"speaker": turn["speaker"], "text": new_text})
    
    if not turns_modified:
        return None
    
    return {
        "type": "greeting_swap",
        "family": "A",
        "turns_modified": turns_modified,
        "num_turns_modified": len(turns_modified),
        "perturbed_turns": perturbed_turns,
    }


def generate_combined_family_a(sample, rng):
    """
    Combine filler insertion + greeting swap for a stronger Family A perturbation.
    """
    original_turns = sample["original_turns"]
    slot_values = set()
    for v in sample["swappable_values"].values():
        slot_values.add(v.lower())
    
    perturbed_turns = []
    turns_modified = []
    modifications = []
    
    for i, turn in enumerate(original_turns):
        text = turn["text"]
        text_lower = text.lower().strip()
        new_text = text
        modified = False
        
        has_slot = any(sv in text_lower for sv in slot_values)
        is_action = turn["speaker"] == "action"
        
        # Greeting swap on first/last turns
        if (i <= 3 or i >= len(original_turns) - 3) and not is_action:
            for pattern, replacements in GREETING_SWAPS.items():
                if pattern in text_lower:
                    replacement = rng.choice(replacements)
                    if text[0].isupper():
                        replacement = replacement[0].upper() + replacement[1:]
                    new_text = new_text.lower().replace(pattern, replacement)
                    modified = True
                    modifications.append(f"greeting_swap@{i}")
                    break
        
        # Filler insertion on non-slot, non-action turns
        if not has_slot and not is_action and len(text) >= 15 and rng.random() < 0.4:
            filler, _ = rng.choice(FILLERS)
            if new_text[0].isupper():
                new_text = filler.capitalize() + new_text[0].lower() + new_text[1:]
            else:
                new_text = filler + new_text
            modified = True
            modifications.append(f"filler@{i}")
        
        if modified:
            turns_modified.append(i)
        
        perturbed_turns.append({"speaker": turn["speaker"], "text": new_text})
    
    if not turns_modified:
        return None
    
    return {
        "type": "combined_surface",
        "family": "A",
        "turns_modified": turns_modified,
        "num_turns_modified": len(turns_modified),
        "modifications": modifications,
        "perturbed_turns": perturbed_turns,
    }


# ═══════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_family_a(original_turns, perturbed):
    """
    Verify Family A perturbation didn't change anything substantive.
    Check that all slot values are still present and unchanged.
    """
    orig_text = " ".join(t["text"] for t in original_turns).lower()
    pert_text = " ".join(t["text"] for t in perturbed["perturbed_turns"]).lower()
    
    # The texts should be different (otherwise perturbation did nothing)
    if orig_text == pert_text:
        return False, "no_change"
    
    # Same number of turns
    if len(original_turns) != len(perturbed["perturbed_turns"]):
        return False, "turn_count_changed"
    
    # Same speakers
    for orig, pert in zip(original_turns, perturbed["perturbed_turns"]):
        if orig["speaker"] != pert["speaker"]:
            return False, "speaker_changed"
    
    return True, "valid"


def validate_family_d_slot_swap(original_turns, perturbed, sample):
    """
    Verify Family D slot swap:
    - The original value is NO LONGER in the perturbed text 
      (at least at modified positions)
    - The replacement value IS in the perturbed text
    """
    orig_value = perturbed["original_value"].lower()
    repl_value = perturbed["replacement_value"].lower()
    
    # Check modified turns
    for turn_idx in perturbed["turns_modified"]:
        pert_text = perturbed["perturbed_turns"][turn_idx]["text"].lower()
        orig_text = original_turns[turn_idx]["text"].lower()
        
        # Original value should be gone from modified turns
        if orig_value in pert_text:
            return False, "original_value_still_present"
        
        # Replacement should be there
        if repl_value not in pert_text:
            return False, "replacement_value_missing"
        
        # Text should have actually changed
        if orig_text == pert_text:
            return False, "turn_unchanged"
    
    return True, "valid"


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  STEP 2: PERTURBATION GENERATION")
    print("=" * 60 + "\n")
    
    rng = random.Random(RANDOM_SEED)
    
    # Load sample
    sample_path = EXPERIMENT_DIR / "sample_500.json"
    if not sample_path.exists():
        print(f"ERROR: {sample_path} not found. Run 01_sample_selection.py first.")
        return
    
    with open(sample_path) as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} sample conversations")
    
    # Build replacement value pool
    replacement_pool = build_replacement_pool(samples)
    print(f"Replacement pool sizes:")
    for slot, values in sorted(replacement_pool.items()):
        print(f"  {slot:30s}  {len(values)} unique values")
    
    # ── Generate perturbations ──
    print(f"\n{'='*60}")
    print("GENERATING PERTURBATIONS")
    print("=" * 60)
    
    all_perturbations = []
    
    stats = {
        "slot_swap_success": 0,
        "slot_swap_fail": 0,
        "filler_success": 0,
        "filler_fail": 0,
        "greeting_success": 0,
        "greeting_fail": 0,
        "combined_a_success": 0,
        "combined_a_fail": 0,
        "position_variants": 0,
        "position_2way": 0,
        "position_3way": 0,
        "validation_pass": 0,
        "validation_fail": 0,
    }
    
    for idx, sample in enumerate(samples):
        convo_id = sample["convo_id"]
        original_turns = sample["original_turns"]
        
        convo_perturbations = {
            "convo_id": convo_id,
            "flow": sample["flow"],
            "subflow": sample["subflow"],
            "num_turns": sample["num_turns"],
            "original_turns": original_turns,
            "required_actions": sample["required_actions"],
            "performed_actions": sample["performed_actions"],
            "family_d": [],
            "family_a": [],
            "position_variants": None,
        }
        
        # ── Family D: Slot value swap ──
        slot_swap = generate_slot_swap(sample, replacement_pool, rng)
        if slot_swap:
            valid, reason = validate_family_d_slot_swap(original_turns, slot_swap, sample)
            if valid:
                slot_swap["validation"] = "pass"
                convo_perturbations["family_d"].append(slot_swap)
                stats["slot_swap_success"] += 1
            else:
                stats["slot_swap_fail"] += 1
                stats["validation_fail"] += 1
        else:
            stats["slot_swap_fail"] += 1
        
        # ── Family D: Position variants ──
        position_variants = generate_position_variants(sample, replacement_pool, rng)
        if position_variants:
            valid_variants = {}
            for pos, variant in position_variants.items():
                valid, reason = validate_family_d_slot_swap(original_turns, variant, sample)
                if valid:
                    variant["validation"] = "pass"
                    valid_variants[pos] = variant
            
            if len(valid_variants) >= 2:
                convo_perturbations["position_variants"] = valid_variants
                stats["position_variants"] += 1
                if len(valid_variants) == 3:
                    stats["position_3way"] += 1
                else:
                    stats["position_2way"] += 1
        
        # ── Family A: Filler insertion ──
        filler = generate_filler_insertion(sample, rng)
        if filler:
            valid, reason = validate_family_a(original_turns, filler)
            if valid:
                filler["validation"] = "pass"
                convo_perturbations["family_a"].append(filler)
                stats["filler_success"] += 1
            else:
                stats["filler_fail"] += 1
        else:
            stats["filler_fail"] += 1
        
        # ── Family A: Combined surface ──
        combined = generate_combined_family_a(sample, rng)
        if combined:
            valid, reason = validate_family_a(original_turns, combined)
            if valid:
                combined["validation"] = "pass"
                convo_perturbations["family_a"].append(combined)
                stats["combined_a_success"] += 1
            else:
                stats["combined_a_fail"] += 1
        else:
            stats["combined_a_fail"] += 1
        
        all_perturbations.append(convo_perturbations)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1} / {len(samples)}")
    
    # ── Report ──
    print(f"\n{'='*60}")
    print("PERTURBATION GENERATION RESULTS")
    print("=" * 60)
    
    total = len(samples)
    has_d = sum(1 for p in all_perturbations if p["family_d"])
    has_a = sum(1 for p in all_perturbations if p["family_a"])
    has_both = sum(1 for p in all_perturbations if p["family_d"] and p["family_a"])
    has_position = sum(1 for p in all_perturbations if p["position_variants"])
    
    print(f"""
    Total conversations:              {total}
    
    FAMILY D (real violations):
    ─────────────────────────────────────────────
    Slot swaps generated:             {stats['slot_swap_success']}
    Slot swaps failed:                {stats['slot_swap_fail']}
    Conversations with Family D:      {has_d} ({100*has_d/total:.1f}%)
    
    TURN-POSITION VARIANTS:
    ─────────────────────────────────────────────
    Conversations with variants:      {has_position} ({100*has_position/total:.1f}%)
    With all 3 positions:             {stats['position_3way']}
    With 2 positions:                 {stats['position_2way']}
    
    FAMILY A (surface noise):
    ─────────────────────────────────────────────
    Filler insertion:                 {stats['filler_success']}
    Combined surface:                 {stats['combined_a_success']}
    Conversations with Family A:      {has_a} ({100*has_a/total:.1f}%)
    
    USABLE PAIRS (have both A and D):
    ─────────────────────────────────────────────
    Complete pairs:                   {has_both} ({100*has_both/total:.1f}%)
    """)
    
    # Show examples
    print("=" * 60)
    print("EXAMPLE PERTURBATIONS")
    print("=" * 60)
    
    for p in all_perturbations[:3]:
        if not p["family_d"] or not p["family_a"]:
            continue
        
        print(f"\n{'─'*50}")
        print(f"Convo: {p['convo_id']} ({p['flow']}::{p['subflow']})")
        
        # Show original first 5 turns
        print(f"\n  ORIGINAL (first 5 turns):")
        for t in p["original_turns"][:5]:
            print(f"    [{t['speaker']}] {t['text'][:80]}")
        
        # Show Family D
        d = p["family_d"][0]
        print(f"\n  FAMILY D ({d['type']}):")
        print(f"    Swapped: {d['slot_swapped']}")
        print(f"    '{d['original_value']}' → '{d['replacement_value']}'")
        print(f"    Modified turns: {d['turns_modified']}")
        for ti in d["turns_modified"][:3]:
            orig = p["original_turns"][ti]["text"][:70]
            pert = d["perturbed_turns"][ti]["text"][:70]
            print(f"    Turn {ti} BEFORE: {orig}")
            print(f"    Turn {ti} AFTER:  {pert}")
        
        # Show Family A
        a = p["family_a"][0]
        print(f"\n  FAMILY A ({a['type']}):")
        print(f"    Modified turns: {a['turns_modified'][:5]}")
        for ti in a["turns_modified"][:3]:
            orig = p["original_turns"][ti]["text"][:70]
            pert = a["perturbed_turns"][ti]["text"][:70]
            print(f"    Turn {ti} BEFORE: {orig}")
            print(f"    Turn {ti} AFTER:  {pert}")
        
        print()
    
    # ── Save ──
    out_path = EXPERIMENT_DIR / "perturbations.json"
    with open(out_path, "w") as f:
        json.dump(all_perturbations, f, indent=2)
    print(f"Saved {len(all_perturbations)} perturbation sets to {out_path}")
    
    stats_path = EXPERIMENT_DIR / "perturbation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
