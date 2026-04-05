"""
Step 1: Sample Selection

Selects 500 policy-compliant conversations from ABCD training data,
stratified by flow, filtered for:
- All required actions (from kb.json) were performed
- Scenario slot values appear in the original text (needed for slot swaps)
- Conversation has 12+ turns (needed for turn-position bias)

Outputs:
- data/experiment/sample_500.json — the selected conversations with metadata
- data/experiment/sample_stats.json — summary statistics

Usage:
    cd ~/research/selective-robustness-auditing
    source .venv/bin/activate
    python 01_sample_selection.py
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TARGET_SAMPLE = 500
MIN_TURNS = 12


def load_all():
    with open(DATA_DIR / "abcd_v1.1.json") as f:
        all_data = json.load(f)
    with open(DATA_DIR / "kb.json") as f:
        kb = json.load(f)
    with open(DATA_DIR / "ontology.json") as f:
        ontology = json.load(f)
    with open(PROJECT_ROOT / "data" / "policy_lookup.json") as f:
        policy_lookup = json.load(f)
    return all_data["train"], kb, ontology, policy_lookup


def get_conversation_actions(convo):
    """Extract action names from delexed turns."""
    actions = []
    delexed = convo.get("delexed", [])
    for i, turn in enumerate(delexed):
        if isinstance(turn, dict):
            targets = turn.get("targets", [])
            if isinstance(targets, list) and len(targets) >= 3:
                action = targets[2]
                values = targets[3] if len(targets) > 3 else []
                if action and str(action) not in ("None", "none", ""):
                    actions.append({
                        "turn_index": i,
                        "action": str(action),
                        "values": values if isinstance(values, list) else [],
                        "speaker": turn.get("speaker", ""),
                        "text": turn.get("text", ""),
                    })
    return actions


def find_slot_values_in_text(convo):
    """Find scenario slot values that appear in the original conversation text."""
    scenario = convo.get("scenario", {})
    
    # Build full text from original conversation
    original = convo.get("original", [])
    turn_texts = []
    for turn in original:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            turn_texts.append({"speaker": turn[0], "text": str(turn[1])})
        elif isinstance(turn, dict):
            turn_texts.append({"speaker": turn.get("speaker", ""), "text": str(turn.get("text", ""))})
    
    full_text_lower = " ".join(t["text"] for t in turn_texts).lower()
    
    # Check each scenario value
    found_values = {}
    value_positions = {}  # which turns contain each value
    
    for section_name in ["personal", "order", "product"]:
        section = scenario.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for k, v in section.items():
            if isinstance(v, str) and len(v) >= 2:
                v_lower = v.lower().strip()
                if v_lower in full_text_lower:
                    slot_key = f"{section_name}.{k}"
                    found_values[slot_key] = v
                    
                    # Find which turns contain this value
                    positions = []
                    for ti, tt in enumerate(turn_texts):
                        if v_lower in tt["text"].lower():
                            positions.append(ti)
                    value_positions[slot_key] = positions
    
    return found_values, value_positions, turn_texts


def is_compliant(convo, kb):
    """Check if conversation performed all required actions."""
    scenario = convo.get("scenario", {})
    subflow = scenario.get("subflow", "")
    
    required = kb.get(subflow, [])
    if not required:
        return False, [], []  # Skip conversations with no KB entry
    
    performed = set(a["action"] for a in get_conversation_actions(convo))
    missing = [r for r in required if r not in performed]
    
    return len(missing) == 0, required, list(missing)


def select_swappable_values(found_values):
    """Identify which slot values are good candidates for swapping."""
    # Prioritize values that are specific and meaningful
    priority_slots = [
        "personal.customer_name",
        "order.order_id", 
        "personal.account_id",
        "personal.email",
        "personal.username",
        "order.zip_code",
        "personal.phone",
    ]
    
    swappable = {}
    for slot in priority_slots:
        if slot in found_values:
            swappable[slot] = found_values[slot]
    
    return swappable


def main():
    print("\n" + "=" * 60)
    print("  STEP 1: SAMPLE SELECTION")
    print("=" * 60 + "\n")
    
    random.seed(RANDOM_SEED)
    
    train, kb, ontology, policy_lookup = load_all()
    print(f"Loaded {len(train)} training conversations")
    print(f"KB has {len(kb)} subflow entries")
    
    # ── Filter candidates ──
    print("\nFiltering candidates...")
    
    candidates = []
    rejection_reasons = Counter()
    
    for convo in train:
        convo_id = convo.get("convo_id", "unknown")
        scenario = convo.get("scenario", {})
        flow = scenario.get("flow", "")
        subflow = scenario.get("subflow", "")
        
        # Check turn count
        original = convo.get("original", [])
        if len(original) < MIN_TURNS:
            rejection_reasons["too_few_turns"] += 1
            continue
        
        # Check policy compliance
        compliant, required, missing = is_compliant(convo, kb)
        if not compliant:
            rejection_reasons["not_compliant"] += 1
            continue
        
        # Check slot values in text
        found_values, value_positions, turn_texts = find_slot_values_in_text(convo)
        swappable = select_swappable_values(found_values)
        
        if not swappable:
            rejection_reasons["no_swappable_values"] += 1
            continue
        
        # Check for multi-turn value references (for position bias)
        multi_turn_values = {
            k: v for k, v in value_positions.items() 
            if len(v) >= 2 and k in swappable
        }
        
        candidates.append({
            "convo_id": convo_id,
            "flow": flow,
            "subflow": subflow,
            "num_turns": len(original),
            "num_delexed_turns": len(convo.get("delexed", [])),
            "required_actions": required,
            "performed_actions": [a["action"] for a in get_conversation_actions(convo)],
            "swappable_values": swappable,
            "all_found_values": found_values,
            "value_positions": value_positions,
            "multi_turn_values": {k: len(v) for k, v in multi_turn_values.items()},
            "has_multi_turn": len(multi_turn_values) > 0,
            "original_turns": [
                {"speaker": t["speaker"], "text": t["text"]} 
                for t in turn_texts
            ],
            "scenario": scenario,
        })
    
    print(f"\nCandidates passing all filters: {len(candidates)}")
    print(f"Rejection reasons:")
    for reason, count in rejection_reasons.most_common():
        print(f"  {reason}: {count}")
    
    # ── Stratified sampling ──
    print(f"\n{'='*60}")
    print("STRATIFIED SAMPLING")
    print("=" * 60)
    
    # Group by flow
    by_flow = defaultdict(list)
    for c in candidates:
        by_flow[c["flow"]].append(c)
    
    print(f"\nCandidates per flow:")
    for flow, convos in sorted(by_flow.items(), key=lambda x: -len(x[1])):
        print(f"  {flow:30s}  {len(convos)}")
    
    # Proportional sampling
    total_candidates = len(candidates)
    selected = []
    
    for flow, convos in by_flow.items():
        # Proportional share of 500
        n_to_select = max(1, round(TARGET_SAMPLE * len(convos) / total_candidates))
        n_to_select = min(n_to_select, len(convos))
        
        # Prioritize conversations with multi-turn values
        convos_sorted = sorted(convos, key=lambda c: (
            c["has_multi_turn"],           # prefer multi-turn
            len(c["swappable_values"]),     # prefer more swappable values
            c["num_turns"],                 # prefer longer conversations
        ), reverse=True)
        
        flow_selected = convos_sorted[:n_to_select]
        selected.extend(flow_selected)
        print(f"  {flow:30s}  selected {len(flow_selected)} / {len(convos)}")
    
    # Trim to exactly TARGET_SAMPLE if over, or pad if under
    random.shuffle(selected)
    if len(selected) > TARGET_SAMPLE:
        selected = selected[:TARGET_SAMPLE]
    elif len(selected) < TARGET_SAMPLE:
        # Add more from underrepresented flows
        selected_ids = {s["convo_id"] for s in selected}
        remaining = [c for c in candidates if c["convo_id"] not in selected_ids]
        random.shuffle(remaining)
        needed = TARGET_SAMPLE - len(selected)
        selected.extend(remaining[:needed])
    
    print(f"\nFinal sample: {len(selected)} conversations")
    
    # ── Summary statistics ──
    print(f"\n{'='*60}")
    print("SAMPLE STATISTICS")
    print("=" * 60)
    
    flow_dist = Counter(s["flow"] for s in selected)
    subflow_dist = Counter(f"{s['flow']}::{s['subflow']}" for s in selected)
    multi_turn_count = sum(1 for s in selected if s["has_multi_turn"])
    
    turn_counts = [s["num_turns"] for s in selected]
    swap_counts = [len(s["swappable_values"]) for s in selected]
    
    print(f"\n  Total conversations:     {len(selected)}")
    print(f"  With multi-turn values:  {multi_turn_count} ({100*multi_turn_count/len(selected):.1f}%)")
    print(f"  Avg turns:               {sum(turn_counts)/len(turn_counts):.1f}")
    print(f"  Avg swappable values:    {sum(swap_counts)/len(swap_counts):.1f}")
    print(f"  Unique flows:            {len(flow_dist)}")
    print(f"  Unique subflows:         {len(subflow_dist)}")
    
    print(f"\n  Flow distribution:")
    for flow, count in flow_dist.most_common():
        print(f"    {flow:30s}  {count:4d} ({100*count/len(selected):.1f}%)")
    
    # Swappable value types
    value_type_counts = Counter()
    for s in selected:
        for k in s["swappable_values"]:
            value_type_counts[k] += 1
    
    print(f"\n  Swappable value types across sample:")
    for vtype, count in value_type_counts.most_common():
        print(f"    {vtype:30s}  {count:4d}")
    
    # ── Save ──
    out_path = OUTPUT_DIR / "sample_500.json"
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved sample to {out_path}")
    
    stats = {
        "total_selected": len(selected),
        "multi_turn_count": multi_turn_count,
        "avg_turns": round(sum(turn_counts)/len(turn_counts), 1),
        "avg_swappable_values": round(sum(swap_counts)/len(swap_counts), 1),
        "flow_distribution": dict(flow_dist.most_common()),
        "value_type_distribution": dict(value_type_counts.most_common()),
        "subflow_count": len(subflow_dist),
    }
    
    stats_path = OUTPUT_DIR / "sample_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
