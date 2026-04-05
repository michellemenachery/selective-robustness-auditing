"""
ABCD Data Audit for Selective Robustness Auditing (FIXED)

Based on actual ABCD repo structure:
- Single file: abcd_v1.1.json with keys: train, dev, test
- Conversations have: convo_id, scenario, original, delexed
- Turns have: speaker, text, turn_count, targets, candidates
- Targets is a list of 5 items:
    [0] Intent Classification (subflow, 55 options)
    [1] Nextstep Selection (take_action / retrieve_utterance / end_conversation)
    [2] Action Prediction (button clicked, 30 options)
    [3] Value Filling (list of slot values)
    [4] Utterance Ranking (int)
- Speaker can be "agent", "customer", or "action"

Usage:
    cd ~/research/selective-robustness-auditing
    source .venv/bin/activate
    python data_audit.py
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ═══════════════════════════════════════════════════════════
# STEP 1: Load everything
# ═══════════════════════════════════════════════════════════

def load_data():
    """Load all ABCD components."""
    print("Loading data...")

    main_file = DATA_DIR / "abcd_v1.1.json"
    if not main_file.exists():
        print(f"ERROR: {main_file} not found.")
        print("Make sure you ran: gunzip abcd_v1.1.json.gz")
        sys.exit(1)

    with open(main_file) as f:
        all_data = json.load(f)

    train = all_data.get("train", [])
    dev = all_data.get("dev", [])
    test = all_data.get("test", [])
    print(f"  Train conversations: {len(train)}")
    print(f"  Dev conversations:   {len(dev)}")
    print(f"  Test conversations:  {len(test)}")

    with open(DATA_DIR / "ontology.json") as f:
        ontology = json.load(f)
    print(f"  Ontology loaded")

    with open(DATA_DIR / "guidelines.json") as f:
        guidelines = json.load(f)
    print(f"  Guidelines loaded")

    kb = {}
    kb_path = DATA_DIR / "kb.json"
    if kb_path.exists():
        with open(kb_path) as f:
            kb = json.load(f)
        print(f"  KB loaded")

    print()
    return train, dev, test, ontology, guidelines, kb


# ═══════════════════════════════════════════════════════════
# STEP 2: Understand the raw data structure
# ═══════════════════════════════════════════════════════════

def inspect_structure(conversations, ontology, guidelines, kb):
    print("=" * 60)
    print("DATA STRUCTURE INSPECTION")
    print("=" * 60)

    sample = conversations[0]
    print(f"\n── Conversation top-level keys ──")
    print(f"  {list(sample.keys())}")

    if "scenario" in sample:
        scenario = sample["scenario"]
        print(f"\n── Scenario keys ──")
        print(f"  {list(scenario.keys())}")
        for k, v in scenario.items():
            if isinstance(v, dict):
                print(f"  {k}: {dict(list(v.items())[:3])}...")
            elif isinstance(v, str):
                print(f"  {k}: {repr(v)[:100]}")

    if "original" in sample:
        orig = sample["original"]
        print(f"\n── 'original' conversation (first 3 turns) ──")
        print(f"  Type: {type(orig).__name__}, Length: {len(orig)}")
        for i, turn in enumerate(orig[:3]):
            print(f"  Turn {i}: {repr(turn)[:120]}")

    if "delexed" in sample:
        delexed = sample["delexed"]
        print(f"\n── 'delexed' conversation (first 5 turns) ──")
        print(f"  Type: {type(delexed).__name__}, Length: {len(delexed)}")
        for i, turn in enumerate(delexed[:5]):
            if isinstance(turn, dict):
                print(f"\n  Turn {i}:")
                for k, v in turn.items():
                    print(f"    {k}: {repr(v)[:120]}")
            elif isinstance(turn, (list, tuple)):
                print(f"  Turn {i}: {repr(turn)[:150]}")

    print(f"\n── Ontology structure ──")
    if isinstance(ontology, dict):
        for k, v in list(ontology.items())[:3]:
            print(f"  '{k}': {repr(v)[:200]}")

    print(f"\n── Guidelines structure ──")
    if isinstance(guidelines, dict):
        for k, v in list(guidelines.items())[:2]:
            print(f"\n  Flow '{k}':")
            if isinstance(v, dict):
                for k2, v2 in list(v.items())[:2]:
                    print(f"    Subflow '{k2}': {repr(v2)[:200]}")
            elif isinstance(v, list):
                for item in v[:2]:
                    print(f"    {repr(item)[:150]}")

    if kb:
        print(f"\n── KB structure ──")
        for k, v in list(kb.items())[:3]:
            print(f"  '{k}': {repr(v)[:200]}")

    print()


# ═══════════════════════════════════════════════════════════
# STEP 3: Flow/Subflow distribution
# ═══════════════════════════════════════════════════════════

def audit_flows(conversations):
    print("=" * 60)
    print("FLOW / SUBFLOW DISTRIBUTION")
    print("=" * 60)

    flow_counts = Counter()
    subflow_counts = Counter()
    flow_subflow_counts = Counter()

    for convo in conversations:
        scenario = convo.get("scenario", {})
        flow = scenario.get("flow", "MISSING")
        subflow = scenario.get("subflow", "MISSING")
        flow_counts[flow] += 1
        subflow_counts[subflow] += 1
        flow_subflow_counts[f"{flow}::{subflow}"] += 1

    total = len(conversations)
    print(f"\nTotal conversations: {total}")
    print(f"Unique flows: {len(flow_counts)}")
    print(f"Unique subflows: {len(subflow_counts)}")
    print(f"Unique flow::subflow pairs: {len(flow_subflow_counts)}")

    print(f"\n── Flows ──")
    for flow, count in flow_counts.most_common():
        pct = 100 * count / total
        print(f"  {flow:30s}  {count:5d}  ({pct:.1f}%)")

    print(f"\n── All flow::subflow pairs ──")
    for pair, count in flow_subflow_counts.most_common():
        pct = 100 * count / total
        print(f"  {pair:55s}  {count:5d}  ({pct:.1f}%)")

    print()
    return flow_counts, subflow_counts, flow_subflow_counts


# ═══════════════════════════════════════════════════════════
# STEP 4: Guidelines coverage audit
# ═══════════════════════════════════════════════════════════

def audit_guidelines_coverage(conversations, guidelines):
    print("=" * 60)
    print("GUIDELINES COVERAGE AUDIT")
    print("=" * 60)

    print("\n── Guidelines map ──")
    guideline_lookup = {}

    if isinstance(guidelines, dict):
        for flow, flow_val in guidelines.items():
            if isinstance(flow_val, dict):
                for subflow, subflow_val in flow_val.items():
                    key = f"{flow}::{subflow}"
                    steps = []
                    if isinstance(subflow_val, list):
                        steps = subflow_val
                    elif isinstance(subflow_val, dict):
                        for sk, sv in subflow_val.items():
                            if isinstance(sv, list):
                                steps = sv
                                break
                            elif isinstance(sv, str):
                                steps.append(sv)
                    guideline_lookup[key] = {"steps": steps, "raw": subflow_val}
                    print(f"  {key}: {len(steps)} steps")
                    for s in steps[:2]:
                        print(f"    → {repr(s)[:100]}")

    print(f"\n  Total guideline entries: {len(guideline_lookup)}")
    entries_with_steps = sum(1 for v in guideline_lookup.values() if len(v["steps"]) > 0)
    print(f"  Entries with ≥1 step: {entries_with_steps}")

    print(f"\n── Matching conversations to guidelines ──")
    matched = 0
    matched_with_steps = 0
    unmatched = Counter()

    for convo in conversations:
        scenario = convo.get("scenario", {})
        flow = scenario.get("flow", "")
        subflow = scenario.get("subflow", "")
        key = f"{flow}::{subflow}"

        if key in guideline_lookup:
            matched += 1
            if len(guideline_lookup[key]["steps"]) > 0:
                matched_with_steps += 1
        else:
            unmatched[key] += 1

    total = len(conversations)
    print(f"  Conversations with guideline match:        {matched:5d} / {total} ({100*matched/total:.1f}%)")
    print(f"  Conversations with step-level guidelines:  {matched_with_steps:5d} / {total} ({100*matched_with_steps/total:.1f}%)")

    if unmatched:
        print(f"\n── Unmatched flow::subflow pairs ──")
        for pair, count in unmatched.most_common(20):
            print(f"  {pair:55s}  {count:5d}")

    print()
    return matched_with_steps, guideline_lookup


# ═══════════════════════════════════════════════════════════
# STEP 5: Action & slot value audit
# ═══════════════════════════════════════════════════════════

def audit_actions_and_slots(conversations):
    print("=" * 60)
    print("ACTION & SLOT VALUE AUDIT")
    print("=" * 60)

    total_convos = len(conversations)
    convos_with_actions = 0
    action_type_counts = Counter()
    value_counts = Counter()
    actions_per_convo = []
    nextstep_counts = Counter()
    convos_with_values_in_text = 0
    convos_multi_turn_values = 0
    slot_swap_candidates = []

    for convo in conversations:
        convo_id = convo.get("convo_id", "unknown")
        scenario = convo.get("scenario", {})

        # Gather scenario slot values
        scenario_values = {}
        for section_name in ["personal", "order", "product"]:
            section = scenario.get(section_name, {})
            if isinstance(section, dict):
                for k, v in section.items():
                    if isinstance(v, str) and len(v) > 0:
                        scenario_values[f"{section_name}.{k}"] = v
                    elif isinstance(v, (int, float)):
                        scenario_values[f"{section_name}.{k}"] = str(v)

        # Build text from ORIGINAL conversation
        turns = convo.get("original", [])
        full_text = ""
        turn_texts = []
        for turn in turns:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                speaker, text = turn[0], str(turn[1])
                full_text += " " + text
                turn_texts.append({"speaker": speaker, "text": text})
            elif isinstance(turn, dict):
                text = str(turn.get("text", ""))
                full_text += " " + text
                turn_texts.append({"speaker": turn.get("speaker", ""), "text": text})

        full_text_lower = full_text.lower()

        # Get actions from DELEXED conversation
        delexed = convo.get("delexed", [])
        convo_actions = []
        action_turns = []

        for i, turn in enumerate(delexed):
            if isinstance(turn, dict):
                targets = turn.get("targets", [])
                if isinstance(targets, list) and len(targets) >= 4:
                    nextstep = targets[1] if len(targets) > 1 else ""
                    action = targets[2] if len(targets) > 2 else ""
                    values = targets[3] if len(targets) > 3 else []

                    nextstep_counts[str(nextstep)] += 1

                    if action and str(action) not in ("none", "", "None"):
                        convo_actions.append(str(action))
                        action_type_counts[str(action)] += 1
                        action_turns.append(i)

                    if isinstance(values, list):
                        for v in values:
                            if v and str(v) not in ("none", "", "None"):
                                value_counts[str(v)] += 1

        if convo_actions:
            convos_with_actions += 1
        actions_per_convo.append(len(convo_actions))

        # Check scenario values in original text
        values_found_in_text = {}
        value_turn_positions = defaultdict(list)

        for slot_key, slot_val in scenario_values.items():
            slot_val_lower = str(slot_val).lower().strip()
            if len(slot_val_lower) >= 2 and slot_val_lower in full_text_lower:
                values_found_in_text[slot_key] = slot_val
                for ti, tt in enumerate(turn_texts):
                    if slot_val_lower in tt["text"].lower():
                        value_turn_positions[slot_key].append(ti)

        if values_found_in_text:
            convos_with_values_in_text += 1

        multi_turn = {k: v for k, v in value_turn_positions.items() if len(v) >= 2}
        if multi_turn:
            convos_multi_turn_values += 1

        if values_found_in_text and convo_actions:
            slot_swap_candidates.append({
                "convo_id": convo_id,
                "flow": scenario.get("flow", ""),
                "subflow": scenario.get("subflow", ""),
                "num_actions": len(convo_actions),
                "action_types": list(set(convo_actions)),
                "num_values_in_text": len(values_found_in_text),
                "values_in_text": values_found_in_text,
                "multi_turn_values": {k: len(v) for k, v in multi_turn.items()},
                "num_total_turns": len(turns),
                "num_delexed_turns": len(delexed),
            })

    # Report
    print(f"\n── Action Statistics ──")
    print(f"  Conversations with actions:               {convos_with_actions:5d} / {total_convos} ({100*convos_with_actions/total_convos:.1f}%)")
    print(f"  Conversations with scenario vals in text:  {convos_with_values_in_text:5d} / {total_convos} ({100*convos_with_values_in_text/total_convos:.1f}%)")
    print(f"  Conversations with multi-turn vals:       {convos_multi_turn_values:5d} / {total_convos} ({100*convos_multi_turn_values/total_convos:.1f}%)")

    nonzero = [a for a in actions_per_convo if a > 0]
    if nonzero:
        print(f"\n  Actions per convo (with actions): min={min(nonzero)}, max={max(nonzero)}, avg={sum(nonzero)/len(nonzero):.1f}")

    print(f"\n── Nextstep distribution ──")
    for ns, count in nextstep_counts.most_common():
        print(f"  {str(ns):35s}  {count:6d}")

    print(f"\n── Top 30 action types ──")
    for action, count in action_type_counts.most_common(30):
        print(f"  {action:35s}  {count:5d}")

    print(f"\n── Top 30 slot values ──")
    for val, count in value_counts.most_common(30):
        print(f"  {str(val):40s}  {count:5d}")

    print(f"\n── Scenario value types found in text (sample from first 200 candidates) ──")
    type_counter = Counter()
    for cand in slot_swap_candidates[:200]:
        for k in cand["values_in_text"]:
            type_counter[k.split(".")[1] if "." in k else k] += 1
    for vtype, count in type_counter.most_common(20):
        print(f"  {vtype:35s}  {count:5d}")

    print(f"\n── Slot swap candidates summary ──")
    print(f"  Total slot swap candidates: {len(slot_swap_candidates)}")
    multi = [c for c in slot_swap_candidates if c["multi_turn_values"]]
    print(f"  With multi-turn value refs: {len(multi)} (needed for turn-position bias)")

    if multi:
        print(f"\n  Multi-turn value types:")
        mt_types = Counter()
        for c in multi:
            for k in c["multi_turn_values"]:
                mt_types[k.split(".")[1] if "." in k else k] += 1
        for vtype, count in mt_types.most_common(10):
            print(f"    {vtype:30s}  {count:5d}")

    print()
    return slot_swap_candidates


# ═══════════════════════════════════════════════════════════
# STEP 6: Conversation structure
# ═══════════════════════════════════════════════════════════

def audit_conversation_structure(conversations):
    print("=" * 60)
    print("CONVERSATION STRUCTURE AUDIT")
    print("=" * 60)

    orig_turn_counts = []
    delexed_turn_counts = []
    speaker_types = Counter()

    for convo in conversations:
        orig = convo.get("original", [])
        delexed = convo.get("delexed", [])
        orig_turn_counts.append(len(orig))
        delexed_turn_counts.append(len(delexed))

        for turn in delexed:
            if isinstance(turn, dict):
                speaker_types[turn.get("speaker", "unknown")] += 1

    def stats(values, label):
        if not values:
            return
        s = sorted(values)
        n = len(s)
        print(f"\n  {label}:")
        print(f"    Min: {s[0]}, Max: {s[-1]}, Mean: {sum(s)/n:.1f}")
        print(f"    Median: {s[n//2]}")
        print(f"    25th pct: {s[n//4]}, 75th pct: {s[3*n//4]}")

    stats(orig_turn_counts, "Original turns per conversation")
    stats(delexed_turn_counts, "Delexed turns per conversation")

    print(f"\n── Speaker types (delexed) ──")
    for speaker, count in speaker_types.most_common():
        print(f"  {speaker:15s}  {count:6d}")

    print(f"\n── Turn count distribution (original) ──")
    buckets = {"1-10": 0, "11-20": 0, "21-30": 0, "31-40": 0, "41-50": 0, "51+": 0}
    for tc in orig_turn_counts:
        if tc <= 10:
            buckets["1-10"] += 1
        elif tc <= 20:
            buckets["11-20"] += 1
        elif tc <= 30:
            buckets["21-30"] += 1
        elif tc <= 40:
            buckets["31-40"] += 1
        elif tc <= 50:
            buckets["41-50"] += 1
        else:
            buckets["51+"] += 1

    for bucket, count in buckets.items():
        pct = 100 * count / len(orig_turn_counts)
        bar = "█" * int(pct / 2)
        print(f"    {bucket:8s}: {count:5d} ({pct:5.1f}%) {bar}")

    for min_turns in [10, 12, 16, 20]:
        eligible = sum(1 for tc in delexed_turn_counts if tc >= min_turns)
        print(f"\n  Delexed convos with >= {min_turns} turns: {eligible} ({100*eligible/len(delexed_turn_counts):.1f}%)")

    print()


# ═══════════════════════════════════════════════════════════
# STEP 7: Sample conversations
# ═══════════════════════════════════════════════════════════

def show_sample_conversations(conversations, guidelines, n=3):
    print("=" * 60)
    print(f"SAMPLE CONVERSATIONS (first {n})")
    print("=" * 60)

    for convo in conversations[:n]:
        convo_id = convo.get("convo_id", "unknown")
        scenario = convo.get("scenario", {})
        flow = scenario.get("flow", "?")
        subflow = scenario.get("subflow", "?")

        print(f"\n{'─'*50}")
        print(f"Convo: {convo_id}")
        print(f"Flow: {flow} → Subflow: {subflow}")

        for section in ["personal", "order", "product"]:
            data = scenario.get(section, {})
            if data:
                print(f"  {section}: {data}")

        # Matching guideline
        key = f"{flow}::{subflow}"
        if isinstance(guidelines, dict) and flow in guidelines:
            flow_g = guidelines[flow]
            if isinstance(flow_g, dict) and subflow in flow_g:
                guideline = flow_g[subflow]
                print(f"\n  Guideline for {key}:")
                if isinstance(guideline, list):
                    for i, step in enumerate(guideline):
                        print(f"    Step {i+1}: {step}")
                elif isinstance(guideline, dict):
                    for k, v in guideline.items():
                        print(f"    {k}: {v}")

        # Original conversation
        print(f"\n  Original conversation:")
        orig = convo.get("original", [])
        for i, turn in enumerate(orig[:10]):
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                print(f"    [{turn[0]}] {turn[1][:100]}")
            elif isinstance(turn, dict):
                print(f"    [{turn.get('speaker', '?')}] {turn.get('text', '?')[:100]}")
        if len(orig) > 10:
            print(f"    ... ({len(orig) - 10} more turns)")

        # Delexed actions
        print(f"\n  Delexed turns with actions:")
        delexed = convo.get("delexed", [])
        action_count = 0
        for i, turn in enumerate(delexed):
            if isinstance(turn, dict):
                targets = turn.get("targets", [])
                action = targets[2] if len(targets) > 2 else ""
                values = targets[3] if len(targets) > 3 else []
                if action and str(action) not in ("none", "", "None"):
                    action_count += 1
                    text = turn.get("text", "?")
                    print(f"    Turn {i} [{turn.get('speaker','?')}]: {text[:80]}")
                    print(f"      ACTION: {action}, VALUES: {values}")
        print(f"  Total actions: {action_count}")
        print()


# ═══════════════════════════════════════════════════════════
# STEP 8: Feasibility summary
# ═══════════════════════════════════════════════════════════

def print_feasibility_summary(conversations, matched_with_steps, slot_swap_candidates):
    total = len(conversations)
    multi_turn_slot = len([c for c in slot_swap_candidates if c["multi_turn_values"]])

    enough_12 = sum(1 for c in conversations if len(c.get("delexed", [])) >= 12)
    enough_16 = sum(1 for c in conversations if len(c.get("delexed", [])) >= 16)

    print("=" * 60)
    print("FEASIBILITY SUMMARY")
    print("=" * 60)

    print(f"""
    Total training conversations:              {total}

    CORE EXPERIMENTS (SRG):
    ─────────────────────────────────────────────
    Conversations with step-level guidelines:  {matched_with_steps} ({100*matched_with_steps/total:.1f}%)
    Slot swap candidates (value in text):      {len(slot_swap_candidates)} ({100*len(slot_swap_candidates)/total:.1f}%)
    → Target: 500 conversations               {"✓ FEASIBLE" if min(matched_with_steps, len(slot_swap_candidates)) >= 500 else "⚠ MAY NEED ADJUSTMENT"}

    TURN-POSITION BIAS:
    ─────────────────────────────────────────────
    Multi-turn slot references:                {multi_turn_slot}
    Conversations with >= 12 delexed turns:    {enough_12}
    Conversations with >= 16 delexed turns:    {enough_16}
    → Need ~100 per position (early/mid/late)  {"✓ FEASIBLE" if multi_turn_slot >= 300 else "⚠ LIMITED — " + str(multi_turn_slot) + " available" if multi_turn_slot >= 100 else "✗ NOT VIABLE — only " + str(multi_turn_slot)}

    FAILURE ATTRIBUTION:
    ─────────────────────────────────────────────
    → Need ~50 evaluator failures              ✓ FEASIBLE (subset of SRG experiments)

    OVERALL ASSESSMENT:
    ─────────────────────────────────────────────""")

    if min(matched_with_steps, len(slot_swap_candidates)) >= 500:
        print("    ✓ PROCEED — sufficient data for all core experiments")
    elif min(matched_with_steps, len(slot_swap_candidates)) >= 200:
        print("    ⚠ PROCEED WITH CAUTION — consider using dev+test splits too")
    else:
        print("    ✗ PROBLEM — need to investigate data structure further")

    if multi_turn_slot >= 300:
        print("    ✓ Turn-position bias analysis is viable")
    elif multi_turn_slot >= 100:
        print("    ⚠ Turn-position bias possible but statistically limited")
    else:
        print("    ✗ Turn-position bias may need to be cut or redesigned")
    print()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  ABCD DATA AUDIT — SELECTIVE ROBUSTNESS AUDITING")
    print("═" * 60 + "\n")

    train, dev, test, ontology, guidelines, kb = load_data()
    inspect_structure(train, ontology, guidelines, kb)
    audit_flows(train)
    matched_with_steps, guideline_lookup = audit_guidelines_coverage(train, guidelines)
    slot_swap_candidates = audit_actions_and_slots(train)
    audit_conversation_structure(train)
    show_sample_conversations(train, guidelines, n=3)
    print_feasibility_summary(train, matched_with_steps, slot_swap_candidates)

    output = {
        "total_train": len(train),
        "total_dev": len(dev),
        "total_test": len(test),
        "guideline_matched_with_steps": matched_with_steps,
        "slot_swap_candidates": len(slot_swap_candidates),
        "multi_turn_slot_candidates": len([c for c in slot_swap_candidates if c["multi_turn_values"]]),
    }
    out_path = PROJECT_ROOT / "data" / "audit_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved audit results to {out_path}")


if __name__ == "__main__":
    main()
