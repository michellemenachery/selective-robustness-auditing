"""
ABCD Guidelines Parser

Resolves the naming mismatch between:
- guidelines.json: Capitalized names with spaces ("Product Defect" → "Initiate Refund")
- conversations:   Lowercase with underscores ("product_defect" → "refund_initiate")
- kb.json:         Lowercase with underscores mapping subflows to required action sequences
- ontology.json:   Action names with required slot values

Produces a unified lookup that maps any conversation's flow::subflow
to its complete policy specification: steps, required actions, required slots.

Usage:
    cd ~/research/selective-robustness-auditing
    source .venv/bin/activate
    python guidelines_parser.py
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ── Data Classes ──

@dataclass
class ActionStep:
    """A single action step from guidelines."""
    step_index: int
    action_type: str          # e.g., "interaction", "kb query", "action"
    button: str               # e.g., "Pull up Account", "Verify Identity"
    button_normalized: str    # e.g., "pull-up-account", "verify-identity"
    description: str          # Full text description of the step
    sub_instructions: list[str] = field(default_factory=list)


@dataclass
class PolicySpec:
    """Complete policy specification for a flow::subflow pair."""
    flow_raw: str             # As it appears in guidelines ("Product Defect")
    subflow_raw: str          # As it appears in guidelines ("Initiate Refund")
    flow_normalized: str      # As it appears in conversations ("product_defect")
    subflow_normalized: str   # As it appears in conversations ("refund_initiate")
    steps: list[ActionStep]   # Ordered list of required steps
    required_actions: list[str]  # From kb.json (e.g., ["pull-up-account", "verify-identity"])
    action_slots: dict[str, list[str]]  # From ontology: action -> required slots

    @property
    def key(self) -> str:
        return f"{self.flow_normalized}::{self.subflow_normalized}"

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def button_sequence(self) -> list[str]:
        """Ordered list of button/action names from guidelines."""
        return [s.button_normalized for s in self.steps]

    def summary(self) -> str:
        lines = [f"Policy: {self.key}"]
        lines.append(f"  Guidelines: {self.flow_raw} → {self.subflow_raw}")
        lines.append(f"  Steps ({self.num_steps}):")
        for s in self.steps:
            lines.append(f"    {s.step_index}. [{s.action_type}] {s.button}: {s.description[:80]}")
            for sub in s.sub_instructions:
                lines.append(f"       → {sub[:80]}")
        if self.required_actions:
            lines.append(f"  Required actions (kb.json): {self.required_actions}")
        if self.action_slots:
            lines.append(f"  Action slots:")
            for action, slots in self.action_slots.items():
                lines.append(f"    {action}: {slots}")
        return "\n".join(lines)


# ── Name Normalization ──

def normalize_name(name: str) -> str:
    """
    Convert guideline-style names to conversation-style names.
    "Product Defect" -> "product_defect"
    "Initiate Refund" -> "refund_initiate"  (sometimes order differs)
    "Pull up Account" -> "pull-up-account"
    """
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def normalize_button(button: str) -> str:
    """
    Convert button names to action names used in conversations.
    "Pull up Account" -> "pull-up-account"
    "Verify Identity" -> "verify-identity"
    """
    return button.lower().strip().replace(" ", "-").replace("_", "-")


def build_name_variants(name: str) -> set[str]:
    """
    Generate multiple possible normalized forms for fuzzy matching.
    "Initiate Refund" could map to "refund_initiate" or "initiate_refund"
    """
    words = name.lower().strip().split()
    variants = set()

    # Direct: "initiate refund" -> "initiate_refund"
    variants.add("_".join(words))

    # Reversed: "initiate refund" -> "refund_initiate"
    if len(words) == 2:
        variants.add(f"{words[1]}_{words[0]}")

    # With common prefixes/suffixes
    variants.add("_".join(words).replace("-", "_"))

    # Handle multi-word with different separators
    variants.add("_".join(words).replace(" ", "_"))

    return variants


# ── Parsers ──

def parse_guidelines(guidelines: dict) -> dict[str, dict]:
    """
    Parse guidelines.json into structured format.
    Returns dict keyed by "FlowName::SubflowName" (raw format).
    """
    parsed = {}

    for flow_name, flow_data in guidelines.items():
        if not isinstance(flow_data, dict):
            continue

        # Get subflows - they're nested under a "subflows" key
        subflows_data = flow_data.get("subflows", {})
        if not isinstance(subflows_data, dict):
            continue

        for subflow_name, subflow_data in subflows_data.items():
            raw_key = f"{flow_name}::{subflow_name}"

            steps = []
            if isinstance(subflow_data, dict):
                # Actions are typically in an "actions" key
                actions_list = subflow_data.get("actions", [])
                if isinstance(actions_list, list):
                    for i, action in enumerate(actions_list):
                        if isinstance(action, dict):
                            step = ActionStep(
                                step_index=i + 1,
                                action_type=action.get("type", "unknown"),
                                button=action.get("button", ""),
                                button_normalized=normalize_button(action.get("button", "")),
                                description=action.get("text", ""),
                                sub_instructions=action.get("subtext", []) if isinstance(action.get("subtext"), list) else [],
                            )
                            steps.append(step)

            parsed[raw_key] = {
                "flow_raw": flow_name,
                "subflow_raw": subflow_name,
                "steps": steps,
            }

    return parsed


def parse_kb(kb: dict) -> dict[str, list[str]]:
    """
    Parse kb.json — maps subflow names to required action sequences.
    e.g., {"recover_username": ["pull-up-account", "verify-identity"]}
    """
    parsed = {}
    for subflow, actions in kb.items():
        if isinstance(actions, list):
            parsed[subflow] = actions
    return parsed


def parse_ontology_actions(ontology: dict) -> dict[str, list[str]]:
    """
    Parse ontology.json to extract action -> required slots mapping.
    Structure: {"actions": {"kb_query": {"verify-identity": ["customer_name", ...]}}}
    """
    action_slots = {}
    actions_data = ontology.get("actions", {})

    if isinstance(actions_data, dict):
        for category, category_actions in actions_data.items():
            if isinstance(category_actions, dict):
                for action_name, slots in category_actions.items():
                    if isinstance(slots, list):
                        action_slots[action_name] = slots
                    elif isinstance(slots, dict):
                        # Some might be nested further
                        action_slots[action_name] = list(slots.keys())

    return action_slots


# ── Matching Engine ──

def build_flow_name_map(guidelines: dict, conversations: list) -> dict[str, str]:
    """
    Build mapping from conversation flow names to guideline flow names.
    "product_defect" -> "Product Defect"
    """
    # Get unique flow names from conversations
    convo_flows = set()
    for convo in conversations:
        flow = convo.get("scenario", {}).get("flow", "")
        if flow:
            convo_flows.add(flow)

    # Get guideline flow names
    guideline_flows = set(guidelines.keys())

    # Build mapping
    flow_map = {}
    for convo_flow in convo_flows:
        convo_norm = normalize_name(convo_flow)

        for guide_flow in guideline_flows:
            guide_norm = normalize_name(guide_flow)

            if convo_norm == guide_norm:
                flow_map[convo_flow] = guide_flow
                break

        # Try fuzzy match if exact didn't work
        if convo_flow not in flow_map:
            for guide_flow in guideline_flows:
                guide_variants = build_name_variants(guide_flow)
                convo_variants = build_name_variants(convo_flow)
                if guide_variants & convo_variants:
                    flow_map[convo_flow] = guide_flow
                    break

    return flow_map


def build_subflow_name_map(
    guideline_subflows: dict[str, str],
    convo_subflows: set[str],
) -> dict[str, str]:
    """
    Build mapping from conversation subflow names to guideline subflow names.
    This is harder because naming conventions differ more at subflow level.
    "return_size" -> "Return Wrong Size" (needs fuzzy matching)
    """
    subflow_map = {}

    for convo_sub in convo_subflows:
        convo_norm = normalize_name(convo_sub)
        convo_words = set(convo_norm.replace("_", " ").split())

        best_match = None
        best_score = 0

        for guide_sub in guideline_subflows:
            guide_norm = normalize_name(guide_sub)
            guide_words = set(guide_norm.replace("_", " ").split())

            # Exact match
            if convo_norm == guide_norm:
                best_match = guide_sub
                best_score = 100
                break

            # Check if convo name variants match guideline variants
            convo_variants = build_name_variants(convo_sub)
            guide_variants = build_name_variants(guide_sub)
            if convo_variants & guide_variants:
                best_match = guide_sub
                best_score = 90
                break

            # Word overlap score
            overlap = len(convo_words & guide_words)
            total = len(convo_words | guide_words)
            score = overlap / total if total > 0 else 0

            # Boost if key words match
            if overlap >= 1 and score > best_score:
                best_score = score
                best_match = guide_sub

            # Check substring containment
            if convo_norm in guide_norm or guide_norm in convo_norm:
                if score >= best_score:
                    best_score = max(score, 0.7)
                    best_match = guide_sub

        if best_match and best_score >= 0.3:
            subflow_map[convo_sub] = best_match

    return subflow_map


# ── Main Builder ──

def build_policy_lookup(
    guidelines: dict,
    kb: dict,
    ontology: dict,
    conversations: list,
) -> dict[str, PolicySpec]:
    """
    Build the complete policy lookup table.
    Returns dict keyed by "flow_normalized::subflow_normalized".
    """
    # Parse all sources
    parsed_guidelines = parse_guidelines(guidelines)
    parsed_kb = parse_kb(kb)
    action_slots = parse_ontology_actions(ontology)

    # Build flow name mapping
    flow_map = build_flow_name_map(guidelines, conversations)

    # Get all conversation subflows grouped by flow
    flow_subflows = defaultdict(set)
    for convo in conversations:
        scenario = convo.get("scenario", {})
        flow = scenario.get("flow", "")
        subflow = scenario.get("subflow", "")
        if flow and subflow:
            flow_subflows[flow].add(subflow)

    # Build subflow mappings per flow
    policy_lookup = {}
    matched_count = 0
    unmatched = []

    for convo_flow, convo_subflows in flow_subflows.items():
        guide_flow = flow_map.get(convo_flow)
        if not guide_flow:
            for sub in convo_subflows:
                unmatched.append(f"{convo_flow}::{sub}")
            continue

        # Get guideline subflows for this flow
        guide_flow_data = guidelines.get(guide_flow, {})
        guide_subflows_data = guide_flow_data.get("subflows", {})

        if not isinstance(guide_subflows_data, dict):
            for sub in convo_subflows:
                unmatched.append(f"{convo_flow}::{sub}")
            continue

        # Build subflow mapping
        subflow_map = build_subflow_name_map(
            guide_subflows_data,
            convo_subflows,
        )

        for convo_sub in convo_subflows:
            guide_sub = subflow_map.get(convo_sub)

            if guide_sub:
                raw_key = f"{guide_flow}::{guide_sub}"
                parsed = parsed_guidelines.get(raw_key, {})
                steps = parsed.get("steps", [])

                # Get required actions from kb.json
                required_actions = parsed_kb.get(convo_sub, [])

                # Get slots for each required action
                step_slots = {}
                for action in required_actions:
                    if action in action_slots:
                        step_slots[action] = action_slots[action]
                # Also check button names from guidelines
                for step in steps:
                    btn = step.button_normalized
                    if btn in action_slots and btn not in step_slots:
                        step_slots[btn] = action_slots[btn]

                spec = PolicySpec(
                    flow_raw=guide_flow,
                    subflow_raw=guide_sub,
                    flow_normalized=convo_flow,
                    subflow_normalized=convo_sub,
                    steps=steps,
                    required_actions=required_actions,
                    action_slots=step_slots,
                )
                policy_lookup[spec.key] = spec
                matched_count += 1
            else:
                unmatched.append(f"{convo_flow}::{convo_sub}")

    return policy_lookup, unmatched


# ── Conversation Matcher ──

def match_conversation(convo: dict, policy_lookup: dict) -> Optional[PolicySpec]:
    """Get the policy spec for a conversation."""
    scenario = convo.get("scenario", {})
    flow = scenario.get("flow", "")
    subflow = scenario.get("subflow", "")
    key = f"{flow}::{subflow}"
    return policy_lookup.get(key)


def get_conversation_actions(convo: dict) -> list[dict]:
    """Extract all actions from a conversation's delexed turns."""
    actions = []
    delexed = convo.get("delexed", [])

    for i, turn in enumerate(delexed):
        if not isinstance(turn, dict):
            continue

        speaker = turn.get("speaker", "")
        targets = turn.get("targets", [])

        if isinstance(targets, list) and len(targets) >= 4:
            action = targets[2] if len(targets) > 2 else None
            values = targets[3] if len(targets) > 3 else []

            if action and str(action) not in ("None", "none", ""):
                actions.append({
                    "turn_index": i,
                    "speaker": speaker,
                    "action": str(action),
                    "values": values if isinstance(values, list) else [],
                    "text": turn.get("text", ""),
                })

    return actions


def validate_against_policy(convo: dict, spec: PolicySpec) -> dict:
    """
    Check if a conversation's actions match its policy specification.
    Returns validation result with details.
    """
    convo_actions = get_conversation_actions(convo)
    performed = [a["action"] for a in convo_actions]
    performed_set = set(performed)

    # Check required actions (from kb.json)
    missing_actions = []
    for required in spec.required_actions:
        if required not in performed_set:
            missing_actions.append(required)

    # Check action ordering (from guidelines steps)
    ordering_violations = []
    button_sequence = spec.button_sequence
    if button_sequence and len(performed) >= 2:
        # Check if performed actions follow guideline order
        expected_order = {btn: i for i, btn in enumerate(button_sequence)}
        performed_with_order = [
            (a, expected_order.get(a, -1))
            for a in performed
            if a in expected_order
        ]
        for i in range(len(performed_with_order) - 1):
            curr_action, curr_order = performed_with_order[i]
            next_action, next_order = performed_with_order[i + 1]
            if curr_order > next_order and curr_order >= 0 and next_order >= 0:
                ordering_violations.append({
                    "expected_first": next_action,
                    "actual_first": curr_action,
                })

    # Check slot completeness
    missing_slots = {}
    for action_name, required_slots in spec.action_slots.items():
        if action_name in performed_set:
            # Find this action's values
            action_values = []
            for a in convo_actions:
                if a["action"] == action_name:
                    action_values.extend(a["values"])
            if len(action_values) < len(required_slots):
                missing_slots[action_name] = {
                    "required": required_slots,
                    "provided": action_values,
                }

    return {
        "convo_id": convo.get("convo_id", "unknown"),
        "policy_key": spec.key,
        "performed_actions": performed,
        "required_actions": spec.required_actions,
        "missing_actions": missing_actions,
        "ordering_violations": ordering_violations,
        "missing_slots": missing_slots,
        "is_compliant": len(missing_actions) == 0 and len(ordering_violations) == 0,
        "num_guideline_steps": spec.num_steps,
    }


# ═══════════════════════════════════════════════════════════
# MAIN — Run and report
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  ABCD GUIDELINES PARSER & POLICY MATCHER")
    print("═" * 60 + "\n")

    # Load data
    with open(DATA_DIR / "abcd_v1.1.json") as f:
        all_data = json.load(f)
    train = all_data["train"]

    with open(DATA_DIR / "guidelines.json") as f:
        guidelines = json.load(f)

    with open(DATA_DIR / "kb.json") as f:
        kb = json.load(f)

    with open(DATA_DIR / "ontology.json") as f:
        ontology = json.load(f)

    print(f"Loaded {len(train)} training conversations\n")

    # ── Parse guidelines structure ──
    print("=" * 60)
    print("GUIDELINES PARSING")
    print("=" * 60)

    parsed = parse_guidelines(guidelines)
    print(f"\nParsed {len(parsed)} guideline entries:")
    for key, data in parsed.items():
        steps = data["steps"]
        print(f"  {key}: {len(steps)} steps")
        for s in steps[:3]:
            print(f"    {s.step_index}. [{s.action_type}] {s.button} → {s.description[:70]}")
        if len(steps) > 3:
            print(f"    ... ({len(steps) - 3} more steps)")

    # ── Parse KB ──
    print(f"\n{'='*60}")
    print("KB ACTION SEQUENCES")
    print("=" * 60)

    parsed_kb = parse_kb(kb)
    print(f"\n{len(parsed_kb)} subflow -> action mappings:")
    for subflow, actions in sorted(parsed_kb.items()):
        print(f"  {subflow:40s} → {actions}")

    # ── Parse Ontology Actions ──
    print(f"\n{'='*60}")
    print("ONTOLOGY ACTION SLOTS")
    print("=" * 60)

    action_slots = parse_ontology_actions(ontology)
    print(f"\n{len(action_slots)} action -> slot mappings:")
    for action, slots in sorted(action_slots.items()):
        print(f"  {action:30s} → {slots}")

    # ── Build unified lookup ──
    print(f"\n{'='*60}")
    print("BUILDING UNIFIED POLICY LOOKUP")
    print("=" * 60)

    policy_lookup, unmatched = build_policy_lookup(guidelines, kb, ontology, train)

    print(f"\nMatched policies: {len(policy_lookup)}")
    print(f"Unmatched flow::subflow pairs: {len(unmatched)}")

    print(f"\n── Matched Policies ──")
    for key, spec in sorted(policy_lookup.items()):
        steps_info = f"{spec.num_steps} steps" if spec.steps else "no steps"
        kb_info = f"{len(spec.required_actions)} kb actions" if spec.required_actions else "no kb"
        slots_info = f"{len(spec.action_slots)} slot defs" if spec.action_slots else "no slots"
        print(f"  {key:55s} [{steps_info}, {kb_info}, {slots_info}]")

    if unmatched:
        print(f"\n── Unmatched (need manual mapping or are FAQ-type) ──")
        unmatched_counts = Counter()
        for u in unmatched:
            flow = u.split("::")[0]
            unmatched_counts[flow] += 1
        for flow, count in unmatched_counts.most_common():
            print(f"  {flow}: {count} subflows unmatched")
        print(f"\n  Full list:")
        for u in sorted(set(unmatched)):
            print(f"    {u}")

    # ── Match conversations and validate ──
    print(f"\n{'='*60}")
    print("CONVERSATION COVERAGE & VALIDATION")
    print("=" * 60)

    matched_convos = 0
    matched_with_steps = 0
    matched_with_kb = 0
    compliant = 0
    noncompliant = 0
    validation_results = []

    for convo in train:
        spec = match_conversation(convo, policy_lookup)
        if spec:
            matched_convos += 1
            if spec.num_steps > 0:
                matched_with_steps += 1
            if spec.required_actions:
                matched_with_kb += 1

            result = validate_against_policy(convo, spec)
            validation_results.append(result)
            if result["is_compliant"]:
                compliant += 1
            else:
                noncompliant += 1

    total = len(train)
    print(f"\n  Total conversations:                    {total}")
    print(f"  Matched to a policy:                    {matched_convos} ({100*matched_convos/total:.1f}%)")
    print(f"  With guideline steps:                   {matched_with_steps} ({100*matched_with_steps/total:.1f}%)")
    print(f"  With KB action sequences:               {matched_with_kb} ({100*matched_with_kb/total:.1f}%)")
    print(f"  Policy-compliant conversations:         {compliant} ({100*compliant/total:.1f}%)")
    print(f"  Non-compliant (missing actions/order):  {noncompliant} ({100*noncompliant/total:.1f}%)")

    # Show some compliant and non-compliant examples
    print(f"\n── Sample Compliant Conversations ──")
    compliant_examples = [r for r in validation_results if r["is_compliant"]][:3]
    for r in compliant_examples:
        print(f"  {r['convo_id']} ({r['policy_key']})")
        print(f"    Performed: {r['performed_actions']}")
        print(f"    Required:  {r['required_actions']}")

    print(f"\n── Sample Non-Compliant Conversations ──")
    noncompliant_examples = [r for r in validation_results if not r["is_compliant"]][:5]
    for r in noncompliant_examples:
        print(f"  {r['convo_id']} ({r['policy_key']})")
        print(f"    Performed: {r['performed_actions']}")
        print(f"    Required:  {r['required_actions']}")
        if r["missing_actions"]:
            print(f"    MISSING:   {r['missing_actions']}")
        if r["ordering_violations"]:
            print(f"    ORDER:     {r['ordering_violations']}")

    # ── Feasibility update ──
    print(f"\n{'='*60}")
    print("UPDATED FEASIBILITY ASSESSMENT")
    print("=" * 60)

    # Usable for SRG: matched + has steps or KB + is compliant (baseline is good)
    usable_for_srg = sum(
        1 for r in validation_results
        if r["is_compliant"] and (
            policy_lookup[r["policy_key"]].num_steps > 0
            or policy_lookup[r["policy_key"]].required_actions
        )
    )

    print(f"""
    CORE EXPERIMENTS (SRG):
    ─────────────────────────────────────────────
    Conversations matched to policy:           {matched_convos}
    Compliant (usable as baselines):           {compliant}
    With steps OR kb actions (ground truth):   {usable_for_srg}
    → Target: 500                              {"✓ FEASIBLE" if usable_for_srg >= 500 else "⚠ LIMITED" if usable_for_srg >= 200 else "✗ INSUFFICIENT"}

    GROUND TRUTH SOURCES:
    ─────────────────────────────────────────────
    Guidelines steps (what agent should do):   {matched_with_steps} conversations
    KB action sequences (required actions):    {matched_with_kb} conversations
    Ontology slot requirements:                {len(action_slots)} action definitions
    
    These three sources together provide deterministic
    ground truth WITHOUT any LLM judgment.
    """)

    # Save policy lookup for use by other scripts
    output = {
        "policy_count": len(policy_lookup),
        "matched_conversations": matched_convos,
        "matched_with_steps": matched_with_steps,
        "matched_with_kb": matched_with_kb,
        "compliant": compliant,
        "noncompliant": noncompliant,
        "usable_for_srg": usable_for_srg,
        "policies": {
            key: {
                "flow": spec.flow_normalized,
                "subflow": spec.subflow_normalized,
                "flow_raw": spec.flow_raw,
                "subflow_raw": spec.subflow_raw,
                "num_steps": spec.num_steps,
                "button_sequence": spec.button_sequence,
                "required_actions": spec.required_actions,
                "action_slots": spec.action_slots,
            }
            for key, spec in policy_lookup.items()
        },
        "unmatched": sorted(set(unmatched)),
    }

    out_path = PROJECT_ROOT / "data" / "policy_lookup.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved policy lookup to {out_path}")

    # Also print full detail of 3 policies for manual verification
    print(f"\n{'='*60}")
    print("FULL POLICY DETAIL (for manual verification)")
    print("=" * 60)

    for key, spec in list(policy_lookup.items())[:3]:
        print(f"\n{spec.summary()}")


if __name__ == "__main__":
    main()
