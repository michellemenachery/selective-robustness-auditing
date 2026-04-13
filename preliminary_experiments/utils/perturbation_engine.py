"""
Perturbation Engine
Generates controlled perturbations across three families (W, V, R) plus nuisance controls.
Perturbations are deterministic where possible (Family W, simple Family V).
Each perturbation is returned with metadata: family, type, ground_truth_check, is_deterministic.
"""
import copy
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils.abcd_loader import ABCDConversation, ABCDDataset


@dataclass
class Perturbation:
    """A single perturbation applied to a conversation."""
    original_convo_id: str
    family: str                          # "W", "V", "R", "N"
    perturbation_type: str               # e.g., "step_deletion", "slot_swap"
    description: str                     # Human-readable description of what changed
    is_deterministic: bool               # Whether the perturbation was generated without LLM
    overlap_tags: List[str] = field(default_factory=list)  # Cross-family overlaps

    # The perturbed dialogue (list of turns)
    perturbed_turns: List = field(default_factory=list)

    # Ground truth validation
    ground_truth_check: str = ""         # How to verify this is a valid perturbation
    ground_truth_result: bool = True     # Whether the check passed
    violated_rule: str = ""              # Specific rule or constraint violated

    # Metadata
    severity: Optional[str] = None       # "minor", "moderate", "severe"
    evidence_distance: Optional[str] = None  # "local", "cross_turn"
    outcome_critical: Optional[bool] = None  # Whether error affects task outcome
    perturbed_action_sequence: List[str] = field(default_factory=list)
    changed_slot: Optional[str] = None
    original_value: Optional[str] = None
    perturbed_value: Optional[str] = None


class PerturbationEngine:
    """
    Generates perturbations for ABCD conversations.
    Priority: deterministic generation wherever possible.
    """

    def __init__(self, dataset: ABCDDataset, seed: int = 42):
        self.dataset = dataset
        self.rng = random.Random(seed)

        # Collect all slot values across dataset for cross-conversation swaps
        self._build_value_pools()

    def _build_value_pools(self):
        """Build pools of valid values for each slot type (for swapping)."""
        self.value_pools: Dict[str, set] = {}
        for conv in self.dataset.conversations.values():
            for slot_key, slot_val in conv.slot_values.items():
                if slot_key not in self.value_pools:
                    self.value_pools[slot_key] = set()
                self.value_pools[slot_key].add(slot_val)

        # Convert to lists for random sampling
        self.value_pools = {k: list(v) for k, v in self.value_pools.items()}

    # ═══════════════════════════════════════════
    # FAMILY W: WORKFLOW PERTURBATIONS
    # ═══════════════════════════════════════════

    def generate_step_deletion(self, conv: ABCDConversation) -> Optional[Perturbation]:
        """
        Delete one required action from the sequence.
        FULLY DETERMINISTIC. No LLM involved.

        How it works:
        1. Read ground_truth_sequence from kb.json
        2. Pick an action to delete (not the first one — pull-up-account is too trivial)
        3. Remove that action AND its corresponding dialogue turns
        4. The perturbed sequence provably mismatches kb.json
        """
        gt_seq = conv.ground_truth_sequence
        if len(gt_seq) < 3:
            return None  # Too short to meaningfully delete from

        # Pick an action to delete (skip first action, which is usually pull-up-account)
        deletable_indices = list(range(1, len(gt_seq)))
        if not deletable_indices:
            return None

        delete_idx = self.rng.choice(deletable_indices)
        deleted_action = gt_seq[delete_idx]

        # Create perturbed action sequence
        perturbed_seq = gt_seq[:delete_idx] + gt_seq[delete_idx + 1:]

        # Remove corresponding turns from dialogue
        perturbed_turns = []
        action_count = 0
        skip_until_next_speaker_change = False

        for turn in conv.turns:
            if isinstance(turn, dict):
                speaker = turn.get("Speaker", turn.get("speaker", "")).lower()
            elif isinstance(turn, (list, tuple)):
                speaker = str(turn[0]).lower()
            else:
                perturbed_turns.append(turn)
                continue

            if speaker == "action":
                if action_count == delete_idx:
                    # Skip this action turn and the preceding agent turn that set it up
                    skip_until_next_speaker_change = True
                    action_count += 1
                    continue
                action_count += 1

            if skip_until_next_speaker_change and speaker != "action":
                # Also skip the agent turn immediately after the deleted action
                # (which usually says the result of the action)
                skip_until_next_speaker_change = False
                # Only skip one post-action turn
                continue

            perturbed_turns.append(turn)

        # Determine overlap: if deleted action is verify-identity before a restricted action,
        # this is also a Rule violation (authorization bypass)
        overlaps = []
        if deleted_action in ["verify-identity", "validate-purchase"]:
            overlaps.append("R")  # Also a rule violation

        return Perturbation(
            original_convo_id=conv.convo_id,
            family="W",
            perturbation_type="step_deletion",
            description=f"Deleted required action '{deleted_action}' (position {delete_idx}) from sequence",
            is_deterministic=True,
            overlap_tags=overlaps,
            perturbed_turns=perturbed_turns,
            ground_truth_check=f"perturbed_seq {perturbed_seq} != kb.json {gt_seq}",
            ground_truth_result=True,
            violated_rule=f"Missing required action: {deleted_action}",
            perturbed_action_sequence=perturbed_seq,
            severity="severe" if deleted_action in ["verify-identity", "validate-purchase"] else "moderate",
        )

    def generate_step_reordering(self, conv: ABCDConversation) -> Optional[Perturbation]:
        """
        Swap two adjacent actions in the sequence.
        DETERMINISTIC for the swap logic. Dialogue turn swapping is mechanical.
        """
        gt_seq = conv.ground_truth_sequence
        if len(gt_seq) < 3:
            return None

        # Pick a swap point (swap positions i and i+1)
        swap_candidates = list(range(1, len(gt_seq) - 1))
        if not swap_candidates:
            return None

        swap_idx = self.rng.choice(swap_candidates)

        # Create perturbed sequence
        perturbed_seq = list(gt_seq)
        perturbed_seq[swap_idx], perturbed_seq[swap_idx + 1] = (
            perturbed_seq[swap_idx + 1],
            perturbed_seq[swap_idx],
        )

        # For turn swapping: find the action turns and swap the corresponding blocks
        # This is a simplified version — for viability testing, we just note what would change
        perturbed_turns = copy.deepcopy(conv.turns)
        # In production, you'd swap the actual turn blocks here

        return Perturbation(
            original_convo_id=conv.convo_id,
            family="W",
            perturbation_type="step_reordering",
            description=f"Swapped actions at positions {swap_idx} and {swap_idx+1}: "
                        f"'{gt_seq[swap_idx]}' <-> '{gt_seq[swap_idx+1]}'",
            is_deterministic=True,
            overlap_tags=[],
            perturbed_turns=perturbed_turns,
            ground_truth_check=f"perturbed_seq {perturbed_seq} != kb.json {gt_seq}",
            ground_truth_result=True,
            violated_rule=f"Action order violation: {gt_seq[swap_idx]} must precede {gt_seq[swap_idx+1]}",
            perturbed_action_sequence=perturbed_seq,
        )

    def generate_subflow_substitution(self, conv: ABCDConversation) -> Optional[Perturbation]:
        """
        Replace the action sequence with one from a different subflow.
        FULLY DETERMINISTIC.
        """
        gt_seq = conv.ground_truth_sequence
        # Find a different subflow with a different action sequence
        candidates = []
        for sf, seq in self.dataset.kb.items():
            if sf != conv.subflow and seq != gt_seq and len(seq) >= 2:
                candidates.append((sf, seq))

        if not candidates:
            return None

        wrong_sf, wrong_seq = self.rng.choice(candidates)

        return Perturbation(
            original_convo_id=conv.convo_id,
            family="W",
            perturbation_type="subflow_substitution",
            description=f"Replaced action sequence for '{conv.subflow}' with sequence for '{wrong_sf}'",
            is_deterministic=True,
            overlap_tags=[],
            perturbed_turns=copy.deepcopy(conv.turns),  # Would need dialogue rewriting in production
            ground_truth_check=f"Applied subflow '{wrong_sf}' actions to '{conv.subflow}' intent",
            ground_truth_result=True,
            violated_rule=f"Wrong workflow: used {wrong_sf} instead of {conv.subflow}",
            perturbed_action_sequence=wrong_seq,
        )

    # ═══════════════════════════════════════════
    # FAMILY V: VALUE PERTURBATIONS
    # ═══════════════════════════════════════════

    def generate_enumerable_slot_swap(self, conv: ABCDConversation) -> Optional[Perturbation]:
        """
        Replace an enumerable slot value with a different valid value.
        FULLY DETERMINISTIC. No LLM needed.

        Example: membership_level "gold" -> "silver"
        """
        from config import MEMBERSHIP_LEVELS, PAYMENT_METHODS, SHIPPING_STATUSES

        # Find enumerable slots in this conversation
        swappable = []
        slot_pools = {
            "membership": MEMBERSHIP_LEVELS,
            "membership_level": MEMBERSHIP_LEVELS,
            "payment_method": PAYMENT_METHODS,
            "shipping_status": SHIPPING_STATUSES,
        }

        for slot_key, slot_val in conv.slot_values.items():
            field_name = slot_key.split(".")[-1]
            if field_name in slot_pools:
                pool = [v for v in slot_pools[field_name] if v.lower() != slot_val.lower()]
                if pool:
                    swappable.append((slot_key, field_name, slot_val, pool))

        if not swappable:
            return None

        # Pick a slot to swap
        slot_key, field_name, original_val, pool = self.rng.choice(swappable)
        new_val = self.rng.choice(pool)

        # Apply the swap in dialogue text
        perturbed_turns = copy.deepcopy(conv.turns)
        for turn in perturbed_turns:
            if isinstance(turn, dict):
                text = turn.get("Text", turn.get("text", ""))
                # Replace the value in text (case-insensitive)
                text = re.sub(
                    re.escape(original_val),
                    new_val,
                    text,
                    flags=re.IGNORECASE,
                )
                if "Text" in turn:
                    turn["Text"] = text
                elif "text" in turn:
                    turn["text"] = text

        return Perturbation(
            original_convo_id=conv.convo_id,
            family="V",
            perturbation_type="enumerable_slot_swap",
            description=f"Changed {field_name} from '{original_val}' to '{new_val}'",
            is_deterministic=True,
            overlap_tags=[],
            perturbed_turns=perturbed_turns,
            ground_truth_check=f"'{new_val}' != scenario.{slot_key} ('{original_val}') AND '{new_val}' in ontology",
            ground_truth_result=True,
            violated_rule=f"Incorrect {field_name}: stated '{new_val}', actual is '{original_val}'",
            changed_slot=field_name,
            original_value=original_val,
            perturbed_value=new_val,
            severity="moderate",
            evidence_distance="local",
            outcome_critical=field_name in ["membership", "membership_level"],
        )

    def generate_entity_swap(self, conv: ABCDConversation) -> Optional[Perturbation]:
        """
        Replace a non-enumerable entity with one from a different conversation.
        DETERMINISTIC — just a string replacement with a value from the dataset.
        """
        # Find non-enumerable slots
        swappable_fields = ["account_id", "order_id", "phone_number", "email"]
        candidates = []

        for slot_key, slot_val in conv.slot_values.items():
            field_name = slot_key.split(".")[-1]
            pool_key = f"personal.{field_name}"
            if field_name in swappable_fields and pool_key in self.value_pools:
                pool = [v for v in self.value_pools[pool_key] if v != slot_val]
                if pool:
                    candidates.append((slot_key, field_name, slot_val, pool))

        if not candidates:
            return None

        slot_key, field_name, original_val, pool = self.rng.choice(candidates)
        new_val = self.rng.choice(pool)

        perturbed_turns = copy.deepcopy(conv.turns)
        for turn in perturbed_turns:
            if isinstance(turn, dict):
                text = turn.get("Text", turn.get("text", ""))
                text = text.replace(original_val, new_val)
                if "Text" in turn:
                    turn["Text"] = text
                elif "text" in turn:
                    turn["text"] = text

        return Perturbation(
            original_convo_id=conv.convo_id,
            family="V",
            perturbation_type="entity_swap",
            description=f"Changed {field_name} from '{original_val}' to '{new_val}' (from another conversation)",
            is_deterministic=True,
            overlap_tags=[],
            perturbed_turns=perturbed_turns,
            ground_truth_check=f"'{new_val}' != scenario.{slot_key} ('{original_val}')",
            ground_truth_result=True,
            violated_rule=f"Incorrect {field_name}: stated '{new_val}', actual is '{original_val}'",
            changed_slot=field_name,
            original_value=original_val,
            perturbed_value=new_val,
            severity="severe" if field_name == "account_id" else "moderate",
            evidence_distance="local",
            outcome_critical=field_name in ["account_id", "order_id"],
        )

    # ═══════════════════════════════════════════
    # FAMILY R: RULE PERTURBATIONS
    # ═══════════════════════════════════════════

    def generate_authorization_bypass(self, conv: ABCDConversation) -> Optional[Perturbation]:
        """
        Remove verify-identity before a restricted action.
        This is a step_deletion that specifically targets a policy-defined constraint.
        DETERMINISTIC — it's step_deletion with a rule-based motivation.
        """
        gt_seq = conv.ground_truth_sequence
        if "verify-identity" not in gt_seq:
            return None

        vi_idx = gt_seq.index("verify-identity")

        # Check that there's a restricted action after verify-identity
        restricted_actions = ["update-account", "update-order", "make-password", "make-purchase"]
        has_restricted_after = any(
            a in restricted_actions for a in gt_seq[vi_idx + 1:]
        )
        if not has_restricted_after:
            return None

        # Delete verify-identity
        perturbed_seq = [a for a in gt_seq if a != "verify-identity"]

        # Remove corresponding turns
        perturbed_turns = []
        action_count = 0
        skip_next = False

        for turn in conv.turns:
            if isinstance(turn, dict):
                speaker = turn.get("Speaker", turn.get("speaker", "")).lower()
            elif isinstance(turn, (list, tuple)):
                speaker = str(turn[0]).lower()
            else:
                perturbed_turns.append(turn)
                continue

            if speaker == "action":
                if action_count == vi_idx:
                    action_count += 1
                    skip_next = True  # Skip the post-action agent response too
                    continue
                action_count += 1

            if skip_next and speaker != "action":
                skip_next = False
                continue

            perturbed_turns.append(turn)

        return Perturbation(
            original_convo_id=conv.convo_id,
            family="R",
            perturbation_type="authorization_bypass",
            description="Removed identity verification before restricted account action",
            is_deterministic=True,
            overlap_tags=["W"],  # Also a workflow step-deletion
            perturbed_turns=perturbed_turns,
            ground_truth_check="verify-identity absent before restricted action in guidelines.json",
            ground_truth_result=True,
            violated_rule="Authorization bypass: account modified without identity verification",
            perturbed_action_sequence=perturbed_seq,
            severity="severe",
        )

    def generate_eligibility_violation(self, conv: ABCDConversation) -> Optional[Perturbation]:
        """
        Change the customer's membership level to one that shouldn't qualify.
        DETERMINISTIC — change the membership reference in dialogue.
        """
        membership = conv.slot_values.get("personal.membership_level", "")
        if not membership or membership.lower() == "guest":
            return None  # Already lowest tier

        # Downgrade to a lower tier
        tier_order = ["gold", "silver", "bronze", "guest"]
        current_idx = next(
            (i for i, t in enumerate(tier_order) if t == membership.lower()), -1
        )
        if current_idx < 0 or current_idx >= len(tier_order) - 1:
            return None

        # Pick a lower tier
        downgraded = tier_order[current_idx + 1]

        perturbed_turns = copy.deepcopy(conv.turns)
        for turn in perturbed_turns:
            if isinstance(turn, dict):
                text = turn.get("Text", turn.get("text", ""))
                text = re.sub(
                    re.escape(membership),
                    downgraded,
                    text,
                    flags=re.IGNORECASE,
                )
                if "Text" in turn:
                    turn["Text"] = text
                elif "text" in turn:
                    turn["text"] = text

        return Perturbation(
            original_convo_id=conv.convo_id,
            family="R",
            perturbation_type="eligibility_violation",
            description=f"Changed membership from '{membership}' to '{downgraded}' — "
                        f"customer may not qualify for the service provided",
            is_deterministic=True,
            overlap_tags=["V"],  # Also a value perturbation
            perturbed_turns=perturbed_turns,
            ground_truth_check=f"Agent provides {membership}-level service to {downgraded} customer",
            ground_truth_result=True,
            violated_rule=f"Eligibility violation: {downgraded} customer received {membership}-tier service",
            changed_slot="membership_level",
            original_value=membership,
            perturbed_value=downgraded,
            severity="moderate",
        )

    # ═══════════════════════════════════════════
    # FAMILY N: NUISANCE / CONTROLS
    # ═══════════════════════════════════════════

    def generate_nuisance_paraphrase(self, conv: ABCDConversation) -> Perturbation:
        """
        Mark a conversation for nuisance paraphrasing.
        The actual paraphrasing requires LLM — this creates the metadata.
        For viability testing, we just flag it.
        """
        return Perturbation(
            original_convo_id=conv.convo_id,
            family="N",
            perturbation_type="nuisance_paraphrase",
            description="Agent utterances reworded; all actions, values, and ordering preserved",
            is_deterministic=False,  # Requires LLM for actual paraphrasing
            perturbed_turns=copy.deepcopy(conv.turns),  # Placeholder
            ground_truth_check="No compliance violation introduced",
            ground_truth_result=True,
            violated_rule="NONE — this is a control",
        )

    def generate_length_matched_control(self, conv: ABCDConversation) -> Optional[Perturbation]:
        """
        Delete non-action turns to match the length reduction of a step-deletion.
        DETERMINISTIC.
        """
        # Find non-action, non-essential turns
        removable_indices = []
        for i, turn in enumerate(conv.turns):
            if isinstance(turn, dict):
                speaker = turn.get("Speaker", turn.get("speaker", "")).lower()
            elif isinstance(turn, (list, tuple)):
                speaker = str(turn[0]).lower()
            else:
                continue

            # Only remove customer small-talk or agent pleasantries (not action turns)
            if speaker in ["customer", "agent"]:
                text = ""
                if isinstance(turn, dict):
                    text = turn.get("Text", turn.get("text", "")).lower()
                elif isinstance(turn, (list, tuple)):
                    text = str(turn[1]).lower()

                # Heuristic: short turns with greetings/pleasantries are removable
                if len(text.split()) < 8 and any(
                    w in text for w in ["thank", "hello", "hi", "great", "sure", "ok", "bye", "welcome"]
                ):
                    removable_indices.append(i)

        # Remove 2-3 turns to match typical step-deletion length reduction
        num_to_remove = min(3, len(removable_indices))
        if num_to_remove < 2:
            return None

        to_remove = set(self.rng.sample(removable_indices, num_to_remove))
        perturbed_turns = [t for i, t in enumerate(conv.turns) if i not in to_remove]

        return Perturbation(
            original_convo_id=conv.convo_id,
            family="N",
            perturbation_type="length_matched_control",
            description=f"Removed {num_to_remove} non-action turns (pleasantries/small talk only)",
            is_deterministic=True,
            perturbed_turns=perturbed_turns,
            ground_truth_check="All kb.json actions still present; only small talk removed",
            ground_truth_result=True,
            violated_rule="NONE — length-matched control",
        )

    # ═══════════════════════════════════════════
    # BATCH GENERATION
    # ═══════════════════════════════════════════

    def generate_all_for_conversation(
        self, conv: ABCDConversation
    ) -> List[Perturbation]:
        """Generate one perturbation of each type for a conversation (where possible)."""
        perturbations = []

        generators = [
            ("W", "step_deletion", self.generate_step_deletion),
            ("W", "step_reordering", self.generate_step_reordering),
            ("W", "subflow_substitution", self.generate_subflow_substitution),
            ("V", "enumerable_slot_swap", self.generate_enumerable_slot_swap),
            ("V", "entity_swap", self.generate_entity_swap),
            ("R", "authorization_bypass", self.generate_authorization_bypass),
            ("R", "eligibility_violation", self.generate_eligibility_violation),
            ("N", "nuisance_paraphrase", self.generate_nuisance_paraphrase),
            ("N", "length_matched_control", self.generate_length_matched_control),
        ]

        for family, ptype, gen_func in generators:
            try:
                result = gen_func(conv)
                if result is not None:
                    perturbations.append(result)
            except Exception as e:
                print(f"  Warning: {ptype} failed for {conv.convo_id}: {e}")

        return perturbations

    def generate_viability_set(
        self, conversations: List[ABCDConversation], per_family: int = 10
    ) -> Dict[str, List[Perturbation]]:
        """
        Generate a small viability set: per_family perturbations per family.
        Returns dict keyed by family.
        """
        by_family: Dict[str, List[Perturbation]] = {"W": [], "V": [], "R": [], "N": []}

        for conv in conversations:
            all_perts = self.generate_all_for_conversation(conv)
            for pert in all_perts:
                if len(by_family[pert.family]) < per_family:
                    by_family[pert.family].append(pert)

            # Check if we have enough
            if all(len(v) >= per_family for v in by_family.values()):
                break

        return by_family


import re  # needed at module level

if __name__ == "__main__":
    from config import ABCD_DIR, VIABILITY_SAMPLE_SIZE, PERTURBATIONS_PER_FAMILY

    dataset = ABCDDataset(ABCD_DIR)
    engine = PerturbationEngine(dataset)

    compliant = dataset.get_compliant_conversations()
    print(f"\nCompliant conversations available: {len(compliant)}")

    if compliant:
        # Generate viability set
        sample = compliant[:VIABILITY_SAMPLE_SIZE]
        viability = engine.generate_viability_set(sample, per_family=PERTURBATIONS_PER_FAMILY)

        print("\n--- Viability Set Summary ---")
        for family, perts in viability.items():
            print(f"\nFamily {family}: {len(perts)} perturbations")
            for p in perts[:3]:
                print(f"  [{p.perturbation_type}] {p.description}")
                print(f"    Deterministic: {p.is_deterministic}")
                print(f"    Ground truth: {p.ground_truth_check}")
                if p.overlap_tags:
                    print(f"    Overlaps with: {p.overlap_tags}")
