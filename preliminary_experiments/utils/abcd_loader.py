"""
ABCD Data Loader
Loads and parses all ABCD data files (abcd_v1.1.json, kb.json, guidelines.json, ontology.json).
Provides structured access to conversations, action sequences, policies, and slot values.
"""
import json
import gzip
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ABCDConversation:
    """Parsed ABCD conversation with all structured annotations."""
    convo_id: str
    flow: str
    subflow: str
    scenario: dict
    turns: List[dict]                    # Raw turn data
    action_sequence: List[str]           # Extracted action sequence from the dialogue
    ground_truth_sequence: List[str]     # Required sequence from kb.json
    slot_values: Dict[str, str]          # Extracted slot values from scenario
    num_turns: int = 0
    is_policy_compliant: bool = False    # Whether action sequence matches kb.json


def load_abcd_data(data_path: str) -> dict:
    """Load the main ABCD dataset (handles both .json and .json.gz)."""
    if data_path.endswith(".gz"):
        with gzip.open(data_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_subflow_name(name: str) -> str:
    """
    Normalize subflow names between different ABCD files.
    guidelines.json uses 'Capitalized Names With Spaces'
    conversations use 'lowercase-with-hyphens' or 'lowercase_with_underscores'
    kb.json uses 'lowercase_with_underscores'
    """
    # Convert to lowercase with underscores
    normalized = name.lower().strip()
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    # Remove extra underscores
    normalized = re.sub(r"_+", "_", normalized)
    return normalized


def extract_action_sequence(conversation: dict) -> List[str]:
    """
    Extract the ordered sequence of actions from a conversation.
    Actions are turns where Speaker == "action".
    """
    actions = []
    turns = conversation.get("delexed", conversation.get("original", []))

    for turn in turns:
        if isinstance(turn, dict):
            speaker = turn.get("Speaker", turn.get("speaker", ""))
            if speaker.lower() == "action":
                # Extract the action type from the turn text
                text = turn.get("Text", turn.get("text", ""))
                # Action turns typically start with the action name
                action_name = text.strip().split()[0] if text.strip() else ""
                # Clean up: remove brackets, parentheses
                action_name = action_name.strip("[]()").lower()
                if action_name:
                    actions.append(action_name)
        elif isinstance(turn, (list, tuple)):
            # Some formats use [speaker, text] tuples
            if len(turn) >= 2:
                speaker = str(turn[0]).lower()
                if speaker == "action":
                    text = str(turn[1]).strip()
                    action_name = text.split()[0].strip("[]()").lower() if text else ""
                    if action_name:
                        actions.append(action_name)
    return actions


def extract_slot_values(scenario: dict) -> Dict[str, str]:
    """Extract all slot values from a conversation scenario."""
    slots = {}
    for category in ["Personal", "Order", "Product"]:
        if category in scenario:
            for key, value in scenario[category].items():
                slots[f"{category.lower()}.{key}"] = str(value)
    return slots


def count_turns(conversation: dict) -> int:
    """Count the number of turns in a conversation."""
    turns = conversation.get("delexed", conversation.get("original", []))
    return len(turns)


def parse_conversation(
    convo_id: str,
    convo_data: dict,
    kb: dict,
) -> ABCDConversation:
    """Parse a single ABCD conversation into a structured object."""
    scenario = convo_data.get("scenario", {})
    flow = scenario.get("Flow", scenario.get("flow", ""))
    subflow = scenario.get("Subflow", scenario.get("subflow", ""))

    # Normalize subflow name for kb.json lookup
    subflow_normalized = normalize_subflow_name(subflow)

    # Get ground truth action sequence from kb.json
    ground_truth = kb.get(subflow_normalized, [])

    # Extract actual action sequence from dialogue
    action_seq = extract_action_sequence(convo_data)

    # Extract slot values
    slot_values = extract_slot_values(scenario)

    # Count turns
    num_turns = count_turns(convo_data)

    # Check policy compliance (action sequence matches kb.json)
    is_compliant = (action_seq == ground_truth) if ground_truth else False

    turns = convo_data.get("delexed", convo_data.get("original", []))

    return ABCDConversation(
        convo_id=convo_id,
        flow=normalize_subflow_name(flow),
        subflow=subflow_normalized,
        scenario=scenario,
        turns=turns,
        action_sequence=action_seq,
        ground_truth_sequence=ground_truth,
        slot_values=slot_values,
        num_turns=num_turns,
        is_policy_compliant=is_compliant,
    )


class ABCDDataset:
    """
    Complete ABCD dataset with all structured annotations.
    Provides methods for sampling, filtering, and accessing conversations.
    """

    def __init__(self, abcd_dir: str):
        self.abcd_dir = abcd_dir
        self.conversations: Dict[str, ABCDConversation] = {}
        self.kb: dict = {}
        self.guidelines: dict = {}
        self.ontology: dict = {}

        self._load_all()

    def _load_all(self):
        """Load all ABCD data files."""
        print("Loading ABCD dataset...")

        # Load kb.json (action sequences)
        kb_path = os.path.join(self.abcd_dir, "kb.json")
        if os.path.exists(kb_path):
            self.kb = load_json(kb_path)
            print(f"  kb.json: {len(self.kb)} subflow action sequences")
        else:
            print(f"  WARNING: kb.json not found at {kb_path}")

        # Load guidelines.json (policy rules)
        guide_path = os.path.join(self.abcd_dir, "guidelines.json")
        if os.path.exists(guide_path):
            self.guidelines = load_json(guide_path)
            print(f"  guidelines.json: {len(self.guidelines)} flow entries")
        else:
            print(f"  WARNING: guidelines.json not found at {guide_path}")

        # Load ontology.json
        onto_path = os.path.join(self.abcd_dir, "ontology.json")
        if os.path.exists(onto_path):
            self.ontology = load_json(onto_path)
            print(f"  ontology.json loaded")
        else:
            print(f"  WARNING: ontology.json not found at {onto_path}")

        # Load main data
        data_path = os.path.join(self.abcd_dir, "abcd_v1.1.json")
        gz_path = data_path + ".gz"

        if os.path.exists(gz_path):
            raw = load_abcd_data(gz_path)
        elif os.path.exists(data_path):
            raw = load_abcd_data(data_path)
        else:
            raise FileNotFoundError(
                f"ABCD data not found at {data_path} or {gz_path}"
            )

        # Parse all conversations
        total = 0
        for split in ["train", "dev", "test"]:
            if split not in raw:
                continue
            convos = raw[split]
            if isinstance(convos, list):
                for item in convos:
                    cid = item.get("convo_id", f"{split}_{total}")
                    parsed = parse_conversation(cid, item, self.kb)
                    self.conversations[cid] = parsed
                    total += 1
            elif isinstance(convos, dict):
                for cid, item in convos.items():
                    parsed = parse_conversation(cid, item, self.kb)
                    self.conversations[cid] = parsed
                    total += 1

        print(f"  Total conversations loaded: {total}")
        compliant = sum(1 for c in self.conversations.values() if c.is_policy_compliant)
        print(f"  Policy-compliant conversations: {compliant}")

    def get_compliant_conversations(
        self,
        min_turns: int = 12,
        require_slot_values: bool = True,
    ) -> List[ABCDConversation]:
        """Get policy-compliant conversations suitable for perturbation."""
        results = []
        for conv in self.conversations.values():
            if not conv.is_policy_compliant:
                continue
            if conv.num_turns < min_turns:
                continue
            if require_slot_values and not conv.slot_values:
                continue
            if not conv.ground_truth_sequence:
                continue
            results.append(conv)
        return results

    def get_subflow_distribution(self) -> Dict[str, int]:
        """Get the distribution of subflows across compliant conversations."""
        dist = {}
        for conv in self.get_compliant_conversations():
            dist[conv.subflow] = dist.get(conv.subflow, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def get_policy_for_subflow(self, subflow: str) -> Optional[str]:
        """Get the policy text for a given subflow from guidelines.json."""
        subflow_norm = normalize_subflow_name(subflow)
        # guidelines.json is structured as {flow: {subflow: policy_text}}
        for flow_name, subflows in self.guidelines.items():
            if isinstance(subflows, dict):
                for sf_name, policy_text in subflows.items():
                    if normalize_subflow_name(sf_name) == subflow_norm:
                        return policy_text
            elif isinstance(subflows, str):
                # Some formats have flat structure
                if normalize_subflow_name(flow_name) == subflow_norm:
                    return subflows
        return None

    def get_action_sequence(self, subflow: str) -> List[str]:
        """Get the required action sequence for a subflow from kb.json."""
        return self.kb.get(normalize_subflow_name(subflow), [])

    def format_dialogue_text(self, conv: ABCDConversation) -> str:
        """Format a conversation as readable dialogue text for judge input."""
        lines = []
        for i, turn in enumerate(conv.turns):
            if isinstance(turn, dict):
                speaker = turn.get("Speaker", turn.get("speaker", "unknown"))
                text = turn.get("Text", turn.get("text", ""))
            elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                speaker = str(turn[0])
                text = str(turn[1])
            else:
                continue

            # Re-lexicalize with actual values for readability
            for slot_key, slot_val in conv.slot_values.items():
                field_name = slot_key.split(".")[-1]
                token = f"<{field_name}>"
                text = text.replace(token, slot_val)

            lines.append(f"[{speaker.upper()}]: {text}")

        return "\n".join(lines)


if __name__ == "__main__":
    from config import ABCD_DIR
    dataset = ABCDDataset(ABCD_DIR)

    print("\n--- Dataset Summary ---")
    print(f"Total conversations: {len(dataset.conversations)}")

    compliant = dataset.get_compliant_conversations()
    print(f"Compliant with >=12 turns: {len(compliant)}")

    dist = dataset.get_subflow_distribution()
    print(f"\nTop 10 subflows:")
    for sf, count in list(dist.items())[:10]:
        seq = dataset.get_action_sequence(sf)
        print(f"  {sf}: {count} convos, actions: {seq}")

    if compliant:
        sample = compliant[0]
        print(f"\nSample conversation: {sample.convo_id}")
        print(f"  Flow: {sample.flow}")
        print(f"  Subflow: {sample.subflow}")
        print(f"  Turns: {sample.num_turns}")
        print(f"  Actions: {sample.action_sequence}")
        print(f"  Ground truth: {sample.ground_truth_sequence}")
        print(f"  Slot values: {sample.slot_values}")
        print(f"\n  First 5 turns of dialogue:")
        text = dataset.format_dialogue_text(sample)
        for line in text.split("\n")[:5]:
            print(f"    {line}")
