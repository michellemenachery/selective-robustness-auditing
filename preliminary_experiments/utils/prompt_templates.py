"""
Prompt Templates for LLM Judge Evaluation
Three prompt conditions: no-policy, compressed-policy, full-policy.
Two evaluation paradigms: pointwise scoring, pairwise comparison.

RUBRIC DESIGN: Questions are specific and diagnostic to avoid 5/5 clustering.
"""

# ═══════════════════════════════════════════
# POINTWISE SCORING PROMPTS
# ═══════════════════════════════════════════

POINTWISE_RUBRIC = """You are auditing a customer service dialogue for compliance failures. Be critical — most real dialogues have at least minor issues. A score of 5 should be rare.

Score on four dimensions using a 1-5 scale:

1. Workflow Correctness — Did the agent execute every required procedural step, in the correct order, without skipping or repeating any?
   5 = Every required step executed in correct order, no extras
   4 = All steps present but minor ordering issue or one unnecessary extra step
   3 = One required step missing OR two steps clearly out of order
   2 = Multiple steps missing or seriously wrong order
   1 = Agent followed a completely wrong procedure

2. Value Accuracy — Are ALL customer details (name, account ID, membership level, phone number, order ID, amounts, addresses) correct and consistent throughout the dialogue?
   5 = Every value matches, fully consistent across all turns
   4 = All values correct but one minor inconsistency in phrasing
   3 = One factual value appears incorrect or contradicts another turn
   2 = Multiple incorrect values or clear contradictions between turns
   1 = Agent references completely wrong customer data

3. Rule Compliance — Did the agent follow business rules? (e.g., verify identity before account changes, respect discount limits, check eligibility before granting benefits)
   5 = All applicable rules followed
   4 = Rules followed but one borderline case handled loosely
   3 = One clear rule violation (e.g., skipped verification, exceeded a limit)
   2 = Multiple rule violations
   1 = Agent ignored fundamental policy requirements

4. Overall Quality — Considering workflow, values, and rules together, how compliant was this dialogue?
   5 = Flawless compliance
   4 = Minor issues only, operationally acceptable
   3 = One significant compliance gap
   2 = Multiple compliance failures
   1 = Dialogue is operationally unacceptable

IMPORTANT: Carefully check each value mentioned by the agent. Verify that membership levels, account IDs, and other details are consistent. Do not assume correctness — verify.

Respond in this exact JSON format:
{
  "workflow_correctness": <1-5>,
  "value_accuracy": <1-5>,
  "rule_compliance": <1-5>,
  "overall_quality": <1-5>,
  "rationale": "<Specifically state what you checked and any issues found. If you found no issues, explain what you verified.>"
}"""


def make_pointwise_prompt(
    dialogue_text: str,
    customer_intent: str,
    condition: str = "no_policy",
    policy_text: str = "",
    compressed_policy: str = "",
) -> str:
    """
    Create a pointwise evaluation prompt under one of three conditions.
    """
    if condition == "no_policy":
        return f"""You are auditing a customer service dialogue for compliance.

The customer's stated request is: {customer_intent}

{POINTWISE_RUBRIC}

Dialogue to evaluate:
{dialogue_text}"""

    elif condition == "compressed_policy":
        return f"""You are auditing a customer service dialogue for compliance.

The customer's stated request is: {customer_intent}

Policy summary for this type of request:
{compressed_policy}

{POINTWISE_RUBRIC}

Dialogue to evaluate:
{dialogue_text}"""

    elif condition == "full_policy":
        return f"""You are auditing a customer service dialogue for compliance.

The customer's stated request is: {customer_intent}

Complete policy and workflow specification for this type of request:
{policy_text}

Required action sequence: This request type requires the agent to execute specific steps in a defined order. Check whether the agent completed all required steps.

{POINTWISE_RUBRIC}

Dialogue to evaluate:
{dialogue_text}"""

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════
# PAIRWISE COMPARISON PROMPTS
# ═══════════════════════════════════════════

def make_pairwise_prompt(
    dialogue_a: str,
    dialogue_b: str,
    customer_intent: str,
    condition: str = "no_policy",
    policy_text: str = "",
    compressed_policy: str = "",
) -> str:
    """
    Create a pairwise comparison prompt.
    Dialogue A and B are presented in randomized order (caller handles randomization).
    """
    policy_block = ""
    if condition == "compressed_policy":
        policy_block = f"\nPolicy summary for this type of request:\n{compressed_policy}\n"
    elif condition == "full_policy":
        policy_block = f"\nComplete policy and workflow specification:\n{policy_text}\n"

    return f"""You are comparing two customer service dialogues handling the same request. One may contain a compliance violation that the other does not. Be critical and specific.

The customer's stated request is: {customer_intent}
{policy_block}
Carefully check both dialogues for:
- Missing or reordered procedural steps
- Incorrect customer details (wrong membership level, wrong account ID, wrong amounts)
- Policy rule violations (skipped verification, exceeded limits)

Dialogue A:
{dialogue_a}

Dialogue B:
{dialogue_b}

Which dialogue is MORE compliant with proper procedures and policies? Respond in this exact JSON format:
{{
  "preferred": "A" or "B",
  "confidence": "high" or "medium" or "low",
  "reason": "<Specifically state what violation you found in the worse dialogue, or what the better dialogue did correctly>"
}}"""


# ═══════════════════════════════════════════
# COMPRESSED POLICY GENERATION
# ═══════════════════════════════════════════

def make_compressed_policy(subflow: str, action_sequence: list) -> str:
    """
    Generate a 2-3 sentence compressed policy summary from the subflow name
    and action sequence. This is deterministic — no LLM needed.
    """
    action_str = " → ".join(action_sequence)
    subflow_readable = subflow.replace("_", " ").replace("-", " ").title()

    return (
        f"For a '{subflow_readable}' request, the agent must follow these steps in order: "
        f"{action_str}. "
        f"Identity verification must be completed before any account modifications. "
        f"All customer information referenced by the agent must match the account records exactly."
    )
