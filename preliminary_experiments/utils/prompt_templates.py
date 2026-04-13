"""
Prompt Templates for LLM Judge Evaluation
Three prompt conditions: no-policy, compressed-policy, full-policy.
Two evaluation paradigms: pointwise scoring, pairwise comparison.
"""

# ═══════════════════════════════════════════
# POINTWISE SCORING PROMPTS
# ═══════════════════════════════════════════

POINTWISE_RUBRIC = """Score this customer service dialogue on four dimensions using a 1-5 scale.

Scoring rubric:
1 = Very Poor: Major failures in this dimension
2 = Poor: Significant issues that would cause real problems
3 = Acceptable: Minor issues but generally functional
4 = Good: Meets expectations with no notable problems
5 = Excellent: Exemplary performance in this dimension

Dimensions:
1. Workflow Correctness: Did the agent follow the correct procedural steps in the right order?
2. Value Accuracy: Did the agent reference correct customer data, account details, and factual information?
3. Rule Compliance: Did the agent follow applicable business rules and policies?
4. Overall Quality: Considering all aspects, how good was this dialogue overall?

Respond in this exact JSON format:
{{
  "workflow_correctness": <1-5>,
  "value_accuracy": <1-5>,
  "rule_compliance": <1-5>,
  "overall_quality": <1-5>,
  "rationale": "<brief explanation of your scores, noting any issues you identified>"
}}"""


def make_pointwise_prompt(
    dialogue_text: str,
    customer_intent: str,
    condition: str = "no_policy",
    policy_text: str = "",
    compressed_policy: str = "",
) -> str:
    """
    Create a pointwise evaluation prompt under one of three conditions.

    Args:
        dialogue_text: The formatted dialogue to evaluate
        customer_intent: The stated customer intent/subflow
        condition: "no_policy", "compressed_policy", or "full_policy"
        policy_text: Full policy text from guidelines.json (for full_policy)
        compressed_policy: 2-3 sentence summary (for compressed_policy)
    """
    if condition == "no_policy":
        return f"""You are evaluating a customer service dialogue.

The customer's intent is: {customer_intent}

{POINTWISE_RUBRIC}

Dialogue:
{dialogue_text}"""

    elif condition == "compressed_policy":
        return f"""You are evaluating a customer service dialogue.

The customer's intent is: {customer_intent}

Policy summary for this type of request:
{compressed_policy}

{POINTWISE_RUBRIC}

Dialogue:
{dialogue_text}"""

    elif condition == "full_policy":
        return f"""You are evaluating a customer service dialogue.

The customer's intent is: {customer_intent}

Complete policy and workflow specification for this type of request:
{policy_text}

{POINTWISE_RUBRIC}

Dialogue:
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

    return f"""You are comparing two customer service dialogues for the same customer request.

The customer's intent is: {customer_intent}
{policy_block}
Which dialogue better follows proper customer service procedures and policies?

Dialogue A:
{dialogue_a}

Dialogue B:
{dialogue_b}

Respond in this exact JSON format:
{{
  "preferred": "A" or "B",
  "confidence": "high" or "medium" or "low",
  "reason": "<brief explanation of why you preferred this dialogue>"
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
        f"For a '{subflow_readable}' request, the agent should follow these steps: "
        f"{action_str}. "
        f"Identity verification must be completed before any account modifications. "
        f"All customer information must be verified against the account records."
    )
