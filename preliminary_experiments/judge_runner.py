"""
Judge Runner
Handles LLM API calls for judge evaluation.
Replace the call_judge_api() function with your actual API implementation.
"""
import json
import time
from typing import Dict, Optional
from utils.prompt_templates import make_pointwise_prompt, make_pairwise_prompt, make_compressed_policy
from utils.metrics import JudgeOutput, parse_judge_response


def call_judge_api(
    prompt: str,
    model_config: dict,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> str:
    """
    Call an LLM judge API.

    *** REPLACE THIS FUNCTION WITH YOUR ACTUAL API CALL ***

    Args:
        prompt: The evaluation prompt
        model_config: Dict with "endpoint", "model_id", "type"
        temperature: Should be 0.0 for deterministic judging
        max_tokens: Max response tokens

    Returns:
        Raw text response from the model
    """
    # ─── PLACEHOLDER ───
    # Replace this with your actual API call. Examples:

    # Option A: OpenAI-compatible API
    # import openai
    # client = openai.OpenAI(base_url=model_config["endpoint"], api_key="YOUR_KEY")
    # response = client.chat.completions.create(
    #     model=model_config["model_id"],
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    # )
    # return response.choices[0].message.content

    # Option B: requests-based API
    # import requests
    # resp = requests.post(
    #     model_config["endpoint"],
    #     json={"model": model_config["model_id"], "prompt": prompt,
    #           "temperature": temperature, "max_tokens": max_tokens},
    #     headers={"Authorization": f"Bearer YOUR_KEY"},
    # )
    # return resp.json()["choices"][0]["text"]

    # Option C: HuggingFace local inference (for Prometheus 2)
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # ... local inference code ...

    raise NotImplementedError(
        "Replace call_judge_api() in judge_runner.py with your actual API implementation. "
        "See the comments in the function for examples."
    )


def run_pointwise_evaluation(
    dialogue_text: str,
    convo_id: str,
    is_original: bool,
    perturbation_family: str,
    perturbation_type: str,
    customer_intent: str,
    model_name: str,
    model_config: dict,
    prompt_condition: str,
    policy_text: str = "",
    action_sequence: list = None,
) -> JudgeOutput:
    """
    Run a single pointwise evaluation.

    Returns a JudgeOutput with scores and rationale.
    """
    compressed = make_compressed_policy(customer_intent, action_sequence or [])

    prompt = make_pointwise_prompt(
        dialogue_text=dialogue_text,
        customer_intent=customer_intent.replace("_", " ").title(),
        condition=prompt_condition,
        policy_text=policy_text,
        compressed_policy=compressed,
    )

    try:
        response_text = call_judge_api(prompt, model_config)
        parsed = parse_judge_response(response_text)
    except NotImplementedError:
        # Return a dummy output for pipeline testing
        parsed = {
            "workflow_correctness": 0,
            "value_accuracy": 0,
            "rule_compliance": 0,
            "overall_quality": 0,
            "rationale": "API NOT CONFIGURED",
        }
    except Exception as e:
        parsed = {
            "workflow_correctness": 0,
            "value_accuracy": 0,
            "rule_compliance": 0,
            "overall_quality": 0,
            "rationale": f"ERROR: {str(e)}",
        }

    return JudgeOutput(
        convo_id=convo_id,
        is_original=is_original,
        perturbation_family=perturbation_family,
        perturbation_type=perturbation_type,
        model=model_name,
        prompt_condition=prompt_condition,
        workflow_correctness=float(parsed.get("workflow_correctness", 0)),
        value_accuracy=float(parsed.get("value_accuracy", 0)),
        rule_compliance=float(parsed.get("rule_compliance", 0)),
        overall_quality=float(parsed.get("overall_quality", 0)),
        rationale=parsed.get("rationale", ""),
    )


def run_pairwise_evaluation(
    original_text: str,
    perturbed_text: str,
    convo_id: str,
    perturbation_family: str,
    perturbation_type: str,
    customer_intent: str,
    model_name: str,
    model_config: dict,
    prompt_condition: str,
    policy_text: str = "",
    action_sequence: list = None,
    original_first: bool = True,
) -> dict:
    """
    Run a single pairwise evaluation.
    Returns dict with preference result.
    """
    compressed = make_compressed_policy(customer_intent, action_sequence or [])

    if original_first:
        dialogue_a, dialogue_b = original_text, perturbed_text
    else:
        dialogue_a, dialogue_b = perturbed_text, original_text

    prompt = make_pairwise_prompt(
        dialogue_a=dialogue_a,
        dialogue_b=dialogue_b,
        customer_intent=customer_intent.replace("_", " ").title(),
        condition=prompt_condition,
        policy_text=policy_text,
        compressed_policy=compressed,
    )

    try:
        response_text = call_judge_api(prompt, model_config)
        parsed = parse_judge_response(response_text)
    except NotImplementedError:
        parsed = {"preferred": "A", "confidence": "low", "reason": "API NOT CONFIGURED"}
    except Exception as e:
        parsed = {"preferred": "A", "confidence": "low", "reason": f"ERROR: {str(e)}"}

    # Map back to original/perturbed
    preferred_raw = parsed.get("preferred", "A")
    if original_first:
        judge_preferred = "original" if preferred_raw == "A" else "perturbed"
    else:
        judge_preferred = "original" if preferred_raw == "B" else "perturbed"

    return {
        "convo_id": convo_id,
        "family": perturbation_family,
        "perturbation_type": perturbation_type,
        "model": model_name,
        "prompt_condition": prompt_condition,
        "judge_preferred": judge_preferred,
        "confidence": parsed.get("confidence", "unknown"),
        "reason": parsed.get("reason", ""),
        "original_first": original_first,
    }
