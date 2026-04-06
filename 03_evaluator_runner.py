"""
Step 3: Evaluator Runner

Sends original, Family A, and Family D conversations to LLM evaluators
and collects structured scores.

This is the core experiment. For each conversation, the evaluator sees:
1. The conversation text
2. The relevant policy guidelines
3. A scoring prompt

The evaluator returns scores on policy compliance, task completion,
slot accuracy, and overall quality.

Setup:
    1. Fill in your model endpoints and API keys in the .env file or below
    2. Copy perturbations.json and policy_lookup.json to this machine
    3. pip install requests python-dotenv tqdm
    4. python 03_evaluator_runner.py

Output:
    data/experiment/evaluator_results.json
"""

import json
import os
import re
import time
import random
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "experiment"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════
# MODEL CONFIGURATION — FILL THESE IN
# ═══════════════════════════════════════════════════════════

MODELS = {
    "llama_70b": {
        "name": "Llama 3.3 70B",
        "endpoint": os.getenv("LLAMA_70B_ENDPOINT", "https://your-cluster/v1/chat/completions"),
        "api_key": os.getenv("LLAMA_70B_API_KEY", "your-api-key"),
        "model_id": os.getenv("LLAMA_70B_MODEL_ID", "llama-3.3-70b"),
    },
    "gpt_oss_120b": {
        "name": "GPT-OSS 120B",
        "endpoint": os.getenv("GPT_OSS_120B_ENDPOINT", "https://your-cluster/v1/chat/completions"),
        "api_key": os.getenv("GPT_OSS_120B_API_KEY", "your-api-key"),
        "model_id": os.getenv("GPT_OSS_120B_MODEL_ID", "gpt-oss-120b"),
    },
    "gpt_oss_20b": {
        "name": "GPT-OSS 20B",
        "endpoint": os.getenv("GPT_OSS_20B_ENDPOINT", "https://your-cluster/v1/chat/completions"),
        "api_key": os.getenv("GPT_OSS_20B_API_KEY", "your-api-key"),
        "model_id": os.getenv("GPT_OSS_20B_MODEL_ID", "gpt-oss-20b"),
    },
}

# ═══════════════════════════════════════════════════════════
# EVALUATION PROMPT
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert quality assurance evaluator for a customer service organization. 
Your job is to evaluate whether a customer service agent followed company policy correctly during a conversation.
You must be thorough and precise. Evaluate based on the policy guidelines provided, not on general impressions."""

EVAL_PROMPT_TEMPLATE = """Evaluate this customer service conversation against the company policy below.

## Company Policy for "{flow} — {subflow}"

### Required Actions (in order):
{required_actions}

### Action Details:
{action_details}

## Conversation:
{conversation}

## Scoring Instructions:
Rate the agent's performance on each dimension using a 1-5 scale:
- 1 = Complete failure
- 2 = Major issues  
- 3 = Partial compliance
- 4 = Mostly compliant with minor issues
- 5 = Fully compliant

Respond ONLY with valid JSON in exactly this format, nothing else:
{{"policy_compliance": <1-5>, "task_completion": <1-5>, "slot_accuracy": <1-5>, "communication_quality": <1-5>, "overall": <1-5>, "violations_detected": ["list any specific violations you noticed"], "reasoning": "brief explanation of your scores"}}"""


# ═══════════════════════════════════════════════════════════
# CONVERSATION FORMATTING
# ═══════════════════════════════════════════════════════════

def format_conversation(turns):
    """Format turns into readable conversation text."""
    lines = []
    for turn in turns:
        speaker = turn["speaker"]
        text = turn["text"]
        if speaker == "action":
            lines.append(f"[SYSTEM] {text}")
        elif speaker == "agent":
            lines.append(f"[Agent] {text}")
        elif speaker == "customer":
            lines.append(f"[Customer] {text}")
    return "\n".join(lines)


def format_required_actions(actions):
    """Format the required action list for the prompt."""
    if not actions:
        return "No specific action sequence required."
    lines = []
    for i, action in enumerate(actions, 1):
        readable = action.replace("-", " ").title()
        lines.append(f"{i}. {readable}")
    return "\n".join(lines)


def format_action_details(policy_spec):
    """Format action slot requirements for the prompt."""
    if not policy_spec:
        return "No detailed action specifications available."
    
    action_slots = policy_spec.get("action_slots", {})
    if not action_slots:
        return "No detailed action specifications available."
    
    lines = []
    for action, slots in action_slots.items():
        readable_action = action.replace("-", " ").title()
        if slots:
            readable_slots = ", ".join(s.replace("_", " ") for s in slots)
            lines.append(f"- {readable_action}: requires {readable_slots}")
        else:
            lines.append(f"- {readable_action}: no specific slot values required")
    return "\n".join(lines)


def build_eval_prompt(turns, flow, subflow, required_actions, policy_spec):
    """Build the full evaluation prompt for a conversation."""
    return EVAL_PROMPT_TEMPLATE.format(
        flow=flow.replace("_", " ").title(),
        subflow=subflow.replace("_", " ").title(),
        required_actions=format_required_actions(required_actions),
        action_details=format_action_details(policy_spec),
        conversation=format_conversation(turns),
    )


# ═══════════════════════════════════════════════════════════
# API CALLING
# ═══════════════════════════════════════════════════════════

def call_evaluator(model_config, prompt, max_retries=3, timeout=60):
    """
    Send evaluation prompt to an LLM endpoint.
    Returns parsed JSON scores or error dict.
    """
    headers = {
        "Content-Type": "application/json",
    }
    
    # Add auth header if API key is provided
    api_key = model_config["api_key"]
    if api_key and api_key != "your-api-key":
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model_config["model_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,  # Deterministic scoring
        "max_tokens": 500,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                model_config["endpoint"],
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from response — handle common API formats
            text = None
            
            # OpenAI-compatible format
            if "choices" in result:
                text = result["choices"][0]["message"]["content"]
            # Simple format
            elif "response" in result:
                text = result["response"]
            elif "output" in result:
                text = result["output"]
            elif "text" in result:
                text = result["text"]
            # Fallback
            else:
                text = json.dumps(result)
            
            # Parse JSON from response
            scores = parse_scores(text)
            scores["raw_response"] = text
            scores["model"] = model_config["name"]
            scores["attempt"] = attempt + 1
            return scores
            
        except requests.exceptions.Timeout:
            print(f"    Timeout (attempt {attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"    Request error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    
    return {
        "error": "max_retries_exceeded",
        "model": model_config["name"],
        "policy_compliance": None,
        "task_completion": None,
        "slot_accuracy": None,
        "communication_quality": None,
        "overall": None,
        "violations_detected": [],
        "reasoning": "API call failed after max retries",
    }


def parse_scores(text):
    """Extract JSON scores from LLM response text."""
    # Try direct JSON parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block in text
    json_patterns = [
        r'\{[^{}]*"policy_compliance"[^{}]*\}',
        r'\{[^{}]*"overall"[^{}]*\}',
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue
    
    # Try to extract individual scores with regex
    scores = {
        "policy_compliance": None,
        "task_completion": None,
        "slot_accuracy": None,
        "communication_quality": None,
        "overall": None,
        "violations_detected": [],
        "reasoning": "Could not parse structured response",
        "parse_error": True,
    }
    
    for field in ["policy_compliance", "task_completion", "slot_accuracy", 
                  "communication_quality", "overall"]:
        match = re.search(rf'"{field}"\s*:\s*(\d+)', text)
        if match:
            scores[field] = int(match.group(1))
    
    return scores


# ═══════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════

def run_experiment(
    perturbations,
    policy_lookup,
    models_to_run=None,
    max_conversations=None,
    resume_from=None,
):
    """
    Run the full evaluation experiment.
    
    For each conversation, evaluates:
    1. Original (baseline)
    2. Family A perturbation (surface noise)
    3. Family D perturbation (real violation)
    
    With each specified model.
    """
    if models_to_run is None:
        models_to_run = list(MODELS.keys())
    
    policies = policy_lookup.get("policies", {})
    
    results = []
    results_path = RESULTS_DIR / "evaluator_results.json"
    
    # Resume support — load existing results
    if resume_from and Path(resume_from).exists():
        with open(resume_from) as f:
            results = json.load(f)
        completed_ids = {r["convo_id"] for r in results}
        print(f"Resuming: {len(completed_ids)} conversations already done")
    else:
        completed_ids = set()
    
    # Select conversations to process
    to_process = perturbations
    if max_conversations:
        to_process = to_process[:max_conversations]
    
    total = len(to_process)
    total_api_calls = 0
    errors = 0
    
    print(f"\nRunning evaluation experiment")
    print(f"  Conversations: {total}")
    print(f"  Models: {models_to_run}")
    print(f"  API calls needed: ~{total * 3 * len(models_to_run)}")
    print(f"  Estimated time: ~{total * 3 * len(models_to_run) * 5 / 60:.0f} minutes")
    print()
    
    for idx, perturb in enumerate(tqdm(to_process, desc="Evaluating")):
        convo_id = perturb["convo_id"]
        
        if convo_id in completed_ids:
            continue
        
        flow = perturb["flow"]
        subflow = perturb["subflow"]
        policy_key = f"{flow}::{subflow}"
        policy_spec = policies.get(policy_key, {})
        required_actions = perturb["required_actions"]
        
        # Prepare the three versions
        versions = {}
        
        # Original
        versions["original"] = perturb["original_turns"]
        
        # Family A — use combined surface if available, else filler
        if perturb["family_a"]:
            # Prefer combined_surface over filler_insertion
            family_a = None
            for a in perturb["family_a"]:
                if a["type"] == "combined_surface":
                    family_a = a
                    break
            if not family_a:
                family_a = perturb["family_a"][0]
            versions["family_a"] = family_a["perturbed_turns"]
        
        # Family D — slot swap
        if perturb["family_d"]:
            versions["family_d"] = perturb["family_d"][0]["perturbed_turns"]
        
        # Run each version through each model
        convo_result = {
            "convo_id": convo_id,
            "flow": flow,
            "subflow": subflow,
            "policy_key": policy_key,
            "required_actions": required_actions,
            "num_turns": perturb["num_turns"],
            "family_d_details": {
                "type": perturb["family_d"][0]["type"] if perturb["family_d"] else None,
                "slot_swapped": perturb["family_d"][0].get("slot_swapped") if perturb["family_d"] else None,
                "original_value": perturb["family_d"][0].get("original_value") if perturb["family_d"] else None,
                "replacement_value": perturb["family_d"][0].get("replacement_value") if perturb["family_d"] else None,
            },
            "scores": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        for model_key in models_to_run:
            model_config = MODELS[model_key]
            model_scores = {}
            
            for version_name, version_turns in versions.items():
                prompt = build_eval_prompt(
                    turns=version_turns,
                    flow=flow,
                    subflow=subflow,
                    required_actions=required_actions,
                    policy_spec=policy_spec,
                )
                
                scores = call_evaluator(model_config, prompt)
                model_scores[version_name] = scores
                total_api_calls += 1
                
                if scores.get("error"):
                    errors += 1
                
                # Rate limiting — adjust based on your cluster's limits
                time.sleep(0.5)
            
            # Compute CSS, PSS, SRG for this conversation + model
            orig_overall = model_scores.get("original", {}).get("overall")
            fam_a_overall = model_scores.get("family_a", {}).get("overall")
            fam_d_overall = model_scores.get("family_d", {}).get("overall")
            
            css = None
            pss = None
            srg = None
            
            if orig_overall is not None and fam_a_overall is not None:
                css = abs(orig_overall - fam_a_overall)
            if orig_overall is not None and fam_d_overall is not None:
                pss = orig_overall - fam_d_overall
            if css is not None and pss is not None:
                srg = pss - css
            
            model_scores["css"] = css
            model_scores["pss"] = pss
            model_scores["srg"] = srg
            
            convo_result["scores"][model_key] = model_scores
        
        results.append(convo_result)
        
        # Save checkpoint every 25 conversations
        if (idx + 1) % 25 == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Checkpoint saved: {len(results)} conversations, {total_api_calls} API calls, {errors} errors")
    
    # Final save
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Conversations evaluated: {len(results)}")
    print(f"  Total API calls: {total_api_calls}")
    print(f"  Errors: {errors}")
    print(f"  Saved to: {results_path}")
    
    return results


# ═══════════════════════════════════════════════════════════
# POSITION BIAS EXPERIMENT
# ═══════════════════════════════════════════════════════════

def run_position_experiment(
    perturbations,
    policy_lookup,
    models_to_run=None,
):
    """
    Run the turn-position bias experiment.
    Only on conversations that have position variants.
    """
    if models_to_run is None:
        models_to_run = list(MODELS.keys())
    
    policies = policy_lookup.get("policies", {})
    
    # Filter to conversations with position variants
    position_convos = [p for p in perturbations if p.get("position_variants")]
    print(f"\nPosition bias experiment: {len(position_convos)} conversations with variants")
    
    if not position_convos:
        print("No position variants available. Skipping.")
        return []
    
    results = []
    
    for perturb in tqdm(position_convos, desc="Position bias"):
        convo_id = perturb["convo_id"]
        flow = perturb["flow"]
        subflow = perturb["subflow"]
        policy_key = f"{flow}::{subflow}"
        policy_spec = policies.get(policy_key, {})
        required_actions = perturb["required_actions"]
        
        position_result = {
            "convo_id": convo_id,
            "flow": flow,
            "subflow": subflow,
            "scores": {},
        }
        
        # Get original score first
        for model_key in models_to_run:
            model_config = MODELS[model_key]
            model_scores = {"original": None}
            
            # Score original
            prompt = build_eval_prompt(
                turns=perturb["original_turns"],
                flow=flow,
                subflow=subflow,
                required_actions=required_actions,
                policy_spec=policy_spec,
            )
            orig_scores = call_evaluator(model_config, prompt)
            model_scores["original"] = orig_scores
            time.sleep(0.5)
            
            # Score each position variant
            for position, variant in perturb["position_variants"].items():
                prompt = build_eval_prompt(
                    turns=variant["perturbed_turns"],
                    flow=flow,
                    subflow=subflow,
                    required_actions=required_actions,
                    policy_spec=policy_spec,
                )
                pos_scores = call_evaluator(model_config, prompt)
                model_scores[position] = pos_scores
                time.sleep(0.5)
            
            # Compute position-specific PSS
            orig_overall = orig_scores.get("overall")
            for position in ["early", "mid", "late"]:
                if position in model_scores and model_scores[position]:
                    pos_overall = model_scores[position].get("overall")
                    if orig_overall is not None and pos_overall is not None:
                        model_scores[f"pss_{position}"] = orig_overall - pos_overall
            
            position_result["scores"][model_key] = model_scores
        
        results.append(position_result)
    
    # Save
    out_path = RESULTS_DIR / "position_bias_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved position bias results to {out_path}")
    
    return results


# ═══════════════════════════════════════════════════════════
# FAILURE ATTRIBUTION EXPERIMENT
# ═══════════════════════════════════════════════════════════

ATTRIBUTION_PROMPTS = {
    "mode_a_explicit": """You previously evaluated a customer service conversation and gave it an overall score of {original_score}/5.

IMPORTANT: In this conversation, there is a specific violation: {violation_description}

Please re-evaluate the conversation with this information in mind.

{conversation}

Respond ONLY with valid JSON:
{{"overall": <1-5>, "reasoning": "brief explanation", "violation_severity": "critical/major/minor/none"}}""",

    "mode_b_list_first": """Evaluate this customer service conversation. 

BEFORE scoring, you MUST list every policy violation you can find in the conversation.

## Company Policy:
Required actions: {required_actions}

## Conversation:
{conversation}

Respond ONLY with valid JSON:
{{"violations_found": ["list every violation"], "overall": <1-5>, "reasoning": "explanation"}}""",
}


def run_failure_attribution(
    main_results,
    perturbations,
    policy_lookup,
    models_to_run=None,
    max_failures=50,
):
    """
    Run failure attribution on cases where evaluators missed Family D violations.
    A "failure" is when Family D score is within 0.5 of original score.
    """
    if models_to_run is None:
        models_to_run = list(MODELS.keys())
    
    policies = policy_lookup.get("policies", {})
    perturbation_lookup = {p["convo_id"]: p for p in perturbations}
    
    # Find failures
    failures = []
    for result in main_results:
        for model_key in models_to_run:
            model_scores = result["scores"].get(model_key, {})
            srg = model_scores.get("srg")
            pss = model_scores.get("pss")
            
            # Failure = evaluator didn't detect the violation (PSS < 0.5)
            if pss is not None and pss < 0.5:
                failures.append({
                    "convo_id": result["convo_id"],
                    "model": model_key,
                    "original_score": model_scores.get("original", {}).get("overall"),
                    "family_d_score": model_scores.get("family_d", {}).get("overall"),
                    "pss": pss,
                    "srg": srg,
                    "flow": result["flow"],
                    "subflow": result["subflow"],
                    "family_d_details": result.get("family_d_details", {}),
                })
    
    print(f"\nFound {len(failures)} evaluator failures (PSS < 0.5)")
    
    # Sample up to max_failures
    random.seed(RANDOM_SEED)
    if len(failures) > max_failures:
        failures = random.sample(failures, max_failures)
    
    print(f"Analyzing {len(failures)} failures")
    
    attribution_results = []
    
    for failure in tqdm(failures, desc="Attribution"):
        convo_id = failure["convo_id"]
        model_key = failure["model"]
        model_config = MODELS[model_key]
        
        perturb = perturbation_lookup.get(convo_id)
        if not perturb or not perturb["family_d"]:
            continue
        
        family_d = perturb["family_d"][0]
        family_d_turns = family_d["perturbed_turns"]
        
        flow = failure["flow"]
        subflow = failure["subflow"]
        
        # Build violation description
        violation_desc = (
            f"The agent used the wrong {family_d.get('slot_swapped', 'slot value')}. "
            f"The correct value was '{family_d.get('original_value')}' but the conversation "
            f"shows '{family_d.get('replacement_value')}'."
        )
        
        result = {
            "convo_id": convo_id,
            "model": model_key,
            "original_score": failure["original_score"],
            "family_d_score": failure["family_d_score"],
            "pss": failure["pss"],
            "violation_description": violation_desc,
            "attribution": {},
        }
        
        # ── Mode A test: tell it about the violation ──
        mode_a_prompt = ATTRIBUTION_PROMPTS["mode_a_explicit"].format(
            original_score=failure["family_d_score"],
            violation_description=violation_desc,
            conversation=format_conversation(family_d_turns),
        )
        
        mode_a_scores = call_evaluator(model_config, mode_a_prompt)
        result["attribution"]["mode_a"] = mode_a_scores
        time.sleep(0.5)
        
        # ── Mode B test: list violations first ──
        mode_b_prompt = ATTRIBUTION_PROMPTS["mode_b_list_first"].format(
            required_actions=format_required_actions(perturb["required_actions"]),
            conversation=format_conversation(family_d_turns),
        )
        
        mode_b_scores = call_evaluator(model_config, mode_b_prompt)
        result["attribution"]["mode_b"] = mode_b_scores
        time.sleep(0.5)
        
        # ── Classify failure mode ──
        mode_a_new_score = mode_a_scores.get("overall")
        mode_b_new_score = mode_b_scores.get("overall")
        mode_b_violations = mode_b_scores.get("violations_found", [])
        
        original_d_score = failure["family_d_score"]
        
        if mode_a_new_score is not None and original_d_score is not None:
            score_drop_after_told = original_d_score - mode_a_new_score
            
            if score_drop_after_told >= 1.0:
                result["classified_mode"] = "A"  # Blind — only caught it when told
                result["mode_description"] = "Evaluator was blind to the violation until explicitly informed"
            elif mode_b_violations and any(
                family_d.get("slot_swapped", "").replace(".", " ") in str(v).lower() or
                family_d.get("replacement_value", "").lower() in str(v).lower() or
                "wrong" in str(v).lower() or "incorrect" in str(v).lower()
                for v in mode_b_violations
            ):
                result["classified_mode"] = "B"  # Detected but underweighted
                result["mode_description"] = "Evaluator detected the violation but did not weight it appropriately"
            else:
                result["classified_mode"] = "C"  # Surface distraction or other
                result["mode_description"] = "Evaluator failed to detect violation even with structured prompting"
        else:
            result["classified_mode"] = "unknown"
            result["mode_description"] = "Could not classify — missing score data"
        
        attribution_results.append(result)
    
    # Save
    out_path = RESULTS_DIR / "failure_attribution_results.json"
    with open(out_path, "w") as f:
        json.dump(attribution_results, f, indent=2)
    
    # Summary
    mode_counts = {}
    for r in attribution_results:
        mode = r.get("classified_mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    print(f"\n{'='*60}")
    print("FAILURE ATTRIBUTION RESULTS")
    print(f"{'='*60}")
    print(f"  Total failures analyzed: {len(attribution_results)}")
    for mode, count in sorted(mode_counts.items()):
        pct = 100 * count / len(attribution_results) if attribution_results else 0
        label = {
            "A": "Mode A (blind to violation)",
            "B": "Mode B (detected but underweighted)",
            "C": "Mode C (surface distraction / other)",
        }.get(mode, mode)
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\nSaved to {out_path}")
    return attribution_results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  STEP 3: EVALUATOR RUNNER")
    print("=" * 60 + "\n")
    
    # Load data
    with open(EXPERIMENT_DIR / "perturbations.json") as f:
        perturbations = json.load(f)
    print(f"Loaded {len(perturbations)} perturbation sets")
    
    with open(PROJECT_ROOT / "data" / "policy_lookup.json") as f:
        policy_lookup = json.load(f)
    print(f"Loaded policy lookup with {policy_lookup['policy_count']} policies")
    
    # ── Verify API connectivity ──
    print(f"\nVerifying model endpoints...")
    active_models = []
    
    for model_key, model_config in MODELS.items():
        endpoint = model_config["endpoint"]
        if "your-cluster" in endpoint or "your-api-key" in model_config["api_key"]:
            print(f"  ✗ {model_config['name']}: endpoint not configured")
            continue
        
        # Quick test call
        test_scores = call_evaluator(model_config, "Respond with: {\"test\": true}")
        if test_scores.get("error"):
            print(f"  ✗ {model_config['name']}: connection failed")
        else:
            print(f"  ✓ {model_config['name']}: connected")
            active_models.append(model_key)
    
    if not active_models:
        print("\n⚠ No models connected. Configure endpoints in .env file or in the MODELS dict above.")
        print("\nExpected .env format:")
        print("  LLAMA_70B_ENDPOINT=https://your-cluster/v1/chat/completions")
        print("  LLAMA_70B_API_KEY=your-key")
        print("  LLAMA_70B_MODEL_ID=llama-3.3-70b")
        print("\nOr edit the MODELS dict at the top of this script.")
        
        # Run a dry run to verify prompt formatting
        print(f"\n{'='*60}")
        print("DRY RUN — showing sample prompts")
        print(f"{'='*60}")
        
        sample = perturbations[0]
        policy_key = f"{sample['flow']}::{sample['subflow']}"
        policy_spec = policy_lookup.get("policies", {}).get(policy_key, {})
        
        print(f"\nConversation: {sample['convo_id']} ({policy_key})")
        
        prompt = build_eval_prompt(
            turns=sample["original_turns"][:10],
            flow=sample["flow"],
            subflow=sample["subflow"],
            required_actions=sample["required_actions"],
            policy_spec=policy_spec,
        )
        
        print(f"\n── Sample evaluation prompt (first 2000 chars) ──")
        print(prompt[:2000])
        print(f"\n... ({len(prompt)} total chars)")
        
        return
    
    # ── Run main experiment ──
    print(f"\nRunning with models: {active_models}")
    
    # Optional: start with a pilot of 10 conversations
    pilot_mode = input("\nRun pilot (10 conversations) first? [y/n]: ").strip().lower()
    
    if pilot_mode == "y":
        pilot_results = run_experiment(
            perturbations,
            policy_lookup,
            models_to_run=active_models,
            max_conversations=10,
        )
        
        # Show pilot results
        print(f"\n{'='*60}")
        print("PILOT RESULTS")
        print(f"{'='*60}")
        
        for model_key in active_models:
            css_vals = []
            pss_vals = []
            srg_vals = []
            
            for r in pilot_results:
                m = r["scores"].get(model_key, {})
                if m.get("css") is not None:
                    css_vals.append(m["css"])
                if m.get("pss") is not None:
                    pss_vals.append(m["pss"])
                if m.get("srg") is not None:
                    srg_vals.append(m["srg"])
            
            print(f"\n  {MODELS[model_key]['name']}:")
            if css_vals:
                print(f"    CSS (cosmetic sensitivity): {sum(css_vals)/len(css_vals):.2f}")
            if pss_vals:
                print(f"    PSS (policy sensitivity):   {sum(pss_vals)/len(pss_vals):.2f}")
            if srg_vals:
                print(f"    SRG (robustness gap):       {sum(srg_vals)/len(srg_vals):.2f}")
                neg = sum(1 for s in srg_vals if s < 0)
                print(f"    SRG negative: {neg}/{len(srg_vals)}")
        
        proceed = input("\nProceed with full experiment? [y/n]: ").strip().lower()
        if proceed != "y":
            print("Stopped after pilot.")
            return
    
    # ── Full experiment ──
    main_results = run_experiment(
        perturbations,
        policy_lookup,
        models_to_run=active_models,
    )
    
    # ── Position bias experiment ──
    print(f"\n{'='*60}")
    print("RUNNING POSITION BIAS EXPERIMENT")
    print(f"{'='*60}")
    
    position_results = run_position_experiment(
        perturbations,
        policy_lookup,
        models_to_run=active_models,
    )
    
    # ── Failure attribution ──
    print(f"\n{'='*60}")
    print("RUNNING FAILURE ATTRIBUTION")
    print(f"{'='*60}")
    
    attribution_results = run_failure_attribution(
        main_results,
        perturbations,
        policy_lookup,
        models_to_run=active_models,
    )
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Files:")
    print(f"    evaluator_results.json — main SRG experiment")
    print(f"    position_bias_results.json — turn-position analysis")
    print(f"    failure_attribution_results.json — Mode A/B/C classification")
    print(f"\n  Next step: python 04_analysis.py")


if __name__ == "__main__":
    main()
