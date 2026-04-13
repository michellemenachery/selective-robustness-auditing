"""
Configuration for preliminary viability experiments.
Update ABCD_DIR to point to your local ABCD dataset directory.
"""
import os

# ─── PATHS ───
# Update this to your actual ABCD data directory
ABCD_DIR = os.path.expanduser("~/research/selective-robustness-auditing/data")

# ABCD data files
ABCD_DATA = os.path.join(ABCD_DIR, "abcd_v1.1.json")
KB_FILE = os.path.join(ABCD_DIR, "kb.json")
GUIDELINES_FILE = os.path.join(ABCD_DIR, "guidelines.json")
ONTOLOGY_FILE = os.path.join(ABCD_DIR, "ontology.json")

# If you already have a policy_lookup.json from your pipeline
POLICY_LOOKUP = os.path.join(ABCD_DIR, "policy_lookup.json")

# Output directory for preliminary experiments
OUTPUT_DIR = os.path.expanduser("~/research/selective-robustness-auditing/preliminary_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── SAMPLING ───
VIABILITY_SAMPLE_SIZE = 30          # Small sample for viability checks
MIN_TURNS = 12                      # Minimum dialogue turns
PERTURBATIONS_PER_FAMILY = 10       # Per family for viability
RANDOM_SEED = 42

# ─── JUDGE MODELS ───
# Update these with your actual API endpoints / model identifiers
JUDGE_MODELS = {
    "llama-3.3-70b": {
        "endpoint": "YOUR_LLAMA_ENDPOINT",
        "model_id": "llama-3.3-70b-instruct",
        "type": "api",  # "api" or "local"
    },
    "gpt-oss-120b": {
        "endpoint": "YOUR_GPT_OSS_120B_ENDPOINT",
        "model_id": "gpt-oss-120b",
        "type": "api",
    },
    "gpt-oss-20b": {
        "endpoint": "YOUR_GPT_OSS_20B_ENDPOINT",
        "model_id": "gpt-oss-20b",
        "type": "api",
    },
}

# ─── EVALUATION SETTINGS ───
POINTWISE_DIMENSIONS = [
    "workflow_correctness",
    "value_accuracy",
    "rule_compliance",
    "overall_quality",
]
LIKERT_SCALE = (1, 5)

PROMPT_CONDITIONS = ["no_policy", "compressed_policy", "full_policy"]

# ─── ENUMERABLE SLOT VALUES (from ABCD ontology) ───
# These are used for deterministic value perturbations
MEMBERSHIP_LEVELS = ["gold", "silver", "bronze", "guest"]
PAYMENT_METHODS = ["credit card", "debit card", "gift card"]
SHIPPING_STATUSES = ["in transit", "delivered", "processing", "shipped", "returned"]
ORDER_STATUSES = ["active", "cancelled", "completed", "pending"]
