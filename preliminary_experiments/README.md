# Preliminary Viability Experiments
## Auditing LLM Judges in Policy-Constrained Task-Oriented Dialogue

This codebase runs preliminary checks to verify that the full experiment is feasible before committing compute and time.

## Setup

### 1. Prerequisites
```bash
pip install numpy --break-system-packages  # Only additional dependency
```

### 2. Configure Paths
Edit `config.py`:
- Set `ABCD_DIR` to your ABCD data directory (where `abcd_v1.1.json`, `kb.json`, `guidelines.json`, `ontology.json` live)
- Set `OUTPUT_DIR` for results

### 3. Configure Judge API
Edit `judge_runner.py`:
- Replace the `call_judge_api()` function body with your actual API call
- See the comments in the function for examples (OpenAI-compatible, requests-based, local HuggingFace)

## Running the Experiments

### Step 1: Data and Perturbation Checks (No API needed)
```bash
python run_viability.py
```
This runs V0 (data loading), V1 (perturbation generation), V2 (perturbation quality), and V6 (metric computation on synthetic data). Takes ~30 seconds. No LLM API calls.

**What it checks:**
- Can we load all ABCD files?
- Can we find policy-compliant conversations with enough turns?
- Can we generate perturbations of each type?
- Do perturbation ground-truth checks work?
- Do all metric functions compute correctly?

**Pass criteria:**
- All ABCD files load without errors
- At least 30 compliant conversations found
- At least 5 perturbations per family generated
- All metrics compute without errors

### Step 2: Judge Checks (Requires API)
```bash
python run_judge_viability.py
python run_judge_viability.py --model llama-3.3-70b --n 10
```
This runs V3 (score variance), V4 (prompt sensitivity), V5 (position bias). Makes ~60-80 LLM API calls.

**What it checks:**
- Do judge scores have enough variance (SD > 0.8)?
- Do different prompt conditions produce different detection outcomes?
- Is there position bias in pairwise evaluation?

**Pass criteria:**
- Score SD > 0.8 across dimensions
- At least 20% of pairs show different outcomes across prompt conditions
- Position preference gap < 15%

### Step 3: Manual Review
After Steps 1-2 pass, manually review:
1. Open `preliminary_results/viability_perturbations.json`
2. For each perturbation, verify:
   - Does it read naturally?
   - Is the family label correct?
   - Is the described violation actually present?
   - Could a judge detect this from surface cues alone (length, awkwardness)?

## File Structure

```
preliminary_experiments/
├── config.py                    # All paths, settings, model configs
├── run_viability.py             # Main runner (V0, V1, V2, V6)
├── run_judge_viability.py       # Judge checks (V3, V4, V5)
├── judge_runner.py              # LLM API interface (EDIT THIS)
├── utils/
│   ├── __init__.py
│   ├── abcd_loader.py           # ABCD data loading and parsing
│   ├── perturbation_engine.py   # Perturbation generation
│   ├── prompt_templates.py      # 3 prompt conditions + pairwise
│   └── metrics.py               # All 7 metrics + viability checks
└── README.md
```

## Viability Checks Summary

| Check | What It Tests | API Needed? | Pass Criterion |
|-------|--------------|-------------|----------------|
| V0 | Data loading and parsing | No | All files load, >= 30 compliant convos |
| V1 | Perturbation generation | No | >= 5 per family, >= 50% deterministic |
| V2 | Perturbation quality (auto) | No | All have descriptions and ground truth checks |
| V3 | Score variance | Yes | SD > 0.8 across dimensions |
| V4 | Prompt sensitivity | Yes | >= 20% different outcomes across conditions |
| V5 | Position bias (pairwise) | Yes | Gap < 15% |
| V6 | Metric computation | No | All metrics compute without errors |

## Decision Tree After Viability

```
V0 fails → Fix data paths / data format parsing
V1 fails → Fix perturbation engine for your ABCD version
V2 fails → Review and fix perturbation quality
V3 fails (low variance) → Revise scoring rubric (make dimensions more specific)
V3 fails (all zeros) → Fix API configuration
V4 fails → Differentiate prompt conditions more (add/remove policy detail)
V5 fails → Always run both orderings and average (mitigate bias)
V6 fails → Bug in metric code (should not happen on synthetic data)
```

## Next Steps After All Checks Pass

1. Scale perturbation generation to full set (150+ per family)
2. Run all 3 models × 3 prompt conditions on full set
3. Run human annotation on 100-150 samples
4. Compute all metrics and breakdowns
5. Write paper
