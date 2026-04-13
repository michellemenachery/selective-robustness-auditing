# Preliminary Viability Experiment Plan

## What This Document Covers

Before committing 3 months to the full experiment, you need to answer 7 questions in roughly 1 week of work. Each question has a specific check, a pass/fail criterion, and instructions for what to do if it fails.

---

## Phase 1: Can We Build the Pipeline? (Days 1-2, No API Needed)

### V0: Data Loading

**Question:** Can we load all ABCD files and extract what we need?

**What to run:**
```bash
cd preliminary_experiments
python run_viability.py --check data
```

**What it checks:**
- abcd_v1.1.json (or .gz) loads and parses into conversations
- kb.json loads and contains subflow → action sequence mappings
- guidelines.json loads and contains policy text per subflow
- ontology.json loads with slot-value definitions
- At least 30 conversations are policy-compliant (action sequence matches kb.json) with ≥12 turns

**Pass criterion:** All files load. ≥ 30 compliant conversations found.

**If it fails:**
- Missing files → Check your ABCD_DIR path in config.py
- Zero compliant conversations → The action sequence matching logic may not align with your ABCD version. Print the first conversation's extracted actions vs. kb.json expected actions and debug the parsing.
- Parsing errors → The ABCD data format may use a different JSON structure than expected. Check whether your version uses `delexed` or `original` keys, dict vs. list format.

**What success tells you:** The data infrastructure works. You can extract conversations, action sequences, slot values, and policy text programmatically.

---

### V1: Perturbation Generation

**Question:** Can we generate perturbations of each type from ABCD's structural annotations?

**What to run:**
```bash
python run_viability.py --check perturb
```

**What it checks for each family:**

| Family | Perturbation Type | What the Code Does | Ground Truth Check |
|--------|------------------|--------------------|--------------------|
| W | Step deletion | Removes one action from kb.json sequence + deletes corresponding dialogue turns | perturbed_seq ≠ kb.json |
| W | Step reordering | Swaps two adjacent actions in the sequence | perturbed_seq ≠ kb.json |
| W | Subflow substitution | Replaces action sequence with one from a different subflow | wrong subflow applied |
| V | Enumerable slot swap | Replaces membership/payment/status with different valid value from ontology | new_value ≠ scenario AND new_value ∈ ontology |
| V | Entity swap | Replaces account_id/order_id/phone with value from different conversation | new_value ≠ scenario |
| R | Authorization bypass | Deletes verify-identity before restricted action | verify-identity absent before update-account |
| R | Eligibility violation | Downgrades membership tier in dialogue text | lower_tier referenced but higher_tier service provided |
| N | Length-matched control | Deletes non-action turns (pleasantries) to match step-deletion length | All kb.json actions still present |

**Pass criterion:** At least 5 perturbations per family. At least 50% are deterministic (no LLM involved).

**If it fails:**
- Family W fails → The action sequence extraction from dialogue turns isn't working. Print the turn structure and check how actions are represented.
- Family V fails → No enumerable slot values found in conversations. Check whether membership_level, payment_method etc. appear in scenario metadata.
- Family R fails → No conversations have verify-identity in their sequence. Check kb.json for which subflows use verify-identity.
- Low deterministic ratio → More perturbation types need LLM rewriting than expected. This is okay but increases validation burden.

**What success tells you:** You can programmatically construct violations of each type. The perturbation generation is the core engineering deliverable — if this works, the rest is evaluation and analysis.

---

### V2: Perturbation Quality (Automated)

**Question:** Are the generated perturbations well-formed?

**Automated checks (run with V1):**
- Every perturbation has a non-empty description
- Every perturbation has a ground truth check
- Family W perturbed sequences actually differ from kb.json
- Family V swapped values actually differ from scenario ground truth
- Overlap tags are applied where perturbations cross family boundaries

**Manual checks (do yourself after V1):**
1. Open `preliminary_results/viability_perturbations.json`
2. For each perturbation, ask:
   - Does the description accurately describe what changed?
   - Would the perturbed dialogue read naturally to someone unfamiliar with the perturbation?
   - Could a reader detect this violation without being told what to look for?
   - Is the family label correct?
3. Rate each on naturalness: 1 (obviously broken) to 5 (indistinguishable from real dialogue)
4. Target: Mean naturalness > 3.0

**If manual check fails:**
- Naturalness < 3 for step deletions → The surrounding dialogue turns need smoothing. You may need to manually rewrite 1-2 transition sentences.
- Naturalness < 3 for value swaps → The replaced value creates a grammatically awkward sentence. Check whether the swap is in the right position in the text.
- Wrong family labels → Review the overlap tag logic in perturbation_engine.py.

---

## Phase 2: Can the Judges Score? (Days 3-4, Requires API)

### V3: Score Variance

**Question:** When judges score original dialogues, do they produce enough variance in scores, or do they cluster everything at 4-5?

**What to run:**
```bash
python run_judge_viability.py --model llama-3.3-70b --n 10
```

**What it does:** Scores 10 original (un-perturbed) dialogues on all 4 dimensions. Computes the standard deviation across all scores.

**Pass criterion:** SD > 0.8 across all scores. Score range spans at least 2 points (e.g., some 3s, some 4s, some 5s).

**Expected output:**
```
Conversation X: W=4 V=5 R=4 O=4
Conversation Y: W=3 V=4 R=3 O=3
Conversation Z: W=5 V=4 R=5 O=5
SD = 0.82 → PASS
```

**If it fails:**
- All scores are 5/5 → The rubric is too vague. Make dimensions more specific (e.g., instead of "workflow correctness" say "did the agent execute every required step in the correct order?")
- All scores are the same → Try a different model or temperature > 0 (though 0 is preferred for determinism)
- All scores are 0 → API is not returning valid JSON. Check the response format.

**What success tells you:** The scoring rubric produces enough signal to distinguish between good and bad dialogues. Without variance in originals, you can't detect perturbation effects.

---

### V4: Prompt Sensitivity

**Question:** Do the three prompt conditions (no-policy, compressed, full) actually produce different judge behavior?

**What to run:** Runs automatically as part of `run_judge_viability.py`. Scores 5 pairs × 3 conditions = 30 evaluations.

**What it checks:** For the same perturbation pair, does the judge detect the violation under one prompt condition but miss it under another?

**Pass criterion:** At least 20% of pairs show different detection outcomes across no-policy and full-policy conditions.

**If it fails (< 20% sensitivity):**
- The prompt conditions are not differentiated enough. Either:
  - The no-policy prompt already gives too much context (trim it)
  - The full-policy prompt doesn't add enough useful information (add more specific policy rules)
  - The perturbations are too easy or too hard regardless of prompt (try different perturbation types)

**What success tells you:** The prompt conditioning experiment will produce analyzable results. If all three conditions give identical results, RQ2 is dead.

---

### V5: Position Bias

**Question:** In pairwise evaluation, does the judge always prefer whatever is in Position A regardless of content?

**What to run:** Runs automatically. Scores 5 pairs with original first, then same 5 pairs with perturbed first.

**Pass criterion:** Preference rate should not differ by more than 15 percentage points across orderings.

**If it fails:**
- Always prefers Position A → Strong position bias. Mitigation: run every pairwise evaluation twice (both orderings) and average. This doubles pairwise compute cost but eliminates bias.
- Always prefers Position B → Same mitigation.

---

## Phase 3: Do the Metrics Work? (Day 5, No API Needed)

### V6: Metric Computation

**Question:** Do all 7 metrics compute correctly on synthetic data?

**What to run:**
```bash
python run_viability.py --check metrics
```

**What it does:** Creates fake judge outputs with known detection rates (W=80%, V=50%, R=65%, N=5%), runs all metric functions, and checks that the computed values approximately match the known rates.

**Pass criterion:** All metrics compute without errors. Computed rates are within ±10% of known rates.

**If it fails:** Bug in the metric code. The error traceback will tell you which function failed.

---

## Decision Matrix After All Checks

| All V0-V6 pass | → Proceed to full experiment |
|----------------|------------------------------|
| V0 fails | → Fix data loading before anything else |
| V1 fails | → Fix perturbation engine; may need to adapt to your ABCD version |
| V3 fails | → Revise scoring rubric; the experiment design is sound but the prompt needs work |
| V4 fails | → Redesign prompt conditions; if they can't be differentiated, drop RQ2 |
| V5 fails | → Add both-ordering averaging; increases compute but doesn't block the project |
| V6 fails | → Fix metric code; should be straightforward |

---

## What "Viability Confirmed" Means

If all checks pass, you know:

1. **Data pipeline works** — ABCD loads, parses, and provides the structured annotations you need
2. **Perturbation generation works** — You can create controlled violations of each type with deterministic ground truth
3. **Judges produce usable scores** — Enough variance to detect effects, different prompt conditions produce different behavior
4. **Metrics compute correctly** — The analysis code is ready
5. **Position bias is manageable** — Pairwise evaluation is valid (possibly with both-ordering averaging)

You can then scale to the full experiment with confidence that the infrastructure works and the results will be interpretable.

---

## Estimated Effort

| Phase | Days | API Calls | Human Time |
|-------|------|-----------|------------|
| Phase 1 (V0-V2) | 1-2 | 0 | 2-4 hours coding + debugging |
| Phase 2 (V3-V5) | 1-2 | ~80 | 1-2 hours + wait for API |
| Phase 3 (V6) | 0.5 | 0 | 30 minutes |
| Manual review | 1 | 0 | 2-3 hours reading perturbations |
| **Total** | **3-5 days** | **~80** | **~8 hours active work** |
