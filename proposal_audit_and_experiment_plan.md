# Process-Aware, Data-Blind: A Principled Audit of LLM Judges in Policy-Grounded Task-Oriented Dialogue

**Working title options:**
- *Process-Aware, Data-Blind: Auditing LLM Judges in Policy-Grounded Task-Oriented Dialogue*
- *What LLM Judges Miss in Multi-Turn Dialogue: A Perturbation-Based Audit of Value-Grounding and Uncertainty Structure*
- *Detecting, Diagnosing, and Explaining Judge Failures: A Signal-Detection and Uncertainty Framework for TOD Evaluator Auditing*

**Target venue:** EMNLP 2026 — Main Track is realistic with full execution; Findings is the honest conservative target.

**Primary dataset:** ABCD v1.1
**Validation dataset:** τ-bench (Sierra AI, 2024)
**Optional stretch:** doc2dial / MultiDoc2Dial for grounding-type generalization

---

## 1. Executive Summary

LLM-as-judge has become the default evaluator for task-oriented dialogue (TOD) in both research and industry. Deployed judges silently score tens of thousands of customer-service, financial, and healthcare conversations per day, and the industry assumption is that a sufficiently capable LLM, prompted with a rubric, reliably detects quality and compliance failures. This paper tests that assumption and finds it false in a specific, mechanistically interesting way.

We audit three LLM judges on ABCD — a customer-service TOD dataset with deterministic action sequences (`kb.json`), machine-readable business rules (`guidelines.json`), and typed slot-value annotations (`ontology.json`) — using four families of controlled perturbations: workflow (W), value (V), rule (R), and nuisance (N) baselines with matched artifact controls. We evaluate judges under three prompt-conditioning regimes and two scoring paradigms (pointwise and pairwise), and we measure judge behavior with a seven-metric suite grounded in signal detection theory, information theory, uncertainty quantification, and psychometrics.

Our central empirical finding is an asymmetry we call **data-blindness**: LLM judges reliably detect procedural and rule violations (high d′ for W and R) but fail to detect value-grounding violations (d′ near zero for V) at rates substantially below both procedural detection and a trivial deterministic baseline. This asymmetry (i) persists under explicit value-verification prompting, (ii) is invariant across model scale from 20B to 120B parameters, (iii) is structured along evidence distance (local → cross-turn → cascading) and shows no correlation with operational importance, and (iv) replicates on τ-bench, a structurally distinct modern tool-use TOD benchmark.

Our central methodological contribution is a two-dimensional failure-mode characterization combining **semantic entropy of judge rationales** (Kuhn et al., 2023) with **signal-detection sensitivity d′**. Violations clustering in the low-d′, low-SE quadrant represent *confident blindness* — the deployment-worst failure mode where sampling, self-consistency, and prompt engineering all fail. We show empirically that a large fraction of V-violations occupy this quadrant while W-violations do not, providing independent evidence for the data-blindness claim and giving deployers a principled tool for triaging judge risk in other domains.

Our prescriptive finding is that **decomposed evaluator architectures** — separating procedural scoring from value-grounded scoring with structured state supplied to the latter — substantially narrow the d′(V) gap. Decomposition beats prompt engineering, beats model scale, and beats supervised fine-tuning (which we include as an additional study). This converts the paper from a diagnostic audit into an actionable architectural recommendation.

---

## 2. Core Contribution and Novelty Assessment

### 2.1 What is the paper actually claiming?

The paper makes **four distinct, non-overlapping contributions**. Each should survive as an independent result if others weaken.

**Contribution 1 (empirical, primary):** LLM judges in policy-grounded TOD exhibit a structured asymmetry — they are *process-aware but data-blind*. Workflow and rule violations are detected at rates meaningfully above chance and above deterministic baselines; value/state-grounding violations are detected at rates statistically indistinguishable from chance and below trivial deterministic baselines. This asymmetry is characterized across four dimensions: evidence distance, outcome criticality, slot type, and model scale.

**Contribution 2 (methodological, secondary-but-independent):** A two-dimensional failure-mode characterization that combines semantic entropy of judge rationales with signal-detection-theory sensitivity. This framework is domain-general and provides deployers a principled tool for diagnosing whether judge failures are recoverable through sampling (high SE) or require architectural intervention (low SE).

**Contribution 3 (prescriptive, architecture-level):** Decomposed evaluator architectures — splitting procedural and value-grounded scoring, with the latter receiving explicit structured state — measurably close the V-gap where prompt engineering, model scale, and supervised fine-tuning do not. This is the single actionable recommendation for practitioners.

**Contribution 4 (artifact):** A perturbation-annotated evaluation benchmark built on ABCD with typed violation taxonomies, matched artifact controls, and validated human annotations, plus a companion evaluation harness implementing the seven-metric suite. Released for community use.

### 2.2 Novelty assessment, by contribution

I'll be direct about where each contribution sits relative to existing work.

**Contribution 1 — empirical V-blindness.** This is genuinely novel. The closest prior work:

- *FBI (Doddapaneni et al., EMNLP 2024)* — perturbation-based judge auditing in general NLG. Does not address TOD, does not have structured ground truth, does not disentangle procedural from value errors.
- *TD-EVAL (SIGDIAL 2025)* — uses LLM judges in TOD but does not audit them under controlled perturbation. Treats judges as measurement instruments, not as objects of study.
- *CompliBench* — compliance-violation detection in dialogue. Treats compliance as a monolithic category; our contribution is that "compliance" decomposes into procedural and value dimensions with mechanistically different failure profiles.
- *CAUSE (Abolghasemi et al., ACL 2024 Findings)* — counterfactual TOD auditing of *satisfaction classifiers*, not quality judges. Single perturbation type; accuracy-under-rebalancing framing.
- *Hallucination detection literature (FActScore, FEVER-style)* — concerned with factuality against external knowledge, not with session-local value grounding in multi-turn dialogue. The verification task is formally different (see §4.2).

The framing "we isolate value-grounding as a mechanistically distinct failure mode in judge evaluation" is not made by any of these papers.

**Contribution 2 — SE × d′ characterization.** This is the strongest novelty claim in the paper. Semantic entropy (Kuhn et al., 2023) has been applied to open-ended LLM question answering for hallucination detection. Signal-detection-theory d′ has been applied sporadically to LLM evaluation. *Nobody has combined these into a 2D failure-mode characterization for LLM-as-judge.* This contribution is independent of the compliance domain — it applies to any judge auditing task.

Cross-area novelty check: I have searched for combinations of (semantic entropy × item response theory) in NLP and found nothing. The combination is also original. If we pursue IRT item difficulty alongside SE, we have a second methodological contribution: *uncertainty-aware item difficulty estimation for LLM evaluation.*

**Contribution 3 — decomposed judge architecture.** Novel in this specific formulation. Prior work on evaluator decomposition has typically split evaluation along quality dimensions (fluency vs. coherence vs. relevance) rather than along the procedural-vs-grounded axis we propose. The decomposition we propose is mechanistically motivated rather than dimensionally motivated.

**Contribution 4 — benchmark artifact.** Modest novelty as a benchmark in the crowded TOD evaluation space, but valuable as a community resource. Not a headline contribution.

### 2.3 Is this EMNLP-acceptable?

Honest assessment:

**For Findings:** Yes, confidently. Any one of contributions 1, 2, or 3 — combined with the empirical rigor of the proposed methodology — clears the bar.

**For Main Track:** Plausible but not guaranteed. The path requires:

1. All four contributions landing (particularly contribution 2 producing a clean finding)
2. Cross-dataset replication on τ-bench showing the asymmetry is not ABCD-specific
3. Human validation coming in above 0.80 Gwet's AC1 on perturbation labels and judge rationale correctness
4. The decomposed-architecture result being large and reproducible across all three judges
5. Clean writing that resists collapse into "just another judge-auditing paper"

The single biggest failure mode for Main Track acceptance is a reviewer arguing that contribution 1 is essentially a TOD-flavored restatement of FBI. The strongest defenses are:

- The explicit-value-verification experiment showing V-blindness is not a prompting artifact
- The SE × d′ characterization being genuinely novel and producing a clear finding
- The decomposed-architecture result being independently publishable

If the explicit-value-verification experiment *closes* the V-gap, the paper pivots to a different but still strong Findings paper: "what information unlocks value-verification in LLM judges."

### 2.4 What's novel but underdeveloped right now?

Five items that would each substantially strengthen novelty if developed:

1. **Formal characterization of why holistic evaluation is mismatched to multi-turn value verification.** See §4.3 below — a plausibility argument from attention-budget allocation.
2. **Evidence-sufficiency formalization.** Currently the prompt-conditioning experiment reports detection rate at three information levels. Reframing as Value of Information gives quantitative measurement of how much each policy-context increment contributes. This is a minor but rigorous upgrade.
3. **Rationale semantic-cluster taxonomy.** When judges miss V-violations, what reasons do they give? If the semantic clusters are systematically shallow (e.g., "dialogue proceeded normally," "agent was professional"), that's a qualitative finding separate from the quantitative SE result.
4. **Cross-dataset transfer of failure-mode signatures.** If a violation type clusters in the low-d′, low-SE quadrant on ABCD, does the same perturbation type cluster there on τ-bench? This tests whether the SE × d′ characterization has predictive structure beyond per-dataset fit.
5. **Connection to retrieval/RAG literature.** Decomposed judges with explicit state supply are a form of retrieval-augmented judging. Framing them this way connects the paper to a larger readership without diluting the contribution.

---

## 3. Main Output and Practical Impact

### 3.1 What does this paper produce, concretely?

**A scientific finding:** LLM judges evaluating task-oriented dialogue have a structured, mechanistically characterizable blind spot for value-grounding errors. This blind spot is not an artifact of specific judges, specific prompts, or specific benchmarks — it replicates across three judge models, three prompt conditions, and two datasets.

**A methodological tool:** The SE × d′ 2D characterization, which any researcher or practitioner can apply to diagnose failure modes in their own judge systems. Low-d′, low-SE violations are the deployment-worst case: judges fail silently and stably.

**An architectural recommendation:** Decomposed evaluator architectures substantially narrow the V-gap. We provide (i) a concrete decomposition design, (ii) empirical evidence of its effect size, (iii) a comparison against the obvious alternative of supervised fine-tuning, and (iv) guidance on where decomposition helps most (outcome-critical values, cascading errors, cross-turn contradictions).

**A benchmark:** ABCD-perturbation + τ-bench-perturbation evaluation suites with typed violation taxonomies, matched controls, and validated human annotations.

**An evaluation harness:** Open-source implementation of the seven-metric suite (d′, c, MI, VoI, SE, ICC, IRT) specialized for LLM-judge auditing.

### 3.2 Who does this help, and how?

**Academics working on LLM-as-judge:** Gives a principled characterization of a failure mode that every downstream judge-evaluation paper will need to account for. The SE × d′ framework provides a methodological upgrade over ad-hoc accuracy/agreement reporting.

**Industry practitioners deploying LLM-judges for QA:** Directly actionable. The headline finding ("your judges are probably missing the errors that cost you the most money") reaches every customer-service AI team. The decomposed-architecture recommendation gives them a concrete fix. The SE × d′ tool lets them audit their own judge deployments.

**TOD system researchers:** Provides an audit layer their evaluation pipelines currently lack. Anyone using an LLM judge on an ABCD-style benchmark can now characterize judge failure modes in their paper.

**Compliance and risk teams in regulated industries:** The claim "holistic LLM-judge evaluation systematically under-detects value errors" has implications for any AI deployment in financial services, healthcare, or legal settings where value-grounding is the core compliance question. This paper provides citable evidence for that claim.

**The wider evaluator-auditing community:** The SE × d′ framework is domain-general. Anyone auditing any LLM system with sensitivity/specificity metrics can incorporate semantic entropy into their failure-mode analysis.

### 3.3 The one-line practitioner pitch

*Deploying holistic LLM judges for task-oriented dialogue QA systematically under-detects the most operationally costly error class, no amount of prompting fully closes the gap, and the architectural fix is to split procedural and value scoring into separate evaluators with explicit structured state.*

This is the sentence that gets the paper cited by industry groups. It is the sentence we write the abstract to support.

---

## 4. Formal Framework

This section specifies the mathematical and conceptual scaffolding. EMNLP does not require theorems for empirical papers, but precise definitions strengthen every downstream claim. I'll be specific about what is definition, what is modeling assumption, and what is testable hypothesis.

### 4.1 Signal detection theory setup

We treat perturbation detection as a binary classification problem per violation family `f ∈ {W, V, R}`.

For each judge `j`, each family `f`, and each prompt condition `p`:

- Let `S+ = {(d_orig, d_pert) : d_pert ∈ f}` be perturbed-dialogue trials.
- Let `S− = {(d_orig, d_control) : d_control ∈ N}` be nuisance/control trials.
- Hit rate `H(j,f,p) = Pr(judge prefers d_orig over d_pert | d_pert ∈ f)`.
- False alarm rate `FA(j,p) = Pr(judge prefers d_orig over d_control | d_control ∈ N)`.

Sensitivity and criterion:

```
d′(j,f,p) = Φ⁻¹(H(j,f,p)) − Φ⁻¹(FA(j,p))
c(j,f,p)  = −½ · [Φ⁻¹(H(j,f,p)) + Φ⁻¹(FA(j,p))]
```

where `Φ⁻¹` is the inverse standard normal CDF.

d′ measures the judge's ability to distinguish perturbed from control dialogues, independent of its tendency to flag. c measures that tendency — negative c means liberal (flags readily), positive c means conservative (flags reluctantly). This disentanglement is important: a reviewer asking "is V-blindness just a criterion shift, not a sensitivity deficit?" is answered directly by comparing d′ rather than raw detection rate.

**Primary empirical hypothesis (H1):** For all judges `j` and prompt conditions `p`,

```
d′(j, W, p) > d′(j, V, p)
d′(j, R, p) > d′(j, V, p)
d′(j, V, p) ≈ 0   (failure to reject at α=0.05 in a one-sample test against zero)
```

### 4.2 Value grounding vs. factuality — why they are different

Standard factuality verification asks: *is claim `x` true in the world?* The evidence source is world knowledge (Wikipedia, curated knowledge bases, retrieval from open documents).

Value grounding in TOD asks: *is claim `x` consistent with the session-local structured state of this dialogue?* The evidence source is session-local: the scenario metadata, the dialogue history, and the policy document.

Formally, let `W` denote world knowledge and `S_d` denote the session state of dialogue `d`. Factuality verification computes something like:

```
f_fact(x, W) = Pr(x is true | W)
```

Value grounding verification computes:

```
f_val(x, d) = Pr(x is consistent with (d_history, scenario(d), policy) | d)
```

These are different functions over different evidence sources. A claim can be true in `W` and inconsistent with `S_d` (e.g., a product that exists generally but isn't the one this customer ordered). A judge trained to check `f_fact` will not detect failures of `f_val`, which is the core theoretical argument for why the hallucination-detection literature does not subsume this work.

### 4.3 Plausibility argument for V-blindness in holistic judges

This is not a theorem; it is a plausibility sketch that would go in the discussion. I include it because a rigorous intuition strengthens the paper.

Consider a judge evaluating a dialogue of length `T` turns with slot vocabulary `V_slot` and `k` mentioned slot-value pairs. Under holistic evaluation, the judge receives the dialogue and a rubric. To verify procedural correctness, the judge must check an ordered sequence of action mentions against the rubric — an `O(T)` operation in attention terms, because the ordering is a local property of adjacent turns.

To verify value correctness without structured state supplied in-prompt, the judge must (i) extract each slot-value pair from the dialogue, (ii) cross-reference it against any stated reference value earlier in the dialogue or against the scenario it must have inferred, and (iii) check consistency. This is at least `O(T · k)` in effective attention operations and requires the judge to maintain an implicit value table across long-range attention spans.

Holistic judges operating at finite context budget will allocate attention to whatever features most reduce training-time evaluation loss. If training-time signal is dominated by procedural and surface-fluency features — which is plausible given how judge-distillation datasets are typically constructed — then value-verification attention is systematically under-allocated. This predicts V-blindness as a structural property, not a prompting artifact.

The decomposed architecture directly addresses this: by separating value-verification into its own call with explicit structured state, the `O(T · k)` cost is replaced with `O(k)` state lookup, which is what a value-checking LLM can actually do reliably.

This argument is not a theorem and does not belong in a formal-methods paper. It belongs in our discussion section as a mechanistic interpretation of the empirical finding.

### 4.4 Information-theoretic formalization

For mutual information between judge output and violation family:

```
I(Y_judge; F) = H(F) − H(F | Y_judge)
```

where `Y_judge` is the judge's structured output (verdict + rationale category, coded into discrete classes via clustering of rationale text) and `F ∈ {W, V, R, N}` is the ground-truth violation family.

I is estimated using plug-in estimation with Miller-Madow bias correction, given we have ~500 samples per family. Standard error reported via 1000-sample bootstrap.

**Predicted result (H2):** `I(Y_judge; F) < I_max / 2` where `I_max = H(F) ≈ 2 bits`. That is: judges' outputs carry less than half of the available family-distinguishing information, consistent with the qualitative claim that judges detect "something wrong" without diagnosing what.

For Value of Information of policy context:

```
VoI(p₁ → p₂) = Σ_f [d′(j, f, p₂) − d′(j, f, p₁)] / Δ(information content)
```

where `Δ` is measured in tokens of policy text supplied. This normalizes prompt-conditioning results to a "sensitivity gain per information unit" scale.

### 4.5 Semantic entropy of rationales

Following Kuhn et al. (2023), adapted to judge rationales:

For each dialogue `d` and judge `j`, sample `k=10` rationales `r_1, …, r_k` at temperature 0.7. Cluster rationales into equivalence classes `C = {c_1, …, c_m}` using bidirectional entailment (DeBERTa-large-MNLI or equivalent). Let `p_i = |c_i| / k`. Semantic entropy is:

```
SE(d, j) = −Σ_i p_i log p_i
```

Low SE means the judge produces semantically coherent rationales across samples (stable reasoning). High SE means unstable reasoning or confabulation.

**Predicted result (H3):** For violations missed by the judge, `E[SE | family = V] < E[SE | family = W]`. That is: judges miss V-violations with stable, coherent (wrong) rationales; they miss W-violations with unstable rationales. This is the signature of *confident blindness* vs. *confusion*.

### 4.6 The SE × d′ failure-mode quadrant characterization

For each (perturbation subtype × judge × prompt condition), compute mean SE and d′. Plot in 2D. Define quadrants by median split on each axis.

| Quadrant | Interpretation | Deployment implication |
|---|---|---|
| Low d′, Low SE | **Confident blindness** | No recovery via sampling. Architectural fix required. |
| Low d′, High SE | Confusion | Sampling, self-consistency, or ensembling may recover. |
| High d′, Low SE | Reliable detection | Deployment-ready. |
| High d′, High SE | Unstable detection | Detection may be heuristic; investigate robustness. |

**Predicted result (H4):** V-violations cluster disproportionately in the low-d′, low-SE quadrant; W-violations cluster in the high-d′ quadrants; R-violations distribute across high-d′ and low-d′-high-SE depending on rule type.

This is the figure the paper gets cited for.

### 4.7 Psychometric modeling

For item difficulty ranking, we use a Rasch (1PL) model within each family, estimating judge ability `θ_j` and item difficulty `β_i`:

```
Pr(correct on item i by judge j) = exp(θ_j − β_i) / (1 + exp(θ_j − β_i))
```

We explicitly acknowledge the constraint of having 3 test-takers (judges), which is thin for 2PL estimation. 1PL with 3 judges is defensible if we report uncertainty appropriately and treat item difficulty as ordinal rather than cardinal. We also report non-IRT per-item detection rates for comparability.

---

## 5. Research Questions

**RQ1 (primary, empirical):** Do LLM judges exhibit a structured asymmetry in sensitivity (d′) across procedural (W), value (V), and rule (R) violation families in policy-grounded TOD, after controlling for artifact confounds and criterion shifts?

**RQ2 (mechanism):** Does the asymmetry survive (i) explicit value-verification prompting, (ii) model scale from 20B to 120B, and (iii) cross-dataset transfer to τ-bench?

**RQ3 (information-structural):** How does judge sensitivity respond to incremental policy and structured-state information (Value of Information per information increment)? Is the response structured differently across families?

**RQ4 (diagnostic):** Do judge outputs carry information about *which* violation family occurred (mutual information with family label), or do they produce undifferentiated complaints?

**RQ5 (uncertainty-structural):** When judges miss violations, is the failure characterized by low semantic entropy (confident blindness) or high semantic entropy (confusion)? Does this differ by family?

**RQ6 (prescriptive):** Does evaluator decomposition — splitting procedural and value-grounded scoring with explicit structured state supplied to the latter — narrow the V-gap? How does it compare against supervised fine-tuning as an alternative fix?

**RQ7 (methodological):** Does the combination of semantic entropy and signal-detection-theory sensitivity (SE × d′) produce a failure-mode characterization that is more informative than either axis alone? Does it transfer across datasets?

---

## 6. Datasets

### 6.1 Primary: ABCD v1.1

Retained from current proposal. Structural advantages remain: deterministic action sequences in `kb.json`, machine-readable policy in `guidelines.json`, typed slots in `ontology.json`, 10,042 dialogues with 22-turn average.

Sampling: ~500 policy-compliant dialogues stratified across 55 subflows, filtered for task-completeness, slot-value recoverability, and minimum length (≥12 turns). The existing pipeline has already validated this sample.

### 6.2 Validation: τ-bench (Sierra AI, 2024)

Modern tool-use TOD benchmark with airline and retail domains. LLM-generated agent responses, structured tool APIs, policy documents. Different structurally from ABCD in every dimension that matters for external validity: different dialogue generation process (LLM vs. crowdworker), different action space (tool APIs vs. fixed 30-action set), different policy format (prose vs. structured).

Sample: ~100 dialogues, stratified across domains and perturbation families. Perturbation generation requires new parsers but the basic structure (action traces, policy documents, structured state) is analogous to ABCD.

Purpose: test whether the asymmetry replicates on a structurally distinct benchmark. This is the single highest-leverage addition for Main Track viability.

### 6.3 Optional stretch: doc2dial or MultiDoc2Dial

Document-grounded dialogue. Grounding source is free-text documents rather than structured slots. Including this arm would extend the finding from "structured-slot value-blindness" to "grounded judge evaluation generally." Strongest novelty posture. Costs: new perturbation design specific to document-span grounding.

Decision rule: include if the SE × d′ methodological contribution produces clean findings on ABCD and τ-bench. Skip if the paper is already strong on two datasets.

### 6.4 Datasets explicitly dropped

- **MultiWOZ 2.4** — no policy grounding, no deterministic action ground truth. Already dropped.
- **SGD** — mechanical success definitions, no policy rules. Already dropped.
- **Anything broader (5+ datasets)** — shallow coverage across many benchmarks is weaker than deep coverage on two or three. Reviewers value depth.

---

## 7. Perturbation Design

Retained largely from current proposal with consolidations. Four families: W (workflow), V (value), R (rule), N (nuisance + matched controls). The V-family internal structure is the primary site of scientific focus.

### 7.1 Family V internal taxonomy — this is the paper's focus

| Subtype | Evidence distance | Outcome criticality | Construction |
|---|---|---|---|
| V-local-enum | local (single turn) | variable | swap enumerable slot (membership, payment method) with another ontology-valid value |
| V-local-entity | local | moderate-severe | swap non-enumerable entity (account_id, order_id) with a value from another conversation |
| V-local-monetary | local | outcome-critical | modify dollar amount (refund, price) |
| V-cross-turn-contradiction | k ≥ 3 turns | moderate | agent states X in turn N, contradicts with Y in turn N+k |
| V-cross-turn-status | k ≥ 3 turns | moderate | status reference (delivered/in-transit) contradicts earlier confirmation |
| V-cross-turn-product | variable | moderate-severe | product detail swap across turns |
| V-cascading | whole dialogue | severe | wrong value in early action propagates to all subsequent actions |

This subtype decomposition is what enables the evidence-distance breakdown (RQ3) and the outcome-criticality breakdown.

### 7.2 Family W, R, N — unchanged

W: required-step deletion, local reordering, incorrect subflow transition, premature action, duplicate insertion, oracle branch mismatch.
R: discount limit, eligibility, authorization bypass, threshold, promo-code, oracle-conditioned breach.
N (controls): nuisance paraphrase, length-matched W control, discourse-disrupted control, salience-matched V control.

### 7.3 Family overlap handling — unchanged

Dominant family label + overlap tags. Analyses reported on both full set and clean non-overlapping subset.

---

## 8. Evaluator Setup

### 8.1 Judge models

Three models, chosen for deployment realism within the author's model-access constraints:

- GPT-OSS 20B — small-capacity judge; tests scale hypothesis
- Llama 3.3 70B — strong open-weight judge
- GPT-OSS 120B — largest available; proprietary-class capability

### 8.2 Prompt conditions (core experiment)

- **No-policy:** dialogue + general quality rubric
- **Compressed-policy:** dialogue + 2-3 sentence subflow-specific policy summary
- **Full-policy:** dialogue + complete rule/workflow description
- **Explicit-value-verification** (new): dialogue + structured reference values (scenario slots) + explicit instruction to verify against supplied values
- **Full-context** (new): dialogue + full policy + structured reference values. Upper bound on information access.

The two added conditions are central. They constitute the main-track go/no-go: if V-d′ rises to parity with W-d′ under explicit-value-verification, the paper pivots to "what information unlocks value-checking" rather than "judges are data-blind."

### 8.3 Paradigms

Pointwise scoring (1–5 Likert on 4 dimensions + free-text rationale) and pairwise comparison (which dialogue is more compliant / higher quality overall). Both core, reported jointly.

### 8.4 Sampling for semantic entropy

k=10 samples per (dialogue × judge × prompt condition) at temperature 0.7 for SE computation. Temperature-0 evaluation run separately for deterministic metrics.

### 8.5 Decomposed judge architecture — the prescriptive contribution

Two separate evaluator calls per dialogue:

**Process-judge:** receives dialogue + policy. Scores (i) workflow adherence vs. `kb.json` sequence, (ii) rule compliance vs. `guidelines.json` constraints. Does *not* score value correctness.

**Value-judge:** receives dialogue + structured reference state (scenario slots, policy values, prior value commitments extracted from dialogue). Scores only whether asserted values are consistent with reference. Produces per-slot verdicts: {consistent, contradicted, insufficient_evidence}.

Aggregation: dialogue passes iff both judges pass. Detection rate and d′ computed on the aggregated verdict.

This is a concrete, implementable architecture. The paper includes it as the fifth experimental condition and reports full results against it.

---

## 9. Metric Framework — the Seven-Metric Suite

Organized by what each metric answers. No redundancy; each addresses a distinct question.

### 9.1 Detection and discrimination

**d′ per family × judge × prompt condition** — sensitivity, disentangled from bias. Core.

**Criterion c per family × judge × prompt condition** — bias (liberal/conservative threshold). Comes free with d′. Core.

**Pairwise preference accuracy** — retained for comparability with FBI and judge-evaluation literature. Core but presented as classical accompaniment to d′.

**Pointwise detection rate** — same reason. Core accompaniment.

### 9.2 Diagnostic quality

**Rationale correctness rate** — human-annotated. Does the judge's free-text rationale identify the *actual* violation type? Core.

**Mutual information I(Y_judge; F)** — does judge output carry information about violation family? Primary.

### 9.3 Uncertainty structure

**Semantic entropy of rationales** over k=10 samples — primary methodological contribution, combined with d′ in §9.7.

**ICC test-retest reliability** — stability of scoring across resamples at temperature 0.7. Supporting.

### 9.4 Information sufficiency

**Value of Information for policy context** — bits of sensitivity gain per token of policy information added. Formalization of prompt-conditioning analysis. Supporting.

### 9.5 Item-level analysis

**Rasch (1PL) item difficulty β_i** — ranks violations by difficulty, controlling for judge ability θ_j. Secondary, with honest reporting of the 3-judge constraint.

**Per-slot-type detection rate** — which ABCD slot types are checked, which are ignored. Practitioner-facing.

### 9.6 Measurement validation

**Gwet's AC1 (human-judge agreement)** — on the validation subset. Validates the measurement layer.

### 9.7 The 2D characterization

**SE × d′ scatter with quadrant labels** — per-violation-subtype characterization across the four failure-mode quadrants. Primary methodological contribution.

### 9.8 Metrics explicitly dropped

- **Raw detection rate as primary metric** — replaced by d′; raw rate retained for classical comparability only.
- **Cohen's kappa** — unstable under class imbalance; Gwet's AC1 strictly preferable.
- **ECE / Brier** — pointwise judges don't natively emit calibrated probabilities. Moving to verbalized confidence would require building a separate pipeline and risks pulling the paper into calibration-benchmark territory. The uncertainty structure is captured through SE more appropriately.
- **Spearman / Kendall** — ranking metrics are misaligned with the detection-pair design.
- **Selective Robustness Gap (SRG)** — abstraction without added clarity; d′ disaggregated by family communicates more.

---

## 10. Experiments — Exhaustive Plan

Organized by logical dependency (not timeline). Each experiment specifies purpose, method, success criterion, and role in the paper narrative.

### 10.1 Stage 0 — Viability on ABCD

**E0.1 — Label sanity check.** Two human annotators verify automatic perturbation labels on 50–100 examples across all families. Success: ≥ 90% agreement with automatic label. Failure mode: ≤ 90% means the construction pipeline is the bug and nothing downstream is trustworthy.

**E0.2 — Noise floor on compliant dialogues.** Score 50 unperturbed, policy-compliant dialogues. What scores do judges assign, and do those scores correlate with whether the judge actually *could* have value-checked (via post-hoc probing)? Purpose: establish that holistic judges don't value-check by default, independent of perturbation.

**E0.3 — Pilot detection grid (50 dialogues, Llama 3.3 70B).** All four families, all V subtypes, holistic prompt. Report d′ per family. Required pattern: d′(W) > 1.0, d′(R) > 0.8, d′(V) < 0.5 with confidence interval including 0. Failure mode: d′(V) substantially above zero — the main empirical claim dissolves.

**E0.4 — Semantic entropy feasibility (20 dialogues).** k=5 pilot samples. Does entailment-based clustering produce interpretable equivalence classes? Does SE vary across dialogues (not saturated at zero or max)? Failure mode: SE uniformly near zero or near max means the metric doesn't discriminate on this data.

**E0.5 — ICC feasibility.** Test-retest on 20 dialogues at temperatures 0.3, 0.5, 0.7. Confirms ICC is non-trivial at workable temperature.

**E0.6 — Symbolic baseline dry run (50 dialogues).** Deterministic NER + exact string match on V perturbations. Threshold-extraction on R perturbations. Required: symbolic beats LLM on V by ≥ 0.20 absolute detection rate.

**E0.7 — Explicit value-verification dry run (20 dialogues).** Give judge structured reference values and explicit instruction. Does V-d′ rise? How much? Failure mode for main empirical claim: V-d′ rises to parity with W-d′. This flips the paper to the "what unlocks value-checking" version.

**E0.8 — Decomposed judge dry run (20 dialogues).** Process-judge + value-judge pipeline. Does decomposed d′(V) exceed holistic d′(V) by a meaningful margin? Target: ≥ 0.5 absolute d′ gap.

### 10.2 Stage 1 — Core empirical findings on ABCD

**E1.1 — Main detection grid.** 3 judges × 5 prompt conditions × 4 families × 2 paradigms × ~500 dialogues. Headline table. Report d′, c, detection rate, pairwise preference, with bootstrap confidence intervals.

**E1.2 — Artifact-control grid.** Same structure restricted to length-matched W controls, discourse-disrupted controls, salience-matched V controls, and nuisance baselines. Expected: d′(W) drops meaningfully under length control (W detection was partly length-driven); d′(V) is unaffected by salience control (V was never artifact-driven).

**E1.3 — Clean non-overlapping subset.** Same structure, overlap-tagged perturbations removed. Confirms family differences aren't a labeling artifact.

**E1.4 — Rationale correctness annotation.** 150 judge outputs, stratified across families and detected/missed cases, two annotators, adjudication protocol. Inter-annotator agreement reported via Gwet's AC1.

**E1.5 — Reverse-direction experiment.** Dialogues where process is correct but values are wrong vs. dialogues where values are correct but process is wrong. Clean asymmetry table. Most visually compelling evidence of the dissociation.

**E1.6 — Mutual information analysis.** I(judge output cluster; violation family) computed via plug-in with Miller-Madow correction, bootstrapped CI. Compared against I_max and against I computed on a random-labeling baseline.

### 10.3 Stage 2 — Mechanism on ABCD

**E2.1 — Evidence-distance breakdown within V.** Separate d′ reporting for V-local, V-cross-turn, V-cascading. Hypothesis: monotonic decrease in d′.

**E2.2 — Outcome-criticality breakdown within V.** Separate d′ for peripheral, moderate, and outcome-critical V perturbations. Hypothesis: no monotonic relationship — judges do not weight by operational importance.

**E2.3 — Per-slot-type breakdown within V.** d′ per slot type (membership_level, order_id, account_id, dollar_amount, etc.). Practitioner-facing finding.

**E2.4 — Decomposed vs. holistic judge, full scale.** Full grid: decomposed judge on all 500 dialogues, all 3 judge backbones, all 4 families. Paired comparison vs. holistic baseline. Required result: decomposed d′(V) > holistic d′(V) by ≥ 0.5 absolute.

**E2.5 — Explicit value-verification at full scale.** Full evaluation under the value-verification prompt. Reports Value of Information from holistic → explicit-verification, per family.

**E2.6 — Symbolic baseline at full scale.** Per-subtype comparison: LLM judges vs. deterministic baseline on V and R perturbations. Expected: symbolic wins on enumerable slots, loses on cross-turn reasoning.

**E2.7 — Model-scale analysis.** d′(V) across 20B / 70B / 120B judges. Hypothesis: invariance with scale — data-blindness is not resolvable by scaling.

**E2.8 — Rationale-verdict consistency.** For judges that produce verdicts plus rationales, does the rationale text actually support the verdict direction? Assessed by a separate evaluation round (human annotators or strong reference judge).

### 10.4 Stage 3 — Uncertainty structure on ABCD

**E3.1 — Semantic entropy per family and subtype.** k=10 samples per query, entailment-clustering with DeBERTa-large-MNLI. SE distribution per family. Hypothesis: E[SE | missed V] < E[SE | missed W].

**E3.2 — ICC test-retest per family.** Stability at temperature 0.7. Is judge reliability itself family-dependent?

**E3.3 — IRT item difficulty.** Rasch model within family. Rank-order violations by difficulty controlling for judge ability. Honest reporting of 3-judge constraint.

**E3.4 — SE × d′ 2D characterization — *the paper's headline methodological figure*.** Scatter each (perturbation subtype × prompt condition) by (SE, d′) with quadrant labels. Qualitative and quantitative analysis of quadrant distribution by family. This is the figure the paper gets cited for.

**E3.5 — Uncertainty-aware item difficulty.** Combine SE and IRT β_i into a joint ranking. Violations with high β_i (hard) AND low SE (confidently missed) are flagged as the most dangerous for deployment. This is the novel methodological combination described in §2.

### 10.5 Stage 4 — Cross-dataset validation on τ-bench

**E4.1 — τ-bench replication arm.** 100 dialogues, one perturbation per family (full V subtype breakdown if possible), best two judges, length-matched and nuisance controls. Core detection grid. Required pattern: d′(W) > d′(V) replicates.

**E4.2 — τ-bench decomposed vs. holistic.** Smaller-scale replication of E2.4 on τ-bench. Required: decomposition benefit replicates.

**E4.3 — τ-bench SE × d′.** Do the same perturbation types that cluster in each failure-mode quadrant on ABCD cluster in the same quadrants on τ-bench? Tests whether the 2D characterization has predictive transfer.

**E4.4 — τ-bench explicit value verification.** Does the V-gap respond to explicit prompting on τ-bench the same way as on ABCD?

### 10.6 Stage 5 — Optional stretch arm: document-grounded extension

**E5.1 — doc2dial / MultiDoc2Dial value-blindness test.** 50–100 dialogues with document-span grounded perturbations. Does the V-blindness asymmetry replicate when "value" means "document-grounded claim" rather than "structured-slot value"? If yes, the finding generalizes to grounded judge evaluation broadly.

### 10.7 Stage 6 — Fine-tuning study (see §11 for full discussion)

**E6.1 — SFT judge training.** Fine-tune GPT-OSS 20B on V-perturbation detection data from a train split, holding out V-subtypes for evaluation. Compare d′(V) pre- and post-SFT.

**E6.2 — SFT generalization test.** Does SFT on certain V-subtypes transfer to held-out subtypes? Or does it only memorize the specific patterns seen in training?

**E6.3 — SFT vs. decomposition comparison.** Apples-to-apples: does SFT on the small judge close the gap as much as decomposition does with the strong judge? Which is the preferred architectural recommendation?

**E6.4 — SFT cross-dataset transfer.** SFT on ABCD V-perturbations, evaluate on τ-bench. Does V-detection transfer, or is it dataset-specific?

### 10.8 Stage 7 — Measurement validation and robustness

**E7.1 — Full Gwet's AC1 reporting.** Human-judge agreement broken down by family, prompt condition, and dataset.

**E7.2 — Surface-cue sensitivity ablation.** Inject typos, filler words, topic shifts into compliant dialogues. Does surface perturbation move scores more than value corruption? Surprising-if-yes result.

**E7.3 — Inter-judge disagreement clustering.** When judges disagree, which family does the disagreement cluster in? Higher V-disagreement is independent evidence that V is a harder task.

**E7.4 — Rationale cluster qualitative analysis.** For missed V-violations, what reasons do judges give? Qualitative taxonomy of failure explanations. Useful for the discussion section.

---

## 11. Fine-Tuning: Whether, Why, and How

You asked specifically about SFT. Here is my honest position.

### 11.1 Should SFT be included?

**Yes — as a comparison condition, not as the paper's main thrust.**

The argument for including it: the paper's prescriptive contribution (decomposed judges) is strengthened by showing that the alternative fix most practitioners would reach for first (SFT) is either less effective or has worse generalization properties.

The argument against: SFT adds methodological surface area. It risks pulling the paper's narrative toward training methodology. Every additional judge-training paper competes in a crowded sub-area.

The case closes in favor of including it because SFT *as a comparison condition* is cheap relative to its narrative benefit. The paper goes from "decomposition helps" to "decomposition helps and is preferable to the obvious training-based alternative, which has these limits." That's a qualitatively stronger prescriptive claim.

### 11.2 How should SFT be framed?

Not as: "We train a better judge."
But as: "Can the value-blindness failure mode be removed by training alone, or does it require architectural intervention?"

The framing matters. The first framing competes with every SFT judge paper ever published. The second framing makes SFT a tool for characterizing the failure mode, not a contribution in itself.

### 11.3 How should SFT be executed?

**Target model:** GPT-OSS 20B (smallest judge, cheapest to fine-tune, most tractable given Capital One infrastructure constraints).

**Training data:** Synthetic. Perturbation-detection pairs generated from ABCD train split. Two training setups:

- *Narrow SFT:* train on all V-subtypes seen in training, evaluate on held-out V-subtypes.
- *Broad SFT:* train on all perturbation families, evaluate on held-out subtypes within each family.

**Training objective:** Verdict prediction (violated / not violated) plus violation family prediction plus free-text rationale. Three-headed output; standard causal LM loss.

**Held-out axes:**

- Held-out V-subtypes (tests whether SFT memorizes patterns or learns value-checking)
- Held-out slot types (tests whether SFT transfers across slot vocabularies)
- Held-out dataset (τ-bench, tests whether SFT transfers beyond training distribution)

**Comparison conditions:**

- Pre-SFT GPT-OSS 20B (baseline)
- Post-narrow-SFT GPT-OSS 20B
- Post-broad-SFT GPT-OSS 20B
- Holistic Llama 3.3 70B (untrained, stronger backbone)
- Decomposed Llama 3.3 70B (no SFT, different architecture)

### 11.4 Expected results and what they mean

**If SFT closes d′(V) gap on in-distribution perturbations but fails to generalize to held-out subtypes or τ-bench:** data-blindness is partially a training-data coverage problem, but the more fundamental fix is architectural. Decomposition still wins the prescriptive argument because it generalizes without training.

**If SFT closes d′(V) gap and generalizes well:** V-blindness was a training-data problem. Decomposition is still useful (simpler, no training required) but the paper's prescriptive claim weakens.

**If SFT fails to close d′(V) gap at all:** V-blindness is an architectural/capability limit of the base model. Decomposition is the only path. Strongest version of the prescriptive claim.

All three outcomes are publishable. The second is the weakest for the paper's current framing but still produces a clear finding. This is why SFT is worth including — no outcome kills the paper.

### 11.5 What SFT does *not* include

- No RLHF/DPO. That's a different research program.
- No confidence-head training. ChatGPT's proposed calibration-plus-abstention pipeline is not within scope — including it would pull the paper back into the crowded calibration sub-area.
- No continual pretraining. Out of scope.
- No judge ensembling beyond the SE-sampling needed for the uncertainty analysis.

### 11.6 Risks with SFT

- Infrastructure: fine-tuning GPT-OSS 20B in the Capital One environment may have practical constraints. If SFT is infeasible, the paper stands without it — contributions 1, 2, and 3 are independent.
- Data contamination: if SFT training data is generated from the same pipeline as evaluation data, there's a within-distribution inflation risk. Held-out V-subtypes and cross-dataset evaluation are the controls.
- Narrative drift: the paper must resist becoming a judge-training paper. Framing SFT as a failure-mode characterization tool, not a contribution, is the anchor.

---

## 12. Math and Theorems — How Much is Needed

### 12.1 What the paper needs

**Required, not optional:**

- Formal definitions of d′, c in the paper's specific SDT setup (§4.1)
- Formal definition of semantic entropy over rationales (§4.5)
- Formal information-theoretic setup for MI and VoI (§4.4)
- Formal specification of the Rasch IRT model (§4.7)
- Formal statement of the SE × d′ quadrant definition (§4.6)

These are definitions, not theorems. They are required for rigorous reporting and would be flagged as missing by any serious reviewer.

### 12.2 What would strengthen the paper

**Plausibility argument for V-blindness from attention budget** (§4.3). Not a theorem; a mechanistic intuition. Goes in the discussion section. Strengthens the paper's claim that V-blindness is structural, not prompting-artifact.

**Theoretical argument for why holistic evaluation is mismatched to multi-turn value verification** — somewhat stronger version of §4.3, possibly framed as a proposition rather than a theorem. Something like: "Under bounded-attention holistic evaluation, value verification cost scales as O(T · k) while procedural verification scales as O(T), so under attention scarcity value-checking is preferentially sacrificed." This is a *sketch*, not a *proof*, but it can be argued rigorously in a few paragraphs. Strengthens the paper's mechanistic story.

**Characterization of the decomposed-judge speedup.** The decomposed architecture replaces O(T·k) with O(k) — argue this formally. Supports the prescriptive contribution.

### 12.3 What the paper does *not* need

- Theorems in the formal-methods sense. This is an empirical paper; theorems are not expected.
- Convergence proofs. Not that kind of paper.
- Optimality arguments. The paper recommends an architecture; it does not claim the architecture is optimal.

### 12.4 EMNLP expectations

Main Track empirical papers typically contain formal definitions of all metrics and estimation procedures, plus any modeling assumptions underlying statistical tests. They do not typically contain theorems. Papers that *do* include theoretical propositions (e.g., a plausibility argument about why their architecture should work) are rewarded for it but not penalized for omitting it.

My recommendation: include §4.1–4.7 as formal framework in a dedicated methods subsection. Include §4.3 and the decomposed-judge formal argument as a "Mechanistic Interpretation" paragraph in the discussion. Do not attempt to prove theorems.

---

## 13. Writing and Paper-Structure Strategy

### 13.1 Structural outline

Suggested EMNLP paper structure (target 8 pages + unlimited refs + appendix):

1. Introduction (1 page) — the practitioner pitch, the finding, the methodological contribution, the architectural recommendation
2. Related Work (0.75 pages) — FBI, TD-EVAL, CompliBench, CAUSE, hallucination detection, semantic entropy, SDT in NLP
3. Formal Framework (0.75 pages) — definitions from §4
4. Datasets and Perturbation Design (1 page) — ABCD, τ-bench, perturbation taxonomy
5. Evaluator Setup (0.5 pages) — judges, prompts, paradigms, decomposed architecture
6. Metrics (0.5 pages) — the seven-metric suite
7. Experiments and Results (3 pages) — headline grid, decomposition result, explicit-verification, cross-dataset replication, SE × d′ figure, IRT results
8. Discussion (0.75 pages) — mechanistic interpretation, implications, limitations
9. Conclusion (0.25 pages)

Appendix: full perturbation examples, annotator guidelines, SFT training details, additional cross-dataset results, rationale cluster analysis.

### 13.2 The headline figure

The SE × d′ 2D scatter, color-coded by violation family, with quadrant labels. This is the paper's iconic figure. It should be designed to be readable at a glance and to convey both the empirical finding (V in low-d′ quadrants) and the methodological contribution (the quadrant framework).

### 13.3 The headline table

A decomposed comparison: holistic Llama-70B vs. decomposed Llama-70B vs. holistic GPT-OSS-120B vs. SFT GPT-OSS-20B, across W/V/R families, with d′ and CI. This table answers "which fix works" at a glance.

### 13.4 The one sentence in the abstract

We write the whole paper to support a single sentence: *LLM judges evaluating task-oriented dialogue systematically miss value-grounding errors via a confident-blindness failure mode that is invariant to model scale, prompting strategy, and supervised fine-tuning but is substantially narrowed by evaluator decomposition.*

### 13.5 Writing hazards to avoid

- Collapsing into "another judge-auditing benchmark paper." The mechanistic framing and SE × d′ characterization are the differentiators; lead with them.
- Overclaiming cross-domain generalization. The paper is about policy-grounded TOD. Document-grounded dialogue is a bonus arm, not the main claim.
- Treating SFT as a contribution rather than a comparison. SFT results serve the decomposition story.
- Writing the abstract as a taxonomy of contributions. Write it as a single finding with supporting evidence.
- Under-citing hallucination detection and calibration literatures. Cite them explicitly and explain the distinction (§4.2). A reviewer who spots a missing citation is a reviewer who rejects.

---

## 14. What's Missing, Risks, Mitigations

### 14.1 Risks to the empirical claim

**Risk:** V-blindness is just a prompting artifact; proper prompting closes the gap.
**Mitigation:** E0.7 and E2.5 explicitly test this. If V-gap closes under explicit value prompting, the paper pivots (see §2.3).

**Risk:** V-blindness is a capability limit of the specific judge models, not a general finding about LLM judges.
**Mitigation:** E2.7 model-scale analysis across 20B / 70B / 120B. Replication on τ-bench (E4.1). If V-blindness does not replicate across scales or datasets, the paper's generality claim is scoped appropriately.

**Risk:** The asymmetry is a perturbation-construction artifact — W perturbations are somehow easier to detect than V perturbations for reasons unrelated to value-grounding.
**Mitigation:** Artifact-control subsets (E1.2), human validation of perturbation naturalness (E1.4), length-matched and salience-matched controls.

### 14.2 Risks to the methodological claim

**Risk:** Semantic entropy clustering produces unreliable equivalence classes on judge rationales.
**Mitigation:** E0.4 feasibility check. Multiple clustering thresholds reported. Human validation of clusters on a sample.

**Risk:** IRT with 3 judges is too thin for reliable item-difficulty estimation.
**Mitigation:** Report Rasch (1PL) not 2PL. Report item difficulty as ordinal not cardinal. Report raw per-item detection rates alongside IRT for comparability.

**Risk:** Reviewers argue SE × d′ is not genuinely novel — "just two metrics plotted together."
**Mitigation:** The novelty is the failure-mode quadrant framework and its empirical predictive value for deployment triage. The paper makes this explicit and demonstrates quadrant-based findings that neither metric alone produces.

### 14.3 Risks to the prescriptive claim

**Risk:** Decomposition is not obviously superior to SFT, or both produce comparable results.
**Mitigation:** E6.3 SFT-vs-decomposition comparison is designed to give an honest answer. If they are comparable, the paper presents decomposition as the simpler-to-deploy option (no training required, no data curation) while acknowledging SFT as a viable alternative.

**Risk:** Decomposition improvements don't transfer to τ-bench.
**Mitigation:** E4.2 explicitly tests this. Non-replication would be reported honestly and would scope the architectural claim to ABCD-style benchmarks.

### 14.4 Risks to publication viability

**Risk:** Main Track rejection due to novelty framing collapse.
**Mitigation:** Findings is the conservative target and the paper clears it easily. Main Track pursuit is not the primary goal.

**Risk:** Slow reviewer turnaround on perturbation construction and human validation.
**Mitigation:** Invest heavily in documentation. Every perturbation has a type tag, construction record, and validation status. Makes reviewer rebuttal easier.

**Risk:** Capital One model-access constraints limit reproducibility reporting.
**Mitigation:** Release all prompts, perturbation generation code, evaluation harness, and aggregate results. Model weights are not released but the experimental infrastructure is.

---

## 15. Summary Table of Contributions and Evidence

| Contribution | Evidence experiments | Sufficient on own? | Main Track dependency |
|---|---|---|---|
| V-blindness empirical finding | E1.1, E1.2, E1.3, E1.5, E2.7, E4.1 | Yes (Findings) | Requires τ-bench replication (E4.1) |
| SE × d′ methodological framework | E3.1, E3.4, E3.5, E4.3 | Yes (Findings) | Requires clear quadrant-based findings |
| Decomposed architecture recommendation | E2.4, E4.2, E6.3 | Yes (Findings) | Requires effect size ≥ 0.5 d′ and cross-dataset replication |
| Perturbation benchmark artifact | Release | No | Strengthens paper but insufficient alone |

All four contributions targeting Main Track.

---

## 16. Final Assessment

**Is the paper truly novel?** Yes. The combination of (V-blindness as a mechanistically distinct failure mode) + (SE × d′ 2D characterization as a general-purpose tool) + (decomposed architecture as a prescriptive fix) is not claimed by any existing paper I have identified. Each contribution has the closest-neighbor prior work explicitly addressed in related work.

**Is it EMNLP-acceptable?** Confidently yes for Findings. Plausibly yes for Main Track, conditional on τ-bench replication and the SE × d′ finding landing cleanly.

**Does more math need to be added?** Formal definitions throughout (§4) — yes, required. Theorems — no. A plausibility argument for the mechanism (§4.3) — strongly recommended.

**Should SFT be included?** Yes, as a comparison condition, framed as a tool for characterizing the failure mode rather than as a paper contribution.

**What is the main output?** An empirical finding with practical consequences for industry deployers of LLM-as-judge, a methodological tool that generalizes beyond this paper's domain, and an architectural recommendation with evidence that it works where the obvious alternatives don't.

**Who does it help?** Anyone deploying or studying LLM-as-judge for task-oriented dialogue — which is a large and growing community — plus the broader evaluator-auditing research area that needs better failure-mode characterization tools.

The paper is worth writing. Execute cleanly, replicate on τ-bench, get the SE × d′ figure to produce clear findings, and Main Track is within reach.
