# Mechanistic Interpretability of LLM Judges in Task-Oriented Dialogue

## A Complete Execution Guide

**Working title (current):** *Are LLM Judges Faithful to Their Rubrics? A Mechanistic Audit of Internal Compliance Representations in Task-Oriented Dialogue Evaluators*

**Target venue:** EMNLP 2026 main track preferred; Findings is the realistic conservative target. Workshop version at BlackBoxNLP or NeurIPS interpretability workshops first.

**Author:** Solo project, with active community engagement as part of the methodology.

**Project type:** Mechanistic interpretability research applied to LLM-as-judge evaluation in policy-grounded TOD. Primary methodologies: activation patching, probing classifiers, sparse autoencoder analysis (optional), and LoRA weight perturbation.

---

## Part 1 — What This Paper Is and Why It Exists

### 1.1 The scientific question

When an LLM judge issues a verdict on a task-oriented dialogue, what is it actually computing internally, and does that computation align with the rubric humans believe it's following?

This is different from existing judge auditing. Prior work asks whether the judge's verdict matches human judgment, or whether the verdict changes when inputs are perturbed. Both treat the judge as a black box. This work asks whether the judge's internal representation of compliance separates the dimensions humans use to define compliance, and whether verdict-driving features causally correspond to those dimensions.

### 1.2 The four claims the paper makes

The paper is structured around four claims, each independently testable. The paper lands if at least two land cleanly; three is a strong paper.

**Claim 1: Judge representations are not faithful to evaluation criteria.** When a judge processes a dialogue, it builds an internal representation that drives the verdict. The hypothesis is that workflow violations, value violations, and rule violations are not represented as cleanly separable directions in this internal space — they collapse into a more generic "this dialogue is bad" signal. If true, judges are not actually evaluating along the dimensions their rubrics specify.

**Claim 2: Verdict-driving features are localizable but not interpretable as criteria.** Causal interventions identify specific layers and components whose modification changes the verdict. The hypothesis is that these cluster in mid-layers and that intervening on them flips verdicts in ways that don't correspond to human notions of what changed about the compliance assessment.

**Claim 3: Rationales are post-hoc.** When judges produce free-text rationales, those rationales don't faithfully reflect the computation that produced the verdict. Testable by intervening on the verdict-driving computation and checking whether rationales update correspondingly or merely "explain" whatever verdict came out.

**Claim 4: LoRA-trained judges have brittle decision functions.** Small weight perturbations in specific adapter components disproportionately change verdicts, suggesting fine-tuning produces concentrated, fragile evaluation functions rather than robust ones.

### 1.3 The output and impact

**Scientific finding:** A characterization of how LLM judges internally represent the evaluation criteria they apply to TOD dialogues.

**Methodological contribution:** A protocol for mechanistic audits of LLM judges that other researchers can apply to their own judge systems.

**Personal goal:** Fluency with mechanistic interpretability techniques, relationships in the interpretability community, credibility as someone who can do this work.

The paper succeeds if all three land, even softly. It fails if you complete experiments without absorbing the field.

### 1.4 Critical novelty context (from literature review)

Several papers occupy near-neighborhoods to this proposal:

- **Cyberey, Ji, Evans (arXiv:2601.16398, January 2026)** uses "white-box sensitivity auditing with steering vectors" terminology for general LLM auditing. **You cannot use the "white-box auditing" framing as the headline.** They own it.
- **JUSSA (Eshuijs et al., arXiv:2505.17760, May 2025)** uses steering vectors to aid LLM-as-judge by steering the *agent*. Your distinction: you intervene on the *judge*, not the agent.
- **"Alignment is Localized" (arXiv:2510.16167, October 2025)** activation-patches DPO-aligned generators to localize preference effects. Your distinction: targets generative judges with reasoning traces, not preference-tuned generators.
- **SAFER, SARM, Christian et al.** do mechanistic interpretability of reward models. Your distinction: judges produce verdicts plus rationales via autoregressive generation, not scalar scores; this creates a faithful-vs-confabulated reasoning question that doesn't exist for scalar reward heads.
- **TOD-ProcBench (arXiv:2511.15976, November 2025)** is a recent ABCD-based compliance benchmark. Build on top of its violation taxonomy rather than reinventing.

**Your defensible novelty positioning:** *"Causal faithfulness of judge representations to multi-dimensional evaluation criteria in policy-grounded TOD."* This is sharper than "white-box auditing" and has no clear scoop in surfaced literature.

---

## Part 2 — Datasets

### 2.1 Primary dataset: ABCD v1.1

**Citation:** Chen, Chen, Yang, Lin, Yu (NAACL 2021), arXiv:2104.00783. Full dataset and documentation: https://github.com/asappresearch/abcd

**Why ABCD:** It is the only publicly available TOD dataset that simultaneously provides deterministic action sequences, machine-readable policy rules, and structured slot-value annotations. All three are required for your perturbation design and for your causal-faithfulness questions.

**Properties:**

- 10,042 dialogues (8,034 train / 1,004 dev / 1,004 test)
- Average 22.08 turns per dialogue
- Average 3.73 actions per dialogue (highest action density among comparable datasets)
- 10 flows → 55 subflows → 30 distinct action types
- 125 enumerable + non-enumerable slot types
- Domain: customer-service for fictional retail company

**Key files:**

- `kb.json`: maps each subflow to its unique, ordered action sequence (deterministic ground truth)
- `guidelines.json`: machine-readable encoding of company policies (discount limits, verification requirements, eligibility rules)
- `ontology.json`: full taxonomy of flows, subflows, actions, and 231 slot-value combinations

**Status:** You already have ~500 policy-compliant dialogues sampled and the perturbation pipeline built. This is reusable infrastructure.

### 2.2 Secondary dataset: τ-bench

**Citation:** Yao et al. (2024), arXiv:2406.12045. https://github.com/sierra-research/tau-bench

**Why τ-bench:** Modern tool-use TOD benchmark. LLM-generated agent responses, structured tool APIs, prose policy documents. Different from ABCD in every external-validity dimension that matters: dialogue generation process (LLM vs. crowdworker), action space (tool APIs vs. fixed 30-action set), policy format (prose vs. structured).

**Use:** ~100 dialogues for cross-dataset replication of your most important findings (probing analysis and activation patching). Not the primary experimental substrate.

**Setup overhead:** Requires writing new perturbation generators for τ-bench's structure. Budget for this.

### 2.3 Build on TOD-ProcBench

**Citation:** Ghazarian et al. (Amazon), arXiv:2511.15976 (November 2025). https://arxiv.org/abs/2511.15976

**Why:** This is the most recent ABCD-based compliance benchmark. It has explicit IF-THEN workflow taxonomies and tests instruction retrieval, violation detection, and compliant response generation. Use their violation taxonomy as the structured framework for your compliance categories rather than inventing your own. This positions you as building on community work and sidesteps reinventing taxonomic structure.

### 2.4 Datasets explicitly NOT to use

- **MultiWOZ (any version):** No policy grounding, no action sequence ground truth. Already correctly dropped.
- **SGD:** Mechanical success definitions, no policy rules. Already correctly dropped.
- **Anything broader than three datasets:** Shallow coverage across many benchmarks is weaker than depth on one or two. Reviewers value depth.
- **doc2dial / MultiDoc2Dial:** Different grounding type (documents vs. structured state). Could extend the contribution if everything else lands cleanly, but not load-bearing. Optional stretch only.

### 2.5 Models you'll be auditing

You have access through Capital One to:

- **Llama 3.3 70B** — Primary subject. Strong open-weight judge. Activations accessible if HuggingFace weights are loadable in your environment. Verify this first (see Part 5).
- **GPT-OSS 120B** — Largest available, proprietary-class capability. Activation access may be more restricted; verify.
- **GPT-OSS 20B** — Smaller capacity. Useful for: (a) the LoRA fine-tuning study, (b) more aggressive interpretability experiments, (c) testing scale invariance of findings.

You should also do foundational technique work on small open models (GPT-2-small, Pythia-1.4B, Pythia-2.8B) where the interpretability community has extensive prior work and where your results can be directly compared against known findings. This is for skill development, not paper experiments.

---

## Part 3 — The Three Levels of Perturbation (Crucial — Don't Conflate)

This paper involves three distinct kinds of perturbation. Conflating them is the most common conceptual error in mechanistic interpretability work. Be precise.

### 3.1 Input perturbations on dialogues

**What they are:** Modifications to the dialogue text itself before it reaches the judge. You take a clean compliant ABCD dialogue and create a violated version.

**Where they come from:** Your existing perturbation pipeline, organized along TOD-ProcBench's violation taxonomy. Three categories:

- **Workflow (W) violations:** required-step deletion, action reordering, incorrect subflow transition, premature action execution, duplicate action insertion
- **Value (V) violations:** enumerable slot swap, non-enumerable entity swap, monetary value change, cross-turn value inconsistency, cascading value error
- **Rule (R) violations:** discount limit exceeded, eligibility violation, authorization bypass, threshold violation, promo-code misapplication

**Role in this paper:** *Stimuli that systematically vary the type of compliance violation present.* You are NOT testing whether judges detect them (that was the old framing). You are using them as controlled inputs that let you ask whether judges' internal representations distinguish between the violation types.

**Status:** ~500 perturbation pairs already constructed. Reusable.

### 3.2 Activation perturbations on the judge

**What they are:** Interventions on the judge's intermediate computations while it processes a (possibly-perturbed) dialogue.

**Two main techniques:**

- **Activation patching:** Run the judge on dialogue A, capture activations at some layer. Run on dialogue B, but at that layer, replace the activation with the captured one from A. Observe whether the verdict changes.
- **Activation steering:** Compute a "direction" in activation space (typically by differencing activations between two sets of contrastive examples). Add or subtract this direction at inference time. Observe behavior changes.

**Role:** Test causality. Which internal computations are necessary or sufficient for the judge's verdict? If patching layer L's activation from a violated dialogue into a compliant-dialogue run flips the verdict, layer L carries verdict-driving information.

**Required infrastructure:** Hook into intermediate layers of the judge model. TransformerLens for supported models; raw PyTorch hooks for others.

### 3.3 Weight perturbations on a trained judge adapter

**What they are:** Perturbations to the weights of a LoRA adapter you've trained on top of a base model to specialize it as a TOD judge.

**Specific interventions:**

- Gaussian noise of varying magnitude (σ = 0.001, 0.01, 0.1) on adapter weights
- Per-rank scaling: scale specific ranks of the LoRA decomposition up or down
- Per-rank zeroing: remove individual rank components and observe behavior change
- A-matrix vs. B-matrix asymmetric perturbation (per Zhu et al. 2024)

**Role:** Test fragility. Does the judge's discriminative function depend on a few specific adapter components, or is it distributed? Brittleness suggests memorization rather than learning a robust evaluation function.

### 3.4 The unifying frame

The three levels operate at different abstraction layers:

- Input perturbations vary *what the judge sees*
- Activation perturbations vary *what the judge computes given fixed input*
- Weight perturbations vary *what computation the judge is set up to perform*

In writing, frame all three as instances of a common methodological idea: **internal intervention as a judge-auditing primitive.** Activations are the base model's internal state; LoRA weights are the trained adapter's internal state. Both are interventions on internals to characterize judge behavior.

---

## Part 4 — Foundational Reading List

This is what you read before serious paper work begins. Tier 1 papers are non-negotiable; Tier 2 supports specific phases; Tier 3 is for community context.

### 4.1 Tier 1 — Read carefully, take notes, work through math

These are the canonical interpretability papers you will be assumed to have read. Cannot be skipped.

**Transformer architecture and circuits foundation:**

- Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits." Anthropic. https://transformer-circuits.pub/2021/framework/index.html — The conceptual foundation for thinking about transformers as composable circuits. Read until you understand what an "induction head" is and why it's notable.
- Olsson et al. (2022), "In-context Learning and Induction Heads." Anthropic. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html — The canonical example of a mechanistic finding.
- Wang et al. (2023), "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small." arXiv:2211.00593 — The reference example of circuit-level analysis on a real model.

**Causal intervention methodology:**

- Meng et al. (2022), "Locating and Editing Factual Associations in GPT" (ROME). NeurIPS 2022. arXiv:2202.05262 — The foundational paper for causal tracing.
- Meng et al. (2023), "Mass-Editing Memory in a Transformer" (MEMIT). ICLR 2023. arXiv:2210.07229 — Extension to mass editing.
- Zhang & Nanda (2024), "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods." ICLR 2024. arXiv:2309.16042 — *Critical*. Read this multiple times. It is the standard reference for activation patching methodology, including the failure modes you'll encounter.
- Heimersheim & Nanda (2024), "How to Use and Interpret Activation Patching." arXiv:2404.15255 — Practical companion to the above.

**Steering and representation engineering:**

- Zou et al. (2023), "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405 — Foundational paper for the steering line. Cited by Cyberey et al. You must cite and differentiate.

**Faithfulness of reasoning:**

- Turpin et al. (2023), "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting." NeurIPS 2023. arXiv:2305.04388 — The reference for unfaithful chain-of-thought. Directly relevant to Claim 3.
- Lanham et al. (2023), "Measuring Faithfulness in Chain-of-Thought Reasoning." Anthropic. arXiv:2307.13702 — Methodological work on testing rationale faithfulness.

**Probing methodology and its limits:**

- Belinkov (2022), "Probing Classifiers: Promises, Shortcomings, and Advances." Computational Linguistics. — The standard reference on what probing does and doesn't show.
- Hewitt & Liang (2019), "Designing and Interpreting Probes with Control Tasks." EMNLP 2019. arXiv:1909.03368 — Why probing accuracy alone is misleading and how to control for it.

### 4.2 Tier 2 — Read for specific phases

**For Phase 1 (probing analysis):**

- Probing classifier methodology papers above (Belinkov, Hewitt)
- "Linear Probes" survey content from ARENA curriculum

**For Phase 2 (causal localization):**

- Activation patching papers above (Zhang & Nanda, Heimersheim & Nanda)
- "Alignment is Localized: A Causal Probe into Preference Layers." arXiv:2510.16167 — Closely adjacent to your method; you must position against it.

**For Phase 3 (rationale faithfulness):**

- Turpin et al. and Lanham et al. above
- Recent papers on "Breaking the Chain" methodology if available

**For Phase 4 (LoRA perturbation):**

- Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685 — Original LoRA paper.
- Zhu et al. (2024), "Asymmetry in Low-Rank Adapters of Foundation Models." arXiv:2402.16842 — B-matrix matters more than A-matrix; informs your perturbation design.
- Goel et al. (2025), "Learning to Interpret Weight Differences in Language Models." — Directly relevant for framing weight-level interpretability.
- Horwitz, Kahana, Hoshen (2024), "Recovering the Pre-Fine-Tuning Weights of Generative Models." arXiv:2402.10208 — Weight-space attack literature; cite.

**Sparse autoencoders (optional but increasingly standard):**

- Bricken et al. (2023), "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic. https://transformer-circuits.pub/2023/monosemantic-features — Foundational.
- Cunningham et al. (2023), "Sparse Autoencoders Find Highly Interpretable Features in Language Models." arXiv:2309.08600
- Recent work from EleutherAI and Apollo on SAEs at scale — search for late 2024 / 2025 papers.

### 4.3 Tier 3 — Adjacent literature for related work

Papers you cite and differentiate against, but don't need to deeply understand:

**LLM-as-judge auditing (black-box):**

- Doddapaneni et al. (2024), FBI. EMNLP 2024. arXiv:2406.13439
- Abolghasemi et al. (2024), CAUSE. ACL Findings 2024. arXiv:2403.19056
- Acikgoz et al. (2025), TD-EVAL. SIGDIAL 2025. arXiv:2504.19982
- Li et al. (2025), RobustJudge. arXiv:2506.09443
- Eiras et al. (2025), Know Thy Judge. ICLR 2025 Workshop. arXiv:2503.04474
- Wu & Aji (2025), Style Over Substance. COLING 2025
- Feuer et al. (2025), SOS-Bench. ICLR 2025. arXiv:2409.15268
- Chen et al. (2024), Humans or LLMs as Judge. EMNLP 2024
- Raina et al. (2024), Is LLM-as-a-Judge Robust? EMNLP 2024

**White-box auditing collisions (must position against):**

- Cyberey, Ji, Evans (2026), "White-Box Sensitivity Auditing with Steering Vectors." arXiv:2601.16398 — *Critical citation. You cannot claim "white-box auditing" framing.*
- Eshuijs et al. (2025), JUSSA. arXiv:2505.17760 — *Critical citation. They're in steering × LLM-judge space.*

**Reward model interpretability:**

- Christian et al. (2025), "Reward Model Interpretability via Optimal and Pessimal Tokens." FAccT 2025. arXiv:2506.07326
- SAFER (Liu et al., 2025) — SAEs on reward models
- SARM (Zhang et al.), AAAI 2026 Oral. arXiv:2508.08746 — Interpretable RM via SAE
- Chai et al. (2025), Activation Reward Models. arXiv:2507.01368

**Evaluation awareness (uses Llama 3.3 70B, your model):**

- Nguyen et al. (2025), "Probing and Steering Evaluation Awareness." ICML 2025 Workshop. arXiv:2507.01786
- Hua et al. (2025), "Steering Evaluation-Aware Language Models." arXiv:2510.20487

**Model tampering and weight-space attacks:**

- Che et al. (2025), "Model Tampering Attacks." ICML 2025. arXiv:2502.05209 — Establishes weight perturbation as evaluation methodology

**TOD-specific recent work:**

- Ghazarian et al. (2025), TOD-ProcBench. arXiv:2511.15976 — Build on this
- Chalamalasetti et al. (2025), clem:todd. SIGDIAL 2025

### 4.4 Hands-on technique resources

You learn interpretability by doing it, not by reading about it.

**Primary curriculum:**

- **ARENA Mechanistic Interpretability curriculum:** https://arena3-chapter1-transformer-interp.streamlit.app/ — *This is the standard learning path.* Work through it on small models (GPT-2-small).
- **Neel Nanda's interpretability tutorial videos:** YouTube. Search "Neel Nanda transformer mechanistic interpretability."
- **Neel Nanda's 200 Concrete Problems list:** https://neelnanda.io/200 — Useful for finding tractable practice problems.

**Libraries:**

- **TransformerLens:** https://github.com/TransformerLensOrg/TransformerLens — The standard library. Supports GPT-2, Llama, Pythia, and others. Use this for activation patching.
- **nnsight:** https://nnsight.net/ — Alternative library, growing in popularity.
- **SAELens:** https://github.com/jbloomAus/SAELens — For sparse autoencoder work if you go that route.

### 4.5 Community and current debates

The field's discussion happens in specific places. Engage with them.

- **Alignment Forum:** https://alignmentforum.org — Most rigorous interpretability discussion outside paper venues
- **LessWrong** (interpretability tag): https://lesswrong.com/tag/interpretability-ml — More accessible
- **AI Alignment Forum interpretability tag**
- **BlackBoxNLP:** https://blackboxnlp.github.io — Annual workshop, lots of relevant work
- **People to follow on Twitter/X:** Neel Nanda, Chris Olah, Anthropic interpretability team accounts, Apollo Research, EleutherAI accounts, Lawrence Chan, Stephen Casper

**Active debates to be aware of:**

- Whether sparse autoencoders are the right abstraction (vs. probes, vs. raw activations)
- Whether activation patching results actually generalize or are mostly artifacts
- Whether mechanistic interpretability claims have empirical content beyond "we found a pattern"
- The role of mechanistic interpretability in AI safety vs. as basic science

You don't need positions on these debates, but you should know they exist and recognize them when they come up in reviews.

---

## Part 5 — Setup and Infrastructure

### 5.1 The make-or-break feasibility test

**Before doing anything else, run this test:**

In your Capital One environment, can you:

1. Load Llama 3.3 70B with intermediate-layer access enabled?
2. Run a forward pass and capture the residual stream at every layer?
3. Run a second forward pass with a captured activation patched into a specific layer at a specific token position?
4. Verify that patched outputs differ from unpatched outputs in expected ways on a toy task?

If yes to all four: project is feasible. Proceed.

If no to any: the paper does not exist in this form. Either resolve the infrastructure issue or pivot to a different direction (the policy audit paper from earlier discussions remains a viable fallback).

### 5.2 Environment setup checklist

**On your personal MacBook Air (already set up for data prep):**

- Python virtual environment for ABCD data processing (already done)
- GitHub repo (already at https://github.com/michellemenachery/selective-robustness-auditing)
- Existing perturbation pipeline (reusable)

**On the Capital One laptop (model evaluation environment):**

- TransformerLens or nnsight installed
- Llama 3.3 70B weights accessible (via HuggingFace or internal model registry — verify which)
- GPU access with sufficient memory (70B in fp16 needs ~140GB VRAM minimum; activation caching adds substantial overhead)
- Storage for cached activations (a single 70B forward pass at 22 turns × ~500 tokens produces multi-GB of intermediate state if you cache everything)
- LoRA training capability via PEFT library or equivalent

**For SAE work (if you go that route):**

- SAELens or equivalent
- Pre-trained SAEs for your judge models (download if available; train if not — training SAEs at 70B scale is expensive)

### 5.3 The toy-task validation pipeline

Before running real experiments, validate the entire pipeline end-to-end on a toy task. Standard choice: indirect object identification (IOI) on GPT-2-small. Wang et al. (2023) is the reference; the experiment is reproducible and the expected results are well-known.

This catches infrastructure bugs, methodology mistakes, and library issues before they corrupt real experiments. Skipping this step is the most common reason mechanistic interpretability projects waste months.

### 5.4 Compute and time estimates

Order-of-magnitude guidance, not commitments:

- **GPT-2-small experiments:** Minutes to hours per experiment. Iterate freely.
- **Pythia-1.4B / 2.8B experiments:** Hours per experiment. Reasonable iteration speed.
- **Llama 3.3 70B activation patching:** Tens of minutes per dialogue × layers patched. Per-experiment runtimes in the multi-hour to multi-day range depending on coverage.
- **GPT-OSS 120B:** Slower than 70B. Verify infrastructure supports it before committing.
- **LoRA training on 20B model:** Hours to a day or two depending on data scale.
- **SAE training at 70B scale:** Days to weeks if attempted at all. Probably skip unless pre-trained SAEs are available.

### 5.5 Code repository organization

Suggested structure beyond what you have:

```
selective-robustness-auditing/
├── data/                    # Existing ABCD pipeline
├── perturbations/           # Existing perturbation generation
├── interpretability/        # New: all mech-interp code
│   ├── probing/             # Phase 1
│   ├── activation_patching/ # Phase 2
│   ├── rationale_analysis/  # Phase 3
│   ├── lora_perturbation/   # Phase 4
│   └── shared/              # Hooks, model loading, utilities
├── experiments/             # Experiment scripts and configs
├── results/                 # Cached results, plots, tables
├── notebooks/               # Exploratory analysis
└── docs/                    # Write-ups, paper drafts
```

---

## Part 6 — The Six Phases of Execution

### 6.1 Phase 0 — Foundational learning and feasibility verification

**Purpose:** Build technique fluency and confirm the project is executable.

**Components:**

1. **Reading.** Tier 1 papers from §4.1, plus relevant Tier 2 for the techniques you'll use most.
2. **ARENA curriculum.** Work through chapters 1.1-1.5 on small models. Reproduce indirect object identification on GPT-2-small.
3. **Library fluency.** Set up TransformerLens, run example notebooks, understand the hook system.
4. **Infrastructure verification.** Test in §5.1 above on Capital One environment.
5. **First community engagement.** Post on LessWrong or Alignment Forum about your project (lightly), or DM 2-3 interpretability researchers asking for early feedback. Most won't respond; some will.

**Exit criteria:** You can articulate what activation patching is, why it works, and what it does and doesn't tell you. You have run an end-to-end IOI replication on GPT-2-small. You've confirmed Llama 3.3 70B activation access in your environment.

**Common failure mode:** Skipping the foundational reading or the ARENA curriculum because the paper experiments seem more important. This produces shallow paper work where you don't know whether your results are meaningful. Don't skip.

### 6.2 Phase 1 — Probing analysis: Do judges represent criteria as separable?

**Question:** Do judge internal representations distinguish between violation types (W/V/R) as separable directions?

**Method:**

1. Stratified sample of dialogues across compliance categories (clean, W-violated, V-violated, R-violated). Target ~100 dialogues per category for probing training.
2. Run judge on each dialogue. Capture residual stream activations at every layer at the final token position (where the verdict-forming computation concentrates).
3. For each layer, train a linear probe (logistic regression or small MLP) to predict the violation category from the captured representation.
4. Evaluate: at what layer do probes achieve highest accuracy? Is accuracy higher for "violated vs. compliant" (binary) than for "W vs. V vs. R" (4-way)? The gap is your finding.

**Controls:**

- Hewitt-Liang control tasks: train probes on randomly assigned labels to establish what accuracy is "trivially achievable." Subtract this from real probe accuracy.
- Probe complexity ablation: linear vs. MLP. If MLP substantially outperforms linear, the structure isn't linearly accessible.
- Layer-wise analysis: report probe accuracy as a function of layer depth.

**What success looks like:** Binary detection (violated vs. compliant) probes achieve high accuracy at mid-layers. Multi-class (W vs. V vs. R) probes either (a) achieve substantially lower accuracy than binary, or (b) confuse W with V at high rates. Either pattern supports Claim 1.

**What failure looks like:** Multi-class probes achieve accuracy comparable to binary detection. This would mean the judge cleanly represents distinct criteria, undermining Claim 1.

**Key risk:** Probing accuracy alone is not enough. A linear probe can extract information from representations even when that information isn't being used by the downstream computation. Combine with causal probing (Phase 2) to make a meaningful claim — if the probe-detected feature is causally manipulable in a way that flips verdicts, the feature is being used.

**Output for paper:** Layer-wise probing accuracy curves per family + control task baseline + linear vs. nonlinear comparison. Probably 1-2 figures.

### 6.3 Phase 2 — Causal localization via activation patching

**Question:** Which internal computations causally drive verdicts? Are they shared or family-specific?

**Method:**

For each violation family (W, V, R) and each judge model:

1. Pair up clean compliant dialogues with their perturbed counterparts (already done — your existing 500 pairs).
2. For each pair: run judge on clean, capture activations at every layer × every token position. Run judge on perturbed dialogue. For each (layer, token) intervention point, replace the activation with the captured clean one and measure verdict change.
3. Compute "patching effect" = magnitude of verdict change attributable to that intervention.
4. Aggregate across pairs to identify layers/positions with consistent large patching effects.
5. Compare: do W, V, R have the same high-effect layers, or do they differ?

**Controls:**

- Random patching: patch with activations from unrelated dialogues. Should produce small effects.
- Patching at the same layer across multiple seeds: test reproducibility.
- Counterfactual baselines: compare "patching restores correct verdict" effect to "patching causes wrong verdict on clean dialogue."

**Methodological standards:** Follow Zhang & Nanda (2024) "Towards Best Practices" rigorously. Common pitfalls they document include:

- Confusing patching effect with importance (a layer can be necessary but not sufficient)
- Using inappropriate metrics (logit difference vs. probability difference)
- Token-position confounds in autoregressive models
- Failing to distinguish direct vs. indirect effects

**What success looks like:** Patching effects concentrate at specific layers, and those layers differ between families. E.g., W violations patched out at layers 30-35; V violations at layers 40-50.

**What failure looks like:** Patching effects are diffuse across all layers, or the same layers affect all families equally. The first is uninformative; the second supports entanglement (which is also a finding, but a different one).

**Output for paper:** Patching effect heatmaps (layer × position) per family. The flagship figure of the paper.

### 6.4 Phase 3 — Rationale faithfulness via causal mediation

**Question:** When a judge produces a rationale alongside a verdict, does the rationale reflect the verdict-driving computation, or is it post-hoc?

**Method:**

1. For judges that produce structured rationale + verdict (probably the larger judges), identify cases where Phase 2's activation patching reliably flips the verdict.
2. Run the patched intervention. Capture both the verdict and the rationale produced under intervention.
3. Compare rationale-under-intervention to rationale-without-intervention for the same input.
4. Annotate (manually or via classifier): does the intervened rationale "explain" the new verdict, or does it inherit reasoning from the original computation?

**Variant: counterfactual perturbation.** Compute steering directions that flip verdicts. Apply varying magnitudes. Track at what point the rationale starts changing — does it lag behind the verdict (rationale changes after verdict has already flipped, suggesting post-hoc) or anticipate it (rationale changes coherently with verdict)?

**What success looks like:** Rationales after intervention "explain" the new verdict in plausible-sounding ways even when the input dialogue is unchanged. This would replicate Turpin et al.-style unfaithfulness in judge settings.

**What failure looks like:** Rationales remain consistent with the original (pre-intervention) computation despite the verdict change. This would support Claim 3's negation — judges' rationales are causally connected to verdicts.

Either outcome is publishable; you just want a clean finding in some direction.

**Output for paper:** Rationale-faithfulness analysis section. Probably qualitative examples + quantitative summary.

### 6.5 Phase 4 — LoRA adapter weight perturbation

**Question:** When a base model is fine-tuned into a TOD judge via LoRA, is the resulting evaluation function robust or fragile to weight perturbations?

**Method:**

1. Fine-tune a LoRA adapter on top of GPT-OSS 20B (or Llama 3.3 8B if 20B is impractical) using TOD-ProcBench-style training data: dialogue + violation type + correct verdict.
2. Verify the trained judge achieves reasonable accuracy on a held-out test set.
3. Apply systematic weight perturbations:
   - Gaussian noise on adapter weights at varying σ
   - Per-rank scaling: scale rank k of the LoRA decomposition by factor f, for each (k, f) pair
   - Per-rank zeroing: set rank k's contribution to zero
   - A-matrix vs. B-matrix asymmetric perturbations (cf. Zhu et al. 2024)
4. Measure verdict change after each perturbation on held-out evaluation set.
5. Identify high-impact components vs. low-impact components.

**Controls:**

- Compare fragility of LoRA-trained judge vs. base model with prompt-only judging
- Train multiple LoRA adapters with different seeds; check whether high-impact components are consistent across runs
- Test fragility separately on W, V, R violations

**What success looks like:** A small subset of adapter components carries most of the discriminative ability. Perturbing them disproportionately changes verdicts. This supports Claim 4 and gives a deployment-actionable finding.

**What failure looks like:** Perturbation effects are evenly distributed across components, suggesting robust learned representations.

**Output for paper:** Weight-perturbation sensitivity tables and visualizations. Connects to Phase 1-3 narratively as "internal intervention as a method, applied at a different level."

### 6.6 Phase 5 — Cross-dataset replication on τ-bench

**Question:** Do findings replicate on a structurally different TOD benchmark?

**Method:** Take your most important findings — typically Phase 1 (probing) and Phase 2 (activation patching) — and run them on a τ-bench subset (~100 dialogues with newly-built perturbations). Don't replicate every experiment; replicate the headline ones.

**Required pattern:** The same family-level asymmetries (or whatever you found on ABCD) show up on τ-bench. If they do, claims generalize. If they don't, scope to ABCD-style benchmarks honestly.

**Output for paper:** A "cross-dataset replication" section with comparison tables.

### 6.7 Phase 6 — Synthesis and writing

The narrative needs to be sharper than "we did several experiments." Required structure:

> "We asked whether LLM judges' internal computation aligns with the multi-dimensional evaluation criteria they're supposed to apply. Using a battery of mechanistic interpretability techniques on three judge models across two TOD benchmarks, we found [main finding]. This implies [practical or scientific consequence] for how LLM-as-judge should be deployed in policy-grounded settings."

The story is: you applied multiple methods because no single method is conclusive (probing has limits; causal interventions have their own confounds; weight perturbation tells you something different than activation patching). Across methods, a coherent picture emerges. That coherence is the paper.

**Workshop submission first.** Submit to BlackBoxNLP at EMNLP, NeurIPS interpretability workshop, or similar before the EMNLP main submission. Workshop deadlines force crystallization and the feedback substantially improves the main submission.

---

## Part 7 — What Will Probably Go Wrong

Forecastable failure modes. Plan for them.

### 7.1 Methodological failures

**Probing finds structure that isn't being used.** This is the most common probing pitfall. If you stop after probing, your story is fragile. Always combine with causal interventions.

**Activation patching produces noisy results.** Real-world tasks (vs. toy tasks like IOI) typically produce messier patching results than the cleanest published examples. Multiple seeds, multiple metrics, and careful controls are necessary. Some experiments will not produce clean results no matter how careful you are; recognize this and don't force a story.

**Cross-model findings don't transfer.** Different model families have different internal organizations. If your story depends on cross-model replication, you may find the patterns differ. Scope claims appropriately.

**Rationale faithfulness ambiguity.** Determining whether a rationale "explains" a verdict requires interpretation. Multiple annotators with adjudication is necessary. Inter-annotator agreement may be low and the bar for "rationale changed" can be subjective.

### 7.2 Infrastructure failures

**Activation access is gated.** If Capital One restricts intermediate layer access on certain models, you may be limited to subsets. Verify before committing.

**Memory issues with 70B models.** Activation caching at scale produces very large files. Disk and memory budgets need real planning.

**Reproducibility issues.** Mechanistic interpretability experiments are sensitive to seeds, library versions, and numerical precision. Pin everything; expect to chase down inconsistencies.

### 7.3 Project-management failures

**Three months on infrastructure, one on findings.** The most common failure mode for solo interpretability work. Mitigate by validating end-to-end pipeline on toy task before scaling up.

**Two halves of paper feel disjoint.** Activation work and LoRA weight work need a connecting narrative. Write the abstract early and revise it constantly to keep both halves serving the same claim.

**Foundational reading deferred indefinitely.** "I'll read the canonical papers later, after I get experiments running." This produces shallow work. Front-load the reading.

**Community engagement deferred.** Publishing in interpretability without engaging with the community produces work that doesn't fit field norms. Engage early, even informally.

### 7.4 Scientific failures

**Findings come out negative or null.** Plausible outcome: probing achieves high accuracy on multi-class violation prediction (no entanglement); activation patching reveals nothing surprising about layer structure; rationales are roughly faithful; LoRA adapters are robust. In this world, the paper is "we looked for evidence of judge unfaithfulness and didn't find it, here's the negative result." Publishable but a different paper. Plan for this contingency.

**Results too clean to trust.** If everything works perfectly the first time, be suspicious. Look for confounds, leakage, or methodological errors. Real interpretability findings are usually messy.

---

## Part 8 — Community Engagement Plan

Don't treat this as optional. The interpretability community is small enough that participating is feasible and large enough that engagement materially helps your work.

### 8.1 Throughout the project

- Read recent papers on Alignment Forum and arXiv weekly. Skim, don't deep-read everything.
- Follow major interpretability researchers on Twitter/X. Saves time vs. searching for new work.
- Lurk on relevant Discord servers (TransformerLens, EleutherAI, etc.).

### 8.2 Specific touchpoints

**Phase 0 (foundational):** DM 2-3 interpretability researchers describing your project briefly, asking for early feedback or pointers to related work. Most won't respond. Some will. Even non-responses tell you things.

**Phase 1-2 (early findings):** Post a short writeup on LessWrong or Alignment Forum describing preliminary results. Frame as "early-stage work, looking for feedback." Usually generates useful comments.

**Phase 4-5 (consolidation):** Submit to BlackBoxNLP workshop or NeurIPS interpretability workshop. The deadline is the forcing function; the feedback is the value.

**Phase 6 (writing):** Share draft with 2-3 people in the field for prereview before EMNLP submission. This is normal in interpretability and substantially improves submissions.

### 8.3 What to avoid

- Don't oversell preliminary results in public posts. Hedging early helps credibility.
- Don't ignore disagreements. If a community member says your method is wrong, engage seriously rather than defensively.
- Don't fake fluency. The field rewards honest "I'm new and learning" framing more than performed expertise.

---

## Part 9 — Writing the Paper

### 9.1 Structure (for EMNLP main, 8 pages + unlimited refs + appendix)

1. **Introduction (1 page).** The mechanistic question, the methodological framework, the headline finding, the practical implication.
2. **Related Work (0.75 pages).** Position against Cyberey, JUSSA, Alignment-is-Localized, SAFER/SARM, FBI, TD-EVAL, TOD-ProcBench. The "white-box" framing collision must be addressed explicitly.
3. **Methods (1.5 pages).** Datasets, perturbation taxonomy (with TOD-ProcBench citation), three levels of intervention, evaluation protocol.
4. **Experiments and Results (4 pages).** Phase 1 probing, Phase 2 patching, Phase 3 rationale faithfulness, Phase 4 LoRA perturbation, Phase 5 cross-dataset.
5. **Discussion (0.5 pages).** Mechanistic interpretation, limitations, scope of claims.
6. **Conclusion (0.25 pages).**

Appendix: full perturbation examples, probing controls, patching reproducibility analysis, LoRA training details, rationale annotation guidelines.

### 9.2 The headline figure

Phase 2's per-family layer-wise patching heatmap is the iconic figure if findings land. Color-coded by family, organized to make the asymmetry visually immediate.

### 9.3 The headline table

A summary across techniques: probing accuracy, patching effect, rationale faithfulness, LoRA fragility — per family, per model. One table that captures the multi-method evidence converging on a single picture.

### 9.4 The one sentence

Write the paper to support a single sentence. Candidate:

*LLM judges in policy-grounded TOD evaluation maintain entangled internal representations of distinct compliance criteria, produce verdicts driven by features that don't cleanly correspond to those criteria, and generate rationales that are post-hoc rather than faithful — across multiple model scales and benchmark datasets.*

If the experiments don't support this sentence, find the sentence they do support. Don't bend evidence to fit the original target.

### 9.5 Writing hazards

- **"White-box auditing" framing.** Burned by Cyberey. Don't lead with it.
- **Overclaiming generalization.** The paper is about specific judges in specific settings. Reviewers punish unscoped claims.
- **Underciting the close-neighbor papers.** Cite Cyberey, JUSSA, Alignment-is-Localized, SAFER, SARM, Model Tampering, TOD-ProcBench. Each one is a potential reviewer.
- **Treating LoRA work as separate paper.** The unifying narrative — internal intervention as auditing primitive — must be carried explicitly.

---

## Part 10 — Decision Points and Off-Ramps

Honest acknowledgment: this paper is not guaranteed to work. Predefined off-ramps reduce sunk-cost reasoning if you hit walls.

### 10.1 Phase 0 off-ramp

If activation access on Llama 3.3 70B is not feasible in Capital One environment: pivot to the policy audit paper (separate direction, doesn't require model internals). Don't try to make it work with workarounds that will haunt the paper later.

### 10.2 Phase 1 off-ramp

If probing analysis shows judges cleanly represent distinct violation criteria as separable directions: Claim 1 collapses. The paper pivots to a different story — "judges represent criteria distinctly; verdicts fail anyway, why?" Still publishable but a different paper.

### 10.3 Phase 2 off-ramp

If activation patching produces uninformative results (effects either nonexistent or impossible to localize): retreat to a probing-only paper, narrower in scope, possibly Findings-tier rather than main track.

### 10.4 Phase 4 off-ramp

If LoRA fine-tuning is infeasible in your environment, or if results are uninteresting: drop Phase 4. The paper is still publishable on activation work alone, especially if Phases 1-3 land.

### 10.5 Whole-paper off-ramp

If by Phase 3 the findings haven't converged into a coherent story, consider a workshop-only submission. This isn't failure — interpretability papers often start as workshop work and grow into main-track submissions on the next iteration.

---

## Part 11 — What You Need to Be Aware Of (Beyond the Technical)

Honest things about doing this work that aren't in any methods paper.

### 11.1 The technique fluency gap

The gap between "I've read papers about activation patching" and "I can do activation patching reliably" is large. Larger than in most NLP subfields. Plan for the ramp-up. Three months is enough to learn techniques to publication-paper level. It is not enough to become fluent. Treat this paper as the first step of a longer arc.

### 11.2 Norms about claims

Mechanistic interpretability has strong norms about precision. Sweeping claims get critiqued sharply. Papers report specific narrow findings with explicit caveats. Read recent discussion sections to calibrate.

### 11.3 The signal-to-noise ratio

Real mechanistic interpretability research produces messier results than the cleanest published examples. You will often face the question "is this finding signal or am I p-hacking?" Multiple seeds, multiple metrics, control conditions, and pre-registered analysis plans all help. So does talking with others in the field.

### 11.4 Solo work is harder here than in NLP generally

Mechanistic interpretability benefits from collaborative debugging more than other empirical NLP work. Patching results that look strange often turn out to be infrastructure bugs or misunderstood techniques rather than findings. Without a collaborator, you'll catch fewer of these. Compensate with community engagement, careful pipeline validation, and toy-task verification.

### 11.5 Negative results

The field is comparatively friendly to negative results — "we looked for X and didn't find it" is publishable. This lowers the stakes on every individual experiment. Use this latitude rather than forcing positive results.

### 11.6 The career move beneath the paper

You said you want to learn the field, not just publish a paper. The career-shaping moves are:

- Becoming someone who can do this work fluently (multi-year)
- Building relationships with people in the field (compounds)
- Producing work that gets cited by mech-interp researchers (feedback loop into more opportunities)

The paper is the visible output. The skill, network, and credibility are the durable capital. Optimize for the durable capital and the visible output mostly takes care of itself.

### 11.7 The thing nobody says explicitly

Mechanistic interpretability work often doesn't generalize the way authors hope. Findings on one model don't replicate on another. Specific circuits identified in one task don't appear in adjacent tasks. The field is in an early stage where we don't yet know which findings are durable and which are brittle. Write papers that are honest about this rather than claiming more than you should.

---

## Part 12 — Quick-Reference Summary

**Goal:** Mechanistic audit of LLM judges' internal representations of TOD compliance criteria.

**Datasets:** ABCD (primary), τ-bench (validation), TOD-ProcBench taxonomy as scaffolding.

**Models:** Llama 3.3 70B, GPT-OSS 120B, GPT-OSS 20B (Capital One access).

**Methods:** Probing classifiers, activation patching, causal mediation analysis on rationales, LoRA weight perturbation.

**Three perturbation levels:** Input (existing pipeline), activation (new methodology), weight (new for LoRA judge).

**Critical positioning:** Lead on "causal faithfulness of judge representations to multi-dimensional evaluation criteria." Don't lead on "white-box auditing" — Cyberey owns that.

**Phases:** 0 foundational learning + feasibility, 1 probing, 2 activation patching, 3 rationale faithfulness, 4 LoRA perturbation, 5 cross-dataset, 6 synthesis.

**First action:** Verify Llama 3.3 70B activation access in Capital One environment. If yes, project is viable. If no, pivot.

**Reading priority:** Tier 1 papers in §4.1. Don't skip. ARENA curriculum in §4.4. Reproduce IOI on GPT-2-small before serious paper work.

**Community:** Engage from Phase 0. Workshop submission before main submission. Get prereviews before EMNLP.

**Honesty about risk:** This is harder than typical NLP empirical work. Plan for messy results, infrastructure friction, and a real ramp-up cost. The skill and community capital are the durable returns; the paper is the visible output.

---

*End of execution guide.*
