# Synthesis — a combined architecture from the nine proposals

This document proposes **one architecture that takes the strongest idea from each collected proposal**, names it, says where it came from, and states the design decision it forces. It is written to be implementable, not contemplative: every block below has a responsibility, an input and an output. Where two proposals conflict, the conflict is stated and a choice is made with its cost.

Attribution is by family: **Apodex**, **Cohere**, **Dots**, **InclusionAI**, **Liquid**, **NVIDIA-L** (nemotron-3.5-lightning), **NVIDIA-S** (nemotron-3-super), **NVIDIA-U** (nemotron-3-ultra), **Poolside**. Raw proposals are in `raw_outputs/`; per-model component lists in `comparison.csv`.

---

## 0. The design in one paragraph

A perceptual front end that predicts its own next input feeds **two coupled representational substrates** — a probabilistic world model and a symbolic causal graph — kept honest against each other by a mutual-information check. Memory is split four ways (episodic, semantic, procedural, working) with an explicit **forgetting** mechanism, and promotion between stores is gated by verification. Actions are verified **three times**: statically before execution (formal proof over skill pre/post-conditions), in a sandbox before execution (simulated rollout), and dynamically after execution (prediction residual, logical consistency, cross-modal consensus). All error signals collapse into one scalar reward **plus a meta-reward for successful self-correction**. A meta-cognitive layer observes error rates per component, adjusts thresholds, and decides when to spend compute on deliberation. Nothing symbolic is ever stored without a trace back to the signal that produced it.

## 1. Perception and grounding

| Component | Responsibility | From |
|---|---|---|
| **Sensor Fusion Engine** | Normalise heterogeneous channels into a common latent | Cohere |
| **Predictive Encoder** | Encode, but also predict the *next* latent; residual is the primary error signal | NVIDIA-S, NVIDIA-L |
| **Grounding Map** | Maintain the bidirectional trace between latent state and originating signal | Cohere |
| **Cross-Modal Consensus** | Channels must agree; disagreement raises a flag rather than being averaged away | NVIDIA-L |

**Design decision:** the encoder is *predictive*, not passive. Making the front end predict its own input turns perception into a continuous error source instead of a one-way pipe — three separate families landed on this, and it costs nothing extra to implement. Averaging conflicting modalities is rejected: consensus-checking surfaces sensor failure whereas averaging hides it.

## 2. Representation — two substrates, deliberately coupled

| Component | Responsibility | From |
|---|---|---|
| **Probabilistic World Model (PWM)** | Maintain competing causal hypotheses over future state; emit predictions with uncertainty | InclusionAI, Poolside |
| **Causal Model Store / Symbolic Graph** | Concepts and rules as a weighted graph with FOL constraints | NVIDIA-S (CMS), InclusionAI (SAL), Liquid (KGM) |
| **Coherence Meter** | Mutual information between PWM predictions and graph causal rules; low value triggers a consistency audit that re-weights edges | Cohere |

**The strongest idea in the sample** is InclusionAI's: keep **two competing causal hypotheses** in the world model and use the **KL divergence between them** as a first-class signal. Uncertainty stops being a scalar bolted onto a prediction and becomes a *disagreement between explanations*. A single-hypothesis world model cannot express "I don't know why", only "I'm unsure"; competing hypotheses can, and the divergence tells the meta-layer exactly when deliberate reasoning is worth its cost.

**Design decision:** dual substrate is kept, not collapsed into one. Graph-only designs (NVIDIA-S) cannot represent graded prediction; vector-only designs cannot express hard constraints. The cost is real — two representations must be kept in sync — and it is paid by making the Coherence Meter a monitored quantity rather than an assumption.

## 3. Memory — four stores, explicit forgetting, gated promotion

| Store | Holds | From |
|---|---|---|
| **Episodic** | What happened, timestamped, with salience tags | Cohere (EMS), Poolside (ES), NVIDIA-U (EB) |
| **Semantic** | What is true, as verified relations | Cohere (SMI), Liquid (KGM) |
| **Procedural** | What works — reusable skills | Cohere (PMB), NVIDIA-U (SL) |
| **Working** | Active context, with an attention allocator | Poolside (WMB), NVIDIA-S (WM), Cohere (AA) |
| **Forgetting & Consolidation** | Promotion, decay, and pruning | **NVIDIA-S (FCM)** |

**Design decision — forgetting is a component, not a side effect.** NVIDIA-S was the only proposal that gave forgetting a name and a mechanism, and it is the right call: an architecture that only accumulates memory degrades into an unsearchable store of stale facts. Concretely: high-salience episodes consolidate episodic → semantic via contrastive replay (InclusionAI); graph edges are re-weighted or pruned using **causal intervention tests** (InclusionAI) rather than by age; a verified novel skill sequence is distilled into one fused skill (NVIDIA-U, DreamCoder-style program synthesis over episodic traces).

**Promotion gate:** nothing is written to semantic memory at write time. Episodes enter with provenance and an *unverified* status; only items that survive verification are promoted. This directly answers the failure mode that more than one proposal named — an error written to memory becomes a permanent false premise because the learning loop later treats stored experience as ground truth.

## 4. Reasoning and planning

| Component | Responsibility | From |
|---|---|---|
| **Hierarchical Planner** | Long-horizon decomposition into subgoals | Poolside (HRL), NVIDIA-U (PL) |
| **Skill Library** | Executable skills with pre/post-conditions | NVIDIA-U (SL) |
| **Symbolic Reasoner** | Constraint-satisfying inference over the graph | Apodex, Liquid (URE), NVIDIA-S (SRE) |
| **Skill Composer** | Fuse verified novel sequences into a single new skill | NVIDIA-U |

**Design decision:** planning runs over *skills with contracts*, not over raw actions. This is what makes the static verification in §5 possible at all — a plan is checkable only if its steps declare what they require and guarantee.

## 5. Verification — three checkpoints, cheap-first

This is the part where combining proposals beats any single one, because the nine families each verified at a *different moment*:

| Checkpoint | Mechanism | From | Cost |
|---|---|---|---|
| **Pre-execution, static** | Prove `Pre(S₁) ∧ ⋀Post(Sᵢ) ⟹ Pre(Sᵢ₊₁) ⟹ Goal` with an SMT solver (Z3); fail fast on type mismatch | **NVIDIA-U** | High, but only on the plan |
| **Pre-execution, simulated** | Roll out proposed action in a sandbox; abort on divergence | **Liquid (VGM)** | Medium |
| **Post-execution, predictive** | Residual between predicted and observed; trigger recalibration past a threshold | Poolside (2σ), NVIDIA-L, NVIDIA-S, Apodex | Low, always on |
| **Post-execution, logical** | SAT/FOL scan of the *active* knowledge subset | NVIDIA-S, InclusionAI | Medium |
| **Post-execution, cross-modal** | Independent channels must agree | NVIDIA-L | Low |

**Design decision — tiered verification, because verification cost is itself the most-named failure mode.** Two families independently admitted their design breaks when the critic is skipped under compute pressure. A single mandatory verifier is therefore rejected: the checks are ordered by cost, the cheap ones always run, and formal proof is reserved for **irreversible actions** — the only place where its cost is justified.

**The unified error signal** combines NVIDIA-S's scalar `r = −(λ₁·residual² + λ₂·conflict)` with Cohere's **meta-reward for successful self-correction**. That addition matters: without it, the cheapest way to score well is to avoid situations where errors could be detected. Paying the system for *catching and fixing* its own errors is what keeps the verifier from being optimised away.

## 6. Meta-cognition

| Component | Responsibility | From |
|---|---|---|
| **Meta-Cognitive Monitor** | Track error rates per component; hold the calibration record | Cohere (MCM), InclusionAI (MM), Poolside (SMS) |
| **Meta-Controller** | Adjust thresholds; decide when to escalate to slow deliberation | Liquid (MCC), NVIDIA-U (MC) |
| **Meta-Learner** | Learn the learning policy itself (which updates, when) | NVIDIA-S (ML), Poolside (MLC) |

**Design decision:** the meta-layer *observes and tunes*, and only the Meta-Controller may *halt*. Three proposals placed meta-cognition at different levels (observer, controller, learner); all three functions are kept but separated, because a single module that both observes and acts on its own observations has no independent reference to detect its own drift.

## 7. The learning loop, as four signals

1. **Extrinsic** — task reward.
2. **Intrinsic surprise** — prediction residual (§5, post-execution).
3. **Logical conflict** — constraint violations (§5, logical).
4. **Meta-reward** — awarded specifically for a successful self-correction (Cohere).

Update order: cheap parametric updates on high-residual events immediately (Poolside); batch consolidation and skill distillation periodically; promotion between memory stores only after verification passes.

## 8. What this design must survive — the four declared failure modes

Across the sample, the same four failures were named independently. Each is a failure of a block that *all* designs include, which is why convergence on architecture also means convergence on weak points. The mitigations above map onto them one-to-one:

| Declared failure | Mitigation in this design |
|---|---|
| **World-model drift** — model diverges from reality faster than it corrects | Predictive encoder + thresholded recalibration + cross-modal consensus; the drift is measured, not assumed away |
| **Memory contamination** — an error stored becomes a permanent premise | Provenance on every write; promotion gated by verification; forgetting/consolidation as a first-class component |
| **Verification cost** — the critic is skipped under pressure, silently removing correction | Tiered verification; cheap checks always on; meta-reward for self-correction so skipping is not free |
| **Grounding collapse** — internally consistent, externally meaningless | Grounding Map trace on every symbolic item; SAL validated against first-order constraints before action gating |

## 9. Rejected, and what rejecting costs

- **Single-substrate representation** (graph-only or vector-only): rejected — cheaper, but cannot express graded prediction or hard constraints respectively. Cost of rejection: two representations to keep coherent.
- **Mandatory formal verification on every action**: rejected — correct in principle, but it is precisely the "verification cost" failure mode in disguise. Cost of rejection: some reversible-but-wrong actions slip past the SMT gate.
- **Single undifferentiated memory**: rejected — and notably, **none of the nine proposals proposed one**, which is itself evidence that the split is a requirement rather than an optimisation.
- **Meta-cognition as one module**: rejected — see §6.

## 10. Honest scope

This synthesis is a *combination of proposals*, not a validated design. Its strongest claims — that competing causal hypotheses outperform a single uncertainty scalar, that a self-correction meta-reward prevents verifier erosion, that tiered verification dominates uniform verification — are **hypotheses suggested by convergence among nine cost-zero open-weight models**, not results. None was tested. Testing them requires an implementation and an evaluation harness, which this packet does not provide.

The convergence itself is evidence about *these models*, and reasonably suggests a broader attractor. It is not proof of one.
