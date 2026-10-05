# Summary — Convergent patterns across AI-generated AGI architectures

**Sample:** 9 valid responses, from 7 distinct model families (Apodex, Cohere, Dots Studio, InclusionAI, Liquid, NVIDIA ×3, Poolside), collected 2026-10-05.
**Method:** one identical prompt, no per-model tailoring. Raw outputs unedited in `raw_outputs/`. Frequencies in this document were computed mechanically by script over the extracted component lists (see `comparison.csv`), not estimated by reading.

---

## 1. The main finding

**Independent model families converge on the same coarse architecture.** Asked to design an AGI architecture from scratch, with no shared template, seven unrelated model families produced the same six functional blocks. The names differ; the *structure* does not.

| Functional block | Models containing it | Share |
|---|---|---|
| Perception / encoder / grounding | 8 / 9 | 89% |
| World model | 7 / 9 | 78% |
| Reasoning / planner | 7 / 9 | 78% |
| Verification / critic / monitor | 7 / 9 | 78% |
| Action / executor | 6 / 9 | 67% |
| Memory | 6 / 9 | 67% |
| Attention / control | 6 / 9 | 67% |
| Meta-cognition / meta-learning | 6 / 9 | 67% |

Every single response (9/9) contains at least a perception block, a reasoning or world-model block, and an action or planning block. No response proposed a monolithic design without internal modular decomposition.

## 2. The convergence is stronger than the block list

The blocks are the surface. The **error-correction mechanism** is the real convergence, and it is more surprising:

| Verification mechanism | Models | Share |
|---|---|---|
| Prediction mismatch / prediction error | 7 / 9 | 78% |
| Uncertainty / calibration signalling | 7 / 9 | 78% |
| Consistency check | 5 / 9 | 56% |

Seven of nine independent proposals correct themselves by the *same* principle: **compare what the world model predicted against what actually happened, and act on the discrepancy.** They name it differently — "predictive mismatch" (Apodex), "KL divergence between competing hypotheses" (InclusionAI), "mutual information between WM predictions and causal rules" (Cohere), "S1/F3 discrepancy operator" (NVIDIA Ultra) — but the underlying loop is identical, and in every case it is what triggers learning, halting, or re-planning.

**Interpretation:** prediction-error-driven correction is not one design choice among many. Across this sample it behaves like an attractor — the design that independently reinvented itself most often.

## 3. Memory: a near-universal three-way split

| Memory type | Models | Share |
|---|---|---|
| Episodic | 8 / 9 | 89% |
| Semantic | 7 / 9 | 78% |
| Procedural | 6 / 9 | 67% |
| Vector-indexed | 6 / 9 | 67% |
| Graph-structured | 5 / 9 | 56% |
| Working memory / buffer | 5 / 9 | 56% |

The episodic/semantic/procedural trichotomy is reproduced almost universally, and it is the single most consistent element in the whole sample (better than the world model itself). Notably, **none of the nine proposed a single undifferentiated memory store.** The separation of *what happened* from *what is true* from *what works* appears to be treated as a requirement, not an optimisation.

## 4. Where they genuinely diverge

Convergence on structure does not mean consensus on implementation. The real disagreements:

- **Representation.** Split between graph-structured (NVIDIA Super, Liquid), vector-indexed (Poolside, Cohere), and hybrid designs. 5/9 name graphs, 6/9 name vectors; several use both without resolving the tension.
- **Degree of formalism.** NVIDIA Ultra proposes machine-checkable verification (Z3 pre/post-condition proofs over skill sequences) — the only response with a formal proof obligation. Dots Studio proposes four components and stops. The gap between the most and least formal response is roughly an order of magnitude in specification depth (the shortest valid output is 3.7k chars; the longest 10.3k).
- **Where meta-cognition sits.** Six responses include a meta-level, but they disagree on whether it observes (Meta-Cognitive Monitor), controls (Meta-Cognitive Controller), or learns across tasks (Meta-Learner). Two designs place meta-cognition inside the critic; two place it as a peer of the executive.
- **Grounding strategy.** Ranges from explicit sensor-fusion pipelines (Cohere SFE, Liquid PWM) to symbolic abstraction layers that assume grounded input (InclusionAI, NVIDIA Super).

## 5. Failure modes the models declared about their own designs

Responses were asked what would concretely break their architecture. Recurring admissions cluster into four classes:

1. **World-model drift** — the model diverges from reality faster than it can correct, and every downstream decision inherits the error.
2. **Memory contamination** — an error written to episodic memory becomes a permanent false premise, because the learning loop treats stored experience as ground truth.
3. **Verification cost** — the critic is expensive; under compute pressure it gets skipped, which silently removes the only corrective mechanism (stated by more than one response as the failure mode it fears most).
4. **Grounding collapse** — symbolic reasoning over ungrounded abstractions produces answers that are internally consistent and externally meaningless.

The interesting property: these are **the same four failures named by different families**, and each one is a failure of a *block that all nine designs include*. The designs converge on the same components and, consequently, on the same weak points.

## 6. What this sample can and cannot support

**Supports:** the claim that coarse architecture converges across independent model families; that prediction-error correction is the dominant self-correction principle; that the episodic/semantic/procedural split is near-universal; and that the declared failure modes cluster around the shared blocks.

**Cannot support:** any claim about *frontier* models. This sample contains open-weight, cost-zero variants accessible without payment. No closed flagship model (GPT, Claude, Gemini, Grok) is represented, because none was reachable at no cost under the collection constraints. Treat the convergence as a property of this sample — it is suggestive of a broader attractor, not proof of one.

**Does not support:** quality ranking. The prompt asked for a design, not for the best design; length and formality here measure specification depth, not correctness. No response was scored, and no response was edited.

---

*All numeric frequencies on this page were produced by `scripts/analyze.py` over `comparison.csv` and can be regenerated from the raw outputs.*
