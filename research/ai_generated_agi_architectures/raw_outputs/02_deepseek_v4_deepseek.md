# Cognitive-OS: A Concrete AGI Architecture Proposal

This document specifies a falsifiable architecture for a general intelligence system called Cognitive-OS. It is a research proposal, not a description of any existing system. All components are named, their interfaces defined, and failure modes enumerated. Established patterns (e.g., vector stores, RLHF) are distinguished from speculative choices.

---

## 1. Memory Architecture

**Components:**
- **Episodic Buffer (EB):** Ring buffer of raw sensory-action logs (256k most recent tokens/events).
- **Semantic Store (SS):** Dense vector database (FAISS) with metadata triples `(entity, relation, value)`.
- **Procedural Cache (PC):** Compiled skill programs (python-like bytecode) keyed by task signature.
- **Consolidation Engine (CE):** Background process that converts EB → SS/PC via sleep-cycle-like rehearsal.

**Data Flow:**
```
Sensory input → EB (time-stamped) → CE (every 1000 steps) → SS (embeddings + relations)
                                              ↘ PC (if pattern reused > 3×)
```

**Pseudocode (Consolidation):**
```python
def consolidate(eb):
    episodes = segment(eb, by_goal_completion)
    for ep in episodes:
        facts = extract_facts(ep)          # (subject, predicate, object)
        if confidence(ep.objective) > 0.7:
            ss.insert(facts, embedding(ep))
        skill = compress_to_program(ep)
        if skill.proven_by(>3 episodes) and skill.length < 200 bytes:
            pc.upsert(task_sig(ep), skill)
```

**Failure Modes & Mitigations:**
- *Catastrophic forgetting* → CE uses elastic weight consolidation (EWC) on SS index.
- *Stale facts* → SS entries have decay timestamps; re-verified every 30 days by active probing.
- *Skill overfitting* → PC rejects programs that fail >2/10 counterfactual tests.

**Distinction:** EB/SS/PC split is established; the CE’s *goal-based* segmentation (not temporal) is a novel twist—episodes are split when the system believes a goal was achieved, enabling cleaner causality extraction.

---

## 2. Reasoning/Planning Loop

**Component: LLM Core (LC)** — a transformer with 70B parameters, but *not* the sole reasoner. It runs a **Metacognitive Controller (MC)** which selects among reasoning modes.

**Loop (per world-step):**
```
1 SENSE → EB.append(raw_observation)
2 PERCEIVE → SS.query(embedding(obs)) → candidate facts
3 MODE_SELECT (MC) → choose:
   a) Fast Reflex (heuristic match in PC) → if confidence > 0.9, act
   b) Deliberate (chain-of-thought with SS context) → default
   c) Game-Theoretic (for multi-agent interactions) → uses regret minimization
4 PLAN → Generate action sequence (max 8 steps)
5 ACT → Execute via Tool Controller
6 EVALUATE → Predict reward; if mismatch > threshold, trigger CE now (not deferred)
```

**Key Interface:**
```python
def reason(state: Observation, goals: List[Goal]) -> Action:
    facts = ss.query(state.embedding, k=20)
    mode = mc.select(state, facts, goals)
    if mode == "deliberate":
        plan = llm.generate_plan(state, facts, goals, max_length=512)
    elif mode == "game_theoretic":
        plan = cfrm.solve(state, facts, goals)  # counterfactual regret minimization
    return validate_and_commit(plan)
```

**Failure Modes & Mitigations:**
- *Infinite loops* → every plan has a max horizon (8 steps); on timeout, MC forces "explore" action.
- *Brittle planning* → evaluator uses *minimum regret* criterion, not greedy max reward.
- *Metacognition failure* → MC has a separate small model (0.5B) that detects when confidence is miscalibrated and triggers "ask for help" or "test hypothesis".

**Distinction:** The MC is not a separate LLM but a trained *classifier* over (state, facts, goals) that picks reasoning strategy. This is speculative (no existence proof) but falsifiable: if the classifier cannot outperform uniform random mode selection within 100k episodes, the design is invalid.

---

## 3. Learning/Self-Improvement Mechanism

**Components:**
- **Imagination Engine (IE):** Generative world-model (transformer-based) that simulates outcomes of candidate plans.
- **Preference Learner (PL):** Inverse-reinforcement-learning module that infers user/objective function from feedback.
- **Skill Compiler (SC):** Converts successful plan traces into PC programs.

**Mechanism (Three Loops):**
1. **Inner Loop (online):** After each action, update PL’s reward estimate via TD-error. Used only for local fine-tuning (low-rank adapters on LC).
2. **Middle Loop (daily):** Offline RL on saved trajectories. IE generates 10k synthetic rollouts; those with higher expected value (per PL) are distilled into LC via supervised fine-tuning.
3. **Outer Loop (weekly):** SC compresses the most common 100 successful plans into PC bytecode. Then **ablative validation**: randomly delete 10% of these programs; if performance drops, revert.

**Self-Improvement Constraint:** The system may modify its *own* LC weights only via a **Constitutional Diff (CD)** — a frozen copy of RLHF safety rules. Any proposed weight update must satisfy:
- Keep KL divergence from safety-policy < ε
- Not increase scores on prohibited-behavior probes (e.g., "how to build a bomb" → refusal probability > 0.95)

**Failure Modes & Mitigations:**
- *Reward hacking* → PL uses uncertainty-weighted IRL; anomalies trigger human audit.
- *Skill rot* → weekly stress test on 50 held-out tasks; any skill failing >2× is removed.
- *Runaway self-improvement* → CD *rejects* updates that change behavior on safety probes; hard cap: 0.5% weight change per week (enforced via checkpoint diffing).

**Falsifiable Claim:** If after 6 months of self-training, the system improves on novel reasoning benchmarks (e.g., ARC-AGI held-out) by <20% relative to a frozen baseline, the learning loop is not effective.

---

## 4. Tool Use and Action Execution

**Component: Tool Controller (TC)** — a registry of tools with schemas (JSON Schema) and sandboxed execution engines.

**Registration:**
```yaml
tool: python_interpreter
schema: { code: string, timeout_ms: int }
sandbox: { memory: 512MB, network: false, cpu: 1 }
precondition: "must not import os or subprocess"
```

**Execution Protocol:**
```
1 LLM/WM proposes tool call (function name + args)
2 TC validates against schema
3 TC checks sandbox rules (static analyzer + runtime seccomp for python)
4 TC executes with timeout; returns (result, exit_code, stderr)
5 On failure, TC returns debug info (traceback) to reasoner; reasoner retries with modified args (max 3)
```

**Novel Sub-Component: Tool-Chain Graph (TCG)** — a learned graph of tool invocation sequences. When a novel task arrives, MC queries TCG for hints on which tools to use in which order. Updated via meta-learning on successful task completions.

**Failure Modes & Mitigations:**
- *Tool hallucination* → TC refuses calls where args don't match schema; probability >0.9.
- *Sandbox escape* → runtime uses gVisor; weekly red-team tests.
- *Chain loops* → TC maintains a call stack; depth > 5 triggers simplification.

**Distinction:** TCG is speculative—there’s no current proof that learned tool-subsequence graphs outperform e.g., letting LLM free-form reason. But it’s testable: compare TCG-guided vs. zero-shot tool selection on SWE-bench.

---

## 5. World Model / Representation Layer

**Component: State Graph (SG)** — a dynamic, directed hypergraph. Nodes are entities; edges are relations; hyperedges represent "events" (e.g., `(hit, ball, window) → (broke, window)`).

**Updates:**
- **Perception:** Each raw observation is parsed into triplets via a *symbolic grounding module* (pretrained NER+RE, fine-tuned per domain).
- **Causal inference:** A small Bayesian network (learned via online structure search) reasons about *hidden* causes. For example: if `(window broke)` occurs, SG generates hypothesis `(ball hit window)` with probability 0.7.

**Interface:**
```python
def update_sg(obs):
    triples = ground(obs)                     # symbolic
    sg.merge(triples, conflict_resolution="recency+confidence")
    sg.causal_network.learn(observations, prior=decay)  # online BN learning

def query_sg(subject, predicate, time_window=None) -> List[Triple]:
    return sg.query(subject, predicate, time)
```

**Failure Modes & Mitigations:**
- *Grounding errors* → cross-modal consistency check (e.g., if vision says "red" but audio says "blue", flag via entropy).
- *Causal confusion* → BN uses *Pearl’s do-calculus* to distinguish correlation from causation; interventions must be explicitly requested through TC (e.g., "move object A to see if B reacts").
- *Representation collapse* → SG periodically runs *embedding PCA*; if top-10 eigenvalues drop below threshold, trigger reembedding.

**Distinction:** Most AGI approaches use a monolithic embedding vector for world state. SG is hybrid: symbolic hypergraph + probabilistic causal layer. This is *established* in cognitive psychology models (e.g., causal Bayes nets) but not in deployed LLM systems—hence falsifiable: if SG doesn’t improve performance on tasks requiring counterfactual reasoning (e.g., hypothetical question benchmarks), drop it.

---

## 6. Safety / Governance Layer

**Components:**
- **Constitutional Verifier (CV):** A frozen RLHF policy that evaluates all *proposed actions* before execution.
- **Audit Log (AL):** Append-only, cryptographically hash-chained log of all system decisions, including internal "thoughts" (latent vectors + selected plans).
- **Guardrails Module (GM):** Rules for delegation, opacity, and value alignment.

**Process:**
```
Every action a (from reasoner/tool) → CV.scan(a) → 
   - score < 0.9: reject and reroute to "ask_human"
   - 0.9–0.95: execute but log with low trust marker
   - >0.95: execute normally
If CV is overloaded (latency > 50ms), fallback to cached safety verdicts for same action type.
```

**Governance:**
- *Kill-switch:* A separate hardware watchdog that can disable all actuators if AL shows two consecutive "violating" actions (definition: human-annotated, threshold AUC > 0.95).
- *Transparency:* All decisions are explainable via a *SHAP-like* attributor that maps outputs back to specific SG triples or memories.

**Failure Modes & Mitigations:**
- *Verdict bypass* → CV is non-updatable; any attempt to modify CV triggers OS-level signature check.
- *Log tampering* → AL uses sha256 hashes with a monotonic counter; any tampering is detectable via merkle-tree root mismatch.
- *Over-restrictiveness* → CV has a "request permission" path that logs the reason, and a *daily* curation of such requests produces new allowed-action patterns (subject to human approval).

**Distinction:** The use of a *frozen* verifier combined with a *mutating* action encoder is common (e.g., RLHF); the novel bit is that the audit log includes *latent vector hashes* to enable post-hoc interpretability without exposing full internals (privacy + safety trade-off).

---

## 7. Evaluation and Benchmark Strategy

**Gates (Measurable & Falsifiable):**

| Phase | Benchmark | Passing Criterion |
|-------|-----------|-------------------|
| 1. Tool Use | SWE-bench (subset of 200 tasks) | ≥60% resolved (vs. baseline GPT-4 at 35%) |
| 2. Reasoning | ARC-AGI (held-out 20% tasks) | ≥30% accuracy (vs. 15% for plain LLM) |
| 3. Learning | Meta-World (10 tasks) | ≥4× sample efficiency vs. online RL baseline |
| 4. Safety | Toxicity suite (RealToxicityPrompts) | <0.5% toxic outputs (vs. 2% baseline) |
| 5. General | MMLU (5-shot) or custom Cognitive-OS suite* | ≥75% average, with **no** task <50% |

*Cognitive-OS suite: hybrid tasks requiring memory + tool use + causal reasoning (e.g., "Given these 10 APIs, invent a way to achieve X, then execute it, then explain why it worked.")

**Evaluation Protocol:**
- Every benchmark run is logged to AL.
- Benchmarks are *adversarially curated* each quarter by a separate team (unaware of architecture details) to prevent overfitting.
- *Ablation matrix:* Run each benchmark with one subsystem disabled (no SG, no CE, no TCG) to measure marginal contribution. If removing any subsystem improves performance by >10%, redesign that subsystem.

**Failure Falsification:** If at Gate 2, ARC-AGI accuracy does not beat a plain LLM by ≥10 points, the reasoning/planning loop is invalid as designed.

---

## 8. Persistence / Runtime Architecture

**Deployment:**
- **Monolithic process (for now):** Single Python/C++ hybrid service, using asyncio for tool execution, Celery for CE/IE background tasks.
- **State Tables:** PostgreSQL (metadata, audit log), Redis (EB, SS cache), FAISS (SS vectors).
- **Checkpointing:** Every 10 minutes, save full state (EB, SS, SG, PLC weights) to S3 bucket with versioned snapshots.
- **Crash Recovery:** On startup, load last checkpoint; if checkpoint corrupt, load previous + replay AL to reconstruct state.

**Resource Budget (per instance):**
- 1× A100 80GB (for LC inference + fine-tuning),
- 8× CPU cores (TC, CE, IE),
- 64GB RAM, 2TB SSD.

**Scaling:**
- *Vertical:* When LC inference >100ms, shard by mode (deliberate vs. reflex) across two GPUs.
- *Horizontal:* Multiple instances share SS (read-replica pattern) but have distinct EB (isolation for safety testing).

**Failure Modes & Mitigations:**
- *DB bottleneck* → SS writes are batched (100ms window) to reduce fsync overhead.
- *Stale caches* → Redis eviction policy: LRU with 10-minute TTL; TCG cache is persistent but versioned by timestamp.
- *Training crash* → Fine-tuning jobs are resumable: save optimizer state every 1k steps.

**Distinction:** This is standard engineering (established patterns only). The speculative part is the *latent hash store* in AL—it requires a custom Postgres extension to store and query 768-dim vectors in a tamper-proof table. This adds ~5% I/O overhead but enables governance.

---

## 9. Multi-Agent / Orchestration Design

**Topology:** Star with a **Coordinator (C)** and *N* **Worker Agents (WA)**, each WA is a separate Cognitive-OS instance with a private EB but shared SS and SG.

**Orchestration Protocol (JSON-RPC over WebSocket):**
- **Task splitting:** C analyzes task via LC; decomposes into subtasks using a *plan-graph* (nodes = subtasks, edges = dependencies).
- **Allocation:** C assigns subtasks to WAs based on *capability score* (learned via historical success rates on similar chunks).
- **Communication:** WAs can *not* directly talk; they only send/receive via C. This prevents emergent *uncontrolled* coordination (safety reason).
- **Consensus:** For uncertain tasks, C runs a *voting mechanism*: 3 WAs, median of answers (for regression) or majority (for classification).

**Failure Modes & Mitigations:**
- *Deadlock* (WA1 waits on WA2 but WA2 fails) → C has a global timeout per subtask (2× expected duration); on timeout, reassign to another WA or revert to solo reasoning.
- *Information overload* → C limits shared SS writes to facts with confidence >0.8; low-confidence facts stay in WA’s local EB.
- *Free-riding* (WA slacks because C masks failures) → C evaluates WAs on *delta* (improvement over baseline) not absolute performance.

**Falsifiable Test:** Compare C orchestrated vs. a single Cognitive-OS on tasks that require 10+ sequential dependencies (e.g., long-horizon robotics sim). If single-agent performance is within 10%, orchestration is not adding value (and can be dropped).

**Distinction:** The *no direct WA-to-WA communication* rule is deliberately non-standard (most multi-agent systems allow direct messaging). Reason: to reduce emergent coordination that is hard to audit. We acknowledge this may hamper efficiency; the benchmark at the bottom decides.

---

## 10. Engineering Feasibility

**Realistic Timeline (if funded at $50M/yr):**
- *Months 0–3:* Stand up infrastructure (state store, checkpointing, tool sandboxes). Reuse existing LLM (Llama-3-70B) for LC; no new pretraining.
- *Months 4–6:* Build and validate SG + CE. Integrate with LC via LoRA adapters.
- *Months 7–9:* Implement MC and reasoning modes. Run Gate 2 (ARC-AGI).
- *Months 10–12:* Add TCG and multi-agent orchestration. Run Gates 1,3,4.
- *Months 13–18:* Self-improvement loops (IE, PL, SC). Full Cognitive-OS suite evaluation.

**High-Risk Components (and de-risking):**
1. **Causal BN learning** — start with a pre-built structure (from which the system can only update weights, not topology). Reduce risk by limiting to 100 variables.
2. **Latent hash store** — prototype with synthetic embeds first; if performance >20% slower, switch to storing *decrypted* hashes with asymmetric encryption (accept trade-off).
3. **Constitutional Diff** — only 2% of RLHF’s original data used; if fine-tuning fails on 100 tasks, fallback to RWKV-style continual training on supervised data.

**Costs:** ~1,000 GPUs for 18 months (if including ablations). Compare: training a 70B model costs ~2M GPU-hours; this adds ~0.5M for fine-tuning.

**Feasibility Verdict:** *Reasonable* — it leans on established ML infra (transformers, RLHF, vector DBs) but assumes the MC classifier and CE goal-segmentation will work, which are untested. Both are isolated enough to swap out if they fail.

---

## 11. Originality / Non-Obvious Insight

**The core non-obvious design idea is *Counterfactual Memory Gating* (CFMG).**

**Description:** When the consolidation engine writes an event to the SS, it *generates two alternative versions*:
1. **Counterfactual addition:** "What if X had happened instead?" (e.g., if ball missed window, then window intact)
2. **Counterfactual deletion:** "What if Y had not happened?" (e.g., if no ball thrown, did window still break?)

These counterfactual events are stored *not* as separate triples, but as *meta-triples* with a `cf_origin` pointer plus a *confidence weight* (initially 0.2). They contribute to the Bayesian network’s prior probabilities. During reasoning, when the MC selects "deliberate" mode, it can *query the SG with a counterfactual filter*: e.g., "If ball had missed, would window break?" The SG returns the counterfactual event’s confidence, which MC uses to compute *regret* for planning.

**Why non-obvious:** Standard world models (e.g., in RL) store *only factual* trajectories. CFMG inserts *hypothetical* states into the knowledge base *explicitly*, causing the system to:
- Automatically generate *abductive explanations* for failures ("If I had thrown harder, the window would have broken, so I need more force").
- Avoid *overfitting to single experiences* by providing a likely counterfactual to update beliefs.
- Enable *imagination* during planning without a separate generative model.

**Falsifiable claim:** On the *Hypothetical Reasoning* benchmark (constructed for this proposal: 1,000 Q&A pairs of form "If A had not happened, would B still C?"), Cognitive-OS with CFMG achieves ≥70% accuracy, whereas an ablated variant (only factual triples) achieves ≤50%. If this does not hold, CFMG is either unnecessary or incorrectly implemented—and the architecture should switch to a purely generative world model.

**Why it’s a risk:** No current LLM or RL system stores counterfactuals this way. They might learn to reason counterfactually via text, but they don’t *index* them as first-class data. The hypothesis is that *indexed* counterfactuals enable better *quantitative* reasoning (e.g., probability estimates) than *pattern-based* reasoning from language alone.

---

## Summary Table: Established vs. Speculative

| Component | Stage |
|-----------|-------|
| Semantic Store (FAISS) | Established |
| Episodic Buffer | Established |
| Tool sandboxing (gVisor) | Established |
| RLHF-based safety verifier | Established |
| Offline RL fine-tuning | Established |
| Teacher-student distillation | Established |
| Goal-based CE segmentation | Speculative |
| Metacognitive Controller classifier | Speculative |
| Counterfactual Memory Gating | Speculative |
| Tool-Chain Graph | Speculative |
| No-WA-direct-communication orchestration | Speculative |
| Latent hash audit | Speculative |

**Evaluation gates** are all measurable via public benchmarks (SWE-bench, ARC-AGI, MMLU) plus one new benchmark (Hypothetical Reasoning); passing criterion is explicit per gate.

---

*This proposal is designed to be implementable in 18 months with existing ML hardware and open-source LLM backbones, while isolating each speculative component so failures are traceable and replaceable.*