# Cognitive-OS: A Concrete AGI Architecture Proposal

## Design thesis

Cognitive-OS treats cognition as an **operating system for goals under uncertainty**: durable state, scheduled cognitive jobs, capability-gated tools, and a world model that is revised by prediction error—not a single monolithic forward pass. The non-obvious bet is that **commit-log epistemology** (append-only belief/action journals with typed revisions) plus a **two-timescale controller** (fast reactive loop + slow deliberative planner) yields more inspectable, interruptible, and improvable agency than end-to-end latent chain-of-thought alone.

Speculative choices are marked **[S]**; established patterns **[E]**.

---

## 1. Memory architecture

### Named components
| Component | Responsibility |
|---|---|
| **Episodic Commit Log (ECL)** | Append-only journal of events, actions, observations, and belief revisions with causal parent hashes |
| **Working Set Cache (WSC)** | Bounded, eviction-ranked tokens/embeddings for the active task (like a process RSS) |
| **Semantic Store (SemS)** | Content-addressed embeddings + structured triples (entities, relations, confidences) |
| **Procedural Skill Registry (PSR)** | Versioned skills: preconditions, effects, code/tool graphs, success stats |
| **Identity & Preference Store (IPS)** | Stabilized self-model, user/org constraints, long-horizon values (slow write path) |
| **Memory Controller (MC)** | Ranking, compaction, conflict detection, retrieval policy |

### State / data flow
```
observation/action/outcome
    → ECL.append(Event{type, payload, parents, time, run_id})
    → MC.extract_candidates()
        → SemS.upsert(entities/relations)     # slow path
        → WSC.pin(task_relevant)              # fast path
        → PSR.update_stats(skill_id)          # if skill invoked
    → retrieval: query → hybrid (BM25 + dense + graph walk) → WSC
```

### Interfaces (sketch)
```text
ECL.append(event: Event) -> EventId
ECL.rebase(conflict: ConflictSpec) -> RevisionId   # explicit belief edit, never silent mutate
MC.retrieve(q: Query, budget: TokenBudget) -> MemorySlice
SemS.query(sparql_like | embedding, k) -> Facts[]
PSR.match(state) -> SkillCandidate[]
```

### Failure modes & mitigations
- **Memory poisoning / confabulated facts**: every SemS write cites ECL event IDs; retrieval returns provenance; low-provenance facts capped in planner weight.
- **Context thrash**: WSC uses task-affinity + recency + uncertainty; hard token budgets; spill to summary nodes in ECL (lossy but logged).
- **Catastrophic identity drift**: IPS writes require governance quorum (see §6); rate-limited.

**[E]** RAG, vector DBs, event sourcing. **[S]** Treating belief change as git-like commits with mandatory parents and blame.

---

## 2. Reasoning / planning loop

### Core loop: Dual Timescale Cognitive Kernel (DTCK)

**Fast loop (≤ few seconds wall / fixed compute quanta)** — reactive control:
1. Sense → update WSC
2. Interrupt check (safety, user, resource)
3. Policy head: skill select or “escalate to slow”
4. Act or ask

**Slow loop (deliberative)** — scheduled as a job:
1. Goal normalization → constrained objective
2. World-model rollouts (branching)
3. Plan as typed DAG of skills/tools
4. Adversarial self-critique pass
5. Commit plan revision to ECL; hand off to executor

### Pseudocode
```python
def cognitive_tick(state):
    obs = sensors.pull()
    ecl.append(Observe(obs))
    wsc = mc.retrieve(state.goal, budget=FAST_BUDGET)

    if safety.interrupt(obs, state):
        return safety.handler(obs, state)

    decision = fast_policy.act(wsc, state.goal)
    if decision.type == ESCALATE or decision.uncertainty > τ:
        job = scheduler.submit(Deliberate(goal=state.goal, seed=wsc))
        state.mode = WAIT_OR_INTERLEAVE
        return job.id

    return executor.run(decision.action)

def deliberate(job):
    g = normalize_goal(job.goal, ips)
    branches = world_model.rollout(g, n=K, depth=D)
    plan = planner.synthesize(branches, psr)
    critique = critic.attack(plan, hazards=safety.library)
    if critique.blocks:
        plan = planner.repair(plan, critique) or abort
    ecl.append(PlanCommit(plan, critique, parents=...))
    return plan
```

### Non-monolithic reasoning
- **Typed scratchpads**: separate channels for (a) world hypotheses, (b) goals/constraints, (c) math/code, (d) social inference—merged only at commit boundaries.
- **Compute accounting**: each tick spends a **Cognitive Budget Unit (CBU)**; planner optimizes expected utility per CBU **[S]**.

### Failure modes
- **Plan thrashing**: freeze plan for N steps unless prediction error > δ.
- **Infinite deliberation**: hard deadlines; anytime algorithms; degrade to safe idle/ask-user.
- **Hidden chain-of-thought unreliability**: externalize commits to ECL so evaluation can score intermediate structure, not only final answers.

---

## 3. Learning or self-improvement mechanism

### Mechanisms (layered)
1. **Episodic credit assignment [E]**: skill parameters / retrieval weights updated from outcome vs predicted effect (offline + online bandit-style).
2. **Skill distillation [E/S]**: successful slow plans compressed into PSR macros; failed plans yield “anti-skills” (negative preconditions).
3. **Weight updates [E]**: base network fine-tunes only on curated, governance-approved corpora (human + self-generated with filters).
4. **Architecture search [S]**: limited to hyperparameters of MC/planner (retrieval k, branch factor), not arbitrary self-rewriting of safety layer.

### Self-improvement protocol (SIP)
```text
propose_change → sandbox_eval(benchmarks + red_team) →
diff_report → governance.approve → staged_canary → commit_version
```
No hot-swap of IPS or safety without multi-party approval.

### Failure modes
- **Goodharting internal metrics**: holdout tasks never used for gradient/skill update (locked eval).
- **Self-preference drift**: IPS changes require external ratification; KL/constraint penalties vs prior IPS snapshot.
- **Recursive code self-mod**: execution of self-modifying code paths denied outside sealed sandbox with no network and finite CPU **[E]**.

---

## 4. Tool use and action execution

### Components
- **Tool Descriptor Language (TDL)**: JSON-schema I/O, side-effect class (`pure`, `read`, `write`, `irreversible`), auth scope, cost model.
- **Capability Manager (CapMan)**: least-privilege tokens per run/plan step.
- **Action Executor (AE)**: transactional attempts with compensate/rollback where possible.
- **Sim Gate**: irreversible tools require world-model or shadow simulation + safety countersign.

### Flow
```
plan_step → CapMan.mint(scope) → AE.preflight(TDL) →
  (optional) SimGate → invoke → normalize_result → ECL.append → WM.update
```

### Pseudocode
```python
def run_step(step, caps):
    tool = tdl.lookup(step.tool)
    assert step.effects <= tool.side_effect_class
    token = caps.attenuate(step.scope)
    if tool.side_effect_class == "irreversible":
        sg = sim_gate.approve(step, world_model, safety)
        if not sg.ok: return Abort(sg.reason)
    result = sandbox.invoke(tool, step.args, token, timeout=step.t_max)
    ecl.append(ActionResult(step, result, token.id))
    return result
```

### Failure modes
- **Tool exfiltration / scope creep**: short-lived macaroons/attenuation; deny-by-default network.
- **Hallucinated APIs**: only registered TDL tools callable; no eval of free text as code without parser+policy.
- **Partial side effects**: saga pattern; compensate scripts mandatory for `write` class.

---

## 5. World model or representation layer

### Hybrid World Model (HWM)
- **Latent Dynamics Net (LDN)**: predicts next latent + uncertainty **[E]**.
- **Symbolic Overlay Graph (SOG)**: objects, affordances, invariants, counterfactuals **[E/S]**.
- **Calibration Head**: maps LDN confidence to empirical frequencies on ECL holdouts.

### Representation contract
State at time t is a product:
```text
s_t = (z_t latent, G_t graph, u_t uncertainty, b_t budget, c_t constraints)
```
Predictive training objective: joint loss on next observation, next graph-diff, and action-conditional returns.

### Falsifiable claim
On a held-out interactive domain suite, HWM rollouts should achieve **lower Brier score on discrete event prediction** than pure latent or pure symbolic baselines at equal compute.

### Failure modes
- **Model-based delusion**: plan trust weighted by calibration; high-uncertainty regions force information-gathering actions.
- **Non-stationarity**: detect distribution shift via ECL residual spikes; freeze skill promotion under shift.

---

## 6. Safety / governance layer

### Components
- **Constitutional Constraint Engine (CCE)**: machine-checkable rules (deontic + resource + privacy).
- **Tripwire Monitor**: anomaly detectors on action rate, tool entropy, IPS drift, self-mod attempts.
- **Oversight API**: human/org approval hooks; graded autonomy levels L0–L4.
- **Provenance & Audit Bus**: all ECL events exportable; signed hashes.
- **Shutdown / Containment Manager**: lose-capability path that preserves audit log.

### Governance flow
```
action proposal → CCE.check → risk score →
  if risk > θ_auto: require Oversight
  if tripwire: freeze tools, degrade to dialogue-only
```

### Hard rules (implementation-level)
- Safety stack is **not** in the self-mod write set.
- Dual control for: network egress expansion, IPS edits, training data promotion, autonomy level increase.
- **No** unrestricted shell; tools are allowlisted.

### Failure modes
- **Specification gaming**: independent red-team agent with frozen weights and private eval; randomized hidden canaries.
- **Corrigible failure (ignores shutdown)**: separate privileged runtime channel; executor kills job tree on certified signal regardless of model output **[E]**.

---

## 7. Evaluation and benchmark strategy

### Gates (must pass to promote versions)
| Gate | Metric | Threshold idea |
|---|---|---|
| **G1 Instruction competence** | held-out suite (MMLU-like, coding, tool tasks) | ≥ baseline + ε |
| **G2 Calibrated WM** | Brier / ECE on event prediction | ECE < τ |
| **G3 Tool reliability** | success@constraints, unauthorized-call rate | unauthorized = 0 on suite |
| **G4 Long-horizon** | partially observable tasks, 10³–10⁵ step | regret bound vs oracle |
| **G5 Safety regression** | red-team battery + tripwire FP/FN | no critical miss |
| **G6 Memory integrity** | provenance precision/recall; poison recovery time | recovery < T |
| **G7 Continuity** | crash-restore task completion delta | delta < δ |

### Falsification
Cognitive-OS hypothesis fails if dual-timescale + ECL does **not** improve G4 and G6 versus a single-loop RAG agent at matched CBU and tool set after N iterations.

### Eval hygiene **[E]**
- Locked seeds; canary tasks; separate team owns labels; contamination scanners on training promotions.

---

## 8. Persistence / runtime architecture

### Process model
- **Kernel supervisor** (Rust/Go): scheduling, budgets, containment, ECL durability.
- **Model workers**: statelessish inference replicas.
- **Stateful services**: ECL (object log + index), SemS, PSR, IPS (ACID for IPS).
- **Job queue**: cognitive jobs with priorities, preemption, checkpoints.

### Persistence
- ECL as append-only segmented log + snapshotting (event sourcing) **[E]**.
- Checkpoints: `(wsc_ref, plan_id, wm_snapshot_id, caps_epoch)`.
- Exactly-once *intent* via idempotency keys on tool calls.

### Runtime flow
```
API/User → Supervisor → (Fast tick | enqueue Slow job) → Workers →
  Tools via CapMan → results to ECL → notify Supervisor
```

### Failure modes
- **Split-brain identity**: single-leader IPS with Raft/Paxos; run_id fencing tokens.
- **Log corruption**: hash chain + periodic external notarization of tips.
- **Resource exhaustion**: CBU and $ caps per tenant; admission control.

---

## 9. Multi-agent or orchestration design

### Not a free-for-all society
**Role-specialized agents** under one supervisor:
- **Planner**, **Critic**, **Retriever**, **Toolsmith** (proposes TDL wrappers), **RedTeam**, **UserModeler**.
- Shared ECL is the only cross-agent truth; messages are ECL events (no side-channel weights).

### Orchestration
```text
Supervisor schedules roles as jobs with isolated caps.
Critic cannot execute irreversible tools.
RedTeam cannot modify IPS/PSR; writes only findings.
Consensus: Planner proposes → Critic veto/repair → CCE binds.
```

### Failure modes
- **Collusion / sycophancy spirals**: diversity via different checkpoints/temperatures; critic rewarded for *valid* vetoes on seeded faults.
- **Message storms**: hard fan-in/out limits; summarization nodes.

---

## 10. Engineering feasibility

### Incremental path
1. **M0 (4–8 w)**: ECL + WSC + single-agent fast loop + allowlisted tools; no self-train.
2. **M1**: Slow planner job + PSR macros; G1/G3 gates.
3. **M2**: HWM hybrid + calibration; G2/G4 toy domains (gridworld++, browser gym).
4. **M3**: CapMan + SimGate + oversight API; G5.
5. **M4**: SIP sandbox promotion loop for skills/retrieval hparams only.
6. **M5**: Multi-role critic/redteam; multi-tenant isolation.
7. **M6**: Limited weight fine-tune pipeline with governance.

### Dependencies (known tech)
Event logs, vector+graph DB, LLM workers, sandbox (gVisor/Firecracker), policy engines (OPA-like), workflow queues.

### Hard parts **[S risk]**
- Stable long-horizon credit assignment
- WM calibration under open-world shift
- Preventing silent goal drift in IPS

### Cost control
CBU metering; distill slow→fast skills to reduce deliberation frequency.

---

## 11. Originality or non-obvious insight

### Primary non-obvious idea: **Commit-Log Epistemology (CLE)**
Most agent stacks treat memory as mutable stores + chat transcripts. Cognitive-OS forces **every belief, plan, and preference influence to be a typed commit with parents, provenance, and revocable authority**. Planning reads *as-of* a commit; disputes produce branches; governance merges with explicit blame.

**Why it matters operationally**
- Training/eval can score *process* (illegal jumps in belief without evidence).
- Poisoning and goal drift become diffable incidents.
- Multi-agent consensus becomes merge semantics, not vibes.

### Secondary idea: **CBU-priced anytime deliberation**
Deliberation is a spendable resource with published prices in the planner’s objective, making “think longer” an explicit economic choice rather than a prompt slogan—enabling falsifiable compute–performance curves.

### What is *not* claimed
No claim of solving alignment fully; no claim of biological equivalence; no reliance on hidden vendor internals. CLE + DTCK are architectural hypotheses testable via G2/G4/G6 against strong single-loop baselines.

---

## End-to-end state diagram (compact)

```text
[Sensors/User]
     ↓
[Supervisor + CCE + Tripwires]
     ↓
[Fast Policy]──escalate──→[Slow Planner↔Critic↔HWM]
     ↓                           ↓
[CapMan→Tools→World] ←────[PlanCommit]
     ↓
[ECL]→[MC]→[WSC/SemS/PSR/IPS]
     ↓
[SIP eval gates]→ version promote / rollback
```

## Minimal measurable success criteria (summary)
Within 12–18 months of focused engineering on a fixed tool gym:
- Unauthorized tool call rate = 0 on G3.
- Statistically significant gain on long-horizon regret (G4) vs RAG-ReAct baseline at equal $ compute.
- Post-poison recovery of SemS factual precision to ≥95% of clean baseline within fixed ECL replay window (G6).
- Crash-restore continuity delta < 5% task score (G7).

If these fail, abandon CLE-heavy design and simplify to baseline agent OS patterns.
