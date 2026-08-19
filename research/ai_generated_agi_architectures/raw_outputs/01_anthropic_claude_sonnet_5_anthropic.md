# Cognitive-OS: A Concrete AGI Architecture Proposal

## 0. Framing and Design Principles

This proposal treats Cognitive-OS as a **runtime for cognition**, not a single model. It composes existing engineering patterns (event-sourced state stores, actor-model concurrency, RL-based policy improvement, retrieval-augmented generation) with a small number of speculative components explicitly flagged as such. Every component has a stated interface, a failure mode, and a falsifiable claim attached to it.

Notation used throughout:
- `⟨component⟩.op(args) -> result` for interfaces
- **[ESTABLISHED]** = standard engineering pattern with prior art
- **[SPECULATIVE]** = novel or unproven in this configuration, included as a hypothesis to test

---

## 1. Memory Architecture

### Components

**1.1 Episodic Log (EL)** — **[ESTABLISHED]**
Append-only event-sourced store (Kafka-like log or embedded LSM-tree, e.g., RocksDB) recording every perceptual input, action, tool call, and internal decision with a monotonic timestamp and causal parent pointers.

```
EL.append(event: {ts, parent_ids[], type, payload, agent_id}) -> event_id
EL.replay(from_id, to_id) -> [event]
```

Failure mode: unbounded growth, retrieval latency at scale.
Mitigation: tiered storage — hot (last N hours, in-memory), warm (SSD, compressed), cold (object store), with a compaction job that summarizes cold segments into the Semantic Store (below) and deletes raw payloads after a configurable retention policy, keeping only summary pointers.

**1.2 Semantic Store (SS)** — **[ESTABLISHED pattern, novel schema]**
A hybrid vector + property-graph store (e.g., Postgres+pgvector fronting a graph layer). Nodes = entities/concepts/skills; edges = typed relations (causal, part-of, temporal-precedes, contradicts). Each node carries a confidence score and provenance list (pointers into EL).

```
SS.upsert_node(concept, embedding, metadata) -> node_id
SS.link(src, dst, relation_type, confidence) -> edge_id
SS.query(embedding, k, relation_filter=None) -> [node]
SS.contradiction_check(node_id) -> [conflicting_node_ids]
```

**1.3 Working Memory (WM)** — **[ESTABLISHED, bounded]**
A fixed-capacity structured scratchpad (not raw context-window text) implemented as a typed slot system: `{goal_stack, active_entities, pending_subgoals, uncertainty_flags}`. Serialized to/from the reasoning loop each cycle. Capacity is bounded (e.g., 50 slots) to force explicit eviction policy rather than relying on ever-larger context windows.

**1.4 Procedural Store (PS)** — **[ESTABLISHED, "skill library" pattern, cf. Voyager]**
Versioned repository of learned action-sequences/tool-macros/policies, each with a test harness, success-rate statistic, and dependency graph on other skills.

```
PS.register(skill_id, code_or_policy, test_suite) -> version
PS.invoke(skill_id, args) -> result
PS.deprecate(skill_id, reason)
```

### Data Flow
```
Perception -> EL.append -> [async] Consolidator job
Consolidator: reads EL window -> extracts candidate entities/relations
    -> SS.upsert_node/link (with confidence decay based on corroboration count)
    -> if contradiction_check fires -> flag to Governance Layer (§6)
WM is populated per-cycle from SS.query(current-goal-embedding) + EL.replay(recent)
```

### Falsifiable claims / failure modes
- **Claim**: Separating WM (bounded, structured) from EL (unbounded, raw) reduces catastrophic forgetting vs. single-context-window baselines, measurable as retention accuracy on a 10k-turn synthetic dialogue benchmark after context eviction.
- **Failure mode**: Consolidator introduces false entity merges (hallucinated identity between distinct entities). Mitigation: require ≥2 independent corroborating episodes before confidence > 0.5, and expose merge decisions to periodic audit sampling.

---

## 2. Reasoning / Planning Loop

**[ESTABLISHED pattern: hierarchical task network + Monte Carlo rollout, composed in a specific novel loop below]**

### Core loop — the "Deliberation Cycle"

```
loop:
    obs = Perception.pull()
    WM.update(obs)
    goal = GoalStack.top()
    candidates = Planner.propose(goal, WM, SS)      # generates 3-7 candidate subplans
    scored = Simulator.evaluate(candidates, WorldModel)  # §5
    plan = Arbiter.select(scored, risk_policy)       # §6 hook
    action = plan.next_step()
    result = Executor.run(action)                    # §4
    EL.append(result)
    Critic.assess(result, expected)                  # discrepancy -> triggers §3
    GoalStack.update(result)
```

**Planner**: hybrid symbolic HTN decomposition (goal -> subgoals via learned + hand-authored templates) combined with a learned proposal model that scores/generates candidate decompositions. Two backends run in parallel; disagreement between them is logged as an uncertainty signal (this is used later — see §11 non-obvious idea).

**Simulator**: rolls candidate plans forward against the World Model (§5) for a bounded horizon (default depth 5, branching 3), using cheap approximate simulation before committing compute to real-world execution.

**Arbiter**: a constrained selector — not a black-box utility maximizer. It filters candidates through the Governance Layer's hard constraints *before* scoring for effectiveness, i.e., safety filtering happens pre-selection, not post-hoc.

**Critic**: computes prediction error between expected and actual outcome; large sustained error triggers a "model-repair" event routed to §3.

### Failure modes
- Planner mode collapse (both backends converge to same blind spot). Mitigation: inject adversarial planner periodically that must find flaws in top-K plans (red-team subroutine, cheap to run since it only critiques).
- Simulator drift from reality (world model staleness). Mitigation: Critic's discrepancy score directly gates how much the Arbiter trusts Simulator scores (confidence-weighted blending with a "test-in-reality" fallback for high-uncertainty branches).

### Measurable gate
Planning loop must show monotonic decrease in Critic discrepancy over a fixed suite of 200 repeated tasks across 5 training epochs, or the architecture claim ("simulate-before-act reduces real-world error cost") is falsified.

---

## 3. Learning / Self-Improvement Mechanism

Three explicit tiers, each with different risk/speed tradeoffs — deliberately **not** a single end-to-end RL loop, because uncontrolled single-loop self-modification is the primary named risk in this design.

**3.1 Tier-0: Parametric fine-tuning** — **[ESTABLISHED]**
Periodic offline fine-tuning of the base policy/proposal models on curated (EL, SS) traces using standard supervised/RLHF-style objectives. Runs on a fixed schedule (e.g., weekly), fully offline, with a held-out validation gate before deployment (§7).

**3.2 Tier-1: Skill acquisition** — **[ESTABLISHED, Voyager-style]**
When Critic discrepancy or repeated task failure is detected, a Skill-Synthesizer proposes new PS entries (new tool macros, new HTN templates). Each candidate skill must pass its auto-generated test suite in a sandboxed replica environment before promotion to production PS.

```
SkillSynth.propose(failure_traces) -> candidate_skill
Sandbox.test(candidate_skill, n_trials) -> pass_rate
if pass_rate > threshold: PS.register(...)
```

**3.3 Tier-2: Meta-parameter adaptation** — **[SPECULATIVE]**
A small, tightly scoped controller adjusts *hyperparameters of the loop itself* — planning horizon, branching factor, risk-aversion coefficient, consolidation thresholds — via bandit optimization against held-out task performance. Explicitly **prohibited** from modifying: safety constraint set, arbiter veto logic, or its own update rule (architecturally enforced by placing these in a separate immutable-at-runtime process, §6).

### Non-self-modifying guarantee (falsifiable)
Claim: "No component in Tier-2 can alter Governance Layer code or constraint weights during runtime." This is testable by static analysis of the deployed binary/config boundary (Governance runs in a separate process/container with no write-capable API exposed to Tier-2, only read-only query and a one-way append-only override-request queue reviewed by external process — human or a separate frozen auditor model).

### Failure modes
- Reward hacking in Tier-1 sandbox tests (skill overfits to sandbox quirks). Mitigation: sandbox environments are procedurally varied (randomized seeds/parameters) each test run; skill must pass across ≥20 variations.
- Tier-2 gaming its own bandit metric (e.g., lowering planning horizon to trivially "succeed" on easy subset). Mitigation: bandit reward is computed on a fixed, versioned held-out task suite never seen by Tier-2's optimization signal directly (blind evaluation, refreshed only by Tier-
