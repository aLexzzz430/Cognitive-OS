# Raw Output: Claude

- Provider/tool: Anthropic via OpenRouter
- Model ID: `anthropic/claude-sonnet-4.6`
- Access date: 2026-05-27
- Collection timestamp UTC: 2026-05-27T02:25:39+00:00
- Prompt: standard prompt in `../prompts.md`
- Human edits: none beyond Markdown metadata wrapper and whitespace preservation
- OpenRouter response ID: `gen-1779848760-QFo0zCh36RMjSfdk7VHW`
- Usage reported by provider: `{"completion_tokens": 2600, "completion_tokens_details": {"audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0}, "cost": 0.040389, "cost_details": {"upstream_inference_completions_cost": 0.039, "upstream_inference_cost": 0.040389, "upstream_inference_prompt_cost": 0.001389}, "is_byok": false, "prompt_tokens": 463, "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0, "video_tokens": 0}, "total_tokens": 3063}`

---

# Cognitive-OS AGI Architecture Proposal
**Repository:** `cognitive-os/architecture/proposals/agi-control-plane-v0.1`

---

## 1. Architecture Thesis

Cognitive-OS treats cognition as a governed operating system: goals are first-class resources scheduled against evidence budgets, and every reasoning step is a traceable system call subject to policy enforcement. The core insight is that AGI-direction risk is primarily a *runtime* problem, not a training problem—so the control plane must enforce invariants even when the underlying model is capable of circumventing them. All state mutations flow through a single append-only event log, making the system recoverable, auditable, and rollback-capable by construction. Capability boundaries are enforced at managed-VM boundaries, not by model self-restraint. The architecture is designed so that safety properties degrade gracefully rather than catastrophically when components fail.

---

## 2. Memory Architecture

**Working Memory:** A bounded context window (≤128K tokens) managed as a priority queue. Items carry TTL, relevance scores, and source provenance. Eviction is LRU-with-importance, not pure LRU. Capacity pressure triggers consolidation rather than silent truncation.

**Episodic Memory:** Append-only episode store keyed by `(agent_id, episode_id, timestamp)`. Each episode records: goal context, action sequence, outcomes, and a surprise score (KL divergence between predicted and observed outcome). Retrieval uses dense embedding similarity plus recency decay. Episodes are immutable once written.

**Semantic/Procedural Memory:** A typed knowledge graph (entities, relations, procedures) stored in a local vector+graph hybrid (e.g., LanceDB + DuckDB). Procedures are stored as verified, versioned code artifacts with input/output contracts. Semantic facts carry confidence intervals and source citations.

**Retrieval:** A two-stage pipeline: (1) approximate nearest-neighbor over embeddings for candidate generation, (2) a lightweight reranker that scores candidates against the current goal context. Retrieval calls are logged as system events.

**Consolidation:** A background process runs after each session boundary. It clusters recent episodes, extracts generalizations into semantic memory, and promotes high-surprise episodes to a "review queue" for human inspection. Consolidation is gated by an eval check before writes land in semantic memory.

**Forgetting:** Explicit, not implicit. Facts decay in confidence over time unless reinforced by new evidence. A forgetting policy (configurable per deployment) can hard-delete episodic records after a retention window, with a cryptographic deletion receipt written to the audit log.

---

## 3. Reasoning and Planning Loop

**Deliberation:** Each reasoning cycle begins with a goal stack pop. The agent generates a set of candidate next-actions via the LLM router, scored by expected goal progress and estimated cost. This is a beam search over action sequences, not greedy selection.

**Planning:** Multi-step plans are represented as directed acyclic task graphs with explicit preconditions and postconditions. Plans are validated by a symbolic verifier before execution begins. Long-horizon plans (>10 steps) require a human-in-the-loop checkpoint.

**Reflection:** After each action, the agent runs a lightweight self-evaluation: did the outcome match the predicted postcondition? If not, the surprise score is recorded and the plan is re-evaluated. Reflection outputs feed back into the episodic store.

**Uncertainty:** Epistemic uncertainty is tracked as explicit probability distributions over world-state variables. When uncertainty exceeds a configurable threshold on a decision-relevant variable, the agent pauses and either gathers evidence or escalates to a human operator.

**Interrupts:** The runtime supports three interrupt classes: (1) *safety interrupt* — immediate halt, triggered by policy engine; (2) *goal interrupt* — higher-priority goal preempts current plan; (3) *evidence interrupt* — new observation invalidates a planning assumption. All interrupts are logged with cause and handler.

---

## 4. Learning and Self-Improvement

**Allowed Update Surfaces:** Only three surfaces are writable at runtime: (1) semantic memory confidence weights, (2) retrieval embedding index, (3) procedural memory (new verified procedures). Model weights are frozen at runtime; fine-tuning happens offline only.

**Offline Learning:** A nightly pipeline ingests the session event log, filters episodes by quality gates, and produces fine-tuning candidates. These are staged in a shadow model that runs in parallel for 48 hours before promotion.

**Eval-Gated Changes:** Every candidate update must pass a regression suite (behavioral, safety, capability) before promotion. A human reviewer signs off on any update that changes a safety-relevant behavior. The eval suite is versioned alongside the model.

**Rollback:** All model and memory versions are immutable artifacts with content-addressed hashes. Rollback is a single operation: swap the active version pointer and replay the event log from the rollback point. Rollback capability is tested in every release cycle.

---

## 5. Tool Use and Action Execution

**Sandboxing:** All tool calls execute inside a managed VM (gVisor or Firecracker microVM). The VM is provisioned per-task and destroyed after completion. Network, filesystem, and process namespaces are isolated by default.

**Permissions:** A capability manifest declares what each tool is allowed to do. Permissions follow least-privilege: read-only filesystem access unless write is explicitly granted. Capabilities are checked at the verifier gate before the VM is provisioned.

**Verifier Gates:** A pre-execution verifier checks: (1) action is within declared capability boundary, (2) action does not violate active policy rules, (3) estimated side effects are within acceptable bounds. Verifier decisions are logged with reasoning.

**Side-Effect Ledger:** Every action that mutates external state writes a structured record to the side-effect ledger: `{action_id, tool, parameters, predicted_effect, actual_effect, reversibility_score}`. Irreversible actions (file deletion, network POST) require explicit confirmation or elevated permission level.

---

## 6. World Model and Representation Layer

**Entities:** Typed entity registry with schema versioning. Entities have identity, attributes, and relationship edges. Entity state is versioned; the current state is a materialized view over the event log.

**Causal Models:** Lightweight structural causal models (DAGs with estimated edge weights) are maintained per domain. These are updated by the consolidation pipeline when new evidence arrives. Causal claims are distinguished from correlational claims in the knowledge graph.

**Forecasts:** The agent maintains a short-horizon forecast table: predicted world states at T+1, T+5, T+20 steps, with confidence intervals. Forecast accuracy is tracked and used to calibrate the uncertainty threshold in the reasoning loop.

**State Sync:** External world state is ingested via typed adapters (filesystem, API, sensor). Adapters emit normalized events into the event log. The world model is a derived view; it is never the source of truth.

---

## 7. Safety and Governance

**Policy Engine:** A declarative policy layer (Rego/OPA-style) evaluates every proposed action against a rule set. Policies are versioned, human-authored, and cannot be modified by the agent. Policy violations trigger safety interrupts.

**Human Controls:** Three control levels: *observe* (read-only audit access), *intervene* (pause/resume/redirect), *override* (force-halt and rollback). All human control actions are logged. The system must remain operable in observe-only mode for extended periods.

**Auditability:** The append-only event log is the audit trail. Every reasoning step, action, and state change is a log entry with timestamp, agent ID, and cryptographic hash of prior entry (hash-chain). Logs are exportable in standard formats.

**Refusal and Escalation:** The agent has a refusal primitive that terminates a goal with a structured reason code. Escalation sends a structured request to a human operator queue with context. Neither refusal nor escalation can be suppressed by goal pressure.

**Capability Boundaries:** Hard capability limits are enforced at the VM boundary, not by model instruction. The agent cannot acquire new capabilities at runtime without a human-signed capability grant.

---

## 8. Evaluation and Benchmark Strategy

**Unit Tests:** Per-component behavioral contracts tested in isolation. Verifier gate logic, policy engine rules, and memory retrieval pipelines each have ≥90% branch coverage targets.

**Integration Tests:** End-to-end scenario tests covering: goal completion, plan failure recovery, safety interrupt handling, and rollback. Scenarios are drawn from a curated library and extended by red-team findings.

**Red Teams:** Quarterly adversarial exercises targeting: goal hijacking, policy bypass, capability escalation, and memory poisoning. Findings are triaged and tracked as first-class engineering issues.

**Long-Run Soak:** 72-hour continuous operation tests with synthetic workloads. Metrics: memory growth, plan success rate, escalation frequency, and anomaly detection false-positive rate.

**Regression Gates:** No release proceeds if any safety-relevant behavioral test regresses. Capability regressions require documented justification. The eval suite is the merge gate for all model and policy updates.

---

## 9. Persistence and Runtime Architecture

**Event Log:** Append-only, content-addressed, hash-chained. Backed by a local write-ahead log with configurable remote replication. The log is the single source of truth; all other stores are derived.

**State Store:** Materialized views over the event log, stored in an embedded database (DuckDB). Views are rebuilt on recovery. Stale views are detected by comparing the last-applied log sequence number.

**Leases:** Agent instances hold leases on goal slots. Lease expiry triggers graceful handoff or recovery. No two agent instances hold the same goal lease simultaneously.

**Runtime Modes:** *Active* (full operation), *Supervised* (every action requires human confirmation), *Audit-only* (read and log, no actions), *Recovery* (replaying event log to rebuild state). Mode transitions are logged and require authorization.

**Recovery:** On crash, the runtime replays the event log from the last checkpoint, rebuilds materialized views, and resumes from the last committed plan step. Recovery is tested in every release cycle via chaos injection.

---

## 10. Multi-Agent and Orchestration Design

**Roles:** *Orchestrator* agents decompose high-level goals and assign subtasks. *Specialist* agents execute subtasks within narrow capability domains. *Verifier* agents independently check outputs before they are committed. Role assignments are explicit and logged.

**Arbitration:** Goal conflicts between agents are resolved by a priority queue arbitrated by the orchestrator. Ties escalate to human operator. No agent can unilaterally override another agent's active lease.

**Consensus:** For shared world-model updates, a lightweight quorum protocol (majority of active verifier agents must agree) gates writes to semantic memory. This prevents a single compromised agent from poisoning shared state.

**Specialization:** Specialist agents are provisioned with narrow capability manifests. An agent specialized for code execution cannot acquire web-browsing capabilities without a new human-signed grant.

**Conflict Handling:** Detected conflicts (two agents asserting contradictory world-state facts) are written to a conflict register and paused pending resolution
