# Summary — Patterns, Disagreements, and Notable Ideas

> **Note:** This analysis is based on 3 outputs from Xunfei Spark (astron-code-latest) using different system prompts to simulate distinct perspectives: (1) Standard AGI Researcher, (2) Oxford FHI Safety Researcher, (3) Pragmatic Production Engineer.

---

## Common Patterns

All three perspectives converge on several core architectural principles:

1. **Tiered Memory is Universal** — Every perspective agrees on separating short-term (working/context), long-term (episodic/vector), and semantic (structured knowledge) memory. The specific implementations differ, but the three-tier model is consistent.

2. **Planning ≠ Generation** — All three reject single-shot LLM generation as sufficient for AGI. They all propose structured planning loops (DAGs, PEOR cycles, or state machines) that decompose goals, execute steps, and revise based on feedback.

3. **Tool Use Requires Sandboxing** — Every perspective insists on isolated execution environments (gVisor, Firecracker microVMs, or containerized sandboxes) with strict resource limits and no unrestricted host access.

4. **Learning Must Be Safe and Bounded** — All three reject continuous online training of base model weights. They converge on adapter-based approaches (LoRA), experience replay, and decoupled learning pipelines with safety gates.

5. **Multi-Agent > Monolithic** — All three favor specialized agents over a single model doing everything, though the engineer is most pragmatic about the trade-offs (cost, latency).

---

## Key Disagreements

| Dimension | Standard Researcher | Safety Researcher | Pragmatic Engineer |
|-----------|-------------------|-------------------|-------------------|
| **World Model** | Central component: neuro-symbolic predictive engine with surprise-driven learning | Skeptical: focuses on formal verification rather than predictive simulation | Pragmatic: relies on RAG + tool outputs as ground truth, no internal simulation |
| **Memory Immutability** | Dynamic: graph updated via GNN, episodic memory appendable | Strict: immutable history, cryptographic hashing, read-only semantic graph | Pragmatic: summarization + deletion of raw history to manage context window |
| **Safety Approach** | Constitutional Firewall: parallel independent agent with veto power | Safety-first: safety gates at every layer, formal specs, cryptographic integrity | Safety as constraint: rule-based filters, canary releases, cost-aware trade-offs |
| **Evaluation** | Dynamic Capability Matrix with continuous adversarial testing | Formal specification verification | Production metrics: latency, cost, reliability |
| **Engineering Path** | 4-phase incremental integration over months/years | Safety gates at each step, no shortcuts | MVP first: ship something that works today, iterate |

**Fundamental Tension:** The safety researcher views the world model as potentially dangerous (it could be wrong and the system could act on wrong predictions), while the standard researcher views it as essential (the system must simulate before acting). The engineer sidesteps this by relying on external ground truth (tool outputs) rather than internal simulation.

---

## Notable Ideas

### From the Standard Researcher
- **"Surprise-Driven" World Model** — Treat prediction error as the primary learning signal, mimicking human curiosity. When the model predicts X but observes Y, it triggers causal investigation, not just weight updates.
- **Governance as a Service** — Safety Layer as a separate, competing microservice agent with its own reasoning path, creating biological-style "checks and balances" (prefrontal cortex vs. amygdala).
- **Event-Sourced Reasoning** — Immutable event log enables "time travel" debugging: replay exact mental state from 5 steps ago and try a different branch.

### From the Safety Researcher
- **Immutable History** — Once an episode is written, it cannot be altered. Prevents "memory laundering" where the model deletes evidence of safety violations.
- **Explicit Context Injection** — Context via structured JSON/XML, not raw text concatenation, to prevent prompt injection attacks.
- **RAG with Verification** — Retrieval must return source hashes that the model must cite. No black-box retrieval.

### From the Pragmatic Engineer
- **Distilled Planner, Large Executor** — Use a cheap 8B model for planning, reserve 70B for complex execution. Dramatic cost savings.
- **Max 5-Step Planning Tree** — Hard limit prevents infinite loops and controls token costs.
- **Per-Tenant Sharding** — Vector DB sharded per user for privacy and latency, aggregated for system-wide knowledge.
- **Top 1% Fine-Tuning** — Only fine-tune on the most valuable, complex cases, not all user logs.

---

## Model-by-Model Highlights

| Perspective | Standout Idea | Weakness |
|-------------|--------------|----------|
| **Standard Researcher** | Surprise-driven world model + event-sourced reasoning | Over-engineered; assumes capabilities that don't exist yet |
| **Safety Researcher** | Immutable memory + cryptographic integrity | May be too restrictive; could limit system flexibility |
| **Pragmatic Engineer** | Distilled planner + 5-step limit + canary releases | Lacks depth on world modeling and long-term safety |

---

## Cross-Perspective Consensus Score

| Dimension | Agreement Level | Notes |
|-----------|----------------|-------|
| Memory Architecture | ⭐⭐⭐⭐⭐ | All three agree on 3-tier model |
| Reasoning/Planning | ⭐⭐⭐⭐⭐ | All reject single-shot generation |
| Learning/Self-Improvement | ⭐⭐⭐⭐ | All agree on adapter-based, bounded learning |
| Tool Use/Sandboxing | ⭐⭐⭐⭐⭐ | Universal agreement on isolation |
| World Model | ⭐⭐ | Major disagreement on necessity and approach |
| Safety/Governance | ⭐⭐⭐ | Agree on importance, disagree on implementation |
| Evaluation | ⭐⭐ | Different priorities (research vs. safety vs. production) |
| Persistence | ⭐⭐⭐ | Engineer least detailed |
| Multi-Agent | ⭐⭐⭐⭐ | All favor specialization |
| Engineering Feasibility | ⭐⭐⭐ | Different time horizons |
| Originality | ⭐⭐⭐⭐ | Each brings unique insights |
