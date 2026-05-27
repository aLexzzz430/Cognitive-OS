# Raw Output: Perplexity Sonar

- Provider/tool: Perplexity via OpenRouter
- Model ID: `perplexity/sonar-pro`
- Access date: 2026-05-27
- Collection timestamp UTC: 2026-05-27T02:25:39+00:00
- Prompt: standard prompt in `../prompts.md`
- Human edits: none beyond Markdown metadata wrapper and whitespace preservation
- OpenRouter response ID: `gen-1779849077-8sTXk7EUz9sQeu3Y1EuE`
- Usage reported by provider: `{"completion_tokens": 2600, "completion_tokens_details": {"audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0}, "cost": 0.04619, "cost_details": {"upstream_inference_completions_cost": 0.045002, "upstream_inference_cost": 0.04619, "upstream_inference_prompt_cost": 0.001188}, "is_byok": false, "prompt_tokens": 396, "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0, "video_tokens": 0}, "total_tokens": 2996}`

---

1. **Architecture thesis (3–5 sentences)**  
Cognitive-OS should be a **local-first cognitive control plane** that orchestrates multiple models and tools under a single, auditable loop of goals → hypotheses → evidence → governed actions. It treats the “AGI direction system” as a **runtime for reasoning processes**, not as a single model, with memory, planning, and tool use all mediated through explicit state and policies. The core thesis: all consequential behavior must flow through **evidence-governed plans**, **verifier-gated tools**, and **persistent world/self state** that can be audited, replayed, and selectively rolled back. Architecturally, it should resemble a **database-backed OS for cognition**: event-sourced, policy-enforced, tool-extensible, and safe to iterate on while remaining far from actual AGI capability.

---

2. **Memory architecture**

Goal: explicit, inspectable memory subsystems with clear lifecycles and limits, suitable for controlled long-horizon behavior.

**2.1 Working memory**

- **Representation**  
  - Structured **Working Memory Frame (WMF)** as a typed object:  
    - `goal_context` (current top-level and subgoals)  
    - `current_hypotheses` with confidence estimates  
    - `evidence_view` (subset of episodic/semantic entries relevant to current goals)  
    - `plan_state` (active plan, next steps, status)  
    - `scratchpad` (LLM-visible scratch text + structured fields, e.g., intermediate calculations).  
- **Scope & lifetime**  
  - Bound to a **deliberation episode** (e.g., one “task” spanning multiple tool calls).  
  - Evicted or snapshot to episodic memory at task end or after time/size thresholds.  

**2.2 Episodic memory**

- **Unit**: an **Episode Record**:
  - `id`, `timestamp`, `actors` (agents/humans), `goals`, `actions_taken`, `tools_used`, `inputs`, `outputs`, `outcome` labels, `side_effects`, `policy_decisions`.  
- **Storage**: append-only **event log** plus **episode index** (SQL/OLAP table).  
- **Indexing**:
  - Symbolic tags (task IDs, entities, tools, domains).
  - Embedding index for natural-language similarity retrieval.
  - Causal links: `precedes`, `contributes_to`, `contradicts`, `reverted_by`.  

**2.3 Semantic/procedural memory**

- **Semantic layer**  
  - **Typed knowledge graph** with entities (files, users, APIs, projects), relations (depends_on, owns, governed_by), plus soft attributes (reliability, last_validated_at).  
  - Backed by a graph DB or a graph-on-relational schema.  
- **Procedural layer**  
  - **Skill descriptors**: templates for recurring workflows (e.g., “safe code edit,” “data export with PII scrub”), each with:
    - Preconditions (checked against world state).
    - Expected steps & tools.
    - Risk profile and required verifier gates.  
  - Stored as versioned YAML/JSON, treated as **code**, with tests and reviews.

**2.4 Retrieval**

- **Multi-stage retrieval**:  
  1. Symbolic filter (by domain, entities, time window, risk level).  
  2. Vector search over episodic & semantic summaries.  
  3. LLM-based re-ranking/summarization into a **Context Package** that fits model limits.  
- Retrieval is **goal-conditioned**: query templates are generated from the current WMF and task type, not free-form.  

**2.5 Consolidation**

- Nightly or idle-time **consolidation jobs**:
  - Cluster similar episodes, infer stable semantic facts (e.g., “Tool X often times out; reliability=low”).  
  - Promote high-value patterns into procedural memory (candidate skills) under human review.  
  - Create compressed “chapter summaries” for long-running projects.  
- Consolidation steps are **one-way** into higher-level structures and always **logged** with provenance.  

**2.6 Forgetting**

- **Policy-driven forgetting**:
  - Time-based decay for low-value episodes (keep summaries, drop raw details).  
  - Legal/privacy-based purging: explicit delete tokens propagate through episodic, semantic, and embeddings.  
- **Soft forgetting**:
  - Reduce retrieval priority instead of deleting: a `retrieval_weight` that decays unless reinforced by use.  
- Forgetting operations are **audited** so humans know what was removed or down-weighted.

---

3. **Reasoning and planning loop**

Core loop per agent:

1. **Perception & goal intake**
   - Ingest events (user request, schedule triggers, environment changes).
   - Normalize into **Goal Objects** with: `goal_id`, `description`, `priority`, `constraints`, `deadline`, `risk_tier`.  

2. **Deliberation**
   - Build/refresh WMF via retrieval.
   - Run **Deliberation Policy**:
     - Choose reasoning mode (fast heuristic vs. deep plan).
     - Decide whether to escalate, ask clarifying questions, or refuse.  

3. **Planning**
   - Use a planning LLM + symbolic planner:
     - Generate candidate plans as sequences/trees of **Plan Steps** (tool calls, subgoals, checks).  
     - Enforce **Plan Schema** constraints per risk tier:  
       - Low risk: direct tool call allowed.  
       - Medium: plan must include explicit verification steps.  
       - High: plan must include human approval nodes and sandbox-only actions.  
   - Plans represented as explicit DAGs, stored in the event log before execution.  

4. **Execution & interrupts**
   - Execute plan step-by-step:
     - Before each step: run **preconditions** and **verifier gates** (e.g., static analyzers, safety filters).  
     - After each step: log outputs and update WMF.  
   - **Interrupts**:
     - External: user cancel/pause; policy revocation of a permission.  
     - Internal: detector flags anomalies (unexpected diff size, policy violation, high uncertainty).  
   - Interrupt handling: freeze plan, snapshot WMF, escalate or roll back side effects if supported.

5. **Reflection**
   - After plan completion or abort:
     - Generate structured **Reflection Record**:
       - What worked, what failed, surprises vs. expectations, updated hypotheses.  
     - Feed into consolidation and learning mechanisms.
   - For long tasks, mid-run reflection checkpoints are scheduled (time or event-based).

6. **Uncertainty handling**
   - All hypotheses and forecasts carry **explicit confidence bands** (e.g., 0–1 or discretized levels).  
   - Policy: above certain uncertainty thresholds, the agent must:
     - Seek more evidence, or  
     - Present options to a human with trade-offs, or  
     - Defer/refuse.

---

4. **Learning or self-improvement**

**4.1 Allowed update surfaces**

- **LLM weights**: *read-only* in the core runtime. No online fine-tuning in production.  
- **Safe update surfaces**:
  - Retrieval indices and embeddings.  
  - Semantic/procedural memory (knowledge graph and skills).  
  - Tool and plan heuristics (non-critical) behind gates.  
  - Per-agent configuration (preferences, prompt templates).  

**4.2 Offline learning**

- **Batch pipelines** run off the production event log:
  - Train/improve retrievers, ranking heuristics.  
  - Suggest new skills or plan templates based on frequent patterns.  
  - Evaluate alternative prompts or routing strategies on **frozen historical traces**.  
- All candidate changes are staged in a **sandbox environment** with replay of real traces.

**4.3 Eval-gated changes**

- Any change that alters:
  - Tool configurations,  
  - Plan schemas, or  
  - Safety policies  
  must pass:
  - Unit tests (behavioral specs).  
  - Regression tests on historical traces.  
  - Safety & governance checks (no expansion of capability beyond declared boundaries without sign-off).  

**4.4 Rollback**

- Every configuration, skill, and policy is **versioned**:
  - Changes are applied via migrations with one-step or multi-step rollback scripts.  
  - Event log is immutable; only **interpretation** changes, never past records.  
- Rollback scenarios:
  - Auto-rollback on anomalies (e.g., error rate spike, repeated policy near-misses).  
  - Manual rollback by operator with clear UI.

---

5. **Tool use and action execution**

**5.1 Sandboxing**

- Tools executed in **isolated OS-level sandboxes** (containers / VMs):
  - File access via **virtual filesystem** with whitelisted paths.  
  - Network access via **egress proxy** with allowlist and rate limits.  
- Each tool call carries a **capability token** that encodes allowed operations.

**5.2 Permissions**

- Permissions are not implicit; they are:
  - Described in a **capability schema** for each tool.  
  - Granted per-agent and per-goal (e.g., Agent A may edit code in Repo X but only read in Repo Y).  
- High-risk actions require **multi-step approval** (agent plan → human review → runtime revalidation).

**5.3 Verifier gates**

- Before execution:
  - Input validators, policy filters (PII checks, dangerous API usage), and static analyzers run.  
- After execution:
  - Diff analyzers (e.g., for code or data modifications).  
  - Post-conditions: expected invariants (tests must pass, schema must hold).  
- Gates are **composable**: a tool invocation can be wrapped by multiple verifiers.

**5.4 Side-effect ledger**

- Every side-effecting action writes a **Side-Effect Entry**:
  - `id`, `actor`, `tool`, `parameters`, `target_resources`, `diff`, `timestamp`, `approvals`, `verifiers_passed`, `rollback_handle` (if supported).  
- Ledger is separate from app data stores, so it persists even if target systems change.  
- Used for:
  - Auditing and accountability.  
  - Automated rollback orchestration (undo scripts or synthetic revert actions).  

---

6. **World model and representation layer**

**6.1 Entities**

- Core model: **typed entities**:
  - Physical/logical resources (files, tables, services, users, agents, policies).  
  - Each with attributes, ownership, and **governance tags** (confidentiality, criticality).  

**6.2 Causal models**

- Represent:
  - `Action → State Change → Outcomes` links based on past episodes.  
  - Dependency structures (e.g., “code in service A affects API B, which affects client C”).  
- Implementation: relational + graph edges for causal links, plus learned causal scores derived from observing effects.

**6.3 Forecasts**

- Agents maintain **Forecast Objects** for relevant metrics:
  - e.g., “probability build will fail after this change,” “expected runtime of this notebook,” “chance of violating latency SLO.”  
- Forecasts are produced by:
  - Simple statistical models, or  
  - LLM-synthesized ranges grounded in evidence and causal links.  

**6.4 State synchronization**

- World model is not a mirror of all external systems; instead:
  - It stores **cached state** with `last_observed_at` timestamps.  
  - Before high-impact actions, the agent must **refresh state** via tools (e.g., git fetch, DB metadata, monitoring APIs).  
- Conflicts:
  - If world model is stale beyond policy limits, the action is blocked or downgraded to read-only inspection.  

---

7. **Safety/governance**

**7.1 Policy engine**

- Central **policy engine** operates on:
  - Subject (agent),
