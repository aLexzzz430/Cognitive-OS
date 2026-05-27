# Raw Output: OpenAI GPT

- Provider/tool: OpenAI via OpenRouter
- Model ID: `openai/gpt-5.4-mini`
- Access date: 2026-05-27
- Collection timestamp UTC: 2026-05-27T02:25:39+00:00
- Prompt: standard prompt in `../prompts.md`
- Human edits: none beyond Markdown metadata wrapper and whitespace preservation
- OpenRouter response ID: `gen-1779848740-c5o4NyAhVIgVX1YTosx1`
- Usage reported by provider: `{"completion_tokens": 2600, "completion_tokens_details": {"audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0}, "cost": 0.012003, "cost_details": {"upstream_inference_completions_cost": 0.0117, "upstream_inference_cost": 0.012003, "upstream_inference_prompt_cost": 0.000303}, "is_byok": false, "prompt_tokens": 404, "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0, "video_tokens": 0}, "total_tokens": 3004}`

---

## 1. Architecture thesis in 3-5 sentences

Cognitive-OS should be organized as a governed control plane for agentic work, not as a single monolithic “smart model.” The core design principle is to separate **intent, evidence, world state, policy, and execution** so every action is explainable, gateable, and recoverable. The runtime should treat the LLM as a probabilistic proposal engine routed through verifier gates and local adapters, with persistent memory and planning state managed outside the model. This makes the system incrementally capable: better models can be swapped in, but safety, auditability, and recovery remain stable.

## 2. Memory architecture: working, episodic, semantic/procedural, retrieval, consolidation, forgetting

**Working memory**
- Holds the active task frame: current goal, subgoals, constraints, open questions, and the next action candidates.
- Implement as an in-memory, versioned “scratchpad state” with small bounded size and explicit TTL.
- Structure:
  - `goal_stack`
  - `active_hypotheses`
  - `current_context`
  - `recent_observations`
  - `pending_actions`
  - `uncertainty_flags`

**Episodic memory**
- Append-only event journal of what happened, when, why, and with what evidence.
- Each episode should record: trigger, context snapshot, model proposal, policy decision, tool call, outcome, and post-hoc evaluation.
- Store as structured event objects plus optional embeddings for retrieval.

**Semantic memory**
- Stable facts, definitions, invariants, preferences, user-specific settings, and learned domain knowledge.
- Only promote items from episodic memory after consolidation and verification.
- Use typed records with provenance:
  - `fact`
  - `preference`
  - `constraint`
  - `procedure`
  - `policy_exception`
  - `domain_ontology`

**Procedural memory**
- Reusable task strategies, action sequences, tool recipes, and planning templates.
- Represent as parameterized workflows or graph fragments rather than raw text.
- A procedural item should include preconditions, expected effects, failure modes, and rollback steps.

**Retrieval**
- Hybrid retrieval: keyword + semantic similarity + graph traversal + recency.
- Retrieval should be query-scoped by task and policy.
- Every retrieved item should carry confidence and provenance metadata.
- The planner must distinguish:
  - direct evidence
  - inferred memory
  - stale memory
  - policy-derived constraints

**Consolidation**
- Batch process that promotes episodes into stable memory only after:
  - consistency checks
  - duplicate merging
  - contradiction detection
  - utility scoring
  - policy review
- Consolidation should happen in a low-priority background worker, never in the critical action path.
- Prefer “evidence-backed compression”: summarize repeated episodes into minimal stable abstractions.

**Forgetting**
- Explicit forgetting is a feature, not a failure.
- Use retention classes:
  - ephemeral working state: minutes/hours
  - episodic: days/weeks
  - semantic/procedural: until invalidated
- Support decay based on age, low utility, contradiction, sensitivity, and storage budgets.
- Sensitive or high-risk items should have shorter retention and stricter access.

## 3. Reasoning and planning loop: deliberation, planning, reflection, uncertainty, interrupts

**Deliberation**
- The system begins by classifying the request: informational, task execution, open-ended planning, or safety-sensitive.
- It then forms a task frame with assumptions, known constraints, and missing information.
- Deliberation is structured as hypothesis generation, not freeform text only.

**Planning**
- Use hierarchical planning:
  1. identify objective
  2. decompose into subgoals
  3. choose candidate plans
  4. estimate cost, risk, and confidence
  5. select a plan or request more information
- Plans should be explicit graphs with nodes for actions and checkpoints.
- Each node should include:
  - expected effect
  - required permissions
  - verification method
  - fallback/rollback

**Reflection**
- Reflection is a separate pass after tool calls or evidence changes.
- The system compares expected vs observed outcomes, updates beliefs, and decides whether to continue, repair, or abort.
- Reflection should be triggered by:
  - failed verification
  - contradictory evidence
  - repeated plan loops
  - confidence drops
  - user interrupt

**Uncertainty**
- Maintain explicit confidence on claims, forecasts, and plan steps.
- Use uncertainty bands rather than a single score when possible.
- When uncertainty crosses a threshold, the system should:
  - ask a question
  - narrow scope
  - switch to conservative mode
  - escalate to human review

**Interrupts**
- Interrupts must preempt the current plan safely.
- The runtime should preserve:
  - current state snapshot
  - pending side-effect ledger entries
  - plan cursor
- On resume, revalidate assumptions because the world may have changed.

## 4. Learning or self-improvement: allowed update surfaces, offline learning, eval-gated changes, rollback

**Allowed update surfaces**
- Prompt templates
- Retrieval ranking weights
- Procedure libraries
- Domain ontologies
- Policy rules and exception tables
- Tool routing heuristics
- Compression/summarization policies
- Task-specific caches

**Not directly mutable by agent runtime**
- Core safety policy without approval
- Model weights in production path
- Permission boundaries
- Audit log contents

**Offline learning**
- Learning should happen in an offline or quarantined pipeline fed by logged episodes.
- Candidate updates are trained or synthesized from:
  - successful task traces
  - failure traces
  - user corrections
  - benchmark regressions
- Prefer constrained learning methods that produce small, reviewable deltas.

**Eval-gated changes**
- No update ships without passing a gated evaluation suite.
- Gating should test:
  - task success improvement
  - regression rate
  - policy compliance
  - tool safety
  - memory corruption risk
  - recovery behavior
- The gate must compare against a pinned baseline.

**Rollback**
- Every deployed change needs a rollback artifact:
  - previous version
  - diff
  - reason for change
  - test evidence
- Support instant rollback of routing rules, prompt configs, memory schemas, and procedures.
- Learning outputs should be versioned and replayable.

## 5. Tool use and action execution: sandboxing, permissions, verifier gates, side-effect ledger

**Sandboxing**
- Run all external actions through adapters with strict capability scopes.
- Local machine access should be partitioned by task and role.
- Sensitive actions should execute inside managed-VM or container boundaries with restricted filesystem, network, and device access.

**Permissions**
- Use least privilege and time-bounded leases.
- Permissions should be explicit per tool and per operation class:
  - read
  - write
  - execute
  - network
  - payment
  - credential access
- Escalation must require policy approval or human authorization.

**Verifier gates**
- Before action: check policy, permissions, preconditions, and risk.
- After action: verify side effects against expected outcomes.
- For high-risk actions, require multi-stage verification:
  - plan review
  - preflight
  - execution
  - post-check
- Verifiers should be independent components, not the same LLM prompt.

**Side-effect ledger**
- Every external effect must be logged as an intent/effect pair:
  - requested action
  - authority used
  - time
  - target
  - observed result
  - rollback status
- The ledger should support idempotency checks and compensating actions.
- If a side effect cannot be verified, mark it as “unconfirmed” and trigger follow-up.

## 6. World model and representation layer: entities, causal models, forecasts, state sync

**Entities**
- Represent the world as typed entities with IDs, attributes, relations, and provenance.
- Entities can include:
  - users
  - files
  - processes
  - projects
  - goals
  - tools
  - policies
  - hypotheses
- The entity layer should support partial observability.

**Causal models**
- Maintain causal graphs for domains where actions have predictable effects.
- A causal record should include:
  - preconditions
  - intervention
  - expected consequences
  - known failure cases
- The model need not be perfect; it must be useful for planning and risk estimation.

**Forecasts**
- Forecasts should be first-class objects with horizon, confidence, dependencies, and revision history.
- The system should compare predicted vs observed outcomes to calibrate itself.
- Short-horizon forecasts can drive tool execution; long-horizon forecasts should be conservative.

**State sync**
- Sync world state from:
  - local sensors/adapters
  - user input
  - tool results
  - event logs
- Use conflict-aware merges when multiple sources update the same entity.
- Preserve provenance on every field so the system can resolve disagreements rather than overwrite them blindly.

## 7. Safety/governance: policy engine, human controls, auditability, refusal/escalation, capability boundaries

**Policy engine**
- Central policy engine evaluates every plan and tool action.
- Policies should be machine-readable and versioned.
- Support rule types for:
  - forbidden actions
  - required approvals
  - data handling constraints
  - domain-specific limitations
  - emergency stop conditions

**Human controls**
- Human operators must be able to:
  - approve or deny actions
  - pause the runtime
  - inspect the plan
  - revoke permissions
  - reset memory scopes
  - pin or delete records where allowed
- Provide a “safe review mode” that exposes evidence and planned effects before execution.

**Auditability**
- Every decision should be reconstructable from logs:
  - input
  - retrieved evidence
  - model outputs
  - policy checks
  - tool calls
  - final outcome
- Audit views should separate raw logs from summarized explanations.

**Refusal/escalation**
- If policy blocks an action, the system should refuse, explain the reason, and propose safe alternatives.
- If uncertainty or risk exceeds threshold, escalate to a human or narrower subtask.
- Refusal should be stable across retries unless the underlying facts change.

**Capability boundaries**
- The runtime must enforce that capability is determined by permissions, not by model confidence.
- High-impact domains need stricter leases, more verifiers, and narrower adapters.
- The system should degrade gracefully into read-only or advisory mode.

## 8. Evaluation and benchmark strategy: unit tests, integration tests, red teams, long-run soak, regression gates

**Unit tests**
- Test policy rules, memory operations, planner transitions, retrieval ranking, and tool adapter contracts.
- Include tests for malformed inputs, stale state, and contradictory evidence.

**Integration tests**
- End-to-end tasks that combine planning, memory, tool use, and verification.
- Scenarios should cover:
  - file operations
  - calendar/task management
  - local data analysis
  - controlled VM actions
  - interrupted/resumed workflows

**Red teams**
- Design adversarial cases for:
  - prompt injection
  - tool misuse
  - memory poisoning
  - overconfident planning
  - policy bypass attempts
  - ambiguous user intent
- Red team outputs should become regression cases.

**Long-run soak**
- Run the system for days/weeks on benign workloads to detect drift, memory bloat, leak paths, and recovery failures.
- Measure action success rate, verification failure rate, false refusals, and stale memory incidence.

**Regression gates**
- Any code, prompt, policy, or retrieval change must pass:
  - functional tests
  - safety tests
  - benchmark tasks
  - replay of historical failures
- A failed gate blocks deployment unless manually waived with recorded justification.

## 9. Persistence/runtime architecture: event log, state store, leases, runtime modes, recovery

**Event log**
- Append-only canonical log of all significant state transitions.
- Use it as the source of truth for replay, audit, and reconstruction.
- Events should be immutable and schema-versioned.

**State store**
- Materialized views built from the log:
  - current goals
  - memory indexes
  - policy state
  - active leases
  - world model snapshot
