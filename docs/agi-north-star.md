# AGI North Star

Cognitive OS is pivoting away from an "LLM amplifier" architecture and toward
an AGI-directed runtime. This is a structural refactor, not a messaging change.
The current system has useful agent/runtime capabilities, but they are too
centered on local-machine coding repair, patching, VM execution, and provider
routing. Those capabilities must become tools around a domain-neutral cognitive
core.

For the concrete refactor boundary, see `docs/agi-refactor-blueprint.md`.

## North Star

Con OS should become a local-first, evidence-governed general intelligence
runtime that can maintain goals, form hypotheses, run discriminating
experiments, act through governed tools, verify outcomes, learn from failure,
and improve its own future behavior without bypassing safety boundaries.

## Definition Boundary

Con OS may only claim AGI-level progress when the claim is backed by repeatable
evidence. The system must not use "AGI" as a marketing shortcut for a wrapper
around one model or one coding workflow.

Current supported claim:

- Con OS is an AGI-directed cognitive runtime prototype.

Unsupported claim:

- Con OS is already a complete AGI.

## Required Capabilities

1. Goal maintenance
   - Durable North Star and active goals.
   - Explicit constraints and forbidden actions.
   - Gap detection tied to evidence, not vibes.
   - Learning pressure from the self-model must enter runtime goals before it
     becomes an executable task.
   - If no explicit user task is active, the highest-priority active learning
     pressure can become the current task focus.

2. World and self model
   - Separate state for external world facts, local machine facts, and internal
     runtime health.
   - Self-model records limits, failures, costs, tool reliability, model
     reliability, and recovery ability.
   - World-model and self-model state must affect action selection, not only
     appear in status reports.
   - Action outcomes must feed back into both models so the next tick can
     change confidence, risk, capability estimates, and adaptation pressure.

3. Hypothesis lifecycle
   - Multiple competing hypotheses.
   - Discriminating experiments bound to hypotheses.
   - Posterior updates that mutate real hypothesis state.
   - Patch/action selection linked to leading hypotheses.

4. Evidence ledger
   - Claim to evidence to action to result to update.
   - Formal outcomes are preferred over chat memory.
   - Failures become future retrieval objects and regression tasks.

5. Governed action
   - Read, propose, execute, network, credential, and sync-back permissions.
   - VM or mirror boundary for side effects.
   - Verifier gates before completion or source sync.
   - Human approval as a first-class state.

6. Learning and autonomy
   - Autonomous task discovery from failures, goal gaps, user signals, code
     health, hypothesis debt, and opportunities.
   - Self-model learning pressure from action reliability and verified
     successes.
   - Failed capabilities become repair goals; repeated verified successes
     become skill-evaluation goals.
   - High-recall discovery and high-precision execution.
   - Long-running operation with recovery and budget policies.

7. Model/tool substrate
   - Provider-agnostic model routing.
   - Cheap deterministic layer first.
   - Small models for monitoring and formatting.
   - Strong models only for uncertainty, disagreement, or high-risk decisions.

## Capability Ladder

| Level | Name | Requirement |
| --- | --- | --- |
| G0 | Cognitive runtime | Runs governed tasks with evidence, verifier gates, and audit. |
| G1 | Autonomous maintainer | Discovers, scores, and executes bounded improvement tasks in a mirror/VM. |
| G2 | Cross-domain operator | Handles unfamiliar repositories and task types without fixture-specific heuristics. |
| G3 | Persistent researcher | Forms research hypotheses, runs experiments, updates beliefs, and creates reusable skills. |
| G4 | AGI claim candidate | Demonstrates repeatable broad autonomy, transfer, self-repair, safe action governance, and external baselines across domains. |

The current project should be treated as G0 with partial G1/G2 evidence. It
must not be represented as G4 until the evidence exists.

## Immediate Direction

The next engineering work should prioritize:

- Active goal ledger as a first-class runtime object.
- Self-model and world-model separation.
- Failure-to-learning pipeline that creates durable future behavior changes.
- Long-running autonomous task discovery with conservative execution gates.
- No-user-instruction ticks as a normal runtime path: when the system is awake
  but has no active run, it can promote safe L0/L1 goal pressure into an
  auditable background task instead of treating idle time as a special case.
- Homeostasis pressure: if no explicit subgoal exists, the runtime can convert
  self-model failures, runtime anomalies, or elevated world-model uncertainty
  into a bounded read-only diagnostic task.
- Runtime behavior policy: self/world state now compiles into a concrete
  behavior envelope before action. It selects goal pressure, chooses
  `IDLE`/`ROUTINE_RUN`/`DEEP_THINK`/`WAITING_HUMAN`, adjusts model tier,
  thinking/budget posture, and applies permission ceilings.
- Homeostasis evidence commit: autonomous diagnostics write a run-local report
  and a formal evidence entry before they are marked verified, so idle-time
  recovery has an auditable object-layer trace.
- Pressure resolution policy: resolved or already-handled pressure is
  suppressed and deprioritized; persistent pressure escalates to
  `WAITING_HUMAN`, `DEEP_THINK`, or approval-gated limited L2 mirror
  investigation instead of repeating the same idle diagnostic forever.
- Failure learning as behavior: structured failure objects compile into
  preferred actions, avoided/blocked actions, governance constraints, regression
  tests, and retrieval objects. Future ticks can therefore change action ranking
  without treating old failures as chat memory.
- Open-ended task benchmarks against real external agents and local models.
- VM-backed execution as the default side-effect boundary.

## Non-Negotiables

- No AGI claim without repeatable evaluations.
- No silent host-side effects.
- No source sync without patch gate and verifier evidence.
- No hidden fallback patching after model timeout.
- LLM failures are converted into structured behavior decisions before the
  system retries, downgrades, escalates, or blocks. The default timeout decision
  is `return_structured_timeout`; configured model fallback is separate from
  deterministic patch fallback and is recorded as an auditable policy decision.
- No benchmark-specific patch heuristics as a substitute for generality.
- No replacing formal evidence with chat transcripts.
