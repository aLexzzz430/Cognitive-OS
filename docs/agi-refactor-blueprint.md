# AGI Refactor Blueprint

This document is a structural correction, not a branding document. The current
Con OS codebase has accumulated strong agent/runtime engineering, but several
capabilities now pull the system away from AGI work:

- local-machine coding repair dominates the loop;
- VM, patch gates, OAuth, service packaging, and CLI reliability dominate the
  roadmap;
- benchmarks are mostly coding-fixture and open-repo oriented;
- task discovery is about project maintenance more than intrinsic goal
  formation;
- hypothesis lifecycle exists, but it is most concrete inside the
  local-machine adapter rather than as a domain-neutral cognitive primitive.

Those are useful capabilities, but they are not the AGI core. The refactor goal
is to move them out of the center and make them tools used by a domain-neutral
cognitive organism.

## Current Misalignment

| Existing emphasis | Why it is useful | Why it is not the AGI center |
| --- | --- | --- |
| Local-machine coding tasks | Gives concrete verifier-gated work | Overfits cognition to code repair and pytest |
| Patch proposal | Creates bounded source changes | Treats intelligence as editing repositories |
| VM and mirror execution | Gives a body boundary and safety | It is embodiment infrastructure, not cognition |
| Provider/model routing | Controls cost and model reliability | Models are organs/tools, not the organism |
| Runtime service packaging | Enables long-running behavior | Uptime is necessary but not sufficient for agency |
| Task discovery | Creates autonomous maintenance tasks | Still goal-gap maintenance, not general motivation |
| Closed-loop probes | Prove a local mechanism chain | Mostly controlled coding fixtures, not broad transfer |

## New Center

The center of Con OS should be a domain-neutral AGI Core:

```text
Drive / Goal System
  -> Attention / Situation Model
  -> World Model + Self Model
  -> Memory Retrieval
  -> Hypothesis Formation
  -> Experiment Design
  -> Action Selection
  -> Governed Actuation
  -> Verification / Outcome Appraisal
  -> Learning / Consolidation
```

No item in that chain should be inherently about files, patches, tests, GitHub,
VMs, or LLM providers. Those belong below the core as tools, bodies, or
evaluation environments.

## Target Layering

### Layer 0: Substrate

Durable storage, event journal, object store, leases, runtime status, resource
watchdog, and VM/mirror boundaries.

This layer answers: "Can the organism stay alive and act safely?"

### Layer 1: Body and Environment Adapters

Local machine, browser, internet, VM, filesystem, terminal, external APIs,
coding repositories, simulated worlds, future robotic or UI surfaces.

This layer answers: "What can the organism sense and affect?"

### Layer 2: AGI Core

Domain-neutral cognition:

- goals and drives;
- attention and salience;
- world model;
- self model;
- episodic and semantic memory;
- hypotheses and causal models;
- experiments and curiosity;
- action policy;
- outcome appraisal;
- learning and consolidation.

This layer answers: "What does the organism believe, want, try, learn, and
avoid?"

### Layer 3: Skills

Reusable procedures learned or installed over time:

- code repair;
- research;
- reading;
- market analysis;
- web navigation;
- debugging;
- planning;
- summarization;
- tool-specific routines.

This layer answers: "What reusable abilities can the organism call?"

### Layer 4: Governance and Ethics

Capability ceilings, approval gates, refusal policy, budget policy, evidence
requirements, user boundaries, sync-back policy, credential control, and network
policy.

This layer answers: "What may the organism do?"

### Layer 5: Evaluation

Controlled probes, open tasks, external baselines, long-run tests, transfer
tests, robustness tests, and refusal tests.

This layer answers: "What can the organism really do, and under what evidence?"

## Refactor Rules

1. Local-machine code must stop defining cognition.
   - It may provide evidence and actions.
   - It must not be the only place where hypotheses, experiments, and learning
     become real.

2. LLM calls must stop being the cognitive identity.
   - LLMs are reasoning organs.
   - The system state must be explicit: evidence, hypotheses, decisions,
     actions, confidence, failure boundaries.

3. Runtime reliability must support cognition, not replace it.
   - 7/24 runtime matters because an AGI-like system needs continuity.
   - Uptime alone is not intelligence.

4. Goal discovery must become motivation-like.
   - Current task discovery is maintenance-oriented.
   - It should evolve into goal pressure, curiosity, anomaly detection,
     capability growth, user alignment, and survival/continuity needs.

5. Skills must be learned objects.
   - A successful workflow should become a skill candidate.
   - A failure should become a failure mode, violated assumption, and future
     test or constraint.
   - Self-model capability estimates should create task pressure: unreliable
     actions produce investigation tasks, while repeated verified successes
     produce skill-candidate tasks with verifier boundaries.

6. Benchmarks must test transfer.
   - Coding repair remains one domain.
   - AGI progress requires unfamiliar domains, hidden rules, multi-step
     exploration, ambiguous goals, tool transfer, long-horizon memory, and
     refusal under uncertainty.

## Demotions

These existing areas should be demoted from "core intelligence" to support
roles:

- `integrations/local_machine/`: body adapter for code/files/tests.
- `modules/local_mirror/`: safety boundary and actuation substrate.
- `core/auth/` and `modules/llm/`: model organs and credentials.
- `experiments/open_task_benchmark/`: one evaluation family, not the main
  definition of progress.
- patch proposal and target binding: code-repair skills, not general cognition.

## Promotions

These areas should be promoted into explicit AGI Core responsibilities:

- `self_model/`: from diagnostics to living self-state.
- `core/world_model/` and `modules/world_model/`: from object graph utilities
  to the central world-state substrate.
- `core/reasoning/`: from task heuristics to domain-neutral hypothesis and
  experiment machinery.
- `core/runtime/evidence_ledger.py`: from audit artifact to cognitive memory.
- `core/task_discovery/`: from maintenance queue to drive/goal discovery.

## P0 Refactor Plan

1. Add an explicit AGI Core contract.
   - Define domain-neutral cycle records: goal, situation, belief, hypothesis,
     experiment, action intent, observation, outcome, learning update.
   - No local-machine terms allowed in the core contract.

2. Move local-machine hypothesis construction behind an adapter boundary.
   - The adapter may translate file/test evidence into generic hypotheses.
   - The core must own hypothesis lifecycle semantics.

3. Split goals from tasks.
   - Goals represent persistent pressure.
   - Tasks are temporary executable commitments.
   - Current `TaskDiscoveryEngine` should become a producer of candidate tasks
     from deeper goal pressure, not the source of agency.
   - `core.cognition.goal_pressure` now converts self-model learning outcomes
     into runtime `goal_stack` pressure before they become task candidates.
   - `UnifiedContextBuilder` and `GoalTaskRuntime` now preserve structured
     `goal_stack.subgoals`, rank active pressure by priority, and promote the
     highest active pressure to the current task when no explicit task is
     present.

4. Make self-model and world-model first-class runtime inputs.
   - Every tick should receive both.
   - Every outcome can update both.
   - Both models must be able to change action ranking, penalties, and
     viability before governance selects an action.
   - The first implementation hook is:
   - `core.cognition.model_influence`: world/self state changes candidate
     scores and blocks before execution.
   - `core.cognition.outcome_model_update`: action outcomes update
     `world_summary`, `self_summary`, and `learning_context` during evidence
     commit.
   - `core.cognition.goal_pressure`: failed capabilities and repeated verified
     successes become active goal pressure for repair or skill evaluation.

5. Add AGI-oriented eval families.
   - Hidden rules.
   - Novel tool use.
   - Long-horizon memory.
   - Cross-domain transfer.
   - Ambiguous objective refusal.
   - Self-repair without code-only assumptions.

## What Not To Do

- Do not rename coding-agent features to AGI.
- Do not add more patch heuristics.
- Do not keep proving intelligence only with repository repair.
- Do not treat VM progress as AGI progress.
- Do not make LLM provider support the center of the architecture.
- Do not claim complete AGI before broad, repeatable, adversarial evidence.

## First Acceptance Gate

Before continuing broad product work, Con OS should pass this architectural
gate:

- A domain-neutral AGI Core contract exists.
- World-model and self-model state can influence action selection before an
  adapter executes a tool.
- At least two non-code environments can produce the same core event types.
- Hypothesis lifecycle works outside local-machine.
- Goal pressure is separate from task execution.
- Self-model updates affect future action selection.
- Failure learning changes future behavior in a measurable way.

Until that gate is passed, Con OS should be described as an AGI-directed
cognitive runtime under refactor, not a complete AGI.
