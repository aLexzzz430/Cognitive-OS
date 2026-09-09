# Findings from the collected proposals

This is a qualitative review of one generation per checkpoint under the exact common prompt. Every system/dimension assessment is traceable to a quote in [comparison.csv](comparison.csv). Quotes are evidence of what a model wrote, not evidence that its claim is true. There is no numerical model leaderboard.

## Patterns worth retaining

**Separate proposals from authorized effects.** Spark describes permission-bound proposed modifications; Granite describes a centralized authorization point; Phi describes a modular governed executor. Those are useful prompts for a concrete interface boundary. They largely echo constraints already supplied to the models. The synthesis makes that boundary explicit with immutable action content, current policy version, file hashes and verifier evidence; those details are analyst design choices, not a claimed verbatim model consensus.

**Start with a small planner and a durable state store.** Qwen's heuristic selection and TinyLlama's FIFO queue can be useful simple baselines. FIFO is scheduling rather than reasoning, and neither proposal proves planning quality. A single bounded worker and SQLite avoid adding distributed coordination before measuring its value. Falcon's on-disk state separates persistence from transient computation more clearly than Granite's in-memory state proposal.

**Evaluate observable effects.** Phi suggests completion, error and resource metrics; Granite explicitly names final-file consistency. These can become falsifiable tests once workloads, crash points and pass criteria are supplied. No collected response by itself establishes a correct recovery implementation. The isolated experiment in this packet adds concrete evidence about two toy algorithms, not about the repository's existing recovery path.

## Disagreements and failures that change implementation decisions

| Topic | Observations | Decision |
|---|---|---|
| Durable memory | Spark calls part of GPU memory persistent; Granite proposes in-memory SQLite; Qwen/Phi/Falcon describe SQLite or disk state | Keep durable state on disk and treat GPU allocation independently. Do not infer durability from a memory diagram. |
| Learning | Several outputs invoke RL; Spark describes approved proposed changes | Begin with offline candidate evaluation and explicitly preserve authority. None supplies enough evidence for autonomous online self-modification. |
| Recovery | Spark snapshots “on crash”; Granite logs state “in case of a crash”; Phi lists task-state steps without placing the actual file effect | Persist before the effect and reconcile after restart. An abrupt process death cannot be relied on to run a final snapshot/log handler. |
| Coordination | TinyLlama requests distributed coordination/DFS; Granite proposes negotiation; other outputs name generic multi-agent layers | Use one writer and bounded queues initially. Communication does not define ownership, conflict handling or recovery. |
| Hardware scope | Falcon proposes cloud scaling and quantum elements | Exclude those from the local 12 GB design. Unavailable resources cannot serve as a feasibility argument. |
| Empirical evidence | TinyLlama asserts over 100,000 tasks/s and over 90% success with no source or experiment | Mark both claims unsupported and do not carry them into the synthesis or PR as benchmark results. |
| Novelty | Qwen repeats simple rules; SmolLM names “insight generation”; Granite repeats the supplied verifier gate | No novelty is established by those labels. Prefer small, testable integration decisions over a new architecture name. |

Spark, SmolLM and Danube hit the token cap. Spark reaches an “Implemented Components” list and stops mid-sentence; SmolLM stops just as it introduces its worked example; Danube stops after introducing its baseline comparison. Their missing endings cannot be interpreted as deliberate design choices. Qwen, Falcon and TinyLlama terminate without the requested concrete crash/schema/experiment combination. Phi provides a SQL table but omits action identity, expected file hashes and authorization in that table. Granite supplies an example and an experiment but no explicit schema/interface. Danube suggests a state machine and decoupled planning, but its backup-based crash example lacks effect ordering or an actual schema; its proposed probabilistic reasoning is not a specified algorithm.

Phi's “Implemented Components” list includes an RL module and graph world representation without checking this repository. Spark makes a similarly unsupported implementation claim. The analysis treats such text as model output; actual repository observations in synthesis.md are separately linked to source files at the inspected commit.

## What the study cannot establish

The samples do not show that small local models generally fail at architecture work, that one family is better than another, or that the resulting design constitutes AGI. Shared prompting, ancestry, training overlap, tokenization and analyst judgment confound such conclusions. Exact token IDs and local model hashes improve auditability; they do not eliminate those methodological limits. There is no independent originality search or formal verification here.

The recommended next engineering step is a bounded integration experiment with verified file effects and explicit authority, followed by measured comparison against a simple planner. Training, additional agents and richer world models should have separate hypotheses and budgets. A long answer or eleven headings alone is not a useful acceptance test for an architecture proposal.
