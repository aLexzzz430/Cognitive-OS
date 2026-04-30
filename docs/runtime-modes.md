# Runtime Modes

Cognitive OS exposes a product-level runtime mode above the lower-level run,
task, heartbeat, watchdog, and local-machine investigation states. The mode now
drives scheduling, LLM budget, model selection, and permission ceilings; it is
not only a status label.

| Mode | Human analogy | Meaning |
| --- | --- | --- |
| `STOPPED` | off | The process is not running; only durable disk state remains. |
| `SLEEP` | low-power sleep | Minimal live standby. Heartbeat and housekeeping can run, but no autonomous thinking, LLM calls, tool execution, or side effects. |
| `IDLE` | awake and available | The runtime is online and ready. With no user instruction, it may schedule bounded L0/L1 autonomous goal-pressure work. |
| `ROUTINE_RUN` | normal work | Low-risk task progress through deterministic, fast-path, or small-model steps. |
| `DEEP_THINK` | high-quality thinking | High-uncertainty analysis, root-cause reasoning, planning, recovery, or final judgment. |
| `CREATING` | inspiration burst | Bounded generation of candidate plans, patches, experiments, or designs. It creates candidate futures; it does not execute them. |
| `ACTING` | doing | Tool execution, tests, file reads, patch application, VM actions, or mirror sync. |
| `DREAM` | offline dreaming | Soak, consolidation, profiling, learning, or maintenance that should not directly change the world. |
| `WAITING_HUMAN` | asking for help | The runtime is blocked for approval, human review, or ambiguous evidence. |
| `DEGRADED_RECOVERY` | tired or recovering | Watchdog degradation, VM boundary recovery, crash recovery, zombie suspicion, retry recovery, or runtime exception handling. |

`SLEEP` is intentionally not the same as `STOPPED`. A sleeping runtime is still
alive enough to write status heartbeats, accept wake commands, clean leases,
checkpoint SQLite, and monitor high-priority external events. It must not plan,
call LLMs, execute tools, fetch the network, or create autonomous tasks.

No-user-instruction ticks are normal in `IDLE`. The daemon reads the durable
goal stack, selects the highest-priority active subgoal that is L0/L1 and limited
to read/report/analysis actions, creates an auditable autonomous run, and then
lets the normal supervisor tick it. If the goal stack has no safe active
subgoal, the daemon may synthesize a homeostasis goal from self-model failures,
runtime anomaly flags, or high world-model uncertainty. Unsafe goals, side
effects, network, credential use, sync-back, and human-approval tasks are not
auto-started. Recently completed autonomous goals are throttled so an unchanged
state file does not churn duplicate runs.

Autonomous no-user tasks now have a real read-only completion path. The
supervisor executes a homeostasis diagnostic, writes
`runtime/runs/<run_id>/homeostasis_report.json`, records the report in the
formal evidence ledger, and only then marks the task verified. This keeps idle
autonomy inside the object/evidence layer instead of treating background work
as a synthetic completed task.

Homeostasis diagnostics also apply a pressure-resolution policy. If the
diagnosis shows the pressure is already known or resolved, the runtime records a
repeat-suppression window and lowers the priority of duplicate triggers. If the
pressure remains active, the supervisor escalates through a bounded path:
runtime anomalies or critical pressure enter `WAITING_HUMAN`; persistent
self-model failures request approval for a limited L2 mirror investigation; and
world-model uncertainty creates a read-only `DEEP_THINK` follow-up. These
transitions are written as audit events before any follow-up task can run.

`CREATING` is distinct from `DEEP_THINK`: thinking weighs explanations and
decisions, while creating generates bounded candidates that still require later
selection, verification, and governance before `ACTING`.

Every mode exposes a `mode_policy` in `conos status` and
`runtime_mode_catalog()`:

- `scheduler`: whether planner ticks may run, tick interval hints, and whether
  the mode is background-only.
- `llm_budget`: max model calls, prompt/completion tokens, wall-clock budget,
  retry count, and whether escalation is allowed.
- `model_selection`: local-small, strong, or none; thinking mode; and whether a
  cloud/strong model may be preferred.
- `permission_policy`: allowed capability layers and approval-required layers.
- `exit_conditions`: the events that should move the runtime out of that mode.

Self/world model state is now compiled through
`core.cognition.runtime_behavior_policy` before no-user goal pressure is
selected. Elevated world uncertainty can move the next tick into `DEEP_THINK`
with a strong-model policy; resource pressure can force local-small/no-cloud
budgets; persistent self-model failure can lift repair-oriented L1/L2 goals
above lower-value background work; and structured failure-learning objects can
block or penalize previously unsafe actions.

`CREATING` is deliberately bounded. It must exit when it produces at least one
candidate plan, patch proposal, hypothesis, or evidence reference; when it marks
the task as needing human review; or when its max creating tick budget is
exhausted. It is allowed to read and propose, but not to execute, fetch the
network, access credentials, or sync back source changes.
