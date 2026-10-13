# AGI Architecture Proposal — Anthropic Claude family perspective

> **Attribution note:** Representative reconstruction drafted by the research author
> (AI-assisted) in the documented architectural philosophy of the Anthropic/Claude
> family, grounded in public materials (Constitutional AI papers, system cards, MCP
> specification, engineering blog posts on agents). Not a verbatim model output.
> See `sources.md`.

## Thesis
AGI is an **agentic loop done with extreme care**: gather context, take actions,
verify results, repeat — with safety properties trained into the model itself
rather than bolted on outside. The differentiators are a universal tool bus
(MCP), disciplined context engineering, and constitutional training that makes
the model a reliable collaborator over long horizons.

## 1. Memory architecture
Memory is **files and context, not databases**. The agent maintains working
notes (a scratchpad of plans, findings, open questions) inside its context, and
persists durable knowledge as plain markdown files the agent itself reads and
writes. This is deliberate: file-based memory is inspectable, versionable, and
works with any model. Retrieval augmentation exists but is secondary — the
primary skill is *context engineering*: deciding what deserves to be in context
right now.

## 2. Reasoning / planning loop
**Extended thinking** blocks give the model a private scratchpad for hard
problems, separable from the user-visible response. The outer loop is the
agentic loop: think → act (tool) → observe → verify. Planning artifacts
(explicit plans, hypotheses with confidence) are written into context so they
persist across steps. Verification is a first-class step, not an afterthought:
after acting, the agent checks the result against the goal before proceeding.

## 3. Learning / self-improvement
Training emphasizes **honesty and calibration** (the model should know what it
doesn't know) via constitutional training and RL from human feedback focused on
long-horizon agentic tasks. In deployment, the model improves within a session
by updating its notes and plans — a form of in-context learning. Cross-session
improvement comes from humans curating the persisted files, keeping a human in
the loop of the learning process.

## 4. Tool use and action execution
The **Model Context Protocol (MCP)** standardizes tool access: any tool server
plugs into any compliant agent. Tools are described with schemas; the model
learns robust tool use from RL on agentic coding and computer-use tasks.
Execution policy: read-only tools are free, state-changing tools go through
explicit approval gates, and destructive tools are namespaced per session.

## 5. World model / representation layer
The world model is the language model plus **grounding through action**: the
agent builds its picture of the environment by probing it (listing files,
running commands, reading outputs) rather than from a prebuilt simulator.
For software domains this is sufficient; for physical domains, tool-mediated
sensors feed the same loop.

## 6. Safety / governance layer
**Constitutional AI**: the model is trained against an explicit written
constitution (harmlessness, honesty, fidelity), so safety judgments happen
inside the model's reasoning, not just in external filters. Deployment adds
**AI Safety Levels (ASL)** — staged safeguards that ratchet up with
capability — plus usage policies, monitoring, and the instruction hierarchy.
The governance philosophy: the strongest safety layer is the model's own
trained judgment.

## 7. Evaluation and benchmark strategy
Holistic **agentic evals** over full task completions (not just next-token
accuracy): can the agent actually fix the bug, complete the research, use the
computer? Safety evals probe for deception, power-seeking, and jailbreak
robustness. Human feedback on real deployments is the ultimate metric.

## 8. Persistence / runtime architecture
Sessions are long-lived and resumable; the runtime checkpoints context,
working files, and tool state. **Subagents** with isolated contexts handle
subtasks in parallel, reporting back summaries — this is both a capability
pattern and a safety pattern (blast radius containment). Cost is managed by
routing: cheap models for retrieval-classification, frontier models for
judgment.

## 9. Multi-agent / orchestration design
A lead agent spawns **specialized subagents** (researcher, coder, reviewer),
each with a narrow context window and toolset, coordinated through the lead's
context. Critic agents review high-stakes outputs. Orchestration is
hierarchical and explicit, avoiding emergent multi-agent chaos.

## 10. Engineering feasibility
High: the entire pattern runs in production (agentic coding tools). MCP makes
tool integration a solved problem. The research risk is concentrated in
training — constitutional methods at larger scales — not in architecture.

## 11. Originality / non-obvious insight
The non-obvious claim: **context engineering beats architecture**. Most
"agent frameworks" fail because they over-structure the loop; a strong model
with good tools, good context, and permission to think does better than any
hand-built planner. And: safety should be *trained in*, because external
guardrails always lag the model's capabilities.
