# Meta Llama — research reproduction

## Memory architecture
Local bounded cache, encrypted episodic database, and replaceable semantic index.

## Reasoning and planning loop
Orchestrator selects specialists, builds a plan, invokes tools, checks postconditions, and supports interruption/replay.

## Learning and self-improvement
Adapters, retrieval updates, and skill libraries from approved data; evaluate locally before release; no production weight rewrite.

## Tools and action
Capability-based plugins with schemas, sandboxing, resource limits, and consent for side effects.

## World model and safety
Task/state graph with learned embeddings; independent policy checks, logs, rollback, and shutdown.
