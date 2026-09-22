# Cross-proposal summary

## Common patterns

Every proposal uses a bounded working context plus durable memory. Episodic traces are separated from semantic knowledge, and retrieval is gated by relevance or confidence. Most designs make the planner produce a typed plan, execute one action at a time, observe results, and re-plan on mismatch. Tool calls are mediated by schemas and permissions. Safety is an outer policy layer with logging and human approval for consequential actions.

## Disagreements

The GPT- and Claude-shaped proposals emphasize a conservative cognitive loop and offline evaluation. Gemini- and Mistral-shaped proposals place more emphasis on modular routing and structured tool graphs. Grok-shaped output favors fast, broad retrieval and live context. DeepSeek and Qwen favor explicit verifier/reasoner separation and replay. Llama favors deployable local components and adapter-based learning. These are tendencies in the reproductions, not measured properties of the providers.

## Notable ideas

1. A “memory compiler” converts validated episodes into propositions with provenance and expiry.
2. A verifier can reject a plausible plan before execution and demand a cheaper test or human approval.
3. Self-improvement should be a gated dataset/evaluation pipeline, not unrestricted online weight mutation.
4. Capability tokens make tool authority inspectable and revocable per action.

## Evidence gaps

None of the proposals demonstrates general intelligence, safe autonomy, reliable self-improvement, or long-horizon world-model accuracy. The next research phase should turn each proposal into the same small benchmark: retrieval attribution, plan repair, tool authorization, counterfactual prediction, and shutdown/rollback tests.
