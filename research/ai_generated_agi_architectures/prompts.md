# Prompt Protocol

## Baseline prompt

All systems receive the same substantive request:

> Design a technically concrete AGI-capable cognitive architecture that could be implemented in stages. Do not give marketing language or generic principles. Specify mechanisms, interfaces, state transitions, and data structures where useful. Cover: memory architecture (working, episodic, semantic, procedural, identity/self-model); reasoning/planning loop; learning/self-improvement; tool use/action execution; world model/representation layer; safety/governance; evaluation/benchmarks; persistence/runtime; multi-agent/orchestration; engineering feasibility/staged roadmap; and 2–4 non-obvious insights. Distinguish durable state from model context. Treat LLM outputs as probabilistic proposals rather than ground truth. Show one end-to-end decision/action trace. Include a compact ASCII or Mermaid diagram. End with unresolved questions and falsifiable tests. Target 1,800–2,800 words.

The byte-for-byte working copy used for local command execution is `prompt_baseline.txt`.

## Adaptation policy

Provider-specific changes may only cover transport mechanics, such as passing the prompt through a CLI/API, selecting a model, or requesting plain-text output. We do not add architecture hints tailored to a particular model. We do not ask one model to imitate another. We do not synthesize missing raw outputs and label them as provider responses.

## Output handling

Responses are stored under `raw_outputs/`. Terminal control sequences may be removed as a mechanical cleanup; substantive wording is not rewritten. If a model emits explicit scratchpad-style planning before its answer, that material is excluded from comparative scoring and the final packet should retain only provider-visible answer text where feasible.