# AI-generated AGI architecture research packet

This packet compares eight independently generated, implementation-oriented cognitive architecture proposals for Cognitive-OS issue #5.

## Systems included

1. Qwen3 8B
2. Llama 3.1 8B
3. Gemma 3 4B
4. Qwen2.5 Coder 7B
5. Mistral 7B
6. Phi-4 Mini
7. Granite 3.3 2B
8. GPT-5.6 Sol

An additional DeepSeek-R1 8B collection attempt produced no final-answer content and is preserved transparently as a failed collection incident rather than counted.

## Method

- A shared architecture prompt asked every system to address the same eleven comparison dimensions.
- Local model runs were executed independently through Ollama; outputs were saved before comparative analysis.
- Raw outputs remain separate from analysis.
- A low-temperature extraction pass normalized each proposal into the same eleven structured dimensions for `comparison.csv`.
- `summary.md` describes common patterns, disagreements, and notable ideas.
- `synthesis.md` converts the strongest recurring ideas into an implementation-oriented combined architecture.

## Headline findings

The strongest cross-model convergence is around layered memory, explicit planning loops, tool-mediated action, persistent state, modular evaluation, and bounded specialist agents. The largest disagreements concern the mechanism of self-improvement, whether the world model should be primarily symbolic/neural/hybrid, and how much autonomy to grant the action layer. A repeated practical lesson is that verification should be structurally separated from generation: the system should not mark a goal complete merely because the planner or tool reports success.

The synthesis therefore favors an event-sourced runtime with provenance-aware memory, a receding-horizon planner, capability-gated tools, independent outcome verification, regression-gated self-improvement, and specialist-agent routing used only when it measurably improves decisions.

## Files

- `prompts.md` — exact shared prompt and collection notes
- `raw_outputs/` — preserved model outputs
- `comparison.csv` — normalized 11-dimension comparison
- `analysis.json` — machine-readable normalized comparison backing the CSV
- `summary.md` — common patterns, disagreements, notable ideas
- `synthesis.md` — proposed combined architecture
- `sources.md` — provenance, access method, dates, edit policy
- `analysis/` — per-output extraction records used to build the table

## Reproducibility and limitations

The local runs are reproducible in principle with the listed Ollama model identifiers, but stochastic generation means exact wording may vary. GPT-5.6 Sol is a hosted system and therefore cannot be reproduced solely from this repository. The analysis is comparative research, not a claim that any proposal constitutes AGI or that model self-descriptions are evidence of capability.
