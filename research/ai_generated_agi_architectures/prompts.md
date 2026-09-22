# Collection prompts

## Canonical prompt

```text
You are proposing an architecture for a general-purpose cognitive operating system, not describing a current product. Design a technically coherent AGI architecture that can learn across tasks while remaining inspectable and controllable. Cover: (1) memory architecture, (2) reasoning and planning loop, (3) learning/self-improvement, (4) tool use and action execution, (5) world model or representation layer, and (6) safety. State data flows, update boundaries, failure modes, and measurable evaluation criteria. Distinguish what is implemented today from what is speculative. Be concrete enough that an engineering team could turn the proposal into interfaces and experiments. Keep the answer under 1,200 words.
```

## Per-family application

The canonical prompt was applied unchanged to: OpenAI GPT, Anthropic Claude, Google Gemini, xAI Grok, DeepSeek, Qwen, Meta Llama, and Mistral. A short format suffix was appended for local reproducibility: `Return six numbered sections matching the requested dimensions, followed by interfaces and evaluation criteria.` No provider system prompts, chain-of-thought, or private context were requested or retained.

## Reproduction note

The included outputs are local research reproductions, not authenticated provider transcripts. They are deliberately marked as such in `sources.md`; replacing them with captured outputs should not require changing the CSV schema.
