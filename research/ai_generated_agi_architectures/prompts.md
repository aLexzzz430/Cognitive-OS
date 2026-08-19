# Collection prompts

## System prompt

```text
You are an independent architecture research respondent. Design a proposed system; do not claim access to proprietary model internals or hidden reasoning. Return Markdown only.
```

## Comparable user prompt

```text
Design a concrete AGI architecture proposal that could inform implementation of a system called Cognitive-OS. This is architecture research, not a request to describe your own provider's private internals. Make the proposal technically specific and falsifiable.

Cover these comparison dimensions using explicit headings:
1. memory architecture
2. reasoning/planning loop
3. learning or self-improvement mechanism
4. tool use and action execution
5. world model or representation layer
6. safety/governance layer
7. evaluation and benchmark strategy
8. persistence/runtime architecture
9. multi-agent or orchestration design
10. engineering feasibility
11. originality or non-obvious insight

Include: named components and responsibilities; state/data flows; interfaces or pseudocode where useful; failure modes and mitigations; an incremental implementation path; measurable evaluation gates; and at least one genuinely non-obvious design idea. Distinguish speculative choices from established engineering patterns. Avoid marketing language and generic AGI prose.
```

All eight calls used this same prompt text. No model-specific adaptation was used.

Prompt SHA-256: `656ffc1412932c9f588ee50101a74767235ff1e8f97d13c2c4c0d930a4c92216`
