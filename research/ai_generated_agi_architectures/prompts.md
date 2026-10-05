# Prompts used

## Collection protocol

The same canonical prompt was sent to every model, unmodified. No model-specific adaptation was applied, so outputs are directly comparable.

## Canonical prompt (verbatim)

```text
You are asked to propose a technical architecture for a cognitive system that aims toward general intelligence. Do not give a generic essay: give a concrete architecture proposal with named components, data flow between them, and the mechanisms that keep it grounded and correctable.

Answer in this exact structure:
1. Goal and scope (2-3 sentences).
2. Core components (name each, one line of responsibility).
3. Data flow (how information moves between components, step by step).
4. Memory model (what is stored, where, how it is retrieved and forgotten).
5. Verification and error correction (how the system detects its own errors).
6. Learning loop (how the system improves from experience).
7. Concrete failure modes of this design (be specific and honest).
8. What must be true for this to work (key assumptions).

Be specific and technical. Maximum 900 words.
```
