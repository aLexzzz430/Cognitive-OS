# Summary of AGI Architecture Proposals

## Common Patterns
- **Modular design**: All proposals decompose AGI into memory, reasoning, learning, tool use, world model, and safety modules.
- **Hierarchical memory**: Working, episodic, and semantic layers appear in every proposal.
- **Feedback loops**: Reasoning includes reflection, verification, or self-critique.
- **Continual learning**: Online updates via human feedback, self-supervised objectives, or RL.
- **Tool integration**: APIs and external executors are considered essential.
- **Safety as an add-on**: Most place safety as a separate module, often after core functionality.

## Disagreements
- **Nature of world model**: Symbolic vs. learned neural vs. hybrid. Grok-1 favors logic, while Gemini and DeepSeek favor learned simulators.
- **Planning method**: MCTS (GPT-4), hierarchical (Claude), SAT (Grok), beam search (Mistral) – wide variation.
- **Learning paradigm**: Meta-learning (Gemini), competitive self-play (Grok), continual pretraining (DeepSeek) – focus differs.
- **Memory representation**: Vector DB (ChatGPT, Llama), relational (Claude), key-value (Grok), SQL (Qwen).

## Notable Ideas
- **Sleep consolidation**: Claude proposes offline memory reorganization inspired by biological sleep.
- **Honeypot safety**: Grok suggests fake vulnerabilities to detect misaligned behavior.
- **Formal verification**: Mistral includes formal checks for safety constraints.
- **Abstract tool contracts**: Grok and Claude define tools with preconditions/postconditions.
- **Hybrid neuro-symbolic**: Grok and Qwen advocate combining neural networks with symbolic reasoning.

---
*These patterns inform the combined architecture in synthesis.md.*