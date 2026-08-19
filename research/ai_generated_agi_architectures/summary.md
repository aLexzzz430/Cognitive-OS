# Summary of Findings

## Common Patterns
- **Memory**: Almost all models propose a dual or tripartite memory system (short-term context window, episodic vector DB, and long-term structural/symbolic graph).
- **Reasoning**: "Tree of Thoughts" and recursive self-reflection loops are universally favored over linear generation for AGI architectures.
- **Safety**: Multi-layered approaches combining Constitutional AI (prompt/context level) with hardcoded sandboxes for action execution.

## Disagreements
- **World Models**: Some systems (Gemini, Grok) advocate for explicit, real-time updated knowledge graphs, while others (GPT-5.6) rely on implicit neural representations supplemented by search.
- **Learning**: DeepSeek heavily emphasizes self-play and reinforcement learning, while Perplexity and Kimi focus on in-context learning over massive context windows.

## Notable Ideas
- **Grok-4.5**: Suggests an adversarial truth-seeking dual-agent system where one agent proposes actions and another specifically searches for reasons it will fail.
- **DeepSeek-v4**: Recommends symbolic+neural execution environments where all planning is translated to verifiable code before execution.
