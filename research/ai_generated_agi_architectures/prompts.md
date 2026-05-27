# Prompts Used for AGI Architecture Collection

## Base Prompt (used for all systems)

```
You are an AGI architecture researcher. Design a complete AGI (Artificial General Intelligence) system architecture. Your proposal should cover:

1. **Memory Architecture**: How does the system store, retrieve, and update knowledge? Include working memory, long-term memory, and any specialized memory types.

2. **Reasoning/Planning Loop**: How does the system reason about problems, plan actions, and handle uncertainty? Include the core inference mechanism.

3. **Learning/Self-Improvement**: How does the system learn from experience and improve itself over time? Include meta-learning capabilities.

4. **Tool Use & Action Execution**: How does the system interact with the external world, use tools, and execute actions?

5. **World Model**: How does the system represent and reason about the external world? Include how it builds and updates its world model.

6. **Safety**: How does the system ensure safe behavior? Include alignment mechanisms, value learning, and shutdown procedures.

For each component, describe:
- The core mechanism
- Key algorithms or approaches
- How it interfaces with other components
- Known limitations and proposed mitigations

Be specific and technical. Propose concrete architectures, not vague aspirations.
```

## Adaptations Per Model

### GPT-4o
- No adaptation needed. Base prompt used directly.

### Claude Opus 4
- Added: "Please be thorough in your safety section given your constitutional training."
- Reasoning: Claude has explicit safety training; prompting for thoroughness leverages this.

### Gemini 2.5 Pro
- Added: "Consider multimodal perception as a core architectural component."
- Reasoning: Gemini is natively multimodal; this elicits its strengths.

### Grok-3
- Added: "Include how the system handles real-time information streams."
- Reasoning: Grok has real-time X/Twitter access; this surfaces unique insights.

### DeepSeek-R1
- Added: "Show your reasoning process step by step before the final architecture."
- Reasoning: R1 is trained for chain-of-thought; this elicits deeper reasoning.

### Qwen-3 235B
- Added: "Consider how this architecture would handle multilingual reasoning natively."
- Reasoning: Qwen excels at multilingual tasks.

### Llama 4 Maverick
- No adaptation needed. Base prompt used directly.

### MiMo v2.5 Pro
- Added: "Consider temporal reasoning as a first-class architectural primitive."
- Reasoning: MiMo's training emphasizes structured reasoning over time.
