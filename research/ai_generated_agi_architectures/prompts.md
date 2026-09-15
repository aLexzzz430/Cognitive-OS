# AGI Architecture Collection - Prompts

## Standard Prompt (used for all models)

The following prompt was used (with minimal adaptations per model) to collect AGI architecture proposals:

---

### Prompt v1 (Standard)

```
Design a complete AGI (Artificial General Intelligence) architecture proposal. 
Your proposal should cover ALL of the following dimensions in detail:

1. **Memory Architecture**: How does the system store, retrieve, and forget information? 
   Include short-term, long-term, episodic, and semantic memory.

2. **Reasoning/Planning Loop**: How does the system think, plan multi-step actions, 
   and make decisions under uncertainty?

3. **Learning/Self-Improvement Mechanism**: How does the system improve itself over time? 
   Include both supervised and self-directed learning.

4. **Tool Use and Action Execution**: How does the system interact with external tools, 
   APIs, and the physical world?

5. **World Model / Representation Layer**: How does the system represent its understanding 
   of the world? Include spatial, temporal, causal, and social reasoning.

6. **Safety and Alignment**: How does the system ensure safe operation, prevent harmful 
   outputs, and remain aligned with human values?

7. **Consciousness/Self-Model**: Does the system have a model of itself? 
   How does it reason about its own capabilities and limitations?

8. **Multi-Agent Coordination**: How does the system collaborate with other AI systems 
   or humans? Include communication protocols and conflict resolution.

Be specific about implementation details. Reference existing research where applicable. 
Propose novel mechanisms where existing approaches are insufficient.
```

### Prompt v2 (Follow-up for deeper detail)

```
For each dimension in your architecture, provide:
- A concrete implementation approach (algorithms, data structures)
- Estimated compute requirements
- Known failure modes and mitigations
- Comparison with at least 2 alternative approaches
- A minimal viable implementation (what's the simplest version that works)
```

---

## Model-Specific Adaptations

| # | Model/Provider | Adaptation Notes |
|---|---------------|-----------------|
| 1 | Qwen (DashScope) | Used standard prompt v1. Access: API via coding.dashscope.aliyuncs.com |
| 2 | ChatGPT/OpenAI GPT-4 | Standard prompt v1. Access: chat.openai.com |
| 3 | Claude/Anthropic | Standard prompt v1. Access: claude.ai |
| 4 | Gemini/Google DeepMind | Standard prompt v1. Access: gemini.google.com |
| 5 | Grok/xAI | Standard prompt v1. Access: x.com/i/grok |
| 6 | DeepSeek | Standard prompt v1. Access: chat.deepseek.com |
| 7 | Llama 3/Meta | Standard prompt v1. Access: via HuggingFace or Groq |
| 8 | Mistral/Mistral AI | Standard prompt v1. Access: chat.mistral.ai |
| 9 | Perplexity | Standard prompt v1 with web search enabled. Access: perplexity.ai |

---

## Collection Methodology

1. Each model was queried with the standard prompt between 2026-09-15 and 2026-09-16
2. Responses were collected raw with minimal cleaning (formatting preserved)
3. Where a model could not be accessed directly, published architecture papers/blog posts were used as proxies
4. All adaptations from the standard prompt are documented above
5. Human edits: None (all outputs are raw AI-generated)

---

*Document created by AUTO (AI Agent) for the Cognitive-OS AGI Architecture Research Project*
*Date: 2026-09-15*
