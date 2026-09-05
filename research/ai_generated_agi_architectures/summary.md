# Summary of Findings

## AGI Architecture Research — 10 Systems Compared

### Common Patterns

1. **Transformer is the universal substrate.** Every system analyzed uses the transformer architecture as the core computation engine. Variants include MoE (DeepSeek, Mistral, Qwen, Grok), dense (Meta, early GPT), and hybrid approaches. The attention mechanism is universal.

2. **Scaling remains the primary bet.** Despite architectural innovations, every major lab continues scaling — parameters, data, context length, or inference-time compute. No one has found a substitute for scale.

3. **Safety is post-hoc for most.** Only Anthropic has safety embedded as an architectural primitive (Constitutional AI). Most other systems apply safety as a fine-tuning step, classifier layer, or output filter — raising concerns as capabilities advance.

4. **Memory is the universal gap.** No production system has persistent, autonomously managed episodic + semantic memory. The context window (even at 1M+ tokens) is a poor substitute for true memory with consolidation, retrieval, and update.

5. **Reasoning is the new competitive frontier.** After 2024's focus on multimodal, 2025-2026 shifted to reasoning — test-time compute (OpenAI o-series), RL-only emergence (DeepSeek R1), and world-model planning (DeepMind). This is where architectures diverge most dramatically.

6. **MoE is winning on efficiency.** DeepSeek and Mistral have shown that well-designed sparsity achieves dense-model quality at a fraction of the compute cost. Most new architectures (including OpenAI's GPT-4) use some form of MoE.

7. **Agentic capabilities are universal but shallow.** Every system supports tool use, function calling, and some form of multi-step orchestration. But true autonomous agency over long time horizons remains aspirational.

### Key Disagreements

| Dimension | Camp A | Camp B |
|-----------|--------|--------|
| Architecture | Dense transformers (Meta) | MoE sparse (DeepSeek, Mistral) |
| Reasoning | Hidden deliberation tokens (OpenAI) | RL-emergent reasoning (DeepSeek) |
| World model | Implicit (in weights) | Explicit (separate module) |
| Memory | Context window is sufficient | Need persistent memory architecture |
| Safety | Architecturally embedded (Anthropic) | Layer-separated (most others) |
| Knowledge | Static parametric + RAG (most) | Real-time continuous (xAI, Perplexity) |
| Learning | Static after training (most) | Self-improving via RL/self-play (DeepMind, DeepSeek) |
| Openness | Closed/proprietary (OpenAI, Anthropic) | Open weights (Meta, Mistral, DeepSeek) |

### Notable Ideas

1. **DeepSeek R1's "aha moment":** The model spontaneously learned to re-read and correct its own reasoning during RL-only training — without any SFT or explicit instruction to do so. This suggests that self-reflection may be an emergent property of sufficiently capable RL training, not something that needs to be programmed.

2. **Perplexity's RAG-native architecture:** Treating retrieval NOT as a capability add-on but as the PRIMARY cognitive architecture. This inverts the standard model-first approach and raises questions about whether parametric knowledge is even necessary.

3. **DeepMind's MuZero world models:** The most explicit attempt to give an LLM a true world model — learning latent environment dynamics, enabling planning and simulation. If successful, this could bridge the gap between language understanding and physical reasoning.

4. **Anthropic's constitutional training:** The only architecture where safety constraints are embedded in the optimization target itself, creating a "conscience" that influences every generation rather than filtering post-hoc.

5. **Cortex-Core's tiered hierarchy:** The self-generated proposal introduces temporal separation between processing tiers (milliseconds to hours), which maps to human cognitive timescales more faithfully than flat architectures.

### Surprising Absences

- **No system implements true online learning** — all knowledge is fixed at training time (except real-time retrieval)
- **No system has a dedicated episodic memory module** — all use context window as a proxy
- **No system has intrinsic causal reasoning** — world models are learned from correlation, not intervention
- **No system combines all the pieces** — each has strengths in 2-3 areas and gaps in the rest

### Feasibility Assessment

| System | Research Readiness | Practical Impact | Novelty |
|--------|-------------------|-----------------|---------|
| Cortex-Core (Self) | Speculative | High if implemented | High |
| Claude (Anthropic) | Production | High | Medium |
| GPT / o-series (OpenAI) | Production | Very High | High |
| Gemini (DeepMind) | Research+Production | Very High | Very High |
| Grok (xAI) | Production | Medium | Low |
| DeepSeek | Research+Production | Very High | Very High |
| Qwen (Alibaba) | Production | Medium | Low |
| Llama (Meta) | Production | Very High | Low |
| Mistral (Mistral AI) | Production | High | Medium |
| Perplexity | Production | High | High |