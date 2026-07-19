# Summary: Common Patterns, Disagreements, and Notable Ideas

## Common Patterns (Shared by 7+ Systems)

### 1. Hierarchical Memory (10/10)
Every system proposes some form of multi-tier memory. The consensus is that a single memory store is insufficient. The typical pattern is: working/short-term + episodic + semantic/long-term. The disagreement is in implementation��vector stores, knowledge graphs, MoE shards, or search caches.

### 2. Multi-Agent Orchestration (10/10)
All systems include multi-agent designs, though the architectures differ significantly. The consensus is that a single monolithic agent is insufficient for general intelligence. Approaches range from hierarchical (GPT-5) to debate-based (Claude) to marketplace (Qwen) to federated (Llama 4).

### 3. Tool Use as First-Class (10/10)
Every system treats tool use as a core capability, not an add-on. The trend is toward learned tool selection and composition rather than manual tool specification.

### 4. Safety as a Distinct Concern (10/10)
All systems address safety explicitly. The split is between "separate safety layer" (6/10) and "integrated safety reasoning" (4/10).

### 5. Self-Improvement Loops (9/10)
All but Perplexity propose some form of self-improvement. The common thread is using task outcomes as training signal, but methods range from LoRA fine-tuning to self-play to skill mining.

## Key Disagreements

### Memory: Learned vs. Structured
- **Learned camp** (DeepSeek, Mistral): Memory is primarily vector embeddings with learned retrieval
- **Structured camp** (Claude, Qwen, GLM): Memory requires explicit structure (knowledge graphs, skill libraries, cultural models)
- **Hybrid camp** (GPT-5, Gemini, Grok, Llama 4, Perplexity): Both are needed, with different roles

### Planning: Linear vs. Tree Search
- **Linear/backtracking** (GPT-5, Grok): Sequential reasoning with backtracking on failure
- **Tree search** (Gemini, DeepSeek): Explicit tree exploration with value functions
- **Skill-based** (Qwen, Mistral): Planning as skill sequence composition
- **Configurable** (Llama 4): User-defined reasoning DAG

### World Models: Essential vs. Emergent vs. Unnecessary
- **Essential** (Gemini, DeepSeek, Perplexity): Explicit world models are critical for prediction and planning
- **Emergent** (GPT-5, Claude, Mistral): World models emerge from memory and reasoning
- **Task-specific** (Qwen, GLM): No single world model; use task-specific or cultural-perspective models
- **Community-provided** (Llama 4): World models should be contributed by domain experts
- **Streaming** (Grok): A full world model is unnecessary; situation awareness suffices

### Safety: Universal vs. Contextual
- **Universal rules** (GPT-5, Gemini, DeepSeek, Mistral, Perplexity): Safety rules apply globally
- **Contextual/pluralistic** (Claude, Grok, Qwen, GLM): Safety depends on context, culture, and stakes
- **Community-governed** (Llama 4): Safety rules should be community-defined and auditable

## Notable Unique Ideas

1. **Cultural memory** (GLM-5): The only system that explicitly models cultural variation as a core memory dimension. This addresses a real gap��AGI must work across cultures.

2. **Skill mining and compilation** (Qwen 3): Converting ad-hoc solutions into reusable, parameterized skills is a powerful self-improvement mechanism that goes beyond simple fine-tuning.

3. **Search-grounded memory** (Perplexity): Building memory entirely on live search results ensures information is always current and verifiable, but sacrifices inference speed.

4. **Constitutional reasoning** (Claude): Integrating safety into the reasoning loop itself, rather than as a post-hoc filter, produces more robust safety guarantees.

5. **Open modular architecture** (Llama 4): Designing for community contribution at every layer could accelerate development far beyond what any single team can achieve.

6. **Sliding window memory injection** (Mistral): A practical compromise between fixed context windows and unbounded memory, with clear engineering advantages.

7. **Self-play triad** (DeepSeek): The solver-verifier-generator triad is a self-contained improvement loop that doesn't require external training data.

8. **Streaming real-time architecture** (Grok): The only system designed for real-time data streams, addressing a use case others overlook.

## Surprising Gaps

1. **No system proposes a concrete solution to the symbol grounding problem.** All assume language or multimodal embeddings are sufficient.
2. **Energy efficiency is mentioned by none.** AGI at scale will have enormous energy costs, yet no proposal addresses this.
3. **Continual learning stability** is hand-waved by all. The catastrophic forgetting problem is acknowledged but not solved.
4. **Interpretability** is addressed only by Claude (constitutional reasoning) and Llama 4 (open audit). Others treat the model as a black box.
