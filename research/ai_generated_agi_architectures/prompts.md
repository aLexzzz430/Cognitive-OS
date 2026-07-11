# Prompts Used for AGI Architecture Collection

## Standardized Prompt

Used for all models (adapted only when API format required):

```
You are an AGI architecture expert. Propose a detailed, implementable architecture for an Artificial General Intelligence system. Your proposal must cover ALL of the following dimensions with concrete technical details:

1. Memory Architecture: How does the system store, retrieve, and forget information? What types of memory (working, episodic, semantic, procedural) are used and how are they implemented?

2. Reasoning/Planning Loop: What is the core cognitive loop? How does the system deliberate, plan multi-step actions, and adapt when plans fail?

3. Learning/Self-Improvement: How does the system learn from experience? What mechanisms enable continual learning without catastrophic forgetting? Can it modify its own architecture?

4. Tool Use & Action Execution: How does the system interact with external tools, APIs, and environments? How does it decide when to use tools vs. reason internally?

5. World Model: How does the system represent and predict the state of the world? What is its internal model of causality, time, and physical/social dynamics?

6. Safety/Governance: What mechanisms prevent harmful behavior? How are goals aligned with human values? What oversight and override mechanisms exist?

7. Evaluation & Benchmarks: How would you measure whether this system has achieved AGI? What specific benchmarks and evaluation criteria would you use?

8. Persistence/Runtime: What is the deployment architecture? How is state persisted across sessions? What is the runtime environment?

9. Multi-Agent/Orchestration: Does the system use multiple specialized agents? If so, how are they coordinated? What is the orchestration topology?

10. Engineering Feasibility: What are the concrete engineering challenges? What existing technologies would you build on? Estimate compute requirements.

11. Originality: What is the most non-obvious or novel idea in your proposal that distinguishes it from standard LLM-based approaches?

Be specific and technical. Avoid vague philosophical statements. Provide concrete data structures, algorithms, and architectural patterns where possible. Limit your response to 3000 words.
```

Collection Date: 2026-07-11
