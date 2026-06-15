# Prompts Used for Each AI System

## Core Prompt (Used for All Systems)

```
Design a comprehensive AGI architecture that could serve as a blueprint for building artificial general intelligence. Include specific details on:

1. Memory Architecture - How does the system store and retrieve information at different timescales?
2. Reasoning/Planning Loop - How does the system reason about problems and plan actions?
3. Learning/Self-Improvement Mechanism - How does the system learn and improve over time?
4. Tool Use and Action Execution - How does the system use tools and take actions?
5. World Model/Representation Layer - How does the system represent and reason about the world?
6. Safety/Governance Layer - How does the system ensure safe and aligned behavior?
7. Evaluation and Benchmark Strategy - How should the system's capabilities be evaluated?
8. Persistence/Runtime Architecture - How does the system maintain state over time?
9. Multi-Agent/Orchestration Design - How do multiple agents coordinate?
10. Engineering Feasibility Assessment - What are the key technical risks and timeline?

Provide specific, actionable architectural details. Avoid vague or purely philosophical discussion.
```

## Model-Specific Adaptations

### Claude 3.5 Sonnet
- No adaptation needed. Core prompt used directly.
- Response format: Detailed markdown with sections for each dimension.

### GPT-4o
- No adaptation needed. Core prompt used directly.
- Response format: Structured markdown with tables and code examples.

### Gemini 1.5 Pro
- Added: "Include specific implementation details and consider multi-modal capabilities."
- Response format: Markdown with tables and structured sections.

### DeepSeek-V3
- Added: "Focus on novel mechanisms and efficiency considerations."
- Response format: Markdown with technical specifications.

### Grok-2
- Added: "Consider real-time information processing and current events awareness."
- Response format: Markdown with emphasis on real-time capabilities.

### Llama 3.1 70B
- Added: "Consider open-source implementation and community governance."
- Response format: Markdown with transparency and auditability focus.

### Mistral Large 2
- Added: "Focus on efficiency and minimal resource requirements."
- Response format: Markdown with lean architecture emphasis.

### Qwen 2.5 72B
- Added: "Consider multi-modal integration (text, vision, audio) throughout."
- Response format: Markdown with multi-modal considerations.

### Command R+
- Added: "Emphasize retrieval-augmented approaches and source grounding."
- Response format: Markdown with retrieval-centric design.

### Perplexity Sonar
- Added: "Focus on real-time web grounding and temporal knowledge validity."
- Response format: Markdown with live information emphasis.

## Collection Methodology

1. Each model was prompted independently on 2026-06-15
2. The core prompt was kept as similar as possible across models
3. Minor adaptations were made only when necessary to elicit model-specific strengths
4. All outputs were collected in a single session to ensure consistency
5. Raw outputs were minimally cleaned (formatting only) and preserved in the `raw_outputs/` directory
