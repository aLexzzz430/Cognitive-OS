# Research Prompts and Methodology

## Primary Prompt

The following base prompt was used to collect AGI architecture proposals:

```
You are an AI researcher designing an AGI (Artificial General Intelligence) architecture.

Please describe your proposed AGI architecture in detail, covering:

1. **Core Architecture**: What is the fundamental computational structure?
2. **Scaling Strategy**: How does the system scale to greater capabilities?
3. **Memory System**: How does the system store and retrieve knowledge?
4. **Training Paradigm**: How is the system trained/learned?
5. **Reasoning Approach**: How does the system perform complex reasoning?
6. **World Model**: How does the system understand and predict the world?
7. **Tool Use**: How does the system interact with external tools?
8. **Multi-modal Support**: How does the system handle different data types?
9. **Safety Mechanisms**: How is the system aligned with human values?
10. **Deployment Strategy**: How would the system be deployed?
11. **Open-source Status**: What components would be open-sourced?
12. **AGI Timeline**: When do you estimate this architecture could achieve AGI?

Please be specific and technical. Avoid vague generalities.
```

## System-Specific Adaptations

### For OpenAI (GPT-4/o3)
- Added: "Focus on the reasoning token approach seen in o1/o3 models"
- Rationale: OpenAI has publicly discussed their reasoning approach

### For Anthropic (Claude)
- Added: "Emphasize the Constitutional AI approach to safety"
- Rationale: Anthropic's primary innovation is Constitutional AI

### For Google DeepMind (Gemini)
- Added: "Include the role of AlphaProof-style mathematical reasoning"
- Rationale: DeepMind has unique expertise in formal reasoning

### For DeepSeek
- Added: "Explain the Mixture-of-Experts (MoE) efficiency advantage"
- Rationale: DeepSeek's primary innovation is MoE architecture

### For Meta (Llama)
- Added: "Discuss the open-source ecosystem and community contribution"
- Rationale: Meta's primary strategy is open-source

### For Academic Frameworks (SOAR, AIXI)
- Added: "Compare symbolic vs neural approaches"
- Rationale: Academic frameworks are often hybrid or symbolic

## Data Collection Process

1. **Research Phase**: Collected published papers, technical reports, and design documents
2. **Synthesis Phase**: Extracted architecture proposals from available materials
3. **Comparison Phase**: Standardized proposals across 12 dimensions
4. **Analysis Phase**: Identified patterns and divergences

## Limitations

1. **Access**: Not all systems were queried directly; some proposals are inferred from published materials
2. **Recency**: Information current as of May 2026
3. **Completeness**: Some systems have limited public documentation
4. **Bias**: Researcher's perspective may influence interpretation

## Verification

All raw outputs include source references where available. The comparison CSV provides structured data for independent verification.
