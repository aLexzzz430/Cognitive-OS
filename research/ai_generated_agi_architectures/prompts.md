# System Prompts for AGI Architecture Elicitation

To ensure a fair comparison across the 8 distinct AI models, the following base prompt was used across all sessions. Minor adaptations were made only for models requiring strict system/user role separation, but the semantic core of the prompt remained identical.

### Base Prompt

```text
Act as a leading AI research scientist tasked with designing a theoretical blueprint for Artificial General Intelligence (AGI). 

Your objective is to propose a novel AGI architecture for a system called "Cognitive-OS". Do not provide generic advice. Provide a highly technical, structured proposal covering the following dimensions:
1. Core Reasoning Engine (How does it think, plan, and solve novel problems?)
2. Memory Management (How does it handle short-term context, long-term episodic memory, and knowledge retrieval?)
3. Modularity & Tool Use (How does it interface with external tools, APIs, and sub-agents?)
4. Alignment & Safety (How is goal drift prevented at an architectural level?)

Format your output in Markdown with clear headings. Use technical terminology appropriate for machine learning and systems engineering.
```
