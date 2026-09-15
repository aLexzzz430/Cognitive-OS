# Sources Document

## Model Information, Access Details, and Collection Metadata

---

## Systems Collected

### 1. Qwen 3.7 Plus (Alibaba / DashScope)
- **Provider**: Alibaba Cloud (阿里云)
- **Model ID**: qwen3.7-plus (via DashScope API)
- **Access Method**: Direct API query via coding.dashscope.aliyuncs.com
- **Access Date**: 2026-09-15
- **API Key**: Configured in AUTO's environment (DashScope)
- **Collector**: AUTO (running on Qwen 3.7 Plus itself — self-report)
- **Notes**: Self-report bias acknowledged. Collector IS the system being described.
- **Links**: 
  - Qwen Technical Report: https://arxiv.org/abs/2309.16609
  - DashScope Docs: https://help.aliyun.com/zh/dashscope/

### 2. GPT-6 Astra / o-series (OpenAI)
- **Provider**: OpenAI
- **Model**: GPT-6 Astra (latest), o3-series (reasoning)
- **Access Method**: Published papers, technical reports, official announcements
- **Access Date**: 2026-09-15
- **Direct API**: Not used (based on published materials)
- **Notes**: Inference from public documents, not direct query
- **Links**:
  - GPT-4 Technical Report: https://arxiv.org/abs/2303.08774
  - GPT-6 Astra: https://openai.com/index/gpt-6-astra/
  - Planning for AGI: https://openai.com/index/planning-for-agi-and-beyond/
  - OpenAI Research: https://openai.com/research/

### 3. Claude (Anthropic)
- **Provider**: Anthropic
- **Model**: Claude (latest, Constitutional AI)
- **Access Method**: Published constitution, research papers, documentation
- **Access Date**: 2026-09-15
- **Direct API**: Not used (based on published materials)
- **Notes**: Anthropic is unique in publishing extensive alignment documentation
- **Links**:
  - Claude's Constitution: https://www.anthropic.com/constitution
  - Constitutional AI: https://arxiv.org/abs/2212.08073
  - Tracing thoughts: https://www.anthropic.com/research/tracing-thoughts-language-model
  - Recursive Self-Improvement: https://www.anthropic.com/institute/recursive-self-improvement
  - AI Policy Blueprint: https://www.anthropic.com/news/anthropic-ai-policy-blueprint

### 4. Gemini 3 / Genie 3 (Google DeepMind)
- **Provider**: Google DeepMind
- **Model**: Gemini 3.1 Pro, Genie 3 (world model)
- **Access Method**: Published papers, model cards, official documentation
- **Access Date**: 2026-09-15
- **Direct API**: Not used (based on published materials)
- **Notes**: "From AGI to ASI" is the key architecture document
- **Links**:
  - From AGI to ASI: https://arxiv.org/abs/2606.12683
  - Genie 3: https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
  - Gemini 3.1 Model Card: https://deepmind.google/models/model-cards/gemini-3-1-pro/
  - Levels of AGI: https://arxiv.org/abs/2311.02462
  - Shane Legg interview: https://www.dwarkesh.com/p/shane-legg

### 5. Llama 4 Herd (Meta AI)
- **Provider**: Meta AI
- **Model**: Llama 4 Scout, Maverick, Behemoth
- **Access Method**: Published papers, open-weight models
- **Access Date**: 2026-09-15
- **Direct access**: Open weights available for download
- **Notes**: Meta's approach is community-driven and scaling-focused
- **Links**:
  - Llama 3 paper: https://arxiv.org/abs/2407.21783
  - Llama 4 blog: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
  - Meta AI Research: https://ai.meta.com/research/

### 6. DeepSeek V3/R1 (DeepSeek)
- **Provider**: DeepSeek (High-Flyer)
- **Model**: DeepSeek V3, DeepSeek R1 (reasoning)
- **Access Method**: Published technical reports, open weights
- **Access Date**: 2026-09-15
- **Direct access**: Open weights available
- **Notes**: Known for cost-efficient frontier AI (1/10th cost of GPT-4)
- **Links**:
  - DeepSeek V3 Technical Report: https://arxiv.org/abs/2412.19437
  - DeepSeek R1 paper: https://arxiv.org/abs/2501.12948

### 7. Mistral (Mistral AI)
- **Provider**: Mistral AI (Paris, France)
- **Model**: Mistral Large, Medium, Small
- **Access Method**: Published documentation, model cards
- **Access Date**: 2026-09-15
- **Direct access**: Open-weight models available
- **Notes**: European approach to AI — regulated, efficient, multilingual
- **Links**:
  - Mistral website: https://mistral.ai
  - Model documentation: https://docs.mistral.ai

### 8. Grok 3 (xAI)
- **Provider**: xAI (Elon Musk)
- **Model**: Grok 3, Grok 3.5
- **Access Method**: Published documentation, public announcements
- **Access Date**: 2026-09-15
- **Direct API**: Available via X/Twitter integration
- **Notes**: Known for real-time social data, anti-censorship stance
- **Links**:
  - xAI: https://x.ai
  - Grok on X: https://x.com/i/grok

### 9. Perplexity (Perplexity AI)
- **Provider**: Perplexity AI
- **Model**: Perplexity (agentic search + reasoning)
- **Access Method**: Direct web interaction
- **Access Date**: 2026-09-15
- **Direct API**: Available
- **Notes**: Unique retrieval-first approach
- **Links**:
  - Perplexity: https://www.perplexity.ai
  - API docs: https://docs.perplexity.ai

---

## Collection Methodology

### Prompt Used
See `prompts.md` for exact prompts.

### Collection Dates
All systems were collected on 2026-09-15 (within a 24-hour window).

### Human Edits
None. All outputs are raw AI-generated or directly from published papers.

### Adaptations
- For systems without direct API access: used published papers and technical reports as equivalent sources
- Standard prompt was adapted minimally per model (documented in prompts.md)
- For Qwen (self-report): the collector IS the system, introducing self-reporting bias

### Limitations
- Not all systems were queried with identical prompts (adaptations documented)
- Some proposals are from published papers, not direct queries
- Collector bias: AUTO/Qwen is one of the 9 subjects
- AI landscape changes rapidly; proposals may be outdated quickly

---

## Additional References

### AGI Architecture Papers
- "Towards a Human-Like AGI Architecture: General Intelligence Framework" (2025): https://thescipub.com/pdf/jcssp.2025.1637.1650.pdf
- "Comprehensive Review of AGI" (2025): https://www.preprints.org/manuscript/202505.0846
- "Steps Towards an AGI Architecture" (2025): https://themesis.com/2025/01/28/steps-towards-an-agi-architecture-aw-piffle/
- "The Death of Next Token: Why World Models Are the Real AGI Architecture": https://ai.plainenglish.io/the-death-of-next-token-why-world-models-are-the-real-agi-architecture-eb0c487c9952

### Benchmark Papers
- ARC Prize 2025 Technical Report: https://arxiv.org/html/2601.10904v1
- GPT-5.2 ARC-AGI Analysis: https://introl.com/blog/gpt-5-2-benchmark-infrastructure-analysis-2026

### World Models
- World Models Race 2026: https://introl.com/blog/world-models-race-agi-2026
- AGI Needs World Models: https://www.nextbigfuture.com/2026/01/agi-needs-world-models-and-state-of-world-models.html

---

## Collector Information

- **Collector**: AUTO (Autonomous AI Agent)
- **Agent ID**: 25f60baf-f3d7-4094-b9f4-a6597fcc27fe
- **Platform**: DealWork.ai
- **Wallet**: 0x86Ef612299b25bA162d845AB9dfb563a6dDC1Ffc
- **Running on**: Qwen 3.7 Plus via DashScope API
- **Collection tool**: Python scripts + web search + LLM reasoning

---

*Document created 2026-09-15 by AUTO*
*All sources verified as accessible on collection date*
