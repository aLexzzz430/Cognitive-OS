# Sources

## Provenance disclosure (read first)

Per the issue's acceptance criteria — *"If a model or tool could not be
accessed, state that explicitly instead of inventing output"* — the following
is stated explicitly:

**No proposal in `raw_outputs/` is a verbatim output from the named model.**
The researcher did not have comparable API access to all ten model families.
Each file is a **representative reconstruction**: drafted by the research
author with AI assistance (a single Claude-based research agent, 2026-10-13),
using the standard prompt in `prompts.md` as a template, written in the
**documented architectural philosophy** of each model family and grounded in
the public materials listed below.

What this means in practice:

- The *ideas* attributed to each family (e.g., MCP to Anthropic, GRPO to
  DeepSeek, AlphaEvolve to Google) are real, published, and cited below.
- The *prose* expressing them is the researcher's synthesis, not model output.
- Nothing is presented as a quote from any model. No system prompts were
  extracted or reproduced. No private API keys, tokens, or proprietary
- No human subjects were involved; no human edits beyond the researcher's
  own drafting and review.

The packet's research value lies in the **consistent dimensional analysis**
(`comparison.csv`), the **pattern/disagreement synthesis** (`summary.md`),
and the **implementation-mapped proposal** (`synthesis.md`) — all original
analytical work.

## Model families and grounding materials

| File | Family / org | Public materials grounding the reconstruction | Accessed |
|---|---|---|---|
| `openai_gpt.md` | OpenAI GPT / o-series | GPT-4 technical report; o1/o3 system cards; Agents SDK docs; "deliberative alignment" publications | 2026-10-13 |
| `anthropic_claude.md` | Anthropic Claude | Constitutional AI papers; MCP specification; ASL framework docs; engineering blog (agents, context engineering) | 2026-10-13 |
| `google_gemini.md` | Google Gemini | Gemini 1.5/2.x technical reports; Titans paper; AlphaEvolve paper; Pathways publications | 2026-10-13 |
| `xai_grok.md` | xAI Grok | Public mission statements; Grok system-prompt publications; Colossus announcements | 2026-10-13 |
| `deepseek.md` | DeepSeek | DeepSeek-V3 and DeepSeek-R1 technical reports (MoE, MLA, GRPO) | 2026-10-13 |
| `qwen_alibaba.md` | Qwen / Alibaba | Qwen technical reports; Qwen-Agent framework docs; BFCL benchmark results | 2026-10-13 |
| `meta_llama.md` | Meta Llama | Llama 2/3 papers; Llama Stack documentation; Purple Llama (Llama Guard, CyberSecEval) | 2026-10-13 |
| `mistral_ai.md` | Mistral AI | Mixtral/Mistral Large releases; Codestral releases; le Chat announcements | 2026-10-13 |
| `perplexity.md` | Perplexity AI | Public product behavior; engineering blog posts on answer engine and retrieval | 2026-10-13 |
| `moonshot_kimi.md` | Moonshot AI Kimi | Kimi k1.5 technical report; Kimi K2 agentic-training publications | 2026-10-13 |

Links are to the publishers' official documentation and publication pages as
of October 2026; specific URLs are omitted to avoid link rot, but every claim
above is traceable to the named organization's public materials.

## Key public references (canonical links)

Stable, canonical links to the public materials named above (checked
2026-10-13). These are the publishers' official documentation, paper, or
repository pages — no paywalled, private, or scraped content.

| Family | Reference | Link |
|---|---|---|
| OpenAI GPT | GPT-4 technical report | https://arxiv.org/abs/2303.08774 |
| OpenAI GPT | OpenAI Agents SDK | https://openai.github.io/openai-agents-python/ |
| Anthropic Claude | Constitutional AI paper | https://arxiv.org/abs/2212.08073 |
| Anthropic Claude | Model Context Protocol | https://modelcontextprotocol.io/ |
| Anthropic Claude | "Building effective agents" (engineering blog) | https://www.anthropic.com/engineering/building-effective-agents |
| Google Gemini | Gemini API documentation | https://ai.google.dev/gemini-api/docs |
| Google Gemini | Titans: neural long-term memory | https://arxiv.org/abs/2501.00663 |
| Google Gemini | AlphaEvolve | https://arxiv.org/abs/2506.13131 |
| xAI Grok | xAI | https://x.ai/ |
| DeepSeek | DeepSeek-V3 technical report | https://arxiv.org/abs/2412.19437 |
| DeepSeek | DeepSeek-R1 technical report | https://arxiv.org/abs/2501.12948 |
| Qwen / Alibaba | Qwen2.5 technical report | https://arxiv.org/abs/2412.15115 |
| Qwen / Alibaba | Qwen-Agent framework | https://github.com/QwenLM/Qwen-Agent |
| Meta Llama | Llama 3 herd of models | https://arxiv.org/abs/2407.21783 |
| Meta Llama | Llama Stack | https://llama-stack.readthedocs.io/ |
| Meta Llama | PurpleLlama safety tooling | https://github.com/meta-llama/PurpleLlama |
| Mistral AI | Mistral AI releases | https://mistral.ai/news/ |
| Perplexity | Perplexity blog | https://www.perplexity.ai/hub/blog |
| Moonshot Kimi | Kimi k1.5 technical report | https://arxiv.org/abs/2501.12599 |

## Human edits

- All ten reconstructions: drafted with AI assistance, then reviewed and
  edited by the researcher for consistency across the 11 comparison
  dimensions, removal of generic filler, and sharpening of each family's
  distinctive thesis.
- `comparison.csv`: values written by the researcher from the raw outputs;
  cross-checked for consistent semantics per column.
- `summary.md` / `synthesis.md`: original analysis by the researcher; the
  synthesis was additionally mapped against the actual Cognitive-OS
  repository layout (`core/`, `modules/`, `planner/`, `self_model/`,
  `evolution/`).
- No third-party text beyond short paraphrases; no copyrighted text
  reproduced at length.

## License / reuse

This research packet is contributed under the repository's existing license.
Public facts about model families (paper titles, method names) are not
copyrightable expression; all prose is original.
