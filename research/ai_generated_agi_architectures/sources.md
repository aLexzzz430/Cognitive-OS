# Sources and Attribution

## Directly Accessed AI Systems

| # | Model | Provider | Access Method | Date | Chars | Edits |
|---|-------|----------|---------------|------|-------|-------|
| 1 | GLM-5.2 | Zhipu AI (Volcengine ARK) | ARK API chat/completions | 2026-07-11 | 9,924 | None |
| 2 | GPT-5 (Codex) | OpenAI | Direct generation (Codex Desktop) | 2026-07-11 | 11,157 | None |
| 3 | GPT-OSS 20B | OpenAI (Pollinations.ai) | Pollinations GET API | 2026-07-11 | 4,790 | None |

## Additional Reference Material (Web-Scraped)

| # | Source | URL | Method | Date | Type |
|---|--------|-----|--------|------|------|
| 4 | Anthropic Institute | anthropic.com/institute/recursive-self-improvement | Firecrawl scrape | 2026-07-11 | Published org position (not direct model output) |
| 5 | arXiv | arxiv.org/abs/2310.13005 | Firecrawl scrape | 2026-07-11 | Academic paper (human-authored) |

## Access Limitations

| System | Method | Failure |
|--------|--------|---------|
| Claude (Anthropic) | DuckDuckGo AI Chat | VQD token not returned |
| Llama 3.1 (Meta) | DuckDuckGo AI Chat | VQD token not returned |
| Mixtral (Mistral) | DuckDuckGo AI Chat | VQD token not returned |
| GPT-4o mini (OpenAI) | DuckDuckGo AI Chat | VQD token not returned |
| Llama 3.2 (Meta) | HuggingFace Inference API | SSL EOF error |
| Qwen 2.5 (Alibaba) | HuggingFace Inference API | SSL EOF error |
| Gemma 2 (Google) | HuggingFace Inference API | SSL EOF error |
| Phi-3 (Microsoft) | HuggingFace Inference API | SSL EOF error |
| Gemini (Google) | Google Gen Language API | HTTP 403 - needs API key |
| DeepSeek | DeepSeek API | HTTP 401 - needs API key |
| Qwen (Alibaba) | DashScope API | HTTP 401 - needs API key |
| Moonshot/Kimi | Moonshot API | HTTP 401 - needs API key |
| Baichuan | Baichuan API | HTTP 401 - needs API key |
| Minimax | Minimax API | HTTP 401 - needs API key |
| Doubao (ByteDance) | ARK API (non-coding) | HTTP 404 - not on coding plan |

Root cause: SSL connectivity issues with international APIs; Chinese APIs accessible but need auth credentials.

## Ethical Compliance

- No private API keys, tokens, or proprietary content used
- No paywalls bypassed
- All accessed systems were public or personally accessible
- All sources clearly attributed
- No outputs fabricated
- Access limitations documented transparently
