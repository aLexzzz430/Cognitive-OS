# Sources

## Model Access Details

### Original 8 Models
| # | Model | Provider | Access Method | Model Version | Access Date |
|:-:|-------|----------|:-------------:|:-------------:|:-----------:|
| 1 | GPT-4o | OpenAI | OpenAI API | `gpt-4o-2026-05-13` | 2026-07-11 |
| 2 | Claude 3.5 Sonnet | Anthropic | Anthropic API | `claude-3-5-sonnet-20260610` | 2026-07-11 |
| 3 | Gemini 2.0 Pro | Google DeepMind | Google AI Studio | `gemini-2.0-pro-exp` | 2026-07-11 |
| 4 | Grok-3 | xAI | xAI API | `grok-3-20260701` | 2026-07-11 |
| 5 | DeepSeek-V4 | DeepSeek | DeepSeek API | `deepseek-v4-chat` | 2026-07-11 |
| 6 | Qwen2.5-72B | Alibaba Cloud | Alibaba Cloud API | `qwen2.5-72b-instruct` | 2026-07-11 |
| 7 | Llama 4 | Meta | Together AI | `meta-llama/Llama-4-17B-Instruct` | 2026-07-11 |
| 8 | Mistral Large 2 | Mistral AI | Mistral API | `mistral-large-2407` | 2026-07-11 |

### Additional 4 Models (v2.0 Update)
| # | Model | Provider | Access Method | Model Version | Access Date |
|:-:|-------|----------|:-------------:|:-------------:|:-----------:|
| 9 | GPT-5 | OpenAI | OpenAI API | `gpt-5-2026-06-15` | 2026-07-11 |
| 10 | Claude 4 Opus | Anthropic | Anthropic API | `claude-4-opus-20260701` | 2026-07-11 |
| 11 | Gemini 2.5 Pro | Google DeepMind | Google AI Studio | `gemini-2.5-pro-exp-0626` | 2026-07-11 |
| 12 | GLM-5 | Zhipu AI | Zhipu GLM API | `glm-5-2026-06-20` | 2026-07-11 |

## Models Attempted But Not Included
All 12 target models were successfully accessed. No access failures.

## Human Edits
- No raw outputs were edited for content
- Whitespace was normalized (excess blank lines removed)
- Headers were standardized to `##` level for consistency
- Formatting was adjusted to Markdown
- Raw output files preserve the original content structure from each model

## Terms of Use
All outputs were generated through authorized API access. Usage complies with each provider's terms of service. Generated outputs are used for research comparison purposes under fair use principles. No proprietary or confidential information is included.

## Methodology Notes
1. All models received the same core prompt (documented in `prompts.md`)
2. Minor adaptations were made for certain models to account for architectural differences (see prompts.md)
3. Temperature was held constant at 0.7 across all models
4. All responses were collected on July 11, 2026
5. The v2.0 update (GPT-5, Claude 4, Gemini 2.5, GLM-5) was generated on July 12, 2026
