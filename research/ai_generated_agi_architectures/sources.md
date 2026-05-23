# Sources and Attribution

## AI Systems Used

| Model | Provider | Access Method | Collection Date | Version |
|-------|----------|---------------|-----------------|---------|
| Claude 3.5 Sonnet | Anthropic | API | 2026-05-23 | claude-3-5-sonnet-20241022 |
| GPT-4 Turbo | OpenAI | API | 2026-05-23 | gpt-4-turbo-2024-04-09 |
| Gemini 1.5 Pro | Google DeepMind | API | 2026-05-23 | gemini-1.5-pro-latest |
| DeepSeek-V3 | DeepSeek AI | API | 2026-05-23 | deepseek-chat |
| Qwen2.5-Max | Alibaba Cloud | API | 2026-05-23 | qwen-max |
| Llama 3.1 405B | Meta AI | Open Weights | 2026-05-23 | llama-3.1-405b |
| Mistral Large 2 | Mistral AI | API | 2026-05-23 | mistral-large-latest |
| Grok-2 | xAI | X Platform | 2026-05-23 | grok-2-latest |

## Access Details

### Claude (Anthropic)
- **Access Method**: Anthropic API
- **Endpoint**: https://api.anthropic.com/v1/messages
- **Pricing**: Pay-per-token
- **Documentation**: https://docs.anthropic.com/

### GPT-4 (OpenAI)
- **Access Method**: OpenAI API
- **Endpoint**: https://api.openai.com/v1/chat/completions
- **Pricing**: Pay-per-token
- **Documentation**: https://platform.openai.com/docs/

### Gemini (Google)
- **Access Method**: Google AI API
- **Endpoint**: https://generativelanguage.googleapis.com/v1beta/models/
- **Pricing**: Pay-per-token
- **Documentation**: https://ai.google.dev/docs/

### DeepSeek
- **Access Method**: DeepSeek API
- **Endpoint**: https://api.deepseek.com/v1/chat/completions
- **Pricing**: Pay-per-token
- **Documentation**: https://platform.deepseek.com/docs/

### Qwen (Alibaba)
- **Access Method**: Alibaba Cloud API
- **Endpoint**: https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation
- **Pricing**: Pay-per-token
- **Documentation**: https://help.aliyun.com/document_detail/2712195.html

### Llama (Meta)
- **Access Method**: Open weights, Hugging Face
- **Repository**: https://huggingface.co/meta-llama/Meta-Llama-3.1-405B
- **License**: Llama Community License
- **Documentation**: https://llama.meta.com/

### Mistral
- **Access Method**: Mistral API
- **Endpoint**: https://api.mistral.ai/v1/chat/completions
- **Pricing**: Pay-per-token
- **Documentation**: https://docs.mistral.ai/

### Grok (xAI)
- **Access Method**: X Platform (Twitter)
- **Platform**: https://x.com/grok
- **Pricing**: X Premium subscription
- **Documentation**: https://x.ai/

## Human Edits

The following human edits were performed:
1. **Formatting cleanup**: Standardized markdown formatting across all outputs
2. **Structure alignment**: Ensured consistent section headers across proposals
3. **Language consistency**: Qwen output was generated in Chinese with English section headers added for consistency
4. **No content changes**: The core content of each AI's proposal was preserved without modification

## Data Freshness

- All outputs were collected on 2026-05-23
- Knowledge cutoff dates vary by model:
  - Claude: April 2024
  - GPT-4: April 2024
  - Gemini: January 2024
  - DeepSeek: July 2024
  - Qwen: November 2024
  - Llama: December 2023
  - Mistral: Unknown
  - Grok: Real-time (no cutoff)

## Reproducibility

To reproduce this research:
1. Use the prompts provided in `prompts.md`
2. Query each model with the standard prompt
3. Capture the full output without editing
4. Perform comparison using the dimensions in `comparison.csv`

## License and Usage

- This research packet is provided for academic and research purposes
- Individual AI outputs are subject to their respective providers' terms of service
- The synthesis and analysis are original work
- Attribution to this research packet is required for derivative works
