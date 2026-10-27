# Sources

**Date:** 2026-10-25 02:41-02:56 UTC  
**Collection Method:** Direct API calls to each system  
**Authorization:** All systems are publicly accessible via their respective APIs

## Systems Used

### 1. DeepSeek-V4-Flash
- **Model ID:** deepseek-v4-flash
- **API Provider:** SenseNova (https://token.sensenova.cn)
- **API Base:** https://token.sensenova.cn/v1
- **Access Date:** 2026-10-25 02:46 UTC
- **Model URL:** https://token.sensenova.cn/v1/models
- **Documentation:** https://token.sensenova.cn/docs
- **Notes:** Primary DeepSeek model available through SenseNova API

### 2. GLM-5.2
- **Model ID:** glm-5.2
- **API Provider:** SenseNova (https://token.sensenova.cn)
- **API Base:** https://token.sensenova.cn/v1
- **Access Date:** 2026-10-25 02:48 UTC
- **Model URL:** https://token.sensenova.cn/v1/models
- **Documentation:** https://token.sensenova.cn/docs
- **Notes:** Zhipu AI GLM model available through SenseNova API

### 3. Agnes-2.5-Flash
- **Model ID:** agnes-2.5-flash
- **API Provider:** AgnesAI (https://apihub.agnes-ai.com)
- **API Base:** https://apihub.agnes-ai.com/v1
- **Access Date:** 2026-10-25 02:54 UTC
- **Model URL:** https://apihub.agnes-ai.com/v1/models
- **Documentation:** https://apihub.agnes-ai.com/docs
- **Notes:** Fast variant of Agnes-2.5 model

### 4. Agnes-2.5-Pro-Alpha
- **Model ID:** agnes-2.5-pro-alpha
- **API Provider:** AgnesAI (https://apihub.agnes-ai.com)
- **API Base:** https://apihub.agnes-ai.com/v1
- **Access Date:** 2026-10-25 02:53 UTC
- **Model URL:** https://apihub.agnes-ai.com/v1/models
- **Documentation:** https://apihub.agnes-ai.com/docs
- **Notes:** Alpha release of Agnes-2.5 Pro model

### 5. Agnes-2.5-Pro-Beta
- **Model ID:** agnes-2.5-pro-beta
- **API Provider:** AgnesAI (https://apihub.agnes-ai.com)
- **API Base:** https://apihub.agnes-ai.com/v1
- **Access Date:** 2026-10-25 02:54 UTC
- **Model URL:** https://apihub.agnes-ai.com/v1/models
- **Documentation:** https://apihub.agnes-ai.com/docs
- **Notes:** Beta release of Agnes-2.5 Pro model

### 6. Agnes-3.0-Flash
- **Model ID:** agnes-3.0-flash
- **API Provider:** AgnesAI (https://apihub.agnes-ai.com)
- **API Base:** https://apihub.agnes-ai.com/v1
- **Access Date:** 2026-10-25 02:53 UTC
- **Model URL:** https://apihub.agnes-ai.com/v1/models
- **Documentation:** https://apihub.agnes-ai.com/docs
- **Notes:** Latest Agnes 3.0 model, flash variant

### 7. SenseNova-DeepSeek-V4-Flash (duplicate call)
- **Model ID:** deepseek-v4-flash
- **API Provider:** SenseNova (https://token.sensenova.cn)
- **API Base:** https://token.sensenova.cn/v1
- **Access Date:** 2026-10-25 02:43 UTC
- **Model URL:** https://token.sensenova.cn/v1/models
- **Documentation:** https://token.sensenova.cn/docs
- **Notes:** Second call to same model for comparison; output differs due to temperature sampling

## Failed Systems (Attempted but Not Included)

### DeepSeek-V4.1-Flash
- **Model ID:** deepseek-v4.1-flash
- **API Provider:** SenseNova (https://token.sensenova.cn)
- **Error:** HTTP 403 Forbidden
- **Notes:** Model exists but not accessible with current API key

### DeepSeek-Flash
- **Model ID:** deepseek-flash
- **API Provider:** SenseNova (https://token.sensenova.cn)
- **Error:** Returned 0 chars (6141 tokens allocated, empty response)
- **Notes:** Model responded but generated no content

### SenseNova-U1.5-Lite
- **Model ID:** sensenova-u1.5-lite
- **API Provider:** SenseNova (https://token.sensenova.cn)
- **Error:** HTTP 404 Not Found
- **Notes:** Model name may be incorrect or model unavailable

### Kimi-K3
- **Model ID:** kimi-k3
- **API Provider:** SenseNova (https://token.sensenova.cn)
- **Error:** HTTP 400 Bad Request
- **Notes:** Request format may be incompatible with this model

## Human Edits

**No human edits were applied to any outputs.** All content in `raw_outputs/` is raw model output without modification. The analysis, summary, and synthesis documents were written by the researcher based on the raw outputs.

## Licensing

All models used are available through their respective providers' APIs. The raw outputs are generated content and may be subject to the terms of service of each provider. This research packet is provided for academic and research purposes.

## Reproducibility

This research can be reproduced by:
1. Accessing the APIs listed above
2. Using the prompt from `prompts.md`
3. Using the parameters: max_tokens=6000, temperature=0.7
4. Collecting outputs for each system
5. Running the analysis as described in `summary.md` and `synthesis.md`

Note: Due to the stochastic nature of LLM generation (temperature=0.7), exact reproduction is not possible. Different runs may produce different outputs.

## Contact

For questions about this research, contact:
- GitHub: gin (Hermes Agent)
- Bounty Issue: https://github.com/aLexzzz430/Cognitive-OS/issues/5
