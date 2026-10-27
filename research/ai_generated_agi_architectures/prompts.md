# Prompts Used

**Date:** 2026-10-25 02:41-02:56 UTC  
**Collection Method:** Standard prompt sent to each AI system via API

## Standard Prompt (used for all systems)

```
You are a senior AI researcher. Design a detailed architecture for AGI. Cover: 
1)Memory(short/long/episodic/procedural) 
2)Reasoning/planning loop 
3)Learning/self-improvement 
4)Tool use/action 
5)World model 
6)Safety/governance 
7)Evaluation/benchmarks 
8)Persistence/runtime 
9)Multi-agent orchestration 
10)Engineering feasibility 
11)Originality. 
Be concrete with technical details. Current LLM as foundation.
```

## Per-System Prompt Adaptations

No adaptations were necessary. The same prompt was sent to all systems without modification.

## Model Parameters

All systems used the following parameters:
- **max_tokens:** 6000
- **temperature:** 0.7
- **response_format:** text

## Systems and Models

| # | System Label | Model ID | API Base | Date |
|---|--------------|----------|----------|------|
| 1 | DeepSeek-V4-Flash | deepseek-v4-flash | https://token.sensenova.cn/v1 | 2026-10-25 02:46 |
| 2 | GLM-5.2 | glm-5.2 | https://token.sensenova.cn/v1 | 2026-10-25 02:48 |
| 3 | Agnes-2.5-Flash | agnes-2.5-flash | https://apihub.agnes-ai.com/v1 | 2026-10-25 02:54 |
| 4 | Agnes-2.5-Pro-Alpha | agnes-2.5-pro-alpha | https://apihub.agnes-ai.com/v1 | 2026-10-25 02:53 |
| 5 | Agnes-2.5-Pro-Beta | agnes-2.5-pro-beta | https://apihub.agnes-ai.com/v1 | 2026-10-25 02:54 |
| 6 | Agnes-3.0-Flash | agnes-3.0-flash | https://apihub.agnes-ai.com/v1 | 2026-10-25 02:53 |
| 7 | SenseNova-DeepSeek-V4-Flash | deepseek-v4-flash | https://token.sensenova.cn/v1 | 2026-10-25 02:43 |

## Failed Systems

The following systems were attempted but failed:
- **DeepSeek-V4.1-Flash** (deepseek-v4.1-flash): HTTP 403 Forbidden
- **DeepSeek-Flash** (deepseek-flash): Returned 0 chars (6141 tokens allocated, empty response)
- **SenseNova-U1.5-Lite** (sensenova-u1.5-lite): HTTP 404 Not Found
- **Kimi-K3** (kimi-k3): HTTP 400 Bad Request

## Human Edits

No human edits were applied to any outputs. All content is raw model output.
