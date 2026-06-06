# Sources and Model Metadata

## Successfully Queried Models

All 7 models were queried via the **linkapi.org** API on **June 6, 2026**, using the **exact same prompt** (see prompts.md) with temperature 0.7 and max tokens 4096.

| # | Model Name | API Provider | Access Date | File Name | Notes |
|---|---|---|---|---|---|
| 1 | **GPT-4o** | linkapi.org | 2026-06-06 | `gpt_4o_RAW.txt` | OpenAI's multimodal flagship model. Proposed OODA-loop AGI with hybrid neural-symbolic architecture. |
| 2 | **GLM-4 (v4.7)** | linkapi.org | 2026-06-06 | `glm_4_7_RAW.txt` | Zhipu AI's latest GLM model. Proposed "Project AETHER" with Active Inference framework and Video Diffusion world model. |
| 3 | **Llama-3.3-70B-Instruct (FP8 fast)** | linkapi.org | 2026-06-06 | `llama_3_3_70b_instruct_fp8_fast_RAW.txt` | Meta's open-weight Llama 3.3 (70B, FP8 quantized, fast inference). Proposed "Erebus" with graph-based STM and self-modifying code. |
| 4 | **Qwen3.5-Plus** | linkapi.org | 2026-06-06 | `qwen3_5_plus_RAW.txt` | Alibaba's Qwen 3.5 Plus model. Proposed "Cognitive Kernel Architecture (CKA)" with Event Sourcing and Do-Calculus causal reasoning. |
| 5 | **Qwen3-Max** | linkapi.org | 2026-06-06 | `qwen3_max_RAW.txt` | Alibaba's larger Qwen3-Max model. Proposed "Modular Recursive Intelligence Engine (MoRIE)" with hyperdimensional vectors and genetic programming skill synthesis. |
| 6 | **DeepSeek-Chat** | linkapi.org | 2026-06-06 | `deepseek_chat_RAW.txt` | DeepSeek's chat-optimized model. Proposed "Project Prometheus" with Dual-Process (System 1/System 2) architecture and Differentiable Neural Dictionary. |
| 7 | **DeepSeek-R1** | linkapi.org | 2026-06-06 | `deepseek_r1_RAW.txt` | DeepSeek's advanced reasoning model. Proposed "Sylvan" architecture with formal verification (Dafny/Coq), Control Vectors, and blockchain audit. |

## Models Attempted But Unavailable

The following models were attempted but could not be accessed:

| Model | Reason for Failure |
|---|---|
| **Claude Sonnet 4** | API timeout — the linkapi.org endpoint did not return a response within the configured timeout period. Repeated attempts (n=3) all resulted in timeouts. |
| **Gemini 2.5 Pro** | API timeout — the linkapi.org endpoint for this model consistently timed out. May indicate the model was not yet available through this API provider or experienced capacity issues at query time. |

## Query Metadata

| Parameter | Value |
|---|---|
| Query Date | 2026-06-06 |
| API Endpoint | `https://api.linkapi.org/v1/chat/completions` |
| API Provider | linkapi.org (multi-model aggregation API) |
| Temperature | 0.7 (default) |
| Max Tokens | 4096 |
| System Prompt | None (user prompt only) |
| Model Parameters | All default settings per model |

## Raw Output Files

All raw outputs are stored in the `raw_outputs/` directory with the naming convention `{model_name}_RAW.txt`. Each file contains:
- A header with model name, prompt reference, and date
- The complete, unedited model response to the AGI architecture prompt

No post-processing, filtering, or editing was applied to any raw output. These are the exact responses returned by the API.

## Research Packet Files

| File | Description |
|---|---|
| `README.md` | Overview, collection methodology, headline findings |
| `prompts.md` | Exact prompt text and query metadata |
| `raw_outputs/` | 7 raw model outputs (unedited) |
| `comparison.csv` | Structured 10-dimension comparison across all 7 models |
| `summary.md` | Common patterns, disagreements, notable ideas |
| `synthesis.md` | Proposed combined architecture (Project ARGUS) |
| `sources.md` | This file — model metadata and access information |

## Citation

If referencing this research packet, please cite as:

> Cognitive-OS Research Team. (2026). *AI-Generated AGI Architectures: A Comparative Study of 7 Frontier Models*. Retrieved from https://github.com/qingluan-bot/Cognitive-OS

---

*Research conducted June 2026. All model outputs are the responses of their respective AI systems to a public prompt and do not necessarily reflect the views of the Cognitive-OS project or its contributors.*
