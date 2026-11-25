# Sources and attribution

All final answers were generated for the same public research question on 25 November 2026. Exact service-returned model labels, timestamps, receipt identifiers where available, completion status and SHA-256 digests are in [provenance.csv](provenance.csv). Each digest covers the UTF-8 bytes of the named file in [raw_outputs/](raw_outputs/). A digest verifies file integrity. A request identifier records what the interface reported; it does not attest to installed model weights.

| Record | Access and lineage evidence |
| --- | --- |
| qwen | AI Horde returned `Qwen/Qwen3.8-Flash-Next`. The worker name included `Uncensored-NVFP4`; this run cannot be equated to an official provider endpoint. |
| lfm | The AI Horde worker reported a quantized LFM3-1.7B label. [Liquid AI model card](https://huggingface.co/LiquidAI/LFM3-1.7B) |
| gemma | Worker-reported Gemma 4 E4B instruction model on AI Horde. [Google model card](https://huggingface.co/google/gemma-4-E4B-it) |
| llama | Worker-reported Llama 3.2 3B Instruct on AI Horde. [Meta model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| rocinante | AI Horde served a Rocinante-X-12B-v1 worker label. The publisher lists Mistral NeMo Instruct as its base. [Publisher's card](https://huggingface.co/TheDrummer/Rocinante-X-12B-v1) |
| magnum | AI Horde served magnum-v2-4b. Its published base is NVIDIA's Llama 3.1 Minitron. [Publisher's card](https://huggingface.co/anthracite-org/magnum-v2-4b) |
| cydonia | The worker returned a Cydonia 24B v4zg quantization label. The publisher has several Cydonia versions. The exact weight revision and ancestry of this worker were not attested, so it supplies no additional independent-lineage claim. [Publisher's model list](https://huggingface.co/TheDrummer/models) |
| stheno | AI Horde served L3-8B-Stheno-v3.2. Its Llama lineage overlaps with other entries. [Publisher's card](https://huggingface.co/Sao10K/L3-8B-Stheno-v3.2) |
| rwkv | The AI Horde native text endpoint returned the label `RWKV/RWKV7-2.9B-20260805` and state `ok`. The weight revision was not independently verified. [Native API](https://aihorde.net/api/) |
| codex | A fresh OpenAI Codex agent received only the common task text as its user task, with no prior conversation inherited. The final answer was transcribed directly. An exact serving-model SKU and a public completion receipt were unavailable. |
| smollm | A public Hugging Face Space returned a completed Gradio event. The pinned application source maps the chosen menu entry to HuggingFaceTB/SmolLM2-1.7B-Instruct. [Space code](https://huggingface.co/spaces/Harley-ml/SmolLM2_Chat/blob/0b3b94104b208b146ad2f8b8d892c5658be9adf8/app.py#L19) · [Model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/README.md) |

AI Horde distributes public prompts to volunteer workers and publishes its access terms. Eight main records used its chat interface and one used the native text endpoint. Those nine answers share infrastructure. The remaining records came from Codex and a Hugging Face Space. The set spans Qwen, Liquid, Gemma, Llama, Mistral NeMo-derived, RWKV, OpenAI Codex and SmolLM systems under the attribution limits above. [Service description](https://github.com/Haidra-Org/AI-Horde#openai-compatible-chat-completions) · [Service terms](https://aihorde.net/terms/)

The raw files contain final user-visible response text only, with no content edits. Codex message framing and its terminal newline were omitted. The commentary, engineering assessments and executable example are AI-assisted editorial work. Missing cut-off text was never reconstructed. Public demo requests for GPT-OSS, Cohere, Falcon and Granite returned service errors and contributed zero rows.
