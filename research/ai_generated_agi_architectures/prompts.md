# Prompts and collection protocol

The exact common user message is [collection/prompt.txt](collection/prompt.txt). That file was frozen before the first successful generation. Each `collection/runs/<system>/request.json` preserves the same message, its SHA-256, the model-specific rendered prompt and input IDs. No system message, simulated expert identity or additional model turn was supplied by the collector.

The request asks for an implementable AGI-oriented proposal without claiming achieved AGI, constrained to a local Python/SQLite runtime and a 12 GB GPU with no paid hosted API. It asks for eleven numbered headings, a mechanism and tradeoff under each, an explicit schema/interface, a worked file-edit crash example, a falsifiable comparison against a simpler baseline, and separation of implemented components from proposals/research. Read the exact file for all wording; this paragraph is only an index.

## Per-model adaptations

There are no semantic edits to the user prompt. `AutoTokenizer.apply_chat_template` supplies each checkpoint's chat serialization with `add_generation_prompt=True` and `enable_thinking=False`. The latter is a template keyword and has no effect if a template does not use it. The actual serialization, not an assumption about that keyword, is preserved in the request.

All successful runs use `max_new_tokens=1400`, `do_sample=False`, `use_cache=True`, seed `20260909`, BF16 model weights, eager attention, one CUDA device and one request at a time. Model-specific BOS/EOS IDs are retained. `use_model_defaults=False` prevents the checkpoint's default sampling configuration from overriding this protocol. The exact expanded generation configuration is in each request. Tokenization differs between models, so the cap is not a matched semantic output length. Stop reasons and token counts are listed in sources.md.

No model was told what another model wrote. No output was repaired to fill missing headings, invented experiments or truncated examples. Analysis is separate from raw files. Qwen2.5-3B was considered and rejected on license grounds before inference; Qwen2.5-1.5B is the actual collected model. Interrupted downloads and a pre-generation Spark configuration failure are recorded, not counted as additional systems.

## Reproduction

Use the exact revisions in `collection/models.json`; install the package versions recorded in the responses on a compatible CUDA/BF16 machine. Downloads use public Hugging Face snapshots without authentication. No weights are committed. Example:

```bash
python research/ai_generated_agi_architectures/collection/download_models.py --slug qwen --model-root /path/to/models
python research/ai_generated_agi_architectures/collection/collect.py --slug qwen --model-dir /path/to/models/qwen
```

The collector deliberately refuses to overwrite a completed response. Reproduction should use a separate checkout/output location and retain a separate result, not modify this evidence. Spark uses custom model code; inspect the pinned Python sources before allowing local execution. Other models use native Transformers implementations. `model-files.json` records hashes of the actual local weights, tokenizer/configuration and Python model files where present. Hardware/library differences can change greedy output, so exact replay on another platform is not promised.
