# Sources, attribution and editing record


## Primary local systems

Each system is executed through the same local llama.cpp runtime, using its public GGUF chat template. The linked repositories identify the actual weight distribution, including community quantizers where applicable. A community GGUF distributor is not presented as the original model developer. Dates below are UTC.

| Alias / model | Developer and weight distributor | Weight source | Source downloaded / verified | Response completed |
| --- | --- | --- | --- | --- |
| `smollm2` / SmolLM2 1.7B Instruct | HuggingFaceTB | [HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/tree/2d4a76a30b4af41ecd395c35725ac11688d4cfe4) | 2026-09-08T18:56:08.468289+00:00 | 2026-09-08T19:03:31.552280+00:00 |
| `qwen25` / Qwen2.5 1.5B Instruct | Qwen | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9) | 2026-09-08T19:04:07.535354+00:00 | 2026-09-08T19:12:21.575675+00:00 |
| `granite33` / Granite 3.3 2B Instruct | IBM Granite | [ibm-granite/granite-3.3-2b-instruct-GGUF](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF/tree/7cdf86ccd1f1bb3491c9b7017b033f2e51367397) | 2026-09-08T19:39:51.443219+00:00 | 2026-09-08T19:44:36.118101+00:00 |
| `tinyllama` / TinyLlama 1.1B Chat v1.0 | TinyLlama; GGUF by TheBloke | [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/52e7645ba7c309695bec7ac98f4f005b139cf465) | 2026-09-08T19:45:35.083461+00:00 | 2026-09-08T19:47:32.830910+00:00 |
| `phi3` / Phi-3 mini 4k Instruct | Microsoft | [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/a64113399c2f6b8ad3e11c394733a2ddadaa7f33) | 2026-09-08T19:51:30.167408+00:00 | 2026-09-08T20:03:18.180745+00:00 |
| `qwen3` / Qwen3 1.7B | Qwen | [Qwen/Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/tree/90862c4b9d2787eaed51d12237eafdfe7c5f6077) | 2026-09-08T20:04:02.023673+00:00 | 2026-09-08T20:12:33.103579+00:00 |
| `falcon3` / Falcon3 1B Instruct | TII; GGUF by bartowski | [bartowski/Falcon3-1B-Instruct-GGUF](https://huggingface.co/bartowski/Falcon3-1B-Instruct-GGUF/tree/9e280a1e3b7f287edb7248042e3d14f893227b21) | 2026-09-08T20:13:09.868542+00:00 | 2026-09-08T20:16:49.049787+00:00 |
| `danube3` / H2O Danube3 500m Chat | H2O.ai; GGUF by bartowski | [bartowski/h2o-danube3-500m-chat-GGUF](https://huggingface.co/bartowski/h2o-danube3-500m-chat-GGUF/tree/34005c67faf918e19c8748e501fa8418b51060fe) | 2026-09-08T20:17:22.738887+00:00 | 2026-09-08T20:18:28.180171+00:00 (prompt echo; no proposal) |

For each completed system, `collection/<alias>/model.json` is the public repository metadata, including the LFS object identity. `source.json` records the pinned revision, exact weight URL, file size and SHA-256 verified after download. `request.json`, `response.json` and `run.json` preserve the input, full response, finish reason, usage, timestamps and hashes. `raw_outputs/<alias>.txt` contains the exact response content. The metadata and hashes permit consistency checks and repeat retrieval; they are not a provider-signed attestation of model authorship.

The inspected distribution cards identify Apache-2.0 for SmolLM2, Qwen2.5, Granite, TinyLlama, Qwen3 and Danube3; MIT for Phi-3; and the custom Falcon LLM license for Falcon3. The Falcon card links the [TII terms](https://falconllm.tii.ae/falcon-terms-and-conditions.html), which include acceptable-use conditions. Falcon3 is therefore not described here as Apache-2.0. Model weights are not redistributed in this packet.

## Runtime and method

The generation runtime is [llama.cpp release b10809](https://github.com/ggml-org/llama.cpp/releases/tag/b10809), commit `5266f24da`. The official macOS arm64 archive was retrieved on 2026-09-08 and its SHA-256 verified as `7d692df9e1e386e62f1c12b843903218041e6cd74c9415aa39a7ed3176f9eaa2`. Its version command reported `0.4.0-dev` and the recorded build/commit. All eight local runs completed on 2026-09-08; Danube3 returned a prompt echo and is recorded as a failed proposal request. The unmodified model-specific requests and run commands preserve generation settings. The exact user prompt is in [prompts.md](prompts.md) and [collection/prompt.txt](collection/prompt.txt).

The roster comprises eight distinct model systems, not eight independent training lineages. Qwen2.5 and Qwen3 are two generations in one named family. Parameter count, quantization and provider chat templates differ. StableLM was considered but excluded because its public model terms restrict commercial use. DeepSeek was not collected. No unavailable system output is fabricated.

## Exploratory hosted reference

[Google Gemini web app](https://gemini.google.com/app), UI mode Flash, accessed 2026-09-08 through an authorized account. One fresh prompt produced two automatically offered A/B candidate answers, preserved at 19:27:26 UTC in [collection/gemini-flash-reference](collection/gemini-flash-reference/analysis-notes.md). Backend model revision, service instructions, sampling parameters and token counts are unknown. No preference vote was submitted. This is one hosted reference, not two independently identified models.

Extraction copied the rendered `innerText` of each observed answer panel. No wording was edited; rendered plaintext can change code or mathematical formatting from the service’s internal representation. The combined file `raw_outputs/gemini_flash_reference.txt` adds explicitly collector-authored A/B labels and separating newlines around the unchanged strings; the original panel texts remain in the collection folder, with their own hashes. This is an assembled view of one system’s two candidates, not two independent systems. Exact character counts and hashes are in `source.json`; the prompt is preserved as `request.txt`. Private account screenshots, service instructions and authenticated conversation references are not included. The public app link identifies the service, not a public replay of that private conversation.

The supplement was added after examining SmolLM2 and Qwen2.5 and is explicitly exploratory. It is not parameter matched with the local collection.

## Repository and technical sources

- [Cognitive-OS issue #5](https://github.com/aLexzzz430/Cognitive-OS/issues/5), accessed 2026-09-08: required packet and eleven comparison dimensions. Reward and submission administration are not research results.
- [Pinned Cognitive-OS source](https://github.com/aLexzzz430/Cognitive-OS/tree/e20d2ff4d5c84d4c11c87218c4ae9a04ab0046ca), inspected 2026-09-08: the specific source surfaces and interpretations are listed in [repository-context.md](collection/repository-context.md). The models received the same short context paragraph, not this full source inspection.
- [SQLite Atomic Commit](https://sqlite.org/atomiccommit.html) and [Write-Ahead Logging](https://sqlite.org/wal.html), accessed 2026-09-08: context for database transaction boundaries and the analyst’s [isolated example](collection/transaction-example.md). The local measured cases are evidence from the supplied script; the documentation is not a report of executing Cognitive-OS.

## Human and analyst editing

No human edits or analyst wording changes were made to the primary raw response content. Whole HTTP responses are preserved separately. Repeated passages, unsupported claims, omissions and truncation remain present. Hosted content undergoes only the stated rendered-text extraction, with no word-level cleanup or A/B selection.

The comparison, notes, overview, summary, synthesis, scripts and source catalog were authored by the assisting Codex agent. There was no independent human review of the generated proposals. They are labelled separately from generated proposals and are not counted as additional independent model outputs. The analyst’s transaction-ordering correction and wider proposed experiments are additions, not statements secretly inserted into a model’s response. The original transaction-example result is retained; broader proposed evaluations have not been run.

No private API keys, account tokens, hidden system prompts, proprietary repository content or payment credentials are used as research evidence.
