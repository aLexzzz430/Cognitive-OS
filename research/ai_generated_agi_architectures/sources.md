# Sources and attribution

All outputs were generated locally on 2026-09-09 using public Hugging Face model snapshots. Links below pin the model revision rather than a mutable model name. Provider/tool for every execution: local Hugging Face Transformers on the contributor's workstation; the named model organizations did not host these requests.

| System/model | Revision | License in model metadata | UTC collection start | New tokens | Stop |
|---|---|---|---|---:|---|
| [XHToken/Spark-X2.5-1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B/tree/448e61eb392c00f2c403185c5b56d5e0665bfaab) | `448e61eb392c00f2c403185c5b56d5e0665bfaab` | apache-2.0 | 2026-09-09T10:16:24.294704+00:00 | 1400 | length |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | apache-2.0 | 2026-09-09T10:20:35.944982+00:00 | 704 | eos |
| [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/tree/2fe192450127e6a83f7441aef6e3ca586c338b77) | `2fe192450127e6a83f7441aef6e3ca586c338b77` | mit | 2026-09-09T10:24:23.351118+00:00 | 1244 | eos |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) | `31b70e2e869a7173562077fd711b654946d38674` | apache-2.0 | 2026-09-09T10:25:21.982553+00:00 | 1400 | length |
| [ibm-granite/granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct/tree/707f574c62054322f6b5b04b6d075f0a8f05e0f0) | `707f574c62054322f6b5b04b6d075f0a8f05e0f0` | apache-2.0 | 2026-09-09T10:49:45.157700+00:00 | 1275 | eos |
| [tiiuae/Falcon3-1B-Instruct](https://huggingface.co/tiiuae/Falcon3-1B-Instruct/tree/28ba2251970a01dd1edc7ba7dad2eb71216ccfdf) | `28ba2251970a01dd1edc7ba7dad2eb71216ccfdf` | TII Falcon-LLM License 2.0 | 2026-09-09T10:50:29.570799+00:00 | 1115 | eos |
| [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6) | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | apache-2.0 | 2026-09-09T10:53:35.533665+00:00 | 1144 | eos |
| [h2oai/h2o-danube3-4b-chat](https://huggingface.co/h2oai/h2o-danube3-4b-chat/tree/1e5c6fa6620f8bf078958069ab4581cd88e0202c) | `1e5c6fa6620f8bf078958069ab4581cd88e0202c` | apache-2.0 | 2026-09-09T11:09:30.948093+00:00 | 1400 | length |

## Access, ancestry and licenses

The publishers are XHToken (Spark), Qwen/Alibaba, Microsoft (Phi), Hugging Face (SmolLM), IBM (Granite), Technology Innovation Institute/Falcon-LLM Team, TinyLlama and H2O.ai (Danube). These are eight distinct named projects/checkpoints, not eight proven independent training lineages. TinyLlama's card states it adopts the Llama2 architecture/tokenizer; Danube3 adjusts the Llama2 architecture and uses the Mistral tokenizer. Shared architecture, synthetic training material and training-data overlap limit independence. No frontier-service names are used to label local outputs.

License metadata and public model cards were inspected before use. Falcon is not Apache-2.0: its card links the [TII terms](https://falconllm.tii.ae/falcon-terms-and-conditions.html), including the applicable acceptable-use conditions. The Falcon-LLM Team's [release article](https://huggingface.co/blog/falcon3) is an additional attribution source. The intended use is lawful local architecture research. Model licensing is not a warranty that generated text is correct or unique. We redistribute generated proposals and small collection records, not model weights or the full downloaded model-card archive.

## Runtime and evidence

The responses record Python3.13.5, Windows, an NVIDIA GeForce RTX4070, CUDA12.8, torch2.8.0+cu128 and Transformers4.57.1, plus exact supporting package versions. Each response is the authoritative runtime record. Danube additionally required sentencepiece0.2.2 and protobuf7.36.1 before its successful invocation. The collector remains the same for all eight successful runs; its SHA-256 is recorded in every response. No model was re-run after producing a completed response.

`request.json` stores the exact user message, rendered chat prompt, input IDs and generation settings. `response.json` stores generated token IDs, unedited decoded text, timestamps, stop condition and hashes. `model-files.json` records the actual weight/config/tokenizer/custom-code files used locally. File hashes can establish a match to these records but cannot independently attest execution or validate claims in generated text.

## Edits and byte handling

Raw decoded outputs were not rewritten, corrected, translated, continued or deduplicated. Special tokens remain in raw_outputs; readable.txt removes tokenizer special tokens only. The collector wrote text on Windows, which can serialize newlines as CRLF. Its output_sha256 is over the decoded UTF-8 string (LF), while request_sha256 and the packet manifest are over actual file bytes. verify_packet.py checks both deliberately. Packet .gitattributes prevents Git from silently normalizing preserved evidence. Line references in comparison.csv count text lines after universal-newline decoding.

Analysis, comparison assessments, synthesis and scripts are separate from the eight recorded generations. The raw outputs have no human edits. The model assertions that components are implemented or benchmarks achieved have not been promoted to facts. Failed downloads/tokenizer setup and other incidents are disclosed in collection/incidents.md.

## Repository and external references

- [Original task and acceptance criteria](https://github.com/aLexzzz430/Cognitive-OS/issues/5).
- [Inspected repository baseline](https://github.com/aLexzzz430/Cognitive-OS/tree/e20d2ff4d5c84d4c11c87218c4ae9a04ab0046ca).
- Source paths supporting integration observations are linked in synthesis.md. Those observations are separate from model output.
- No proprietary prompts, private API output, credentials, account screenshots or private customer data were used. Existing competing submissions were not used as sources for model outputs or the comparison.
