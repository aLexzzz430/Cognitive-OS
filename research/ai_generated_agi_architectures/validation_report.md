# Validation Report

Collection date: 2026-05-27

This report maps the issue acceptance criteria to the submitted packet and records local verification results for reviewers.

## Acceptance Criteria Mapping

| Requirement | Evidence in this packet |
| --- | --- |
| At least 8 model/system outputs | 10 model-family outputs in `raw_outputs/` and 10 entries in `collection_manifest.json`. |
| Outputs clearly attributed | Every raw file includes provider/tool, model ID, access date, UTC batch timestamp, response ID, token usage, and human-edit statement. `collection_manifest.json` adds per-response `response_created_utc` timestamps. |
| Raw outputs separate from analysis | Raw outputs are stored only in `raw_outputs/`; analysis is in `comparison.csv`, `summary.md`, and `synthesis.md`. |
| Structured comparison table | `comparison.csv` has 10 rows across the requested architecture dimensions; `comparison_by_dimension.csv` adds a dimension-first companion view. |
| Concrete synthesis | `synthesis.md` defines an event ledger, state store, memory manager, planner, policy/lease engine, verifier gate, executor, runtime modes, a 90-day implementation plan, SQL/JSON schemas, and a verifier interface. |
| No private keys/tokens/screenshots/private content | No screenshots or private prompts are included; credential-pattern scan returned no matches. |
| No fabricated sources | `collection_manifest.json` and the raw files record provider-returned response IDs and usage metadata for each collected output. The manifest also records public OpenRouter model-catalog verification from 2026-05-27T04:18:00+00:00. |
| Traceability from analysis to raw outputs | `evidence_matrix.md` maps every comparison dimension and synthesis component back to raw-output sections and files. |

## Manifest Traceability Fields

- `prompt_sha256`: SHA-256 of the exact user prompt recorded in `collection_manifest.json`.
- `model_catalog_source`: public OpenRouter models endpoint used for model ID verification.
- `model_catalog_verified_at_utc`: timestamp for public model catalog verification.
- `response_created_utc`: per-response timestamp decoded from the returned OpenRouter response ID.
- `raw_sha256`: SHA-256 of the preserved raw-output file.
- `raw_bytes`: byte count of the preserved raw-output file.
- `raw_word_count`: word count of the model output body after the metadata wrapper.

## Local Verification Results

Manifest and file-count consistency:

```text
manifest_models 10
raw_files 10
comparison_rows 10
missing_raw_refs []
extra_raw_files []
comparison_models
OpenAI GPT
Claude
Gemini
Grok
DeepSeek
Qwen
Llama
Mistral
Perplexity Sonar
Cohere Command
```

Repository boundary check:

```text
python3 scripts/check_conos_repo_layout.py

Repository root: /Users/husky/new-new-revenue/external/Cognitive-OS
Logical layers:
- conos-core: core/, decision/, evolution/, memory/, modules/, planner/, self_model/, state/, trace/
- adapter: integrations/local_machine/
- private-cognitive-core: core/orchestration/structured_answer.py, modules/hypothesis/mechanism_posterior_updater.py
- runtime: runtime/, audit/, reports/
No forbidden public/private boundary imports detected.
Adapter registry points only to adapter-layer paths.
```

Whitespace check:

```text
git diff --check
```

Result: passed with no output.

Credential-pattern scan:

```text
grep -RInE '(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|re_[A-Za-z0-9_]{20,}|xai-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20,}|pplx-[A-Za-z0-9]{20,})' research/ai_generated_agi_architectures
```

Result: no matches.

## Raw Output SHA-256 Hashes

| Raw output file | SHA-256 |
| --- | --- |
| `raw_outputs/anthropic_claude_sonnet_46.md` | `2ae3a4a9cb4f18fd302b6d52f099a92ee717b8c815988b776a20076542561a64` |
| `raw_outputs/cohere_command_a.md` | `aa41ddbcdeaaa9c051f10d9a798d7c59ca51e9931e87fce47d3497f2805432ff` |
| `raw_outputs/deepseek_v32.md` | `22d9806fe56ac4508b5698934aa720919896f5fc6c453a95c1a7a1847491c510` |
| `raw_outputs/google_gemini_31_pro.md` | `3dc26b7d116c323c592d00559606c384479dbd8cd8851ac196e80eeaeaa80947` |
| `raw_outputs/meta_llama4_maverick.md` | `fec0b1e847e1bbaadd36d0239f8be3acb6fa7ef35b7d5fa249054c151aa6b3eb` |
| `raw_outputs/mistral_large_2512.md` | `5d6022c2ad6b669d7ad7554e039cef6c887bc9bc1b929d0fc86b7129b2282925` |
| `raw_outputs/openai_gpt54_mini.md` | `e6e69aa5ab8fae53373a1e73953a93366d349fdb952a1cc3d333ed9c0bb32335` |
| `raw_outputs/perplexity_sonar_pro.md` | `6f429cf807918e2c464ef498e2fa5272683e305ca5af02a4cc829f202bef6c7c` |
| `raw_outputs/qwen37_max.md` | `fce5eedea72813c5402b78618035730246aa35f6d8ff6a6519ca03e5d815ead4` |
| `raw_outputs/xai_grok_43.md` | `303633da1c12385fff05584169c4bb4a1c512c436638bc52ea9391da7b20fcd1` |
