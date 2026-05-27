# Evidence Matrix

This matrix links comparison and synthesis claims back to the preserved model outputs. All raw files used the same 12-section prompt, so section numbers are comparable across model families.

## Section Key

| Section | Evidence dimension |
| --- | --- |
| 1 | Architecture thesis |
| 2 | Memory architecture |
| 3 | Reasoning and planning loop |
| 4 | Learning or self-improvement |
| 5 | Tool use and action execution |
| 6 | World model and representation layer |
| 7 | Safety and governance |
| 8 | Evaluation and benchmark strategy |
| 9 | Persistence and runtime architecture |
| 10 | Multi-agent and orchestration design |
| 11 | Engineering feasibility |
| 12 | Original or non-obvious insight |

## Dimension-To-Output Map

| Dimension | Raw-output sections used | Analysis files using it |
| --- | --- | --- |
| Memory architecture | Section 2 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |
| Reasoning/planning loop | Section 3 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |
| Learning/self-improvement | Section 4 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |
| Tool use/action execution | Section 5 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |
| World model/representation | Section 6 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |
| Safety/governance layer | Section 7 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |
| Evaluation/benchmark strategy | Section 8 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |
| Persistence/runtime architecture | Section 9 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |
| Multi-agent/orchestration | Section 10 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |
| Engineering feasibility | Section 11 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `synthesis.md` |
| Original/non-obvious insight | Section 12 in all 10 raw files | `comparison.csv`, `comparison_by_dimension.csv`, `summary.md`, `synthesis.md` |

## Model-Specific Evidence Notes

| Model family | Raw file | Evidence carried into analysis |
| --- | --- | --- |
| OpenAI GPT | `raw_outputs/openai_gpt54_mini.md` | Separation of intent, evidence, world state, policy, and execution appears in sections 1, 5, 7, and 12. This supports the control-plane framing in `summary.md` and the service split in `synthesis.md`. |
| Claude | `raw_outputs/anthropic_claude_sonnet_46.md` | Uncertainty as a runtime object appears in sections 3, 7, and 12. This supports escalation thresholds and interrupt handling in the synthesis. |
| Gemini | `raw_outputs/google_gemini_31_pro.md` | Cold-storage tiering and asynchronous OODA flow appear in sections 2, 3, and 9. This supports the disagreement note about forgetting versus audit retention. |
| Grok | `raw_outputs/xai_grok_43.md` | Capability leases, verifier outcomes, and governance precedent appear in sections 5, 7, 9, and 12. This supports the policy-and-lease engine in `synthesis.md`. |
| DeepSeek | `raw_outputs/deepseek_v32.md` | Cost-aware checks and efficient routing appear in sections 3, 4, 8, and 12. This supports the decision to keep first implementations cheap enough to run on every action. |
| Qwen | `raw_outputs/qwen37_max.md` | Context-as-cache, constrained planning search, and stronger causal modeling appear in sections 2, 3, 6, and 12. This supports both the headline combined insight and the causal-world-model disagreement. |
| Llama | `raw_outputs/meta_llama4_maverick.md` | Open/local reproducibility appears in sections 4, 7, 8, 9, and 12. This supports the local-first and inspectable implementation stance. |
| Mistral | `raw_outputs/mistral_large_2512.md` | Sovereignty, privacy, and smaller governed routine models appear in sections 2, 4, 7, 8, and 12. This supports the privacy/local-first constraints in the summary. |
| Perplexity Sonar | `raw_outputs/perplexity_sonar_pro.md` | Freshness, provenance, source reconciliation, and contradiction detection appear in sections 2, 3, 6, 7, 8, and 12. This supports typed source quality in memory and retrieval. |
| Cohere Command | `raw_outputs/cohere_command_a.md` | Side-effect ledger, causal graphs, HTN/MCTS-style planning, and transaction-style auditing appear in sections 3, 5, 6, 9, and 12. This supports the action ledger and implementation appendix. |

## Synthesis Traceability

| Synthesis component | Primary raw evidence | Existing repository surface |
| --- | --- | --- |
| Event ledger | OpenAI section 2, Grok section 9, Qwen section 9, Cohere section 9 | `core/runtime/event_journal.py`, `core/runtime/evidence_ledger.py` |
| State store | OpenAI section 9, Claude section 9, Gemini section 9 | `core/runtime/state_store.py`, `modules/state/` |
| Memory manager | All section 2 outputs | `modules/memory/` |
| Planner | OpenAI section 3, Claude section 3, Qwen section 3, Cohere section 3 | `core/orchestration/planner_runtime.py`, `planner/` |
| Policy and lease engine | Grok sections 5 and 7, Qwen sections 5 and 7, Mistral section 7 | `modules/governance/`, `core/orchestration/governance_runtime.py` |
| Verifier gate | OpenAI section 5, Claude section 5, Grok section 5, Cohere section 5 | `core/orchestration/verifier_runtime.py` |
| Action executor | Gemini section 5, Qwen section 5, Cohere section 5 | `core/orchestration/execution_control.py`, `tools/managed_vm/` |
| Reflection and learning | DeepSeek section 4, Mistral section 4, Qwen section 4 | `core/runtime/failure_learning.py`, `core/runtime/end_to_end_learning.py` |
