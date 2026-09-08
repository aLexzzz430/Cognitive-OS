# AI-generated architecture proposals for Cognitive-OS

This packet compares actual generated proposals with their omissions and failure modes. It is intended to inform an implementable next Cognitive-OS iteration, not establish that the project or any model achieves AGI.

**Coverage: nine systems, with proposal prose from eight.** Eight compact local systems received one identical prompt each. Seven produced proposal prose of varying completeness; Danube3 echoed the prompt and is explicitly recorded as a failed proposal request. An exploratory Gemini Flash web conversation supplied two A/B candidate proposals from one additional system. The hosted reference is separated from the primary local cohort throughout the comparison.

## Headline findings

1. **Repeated terminology is not independent agreement or an implementation contract.** Several local responses restate requested features; TinyLlama repeats sections and Danube3 supplies no proposal. The prompt itself requested memory categories, bounded control and offline learning, so recurrence of those words does not establish independent discovery or consensus.
2. **Concrete-looking recovery pseudocode can fail at a precise boundary.** An executed local example rejects the transaction ordering in hosted candidate A: a separate file effect remains after abrupt exit, while its uncommitted pending-action record disappears. Committing intent before the effect preserves the record for this example's reconciliation probe. This is a test of generated pseudocode, not an upstream bug or a general exactly-once guarantee.
3. **A useful minimum design needs observable task verification and measured tradeoffs.** The synthesis combines a single controller, evidence-linked state, durable intent, effect-specific reconciliation and offline candidate evaluation. It maps these proposals to existing source surfaces and proposes a documentation-edit slice before adding new stores or distributed coordination.

## Read the packet

- [Exact prompt and method](prompts.md), [protocol and amendments](collection/protocol.md), [collection incidents](collection/collection-incidents.md), and [sources/editing record](sources.md).
- [Raw outputs](raw_outputs/) and [comparison.csv](comparison.csv): 99 rows, one per system and requested dimension, comprising 88 local rows plus 11 hosted-reference rows. The `cohort` field identifies the difference.
- [Patterns and disagreements](summary.md) and [combined architecture](synthesis.md), including concrete interfaces, recovery pseudocode, proposed experiments and a six-week estimate.
- [Hosted A/B analysis](collection/gemini-flash-reference/analysis-notes.md) and its [22-row candidate comparison](collection/gemini-flash-reference/comparison.csv). Two candidates from one prompt are not two independent systems.
- [Executed example](collection/transaction-example.md), with code, original results and strict limits.
- [Repository context](collection/repository-context.md): source observations at the pinned upstream commit, distinct from generated proposals.

## Collected systems

All local responses were collected on 2026-09-08. Exact timestamps and source revisions are in [sources.md](sources.md) and the per-model records.

| System | Cohort | Completion tokens | Finish | Observed result |
| --- | --- | ---: | --- | --- |
| SmolLM2 1.7B Instruct | Local | 1,778 | stop | All topic headings, mostly restatement; concrete final example absent. |
| Qwen2.5 1.5B Instruct | Local | 2,400 | length | Repetitive prose; final paragraph cut off and example absent. |
| Granite 3.3 2B Instruct | Local | 1,168 | stop | Specific components and example; missing contracts and an unsubstantiated distributed-stack tradeoff. |
| TinyLlama 1.1B Chat v1.0 | Local | 1,193 | stop | Repeated sections; missing memory distinctions, orchestration decision and actual example. |
| Phi-3 mini 4k Instruct | Local | 1,384 | stop | Clearer baseline measures; unsupported receipt recovery and no enumerated six-week sequence. |
| Qwen3 1.7B | Local | 1,867 | stop | Useful incomplete-outcome handling and task/context memory idea; unjustified retention periods and incomplete contracts. |
| Falcon3 1B Instruct | Local | 1,184 | stop | Broad architecture prose and coordination themes; missing concrete interfaces and ablation. |
| H2O Danube3 500m Chat | Local | 574 | stop | Prompt echo, not an architecture proposal. Preserved as a failed request. |
| Gemini web, Flash mode | Exploratory hosted | Unknown | Not exposed | Two A/B proposals from one prompt; more concrete contracts, including one reproduced transaction-ordering flaw. Exact backend revision unknown. |

A `stop` result describes generation termination, not successful instruction following. Missing mechanisms remain missing in the comparison. None of these observations estimates a system's best possible performance. The hosted reference's unknown settings, selection after two local responses, and different compute/service conditions prevent a parameter-matched or causal comparison.

## Collection and attribution

Local generation used llama.cpp b10809 / commit `5266f24da`, two CPU threads, no GPU offload, a 4,096-token context, temperature 0.2, top-p 0.9, seed 20260908 and at most 2,400 generated tokens. Public model-specific GGUF chat templates remained in use. There was no separate system message, history, retrieval or tool execution. The eight local systems span seven named project families; Qwen2.5 and Qwen3 are related generations. Model size, training, quantization and template differences limit generalization and independent-consensus claims.

For every local system, `collection/<alias>/` preserves public model metadata, pinned weight source/hash, exact request and response, timestamps, usage, finish reason and raw-content hash. The local raw file is the exact response content string. Repetitions, omissions, truncation and the echo are not cleaned away.

Gemini's two rendered answer strings remain unedited in separate collection files. Its combined raw-output view adds only explicitly collector-authored candidate labels and separator newlines. Both original and combined hashes are recorded. Service instructions, sampling, token counts and backend revision are unavailable. No private account screenshots, credentials or hidden prompts are included.

The comparison, report and scripts were authored by the assisting Codex agent; no independent human review of the model proposals was performed. Analyst additions are not counted as another independent model response.

## Validation and replay

From the packet root:

```sh
python collection/verify.py
python collection/transaction_example.py
```

The verifier checks exact recorded content, hashes, pinned source identities, echo classification, all 99 comparison pairs and evidence-anchor bounds, plus the hosted A/B assembly. It checks consistency and structural coverage, not provider-signed authorship, research quality or architectural effectiveness. Review the raw outputs and analysis when assessing those claims.

The transaction example uses temporary local files and SQLite. Its two assertions passed: the one-transaction ordering leaves an effect without a pending record; precommitting intent preserves the record and permits that effect to be reconciled without another append. It models abrupt process exit only. No power-loss, third-party-effect or end-to-end Cognitive-OS recovery test was performed. The three broader experiments in the synthesis remain proposed and unrun.

The maintainer-required repository layout check and all ten public smoke tests passed on the unchanged upstream runtime with the four-response draft staged. Later changes are confined to this research packet. Those checks do not validate the proposed architecture. The packet adds no public runtime/API change or adapter dependency.

Optional replay uses a separately supplied b10809 binary and downloads the chosen pinned model into a new directory:

```sh
python collection/replay.py smollm2 --server /path/to/llama-server --out /tmp/smollm2-replay
```

Replay's help entry point was checked, but a second full generation was not run. Fixed seeds do not guarantee bitwise identity across hardware/builds. Replay preserves the original records. Model weights and the local runtime binary are not redistributed in the packet.
