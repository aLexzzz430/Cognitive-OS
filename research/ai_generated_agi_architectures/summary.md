# Patterns, disagreements and useful ideas

The dataset contains eight primary local responses and one exploratory hosted system with two A/B candidates. Danube3 returned a prompt echo, not an architecture proposal. Thus seven local systems and one hosted system supplied proposal prose, ranging from requirement restatement to concrete but imperfect workflows. All results, including failures and truncation, remain visible.

## A heading is not a mechanism

SmolLM2 and Qwen2.5 largely rephrase the requested topics. TinyLlama's memory section describes a control loop; it repeats evaluation/persistence sections and names an example without supplying one. Danube3 reproduces the prompt. Filling those gaps in a cleaned summary would misrepresent the evidence. The comparison records omissions instead. Sources: [SmolLM2](collection/smollm2/analysis-notes.md), [Qwen2.5](collection/qwen25/analysis-notes.md), [TinyLlama](collection/tinyllama/analysis-notes.md), [Danube3](collection/danube3/analysis-notes.md).

Granite supplies identifiable technologies and a report-edit scenario. Its graph store, transaction manager, replicated log, queue and consensus algorithm create a distinct proposal, but it gives neither the transaction participants nor a workload requiring multiple coordination layers in the minimum local runtime. Specificity makes the gaps inspectable; it does not establish benefit. [Granite notes](collection/granite33/analysis-notes.md).

Phi-3 offers clearer same-task time/resource comparisons and candidate-learning prose, but claims that flushing a log will recover a missing receipt without supplying a durable source or reconstruction mechanism. Qwen3 leaves such completion explicitly unacknowledged and proposes task/context-based memory, while giving arbitrary retention periods and unclear training/held-out terminology. Their differences are descriptive, not controlled causal findings. [Phi-3](collection/phi3/analysis-notes.md), [Qwen3](collection/qwen3/analysis-notes.md).

Falcon3 emphasizes probabilistic reasoning, recovery and coordination overhead, but provides few interfaces, no ablation and an unsupported interrupted-receipt assurance. [Falcon3 notes](collection/falcon3/analysis-notes.md).

## Eleven-dimension synthesis of the differences

| Dimension | Observed pattern or disagreement | Implementation implication |
| --- | --- | --- |
| Memory | Most responsive systems repeat the prompted categories. Granite chooses cache/SQLite/graph/modules; Qwen3 gives retention periods and a context hierarchy; hosted answers add schemas and source IDs. | Specify retrieval results and evidence retention. Evaluate a task/family index against existing failure retrieval before adding stores or unvalidated expiry defaults. |
| Planning | Bounded loops recur. Hosted answers specify step/depth values and recovery states; local outputs often omit actual transition contracts. | Distinguish depth, total steps and replans. Define a task success predicate; a suggested number is an unvalidated configuration choice. |
| Learning | Versions, evaluation and rollback recur, partly because the prompt explicitly requested them. Phi-3/hosted prose describes candidates; Qwen3 ambiguously trains on held-out data. | Separate training, tuning and final evaluation cases. Common prompted words do not demonstrate independent discovery or measured learning. |
| Tools | Several local answers turn uncertainty into confidence values. Hosted answers add request/receipt fields and observation before retry. | A score, request ID or checksum alone is not effect reconciliation. Preserve durable action identity and use an action-specific probe. |
| World representation | Categories are common; concrete evidence schemas and contradiction rules are less common. Qwen3 allows deferral; hosted B zeroes confidence on contradictory observation. | Retain conflicting evidence and assess its reliability. Do not automatically discard a valid claim when a new observation may be mistaken. |
| Governance | User control, permission, budgets and stopping are broadly repeated. Hosted candidates declare more concrete tiers/counters. | These are proposed interfaces, not validated enforcement. No cybersecurity experiment or security guarantee is established by this packet. |
| Evaluation | Granite/Phi-3 give identifiable time/resource/ablation ideas; Qwen3 selects learning ablation; Falcon3 gives themes without an ablation; other local outputs provide little or no experiment content. Hosted answers specify more detail. | Freeze tasks, verifier and budgets. Avoid an unbounded-context baseline, undefined significance or invented achieved metrics. |
| Persistence | Granite names distributed mechanisms; Phi-3 assumes missing receipts appear on flush; Falcon3 asserts interrupted completions remain logged; Qwen3 explicitly leaves missing-receipt work incomplete. Hosted A specifies a flawed order and B assumes a logged pending invocation. | Test the exact boundary. Commit intent before dispatch and reconcile observable effects; retain uncertainty when observation cannot decide. |
| Orchestration | Granite combines a controller with distributed coordination; hosted answers use bounded workers; Phi-3/Qwen3 vary workers with complexity; Falcon3 names a controller/worker pattern. | Begin with the existing single-loop baseline. State worker contracts and conflict rules; ordering messages is not resolving contradictory content. |
| Feasibility | Granite gives three broad two-week phases; hosted answers give weekly plans. Phi-3/Qwen3/Falcon3 name but do not enumerate a six-week sequence. | Map proposals to code already present and choose one testable task/effect slice. Treat schedules as estimates, not evidence of capability. |
| Originality | Pluggability, shadow staging, explicit state transitions and task/context memory are useful choices with uneven rejection criteria. Other outputs repeat common concepts or supply no insight. | Do not label common patterns as inventions. Each added mechanism needs a measured reason to keep or reject it. |

The main [comparison.csv](comparison.csv) contains all 99 system/dimension pairs with raw-text anchors and a cohort label. Danube3's rows explicitly describe a prompt echo, not a model-endorsed architecture. The hosted reference has a separate [A/B comparison](collection/gemini-flash-reference/comparison.csv) and [analysis](collection/gemini-flash-reference/analysis-notes.md). It is one system with unknown revision/settings, not two independent samples.

## The one measured research finding

Hosted A places pending intent, a filesystem effect and the receipt before one SQL commit, then proposes recovering through a pending-action scan. In the analyst's local example, the child exits after an append and before receipt/commit. Reopening reveals one file effect and zero pending actions. Moving the intent commit before the append leaves one pending action and the same single effect; a read probe reconciles it without another append. [Code, original results and limits](collection/transaction-example.md).

This rejects one generated transaction-ordering claim at one process-exit boundary. It is not an upstream defect report, a measurement of Cognitive-OS recovery, a power-loss simulation or a general exactly-once proof. The other experiments in the proposals and synthesis were not executed. Detailed pseudocode can still contain a flaw that a narrowly scoped experiment exposes.

## What to retain and what to test

The [combined design](synthesis.md) retains evidence-linked memory, one controller, explicit task verification, durable action identity, effect-specific reconciliation and versioned offline candidates. Its minimum slice is a documentation edit whose source facts and final content can both be checked. It uses existing state/failure-memory/mirror surfaces as the baseline.

Do not adopt distributed coordination merely because it is named, expire needed evidence using unvalidated retention examples, treat an uncertain-effect confidence score as proof, or infer an improvement from adding logs. Those choices require experiments against the current implementation. The supplied six-week estimate starts with a fixed task corpus and baseline, then local recovery, memory applicability and ablations.

The recommendations are analyst additions informed by the full dataset and pinned source inspection. They are not a vote, a claim of independent expert consensus, a model-quality ranking or demonstrated architectural improvement. The primary protocol selected compact ungated models for an 8 GiB CPU-only VM, and uses one sample per system with different sizes/quantizations/templates. The Gemini reference was added after two local responses and remains uncontrolled. These limitations are part of the finding, not reasons to hide weak outputs.
