# Local model architecture proposals: an auditable constraint study

This packet addresses [issue #5](https://github.com/aLexzzz430/Cognitive-OS/issues/5) by collecting actual proposals from eight named open-weight instruction/chat models on one 12 GB workstation. It studies what compact local systems propose under Cognitive-OS constraints; it is **not a comparison of eight frontier services, eight independent lineages, or demonstrated AGI systems**.

The useful result is a set of implementation decisions grounded in the proposals' omissions and contradictions. A section labelled “recovery” does not define recoverable file effects; a section labelled “governance” does not define authority. Several models repeat the prompt's SQLite/verifier concepts, some introduce incompatible cloud or distributed infrastructure, and TinyLlama invents empirical performance numbers. Those claims remain in the raw evidence and are explicitly rejected in the analysis.

## Read the packet

- [prompts.md](prompts.md): exact common prompt, serialization and generation protocol.
- [sources.md](sources.md): pinned models, attribution, dates, licenses, local environment and edits.
- [raw_outputs/](raw_outputs/): unedited decoded generations, including termination tokens.
- [comparison.csv](comparison.csv): eight systems × eleven dimensions, with a verbatim quote, source line and analyst assessment in every row.
- [summary.md](summary.md): agreements, disagreements, failures and limits of inference.
- [synthesis.md](synthesis.md): a bounded local architecture and integration decisions tied to the current repository.
- [collection/runs/](collection/runs/): exact requests, rendered prompts, input/output token IDs, generation settings, timestamps, hashes and runtime versions.
- [recovery_experiment.py](recovery_experiment.py): an isolated, executable process-crash example; it does not exercise the actual Cognitive-OS runtime.

## Method and evidential limits

Collection uses one frozen user prompt per model, its own chat template, greedy decoding, BF16, eager attention, batch one and a 1,400-new-token cap. The requested length is at most 850 words, but equal token budgets do not imply equal word budgets. A length-capped answer is retained rather than extended or replaced. Completed model outputs are never regenerated to choose a better answer. Failed attempts are documented in [collection/incidents.md](collection/incidents.md).

The sample is selected for public access, local hardware and license compatibility. Its small size, shared architecture ancestry, one prompt, one output per model and lack of blinding do not support statistical rankings or broad claims about model families. “SQLite consensus” is especially confounded: SQLite, journaling, governed tools and verifier-gated patches were explicitly supplied in the prompt. Instruction-following failures are observations for these exact runs, not universal model properties.

The collection records support inspection and local reproduction, not independent proof from a model provider. Hashes establish correspondence between recorded files; they do not establish the truth of a model's assertions or prove that another device will emit identical tokens. No hidden prompts, account tokens, paid-service screenshots or model weights are included. Research analysis and scripts are separate from the raw generations attributed to the named models.

## Recheck

From the repository root:

```bash
python research/ai_generated_agi_architectures/verify_packet.py
python research/ai_generated_agi_architectures/recovery_experiment.py
python scripts/check_conos_repo_layout.py
pytest -q tests/test_public_repo_smoke.py
```

The repository's public smoke tests require a Unix environment because existing runtime code imports `resource`. The required Linux checks are recorded separately from the isolated experiment. The packet makes no claim that all repository tests, production crash recovery or AGI capabilities have been validated. See [validation.md](validation.md) for exact completed checks and their scope.
