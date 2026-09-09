# Validation scope

The packet contains research and an isolated experiment; it changes no runtime implementation or adapter dependency.

- The original repository Linux CI at baseline `e20d2ff4d5c84d4c11c87218c4ae9a04ab0046ca` passed the layout check and all **10 public smoke tests** on both Python3.10 and Python3.11: [run34343696338](https://github.com/zbzbdzb/Cognitive-OS/actions/runs/34343696338). The validation branch adds only manual workflow dispatch; it does not change the tested runtime.
- Windows ran the repository layout check successfully. Public smoke-test collection on Windows failed because existing code imports the Unix-only `resource` module. This result is retained as an environment limitation, not counted as a passing run.
- The recorded Windows process-crash experiment completed **14 cases**. All7 reconciliation cases satisfy the declared final-content/status invariants and repeat-recovery invariant. The simple baseline passes4/7 and exhibits the three expected failure cases described in synthesis.md. Full results are in `recovery-results-windows.json`.
- `verify_packet.py` checks eight unique model records, exact requests/raw outputs, stop conditions, model-file hash records, all88 dimension rows with actual quoted-source lines, the recorded experiment and packet file checksums. These checks do not independently authenticate execution or prove the analyst's judgments.
- The optional local `collection/verify_tokenizers.py` check passed for all8 models: each pinned local tokenizer reproduces its recorded rendered prompt, input IDs and decoded output IDs. Local cached LFS checksums were compared with recorded weight hashes where metadata exists. Results are preserved in `collection/tokenizer-verification.json`; this performs no generation and does not replace the recorded outputs.
- Serious-error Python lint (`E9,F63,F7,F82`) passed on the packet scripts.

Before submission, the candidate packet is additionally checked in the fork on Linux using the original public tests and the packet/experiment commands. The PR's validation section links the completed candidate run. No claim is made that every repository test, all supported platforms, production recovery or AGI capabilities have been validated.
