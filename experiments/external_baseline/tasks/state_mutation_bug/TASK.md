# External Coding Agent Baseline Task

You are given a small Python repository in `repo/`.

## Objective

Investigate why the repository's tests fail, make the smallest source-code
change that fixes the behavior, and verify the full test suite.

## Rules

- Do not modify files under `tests/`.
- Prefer a minimal source patch over broad refactors.
- Do not rewrite project structure.
- Run the full verifier before reporting success.
- If the evidence is insufficient or the expected behavior is ambiguous, do not
  guess. Report `needs_human_review=true` with a short refusal reason.

## Verifier

Run this command from `repo/`:

```text
python -m pytest -q
```

## Output To Preserve

Keep a transcript of:

- commands run
- files inspected
- changed paths
- final test result
- final diff summary
- any rollback or rejected patch attempts
