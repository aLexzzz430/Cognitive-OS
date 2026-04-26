# True External Agent Baseline Protocol

This harness evaluates real external coding agents without giving them Con OS
internal traces, fixture metadata, true bug files, or expected patches.

## Task Package

Each generated task package lives at:

```text
experiments/external_baseline/tasks/<fixture_id>/
```

The package contains:

- `repo/`: a clean copy of the fixture repository.
- `TASK.md`: the natural-language task and rules.
- `verifier.json`: the verifier command and protected-test rule.

The package intentionally does not include:

- true bug file metadata
- traceback-file metadata from the Con OS suite
- internal Con OS reports
- expected patch content
- target-binding or posterior traces

The shared result schema lives outside individual task packages at:

```text
experiments/external_baseline/external_result_schema.json
```

## External Agent Rules

Run the external agent inside the task package. The agent may inspect files,
run tests, and make a minimal source-code patch. It must not modify tests unless
the task prompt explicitly permits that. In the Phase 1J suite, tests are
protected by default.

The verifier is always:

```text
python -m pytest -q
```

Run it from the package `repo/` directory after the patch.

## Result Import Modes

The harness supports three non-vendor-specific result paths:

1. Manual transcript import:
   save the agent transcript and optional diff, then normalize it with
   `normalize_external_report.py`.
2. Command adapter:
   run a local command in each generated task package and capture stdout/stderr
   as a raw transcript.
3. JSON report import:
   provide a JSON report produced by any external agent wrapper and normalize it
   to the shared schema.

For final scoring, use `normalize_external_report.py --run-verifier --task-dir`
to rerun the same pytest verifier inside the task package. The normalized
`final_pytest_passed` field is then derived from that verifier result rather
than from the external agent's self-report.

## Required Normalized Fields

External reports are normalized into:

- `agent_name`
- `fixture_id`
- `max_turns_or_time_budget`
- `commands_run`
- `files_read`
- `changed_paths`
- `tests_modified`
- `final_pytest_passed`
- `wrong_patch_attempt_count`
- `rollback_count`
- `final_diff_summary`
- `raw_transcript_path`
- `normalized_report_path`

Optional fields such as `needs_human_review`, `refusal_reason`, and
`verification_waste_estimate` improve ambiguous-fixture analysis when available.

## Leak Prevention

The package generator scans non-repository package files for forbidden metadata
keys and configured answer paths. The repository copy itself is not scanned for
answer filenames because the source tree naturally contains its own files.

The comparison analyzer may use suite metadata on the evaluator side to judge
whether an external agent patched a true bug file or a traceback/surface file.
That metadata is never written into task prompts.
