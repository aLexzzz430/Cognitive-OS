from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = Path(__file__).resolve().parent
TASKS_DIR = EXTERNAL_DIR / "tasks"
TRANSCRIPTS_DIR = EXTERNAL_DIR / "raw_transcripts"
REPORTS_DIR = EXTERNAL_DIR / "reports"
DIFFS_DIR = EXTERNAL_DIR / "diffs"
VERIFIER_LOG_DIR = EXTERNAL_DIR / "verifier_logs"
SUITE_CONFIG_PATH = REPO_ROOT / "experiments" / "phase1c_suite" / "suite_config.json"

FORBIDDEN_METADATA_KEYS = {
    "true_bug_file",
    "traceback_file",
    "direct_unit_test_file",
    "discriminating_test_files",
    "decoy_files",
    "expected_patch",
    "posterior_summary",
    "target_binding",
    "patch_proposal",
    "source_hypothesis_id",
}

IGNORED_COPY_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    ".DS_Store",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _copy_repo_clean(src: Path, dst: Path) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    copied = 0

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in IGNORED_COPY_NAMES}

    shutil.copytree(src, dst, ignore=ignore)
    for path in dst.rglob("*"):
        if path.is_file():
            copied += 1
    return copied


def _iter_repo_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_COPY_NAMES for part in path.relative_to(root).parts):
            continue
        files.append(path.relative_to(root))
    return sorted(files)


def _read_text_lossy(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        return [f"<binary file: {path.name}>\n"]


def _repo_diff(before_root: Path, after_root: Path) -> tuple[str, list[str]]:
    before_files = set(_iter_repo_files(before_root)) if before_root.exists() else set()
    after_files = set(_iter_repo_files(after_root)) if after_root.exists() else set()
    changed_paths: list[str] = []
    chunks: list[str] = []
    for rel in sorted(before_files | after_files):
        before_path = before_root / rel
        after_path = after_root / rel
        before_lines = _read_text_lossy(before_path) if before_path.exists() else []
        after_lines = _read_text_lossy(after_path) if after_path.exists() else []
        if before_lines == after_lines:
            continue
        rel_s = str(rel)
        changed_paths.append(rel_s)
        chunks.extend(difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{rel_s}",
            tofile=f"b/{rel_s}",
            lineterm="",
        ))
    return "\n".join(chunks) + ("\n" if chunks else ""), changed_paths


def _diff_summary(changed_paths: list[str], diff_text: str) -> dict[str, Any]:
    additions = 0
    deletions = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return {
        "changed_paths": changed_paths,
        "line_additions": additions,
        "line_deletions": deletions,
    }


def _tests_modified(changed_paths: list[str]) -> bool:
    return any(path == "tests" or path.startswith("tests/") or "/tests/" in path for path in changed_paths)


def _parse_transcript_commands(transcript: str) -> list[str]:
    commands: list[str] = []
    for line in transcript.splitlines():
        if "/bin/zsh -lc " not in line or " in /" not in line:
            continue
        command = line.split("/bin/zsh -lc ", 1)[1].rsplit(" in /", 1)[0].strip()
        if command and command not in commands:
            commands.append(command)
    return commands


def _parse_transcript_files_read(transcript: str) -> list[str]:
    files: list[str] = []
    for command in _parse_transcript_commands(transcript):
        cleaned = command.strip().strip("'\"")
        try:
            tokens = shlex.split(cleaned)
        except ValueError:
            tokens = cleaned.replace('"', " ").replace("'", " ").split()
        for token in tokens:
            path = token.strip().strip("'\"")
            if path.endswith((".py", ".toml", ".md", ".json")) and path not in files:
                files.append(path)
    return files


def _parse_transcript_refusal(transcript: str, changed_paths: list[str], final_pytest_passed: bool) -> tuple[bool, str]:
    lowered = transcript.lower()
    if "needs_human_review=true" not in lowered or changed_paths or final_pytest_passed:
        return False, ""
    if "ambiguous" in lowered or "conflict" in lowered or "contradict" in lowered:
        return True, "ambiguous_spec"
    if "insufficient" in lowered:
        return True, "evidence_insufficient"
    return True, "needs_human_review"


def _run_verifier(task_dir: Path, agent_name: str, fixture_id: str, timeout_seconds: int) -> dict[str, Any]:
    VERIFIER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = VERIFIER_LOG_DIR / agent_name / f"{fixture_id}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=task_dir / "repo",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "command": [sys.executable, "-m", "pytest", "-q"],
        "cwd": str(task_dir / "repo"),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "log_path": str(log_path),
    }


def _task_markdown() -> str:
    return """# External Coding Agent Baseline Task

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
"""


def _verifier_json() -> dict[str, Any]:
    return {
        "schema_version": "conos.external_baseline.verifier/v1",
        "working_directory": "repo",
        "command": ["python", "-m", "pytest", "-q"],
        "protected_paths": ["tests/"],
        "tests_may_be_modified": False,
        "success_requires": {
            "full_pytest_passed": True,
            "tests_modified": False,
        },
    }


def _external_result_schema() -> dict[str, Any]:
    return {
        "schema_version": "conos.external_baseline.result_schema/v1",
        "required_fields": [
            "agent_name",
            "fixture_id",
            "max_turns_or_time_budget",
            "commands_run",
            "files_read",
            "changed_paths",
            "tests_modified",
            "final_pytest_passed",
            "final_diff_summary",
            "raw_transcript_path",
            "normalized_report_path",
        ],
        "optional_fields": [
            "wrong_patch_attempt_count",
            "rollback_count",
            "needs_human_review",
            "refusal_reason",
            "verification_waste_estimate",
        ],
    }


def _suite_fixtures() -> list[dict[str, Any]]:
    config = _load_json(SUITE_CONFIG_PATH)
    return [dict(row) for row in config.get("fixtures", [])]


def _configured_answer_strings(fixtures: Iterable[Mapping[str, Any]]) -> set[str]:
    values: set[str] = set()
    for fixture in fixtures:
        for key in ("true_bug_file", "traceback_file", "direct_unit_test_file"):
            value = str(fixture.get(key) or "").strip()
            if value:
                values.add(value)
        for key in ("discriminating_test_files", "decoy_files"):
            for item in fixture.get(key) or []:
                value = str(item or "").strip()
                if value:
                    values.add(value)
    return values


def _scan_non_repo_files_for_leaks(task_dir: Path, answer_strings: set[str]) -> dict[str, Any]:
    checked: list[str] = []
    findings: list[dict[str, str]] = []
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(task_dir)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "repo":
            continue
        checked.append(str(rel))
        text = path.read_text(encoding="utf-8", errors="ignore")
        for key in sorted(FORBIDDEN_METADATA_KEYS):
            if key in text:
                findings.append({"path": str(rel), "leak": key})
        for value in sorted(answer_strings):
            if value and value in text:
                findings.append({"path": str(rel), "leak": value})
    return {
        "checked_non_repo_files": checked,
        "finding_count": len(findings),
        "findings": findings,
        "passed": len(findings) == 0,
    }


def generate_task_packages(fixture_ids: set[str] | None = None) -> dict[str, Any]:
    fixtures = _suite_fixtures()
    answer_strings = _configured_answer_strings(fixtures)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    leak_findings: list[dict[str, Any]] = []

    for fixture in fixtures:
        fixture_id = str(fixture["fixture_id"])
        if fixture_ids and fixture_id not in fixture_ids:
            continue
        fixture_root = REPO_ROOT / str(fixture["fixture_root"])
        if not fixture_root.exists():
            raise FileNotFoundError(f"missing fixture root: {fixture_root}")

        task_dir = TASKS_DIR / fixture_id
        if task_dir.exists():
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        copied_files = _copy_repo_clean(fixture_root, task_dir / "repo")
        (task_dir / "TASK.md").write_text(_task_markdown(), encoding="utf-8")
        _write_json(task_dir / "verifier.json", _verifier_json())

        leak_check = _scan_non_repo_files_for_leaks(task_dir, answer_strings)
        if not leak_check["passed"]:
            leak_findings.append({"fixture_id": fixture_id, **leak_check})
        generated.append({
            "fixture_id": fixture_id,
            "task_dir": str(task_dir),
            "repo_dir": str(task_dir / "repo"),
            "copied_file_count": copied_files,
            "non_repo_leak_check": leak_check,
        })

    manifest = {
        "schema_version": "conos.external_baseline.package_manifest/v1",
        "generated_at": _utc_now(),
        "suite_config": str(SUITE_CONFIG_PATH),
        "package_count": len(generated),
        "packages": generated,
        "leak_prevention": {
            "non_repo_files_scanned": True,
            "repo_files_excluded_from_leak_scan_reason": "source tree naturally contains its own filenames",
            "forbidden_metadata_keys": sorted(FORBIDDEN_METADATA_KEYS),
            "leak_finding_count": sum(row["non_repo_leak_check"]["finding_count"] for row in generated),
            "leak_findings": leak_findings,
        },
    }
    _write_json(EXTERNAL_DIR / "external_result_schema.json", _external_result_schema())
    _write_json(EXTERNAL_DIR / "package_manifest.json", manifest)
    return manifest


def _run_command_adapter(agent_command: str, fixture_ids: set[str] | None, agent_name: str, timeout_seconds: int) -> dict[str, Any]:
    manifest = generate_task_packages(fixture_ids)
    fixtures_by_id = {str(row["fixture_id"]): row for row in _suite_fixtures()}
    argv = shlex.split(agent_command)
    if not argv:
        raise ValueError("--agent-command cannot be empty in command-adapter mode")

    results: list[dict[str, Any]] = []
    for package in manifest["packages"]:
        fixture_id = package["fixture_id"]
        task_dir = Path(package["task_dir"])
        transcript_dir = TRANSCRIPTS_DIR / agent_name
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript_dir / f"{fixture_id}.txt"
        env = dict(os.environ)
        env.update({
            "CONOS_EXTERNAL_TASK_DIR": str(task_dir),
            "CONOS_EXTERNAL_REPO_DIR": str(task_dir / "repo"),
            "CONOS_EXTERNAL_FIXTURE_ID": fixture_id,
        })
        completed = subprocess.run(
            argv,
            cwd=task_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        transcript_path.write_text(completed.stdout, encoding="utf-8")
        transcript_text = completed.stdout
        fixture_root = REPO_ROOT / str(fixtures_by_id[fixture_id]["fixture_root"])
        diff_text, changed_paths = _repo_diff(fixture_root, task_dir / "repo")
        diff_path = DIFFS_DIR / agent_name / f"{fixture_id}.diff"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(diff_text, encoding="utf-8")
        verifier = _run_verifier(task_dir, agent_name, fixture_id, min(timeout_seconds, 180))
        normalized_path = REPORTS_DIR / f"{agent_name}_{fixture_id}_normalized.json"
        needs_human_review, refusal_reason = _parse_transcript_refusal(transcript_text, changed_paths, bool(verifier["passed"]))
        normalized_report = {
            "schema_version": "conos.external_baseline.normalized_report/v1",
            "normalized_at": _utc_now(),
            "agent_name": agent_name,
            "fixture_id": fixture_id,
            "max_turns_or_time_budget": f"timeout_seconds={timeout_seconds}",
            "commands_run": _parse_transcript_commands(transcript_text) or argv,
            "files_read": _parse_transcript_files_read(transcript_text),
            "changed_paths": changed_paths,
            "tests_modified": _tests_modified(changed_paths),
            "final_pytest_passed": bool(verifier["passed"]),
            "wrong_patch_attempt_count": 0,
            "rollback_count": 0,
            "final_diff_summary": _diff_summary(changed_paths, diff_text),
            "raw_transcript_path": str(transcript_path),
            "normalized_report_path": str(normalized_path),
            "needs_human_review": needs_human_review,
            "refusal_reason": refusal_reason,
            "verification_waste_estimate": 0,
            "verifier": verifier,
            "command_adapter_returncode": completed.returncode,
            "diff_path": str(diff_path),
        }
        _write_json(normalized_path, normalized_report)
        results.append({
            "fixture_id": fixture_id,
            "agent_name": agent_name,
            "returncode": completed.returncode,
            "raw_transcript_path": str(transcript_path),
            "diff_path": str(diff_path),
            "normalized_report_path": str(normalized_path),
            "final_pytest_passed": bool(verifier["passed"]),
            "changed_paths": changed_paths,
            "tests_modified": _tests_modified(changed_paths),
            "task_dir": str(task_dir),
        })
    payload = {
        "schema_version": "conos.external_baseline.command_adapter_run/v1",
        "generated_at": _utc_now(),
        "agent_name": agent_name,
        "agent_command": argv,
        "results": results,
    }
    _write_json(REPORTS_DIR / f"{agent_name}_command_adapter_run.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and optionally run true external coding-agent baseline packages.")
    parser.add_argument("--mode", choices=["package-only", "command-adapter"], default="package-only")
    parser.add_argument("--fixture-id", action="append", default=[], help="Limit to one or more fixture ids.")
    parser.add_argument("--agent-name", default="external_agent")
    parser.add_argument("--agent-command", default="", help="Command to run in each task package for command-adapter mode.")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)

    fixture_ids = set(args.fixture_id) if args.fixture_id else None
    if args.mode == "package-only":
        payload = generate_task_packages(fixture_ids)
    else:
        if not args.agent_command:
            parser.error("--agent-command is required for command-adapter mode")
        payload = _run_command_adapter(args.agent_command, fixture_ids, args.agent_name, args.timeout_seconds)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
