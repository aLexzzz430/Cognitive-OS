from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXTERNAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EXTERNAL_DIR / "reports"
VERIFIER_LOG_DIR = EXTERNAL_DIR / "verifier_logs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _load_json(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path | None) -> str:
    if not path:
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_commands(transcript: str) -> list[str]:
    commands: list[str] = []
    patterns = [
        re.compile(r"^\s*\$\s+(.+?)\s*$"),
        re.compile(r"^\s*(?:command|cmd)\s*:\s*(.+?)\s*$", re.IGNORECASE),
        re.compile(r"^\s*>\s*(python|pytest|rg|grep|sed|cat|ls|find|git)\b(.+?)\s*$"),
        re.compile(r"^\s*/bin/zsh\s+-lc\s+(.+?)\s+in\s+/.+?$"),
    ]
    for line in transcript.splitlines():
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                command = " ".join(part for part in match.groups() if part).strip()
                if command and command not in commands:
                    commands.append(command)
                break
    return commands


def _parse_files_read(transcript: str) -> list[str]:
    files: list[str] = []
    for command in _parse_commands(transcript):
        cleaned = command.strip().strip("'\"")
        try:
            tokens = shlex.split(cleaned)
        except ValueError:
            tokens = cleaned.replace('"', " ").replace("'", " ").split()
        for token in tokens:
            path = token.strip().strip("'\"")
            if path.endswith((".py", ".toml", ".md", ".json")) and path not in files:
                files.append(path)
    for line in transcript.splitlines():
        for pattern in (
            r"\b(?:cat|sed|less|open|read)\s+([A-Za-z0-9_./-]+\.py)",
            r"\b(?:rg|grep)\b.+?\s+([A-Za-z0-9_./-]+\.py)",
            r"\bfile(?:_read)?\s*[:=]\s*([A-Za-z0-9_./-]+\.py)",
        ):
            for match in re.findall(pattern, line):
                path = str(match).strip()
                if path and path not in files:
                    files.append(path)
    return files


def _parse_changed_paths(diff_text: str) -> list[str]:
    paths: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/").strip()
            if path != "/dev/null" and path not in paths:
                paths.append(path)
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                path = parts[3].removeprefix("b/")
                if path not in paths:
                    paths.append(path)
    return paths


def _tests_modified(paths: list[str]) -> bool:
    return any(path == "tests" or path.startswith("tests/") or "/tests/" in path for path in paths)


def _final_diff_summary(changed_paths: list[str], diff_text: str) -> dict[str, Any]:
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {
        "changed_paths": changed_paths,
        "line_additions": added,
        "line_deletions": removed,
    }


def _parse_refusal(transcript: str, raw: Mapping[str, Any], changed_paths: list[str], final_pytest_passed: bool) -> tuple[bool, str]:
    explicit = re.search(r"\bneeds_human_review\s*=\s*true\b", transcript, re.IGNORECASE)
    if raw.get("needs_human_review") is not None:
        needs_review = bool(raw.get("needs_human_review"))
    else:
        needs_review = bool(explicit) and not changed_paths and not final_pytest_passed
    if explicit and not changed_paths and not final_pytest_passed:
        needs_review = True
    reason = str(raw.get("refusal_reason") or "")
    lowered = transcript.lower()
    if needs_review and not reason:
        if "ambiguous" in lowered or "conflict" in lowered or "contradict" in lowered:
            reason = "ambiguous_spec"
        elif "insufficient" in lowered:
            reason = "evidence_insufficient"
    return needs_review, reason


def _merge_report(raw: Mapping[str, Any], args: argparse.Namespace, transcript: str, diff_text: str) -> dict[str, Any]:
    changed_paths = [str(path) for path in _as_list(raw.get("changed_paths")) if str(path)]
    if not changed_paths and diff_text:
        changed_paths = _parse_changed_paths(diff_text)

    commands = [str(command) for command in _as_list(raw.get("commands_run")) if str(command)]
    if not commands and transcript:
        commands = _parse_commands(transcript)

    files_read = [str(path) for path in _as_list(raw.get("files_read")) if str(path)]
    if not files_read and transcript:
        files_read = _parse_files_read(transcript)

    final_pytest_passed = raw.get("final_pytest_passed")
    if final_pytest_passed is None:
        final_pytest_passed = bool(args.final_pytest_passed)

    tests_modified = raw.get("tests_modified")
    if tests_modified is None:
        tests_modified = _tests_modified(changed_paths)

    final_diff_summary = raw.get("final_diff_summary")
    if not isinstance(final_diff_summary, Mapping):
        final_diff_summary = _final_diff_summary(changed_paths, diff_text)

    verifier_result = _run_verifier(args) if args.run_verifier else {}
    if verifier_result:
        final_pytest_passed = bool(verifier_result.get("passed"))
    needs_human_review, refusal_reason = _parse_refusal(transcript, raw, changed_paths, bool(final_pytest_passed))

    return {
        "schema_version": "conos.external_baseline.normalized_report/v1",
        "normalized_at": _utc_now(),
        "agent_name": str(raw.get("agent_name") or args.agent_name),
        "fixture_id": str(raw.get("fixture_id") or args.fixture_id),
        "max_turns_or_time_budget": str(raw.get("max_turns_or_time_budget") or args.max_turns_or_time_budget or ""),
        "commands_run": commands,
        "files_read": files_read,
        "changed_paths": changed_paths,
        "tests_modified": bool(tests_modified),
        "final_pytest_passed": bool(final_pytest_passed),
        "wrong_patch_attempt_count": int(raw.get("wrong_patch_attempt_count", 0) or 0),
        "rollback_count": int(raw.get("rollback_count", 0) or 0),
        "final_diff_summary": dict(final_diff_summary),
        "raw_transcript_path": str(raw.get("raw_transcript_path") or args.transcript or ""),
        "normalized_report_path": str(args.output or ""),
        "needs_human_review": needs_human_review,
        "refusal_reason": refusal_reason,
        "verification_waste_estimate": int(raw.get("verification_waste_estimate", 0) or 0),
        "verifier": verifier_result,
    }


def _run_verifier(args: argparse.Namespace) -> dict[str, Any]:
    if not args.task_dir:
        raise ValueError("--task-dir is required with --run-verifier")
    task_dir = Path(args.task_dir)
    repo_dir = task_dir / "repo"
    cwd = repo_dir if repo_dir.exists() else task_dir
    argv = shlex.split(args.verifier_command)
    if argv and argv[0] == "python":
        argv[0] = sys.executable
    if not argv:
        raise ValueError("--verifier-command cannot be empty")
    VERIFIER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    fixture = args.fixture_id or "unknown_fixture"
    agent = args.agent_name or "external_agent"
    log_path = VERIFIER_LOG_DIR / f"{agent}_{fixture}_pytest.txt"
    completed = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.verifier_timeout_seconds,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "command": argv,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "log_path": str(log_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize an external coding-agent transcript, diff, or JSON report.")
    parser.add_argument("--input-json", type=Path, help="External agent JSON report to normalize.")
    parser.add_argument("--transcript", type=Path, help="Raw transcript path.")
    parser.add_argument("--diff", type=Path, help="Unified diff path.")
    parser.add_argument("--task-dir", type=Path, help="Task package directory for optional verifier rerun.")
    parser.add_argument("--fixture-id", default="", help="Fixture id when not present in input JSON.")
    parser.add_argument("--agent-name", default="external_agent")
    parser.add_argument("--max-turns-or-time-budget", default="")
    parser.add_argument("--final-pytest-passed", action="store_true")
    parser.add_argument("--run-verifier", action="store_true", help="Run the package verifier and use it as final_pytest_passed.")
    parser.add_argument("--verifier-command", default="python -m pytest -q")
    parser.add_argument("--verifier-timeout-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path, help="Write normalized report to this path.")
    args = parser.parse_args(argv)

    raw = _load_json(args.input_json)
    transcript = _read_text(args.transcript)
    diff_text = _read_text(args.diff)
    if not raw and not args.fixture_id:
        parser.error("--fixture-id is required unless --input-json provides fixture_id")

    report = _merge_report(raw, args, transcript, diff_text)
    if not report["normalized_report_path"]:
        out = REPORTS_DIR / f"{report['agent_name']}_{report['fixture_id']}_normalized.json"
        report["normalized_report_path"] = str(out)
    else:
        out = Path(report["normalized_report_path"])

    if args.output:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
