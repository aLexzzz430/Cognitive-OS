from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.local_machine.patch_proposal import generate_patch_proposals  # noqa: E402
from integrations.local_machine.target_binding import bind_target  # noqa: E402


EXTERNAL_BASELINE_VERSION = "conos.phase1i.external_coding_agent_baseline/v1"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _run_ref(tick: int) -> str:
    return f"run_external_{tick:03d}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _source_files(root: Path) -> list[str]:
    paths: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        if "__pycache__" in Path(relative).parts:
            continue
        paths.append(relative)
    return paths


def _tree_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        if any(part in {".pytest_cache", "__pycache__"} for part in Path(relative).parts):
            continue
        entries.append({"path": relative, "kind": "dir" if path.is_dir() else "file"})
    return entries[:500]


def _run_pytest(root: Path, target: str, run_output_root: Path, tick: int, timeout: int = 30) -> dict[str, Any]:
    normalized_target = str(target or ".")
    command = [sys.executable, "-m", "pytest", "-q", normalized_target]
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        returncode = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "") + "\nTIMEOUT"
    run_ref = _run_ref(tick)
    payload = {
        "run_ref": run_ref,
        "command": command,
        "target": normalized_target,
        "returncode": returncode,
        "success": returncode == 0,
        "stdout": stdout,
        "stderr": stderr,
    }
    _write_json(run_output_root / f"{run_ref}.json", payload)
    return payload


def _patch_target(diff: str) -> str:
    for line in str(diff or "").splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].strip().split("\t", 1)[0]
        if raw.startswith("b/"):
            raw = raw[2:]
        if raw.startswith("a/"):
            raw = raw[2:]
        return "" if raw in {"dev/null", "/dev/null"} else raw
    return ""


def _parse_single_file_diff(diff: str) -> tuple[str, list[str]]:
    target = _patch_target(diff)
    if not target:
        raise ValueError("patch proposal has no target file")
    lines = str(diff or "").splitlines()
    hunks = [line for line in lines if line.startswith("@@")]
    if not hunks:
        raise ValueError("patch proposal has no hunks")
    return target, lines


def _apply_single_file_unified_diff(root: Path, diff: str) -> list[str]:
    target, lines = _parse_single_file_diff(diff)
    path = (root / target).resolve()
    path.relative_to(root.resolve())
    original = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    old_index = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("@@"):
            index += 1
            continue
        match = re.match(r"@@ -(?P<old>\d+)(?:,\d+)? \+(?P<new>\d+)(?:,\d+)? @@", line)
        if not match:
            raise ValueError(f"unsupported hunk header: {line}")
        start = int(match.group("old")) - 1
        output.extend(original[old_index:start])
        old_index = start
        index += 1
        while index < len(lines) and not lines[index].startswith("@@"):
            hunk_line = lines[index]
            if hunk_line.startswith(("--- ", "+++ ")):
                index += 1
                continue
            if not hunk_line:
                output.append("\n")
                old_index += 1
            elif hunk_line[0] == " ":
                output.append(original[old_index])
                old_index += 1
            elif hunk_line[0] == "-":
                old_index += 1
            elif hunk_line[0] == "+":
                output.append(hunk_line[1:] + "\n")
            elif hunk_line.startswith("\\"):
                pass
            else:
                raise ValueError(f"unsupported patch line: {hunk_line}")
            index += 1
    output.extend(original[old_index:])
    path.write_text("".join(output), encoding="utf-8")
    return [target]


def _snapshot(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        if "__pycache__" in Path(relative).parts:
            continue
        result[relative] = path.read_text(encoding="utf-8")
    return result


def _diff_summary(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, Any]:
    changed = sorted(path for path in set(before) | set(after) if before.get(path, "") != after.get(path, ""))
    patch = ""
    for path in changed:
        patch += "".join(
            difflib.unified_diff(
                str(before.get(path, "")).splitlines(keepends=True),
                str(after.get(path, "")).splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            )
        )
    return {
        "changed_paths": changed,
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else "",
        "patch": patch,
    }


def _selected_action(
    *,
    tick: int,
    function_name: str,
    kwargs: Mapping[str, Any] | None = None,
    success: bool = True,
    state: str = "",
) -> dict[str, Any]:
    return {
        "tick": int(tick),
        "function_name": function_name,
        "kwargs": dict(kwargs or {}),
        "executed_function_name": function_name,
        "executed_kwargs": dict(kwargs or {}),
        "success": bool(success),
        "state": state,
        "source": "external_coding_agent_baseline",
        "candidate_meta": {},
        "action_grounding_status": "",
    }


def run_external_baseline(
    *,
    fixture_root: str | Path,
    instruction: str,
    run_id: str,
    max_ticks: int,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    temp = tempfile.TemporaryDirectory(prefix="conos_external_baseline_")
    try:
        temp_root = Path(temp.name)
        source_root = temp_root / "source"
        run_output_root = temp_root / "run_outputs"
        shutil.copytree(
            Path(fixture_root),
            source_root,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
        before = _snapshot(source_root)
        state: dict[str, Any] = {
            "schema_version": EXTERNAL_BASELINE_VERSION,
            "validation_runs": [],
            "last_tree": {"entries": _tree_entries(source_root)},
            "read_files": [],
            "target_binding": {},
        }
        selected_actions: list[dict[str, Any]] = []
        target_binding_by_tick: list[dict[str, Any]] = []
        patch_events: list[dict[str, Any]] = []
        rollback_count = 0
        tick = 1

        selected_actions.append(_selected_action(tick=tick, function_name="repo_tree", kwargs={"path": ".", "depth": 3}, state="TREE_READ"))
        tick += 1
        initial = _run_pytest(source_root, ".", run_output_root, tick)
        state["validation_runs"].append(
            {"run_ref": initial["run_ref"], "target": ".", "success": bool(initial["success"])}
        )
        selected_actions.append(
            _selected_action(
                tick=tick,
                function_name="run_test",
                kwargs={"target": ".", "timeout_seconds": 30},
                success=bool(initial["success"]),
                state="TESTS_PASSED" if initial["success"] else "TESTS_FAILED",
            )
        )
        tick += 1
        if initial["success"]:
            after = _snapshot(source_root)
            report = {
                "schema_version": EXTERNAL_BASELINE_VERSION,
                "variant": "external_coding_agent_baseline",
                "runner": "ordinary coding-agent style deterministic baseline",
                "instruction": instruction,
                "run_id": run_id,
                "success": False,
                "ticks": len(selected_actions),
                "selected_actions": selected_actions,
                "target_binding_by_tick": target_binding_by_tick,
                "patch_proposal_events": patch_events,
                "patch_proposals_generated": 0,
                "patch_proposal_verified": False,
                "patch_proposal_rollback_count": 0,
                "final_tests_passed": True,
                "completed_before_verification": False,
                "final_diff_summary": _diff_summary(before, after),
            }
            if report_path:
                _write_json(Path(report_path), report)
            return report

        selected_actions.append(
            _selected_action(tick=tick, function_name="read_test_failure", kwargs={"run_ref": initial["run_ref"], "max_chars": 6000}, state="TEST_FAILURE_READ")
        )
        tick += 1
        context = {
            "instruction": instruction,
            "source_root": str(source_root),
            "workspace_root": str(source_root),
            "run_output_root": str(run_output_root),
            "investigation_state": state,
        }
        binding = bind_target(context)
        state["target_binding"] = dict(binding)
        target_binding_by_tick.append({"tick": tick, **dict(binding)})
        selected_actions.append(
            _selected_action(tick=tick, function_name="repo_grep", kwargs={"query": "failure symbols", "root": "."}, state="SEARCHED")
        )
        tick += 1
        target_file = str(binding.get("top_target_file") or "")
        if target_file:
            state["read_files"].append({"path": target_file})
            selected_actions.append(
                _selected_action(
                    tick=tick,
                    function_name="file_read",
                    kwargs={"path": target_file, "start_line": 1, "end_line": 240},
                    state="FILE_READ",
                )
            )
            tick += 1
        proposal_payload = generate_patch_proposals(
            {**context, "investigation_state": state},
            top_target_file=target_file,
            max_changed_lines=20,
        )
        proposals = [
            dict(row)
            for row in _as_list(proposal_payload.get("patch_proposals"))
            if isinstance(row, Mapping)
        ]
        selected = dict(proposals[0]) if proposals else {}
        patch_event = {
            "tick": tick,
            "state": "PATCH_PROPOSAL_GENERATED" if proposals else "PATCH_PROPOSAL_NOT_GENERATED",
            "success": bool(proposals),
            "patch_proposals_generated": len(proposals),
            "patch_proposal_selected": selected,
            "patch_proposal_source": str(selected.get("proposal_source") or ""),
            "patch_proposal_rationale": str(selected.get("rationale") or ""),
            "patch_proposal_applied": False,
            "patch_proposal_verified": False,
            "patch_proposal_rollback_count": 0,
            "rejected_patch_proposals": [],
            "proposal_test_results": [],
            "needs_human_review": bool(not proposals),
            "refusal_reason": str(proposal_payload.get("refusal_reason") or proposal_payload.get("rejection_reason") or "evidence_insufficient") if not proposals else "",
            "target_binding": dict(binding),
        }
        selected_actions.append(
            _selected_action(
                tick=tick,
                function_name="propose_patch",
                kwargs={"target_file": target_file, "max_changed_lines": 20},
                success=bool(proposals),
                state=str(patch_event["state"]),
            )
        )
        tick += 1
        final_tests_passed = False
        if selected:
            backups = _snapshot(source_root)
            try:
                touched = _apply_single_file_unified_diff(source_root, str(selected.get("unified_diff") or ""))
                patch_event["patch_proposal_applied"] = True
                patch_event["touched_files"] = touched
                expected_tests = [str(item) for item in _as_list(selected.get("expected_tests")) if str(item)]
                if "." not in expected_tests:
                    expected_tests.append(".")
                verified = True
                for test_target in list(dict.fromkeys(expected_tests)):
                    if tick > int(max_ticks):
                        verified = False
                        break
                    test_result = _run_pytest(source_root, test_target, run_output_root, tick)
                    state["validation_runs"].append(
                        {"run_ref": test_result["run_ref"], "target": test_target, "success": bool(test_result["success"])}
                    )
                    patch_event["proposal_test_results"].append(
                        {
                            "target": test_target,
                            "success": bool(test_result["success"]),
                            "run_ref": test_result["run_ref"],
                            "returncode": int(test_result["returncode"]),
                        }
                    )
                    selected_actions.append(
                        _selected_action(
                            tick=tick,
                            function_name="run_test",
                            kwargs={"target": test_target, "timeout_seconds": 30},
                            success=bool(test_result["success"]),
                            state="TESTS_PASSED" if test_result["success"] else "TESTS_FAILED",
                        )
                    )
                    tick += 1
                    if not test_result["success"]:
                        verified = False
                        break
                if verified:
                    patch_event["patch_proposal_verified"] = True
                    final_tests_passed = True
                else:
                    for relative, content in backups.items():
                        (source_root / relative).write_text(content, encoding="utf-8")
                    rollback_count = 1
                    patch_event["patch_proposal_rollback_count"] = 1
                    patch_event["rejected_patch_proposals"].append(
                        {"reason": "verifier_rejected_patch", "proposal": selected, "rollback_count": 1}
                    )
            except Exception as exc:
                for relative, content in backups.items():
                    (source_root / relative).write_text(content, encoding="utf-8")
                rollback_count = 1
                patch_event["patch_proposal_rollback_count"] = 1
                patch_event["rejected_patch_proposals"].append(
                    {"reason": str(exc), "proposal": selected, "rollback_count": 1}
                )
        patch_events.append(patch_event)
        after = _snapshot(source_root)
        diff = _diff_summary(before, after)
        report = {
            "schema_version": EXTERNAL_BASELINE_VERSION,
            "variant": "external_coding_agent_baseline",
            "runner": "ordinary coding-agent style deterministic baseline",
            "manual_action_selection": False,
            "instruction": instruction,
            "run_id": run_id,
            "success": bool(final_tests_passed and diff["changed_paths"]),
            "ticks": len(selected_actions),
            "selected_actions": selected_actions,
            "competing_hypotheses_by_tick": [],
            "ranked_discriminating_experiments_by_tick": [],
            "posterior_summary_by_tick": [],
            "hypothesis_posterior_events_by_tick": [],
            "target_binding_by_tick": target_binding_by_tick,
            "patch_proposal_events": patch_events,
            "patch_proposals_generated": len(proposals),
            "patch_proposal_selected": bool(selected),
            "patch_proposal_source": str(selected.get("proposal_source") or ""),
            "patch_proposal_rationale": str(selected.get("rationale") or ""),
            "patch_proposal_applied": bool(patch_event.get("patch_proposal_applied")),
            "patch_proposal_verified": bool(patch_event.get("patch_proposal_verified")),
            "patch_proposal_rollback_count": rollback_count,
            "rejected_patch_proposals": list(patch_event.get("rejected_patch_proposals", []) or []),
            "needs_human_review": bool(patch_event.get("needs_human_review")),
            "refusal_reason": str(patch_event.get("refusal_reason") or ""),
            "final_tests_passed": final_tests_passed,
            "completed_before_verification": False,
            "whether_patch_was_selected_by_mainloop": False,
            "whether_patch_was_selected_after_posterior_bridge": False,
            "whether_posterior_changed_action": False,
            "repair_dependency_ratio": 0.0,
            "side_effect_after_verified_completion_count": 0,
            "terminal_state_by_tick": [],
            "terminal_state": "completed_verified" if final_tests_passed else ("needs_human_review" if patch_event.get("needs_human_review") else ""),
            "terminal_tick": tick - 1 if final_tests_passed or patch_event.get("needs_human_review") else None,
            "verified_completion": final_tests_passed,
            "final_diff_summary": diff,
            "audit_excerpt": {
                "external_baseline": True,
                "source_file_count": len(_source_files(source_root)),
            },
        }
        if report_path:
            _write_json(Path(report_path), report)
        return report
    finally:
        temp.cleanup()
