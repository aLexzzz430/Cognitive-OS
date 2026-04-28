"""Patch, edit, and validation actions for the local-machine surface."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, Mapping, Sequence

from integrations.local_machine.action_grounding import pytest_context_paths_from_tree
from integrations.local_machine.patch_proposal import generate_patch_proposals
from integrations.local_machine.target_binding import bind_target
from modules.local_mirror.mirror import (
    MirrorScopeError,
    materialize_files,
    open_mirror,
    run_mirror_command,
)


LOCAL_MACHINE_RUN_OUTPUT_VERSION = "conos.local_machine.run_output/v1"
DEFAULT_REPO_EXCLUDES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "node_modules",
        "__pycache__",
        ".venv",
    }
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


class LocalMachineExecutionActionsMixin:
    def _validate_patch_bounds(self, patch: str, *, max_files: int, max_hunks: int) -> None:
        files = set()
        hunks = 0
        for line in patch.splitlines():
            if line.startswith("+++ "):
                files.add(line[4:].strip().split("\t", 1)[0])
            elif line.startswith("@@"):
                hunks += 1
        if not patch.strip():
            raise MirrorScopeError("apply_patch requires a non-empty patch")
        if len(files) > max_files:
            raise MirrorScopeError(f"patch touches too many files: {len(files)} > {max_files}")
        if hunks > max_hunks:
            raise MirrorScopeError(f"patch has too many hunks: {hunks} > {max_hunks}")

    @staticmethod
    def _patch_header_path(raw: str) -> str:
        value = str(raw or "").strip().split("\t", 1)[0]
        if value in {"/dev/null", "dev/null"}:
            return ""
        for prefix in ("a/", "b/"):
            if value.startswith(prefix):
                return value[len(prefix):]
        return value

    def _parse_unified_patch(self, patch: str) -> list[Dict[str, Any]]:
        lines = patch.splitlines(keepends=True)
        entries: list[Dict[str, Any]] = []
        index = 0
        while index < len(lines):
            if not lines[index].startswith("--- "):
                index += 1
                continue
            old_path = self._patch_header_path(lines[index][4:])
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise MirrorScopeError("unified patch is missing +++ header")
            new_path = self._patch_header_path(lines[index][4:])
            index += 1
            hunks: list[Dict[str, Any]] = []
            while index < len(lines) and not lines[index].startswith("--- "):
                header = lines[index]
                if not header.startswith("@@"):
                    index += 1
                    continue
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
                context_only_hunk = header.strip() == "@@"
                if not match and not context_only_hunk:
                    raise MirrorScopeError(f"unsupported patch hunk header: {header.strip()}")
                index += 1
                hunk_lines: list[str] = []
                while index < len(lines) and not lines[index].startswith("@@") and not lines[index].startswith("--- "):
                    hunk_lines.append(lines[index])
                    index += 1
                hunks.append({
                    "old_start": int(match.group(1)) if match else 1,
                    "old_count": int(match.group(2) or "1") if match else 1,
                    "new_start": int(match.group(3)) if match else 1,
                    "new_count": int(match.group(4) or "1") if match else 1,
                    "lines": hunk_lines,
                })
            path = new_path or old_path
            if not path:
                raise MirrorScopeError("patch file path is empty")
            if not hunks:
                raise MirrorScopeError(f"unified patch has no hunks for {path}")
            entries.append({"path": path, "old_path": old_path, "new_path": new_path, "hunks": hunks})
        if not entries:
            raise MirrorScopeError("no file patches found")
        return entries

    def _rollback_workspace_files(self, backups: Mapping[str, str]) -> None:
        workspace_root = self._workspace_root()
        for relative, content in backups.items():
            path = (workspace_root / relative).resolve()
            try:
                path.relative_to(workspace_root)
            except ValueError:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")

    def _apply_patch_entry(self, entry: Mapping[str, Any]) -> str:
        workspace_path, relative = self._ensure_workspace_file(entry.get("path"))
        original_lines = workspace_path.read_text(encoding="utf-8").splitlines(keepends=True) if workspace_path.exists() else []
        output: list[str] = []
        cursor = 0
        for hunk in list(entry.get("hunks", []) or []):
            old_start = int(hunk.get("old_start", 1) or 1)
            target = max(0, old_start - 1)
            old_sequence = [
                raw_line[1:].rstrip("\n")
                for raw_line in list(hunk.get("lines", []) or [])
                if raw_line and raw_line[:1] in {" ", "-"}
            ]
            if old_sequence:
                exact_end = target + len(old_sequence)
                current_sequence = [line.rstrip("\n") for line in original_lines[target:exact_end]]
                if current_sequence != old_sequence:
                    fuzzy_target = -1
                    for candidate in range(cursor, len(original_lines) - len(old_sequence) + 1):
                        candidate_sequence = [
                            line.rstrip("\n")
                            for line in original_lines[candidate : candidate + len(old_sequence)]
                        ]
                        if candidate_sequence == old_sequence:
                            fuzzy_target = candidate
                            break
                    if fuzzy_target >= 0:
                        target = fuzzy_target
            if target < cursor:
                raise MirrorScopeError(f"overlapping patch hunks for {relative}")
            output.extend(original_lines[cursor:target])
            cursor = target
            for raw_line in list(hunk.get("lines", []) or []):
                if raw_line.startswith("\\ No newline"):
                    continue
                marker = raw_line[:1]
                content = raw_line[1:]
                if marker == " ":
                    if cursor >= len(original_lines) or original_lines[cursor].rstrip("\n") != content.rstrip("\n"):
                        raise MirrorScopeError(f"patch context mismatch for {relative}")
                    output.append(original_lines[cursor])
                    cursor += 1
                elif marker == "-":
                    if cursor >= len(original_lines) or original_lines[cursor].rstrip("\n") != content.rstrip("\n"):
                        raise MirrorScopeError(f"patch deletion mismatch for {relative}")
                    cursor += 1
                elif marker == "+":
                    output.append(content)
                else:
                    raise MirrorScopeError(f"unsupported patch line marker for {relative}: {marker}")
        output.extend(original_lines[cursor:])
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text("".join(output), encoding="utf-8")
        return relative

    def _proposal_expected_tests(self, proposal: Mapping[str, Any], kwargs: Mapping[str, Any]) -> list[str]:
        tests = _string_list(kwargs.get("expected_tests")) or _string_list(proposal.get("expected_tests"))
        tests = [self._normalize_expected_test_target(test) for test in tests]
        tests = [test for test in tests if test and test != "."]
        failed_target = ""
        state = self._load_investigation_state()
        for row in reversed(list(state.get("validation_runs", []) or [])):
            item = dict(row) if isinstance(row, dict) else {}
            if bool(item.get("success", False)):
                continue
            payload = self._load_run_output_for_state(str(item.get("run_ref") or ""))
            command = [str(part) for part in list(payload.get("command", []) or [])]
            if command:
                failed_target = str(command[-1] or "")
                break
        if failed_target and failed_target != ".":
            tests.insert(0, failed_target)
        tests.append(".")
        return list(dict.fromkeys(tests))

    @staticmethod
    def _normalize_expected_test_target(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()
        if not tokens:
            return ""
        if len(tokens) >= 3 and tokens[0] in {"python", "python3", sys.executable} and tokens[1:3] == ["-m", "pytest"]:
            tokens = tokens[3:]
        elif Path(tokens[0]).name == "pytest":
            tokens = tokens[1:]
        candidates: list[str] = []
        for token in tokens:
            if not token or token.startswith("-"):
                continue
            if token in {"pytest", "python", "python3"}:
                continue
            if token.startswith(("tests/", "test_")) or "/test_" in token or token.endswith(".py"):
                candidates.append(token)
        target = candidates[0] if candidates else ("." if any(token.startswith("-") for token in tokens) else tokens[0])
        if "::" in target:
            target = target.split("::", 1)[0]
        if target in {"tests", "tests/"}:
            return "."
        return target

    def _load_run_output_for_state(self, run_ref: str) -> Dict[str, Any]:
        if not re.fullmatch(r"run_[A-Za-z0-9_.-]+", str(run_ref or "")):
            return {}
        return _load_json(self._run_output_root() / f"{run_ref}.json")

    def _record_patch_proposal(self, event: Mapping[str, Any]) -> None:
        state = self._load_investigation_state()
        proposals = [
            dict(row)
            for row in list(state.get("patch_proposals", []) or [])
            if isinstance(row, dict)
        ]
        proposals.append(dict(event))
        state["patch_proposals"] = proposals[-100:]
        self._save_investigation_state(state)

    def _act_propose_patch(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        state = self._load_investigation_state()
        context = self._action_grounding_context(state_override=state)
        binding = bind_target(context)
        state["target_binding"] = dict(binding)
        state = self._ensure_explicit_hypothesis_lifecycle(
            state,
            function_name="propose_patch",
            kwargs=kwargs,
            raw_result={"success": False, "state": "PRE_PATCH_PROPOSAL"},
            binding=dict(binding),
            target_file=str(kwargs.get("target_file") or binding.get("top_target_file") or ""),
        )
        leading = self._leading_hypothesis_from_state(state)
        if not leading:
            event = {
                "event_type": "patch_proposal_blocked",
                "reason": "missing_leading_hypothesis",
                "target_binding": dict(binding),
                "created_at": _now(),
            }
            self._record_patch_proposal(event)
            return self._raw_success(
                function_name="propose_patch",
                reward=0.0,
                state="PATCH_PROPOSAL_NEEDS_HYPOTHESIS",
                success=False,
                patch_proposals_generated=0,
                needs_human_review=True,
                refusal_reason="missing_leading_hypothesis",
                target_binding=dict(binding),
                source_hypothesis_id="",
                leading_hypothesis_before_patch={},
                hypothesis_target_file="",
                rejected_patch_proposals=[event],
            )
        leading_id = str(leading.get("hypothesis_id") or "")
        leading_target = str(leading.get("target_file") or "")
        target_file = str(kwargs.get("target_file") or binding.get("top_target_file") or "").strip()
        if not target_file and leading_target:
            target_file = leading_target
        leading_payload = {
            "source_hypothesis_id": leading_id,
            "leading_hypothesis_before_patch": dict(leading),
            "hypothesis_target_file": leading_target,
        }
        proposal_payload = generate_patch_proposals(
            {
                **context,
                "llm_thinking_mode": self.llm_thinking_mode,
                "investigation_state": {**state, "target_binding": binding},
            },
            top_target_file=target_file,
            max_changed_lines=self._bounded_int(kwargs.get("max_changed_lines"), default=20, minimum=1, maximum=20),
            llm_client=self.llm_client if self.prefer_llm_patch_proposals else None,
            prefer_llm=False,
            allow_fallback_patch=bool(kwargs.get("allow_fallback_patch", False)),
        )
        proposals = [
            dict(row)
            for row in list(proposal_payload.get("patch_proposals", []) or [])
            if isinstance(row, Mapping)
        ]
        fallback_patch = str(kwargs.get("fallback_patch") or "").strip()
        if fallback_patch and bool(kwargs.get("allow_fallback_patch", False)):
            fallback_changed_lines = sum(
                1
                for line in fallback_patch.splitlines()
                if line
                and not line.startswith(("+++", "---", "@@"))
                and line[0] in {"+", "-"}
            )
            if fallback_changed_lines <= self._bounded_int(kwargs.get("max_changed_lines"), default=20, minimum=1, maximum=20):
                proposals.append(
                    {
                        "schema_version": "conos.local_machine.patch_proposal/v1",
                        "proposal_id": f"grounded_{hashlib.sha256(fallback_patch.encode('utf-8')).hexdigest()[:12]}",
                        "target_file": target_file,
                        "unified_diff": fallback_patch,
                        "rationale": "grounded bounded patch inferred from the current hypothesis and failure evidence; verifier gate decides acceptance",
                        "evidence_refs": _string_list(kwargs.get("fallback_evidence_refs")),
                        "expected_tests": _string_list(kwargs.get("expected_tests")),
                        "risk": 0.28,
                        "proposal_source": "grounded_patch_fallback",
                        "patch_sha256": hashlib.sha256(fallback_patch.encode("utf-8")).hexdigest(),
                    }
                )
        generated_event = {
            "event_type": "patch_proposals_generated",
            "target_binding": dict(binding),
            "proposal_payload": dict(proposal_payload),
            "patch_proposal_llm_trace": list(proposal_payload.get("llm_trace", []) or []),
            **leading_payload,
            "created_at": _now(),
        }
        self._record_patch_proposal(generated_event)
        if not proposals:
            refusal_reason = str(proposal_payload.get("refusal_reason") or proposal_payload.get("rejection_reason") or "evidence_insufficient")
            state_name = "PATCH_PROPOSAL_TIMEOUT" if refusal_reason == "timeout" else "PATCH_PROPOSAL_NOT_GENERATED"
            return self._raw_success(
                function_name="propose_patch",
                reward=0.0,
                state=state_name,
                success=False,
                patch_proposals_generated=0,
                target_binding=dict(binding),
                patch_proposal_llm_trace=list(proposal_payload.get("llm_trace", []) or []),
                **leading_payload,
                needs_human_review=refusal_reason != "timeout",
                refusal_reason=refusal_reason,
                llm_timeout=bool(proposal_payload.get("llm_timeout", False)),
                suggested_replan_reason="llm_patch_proposal_timeout" if refusal_reason == "timeout" else "patch_proposal_not_generated",
                rejected_patch_proposals=[
                    {
                        "reason": refusal_reason,
                        "target_file": target_file,
                        "needs_human_review": refusal_reason != "timeout",
                        "llm_timeout": bool(proposal_payload.get("llm_timeout", False)),
                    }
                ],
            )
        selected = {**proposals[0], **leading_payload}
        patch = str(selected.get("unified_diff") or "")
        try:
            self._validate_patch_bounds(patch, max_files=1, max_hunks=5)
            entries = self._parse_unified_patch(patch)
            if len(entries) != 1:
                raise MirrorScopeError("patch proposal must touch exactly one source file")
            target = str(entries[0].get("path") or "")
            if target.startswith("tests/") or "/tests/" in target:
                raise MirrorScopeError("patch proposal may not modify tests")
        except MirrorScopeError as exc:
            rejected = {
                "event_type": "patch_proposal_rejected",
                "proposal": dict(selected),
                "reason": str(exc),
                "rollback_count": 0,
                "created_at": _now(),
            }
            self._record_patch_proposal(rejected)
            return self._raw_success(
                function_name="propose_patch",
                reward=0.0,
                state="PATCH_PROPOSAL_REJECTED",
                success=False,
                patch_proposals_generated=len(proposals),
                patch_proposal_selected=dict(selected),
                patch_proposal_source=str(selected.get("proposal_source") or ""),
                patch_proposal_rationale=str(selected.get("rationale") or ""),
                patch_proposal_llm_trace=list(proposal_payload.get("llm_trace", []) or []),
                **leading_payload,
                patch_proposal_applied=False,
                patch_proposal_verified=False,
                patch_proposal_rollback_count=0,
                rejected_patch_proposals=[rejected],
                proposal_test_results=[],
                target_binding=dict(binding),
            )
        backups: dict[str, str] = {}
        touched: list[str] = []
        rollback_count = 0
        try:
            for entry in entries:
                workspace_path, relative = self._ensure_workspace_file(entry.get("path"))
                backups[relative] = workspace_path.read_text(encoding="utf-8") if workspace_path.exists() else ""
            touched = [self._apply_patch_entry(entry) for entry in entries]
            test_results: list[Dict[str, Any]] = []
            verified = True
            for test_target in self._proposal_expected_tests(selected, kwargs):
                result = self._act_run_test({"target": test_target, "timeout_seconds": 30})
                test_results.append(
                    {
                        "target": test_target,
                        "success": bool(result.get("success", False)),
                        "run_ref": str(result.get("run_ref") or ""),
                        "returncode": int(result.get("returncode", 0) or 0),
                    }
                )
                if not bool(result.get("success", False)):
                    verified = False
                    break
            if not verified:
                self._rollback_workspace_files(backups)
                rollback_count = 1
                rejected = {
                    "event_type": "patch_proposal_rejected",
                    "proposal": dict(selected),
                    "test_results": test_results,
                    "rollback_count": rollback_count,
                    "created_at": _now(),
                }
                self._record_patch_proposal(rejected)
                return self._raw_success(
                    function_name="propose_patch",
                    reward=0.0,
                    state="PATCH_PROPOSAL_REJECTED",
                    success=False,
                    patch_proposals_generated=len(proposals),
                    patch_proposal_selected=dict(selected),
                    patch_proposal_source=str(selected.get("proposal_source") or ""),
                    patch_proposal_rationale=str(selected.get("rationale") or ""),
                    patch_proposal_llm_trace=list(proposal_payload.get("llm_trace", []) or []),
                    **leading_payload,
                    patch_proposal_applied=True,
                    patch_proposal_verified=False,
                    patch_proposal_rollback_count=rollback_count,
                    rejected_patch_proposals=[rejected],
                    proposal_test_results=test_results,
                    target_binding=dict(binding),
                )
        except Exception:
            if backups:
                self._rollback_workspace_files(backups)
            raise
        accepted = {
            "event_type": "patch_proposal_accepted",
            "proposal": dict(selected),
            "touched_files": touched,
            "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            **leading_payload,
            "created_at": _now(),
        }
        self._record_patch_proposal(accepted)
        return self._raw_success(
            function_name="propose_patch",
            reward=0.55,
            state="PATCH_PROPOSAL_VERIFIED",
            success=True,
            touched_files=touched,
            patch_sha256=hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            patch_proposals_generated=len(proposals),
            patch_proposal_selected=dict(selected),
            patch_proposal_source=str(selected.get("proposal_source") or ""),
            patch_proposal_rationale=str(selected.get("rationale") or ""),
            patch_proposal_llm_trace=list(proposal_payload.get("llm_trace", []) or []),
            **leading_payload,
            patch_proposal_applied=True,
            patch_proposal_verified=True,
            patch_proposal_rollback_count=0,
            rejected_patch_proposals=[],
            proposal_test_results=test_results,
            target_binding=dict(binding),
            final_tests_passed=True,
        )

    def _act_apply_patch(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        patch = str(kwargs.get("patch") or "")
        max_files = self._bounded_int(kwargs.get("max_files"), default=1, minimum=1, maximum=5)
        max_hunks = self._bounded_int(kwargs.get("max_hunks"), default=3, minimum=1, maximum=20)
        self._validate_patch_bounds(patch, max_files=max_files, max_hunks=max_hunks)
        entries = self._parse_unified_patch(patch)
        if len(entries) > max_files:
            raise MirrorScopeError(f"patch touches too many files: {len(entries)} > {max_files}")
        state = self._load_investigation_state()
        leading = self._leading_hypothesis_from_state(state)
        if not leading:
            return self._raw_success(
                function_name="apply_patch",
                reward=0.0,
                state="PATCH_NEEDS_LEADING_HYPOTHESIS",
                success=False,
                needs_human_review=True,
                refusal_reason="missing_leading_hypothesis",
                source_hypothesis_id="",
                leading_hypothesis_before_patch={},
                hypothesis_target_file="",
            )
        leading_payload = {
            "source_hypothesis_id": str(leading.get("hypothesis_id") or ""),
            "leading_hypothesis_before_patch": dict(leading),
            "hypothesis_target_file": str(leading.get("target_file") or ""),
        }
        touched = [self._apply_patch_entry(entry) for entry in entries]
        self._append_note(kind="edit", content=f"Applied bounded patch to {', '.join(touched)}", evidence_refs=[f"patch:{_json_hash(patch)[:12]}"])
        return self._raw_success(
            function_name="apply_patch",
            reward=0.28,
            state="PATCH_APPLIED_TO_MIRROR",
            touched_files=touched,
            patch_sha256=hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            **leading_payload,
        )

    def _act_edit_replace_range(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        workspace_path, relative = self._ensure_workspace_file(kwargs.get("path"))
        existing = workspace_path.read_text(encoding="utf-8").splitlines(keepends=True) if workspace_path.exists() else []
        start_line = self._bounded_int(kwargs.get("start_line"), default=1, minimum=1, maximum=max(1, len(existing) + 1))
        end_line = self._bounded_int(kwargs.get("end_line"), default=start_line, minimum=start_line, maximum=max(start_line, len(existing)))
        replacement = str(kwargs.get("replacement") or "")
        replacement_lines = replacement.splitlines(keepends=True)
        if replacement and not replacement.endswith(("\n", "\r")):
            replacement_lines[-1:] = [replacement_lines[-1] + "\n"] if replacement_lines else [replacement + "\n"]
        updated = existing[: start_line - 1] + replacement_lines + existing[end_line:]
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text("".join(updated), encoding="utf-8")
        return self._raw_success(
            function_name="edit_replace_range",
            reward=0.24,
            state="RANGE_REPLACED_IN_MIRROR",
            path=relative,
            start_line=start_line,
            end_line=end_line,
        )

    def _act_edit_insert_after(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        workspace_path, relative = self._ensure_workspace_file(kwargs.get("path"))
        existing = workspace_path.read_text(encoding="utf-8").splitlines(keepends=True) if workspace_path.exists() else []
        line = self._bounded_int(kwargs.get("line"), default=len(existing), minimum=0, maximum=len(existing))
        insertion = str(kwargs.get("text") or kwargs.get("content") or "")
        insertion_lines = insertion.splitlines(keepends=True)
        if insertion and not insertion.endswith(("\n", "\r")):
            insertion_lines[-1:] = [insertion_lines[-1] + "\n"] if insertion_lines else [insertion + "\n"]
        updated = existing[:line] + insertion_lines + existing[line:]
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text("".join(updated), encoding="utf-8")
        return self._raw_success(
            function_name="edit_insert_after",
            reward=0.22,
            state="TEXT_INSERTED_IN_MIRROR",
            path=relative,
            line=line,
        )

    def _act_create_file(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        workspace_path, relative = self._resolve_workspace_path(kwargs.get("path"))
        if workspace_path.exists() and not bool(kwargs.get("overwrite", False)):
            raise MirrorScopeError(f"workspace file already exists: {relative}")
        content = str(kwargs.get("content") or "")
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text(content, encoding="utf-8")
        return self._raw_success(
            function_name="create_file",
            reward=0.24,
            state="FILE_CREATED_IN_MIRROR",
            path=relative,
            size_bytes=int(workspace_path.stat().st_size),
        )

    def _act_delete_file(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        workspace_path, relative = self._resolve_workspace_path(kwargs.get("path"))
        if not workspace_path.exists():
            raise FileNotFoundError(str(workspace_path))
        if not workspace_path.is_file():
            raise MirrorScopeError(f"workspace path is not a regular file: {relative}")
        workspace_path.unlink()
        return self._raw_success(
            function_name="delete_file",
            reward=0.15,
            state="FILE_DELETED_FROM_MIRROR",
            path=relative,
        )

    def _write_run_output(self, *, function_name: str, command: Sequence[str], completed: subprocess.CompletedProcess[str], timeout_seconds: int) -> Dict[str, Any]:
        payload = {
            "schema_version": LOCAL_MACHINE_RUN_OUTPUT_VERSION,
            "run_ref": "",
            "function_name": function_name,
            "command": [str(part) for part in command],
            "returncode": int(completed.returncode),
            "stdout": str(completed.stdout or ""),
            "stderr": str(completed.stderr or ""),
            "timeout_seconds": int(timeout_seconds),
            "created_at": _now(),
        }
        payload["run_ref"] = f"run_{_json_hash(payload)[:16]}"
        path = self._run_output_root() / f"{payload['run_ref']}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        state = self._load_investigation_state()
        state["last_run_ref"] = payload["run_ref"]
        runs = [
            dict(row)
            for row in list(state.get("validation_runs", []) or [])
            if isinstance(row, dict)
        ]
        runs.append(
            {
                "run_ref": payload["run_ref"],
                "function_name": function_name,
                "returncode": int(completed.returncode),
                "success": int(completed.returncode) == 0,
                "created_at": payload["created_at"],
                "timeout_seconds": int(timeout_seconds),
            }
        )
        state["validation_runs"] = runs[-100:]
        self._save_investigation_state(state)
        return payload

    def _run_workspace_command(self, *, function_name: str, command: Sequence[str], timeout_seconds: int) -> Dict[str, Any]:
        mirror_result = run_mirror_command(
            self.source_root,
            self.mirror_root,
            [str(part) for part in command],
            allowed_commands=[*self.allowed_commands, sys.executable, "python", "python3"],
            timeout_seconds=timeout_seconds,
            backend=self.execution_backend,
            docker_image=self.docker_image,
            vm_provider=self.vm_provider,
            vm_name=self.vm_name,
            vm_host=self.vm_host,
            vm_workdir=self.vm_workdir,
            vm_network_mode=self.vm_network_mode,
            vm_sync_mode=self.vm_sync_mode,
            extra_env=self.extra_env,
        )
        completed = subprocess.CompletedProcess(
            args=[str(part) for part in command],
            returncode=int(mirror_result.returncode),
            stdout=str(mirror_result.stdout or ""),
            stderr=str(mirror_result.stderr or ""),
        )
        output_text = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        if function_name in {"run_lint", "run_typecheck", "run_build"} and "Can't list" in output_text:
            completed = subprocess.CompletedProcess(
                args=[str(part) for part in command],
                returncode=2,
                stdout=str(completed.stdout or ""),
                stderr=str(completed.stderr or "") + "\nvalidation target is missing or not materialized",
            )
        payload = self._write_run_output(
            function_name=function_name,
            command=command,
            completed=completed,
            timeout_seconds=timeout_seconds,
        )
        return self._raw_success(
            function_name=function_name,
            reward=0.3 if completed.returncode == 0 else 0.0,
            state="VALIDATION_RUN_COMPLETED",
            success=completed.returncode == 0,
            run_ref=payload["run_ref"],
            returncode=int(completed.returncode),
            stdout_tail=str(completed.stdout or "")[-4000:],
            stderr_tail=str(completed.stderr or "")[-4000:],
            mirror_command=mirror_result.to_dict(),
            execution_boundary=dict(mirror_result.execution_boundary),
            sandbox_label=str(mirror_result.security_boundary or "best_effort_local_mirror"),
            not_os_security_sandbox=not bool(mirror_result.real_vm_boundary),
        )

    @staticmethod
    def _validation_path_part(target: str) -> str:
        return str(target or "").split("::", 1)[0].strip()

    def _ensure_validation_target_available(self, target: str) -> str:
        path_part = self._validation_path_part(target)
        if not path_part or path_part == ".":
            return "."
        workspace_path, relative = self._resolve_workspace_path(path_part)
        if workspace_path.exists():
            return relative
        source_path = (self.source_root / relative).resolve()
        try:
            source_path.relative_to(self.source_root)
        except ValueError as exc:
            raise MirrorScopeError(f"validation target is outside source root: {path_part}") from exc
        if source_path.exists() and source_path.is_file():
            materialize_files(self.source_root, self.mirror_root, [relative])
            return relative
        if source_path.exists() and source_path.is_dir():
            paths: list[str] = []
            for child in sorted(source_path.rglob("*")):
                if len(paths) >= 250:
                    break
                if not child.is_file() or child.is_symlink():
                    continue
                child_relative = child.resolve().relative_to(self.source_root).as_posix()
                if self._path_excluded(child_relative, DEFAULT_REPO_EXCLUDES):
                    continue
                paths.append(child_relative)
            if paths:
                materialize_files(self.source_root, self.mirror_root, paths)
                return relative
        raise MirrorScopeError(f"validation target does not exist in mirror or source: {path_part}")

    def _act_run_test(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        target = str(kwargs.get("target") or ".").strip() or "."
        timeout_seconds = self._bounded_int(kwargs.get("timeout_seconds"), default=30, minimum=1, maximum=600)
        materialized_for_full_test: list[str] = []
        if target != ".":
            self._ensure_validation_target_available(target)
            target_path = self._validation_path_part(target)
            if target_path == "tests" or target_path.startswith("tests/") or Path(target_path).name.startswith("test_"):
                context = self._action_grounding_context()
                materialized_for_full_test = self._materialize_missing_pytest_context(pytest_context_paths_from_tree(context, limit=200))
        else:
            context = self._action_grounding_context()
            materialized_for_full_test = self._materialize_missing_pytest_context(pytest_context_paths_from_tree(context, limit=200))
        raw = self._run_workspace_command(
            function_name="run_test",
            command=[sys.executable, "-m", "pytest", target],
            timeout_seconds=timeout_seconds,
        )
        if materialized_for_full_test:
            raw["materialized_for_full_test"] = materialized_for_full_test
        return raw

    def _materialize_missing_pytest_context(self, paths: Sequence[str]) -> list[str]:
        missing: list[str] = []
        workspace_root = self._workspace_root()
        for path in [str(item) for item in list(paths or []) if str(item)]:
            workspace_path = (workspace_root / path).resolve()
            try:
                workspace_path.relative_to(workspace_root)
            except ValueError:
                continue
            if not workspace_path.exists():
                missing.append(path)
        if missing:
            materialize_files(self.source_root, self.mirror_root, missing)
        return missing

    def _act_run_lint(self, kwargs: Mapping[str, Any], *, function_name: str) -> Dict[str, Any]:
        target = str(kwargs.get("target") or ".").strip() or "."
        timeout_seconds = self._bounded_int(kwargs.get("timeout_seconds"), default=30, minimum=1, maximum=600)
        materialized_for_validation: list[str] = []
        if target == ".":
            context = self._action_grounding_context()
            materialized_for_validation = self._materialize_missing_pytest_context(pytest_context_paths_from_tree(context, limit=200))
            if not materialized_for_validation and int(open_mirror(self.source_root, self.mirror_root).to_manifest().get("workspace_file_count", 0) or 0) <= 0:
                raise MirrorScopeError("validation target is empty; inspect or materialize files before running validation")
            command = [sys.executable, "-m", "compileall", "-q", "."]
        else:
            relative = self._ensure_validation_target_available(target)
            workspace_path, relative = self._resolve_workspace_path(relative)
            command = (
                [sys.executable, "-m", "py_compile", relative]
                if workspace_path.is_file() and workspace_path.suffix == ".py"
                else [sys.executable, "-m", "compileall", "-q", relative]
            )
        raw = self._run_workspace_command(
            function_name=function_name,
            command=command,
            timeout_seconds=timeout_seconds,
        )
        if materialized_for_validation:
            raw["materialized_for_validation"] = materialized_for_validation
        return raw

    def _act_read_run_output(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        run_ref = str(kwargs.get("run_ref") or "").strip()
        if not run_ref:
            run_ref = str(self._load_investigation_state().get("last_run_ref", "") or "")
        if not re.fullmatch(r"run_[A-Za-z0-9_.-]+", run_ref):
            raise MirrorScopeError("read_run_output requires a valid run_ref")
        path = self._run_output_root() / f"{run_ref}.json"
        payload = _load_json(path)
        if not payload:
            raise FileNotFoundError(str(path))
        max_chars = self._bounded_int(kwargs.get("max_chars"), default=6000, minimum=100, maximum=50000)
        stdout = str(payload.get("stdout", "") or "")
        stderr = str(payload.get("stderr", "") or "")
        return self._raw_success(
            function_name="read_run_output",
            reward=0.05,
            state="RUN_OUTPUT_READ",
            run_ref=run_ref,
            returncode=int(payload.get("returncode", 0) or 0),
            stdout=stdout[-max_chars:],
            stderr=stderr[-max_chars:],
            command=list(payload.get("command", []) or []),
        )
