"""Investigation and read-only atomic actions for the local-machine surface."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import fnmatch
from pathlib import Path
import re
from typing import Any, Dict, Mapping

from core.runtime.hypothesis_lifecycle import (
    HYPOTHESIS_LIFECYCLE_VERSION,
    apply_hypothesis_evidence,
    build_discriminating_test,
    hypothesis_lifecycle_summary,
    mark_competing,
    normalize_hypothesis,
)
from modules.local_mirror.mirror import MirrorScopeError


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LocalMachineInvestigationActionsMixin:
    def _act_repo_tree(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        depth = self._bounded_int(kwargs.get("depth"), default=2, minimum=0, maximum=6)
        max_entries = self._bounded_int(kwargs.get("max_entries"), default=200, minimum=1, maximum=1000)
        excludes = set(DEFAULT_REPO_EXCLUDES)
        excludes.update(_string_list(kwargs.get("exclude") or kwargs.get("excludes")))
        root_path, root_relative = self._resolve_source_path(kwargs.get("path") or kwargs.get("root") or ".", allow_empty=True)
        if not root_path.exists():
            raise FileNotFoundError(str(root_path))
        entries: list[Dict[str, Any]] = []

        def walk(directory: Path, current_depth: int) -> None:
            if len(entries) >= max_entries or current_depth >= depth:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            except OSError:
                return
            for child in children:
                if len(entries) >= max_entries:
                    return
                try:
                    relative = child.resolve().relative_to(self.source_root).as_posix()
                except ValueError:
                    continue
                if self._path_excluded(relative, excludes) or child.is_symlink():
                    continue
                row = {
                    "path": relative,
                    "name": child.name,
                    "kind": "dir" if child.is_dir() else "file",
                    "size_bytes": int(child.stat().st_size) if child.is_file() else 0,
                    "depth": current_depth + 1,
                }
                entries.append(row)
                if child.is_dir():
                    walk(child, current_depth + 1)

        if root_path.is_file():
            entries.append({
                "path": root_relative,
                "name": root_path.name,
                "kind": "file",
                "size_bytes": int(root_path.stat().st_size),
                "depth": 0,
            })
        else:
            walk(root_path, 0)
        state = self._load_investigation_state()
        stored_entry_limit = min(max_entries, 500)
        state["last_tree"] = {
            "root": root_relative,
            "depth": depth,
            "entry_count": len(entries),
            "entries": entries[:stored_entry_limit],
            "entries_stored": min(len(entries), stored_entry_limit),
        }
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="repo_tree",
            reward=0.15 if entries else 0.05,
            state="REPO_TREE_READ",
            root=root_relative,
            depth=depth,
            entry_count=len(entries),
            truncated=len(entries) >= max_entries,
            entries=entries,
        )

    def _act_repo_find(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        root_path, root_relative = self._resolve_source_path(kwargs.get("root") or kwargs.get("path") or ".", allow_empty=True)
        pattern = str(kwargs.get("name_pattern") or kwargs.get("pattern") or "*").strip() or "*"
        kind = str(kwargs.get("kind") or "any").strip().lower()
        max_results = self._bounded_int(kwargs.get("max_results"), default=50, minimum=1, maximum=500)
        excludes = set(DEFAULT_REPO_EXCLUDES)
        excludes.update(_string_list(kwargs.get("exclude") or kwargs.get("excludes")))
        results: list[Dict[str, Any]] = []
        iterator = root_path.rglob("*") if root_path.is_dir() else [root_path]
        for path in iterator:
            if len(results) >= max_results:
                break
            if path.is_symlink():
                continue
            rel = path.resolve().relative_to(self.source_root).as_posix()
            if self._path_excluded(rel, excludes) or not fnmatch.fnmatch(path.name, pattern):
                continue
            row_kind = "dir" if path.is_dir() else "file"
            if kind not in {"any", row_kind}:
                continue
            results.append({"path": rel, "name": path.name, "kind": row_kind})
        state = self._load_investigation_state()
        state["last_search"] = {"action": "repo_find", "root": root_relative, "pattern": pattern, "result_count": len(results), "results": results[:50]}
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="repo_find",
            reward=0.15 if results else 0.03,
            state="REPO_FIND_READ",
            root=root_relative,
            name_pattern=pattern,
            result_count=len(results),
            results=results,
        )

    def _act_repo_grep(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        root_path, root_relative = self._resolve_source_path(kwargs.get("root") or kwargs.get("path") or ".", allow_empty=True)
        query = str(kwargs.get("query") or kwargs.get("pattern") or "").strip()
        if not query:
            raise MirrorScopeError("repo_grep requires a query")
        globs = _string_list(kwargs.get("globs")) or ["*.py", "*.ts", "*.tsx", "*.js", "*.md", "*.toml", "*.json", "*.yaml", "*.yml"]
        max_matches = self._bounded_int(kwargs.get("max_matches"), default=50, minimum=1, maximum=500)
        case_sensitive = bool(kwargs.get("case_sensitive", False))
        use_regex = bool(kwargs.get("regex", False))
        matcher = re.compile(query, 0 if case_sensitive else re.IGNORECASE) if use_regex else None
        needle = query if case_sensitive else query.lower()
        matches: list[Dict[str, Any]] = []
        for path in sorted(root_path.rglob("*") if root_path.is_dir() else [root_path]):
            if len(matches) >= max_matches:
                break
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 512 * 1024:
                continue
            rel = path.resolve().relative_to(self.source_root).as_posix()
            if self._path_excluded(rel, DEFAULT_REPO_EXCLUDES):
                continue
            if not any(fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(path.name, glob) for glob in globs):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.lower()
                found = bool(matcher.search(line)) if matcher is not None else needle in haystack
                if not found:
                    continue
                matches.append({"path": rel, "line": line_number, "text": line[:500]})
                if len(matches) >= max_matches:
                    break
        state = self._load_investigation_state()
        state["last_search"] = {"action": "repo_grep", "root": root_relative, "query": query, "match_count": len(matches), "matches": matches[:50]}
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="repo_grep",
            reward=0.18 if matches else 0.02,
            state="REPO_GREP_READ",
            root=root_relative,
            query=query,
            match_count=len(matches),
            matches=matches,
        )

    def _read_path_for_investigation(self, raw_path: Any) -> tuple[Path, str, str]:
        source_path, relative = self._resolve_source_path(raw_path)
        source_relative = Path(relative)
        path_correction = ""
        if not source_path.exists():
            repaired_relative, path_correction = self._repair_missing_source_relative_path(source_relative)
            if repaired_relative != source_relative:
                source_path = (self.source_root / repaired_relative).resolve()
                relative = repaired_relative.as_posix()
        workspace_path = (self._workspace_root() / relative).resolve()
        if workspace_path.exists() and workspace_path.is_file():
            return workspace_path, relative, "workspace"
        if path_correction:
            state = self._load_investigation_state()
            state["last_path_correction"] = {
                "requested_path": str(raw_path or ""),
                "correction": path_correction,
                "resolved_path": relative,
                "created_at": _now(),
            }
            self._save_investigation_state(state)
        return source_path, relative, "source"

    def _act_file_read(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        path, relative, location = self._read_path_for_investigation(kwargs.get("path"))
        text = self._read_text_file(path)
        lines = text.splitlines()
        start_line = self._bounded_int(kwargs.get("start_line"), default=1, minimum=1, maximum=max(1, len(lines) + 1))
        end_default = min(len(lines), start_line + 199)
        end_line = self._bounded_int(kwargs.get("end_line"), default=end_default, minimum=start_line, maximum=max(start_line, len(lines)))
        selected = lines[start_line - 1:end_line]
        numbered = [{"line": start_line + idx, "text": line} for idx, line in enumerate(selected)]
        state = self._load_investigation_state()
        state["last_read"] = {"path": relative, "location": location, "start_line": start_line, "end_line": end_line}
        read_files = [dict(row) for row in list(state.get("read_files", []) or []) if isinstance(row, dict)]
        read_files.append({"path": relative, "location": location, "start_line": start_line, "end_line": end_line, "created_at": _now()})
        state["read_files"] = read_files[-100:]
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="file_read",
            reward=0.2,
            state="FILE_READ",
            path=relative,
            location=location,
            start_line=start_line,
            end_line=end_line,
            total_lines=len(lines),
            lines=numbered,
            content="\n".join(selected),
        )

    def _outline_python(self, path: Path, text: str) -> list[Dict[str, Any]]:
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return [{"kind": "syntax_error", "name": str(exc), "line": int(exc.lineno or 0)}]
        rows: list[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rows.append({
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "name": node.name,
                    "line": int(getattr(node, "lineno", 0) or 0),
                    "end_line": int(getattr(node, "end_lineno", 0) or 0),
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                rows.append({
                    "kind": "import",
                    "name": path.name,
                    "line": int(getattr(node, "lineno", 0) or 0),
                })
        return sorted(rows, key=lambda row: (int(row.get("line", 0) or 0), str(row.get("kind", ""))))

    def _act_file_outline(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        path, relative, location = self._read_path_for_investigation(kwargs.get("path"))
        text = self._read_text_file(path)
        if path.suffix == ".py":
            outline = self._outline_python(path, text)
        else:
            outline = [
                {"kind": "heading", "name": line.strip("# "), "line": idx}
                for idx, line in enumerate(text.splitlines(), start=1)
                if line.lstrip().startswith("#")
            ][:80]
        return self._raw_success(
            function_name="file_outline",
            reward=0.16 if outline else 0.05,
            state="FILE_OUTLINE_READ",
            path=relative,
            location=location,
            outline=outline[:200],
            outline_count=len(outline),
        )

    def _act_file_summary(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        path, relative, location = self._read_path_for_investigation(kwargs.get("path"))
        text = self._read_text_file(path)
        lines = text.splitlines()
        outline = self._outline_python(path, text)[:30] if path.suffix == ".py" else []
        return self._raw_success(
            function_name="file_summary",
            reward=0.12,
            state="FILE_SUMMARY_READ",
            path=relative,
            location=location,
            suffix=path.suffix,
            size_bytes=int(path.stat().st_size),
            total_lines=len(lines),
            first_lines=lines[:40],
            outline=outline,
        )

    def _act_note_write(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        note = self._append_note(
            kind=str(kwargs.get("kind") or "finding"),
            content=str(kwargs.get("content") or ""),
            evidence_refs=_string_list(kwargs.get("evidence_refs")),
        )
        return self._raw_success(
            function_name="note_write",
            reward=0.08,
            state="INVESTIGATION_NOTE_WRITTEN",
            note=note,
        )

    def _act_hypothesis_add(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        claim = str(kwargs.get("claim") or kwargs.get("content") or "").strip()
        if not claim:
            raise MirrorScopeError("hypothesis_add requires a claim")
        evidence_refs = self._validate_evidence_refs(kwargs.get("evidence_refs", []))
        predictions = kwargs.get("predictions", {})
        falsifiers = kwargs.get("falsifiers", {})
        hypothesis = normalize_hypothesis(
            hypothesis_id=str(kwargs.get("hypothesis_id") or f"hyp_{len(hypotheses) + 1:04d}"),
            run_id=self.task_id,
            task_family="local_machine",
            family=str(kwargs.get("family") or kwargs.get("hypothesis_type") or "codebase"),
            claim=claim,
            confidence=float(kwargs.get("confidence", 0.5) or 0.5),
            evidence_refs=evidence_refs,
            competing_with=_string_list(kwargs.get("competing_with")),
            predictions=dict(predictions) if isinstance(predictions, Mapping) else {},
            falsifiers=dict(falsifiers) if isinstance(falsifiers, Mapping) else {},
            metadata={
                "source": "local_machine_hypothesis_add",
                "created_at_iso": _now(),
            },
        )
        event = {
            "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
            "hypothesis_id": hypothesis["hypothesis_id"],
            "run_id": self.task_id,
            "event_type": "hypothesis_created",
            "evidence_refs": list(hypothesis.get("evidence_refs", []) or []),
            "delta": 0.0,
            "created_at": hypothesis["created_at"],
        }
        hypotheses.append(hypothesis)
        state["hypotheses"] = hypotheses[-100:]
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        events.append(event)
        state["hypothesis_events"] = events[-200:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        self._save_investigation_state(state)
        self._persist_hypothesis_lifecycle(hypothesis, event)
        return self._raw_success(
            function_name="hypothesis_add",
            reward=0.08,
            state="INVESTIGATION_HYPOTHESIS_ADDED",
            hypothesis=hypothesis,
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

    def _act_hypothesis_update(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        hypothesis_id = str(kwargs.get("hypothesis_id") or kwargs.get("id") or "").strip()
        if not hypothesis_id:
            raise MirrorScopeError("hypothesis_update requires hypothesis_id")
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        index = next(
            (idx for idx, row in enumerate(hypotheses) if str(row.get("hypothesis_id") or "") == hypothesis_id),
            -1,
        )
        if index < 0:
            raise MirrorScopeError(f"unknown hypothesis_id: {hypothesis_id}")
        refs = _string_list(kwargs.get("evidence_refs"))
        single_ref = str(kwargs.get("evidence_ref") or "").strip()
        if single_ref and single_ref not in refs:
            refs.append(single_ref)
        try:
            strength = float(kwargs.get("strength", 0.2) or 0.2)
        except (TypeError, ValueError):
            strength = 0.2
        updated, event = apply_hypothesis_evidence(
            hypotheses[index],
            signal=str(kwargs.get("signal") or kwargs.get("verdict") or kwargs.get("direction") or "neutral"),
            evidence_refs=refs,
            strength=strength,
            rationale=str(kwargs.get("rationale") or kwargs.get("why") or ""),
        )
        event["run_id"] = self.task_id
        hypotheses[index] = updated
        state["hypotheses"] = hypotheses[-100:]
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        events.append(event)
        state["hypothesis_events"] = events[-200:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        self._save_investigation_state(state)
        self._persist_hypothesis_lifecycle(updated, event)
        return self._raw_success(
            function_name="hypothesis_update",
            reward=0.1 if str(event.get("signal") or "") in {"support", "contradiction"} else 0.04,
            state="HYPOTHESIS_UPDATED",
            hypothesis=updated,
            hypothesis_event=event,
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

    def _act_hypothesis_compete(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        left = str(
            kwargs.get("hypothesis_a")
            or kwargs.get("left")
            or kwargs.get("hypothesis_id_a")
            or kwargs.get("hypothesis_id")
            or ""
        ).strip()
        right = str(
            kwargs.get("hypothesis_b")
            or kwargs.get("right")
            or kwargs.get("hypothesis_id_b")
            or kwargs.get("competes_with")
            or ""
        ).strip()
        if not left or not right or left == right:
            raise MirrorScopeError("hypothesis_compete requires two distinct hypothesis ids")
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        known = {str(row.get("hypothesis_id") or "") for row in hypotheses}
        missing = [item for item in (left, right) if item not in known]
        if missing:
            raise MirrorScopeError(f"unknown competing hypothesis id(s): {', '.join(missing)}")
        hypotheses = mark_competing(hypotheses, left, right)
        event = {
            "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
            "run_id": self.task_id,
            "event_type": "hypothesis_competition_recorded",
            "hypotheses": [left, right],
            "reason": str(kwargs.get("reason") or kwargs.get("why") or ""),
            "delta": 0.0,
            "created_at": _now(),
        }
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        events.append(event)
        state["hypotheses"] = hypotheses[-100:]
        state["hypothesis_events"] = events[-200:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        self._save_investigation_state(state)
        for row in hypotheses:
            if str(row.get("hypothesis_id") or "") in {left, right}:
                scoped_event = dict(event)
                scoped_event["hypothesis_id"] = str(row.get("hypothesis_id") or "")
                self._persist_hypothesis_lifecycle(row, scoped_event)
        return self._raw_success(
            function_name="hypothesis_compete",
            reward=0.08,
            state="HYPOTHESIS_COMPETITION_RECORDED",
            competition={"hypothesis_a": left, "hypothesis_b": right},
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

    def _act_discriminating_test_add(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        left = str(kwargs.get("hypothesis_a") or kwargs.get("hypothesis_id_a") or "").strip()
        right = str(kwargs.get("hypothesis_b") or kwargs.get("hypothesis_id_b") or "").strip()
        if not left or not right or left == right:
            raise MirrorScopeError("discriminating_test_add requires two distinct hypothesis ids")
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        by_id = {str(row.get("hypothesis_id") or ""): row for row in hypotheses}
        if left not in by_id or right not in by_id:
            missing = [item for item in (left, right) if item not in by_id]
            raise MirrorScopeError(f"unknown discriminating-test hypothesis id(s): {', '.join(missing)}")
        raw_action = kwargs.get("action")
        if isinstance(raw_action, Mapping):
            action = dict(raw_action)
        else:
            action_name = str(kwargs.get("action_name") or kwargs.get("target_action") or "").strip()
            if not action_name:
                raise MirrorScopeError("discriminating_test_add requires action or action_name")
            raw_args = kwargs.get("args", {})
            action = {
                "action": action_name,
                "args": dict(raw_args) if isinstance(raw_args, Mapping) else {},
            }
        expected_if_a = str(kwargs.get("expected_if_a") or "").strip()
        expected_if_b = str(kwargs.get("expected_if_b") or "").strip()
        why = str(kwargs.get("why") or kwargs.get("why_discriminating") or "").strip()
        if not expected_if_a or not expected_if_b or not why:
            raise MirrorScopeError("discriminating_test_add requires expected_if_a, expected_if_b, and why")
        test = build_discriminating_test(
            hypothesis_a=by_id[left],
            hypothesis_b=by_id[right],
            action=action,
            expected_if_a=expected_if_a,
            expected_if_b=expected_if_b,
            why=why,
            test_id=str(kwargs.get("test_id") or ""),
        )
        test["discriminates_between"] = [left, right]
        test["expected_outcomes_by_hypothesis"] = {
            left: expected_if_a,
            right: expected_if_b,
        }
        try:
            test["expected_information_gain"] = float(kwargs.get("expected_information_gain", 0.42) or 0.42)
        except (TypeError, ValueError):
            test["expected_information_gain"] = 0.42
        tests = [dict(row) for row in list(state.get("discriminating_tests", []) or []) if isinstance(row, dict)]
        tests.append(test)
        event = {
            "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
            "run_id": self.task_id,
            "event_type": "discriminating_test_proposed",
            "hypotheses": [left, right],
            "discriminates_between": [left, right],
            "test_id": test["test_id"],
            "delta": 0.0,
            "created_at": test["created_at"],
        }
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        events.append(event)
        state["discriminating_tests"] = tests[-100:]
        state["hypothesis_events"] = events[-200:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        self._save_investigation_state(state)
        for row in (by_id[left], by_id[right]):
            scoped_event = dict(event)
            scoped_event["hypothesis_id"] = str(row.get("hypothesis_id") or "")
            self._persist_hypothesis_lifecycle(row, scoped_event)
        return self._raw_success(
            function_name="discriminating_test_add",
            reward=0.09,
            state="DISCRIMINATING_TEST_ADDED",
            discriminating_test=test,
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

    def _act_candidate_files_set(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        files = []
        for raw in _string_list(kwargs.get("files") or kwargs.get("paths")):
            source_path, relative = self._resolve_source_path(raw)
            if not source_path.exists() or not source_path.is_file():
                raise FileNotFoundError(str(source_path))
            files.append(relative)
        state = self._load_investigation_state()
        state["candidate_files"] = files
        state["candidate_reason"] = str(kwargs.get("reason") or "").strip()
        self._save_investigation_state(state)
        return self._raw_success(
            function_name="candidate_files_set",
            reward=0.1 if files else 0.02,
            state="CANDIDATE_FILES_SET",
            candidate_files=files,
            reason=state["candidate_reason"],
        )

    def _act_investigation_status(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        state = self._load_investigation_state()
        hypotheses = [dict(row) for row in list(state.get("hypotheses", []) or []) if isinstance(row, dict)]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        return self._raw_success(
            function_name="investigation_status",
            reward=0.03,
            state="INVESTIGATION_STATUS_READ",
            investigation=state,
            hypothesis_lifecycle=state["hypothesis_lifecycle"],
        )

