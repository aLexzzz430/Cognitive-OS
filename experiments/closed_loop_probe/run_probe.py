from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.runtime.hypothesis_lifecycle import (  # noqa: E402
    apply_hypothesis_evidence,
    hypothesis_lifecycle_summary,
    normalize_hypothesis,
)
from integrations.local_machine.task_adapter import LocalMachineSurfaceAdapter  # noqa: E402
from modules.local_mirror.mirror import compute_mirror_diff  # noqa: E402


VALID_VARIANTS = {"full", "no_posterior", "no_discriminating_experiment"}
FIXTURE_ROOT = REPO_ROOT / "fixtures" / "closed_loop_bug_repo"
REPORT_ROOT = REPO_ROOT / "experiments" / "closed_loop_probe" / "reports"


def _tool_action(name: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "kind": "call_tool",
        "function_name": str(name),
        "kwargs": dict(args or {}),
    }


def _action_signature(action: Mapping[str, Any]) -> str:
    name = str(action.get("function_name") or action.get("action") or "")
    args = action.get("kwargs") if isinstance(action.get("kwargs"), Mapping) else action.get("args")
    return json.dumps({"action": name, "args": dict(args or {})}, sort_keys=True, default=str)


def _action_name(action: Mapping[str, Any]) -> str:
    return str(action.get("function_name") or action.get("action") or "")


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def _line_text(lines: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(row.get("text", "")) for row in lines if isinstance(row, Mapping))


def _first_non_test_python(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    paths = []
    for row in entries:
        path = str(row.get("path") or "")
        if str(row.get("kind") or "") != "file" or not path.endswith(".py"):
            continue
        if path.startswith("tests/") or path.endswith("/__init__.py") or path == "__init__.py":
            continue
        paths.append(path)
    return sorted(dict.fromkeys(paths))


def _test_files(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    paths = []
    for row in entries:
        path = str(row.get("path") or "")
        if str(row.get("kind") or "") == "file" and path.startswith("tests/") and path.endswith(".py"):
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def _extract_failure_query(output: str) -> str:
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", output)
    blocked = {
        "File",
        "ValueError",
        "InvalidOperation",
        "Decimal",
        "assert",
        "where",
        "pytest",
        "tests",
        "return",
        "raise",
    }
    scored: dict[str, int] = {}
    for token in tokens:
        if token in blocked or token.startswith("test_"):
            continue
        score = 1
        lowered = token.lower()
        if "parse" in lowered:
            score += 8
        if "amount" in lowered:
            score += 6
        if "invoice" in lowered:
            score += 2
        scored[token] = max(scored.get(token, 0), score)
    if not scored:
        return "amount"
    return sorted(scored.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _failure_mentions_comma_currency(output: str) -> bool:
    return bool(re.search(r"[$]?\d{1,3},\d{3}", output))


def _make_single_line_patch(path: str, original_text: str, new_text: str) -> str:
    original_lines = original_text.splitlines()
    new_lines = new_text.splitlines()
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff) + "\n"


def _patch_for_discovered_amount_bug(path: str, content: str, failure_output: str) -> str:
    if not _failure_mentions_comma_currency(failure_output):
        return ""
    candidates = [
        ('.replace("$", "")', '.replace("$", "").replace(",", "")'),
        (".replace('$', '')", ".replace('$', '').replace(',', '')"),
    ]
    for old, new in candidates:
        if old in content and new not in content:
            return _make_single_line_patch(path, content, content.replace(old, new, 1))
    return ""


def _copy_fixture_to_temp() -> tuple[Path, Path, tempfile.TemporaryDirectory[str]]:
    temp = tempfile.TemporaryDirectory(prefix="conos_closed_loop_probe_")
    root = Path(temp.name)
    source_root = root / "source"
    mirror_root = root / "mirror"
    shutil.copytree(FIXTURE_ROOT, source_root, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return source_root, mirror_root, temp


class ClosedLoopProbe:
    def __init__(self, *, variant: str, max_ticks: int, model: str = "", llm_provider: str = "") -> None:
        if variant not in VALID_VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.max_ticks = max(1, int(max_ticks))
        self.model = str(model or "")
        self.llm_provider = str(llm_provider or "")
        self.source_root, self.mirror_root, self._temp = _copy_fixture_to_temp()
        self.adapter = LocalMachineSurfaceAdapter(
            instruction="Investigate the repository, locate the failure cause, make the smallest code change, and verify tests pass.",
            source_root=self.source_root,
            mirror_root=self.mirror_root,
            reset_mirror=True,
            allow_empty_exec=True,
            terminal_after_plan=False,
            default_command=None,
            default_command_timeout_seconds=30,
            task_id=f"closed_loop_probe_{variant}",
        )
        self.trace: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.action_sequence: list[dict[str, Any]] = []
        self.source_files: list[str] = []
        self.test_files: list[str] = []
        self.materialized_sources: set[str] = set()
        self.primary_test = ""
        self.secondary_tests: list[str] = []
        self.primary_run_ref = ""
        self.primary_failed = False
        self.failure_output = ""
        self.failure_query = ""
        self.grep_matches: list[dict[str, Any]] = []
        self.selected_source_file = ""
        self.selected_source_content = ""
        self.patch_text = ""
        self.patch_applied = False
        self.primary_verified_after_patch = False
        self.secondary_verified: set[str] = set()
        self.final_tests_passed = False
        self.final_run_ref = ""
        self.completed_before_verification = False
        self.plan_built = False
        self.residual_failure = ""

        self.hypotheses: list[dict[str, Any]] = []
        self.trace_only_hypotheses: list[dict[str, Any]] = []
        self.ranked_experiments: list[dict[str, Any]] = []
        self.hypothesis_events: list[dict[str, Any]] = []
        self.pending_posterior_events: list[dict[str, Any]] = []
        self.experiment_selected = False
        self.top_experiment_selected = False
        self.posterior_changed_after_test_failure = False
        self.posterior_changed_after_success = False
        self.next_action_changed_after_posterior_update = False
        self._before_after_posterior_actions: dict[str, Any] = {}
        self._source_evidence_applied = False
        self._success_evidence_applied = False
        self._non_discriminating_read_done = False
        self._non_discriminating_probe_run_ref = ""
        self._non_discriminating_probe_output_read = False

    def close(self) -> None:
        self._temp.cleanup()

    def _available_actions(self) -> list[str]:
        return [tool.name for tool in self.adapter.observe().available_tools]

    def _workspace_hypotheses(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.hypotheses]

    def _posterior_summary(self) -> dict[str, Any]:
        return hypothesis_lifecycle_summary(self._workspace_hypotheses())

    def _trace_only_posterior_summary(self) -> dict[str, Any]:
        return hypothesis_lifecycle_summary(self.trace_only_hypotheses)

    def _record_cognitive_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.events.append({"event_type": event_type, "payload": _jsonable(dict(payload))})

    def _ensure_hypotheses_and_experiments(self) -> None:
        if self.hypotheses:
            return
        query = self.failure_query or _extract_failure_query(self.failure_output)
        self.failure_query = query
        h1 = normalize_hypothesis(
            hypothesis_id="hyp_parser_normalization",
            run_id=f"closed_loop_probe_{self.variant}",
            task_family="local_machine",
            family="bug_localization",
            claim="The failing behavior is caused by input normalization in the amount parsing path.",
            confidence=0.5,
            evidence_refs=[f"run:{self.primary_run_ref}"] if self.primary_run_ref else [],
            predictions={
                "repo_grep": f"Searching for {query} should reveal the implementation used by the failing test.",
                "file_read": "The implementation should normalize currency syntax before Decimal construction.",
            },
            falsifiers={"source_evidence": "The selected implementation already strips comma separators."},
        )
        h2 = normalize_hypothesis(
            hypothesis_id="hyp_invoice_logic",
            run_id=f"closed_loop_probe_{self.variant}",
            task_family="local_machine",
            family="bug_localization",
            claim="The failing behavior is caused by invoice aggregation or threshold comparison logic.",
            confidence=0.5,
            evidence_refs=[f"run:{self.primary_run_ref}"] if self.primary_run_ref else [],
            predictions={
                "run_test": "Invoice-focused tests should fail independently of parser-focused tests.",
                "file_read": "Invoice code should contain the local defect.",
            },
            falsifiers={"failure_output": "A stack trace ending inside parser normalization weakens this hypothesis."},
        )
        self.hypotheses = [h1, h2]
        self.trace_only_hypotheses = deepcopy(self.hypotheses)
        self._record_cognitive_event("hypotheses_created", {"hypotheses": self.hypotheses})
        self._build_ranked_experiments()

    def _build_ranked_experiments(self) -> None:
        if not self.hypotheses:
            return
        query = self.failure_query or "amount"
        invoice_test = next((path for path in self.test_files if "invoice" in Path(path).name), "")
        experiments = [
            {
                "experiment_id": "exp_grep_failure_symbol",
                "action": _tool_action("repo_grep", {"root": ".", "query": query, "globs": ["*.py"], "max_matches": 50}),
                "hypothesis_a": "hyp_parser_normalization",
                "hypothesis_b": "hyp_invoice_logic",
                "expected_if_a": "The failing symbol appears in a parser implementation file.",
                "expected_if_b": "The failing symbol is absent or only appears in tests.",
                "base_score": 0.36,
                "expected_information_gain": 0.82,
            },
            {
                "experiment_id": "exp_run_invoice_test",
                "action": _tool_action("run_test", {"target": invoice_test or ".", "timeout_seconds": 30}),
                "hypothesis_a": "hyp_parser_normalization",
                "hypothesis_b": "hyp_invoice_logic",
                "expected_if_a": "Invoice tests fail through the same parser path.",
                "expected_if_b": "Invoice tests reveal a distinct invoice stack.",
                "base_score": 0.34,
                "expected_information_gain": 0.62,
            },
            {
                "experiment_id": "exp_read_first_source",
                "action": _tool_action("file_read", {"path": self.source_files[0] if self.source_files else ".", "start_line": 1, "end_line": 220}),
                "hypothesis_a": "hyp_parser_normalization",
                "hypothesis_b": "hyp_invoice_logic",
                "expected_if_a": "Parser or normalization code is visible.",
                "expected_if_b": "The first source file points elsewhere.",
                "base_score": 0.22,
                "expected_information_gain": 0.28,
            },
        ]
        use_bonus = self.variant != "no_discriminating_experiment"
        for row in experiments:
            info_gain = float(row.get("expected_information_gain", 0.0) or 0.0)
            bonus = round(info_gain * 0.45, 6) if use_bonus else 0.0
            row["score"] = round(float(row["base_score"]) + bonus, 6)
            if use_bonus:
                row["discriminating_experiment_score"] = bonus
        self.ranked_experiments = sorted(experiments, key=lambda item: (-float(item.get("score", 0.0)), str(item.get("experiment_id", ""))))
        self._record_cognitive_event("ranked_discriminating_experiments", {"experiments": self.ranked_experiments})

    def _hypothesis_index(self, hypothesis_id: str) -> int:
        for index, row in enumerate(self.hypotheses):
            if str(row.get("hypothesis_id") or "") == hypothesis_id:
                return index
        return -1

    def _apply_posterior_evidence(
        self,
        hypothesis_id: str,
        *,
        signal: str,
        evidence_refs: Sequence[str],
        strength: float,
        rationale: str,
        stage: str,
    ) -> None:
        index = self._hypothesis_index(hypothesis_id)
        if index < 0:
            return
        updated, event = apply_hypothesis_evidence(
            self.hypotheses[index],
            signal=signal,
            evidence_refs=evidence_refs,
            strength=strength,
            rationale=rationale,
        )
        event["stage"] = stage
        event["writeback_blocked"] = self.variant == "no_posterior"
        self.pending_posterior_events.append(dict(event))
        self.hypothesis_events.append(dict(event))
        if self.variant == "no_posterior":
            trace_index = next(
                (idx for idx, row in enumerate(self.trace_only_hypotheses) if row.get("hypothesis_id") == hypothesis_id),
                -1,
            )
            if trace_index >= 0:
                trace_row, _ = apply_hypothesis_evidence(
                    self.trace_only_hypotheses[trace_index],
                    signal=signal,
                    evidence_refs=evidence_refs,
                    strength=strength,
                    rationale=rationale,
                )
                self.trace_only_hypotheses[trace_index] = trace_row
            return
        self.hypotheses[index] = updated

    def _rank_candidate_actions(self, *, include_patch: bool = True) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if self.ranked_experiments and not self.experiment_selected:
            candidates.extend(deepcopy(self.ranked_experiments))
        if self.selected_source_file and not self.selected_source_content:
            candidates.append(
                {
                    "experiment_id": "read_selected_source",
                    "action": _tool_action("file_read", {"path": self.selected_source_file, "start_line": 1, "end_line": 240}),
                    "score": 0.58,
                    "base_score": 0.58,
                    "expected_information_gain": 0.5,
                }
            )
        if include_patch and self.patch_text and not self.patch_applied:
            leading = self._posterior_summary().get("leading_hypothesis_id", "")
            leading_posterior = float(self._posterior_summary().get("leading_posterior", 0.0) or 0.0)
            score = 0.9 if leading == "hyp_parser_normalization" and leading_posterior >= 0.62 else 0.25
            candidates.append(
                {
                    "experiment_id": "apply_minimal_patch",
                    "action": _tool_action(
                        "apply_patch",
                        {
                            "patch": self.patch_text,
                            "max_files": 1,
                            "max_hunks": 2,
                            "evidence_refs": [f"run:{self.primary_run_ref}", f"file:{self.selected_source_file}:1-240"],
                        },
                    ),
                    "score": score,
                    "base_score": score,
                    "expected_information_gain": 0.0,
                }
            )
        return sorted(candidates, key=lambda item: (-float(item.get("score", 0.0)), str(item.get("experiment_id", ""))))

    def _apply_failure_evidence_once(self) -> None:
        if not self.failure_output or self.posterior_changed_after_test_failure:
            return
        self._ensure_hypotheses_and_experiments()
        before = {row["hypothesis_id"]: float(row.get("posterior", 0.0) or 0.0) for row in self.hypotheses}
        refs = [f"run:{self.primary_run_ref}"] if self.primary_run_ref else []
        parser_signal = "support" if (_failure_mentions_comma_currency(self.failure_output) or self.failure_query) else "neutral"
        invoice_signal = "contradict" if parser_signal == "support" else "neutral"
        self._apply_posterior_evidence(
            "hyp_parser_normalization",
            signal=parser_signal,
            evidence_refs=refs,
            strength=0.35,
            rationale="The failing validation output points at currency input normalization.",
            stage="test_failure",
        )
        self._apply_posterior_evidence(
            "hyp_invoice_logic",
            signal=invoice_signal,
            evidence_refs=refs,
            strength=0.25,
            rationale="The first failure is already explained before invoice aggregation is inspected.",
            stage="test_failure",
        )
        after = {row["hypothesis_id"]: float(row.get("posterior", 0.0) or 0.0) for row in self.hypotheses}
        self.posterior_changed_after_test_failure = self.variant != "no_posterior" and before != after

    def _apply_source_evidence_once(self) -> None:
        if self._source_evidence_applied or not self.selected_source_content:
            return
        before_action = self._baseline_action_before_source_posterior()
        before = {row["hypothesis_id"]: float(row.get("posterior", 0.0) or 0.0) for row in self.hypotheses}
        refs = [f"file:{self.selected_source_file}:1-240"] if self.selected_source_file else []
        parser_supported = '.replace("$", "")' in self.selected_source_content and ".replace(\",\", \"\")" not in self.selected_source_content
        self._apply_posterior_evidence(
            "hyp_parser_normalization",
            signal="support" if parser_supported else "neutral",
            evidence_refs=refs,
            strength=0.25,
            rationale="The selected source shows currency normalization without comma handling.",
            stage="source_read",
        )
        self._apply_posterior_evidence(
            "hyp_invoice_logic",
            signal="contradict" if parser_supported else "neutral",
            evidence_refs=refs,
            strength=0.2,
            rationale="The source evidence localizes the defect away from invoice aggregation.",
            stage="source_read",
        )
        after = {row["hypothesis_id"]: float(row.get("posterior", 0.0) or 0.0) for row in self.hypotheses}
        self._source_evidence_applied = True
        if self.variant != "no_posterior" and before != after:
            self.patch_text = _patch_for_discovered_amount_bug(
                self.selected_source_file,
                self.selected_source_content,
                self.failure_output,
            )
            after_rank = self._rank_candidate_actions(include_patch=True)
            after_action = dict(after_rank[0].get("action", {})) if after_rank else {}
            self._before_after_posterior_actions = {
                "stage": "source_read",
                "before_action": before_action,
                "after_action": after_action,
                "before_posterior": before,
                "after_posterior": after,
            }
            self.next_action_changed_after_posterior_update = bool(
                before_action and after_action and _action_signature(before_action) != _action_signature(after_action)
            )

    def _baseline_action_before_source_posterior(self) -> dict[str, Any]:
        if self.variant == "no_discriminating_experiment":
            target = next((path for path in self.test_files if path not in {self.primary_test}), self.primary_test or ".")
            return _tool_action("run_test", {"target": target or ".", "timeout_seconds": 30})
        if self.variant == "no_posterior":
            target = next((path for path in self.test_files if path not in {self.primary_test}), self.primary_test or ".")
            return _tool_action("run_test", {"target": target or ".", "timeout_seconds": 30})
        return _tool_action("file_read", {"path": self.source_files[-1] if self.source_files else ".", "start_line": 1, "end_line": 220})

    def _apply_success_evidence_once(self) -> None:
        if self._success_evidence_applied or not self.primary_verified_after_patch:
            return
        before = {row["hypothesis_id"]: float(row.get("posterior", 0.0) or 0.0) for row in self.hypotheses}
        refs = [f"run:{self.final_run_ref or self.primary_run_ref}"]
        self._apply_posterior_evidence(
            "hyp_parser_normalization",
            signal="support",
            evidence_refs=refs,
            strength=0.2,
            rationale="The targeted validation passed after the parser-normalization patch.",
            stage="test_success",
        )
        after = {row["hypothesis_id"]: float(row.get("posterior", 0.0) or 0.0) for row in self.hypotheses}
        self.posterior_changed_after_success = self.variant != "no_posterior" and before != after
        self._success_evidence_applied = True

    def _choose_next_action(self) -> dict[str, Any]:
        if not self.source_files:
            return _tool_action("repo_tree", {"path": ".", "depth": 3, "max_entries": 200, "exclude": [".git", "__pycache__", ".pytest_cache", ".venv"]})
        for path in self.source_files:
            if path not in self.materialized_sources:
                return _tool_action("run_typecheck", {"target": path, "timeout_seconds": 30})
        if not self.primary_run_ref:
            return _tool_action("run_test", {"target": self.primary_test or (self.test_files[0] if self.test_files else "."), "timeout_seconds": 30})
        if self.primary_failed and not self.failure_output:
            return _tool_action("read_test_failure", {"run_ref": self.primary_run_ref, "max_chars": 12000})
        if self.failure_output and not self.hypotheses:
            self._ensure_hypotheses_and_experiments()
        if self.failure_output and not self.posterior_changed_after_test_failure and self.variant != "no_posterior":
            self._apply_failure_evidence_once()
        elif self.failure_output and self.variant == "no_posterior" and not self.hypothesis_events:
            self._apply_failure_evidence_once()
        if self.failure_output and not self.experiment_selected:
            ranked = self._rank_candidate_actions(include_patch=False)
            if self.variant == "no_discriminating_experiment" and not self._non_discriminating_read_done:
                invoice_file = next((path for path in self.source_files if "invoice" in Path(path).name), self.source_files[0])
                return _tool_action("file_read", {"path": invoice_file, "start_line": 1, "end_line": 220})
            if ranked:
                return dict(ranked[0]["action"])
        if self.grep_matches and not self.selected_source_file:
            self.selected_source_file = self._select_source_from_grep()
        if self.selected_source_file and not self.selected_source_content:
            return _tool_action("file_read", {"path": self.selected_source_file, "start_line": 1, "end_line": 240})
        if self.selected_source_content and not self._source_evidence_applied:
            self._apply_source_evidence_once()
        if self.selected_source_content and self.patch_text and not self.patch_applied and self.variant != "no_posterior":
            ranked = self._rank_candidate_actions(include_patch=True)
            if ranked:
                return dict(ranked[0]["action"])
        if self.variant == "no_posterior" and self.selected_source_content and not self.final_run_ref:
            return _tool_action("run_test", {"target": ".", "timeout_seconds": 30})
        if self.patch_applied and not self.primary_verified_after_patch:
            return _tool_action("run_test", {"target": self.primary_test or ".", "timeout_seconds": 30})
        if self.primary_verified_after_patch:
            self._apply_success_evidence_once()
            for path in self.secondary_tests:
                if path not in self.secondary_verified:
                    return _tool_action("run_test", {"target": path, "timeout_seconds": 30})
            if not self.final_tests_passed and not self.final_run_ref:
                return _tool_action("run_test", {"target": ".", "timeout_seconds": 30})
            if self.final_tests_passed and not self.plan_built:
                return _tool_action("mirror_plan", {})
        if self.variant == "no_discriminating_experiment" and self.failure_output:
            if self._non_discriminating_read_done and not self._non_discriminating_probe_run_ref:
                target = next((path for path in self.secondary_tests if path != self.primary_test), self.primary_test or ".")
                return _tool_action("run_test", {"target": target, "timeout_seconds": 30})
            if self._non_discriminating_probe_run_ref and not self._non_discriminating_probe_output_read:
                return _tool_action("read_test_failure", {"run_ref": self._non_discriminating_probe_run_ref, "max_chars": 12000})
            if not any(row.get("function_name") == "repo_grep" for row in self.action_sequence):
                return _tool_action("repo_grep", {"root": ".", "query": self.failure_query or "amount", "globs": ["*.py"], "max_matches": 50})
            if self.grep_matches and not self.selected_source_content:
                self.selected_source_file = self._select_source_from_grep()
                return _tool_action("file_read", {"path": self.selected_source_file, "start_line": 1, "end_line": 240})
        return _tool_action("investigation_status", {})

    def _select_source_from_grep(self) -> str:
        for row in self.grep_matches:
            path = str(row.get("path") or "")
            if path.endswith(".py") and not path.startswith("tests/"):
                return path
        return self.source_files[0] if self.source_files else ""

    def _after_action(self, action: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
        name = _action_name(action)
        if name == "repo_tree" and raw.get("success"):
            entries = list(raw.get("entries", []) or [])
            self.source_files = _first_non_test_python(entries)
            self.test_files = _test_files(entries)
            self.primary_test = next((path for path in self.test_files if "amount" in Path(path).name), self.test_files[0] if self.test_files else ".")
            self.secondary_tests = [path for path in self.test_files if path != self.primary_test]
        elif name == "run_typecheck":
            target = str(dict(action.get("kwargs", {}) or {}).get("target") or "")
            if raw.get("success") and target:
                self.materialized_sources.add(target)
        elif name == "run_test":
            target = str(dict(action.get("kwargs", {}) or {}).get("target") or ".")
            run_ref = str(raw.get("run_ref") or "")
            if target == ".":
                self.final_run_ref = run_ref
                self.final_tests_passed = bool(raw.get("success"))
                if not self.final_tests_passed:
                    self.residual_failure = str(raw.get("stdout_tail", "") or "")[-3000:] + str(raw.get("stderr_tail", "") or "")[-3000:]
            elif target == self.primary_test and not self.patch_applied:
                self.primary_run_ref = run_ref
                self.primary_failed = not bool(raw.get("success"))
                if not self.primary_failed:
                    self.completed_before_verification = True
            elif target == self.primary_test and self.patch_applied:
                self.primary_verified_after_patch = bool(raw.get("success"))
                if not self.primary_verified_after_patch:
                    self.residual_failure = str(raw.get("stdout_tail", "") or "")[-3000:] + str(raw.get("stderr_tail", "") or "")[-3000:]
            elif target in self.secondary_tests and bool(raw.get("success")):
                self.secondary_verified.add(target)
            elif self.variant == "no_discriminating_experiment" and target in self.secondary_tests:
                self._non_discriminating_probe_run_ref = run_ref
        elif name in {"read_test_failure", "read_run_output"} and raw.get("success"):
            self.failure_output = f"{raw.get('stdout', '')}\n{raw.get('stderr', '')}"
            self.failure_query = _extract_failure_query(self.failure_output)
            requested_run_ref = str(dict(action.get("kwargs", {}) or {}).get("run_ref") or "")
            if requested_run_ref and requested_run_ref == self._non_discriminating_probe_run_ref:
                self._non_discriminating_probe_output_read = True
        elif name == "repo_grep" and raw.get("success"):
            self.grep_matches = [dict(row) for row in list(raw.get("matches", []) or []) if isinstance(row, Mapping)]
            self.experiment_selected = True
            if (
                self.variant != "no_discriminating_experiment"
                and self.ranked_experiments
                and _action_signature(action) == _action_signature(self.ranked_experiments[0].get("action", {}))
            ):
                self.top_experiment_selected = True
        elif name == "file_read" and raw.get("success"):
            path = str(raw.get("path") or "")
            content = str(raw.get("content") or _line_text(raw.get("lines", []) or []))
            if path and path not in self.selected_source_file and not path.startswith("tests/"):
                if self.variant == "no_discriminating_experiment" and "invoice" in Path(path).name:
                    self._non_discriminating_read_done = True
                    self.experiment_selected = True
                    self._record_cognitive_event("non_discriminating_read", {"path": path})
                else:
                    self.selected_source_file = path
            if path == self.selected_source_file:
                self.selected_source_content = content
        elif name == "apply_patch" and raw.get("success"):
            self.patch_applied = True
        elif name == "mirror_plan" and raw.get("success"):
            self.plan_built = True

    def run(self) -> dict[str, Any]:
        self.adapter.reset()
        for tick in range(self.max_ticks):
            available = self._available_actions()
            action = self._choose_next_action()
            pending_events = list(self.pending_posterior_events)
            self.pending_posterior_events = []
            result = self.adapter.act(action)
            raw = dict(result.raw or {})
            self.action_sequence.append(
                {
                    "tick": tick,
                    "function_name": _action_name(action),
                    "kwargs": _jsonable(dict(action.get("kwargs", {}) or {})),
                    "success": bool(raw.get("success")),
                    "state": str(raw.get("state") or ""),
                }
            )
            tick_record = {
                "tick": tick,
                "selected_action": self.action_sequence[-1],
                "available_actions": available,
                "competing_hypotheses": _jsonable(self._workspace_hypotheses()),
                "ranked_discriminating_experiments": _jsonable(self.ranked_experiments),
                "posterior_summary": _jsonable(self._posterior_summary()),
                "hypothesis_posterior_events": _jsonable(pending_events),
                "run_test_output": self._run_output_excerpt(raw),
                "raw_state": str(raw.get("state") or ""),
                "raw_success": bool(raw.get("success")),
            }
            self.trace.append(tick_record)
            self._after_action(action, raw)
            if self.plan_built or (self.final_tests_passed and tick >= self.max_ticks - 1):
                break
        return self.report()

    @staticmethod
    def _run_output_excerpt(raw: Mapping[str, Any]) -> dict[str, Any]:
        if raw.get("function_name") not in {"run_test", "read_run_output", "read_test_failure"}:
            return {}
        return {
            "run_ref": str(raw.get("run_ref") or ""),
            "returncode": raw.get("returncode"),
            "stdout": str(raw.get("stdout") or raw.get("stdout_tail") or "")[-3000:],
            "stderr": str(raw.get("stderr") or raw.get("stderr_tail") or "")[-3000:],
        }

    def _final_diff(self) -> list[dict[str, Any]]:
        rows = []
        for entry in compute_mirror_diff(self.source_root, self.mirror_root):
            payload = entry.to_dict()
            if payload.get("status") in {"modified", "added", "removed_in_mirror"}:
                rows.append(payload)
        return rows

    def _final_diff_summary(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        changed_paths = []
        for row in rows:
            status = str(row.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
            changed_paths.append(str(row.get("relative_path") or ""))
        return {
            "changed_paths": changed_paths,
            "status_counts": counts,
            "patches": [
                {
                    "path": str(row.get("relative_path") or ""),
                    "patch_sha256": str(row.get("patch_sha256") or ""),
                    "text_patch": str(row.get("text_patch") or ""),
                }
                for row in rows
            ],
        }

    def report(self) -> dict[str, Any]:
        final_diff = self._final_diff()
        success = bool(self.final_tests_passed and self.patch_applied and not self.completed_before_verification)
        residual = self.residual_failure
        if not success and not residual:
            residual = "probe did not reach final passing verification within max_ticks"
        return {
            "schema_version": "conos.closed_loop_reality_probe/v1",
            "variant": self.variant,
            "model": self.model,
            "llm_provider": self.llm_provider,
            "llm_controlled": False,
            "source_fixture": str(FIXTURE_ROOT),
            "ephemeral_source_root": str(self.source_root),
            "ephemeral_mirror_root": str(self.mirror_root),
            "success": success,
            "ticks": len(self.trace),
            "action_sequence": self.action_sequence,
            "hypotheses_created": len(self.hypotheses),
            "experiments_created": len(self.ranked_experiments),
            "top_experiment_selected": bool(self.top_experiment_selected),
            "posterior_changed_after_test_failure": bool(self.posterior_changed_after_test_failure),
            "posterior_changed_after_success": bool(self.posterior_changed_after_success),
            "next_action_changed_after_posterior_update": bool(self.next_action_changed_after_posterior_update),
            "completed_before_verification": bool(self.completed_before_verification),
            "final_tests_passed": bool(self.final_tests_passed),
            "final_diff_summary": self._final_diff_summary(final_diff),
            "final_diff": final_diff,
            "residual_failure": residual,
            "competing_hypotheses": _jsonable(self._workspace_hypotheses()),
            "ranked_discriminating_experiments": _jsonable(self.ranked_experiments),
            "posterior_summary": _jsonable(self._posterior_summary()),
            "trace_only_posterior_summary": _jsonable(self._trace_only_posterior_summary()),
            "trace_only_hypotheses": _jsonable(self.trace_only_hypotheses),
            "hypothesis_posterior_events": _jsonable(self.hypothesis_events),
            "posterior_action_change_evidence": _jsonable(self._before_after_posterior_actions),
            "events": _jsonable(self.events),
            "ticks_trace": _jsonable(self.trace),
        }


def run_probe(
    *,
    variant: str,
    max_ticks: int,
    model: str = "",
    llm_provider: str = "",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    probe = ClosedLoopProbe(variant=variant, max_ticks=max_ticks, model=model, llm_provider=llm_provider)
    try:
        report = probe.run()
    finally:
        probe.close()
    output_path = Path(report_path) if report_path is not None else REPORT_ROOT / f"{variant}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the minimal Closed Loop Reality Probe Pack.")
    parser.add_argument("--variant", choices=sorted(VALID_VARIANTS), default="full")
    parser.add_argument("--max-ticks", type=int, default=12)
    parser.add_argument("--model", default="")
    parser.add_argument("--llm-provider", default="")
    args = parser.parse_args(argv)
    report = run_probe(
        variant=args.variant,
        max_ticks=args.max_ticks,
        model=args.model,
        llm_provider=args.llm_provider,
    )
    print(
        json.dumps(
            {
                "variant": report["variant"],
                "success": report["success"],
                "ticks": report["ticks"],
                "final_tests_passed": report["final_tests_passed"],
                "top_experiment_selected": report["top_experiment_selected"],
                "posterior_changed_after_test_failure": report["posterior_changed_after_test_failure"],
                "next_action_changed_after_posterior_update": report["next_action_changed_after_posterior_update"],
                "report_path": str(REPORT_ROOT / f"{args.variant}.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["success"] or args.variant != "full" else 1


if __name__ == "__main__":
    raise SystemExit(main())
