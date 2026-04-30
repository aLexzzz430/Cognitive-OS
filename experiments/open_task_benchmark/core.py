from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Sequence

try:
    from modules.llm.budget import amplification_efficiency
except Exception:  # pragma: no cover - direct script fallback in unusual checkouts.
    amplification_efficiency = None  # type: ignore[assignment]


OPEN_TASK_BENCHMARK_CONFIG_VERSION = "conos.open_task_benchmark.config/v1"
OPEN_TASK_TASK_PACKAGE_VERSION = "conos.open_task_benchmark.task_package/v1"
OPEN_TASK_EXECUTION_CONTRACT_VERSION = "conos.open_task_benchmark.execution_contract/v1"
OPEN_TASK_RESULT_VERSION = "conos.open_task_benchmark.result/v1"
OPEN_TASK_BENCHMARK_SUMMARY_VERSION = "conos.open_task_benchmark.summary/v1"

LEAKY_PROJECT_KEYS = frozenset(
    {
        "answer",
        "expected_patch",
        "expected_diff",
        "hidden_solution",
        "solution",
        "target_file",
        "true_bug_file",
        "true_root_cause",
    }
)
NON_SOURCE_PATTERNS = (
    ".DS_Store",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    "node_modules/",
    ".venv/",
    "uv.lock",
    "poetry.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
)
LOCKFILE_NAMES = frozenset({"uv.lock", "poetry.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"})


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _safe_slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:80] or "task"


def _stable_short_hash(value: Any, *, size: int = 10) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:size]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(Path(path))
    projects = _as_list(config.get("projects"))
    if not projects:
        raise ValueError("benchmark config must contain at least one project")
    return config


def assert_project_metadata_is_leak_free(project: Mapping[str, Any]) -> None:
    leaked = sorted(key for key in project.keys() if str(key) in LEAKY_PROJECT_KEYS)
    if leaked:
        raise ValueError(f"project metadata leaks answer fields: {', '.join(leaked)}")
    for key, value in project.items():
        if str(key).startswith("hidden_") or str(key).startswith("solution_"):
            raise ValueError(f"project metadata leaks answer field: {key}")
        if isinstance(value, Mapping):
            assert_project_metadata_is_leak_free(value)


def _project_task_id(project: Mapping[str, Any], index: int) -> str:
    explicit = str(project.get("project_id") or "").strip()
    if explicit:
        return _safe_slug(explicit)
    repo = str(project.get("repo_url") or project.get("source_path") or f"project-{index}")
    return f"{_safe_slug(repo)}-{_stable_short_hash(repo)}"


def _select_projects(config: Mapping[str, Any], *, seed: int | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    projects = [dict(row) for row in _as_list(config.get("projects")) if isinstance(row, Mapping)]
    rng = random.Random(int(seed if seed is not None else config.get("seed", 0) or 0))
    rng.shuffle(projects)
    if limit is not None:
        projects = projects[: max(0, int(limit))]
    return projects


def _verifier_for_project(project: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    defaults = _as_dict(config.get("verifier_defaults"))
    command = project.get("verifier_command") or defaults.get("command") or ["python", "-m", "pytest", "-q"]
    if isinstance(command, str):
        command = command.split()
    return {
        "schema_version": "conos.open_task_benchmark.verifier/v1",
        "command": list(command),
        "timeout_seconds": int(project.get("verifier_timeout_seconds") or defaults.get("timeout_seconds") or 300),
        "must_run_full_suite": True,
        "tests_may_be_modified": False,
    }


def _budget_for_project(project: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    defaults = {
        "max_llm_calls": 12,
        "max_prompt_tokens": 60000,
        "max_completion_tokens": 12000,
        "max_wall_clock_seconds": 1800,
        "max_retry_count": 2,
        "escalation_allowed": True,
    }
    defaults.update(_as_dict(config.get("budget_defaults")))
    defaults.update(_as_dict(project.get("budget")))
    return {
        "max_llm_calls": int(defaults.get("max_llm_calls", 12) or 12),
        "max_prompt_tokens": int(defaults.get("max_prompt_tokens", 60000) or 60000),
        "max_completion_tokens": int(defaults.get("max_completion_tokens", 12000) or 12000),
        "max_wall_clock_seconds": int(defaults.get("max_wall_clock_seconds", 1800) or 1800),
        "max_retry_count": int(defaults.get("max_retry_count", 2) or 2),
        "escalation_allowed": bool(defaults.get("escalation_allowed", True)),
    }


def _execution_contract_for_project(
    project: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    task_id: str,
    verifier: Mapping[str, Any],
) -> dict[str, Any]:
    repo_url = str(project.get("repo_url") or "").strip()
    revision = str(project.get("revision") or "").strip()
    return {
        "schema_version": OPEN_TASK_EXECUTION_CONTRACT_VERSION,
        "task_id": task_id,
        "project_id": str(project.get("project_id") or task_id),
        "source_acquisition": {
            "mode": "remote_git_reference" if repo_url else "local_source_reference",
            "repo_url": repo_url,
            "revision": revision,
            "network_required_to_execute": bool(repo_url),
            "package_only_is_offline": True,
            "answer_leak_check": "passed",
        },
        "allowed_actions": [
            "clone_or_prepare_source",
            "repo_tree",
            "repo_find",
            "repo_grep",
            "file_read",
            "run_test",
            "read_test_failure",
            "propose_bounded_patch",
            "apply_patch_in_sandbox_or_vm",
            "run_full_verifier",
            "rollback_failed_patch",
            "write_audit_report",
        ],
        "forbidden_actions": [
            "modify_tests",
            "sync_back_without_verified_patch",
            "use_hidden_metadata",
            "start_unapproved_fallback_patch_after_timeout",
            "write_credentials_to_repo",
        ],
        "budget": _budget_for_project(project, config),
        "cost_policy": {
            "deterministic_first": True,
            "small_model_allowed_for": ["classification", "short_summary", "schema_rewrite", "fixed_option_ranking"],
            "strong_model_allowed_for": ["root_cause_analysis", "patch_design", "ambiguous_spec_review"],
            "record_cost_by_layer": True,
            "record_provider_model_route": True,
        },
        "context_policy": {
            "raw_diff_is_object_layer_evidence": True,
            "retrieve_relevant_files_only": True,
            "drop_raw_model_thinking": True,
            "store_reasoning_state": True,
        },
        "failure_recovery_policy": [
            {
                "failure_type": "model_timeout",
                "expected_behavior": "return_structured_timeout_or_audited_escalation",
            },
            {
                "failure_type": "format_error",
                "expected_behavior": "normalize_with_output_adapter_before_retry",
            },
            {
                "failure_type": "invalid_kwargs",
                "expected_behavior": "repair_or_replan_before_execution",
            },
            {
                "failure_type": "verifier_failed",
                "expected_behavior": "read_failure_then_retry_or_rollback",
            },
            {
                "failure_type": "evidence_insufficient",
                "expected_behavior": "needs_human_review_without_patch",
            },
        ],
        "required_evidence": [
            "commands_run",
            "files_read",
            "changed_paths",
            "final_diff_summary",
            "final_verifier_result",
            "llm_route_usage_or_no_llm_reason",
            "cost_or_unknown_cost_reason",
            "failure_recovery_events",
        ],
        "verifier_policy": dict(verifier),
        "completion_gate": {
            "final_pytest_passed_required": True,
            "tests_modified_allowed": False,
            "source_patch_required_unless_needs_human_review": True,
            "non_source_artifacts_penalized_not_failed": True,
        },
    }


def _runbook_text(metadata: Mapping[str, Any], verifier: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    command = " ".join(str(part) for part in _as_list(verifier.get("command")))
    budget = _as_dict(contract.get("budget"))
    return "\n".join(
        [
            "# Open Task Runbook",
            "",
            f"Task ID: {metadata.get('task_id')}",
            f"Project: {metadata.get('project_id')}",
            f"Repository: {metadata.get('repo_url') or '<provided in metadata.json>'}",
            f"Revision: {metadata.get('revision') or 'default'}",
            "",
            "## Required Flow",
            "1. Prepare a clean workspace from the public source reference.",
            "2. Inspect the repository and tests before patching.",
            "3. Keep all changes inside the execution sandbox or VM boundary.",
            "4. Apply only a bounded source patch unless the task explicitly allows more.",
            "5. Run the verifier and record the result.",
            "6. Roll back failed patches or mark `needs_human_review=true` if evidence is insufficient.",
            "",
            "## Verifier",
            f"`{command}`",
            "",
            "## Budget",
            f"- max_llm_calls: {budget.get('max_llm_calls')}",
            f"- max_wall_clock_seconds: {budget.get('max_wall_clock_seconds')}",
            f"- escalation_allowed: {budget.get('escalation_allowed')}",
            "",
            "## Evidence To Return",
            "- commands_run",
            "- files_read",
            "- changed_paths",
            "- final_diff_summary",
            "- final_pytest_passed",
            "- llm_route_usage or no_llm_reason",
            "- cost or unknown_cost_reason",
            "- failure_recovery_events",
        ]
    )


def _task_prompt(project: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    repo_url = str(project.get("repo_url") or "").strip()
    revision = str(project.get("revision") or "default").strip()
    goal = str(
        project.get("task_goal")
        or config.get("default_task_goal")
        or "Investigate the repository, identify one small verifiable bug or reliability improvement, make a minimal source patch, and run the verifier."
    ).strip()
    rules = [
        "Do not modify tests unless the task explicitly asks for it.",
        "Prefer the smallest source-only patch that makes the verifier pass.",
        "Record commands run, files read, files changed, final diff summary, and cost if available.",
        "If evidence is insufficient for a safe patch, report needs_human_review instead of guessing.",
    ]
    return "\n".join(
        [
            "# Open Task Benchmark Package",
            "",
            f"Repository: {repo_url or '<provided in metadata.json>'}",
            f"Revision: {revision}",
            "",
            "## Objective",
            goal,
            "",
            "## Rules",
            *[f"- {rule}" for rule in rules],
        ]
    )


def create_task_packages(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    seed: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    tasks_root = output_root / "tasks"
    packages: list[dict[str, Any]] = []
    for index, project in enumerate(_select_projects(config, seed=seed, limit=limit), start=1):
        assert_project_metadata_is_leak_free(project)
        task_id = _project_task_id(project, index)
        task_dir = tasks_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        verifier = _verifier_for_project(project, config)
        metadata = {
            "schema_version": OPEN_TASK_TASK_PACKAGE_VERSION,
            "task_id": task_id,
            "project_id": str(project.get("project_id") or task_id),
            "repo_url": str(project.get("repo_url") or ""),
            "revision": str(project.get("revision") or ""),
            "language": str(project.get("language") or ""),
            "benchmark_source": "open_task_benchmark",
            "answer_leak_check": "passed",
            "source_mode": "remote_git_reference" if project.get("repo_url") else "local_source_reference",
            "expected_lockfiles": list(project.get("expected_lockfiles") or []),
        }
        result_template = {
            "schema_version": OPEN_TASK_RESULT_VERSION,
            "agent_name": "",
            "task_id": task_id,
            "commands_run": [],
            "files_read": [],
            "changed_paths": [],
            "tests_modified": False,
            "final_pytest_passed": False,
            "needs_human_review": False,
            "cost": {"total_usd": None, "prompt_tokens": None, "completion_tokens": None},
            "llm_route_usage": [],
            "llm_failure_policy_decisions": [],
            "failure_recovery_events": [],
            "budget_summary": {},
            "runtime_mode_sequence": [],
            "side_effect_audit_events": [],
            "unknown_cost_reason": "",
            "no_llm_reason": "",
            "final_diff_summary": "",
            "raw_transcript_path": "",
        }
        execution_contract = _execution_contract_for_project(project, config, task_id=task_id, verifier=verifier)
        (task_dir / "TASK.md").write_text(_task_prompt(project, config) + "\n", encoding="utf-8")
        (task_dir / "RUNBOOK.md").write_text(_runbook_text(metadata, verifier, execution_contract) + "\n", encoding="utf-8")
        _write_json(task_dir / "metadata.json", metadata)
        _write_json(task_dir / "verifier.json", verifier)
        _write_json(task_dir / "execution_contract.json", execution_contract)
        _write_json(task_dir / "agent_result_template.json", result_template)
        packages.append(
            {
                "task_id": task_id,
                "project_id": metadata["project_id"],
                "task_dir": task_dir.as_posix(),
                "repo_url": metadata["repo_url"],
                "verifier_command": verifier["command"],
                "execution_contract": (task_dir / "execution_contract.json").as_posix(),
            }
        )
    summary = {
        "schema_version": OPEN_TASK_TASK_PACKAGE_VERSION,
        "mode": "package_only",
        "task_count": len(packages),
        "tasks": packages,
        "output_dir": output_root.as_posix(),
        "leak_prevention": {
            "blocked_keys": sorted(LEAKY_PROJECT_KEYS),
            "status": "passed",
        },
        "execution_contract": {
            "schema_version": OPEN_TASK_EXECUTION_CONTRACT_VERSION,
            "status": "written",
            "requires_route_and_cost_evidence": True,
            "requires_failure_recovery_evidence": True,
        },
    }
    _write_json(output_root / "task_packages_manifest.json", summary)
    return summary


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith("tests/") or "/tests/" in normalized or normalized.endswith("_test.py")


def _is_source_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if _is_test_path(normalized):
        return False
    return normalized.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb"))


def _is_non_source_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(pattern in normalized or normalized.endswith(pattern) for pattern in NON_SOURCE_PATTERNS)


def _cost_usd(report: Mapping[str, Any]) -> float | None:
    for key in ("cost_usd", "total_cost_usd"):
        if report.get(key) is not None:
            try:
                return float(report.get(key))
            except (TypeError, ValueError):
                return None
    cost = _as_dict(report.get("cost"))
    if cost.get("total_usd") is not None:
        try:
            return float(cost.get("total_usd"))
        except (TypeError, ValueError):
            return None
    return None


def _changed_lines(report: Mapping[str, Any]) -> int:
    stats = _as_dict(report.get("diff_stats"))
    for key in ("changed_lines", "total_changed_lines", "lines_changed"):
        if stats.get(key) is not None:
            try:
                return max(0, int(stats.get(key)))
            except (TypeError, ValueError):
                pass
    return max(0, int(report.get("changed_lines", 0) or 0))


def _score_minimal_patch(*, source_files: int, non_source_files: int, tests_modified: bool, changed_lines: int) -> float:
    score = 1.0
    if source_files > 1:
        score -= min(0.4, 0.12 * (source_files - 1))
    if changed_lines > 20:
        score -= min(0.35, 0.01 * (changed_lines - 20))
    if non_source_files:
        score -= min(0.25, 0.08 * non_source_files)
    if tests_modified:
        score -= 0.5
    return round(max(0.0, score), 4)


def _score_cleanliness(*, non_source_files: int, lockfile_without_need: int, cache_artifacts: int, tests_modified: bool) -> float:
    score = 1.0
    score -= min(0.4, 0.1 * non_source_files)
    score -= min(0.25, 0.15 * lockfile_without_need)
    score -= min(0.25, 0.1 * cache_artifacts)
    if tests_modified:
        score -= 0.6
    return round(max(0.0, score), 4)


def _score_traceability(report: Mapping[str, Any]) -> float:
    checks = [
        bool(_as_list(report.get("commands_run"))),
        bool(_as_list(report.get("files_read"))),
        bool(report.get("final_diff_summary")),
        bool(report.get("final_pytest_passed") is not None),
        bool(report.get("raw_transcript_path") or report.get("audit_trace_path") or report.get("mechanism_path")),
    ]
    return round(sum(1 for item in checks if item) / len(checks), 4)


def _route_traceability_score(report: Mapping[str, Any]) -> float:
    route_usage = _as_list(report.get("llm_route_usage")) or _as_list(report.get("route_usage"))
    failure_decisions = (
        _as_list(report.get("llm_failure_policy_decisions"))
        or _as_list(report.get("failure_policy_events"))
        or _as_list(report.get("llm_failure_policy_events"))
    )
    budget_summary = _as_dict(report.get("budget_summary")) or _as_dict(report.get("llm_budget_summary"))
    cost = _as_dict(report.get("cost"))
    checks = [
        bool(route_usage) or bool(report.get("no_llm_reason")),
        bool(failure_decisions) or bool(report.get("no_llm_failure_reason")) or not bool(report.get("llm_failure_observed")),
        bool(budget_summary) or cost.get("prompt_tokens") is not None or cost.get("completion_tokens") is not None,
        report.get("cost_usd") is not None or cost.get("total_usd") is not None or bool(report.get("unknown_cost_reason")),
    ]
    return round(sum(1 for item in checks if item) / len(checks), 4)


def _budget_observability_score(report: Mapping[str, Any]) -> float:
    budget_summary = _as_dict(report.get("budget_summary")) or _as_dict(report.get("llm_budget_summary"))
    cost = _as_dict(report.get("cost"))
    checks = [
        "max_llm_calls" in budget_summary or "llm_call_count" in budget_summary,
        "prompt_tokens" in cost or "prompt_tokens" in budget_summary,
        "completion_tokens" in cost or "completion_tokens" in budget_summary,
        "total_usd" in cost or report.get("cost_usd") is not None or bool(report.get("unknown_cost_reason")),
    ]
    return round(sum(1 for item in checks if item) / len(checks), 4)


def _hidden_fallback_patch_violation(report: Mapping[str, Any]) -> bool:
    if bool(report.get("hidden_fallback_patch_violation")):
        return True
    fallback_used = bool(report.get("fallback_patch_used") or report.get("fallback_patch_started"))
    fallback_approved = bool(report.get("fallback_patch_approved") or report.get("fallback_patch_audit_event"))
    if fallback_used and not fallback_approved:
        return True
    for event in _as_list(report.get("failure_recovery_events")) + _as_list(report.get("side_effect_audit_events")):
        if not isinstance(event, Mapping):
            continue
        text = json.dumps(event, sort_keys=True, default=str).lower()
        if "fallback_patch" in text and "approved" not in text and "audit" not in text:
            return True
    return False


def normalize_agent_report(report: Mapping[str, Any]) -> dict[str, Any]:
    changed_paths = [str(path) for path in _as_list(report.get("changed_paths")) if str(path)]
    expected_lockfiles = set(str(path) for path in _as_list(report.get("expected_lockfiles")) if str(path))
    tests_modified = bool(report.get("tests_modified", False)) or any(_is_test_path(path) for path in changed_paths)
    non_source_paths = [path for path in changed_paths if _is_non_source_artifact(path)]
    cache_paths = [
        path
        for path in changed_paths
        if any(token in path.replace("\\", "/") for token in (".pytest_cache/", "__pycache__/", ".ruff_cache/", ".mypy_cache/"))
    ]
    lockfiles_without_need = [
        path
        for path in changed_paths
        if Path(path).name in LOCKFILE_NAMES and path not in expected_lockfiles and Path(path).name not in expected_lockfiles
    ]
    source_paths = [path for path in changed_paths if _is_source_path(path)]
    non_source_changed = [path for path in changed_paths if path not in source_paths and not _is_test_path(path)]
    changed_lines = _changed_lines(report)
    final_pytest_passed = bool(report.get("final_pytest_passed") or report.get("verified_success") or report.get("final_tests_passed"))
    task_success = bool(report.get("task_success", final_pytest_passed and not tests_modified))
    cognitive_success = bool(
        report.get(
            "cognitive_success",
            task_success
            and bool(report.get("mechanism_path") or report.get("hypothesis_lifecycle_complete") or report.get("audit_trace_path")),
        )
    )
    minimal_patch_score = _score_minimal_patch(
        source_files=len(set(source_paths)),
        non_source_files=len(set(non_source_paths)),
        tests_modified=tests_modified,
        changed_lines=changed_lines,
    )
    cleanliness_score = _score_cleanliness(
        non_source_files=len(set(non_source_paths)),
        lockfile_without_need=len(set(lockfiles_without_need)),
        cache_artifacts=len(set(cache_paths)),
        tests_modified=tests_modified,
    )
    route_traceability_score = _route_traceability_score(report)
    budget_observability_score = _budget_observability_score(report)
    hidden_fallback_patch_violation = _hidden_fallback_patch_violation(report)
    failure_recovery_events = _as_list(report.get("failure_recovery_events"))
    product_open_task_readiness_score = round(
        max(
            0.0,
            (
                (1.0 if not tests_modified else 0.0)
                + cleanliness_score
                + minimal_patch_score
                + _score_traceability(report)
                + route_traceability_score
                + budget_observability_score
                + (1.0 if not hidden_fallback_patch_violation else 0.0)
            )
            / 7.0,
        ),
        4,
    )
    return {
        "schema_version": OPEN_TASK_RESULT_VERSION,
        "agent_name": str(report.get("agent_name") or "unknown"),
        "task_id": str(report.get("task_id") or report.get("fixture_id") or report.get("project_id") or "unknown"),
        "project_id": str(report.get("project_id") or report.get("fixture_id") or report.get("task_id") or "unknown"),
        "task_success": task_success,
        "verified_success": final_pytest_passed and not tests_modified,
        "cognitive_success": cognitive_success,
        "needs_human_review": bool(report.get("needs_human_review", False)),
        "modified_tests": tests_modified,
        "changed_paths": changed_paths,
        "changed_source_files_count": len(set(source_paths)),
        "changed_non_source_files_count": len(set(non_source_changed)),
        "unexpected_file_created_count": int(report.get("unexpected_file_created_count", 0) or 0),
        "non_source_artifact_count": len(set(non_source_paths)),
        "lockfile_created_without_need": len(set(lockfiles_without_need)),
        "cache_artifact_created_count": len(set(cache_paths)),
        "source_only_patch": bool(changed_paths) and not tests_modified and not non_source_paths,
        "minimal_patch_score": minimal_patch_score,
        "repo_cleanliness_score": cleanliness_score,
        "traceability_score": _score_traceability(report),
        "route_traceability_score": route_traceability_score,
        "budget_observability_score": budget_observability_score,
        "product_open_task_readiness_score": product_open_task_readiness_score,
        "hidden_fallback_patch_violation": hidden_fallback_patch_violation,
        "failure_recovery_event_count": len(failure_recovery_events),
        "cost_usd": _cost_usd(report),
        "commands_run_count": len(_as_list(report.get("commands_run"))),
        "files_read_count": len(_as_list(report.get("files_read"))),
        "wrong_patch_attempt_count": int(report.get("wrong_patch_attempt_count", 0) or 0),
        "rollback_count": int(report.get("rollback_count", 0) or 0),
        "verification_waste_ticks": int(report.get("verification_waste_ticks", 0) or 0),
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if bool(row.get(key))) / len(rows), 4)


def _average(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return round(mean(values), 6)


def _sum(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0) or 0) for row in rows))


def _amp_efficiency(
    *,
    os_rate: float,
    baseline_rate: float,
    os_cost: float | None,
    baseline_cost: float | None,
) -> dict[str, Any]:
    if amplification_efficiency is not None:
        return dict(
            amplification_efficiency(
                verified_success_rate_os=os_rate,
                verified_success_rate_baseline=baseline_rate,
                cost_os=os_cost or 0.0,
                cost_baseline=baseline_cost or 0.0,
            )
        )
    if not baseline_rate or not os_cost or not baseline_cost:
        return {
            "amplification_efficiency": None,
            "undefined_reason": "zero_baseline_success_or_nonpositive_cost",
        }
    return {"amplification_efficiency": round((os_rate / baseline_rate) / (os_cost / baseline_cost), 6)}


def analyze_report_payloads(
    reports: Iterable[Mapping[str, Any]],
    *,
    os_agent: str = "conos",
    baseline_agent: str = "baseline_llm",
) -> dict[str, Any]:
    normalized = [normalize_agent_report(report) for report in reports]
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        by_agent.setdefault(str(row["agent_name"]), []).append(row)
    agents = sorted(by_agent)
    task_success_by_agent = {agent: _rate(rows, "task_success") for agent, rows in by_agent.items()}
    verified_success_by_agent = {agent: _rate(rows, "verified_success") for agent, rows in by_agent.items()}
    average_cost_by_agent = {agent: _average(rows, "cost_usd") for agent, rows in by_agent.items()}
    cleanliness_by_agent = {agent: _average(rows, "repo_cleanliness_score") for agent, rows in by_agent.items()}
    minimality_by_agent = {agent: _average(rows, "minimal_patch_score") for agent, rows in by_agent.items()}
    traceability_by_agent = {agent: _average(rows, "traceability_score") for agent, rows in by_agent.items()}
    route_traceability_by_agent = {agent: _average(rows, "route_traceability_score") for agent, rows in by_agent.items()}
    budget_observability_by_agent = {agent: _average(rows, "budget_observability_score") for agent, rows in by_agent.items()}
    product_readiness_by_agent = {agent: _average(rows, "product_open_task_readiness_score") for agent, rows in by_agent.items()}
    os_rate = float(verified_success_by_agent.get(os_agent, 0.0) or 0.0)
    baseline_rate = float(verified_success_by_agent.get(baseline_agent, 0.0) or 0.0)
    os_cost = average_cost_by_agent.get(os_agent)
    baseline_cost = average_cost_by_agent.get(baseline_agent)
    return {
        "schema_version": OPEN_TASK_BENCHMARK_SUMMARY_VERSION,
        "report_count": len(normalized),
        "agents": agents,
        "task_success_by_agent": task_success_by_agent,
        "verified_success_rate_by_agent": verified_success_by_agent,
        "cognitive_success_rate_by_agent": {agent: _rate(rows, "cognitive_success") for agent, rows in by_agent.items()},
        "average_cost_usd_by_agent": average_cost_by_agent,
        "repo_cleanliness_score_by_agent": cleanliness_by_agent,
        "minimal_patch_score_by_agent": minimality_by_agent,
        "traceability_score_by_agent": traceability_by_agent,
        "route_traceability_score_by_agent": route_traceability_by_agent,
        "budget_observability_score_by_agent": budget_observability_by_agent,
        "product_open_task_readiness_score_by_agent": product_readiness_by_agent,
        "pollution_by_agent": {
            agent: {
                "unexpected_file_created_count": _sum(rows, "unexpected_file_created_count"),
                "non_source_artifact_count": _sum(rows, "non_source_artifact_count"),
                "lockfile_created_without_need": _sum(rows, "lockfile_created_without_need"),
                "cache_artifact_created_count": _sum(rows, "cache_artifact_created_count"),
                "modified_tests_count": _sum(rows, "modified_tests"),
            }
            for agent, rows in by_agent.items()
        },
        "waste_by_agent": {
            agent: {
                "wrong_patch_attempt_count": _sum(rows, "wrong_patch_attempt_count"),
                "rollback_count": _sum(rows, "rollback_count"),
                "verification_waste_ticks": _sum(rows, "verification_waste_ticks"),
            }
            for agent, rows in by_agent.items()
        },
        "recovery_and_policy_by_agent": {
            agent: {
                "failure_recovery_event_count": _sum(rows, "failure_recovery_event_count"),
                "hidden_fallback_patch_violation_count": _sum(rows, "hidden_fallback_patch_violation"),
            }
            for agent, rows in by_agent.items()
        },
        "amplification_efficiency": _amp_efficiency(
            os_rate=os_rate,
            baseline_rate=baseline_rate,
            os_cost=os_cost,
            baseline_cost=baseline_cost,
        ),
        "per_task": sorted(normalized, key=lambda row: (str(row["task_id"]), str(row["agent_name"]))),
    }


def load_report_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_dir():
            reports.extend(load_report_files(sorted(candidate.glob("*.json"))))
        elif candidate.exists() and candidate.suffix == ".json":
            payload = _read_json(candidate)
            if isinstance(payload.get("reports"), list):
                reports.extend(dict(row) for row in payload["reports"] if isinstance(row, Mapping))
            else:
                reports.append(payload)
    return reports
