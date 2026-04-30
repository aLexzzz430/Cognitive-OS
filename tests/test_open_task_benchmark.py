from __future__ import annotations

from pathlib import Path

import pytest

from experiments.open_task_benchmark.core import (
    analyze_report_payloads,
    assert_project_metadata_is_leak_free,
    create_task_packages,
)


def test_open_task_package_only_writes_leak_free_task_package(tmp_path: Path) -> None:
    config = {
        "seed": 1,
        "default_task_goal": "Find one small verified improvement.",
        "verifier_defaults": {"command": ["python", "-m", "pytest", "-q"], "timeout_seconds": 120},
        "projects": [
            {
                "project_id": "demo-project",
                "repo_url": "https://github.com/example/demo",
                "revision": "main",
                "language": "python",
            }
        ],
    }

    summary = create_task_packages(config, tmp_path, seed=1, limit=1)

    assert summary["task_count"] == 1
    task_dir = Path(summary["tasks"][0]["task_dir"])
    assert (task_dir / "TASK.md").exists()
    assert (task_dir / "RUNBOOK.md").exists()
    assert (task_dir / "verifier.json").exists()
    assert (task_dir / "execution_contract.json").exists()
    metadata = (task_dir / "metadata.json").read_text(encoding="utf-8")
    execution_contract = (task_dir / "execution_contract.json").read_text(encoding="utf-8")
    result_template = (task_dir / "agent_result_template.json").read_text(encoding="utf-8")
    assert "true_bug_file" not in metadata
    assert "expected_patch" not in metadata
    assert "true_bug_file" not in execution_contract
    assert "expected_patch" not in execution_contract
    assert "answer_leak_check" in metadata
    assert "failure_recovery_policy" in execution_contract
    assert "llm_route_usage" in result_template
    assert summary["execution_contract"]["requires_route_and_cost_evidence"] is True


def test_open_task_package_rejects_answer_leaks() -> None:
    with pytest.raises(ValueError, match="leaks answer"):
        assert_project_metadata_is_leak_free(
            {
                "project_id": "bad",
                "repo_url": "https://github.com/example/bad",
                "true_bug_file": "pkg/secret.py",
            }
        )


def test_open_task_analyzer_scores_cleanliness_minimality_and_cost() -> None:
    summary = analyze_report_payloads(
        [
            {
                "agent_name": "conos",
                "task_id": "task-a",
                "changed_paths": ["pkg/fix.py"],
                "final_pytest_passed": True,
                "changed_lines": 8,
                "cost": {"total_usd": 2.0},
                "budget_summary": {
                    "max_llm_calls": 12,
                    "prompt_tokens": 1200,
                    "completion_tokens": 200,
                },
                "llm_route_usage": [{"route": "patch_proposal", "provider": "openai", "model": "gpt-test"}],
                "failure_recovery_events": [{"event_type": "verifier_failed", "recovered": True}],
                "commands_run": ["pytest"],
                "files_read": ["pkg/fix.py"],
                "final_diff_summary": "one source fix",
                "mechanism_path": {"terminal_completion": True},
            },
            {
                "agent_name": "codex",
                "task_id": "task-a",
                "changed_paths": ["pkg/fix.py", "uv.lock"],
                "final_pytest_passed": True,
                "changed_lines": 8,
                "cost": {"total_usd": 1.0},
                "budget_summary": {"max_llm_calls": 4, "prompt_tokens": 800, "completion_tokens": 300},
                "llm_route_usage": [{"route": "patch_proposal", "provider": "codex-cli", "model": "spark"}],
                "commands_run": ["pytest"],
                "files_read": ["pkg/fix.py"],
                "final_diff_summary": "source fix plus lockfile",
                "raw_transcript_path": "codex.log",
            },
        ],
        os_agent="conos",
        baseline_agent="codex",
    )

    assert summary["task_success_by_agent"]["conos"] == 1.0
    assert summary["verified_success_rate_by_agent"]["codex"] == 1.0
    assert summary["pollution_by_agent"]["codex"]["lockfile_created_without_need"] == 1
    assert summary["repo_cleanliness_score_by_agent"]["conos"] > summary["repo_cleanliness_score_by_agent"]["codex"]
    assert summary["minimal_patch_score_by_agent"]["conos"] > summary["minimal_patch_score_by_agent"]["codex"]
    assert summary["route_traceability_score_by_agent"]["conos"] == 1.0
    assert summary["budget_observability_score_by_agent"]["conos"] == 1.0
    assert summary["recovery_and_policy_by_agent"]["conos"]["failure_recovery_event_count"] == 1
    assert summary["amplification_efficiency"]["amplification_efficiency"] == 0.5


def test_open_task_analyzer_treats_test_modification_as_verification_violation() -> None:
    summary = analyze_report_payloads(
        [
            {
                "agent_name": "baseline_llm",
                "task_id": "task-b",
                "changed_paths": ["tests/test_behavior.py"],
                "final_pytest_passed": True,
                "cost": {"total_usd": 1.0},
            }
        ]
    )

    row = summary["per_task"][0]
    assert row["modified_tests"] is True
    assert row["verified_success"] is False
    assert summary["pollution_by_agent"]["baseline_llm"]["modified_tests_count"] == 1


def test_open_task_analyzer_flags_hidden_fallback_patch_violation() -> None:
    summary = analyze_report_payloads(
        [
            {
                "agent_name": "conos",
                "task_id": "task-c",
                "changed_paths": ["pkg/fix.py"],
                "final_pytest_passed": True,
                "fallback_patch_used": True,
                "fallback_patch_approved": False,
                "commands_run": ["pytest"],
                "files_read": ["pkg/fix.py"],
                "final_diff_summary": "source fix",
                "cost": {"total_usd": 1.0},
                "unknown_cost_reason": "",
            }
        ]
    )

    row = summary["per_task"][0]
    assert row["hidden_fallback_patch_violation"] is True
    assert summary["recovery_and_policy_by_agent"]["conos"]["hidden_fallback_patch_violation_count"] == 1
    assert row["product_open_task_readiness_score"] < 1.0
