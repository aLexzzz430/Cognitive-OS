from __future__ import annotations

import json
import sys
import io
from pathlib import Path
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.local_machine.runner import _artifact_contract_check, _resolve_auto_route_policies, apply_task_template, run_local_machine_task
from integrations.local_machine.patch_proposal import generate_patch_proposals
from integrations.local_machine.target_binding import bind_target
from integrations.local_machine.action_grounding import (
    annotate_local_machine_patch_ranking,
    build_local_machine_posterior_action_bridge_candidate,
    choose_repo_grep_query,
    open_task_patch_evidence_gap,
    pytest_context_paths_from_tree,
    validate_local_machine_action,
)
from integrations.local_machine.task_adapter import LocalMachineSurfaceAdapter
from core.runtime.state_store import RuntimeStateStore
from core.runtime.long_run_supervisor import LongRunSupervisor
from core.orchestration.structured_answer import StructuredAnswerSynthesizer
from core.orchestration.goal_task_control import resolve_effective_task_approval_requirement
from modules.llm.budget import LLMCostLedger, LLMRuntimeBudget, wrap_with_budget
from planner.objective_decomposer import ObjectiveDecomposer
import modules.local_mirror.mirror as mirror_module
import modules.local_mirror.vm_backend as vm_backend_module


def test_local_machine_adapter_blocks_sync_plan_until_verified_completion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="update README",
        source_root=source,
        mirror_root=mirror_root,
        candidate_paths=["README.md"],
        default_command=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path('README.md').write_text('after\\n', encoding='utf-8')",
        ],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        expose_apply_tool=True,
        execution_backend="local",
    )

    first_obs = adapter.reset()
    assert first_obs.raw["local_mirror"]["workspace_file_count"] == 0
    assert [tool.name for tool in first_obs.available_tools] == ["mirror_acquire"]

    acquired = adapter.act({"function_name": "mirror_acquire"})
    assert acquired.ok is True
    assert (mirror_root / "workspace" / "README.md").read_text(encoding="utf-8") == "before\n"
    assert (source / "README.md").read_text(encoding="utf-8") == "before\n"
    assert [tool.name for tool in adapter.observe().available_tools] == ["mirror_exec"]

    executed = adapter.act({"function_name": "mirror_exec"})
    assert executed.ok is True
    assert (mirror_root / "workspace" / "README.md").read_text(encoding="utf-8") == "after\n"
    assert (source / "README.md").read_text(encoding="utf-8") == "before\n"

    planned = adapter.act({"function_name": "mirror_plan"})
    assert planned.ok is False
    assert planned.raw["state"] == "MIRROR_PLAN_BLOCKED"
    assert planned.raw["mirror_plan_blocked_reason"] == "no_verified_changes"
    assert (source / "README.md").read_text(encoding="utf-8") == "before\n"


def test_open_project_task_template_adds_robust_default_scaffold() -> None:
    rendered, report = apply_task_template("调查这个仓库并改进一个小问题", "auto")

    assert report["applied"] is True
    assert report["reason"] == "auto_open_project_detected"
    assert "repo_tree" in rendered
    assert "full verification" in rendered

    unchanged, disabled = apply_task_template("inspect README", "none")
    assert unchanged == "inspect README"
    assert disabled["applied"] is False


def test_core_main_loop_can_run_against_local_machine_adapter(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"

    audit = run_local_machine_task(
        instruction="inspect README",
        source_root=str(source),
        mirror_root=str(mirror_root),
        candidate_paths=["README.md"],
        run_id="local-machine-smoke",
        max_ticks_per_episode=2,
        reset_mirror=True,
        execution_backend="local",
    )

    final_mirror = audit["final_surface_raw"]["local_mirror"]
    assert audit["world_provider_source"] == "integrations.local_machine.runner"
    assert final_mirror["workspace_initial_state"] == "empty"
    assert final_mirror["workspace_file_count"] >= 1
    assert (mirror_root / "workspace" / "README.md").exists()
    assert (source / "README.md").read_text(encoding="utf-8") == "hello\n"


def test_local_machine_adapter_defaults_to_managed_vm_boundary(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="verify default execution boundary",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
    )

    obs = adapter.reset()
    boundary = obs.raw["local_mirror"]["execution_boundary"]

    assert obs.raw["local_mirror"]["execution_backend"] == "managed-vm"
    assert boundary["backend"] == "managed-vm"
    assert boundary["security_boundary"] == "conos_managed_vm_provider"
    assert "does_not_fall_back_to_host_process" in boundary["limitations"]


def test_default_managed_vm_blocks_exec_without_host_fallback(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="try default VM execution",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[sys.executable, "-c", "print('should not run on host')"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
    )
    unavailable = {"status": "UNAVAILABLE", "reason": "managed VM provider is unavailable", "real_vm_boundary": False}
    monkeypatch.setattr(mirror_module, "managed_vm_report", lambda **kwargs: dict(unavailable))
    monkeypatch.setattr(vm_backend_module, "managed_vm_report", lambda **kwargs: dict(unavailable))

    adapter.reset()
    executed = adapter.act({"function_name": "mirror_exec"})

    assert executed.ok is False
    assert executed.raw["execution_boundary"]["backend"] == "managed-vm"
    assert "managed VM" in executed.raw["failure_reason"] or "VM" in executed.raw["failure_reason"]
    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit_events"][-1]["event_type"] == "mirror_vm_backend_unavailable"


def test_local_backend_override_is_blocked_unless_adapter_explicitly_allows_host(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="model must not choose host execution",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[sys.executable, "-c", "print('host')"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
    )

    adapter.reset()
    executed = adapter.act({"function_name": "mirror_exec", "kwargs": {"backend": "local"}})

    assert executed.ok is False
    assert "explicit adapter configuration" in executed.raw["failure_reason"]


def test_empty_first_open_investigation_exposes_atomic_inventory_without_candidates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect this repository and find a small improvement",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
    )

    obs = adapter.reset()
    tool_names = {tool.name for tool in obs.available_tools}

    assert "repo_tree" in tool_names
    assert "file_read" not in tool_names
    assert "run_test" not in tool_names
    assert "mirror_exec" not in tool_names
    assert obs.raw["local_mirror"]["workspace_file_count"] == 0


def test_open_project_default_command_keeps_atomic_inventory_first(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello\n", encoding="utf-8")
    (source / "tests").mkdir()
    (source / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect this repository, find a small improvement, and run tests",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        execution_backend="local",
    )

    obs = adapter.reset()
    tool_names = [tool.name for tool in obs.available_tools]

    assert tool_names == ["repo_tree"]
    assert "mirror_acquire" not in tool_names
    assert obs.raw["local_mirror"]["default_command_present"] is True

    tree = adapter.act({"function_name": "repo_tree"})
    next_action = build_local_machine_posterior_action_bridge_candidate(tree.observation.raw)
    assert next_action is not None
    assert next_action["function_name"] == "run_test"
    assert next_action["kwargs"]["target"] == "tests/test_smoke.py"

    tested = adapter.act(next_action)
    followup = build_local_machine_posterior_action_bridge_candidate(tested.observation.raw)
    if followup is not None:
        assert not (
            followup["function_name"] == "run_test"
            and followup["kwargs"].get("target") == "tests/test_smoke.py"
        )


def test_empty_wait_loop_emits_repo_tree_recovery_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect this repository and find a small improvement",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
    )

    result = None
    for _ in range(4):
        result = adapter.act({"kind": "wait"})

    assert result is not None
    phase = result.raw["local_machine_investigation_phase"]
    assert phase["stalled_event"]["recommended_action"] == "repo_tree"
    assert phase["empty_action_attempt_count"] == 4
    recovery = build_local_machine_posterior_action_bridge_candidate(result.observation.raw)
    assert recovery is not None
    assert recovery["function_name"] == "repo_tree"
    assert recovery["kwargs"]["path"] == "."


def test_stalled_open_improvement_can_escalate_to_bounded_patch_proposal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect this repository and improve one small low-risk issue",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
        prefer_llm_patch_proposals=True,
    )

    adapter.act({"function_name": "repo_tree"})
    adapter.act({"function_name": "file_read", "kwargs": {"path": "app.py", "start_line": 1, "end_line": 20}})
    state = adapter._load_investigation_state()
    state["stalled_events"] = [
        {
            "event_type": "investigation_stalled",
            "recommended_action": "propose_patch",
            "top_target_file": "app.py",
            "target_confidence": 0.16,
            "recent_actions": ["repo_grep", "repo_grep", "repo_grep", "repo_grep"],
        }
    ]
    state["target_binding"] = {
        "top_target_file": "app.py",
        "target_confidence": 0.16,
        "target_file_candidates": [{"target_file": "app.py", "score": 0.16}],
    }
    state.setdefault("grounding", {})["target_file"] = "app.py"
    adapter._save_investigation_state(state)

    recovery = build_local_machine_posterior_action_bridge_candidate(adapter.observe().raw)

    assert recovery is not None
    assert recovery["function_name"] == "propose_patch"
    assert recovery["kwargs"]["target_file"] == "app.py"


def test_open_improvement_full_verification_without_changes_does_not_complete(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "tests").mkdir()
    (source / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="improve this repository with one small low-risk patch",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
    )

    result = adapter.act({"function_name": "run_test", "kwargs": {"target": ".", "timeout_seconds": 30}})

    assert result.ok is True
    phase = result.raw["local_machine_investigation_phase"]
    assert phase["terminal_state"] == ""
    assert phase["phase_after"] == "discover"
    grounding = result.observation.raw["local_mirror"]["action_grounding"]
    assert grounding["completion_blocked_reason"] == "no_verified_changes"
    assert result.observation.terminal is False


def test_open_improvement_mirror_plan_hidden_until_meaningful_change(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="improve this repository with one small low-risk patch",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[sys.executable, "-c", "print('ok')"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        execution_backend="local",
    )

    adapter.act({"function_name": "repo_tree"})
    adapter.act({"function_name": "mirror_exec"})
    tool_names = [tool.name for tool in adapter.observe().available_tools]

    assert "mirror_plan" not in tool_names


def test_run_test_helper_target_repairs_to_runnable_pytest_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "h11" / "tests").mkdir(parents=True)
    (source / "h11" / "tests" / "helpers.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (source / "h11" / "tests" / "test_connection.py").write_text("def test_connection():\n    assert True\n", encoding="utf-8")
    context = {
        "instruction": "improve this repository",
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "h11/tests/helpers.py"},
                    {"kind": "file", "path": "h11/tests/test_connection.py"},
                ]
            }
        },
    }

    result = validate_local_machine_action(
        "run_test",
        {"target": "h11/tests/helpers.py"},
        context,
    )

    assert result["status"] == "repaired"
    assert result["function_name"] == "run_test"
    assert result["kwargs"]["target"] == "h11/tests/test_connection.py"
    assert result["event"]["repair_source"] == "selected test helper is not a runnable pytest target"


def test_open_task_evidence_gate_ignores_test_helper_as_test_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "h11" / "tests").mkdir(parents=True)
    (source / "README.rst").write_text("demo\n", encoding="utf-8")
    (source / "h11" / "tests" / "helpers.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (source / "h11" / "tests" / "test_connection.py").write_text("def test_connection():\n    assert True\n", encoding="utf-8")
    context = {
        "instruction": "improve this repository",
        "source_root": str(source),
        "investigation_state": {
            "read_files": [{"path": "README.rst"}],
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "README.rst"},
                    {"kind": "file", "path": "h11/tests/helpers.py"},
                    {"kind": "file", "path": "h11/tests/test_connection.py"},
                ]
            },
        },
    }

    gate = open_task_patch_evidence_gap(context, target_file="")

    assert gate["sufficient"] is False
    assert gate["reason"] == "test_source_required_before_open_task_patch"
    assert gate["suggested_kwargs"]["path"] == "h11/tests/test_connection.py"
    assert "h11/tests/helpers.py" not in gate["test_files"]


def test_pytest_context_paths_include_nested_test_package_helpers(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "h11" / "tests").mkdir(parents=True)
    (source / "h11" / "__init__.py").write_text("", encoding="utf-8")
    (source / "h11" / "_events.py").write_text("class Event:\n    pass\n", encoding="utf-8")
    (source / "h11" / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (source / "h11" / "tests" / "helpers.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (source / "h11" / "tests" / "test_connection.py").write_text("def test_connection():\n    assert True\n", encoding="utf-8")
    context = {
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "pyproject.toml"},
                    {"kind": "file", "path": "h11/_events.py"},
                ]
            }
        },
    }

    paths = pytest_context_paths_from_tree(context, limit=20)

    assert "h11/tests/__init__.py" in paths
    assert "h11/tests/helpers.py" in paths
    assert "h11/tests/test_connection.py" in paths


def test_open_task_evidence_gate_prioritizes_package_source_over_auxiliary_dirs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "bench" / "benchmarks").mkdir(parents=True)
    (source / "h11").mkdir(parents=True)
    (source / "h11" / "tests").mkdir(parents=True)
    (source / "README.rst").write_text("demo\n", encoding="utf-8")
    (source / "bench" / "benchmarks" / "__init__.py").write_text("", encoding="utf-8")
    (source / "bench" / "benchmarks" / "benchmarks.py").write_text("def bench():\n    return 1\n", encoding="utf-8")
    (source / "h11" / "__init__.py").write_text("from ._events import Event\n", encoding="utf-8")
    (source / "h11" / "_events.py").write_text("class Event:\n    pass\n", encoding="utf-8")
    (source / "h11" / "_connection.py").write_text("class Connection:\n    pass\n", encoding="utf-8")
    (source / "h11" / "tests" / "test_events.py").write_text("def test_event():\n    assert True\n", encoding="utf-8")
    context = {
        "instruction": "improve this repository with one small low-risk source patch",
        "source_root": str(source),
        "episode_run_test_targets": ["h11/tests/test_events.py"],
        "investigation_state": {
            "read_files": [{"path": "README.rst"}, {"path": "h11/tests/test_events.py"}],
            "validation_runs": [{"run_ref": "run_ok", "success": True}],
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "README.rst"},
                    {"kind": "file", "path": "bench/benchmarks/__init__.py"},
                    {"kind": "file", "path": "bench/benchmarks/benchmarks.py"},
                    {"kind": "file", "path": "h11/__init__.py"},
                    {"kind": "file", "path": "h11/_connection.py"},
                    {"kind": "file", "path": "h11/_events.py"},
                    {"kind": "file", "path": "h11/tests/test_events.py"},
                ]
            },
        },
    }

    gate = open_task_patch_evidence_gap(context, target_file="")

    assert gate["sufficient"] is False
    assert gate["reason"] == "related_source_required_before_open_task_patch"
    assert gate["suggested_kwargs"]["path"] in {"h11/_connection.py", "h11/_events.py"}
    assert not gate["suggested_kwargs"]["path"].startswith("bench/")


def test_open_task_no_progress_after_green_validation_terminally_refuses(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (source / "pkg" / "core.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_core.py").write_text("from pkg.core import value\n\n\ndef test_value():\n    assert value() == 1\n", encoding="utf-8")
    adapter = LocalMachineSurfaceAdapter(
        instruction="improve this repository with one small low-risk source patch",
        source_root=source,
        mirror_root=tmp_path / "mirror",
        reset_mirror=True,
        execution_backend="local",
    )

    assert adapter.act({"function_name": "repo_tree", "kwargs": {"path": ".", "depth": 3, "max_entries": 50}}).ok is True
    assert adapter.act({"function_name": "run_test", "kwargs": {"target": ".", "timeout_seconds": 30}}).ok is True
    assert adapter.act({"function_name": "file_read", "kwargs": {"path": "README.md", "start_line": 1, "end_line": 20}}).ok is True
    assert adapter.act({"function_name": "file_read", "kwargs": {"path": "tests/test_core.py", "start_line": 1, "end_line": 40}}).ok is True
    assert adapter.act({"function_name": "file_read", "kwargs": {"path": "pkg/core.py", "start_line": 1, "end_line": 20}}).ok is True
    result = None
    for _ in range(6):
        result = adapter.act({"function_name": "run_typecheck", "kwargs": {}})

    assert result is not None
    assert result.observation.terminal is True
    investigation = adapter._load_investigation_state()
    assert investigation["terminal_state"] == "needs_human_review"
    assert investigation["refusal_reason"] == "evidence_insufficient"
    assert investigation["grounding"]["open_task_no_progress_refusal"]["event_type"] == "open_task_no_progress_refusal"


def test_pytest_context_paths_prioritizes_config_and_tests_from_large_tree() -> None:
    entries = [
        {"path": f"core/module_{index}.py", "kind": "file"}
        for index in range(80)
    ]
    entries.extend(
        [
            {"path": "pyproject.toml", "kind": "file"},
            {"path": "tests/test_runtime_budget.py", "kind": "file"},
            {"path": "tests/test_local_machine_adapter.py", "kind": "file"},
            {"path": "core/app.py", "kind": "file"},
        ]
    )
    context = {
        "source_root": "/nonexistent/conos-test-root",
        "investigation_state": {"last_tree": {"entries": entries}},
    }

    paths = pytest_context_paths_from_tree(context, limit=10)

    assert paths[:3] == [
        "pyproject.toml",
        "tests/test_runtime_budget.py",
        "tests/test_local_machine_adapter.py",
    ]
    assert "core/module_0.py" in paths


def test_pytest_context_paths_falls_back_to_source_tests_when_tree_is_truncated(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (source / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    entries = [{"path": f"core/module_{index}.py", "kind": "file"} for index in range(120)]
    context = {
        "source_root": str(source),
        "investigation_state": {"last_tree": {"entries": entries}},
    }

    paths = pytest_context_paths_from_tree(context, limit=20)

    assert paths[:2] == ["pyproject.toml", "tests/test_smoke.py"]


def test_pytest_context_paths_falls_back_to_src_package_when_tree_is_truncated(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src" / "sample_pkg").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='sample-pkg'\n", encoding="utf-8")
    (source / "src" / "sample_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (source / "src" / "sample_pkg" / "core.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_core.py").write_text(
        "from src.sample_pkg.core import value\n\n"
        "def test_value():\n    assert value() == 1\n",
        encoding="utf-8",
    )
    (source / "tests" / "fixtures.txt").write_text("fixture-data\n", encoding="utf-8")
    entries = [
        {"path": "src", "kind": "dir"},
        {"path": "src/sample_pkg", "kind": "dir"},
        {"path": "tests", "kind": "dir"},
        {"path": "tests/test_core.py", "kind": "file"},
        {"path": "pyproject.toml", "kind": "file"},
    ]
    context = {
        "source_root": str(source),
        "investigation_state": {"last_tree": {"entries": entries}},
    }

    paths = pytest_context_paths_from_tree(context, limit=20)

    assert "tests/test_core.py" in paths
    assert "tests/fixtures.txt" in paths
    assert "src/sample_pkg/core.py" in paths
    assert "src/sample_pkg/__init__.py" in paths


def test_run_test_source_target_repairs_to_direct_test(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "core").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "core" / "test_designer.py").write_text("def build():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_test_designer.py").write_text("def test_build():\n    assert True\n", encoding="utf-8")
    context = {
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"path": "core/test_designer.py", "kind": "file"},
                    {"path": "tests/test_test_designer.py", "kind": "file"},
                ]
            }
        },
    }

    result = validate_local_machine_action(
        "run_test",
        {"target": "core/test_designer.py", "timeout_seconds": 30},
        context,
    )

    assert result["status"] == "repaired"
    assert result["kwargs"]["target"] == "tests/test_test_designer.py"
    assert result["event"]["repair_source"] == "direct pytest file for selected source target"


def test_run_typecheck_full_target_materializes_context_before_validation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "sample_pkg").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n", encoding="utf-8")
    (source / "sample_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (source / "sample_pkg" / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_app.py").write_text("from sample_pkg.app import value\n\ndef test_value():\n    assert value() == 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect and validate",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
    )
    adapter.reset()
    adapter.act({"function_name": "repo_tree", "kwargs": {"path": ".", "depth": 3, "max_entries": 50}})

    result = adapter.act({"function_name": "run_typecheck", "kwargs": {"target": ".", "timeout_seconds": 30}})

    assert result.ok is True
    assert result.raw["success"] is True
    assert result.raw["materialized_for_validation"]
    assert (mirror_root / "workspace" / "tests" / "test_app.py").exists()


def test_run_test_directory_target_materializes_tests_and_source_context(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "sample_pkg").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n", encoding="utf-8")
    (source / "sample_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (source / "sample_pkg" / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_app.py").write_text("from sample_pkg.app import value\n\ndef test_value():\n    assert value() == 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="run tests",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
    )
    adapter.reset()
    adapter.act({"function_name": "repo_tree", "kwargs": {"path": ".", "depth": 3, "max_entries": 50}})

    result = adapter.act({"function_name": "run_test", "kwargs": {"target": "tests", "timeout_seconds": 30}})

    assert result.ok is True
    assert result.raw["success"] is True
    assert result.raw["action_governance"]["status"] == "ALLOWED"
    assert result.raw["action_governance"]["audit_event"]["capability_layers"] == ["execute"]
    assert result.raw["action_governance"]["audit_event"]["side_effect_class"] == "execution"
    assert result.raw["side_effect_audit_enforced"] is True
    assert result.raw["side_effect_audit"]["event_type"] == "action_governance_decision"
    assert result.observation.raw["local_mirror"]["action_governance"]["side_effect_audit_events"][-1]["action_name"] == "run_test"
    assert (mirror_root / "workspace" / "tests" / "test_app.py").exists()
    assert (mirror_root / "workspace" / "sample_pkg" / "app.py").exists()


def test_run_test_supports_src_layout_without_installing_package(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src" / "sample_pkg").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n", encoding="utf-8")
    (source / "src" / "sample_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (source / "src" / "sample_pkg" / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_app.py").write_text(
        "from sample_pkg.app import value\n\n\ndef test_value():\n    assert value() == 1\n",
        encoding="utf-8",
    )
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="run tests",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
    )
    adapter.reset()
    adapter.act({"function_name": "repo_tree", "kwargs": {"path": ".", "depth": 3, "max_entries": 50}})

    result = adapter.act({"function_name": "run_test", "kwargs": {"target": "tests", "timeout_seconds": 30}})

    assert result.ok is True
    assert result.raw["success"] is True
    mirror_command = result.raw["mirror_command"]
    assert "PYTHONPATH" in mirror_command["env_audit"]["explicit_env_keys"]
    assert result.observation.raw["local_mirror"]["command_executed"] is True
    assert (mirror_root / "workspace" / "src" / "sample_pkg" / "app.py").exists()


def test_run_test_missing_pytest_fixture_stops_for_human_review(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_missing_fixture.py").write_text(
        "def test_needs_plugin(testdir):\n    assert testdir\n",
        encoding="utf-8",
    )
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="run tests",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
    )
    adapter.reset()

    result = adapter.act({"function_name": "run_test", "kwargs": {"target": "tests", "timeout_seconds": 30}})

    assert result.raw["success"] is False
    assert result.raw["needs_human_review"] is True
    assert result.raw["refusal_reason"] == "environment_blocked"
    assert result.raw["missing_pytest_fixtures"] == ["testdir"]
    phase = result.raw["local_machine_investigation_phase"]
    assert phase["terminal_state"] == "needs_human_review"
    assert result.observation.terminal is True


def test_local_machine_side_effect_audit_guard_records_without_governance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="run tests with governance disabled",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
        action_governance_enabled=False,
    )

    adapter.reset()
    result = adapter.act({"function_name": "run_test", "kwargs": {"target": "tests", "timeout_seconds": 30}})

    assert result.ok is True
    assert "action_governance" not in result.raw
    assert result.raw["side_effect_audit_enforced"] is True
    assert result.raw["side_effect_audit"]["event_type"] == "side_effect_audit_event"
    assert result.raw["side_effect_audit"]["side_effect_class"] == "execution"
    assert any(event.get("event_type") == "side_effect_audit_event" for event in result.events)
    persisted = result.observation.raw["local_mirror"]["action_governance"]["side_effect_audit_events"]
    assert persisted[-1]["audit_source"] == "local_machine_adapter_final_guard"


def test_local_machine_action_policy_blocks_execute_layer_before_running(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="read-only review",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
        action_governance_policy={"allowed_capability_layers": ["read", "propose_patch"]},
    )

    adapter.reset()
    result = adapter.act({"function_name": "run_test", "kwargs": {"target": "tests", "timeout_seconds": 30}})

    assert result.ok is False
    assert result.raw["state"] == "BLOCKED"
    assert result.raw["action_governance"]["status"] == "BLOCKED"
    assert result.raw["action_governance"]["blocked_reason"] == "capability_layer_not_allowed:execute"
    assert not (mirror_root / "workspace" / "tests" / "test_app.py").exists()


def test_local_machine_daemon_plan_waits_for_approval_without_terminal_shutdown(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="update README",
        source_root=source,
        mirror_root=mirror_root,
        candidate_paths=["README.md"],
        default_command=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path('README.md').write_text('after\\n', encoding='utf-8')",
        ],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        terminal_after_plan=False,
    )

    adapter.reset()
    adapter.act({"function_name": "mirror_acquire"})
    adapter.act({"function_name": "mirror_exec"})
    planned = adapter.act({"function_name": "mirror_plan"})

    assert planned.raw["state"] == "MIRROR_PLAN_BLOCKED"
    assert planned.raw["mirror_plan_blocked_reason"] == "no_verified_changes"
    assert planned.raw["terminal"] is False
    assert planned.observation.terminal is False


def test_failed_daemon_plan_finishes_instead_of_waiting_for_approval(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="build a new AI project from an empty mirror",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[sys.executable, "-c", "raise SystemExit(1)"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        terminal_after_plan=False,
        allow_empty_exec=True,
        execution_backend="local",
    )

    adapter.reset()
    executed = adapter.act({"function_name": "mirror_exec"})
    planned = adapter.act({"function_name": "mirror_plan"})

    assert executed.ok is False
    assert planned.raw["state"] == "COMMAND_FAILED"
    assert planned.raw["waiting_approval"] is False
    assert planned.raw["terminal"] is True
    assert planned.observation.terminal is True


def test_empty_mirror_can_expose_default_command_when_allowed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="build a new AI project from an empty mirror",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[sys.executable, "-c", "print('ready')"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        allow_empty_exec=True,
        execution_backend="local",
    )

    obs = adapter.reset()

    assert obs.raw["local_mirror"]["workspace_file_count"] == 0
    assert [tool.name for tool in obs.available_tools][0] == "mirror_exec"


def test_empty_internet_task_can_expose_exec_when_allowed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="research market and build a product",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        internet_enabled=True,
        allow_empty_exec=True,
    )

    obs = adapter.reset()
    tools = [tool.name for tool in obs.available_tools]

    assert "mirror_exec" in tools
    assert "internet_fetch" in tools


def test_empty_mirror_can_expose_model_generated_exec_without_default_command(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect the source root and decide what to copy into the mirror",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
        execution_backend="local",
    )

    obs = adapter.reset()
    tools = [tool.name for tool in obs.available_tools]

    assert tools[0] == "repo_tree"
    assert "mirror_exec" in tools
    assert "internet_fetch" not in tools


def test_selected_top_level_kwargs_override_stale_payload_kwargs() -> None:
    action = {
        "kind": "call_tool",
        "function_name": "run_test",
        "kwargs": {"target": "tests/test_amounts.py", "timeout_seconds": 30},
        "payload": {
            "tool_args": {
                "function_name": "run_test",
                "kwargs": {"target": "tests", "timeout_seconds": 120},
            }
        },
    }

    function_name, kwargs = LocalMachineSurfaceAdapter._extract_tool_call(action)

    assert function_name == "run_test"
    assert kwargs == {"target": "tests/test_amounts.py", "timeout_seconds": 30}


def test_expected_tests_from_llm_are_normalized_to_test_targets() -> None:
    assert (
        LocalMachineSurfaceAdapter._normalize_expected_test_target(
            "pytest tests/test_amounts.py::test_parse_amount_accepts_common_currency_inputs -q"
        )
        == "tests/test_amounts.py"
    )
    assert LocalMachineSurfaceAdapter._normalize_expected_test_target("python -m pytest -q") == "."
    assert LocalMachineSurfaceAdapter._normalize_expected_test_target("tests") == "."
    assert (
        LocalMachineSurfaceAdapter._normalize_expected_test_target(
            "test_boundary_score_is_high",
            fallback_failed_target="tests/test_score.py",
        )
        == "tests/test_score.py"
    )
    assert LocalMachineSurfaceAdapter._normalize_expected_test_target("test_boundary_score_is_high") == ""


def test_natural_language_expected_tests_are_filtered_to_full_verification(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="improve safely",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="local",
    )

    tests = adapter._proposal_expected_tests(
        {
            "expected_tests": [
                "针对包含空行/注释/前后空白规则的 Gitignore_Matcher 读取与匹配用例",
                "项目全量测试",
            ]
        },
        {},
    )

    assert tests == ["."]


def test_atomic_local_machine_discovery_and_notes_persist_state(tmp_path: Path) -> None:
    source = tmp_path / "AGI3"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    core = source / "core"
    core.mkdir()
    (core / "local_machine.py").write_text("def mirror_exec():\n    return 'ok'\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="investigate AGI3",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
        execution_backend="local",
    )

    tree = adapter.act({"action": "repo_tree", "args": {"path": "AGI3", "depth": 2}})
    assert tree.ok is True
    assert any(row["path"] == "core/local_machine.py" for row in tree.raw["entries"])

    grep = adapter.act({"action": "repo_grep", "args": {"root": "AGI3", "query": "mirror_exec", "globs": ["*.py"]}})
    assert grep.ok is True
    assert grep.raw["matches"][0]["path"] == "core/local_machine.py"

    read = adapter.act({"action": "file_read", "args": {"path": "AGI3/core/local_machine.py", "start_line": 1, "end_line": 1}})
    assert read.ok is True
    assert "mirror_exec" in read.raw["content"]

    note = adapter.act(
        {
            "action": "note_write",
            "args": {
                "kind": "finding",
                "content": "mirror_exec is implemented in core/local_machine.py",
                "evidence_refs": ["file:core/local_machine.py:1"],
            },
        }
    )
    assert note.ok is True

    candidates = adapter.act({"action": "candidate_files_set", "args": {"files": ["AGI3/core/local_machine.py"], "reason": "contains mirror_exec"}})
    assert candidates.raw["candidate_files"] == ["core/local_machine.py"]

    status = adapter.act({"action": "investigation_status", "args": {}})
    investigation = status.raw["investigation"]
    assert investigation["candidate_files"] == ["core/local_machine.py"]
    assert investigation["notes"][-1]["content"].startswith("mirror_exec")


def test_file_read_repairs_unique_missing_path_from_last_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    core = source / "core"
    core.mkdir()
    (core / "runtime_budget.py").write_text("def budget():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect runtime budget",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
    )

    tree = adapter.act({"action": "repo_tree", "args": {"path": ".", "depth": 2}})
    assert tree.ok is True
    wrong_absolute = source / "core" / "runtime" / "runtime_budget.py"
    read = adapter.act(
        {
            "action": "file_read",
            "args": {"path": str(wrong_absolute), "start_line": 1, "end_line": 2},
        }
    )

    assert read.ok is True
    assert read.raw["path"] == "core/runtime_budget.py"
    status = adapter.act({"action": "investigation_status", "args": {}})
    assert status.raw["investigation"]["last_path_correction"]["resolved_path"] == "core/runtime_budget.py"


def test_atomic_local_machine_actions_write_formal_evidence_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
        task_id="formal-ledger-smoke",
    )

    read = adapter.act({"action": "file_read", "args": {"path": "app.py", "start_line": 1, "end_line": 2}})
    assert read.ok is True
    assert read.raw["formal_evidence_id"].startswith("ev_")

    ledger = read.observation.raw["local_mirror"]["formal_evidence_ledger"]
    assert ledger["entry_count"] == 1
    assert ledger["last_evidence_id"] == read.raw["formal_evidence_id"]
    payload = json.loads(Path(ledger["path"]).read_text(encoding="utf-8").splitlines()[0])
    assert payload["run_id"] == "formal-ledger-smoke"
    assert payload["evidence_type"] == "codebase_observation"
    assert payload["source_refs"] == ["file:app.py:1-2"]
    assert payload["formal_commit"]["object_layer"] is True


def test_local_machine_formal_evidence_can_persist_to_runtime_db(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("x = 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    db_path = tmp_path / "runtime.sqlite3"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
        evidence_db_path=db_path,
        task_id="runtime-db-ledger",
    )

    result = adapter.act({"action": "file_summary", "args": {"path": "app.py"}})
    assert result.ok is True

    store = RuntimeStateStore(db_path)
    rows = store.list_evidence_entries(run_id="runtime-db-ledger")
    store.close()
    assert rows
    assert rows[0]["evidence_id"] == result.raw["formal_evidence_id"]
    assert rows[0]["task_family"] == "local_machine"


def test_local_machine_hypothesis_lifecycle_actions_are_persisted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    db_path = tmp_path / "runtime.sqlite3"
    adapter = LocalMachineSurfaceAdapter(
        instruction="investigate competing causes",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
        evidence_db_path=db_path,
        task_id="hypothesis-lifecycle-smoke",
    )

    h1 = adapter.act(
        {
            "action": "hypothesis_add",
            "args": {
                "claim": "The behavior comes from app.py returning 42",
                "family": "code_path",
                "confidence": 0.5,
                "evidence_refs": ["file:app.py:1-2"],
                "predictions": {"run_lint": "passes"},
            },
        }
    )
    h2 = adapter.act(
        {
            "action": "hypothesis_add",
            "args": {
                "claim": "The behavior comes from a hidden generated file",
                "family": "code_path",
                "confidence": 0.5,
            },
        }
    )
    assert h1.ok is True
    assert h2.ok is True

    compete = adapter.act(
        {
            "action": "hypothesis_compete",
            "args": {
                "hypothesis_a": h1.raw["hypothesis"]["hypothesis_id"],
                "hypothesis_b": h2.raw["hypothesis"]["hypothesis_id"],
                "reason": "Both explain the current output but imply different files.",
            },
        }
    )
    assert compete.ok is True
    assert compete.raw["hypothesis_lifecycle"]["needs_discriminating_test"] is True

    dtest = adapter.act(
        {
            "action": "discriminating_test_add",
            "args": {
                "hypothesis_a": h1.raw["hypothesis"]["hypothesis_id"],
                "hypothesis_b": h2.raw["hypothesis"]["hypothesis_id"],
                "action": {"action": "file_read", "args": {"path": "app.py", "start_line": 1, "end_line": 2}},
                "expected_if_a": "app.py contains the return value",
                "expected_if_b": "app.py is irrelevant or absent",
                "why": "A bounded file read distinguishes the file-local cause from a hidden generated-file cause.",
            },
        }
    )
    assert dtest.ok is True
    assert dtest.raw["discriminating_test"]["test_id"].startswith("dtest_")

    update = adapter.act(
        {
            "action": "hypothesis_update",
            "args": {
                "hypothesis_id": h1.raw["hypothesis"]["hypothesis_id"],
                "signal": "support",
                "evidence_ref": "file:app.py:1-2",
                "strength": 1.0,
                "rationale": "The discriminating read found the expected return value.",
            },
        }
    )
    assert update.ok is True
    assert update.raw["hypothesis"]["posterior"] == 0.675
    assert update.raw["hypothesis"]["status"] == "supported"

    status = adapter.act({"action": "investigation_status", "args": {}})
    assert status.raw["hypothesis_lifecycle"]["hypothesis_count"] == 2
    assert status.raw["investigation"]["discriminating_tests"]

    store = RuntimeStateStore(db_path)
    rows = store.list_hypothesis_lifecycle(run_id="hypothesis-lifecycle-smoke")
    events = store.list_hypothesis_lifecycle_events(run_id="hypothesis-lifecycle-smoke")
    evidence_rows = store.list_evidence_entries(run_id="hypothesis-lifecycle-smoke", evidence_type="investigation_state")
    store.close()

    assert len(rows) == 2
    assert any(row["status"] == "supported" for row in rows)
    assert {event["event_type"] for event in events} >= {
        "hypothesis_created",
        "hypothesis_competition_recorded",
        "discriminating_test_proposed",
        "evidence_support",
    }
    assert any(row["update"]["kind"] == "hypothesis_update" for row in evidence_rows)


def test_atomic_edit_validate_and_read_run_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fix app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
        execution_backend="local",
    )

    read = adapter.act({"action": "file_read", "args": {"path": "app.py", "start_line": 1, "end_line": 2}})
    assert read.ok is True

    edited = adapter.act(
        {
            "action": "edit_replace_range",
            "args": {
                "path": "app.py",
                "start_line": 2,
                "end_line": 2,
                "replacement": "    return 42",
            },
        }
    )
    assert edited.ok is True
    assert (mirror_root / "workspace" / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 42\n"
    assert (source / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 1\n"

    lint = adapter.act({"action": "run_lint", "args": {"target": "app.py", "timeout_seconds": 10}})
    assert lint.ok is True
    assert lint.raw["returncode"] == 0

    output = adapter.act({"action": "read_run_output", "args": {"run_ref": lint.raw["run_ref"], "max_chars": 1000}})
    assert output.ok is True
    assert output.raw["command"][:3] == [sys.executable, "-m", "py_compile"]


def test_validation_actions_reject_missing_targets_before_running(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fix app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
    )

    missing_test = adapter.act({"action": "run_test", "args": {"target": "tests/test_runtime_budget.py"}})
    missing_typecheck = adapter.act({"action": "run_typecheck", "args": {"target": "test_runtime_budget_resource_cleanup"}})

    assert missing_test.ok is False
    assert "validation target does not exist" in missing_test.raw["failure_reason"]
    assert missing_typecheck.ok is False
    assert "validation target does not exist" in missing_typecheck.raw["failure_reason"]
    assert not (mirror_root / "workspace" / ".pytest_cache").exists()


def test_mirror_plan_requires_real_change_for_edit_tasks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fix app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
        terminal_after_plan=False,
    )
    adapter.reset()
    cache_file = mirror_root / "workspace" / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("[]", encoding="utf-8")

    planned = adapter.act({"action": "mirror_plan", "args": {}})

    assert planned.ok is False
    assert planned.raw["state"] == "MIRROR_PLAN_BLOCKED"
    assert planned.raw["mirror_plan_blocked_reason"] == "no_verified_changes"
    assert planned.raw["terminal"] is False


def test_competition_tools_are_hidden_until_two_hypotheses(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="investigate competing causes",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
    )

    first_tools = {tool.name for tool in adapter.reset().available_tools}
    assert "hypothesis_compete" not in first_tools
    assert "discriminating_test_add" not in first_tools

    assert adapter.act({"action": "hypothesis_add", "args": {"claim": "app.py is wrong"}}).ok is True
    one_hypothesis_tools = {tool.name for tool in adapter.observe().available_tools}
    assert "hypothesis_compete" not in one_hypothesis_tools
    assert "discriminating_test_add" not in one_hypothesis_tools

    assert adapter.act({"action": "hypothesis_add", "args": {"claim": "tests are stale"}}).ok is True
    two_hypotheses_tools = {tool.name for tool in adapter.observe().available_tools}
    assert "hypothesis_compete" in two_hypotheses_tools
    assert "discriminating_test_add" in two_hypotheses_tools


def test_investigation_notes_and_hypotheses_require_real_evidence_refs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="investigate app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
    )

    empty_note = adapter.act(
        {
            "action": "note_write",
            "args": {"kind": "finding", "content": "app.py might be wrong", "evidence_refs": []},
        }
    )
    bad_hypothesis = adapter.act(
        {
            "action": "hypothesis_add",
            "args": {"claim": "app.py might be wrong", "evidence_refs": ["note_0001"]},
        }
    )
    good_note = adapter.act(
        {
            "action": "note_write",
            "args": {
                "kind": "finding",
                "content": "app.py defines answer",
                "evidence_refs": ["file:app.py:1"],
            },
        }
    )
    good_hypothesis = adapter.act(
        {
            "action": "hypothesis_add",
            "args": {"claim": "answer is defined in app.py", "evidence_refs": ["note_0001"]},
        }
    )

    assert empty_note.ok is False
    assert "requires evidence_refs" in empty_note.raw["failure_reason"]
    assert bad_hypothesis.ok is False
    assert "unknown evidence note reference" in bad_hypothesis.raw["failure_reason"]
    assert good_note.ok is True
    assert good_hypothesis.ok is True


def test_atomic_action_governance_blocks_patch_without_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fix app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
    )

    patched = adapter.act(
        {
            "action": "apply_patch",
            "args": {
                "patch": "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def answer():\n-    return 1\n+    return 2\n",
                "max_files": 1,
                "max_hunks": 1,
            },
        }
    )

    assert patched.ok is False
    assert patched.raw["action_governance"]["status"] == "BLOCKED"
    assert "evidence_refs_required_before_mirror_write" in patched.raw["failure_reason"]
    assert (source / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 1\n"


def test_atomic_action_governance_allows_patch_after_file_read_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fix app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
    )

    read = adapter.act({"action": "file_read", "args": {"path": "app.py", "start_line": 1, "end_line": 2}})
    assert read.ok is True
    patched = adapter.act(
        {
            "action": "apply_patch",
            "args": {
                "patch": "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def answer():\n-    return 1\n+    return 2\n",
                "max_files": 1,
                "max_hunks": 1,
            },
        }
    )

    assert patched.ok is True
    assert patched.raw["action_governance"]["status"] == "ALLOWED"
    assert (mirror_root / "workspace" / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 2\n"


def test_apply_patch_accepts_hunk_with_context_line_offset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("header\nline old\nfooter\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fix app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
    )

    assert adapter.act({"action": "file_read", "args": {"path": "app.py", "start_line": 1, "end_line": 3}}).ok is True
    patched = adapter.act(
        {
            "action": "apply_patch",
            "args": {
                "patch": "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n line old\n-footer\n+footer changed\n",
                "max_files": 1,
                "max_hunks": 1,
            },
        }
    )

    assert patched.ok is True
    assert (mirror_root / "workspace" / "app.py").read_text(encoding="utf-8") == "header\nline old\nfooter changed\n"


def test_apply_patch_accepts_context_only_hunk_header(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fix app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
    )

    assert adapter.act({"action": "file_read", "args": {"path": "app.py", "start_line": 1, "end_line": 2}}).ok is True
    patched = adapter.act(
        {
            "action": "apply_patch",
            "args": {
                "patch": "--- a/app.py\n+++ b/app.py\n@@\n-    return 1\n+    return 2\n",
                "max_files": 1,
                "max_hunks": 1,
            },
        }
    )

    assert patched.ok is True
    assert (mirror_root / "workspace" / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 2\n"


def test_mirror_plan_gate_requires_verified_completion_before_code_sync(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fix app",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
        expose_apply_tool=True,
        execution_backend="local",
    )

    assert adapter.act({"action": "file_read", "args": {"path": "app.py", "start_line": 1, "end_line": 2}}).ok is True
    assert adapter.act(
        {
            "action": "apply_patch",
            "args": {
                "patch": "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def answer():\n-    return 1\n+    return 2\n",
                "max_files": 1,
                "max_hunks": 1,
            },
        }
    ).ok is True
    planned = adapter.act({"action": "mirror_plan", "args": {}})
    assert planned.ok is False
    assert planned.raw["state"] == "MIRROR_PLAN_BLOCKED"
    assert planned.raw["mirror_plan_blocked_reason"] == "no_verified_changes"
    assert (source / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 1\n"

    lint = adapter.act({"action": "run_lint", "args": {"target": "app.py", "timeout_seconds": 10}})
    assert lint.ok is True
    still_blocked = adapter.act({"action": "mirror_plan", "args": {}})
    assert still_blocked.ok is False
    assert still_blocked.raw["mirror_plan_blocked_reason"] == "no_verified_changes"
    assert (source / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 1\n"


def test_mirror_exec_fallback_rejects_unbounded_generated_commands(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="inspect source",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        allow_empty_exec=True,
    )

    missing_purpose = adapter.act({"function_name": "mirror_exec", "kwargs": {"command": [sys.executable, "-c", "print(1)"], "timeout_seconds": 5}})
    assert missing_purpose.ok is False
    assert "requires purpose" in missing_purpose.raw["failure_reason"]

    long_script = adapter.act(
        {
            "function_name": "mirror_exec",
            "kwargs": {
                "command": [sys.executable, "-c", "x=1\n" * 200],
                "purpose": "inspect",
                "target": ".",
                "timeout_seconds": 5,
            },
        }
    )
    assert long_script.ok is False
    assert "python -c script is too long" in long_script.raw["failure_reason"]


def test_empty_acquire_without_matches_does_not_repeat_as_only_action(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("readme\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="build a new AI project",
        source_root=source,
        mirror_root=mirror_root,
        candidate_paths=["README.md"],
        default_command=[sys.executable, "-c", "print('ready')"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        allow_empty_exec=True,
    )

    adapter.reset()
    acquired = adapter.act({"function_name": "mirror_acquire"})
    tools_after_empty_acquire = [tool.name for tool in acquired.observation.available_tools]

    assert acquired.raw["selected_paths"] == []
    assert tools_after_empty_acquire == ["mirror_exec"]


def test_empty_exec_generates_files_then_daemon_plan_waits_for_approval(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="build a new AI project from an empty mirror",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path('generated').mkdir(); Path('generated/app.py').write_text('print(\"ok\")\\n', encoding='utf-8')",
        ],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        terminal_after_plan=False,
        allow_empty_exec=True,
        execution_backend="local",
    )

    adapter.reset()
    executed = adapter.act({"function_name": "mirror_exec"})
    assert executed.ok is True
    assert (mirror_root / "workspace" / "generated" / "app.py").exists()
    assert [tool.name for tool in executed.observation.available_tools] == ["mirror_plan"]

    planned = adapter.act({"function_name": "mirror_plan"})

    assert planned.raw["state"] == "MIRROR_PLAN_BLOCKED"
    assert planned.raw["mirror_plan_blocked_reason"] == "no_verified_changes"


def test_default_command_timeout_is_configurable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="run a slow builder",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path('done.txt').write_text('ok\\n', encoding='utf-8')",
        ],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        allow_empty_exec=True,
        default_command_timeout_seconds=120,
        execution_backend="local",
    )

    adapter.reset()
    executed = adapter.act({"function_name": "mirror_exec"})

    assert executed.ok is True
    assert executed.raw["mirror_command"]["timeout_seconds"] == 120
    assert executed.raw["action_governance"]["status"] == "ALLOWED"


def test_local_machine_blocks_sensitive_extra_env_before_execution(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    captured = {"called": False}

    class Completed:
        returncode = 0
        stdout = "should not run\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["called"] = True
        return Completed()

    monkeypatch.setattr(mirror_module.subprocess, "run", fake_run)
    adapter = LocalMachineSurfaceAdapter(
        instruction="run with a credential env",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[sys.executable, "-c", "print('blocked')"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        allow_empty_exec=True,
        execution_backend="local",
        extra_env={"OPENAI_API_KEY": "secret-value"},
    )

    adapter.reset()
    result = adapter.act({"function_name": "mirror_exec"})

    assert result.ok is False
    assert captured["called"] is False
    assert result.raw["state"] == "BLOCKED"
    assert "inline_credentials_not_allowed" in result.raw["failure_reason"]
    assert result.raw["action_governance"]["request"]["metadata"]["credential_env_keys"] == ["OPENAI_API_KEY"]
    assert "secret-value" not in json.dumps(result.raw)


def test_local_machine_allows_leased_credential_env_with_redacted_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="run with a leased credential env",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[
            sys.executable,
            "-c",
            "import os; print(os.environ.get('OPENAI_API_KEY', 'missing'))",
        ],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        allow_empty_exec=True,
        execution_backend="local",
        credential_env_leases={"OPENAI_API_KEY": {"lease_id": "lease_openai_test", "value": "secret-value"}},
    )

    adapter.reset()
    result = adapter.act({"function_name": "mirror_exec"})

    assert result.ok is True
    assert result.raw["action_governance"]["status"] == "ALLOWED"
    assert "credential_access" in result.raw["action_governance"]["effective_permissions"]
    assert result.raw["mirror_command"]["stdout"].strip() == "<redacted:OPENAI_API_KEY>"
    assert result.raw["mirror_command"]["env_audit"]["sensitive_explicit_env_keys"] == ["OPENAI_API_KEY"]
    assert result.raw["side_effect_audit"]["env_audit"]["sensitive_explicit_env_keys"] == ["OPENAI_API_KEY"]
    assert "secret-value" not in json.dumps(result.raw)


def test_local_machine_leased_credential_can_require_approval(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    adapter = LocalMachineSurfaceAdapter(
        instruction="run with a leased credential env",
        source_root=source,
        mirror_root=tmp_path / "mirror",
        default_command=[sys.executable, "-c", "print('should wait')"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        allow_empty_exec=True,
        execution_backend="local",
        credential_env_leases={"OPENAI_API_KEY": {"lease_id": "lease_openai_test", "value": "secret-value"}},
        action_governance_policy={"approval_required_capability_layers": ["credential"]},
    )

    adapter.reset()
    result = adapter.act({"function_name": "mirror_exec"})

    assert result.ok is False
    assert result.raw["state"] == "WAITING_APPROVAL"
    assert "capability_layer_requires_approval:credential" in result.raw["failure_reason"]
    assert result.raw["action_governance"]["request"]["metadata"]["credential_lease_ids"] == ["lease_openai_test"]
    assert "secret-value" not in json.dumps(result.raw)


def test_local_machine_mirror_exec_uses_configured_docker_boundary(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    captured = {}

    class Completed:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return Completed()

    monkeypatch.setattr(mirror_module.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(mirror_module.subprocess, "run", fake_run)
    adapter = LocalMachineSurfaceAdapter(
        instruction="build in a container boundary",
        source_root=source,
        mirror_root=mirror_root,
        default_command=[sys.executable, "-c", "print('ok')"],
        allowed_commands=[sys.executable],
        reset_mirror=True,
        allow_empty_exec=True,
        execution_backend="docker",
    )

    adapter.reset()
    executed = adapter.act({"function_name": "mirror_exec"})

    assert executed.ok is True
    assert executed.raw["mirror_command"]["backend"] == "docker"
    assert executed.raw["execution_boundary"]["security_boundary"] == "container_best_effort"
    assert executed.raw["execution_boundary"]["network_boundary"] == "none"
    assert executed.observation.raw["local_mirror"]["execution_backend"] == "docker"
    assert captured["cmd"][:5] == ["/usr/bin/docker", "run", "--rm", "--network", "none"]


def test_local_machine_validation_uses_configured_execution_boundary(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    captured = {}

    class Completed:
        returncode = 0
        stdout = "1 passed\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return Completed()

    monkeypatch.setattr(mirror_module.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(mirror_module.subprocess, "run", fake_run)
    adapter = LocalMachineSurfaceAdapter(
        instruction="verify in a container boundary",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        execution_backend="docker",
    )

    adapter.reset()
    result = adapter.act({"function_name": "run_test", "kwargs": {"target": "tests/test_smoke.py", "timeout_seconds": 5}})

    assert result.ok is True
    assert result.raw["mirror_command"]["backend"] == "docker"
    assert result.raw["execution_boundary"]["security_boundary"] == "container_best_effort"
    assert captured["cmd"][:5] == ["/usr/bin/docker", "run", "--rm", "--network", "none"]
    assert captured["cmd"][-4:] == ["python", "-m", "pytest", "tests/test_smoke.py"]


def test_require_artifacts_marks_empty_run_failed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("readme\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"

    audit = run_local_machine_task(
        instruction="build a new AI project",
        source_root=str(source),
        mirror_root=str(mirror_root),
        candidate_paths=["README.md"],
        run_id="artifact-contract-empty",
        max_ticks_per_episode=1,
        reset_mirror=True,
        require_artifacts=True,
        execution_backend="local",
    )

    check = audit["local_machine_artifact_check"]
    assert check["ok"] is False
    assert "command_executed" in check["failures"]
    assert "sync_plan_present" in check["failures"]


def test_require_artifacts_rejects_failed_builder_even_with_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"

    audit = run_local_machine_task(
        instruction="build a new AI project",
        source_root=str(source),
        mirror_root=str(mirror_root),
        default_command=[
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "Path('generated').mkdir(); "
                "Path('generated/app.py').write_text('print(\"partial\")\\n', encoding='utf-8'); "
                "raise SystemExit(1)"
            ),
        ],
        allowed_commands=[sys.executable],
        run_id="artifact-contract-failed-builder",
        max_ticks_per_episode=3,
        reset_mirror=True,
        allow_empty_exec=True,
        require_artifacts=True,
        required_artifact_paths=["generated/app.py"],
        execution_backend="local",
    )

    check = audit["local_machine_artifact_check"]
    assert check["ok"] is False
    assert check["latest_command_returncode"] == 1
    assert "latest_command_succeeded" in check["failures"]
    assert check["required_workspace_path_matches"]["generated/app.py"] == ["generated/app.py"]


def test_require_artifacts_accepts_safe_refusal_after_evidence() -> None:
    audit = {
        "final_surface_raw": {
            "local_mirror": {
                "command_executed": True,
                "workspace_file_count": 3,
                "terminal_state": "needs_human_review",
                "audit_events": [
                    {
                        "event_type": "mirror_command_executed",
                        "payload": {"returncode": 0},
                    }
                ],
                "investigation": {
                    "refusal_reason": "evidence_insufficient",
                    "read_files": [{"path": "README.md"}, {"path": "src/pkg/core.py"}],
                    "validation_runs": [{"run_ref": "run_ok", "success": True}],
                },
            }
        }
    }

    check = _artifact_contract_check(audit)

    assert check["ok"] is True
    assert check["safe_refusal_terminal"] is True
    assert check["safe_refusal_reason"] == "evidence_insufficient"
    assert "sync_plan_present" not in check["failures"]
    assert "actionable_changes_present" not in check["failures"]
    assert check["checks"]["safe_refusal_evidence_present"] is True


def test_require_artifacts_accepts_verifier_rejected_patch_after_rollback() -> None:
    audit = {
        "final_surface_raw": {
            "local_mirror": {
                "command_executed": True,
                "workspace_file_count": 3,
                "terminal_state": "needs_human_review",
                "audit_events": [
                    {
                        "event_type": "mirror_command_executed",
                        "payload": {"returncode": 1},
                    }
                ],
                "investigation": {
                    "refusal_reason": "verifier_rejected_patch",
                    "read_files": [{"path": "README.md"}, {"path": "src/pkg/core.py"}],
                    "validation_runs": [
                        {"run_ref": "run_ok", "success": True},
                        {"run_ref": "run_rejected", "success": False},
                    ],
                    "patch_proposals": [
                        {
                            "event_type": "patch_proposal_rejected",
                            "rollback_count": 1,
                            "test_results": [
                                {"target": "tests/test_core.py", "success": False, "returncode": 1}
                            ],
                        }
                    ],
                },
            }
        }
    }

    check = _artifact_contract_check(audit)

    assert check["ok"] is True
    assert check["safe_refusal_terminal"] is True
    assert check["safe_refusal_reason"] == "verifier_rejected_patch"
    assert check["checks"]["verifier_rejection_evidence_present"] is True
    assert check["checks"]["latest_command_failed_by_verifier"] is True
    assert "latest_command_succeeded" not in check["failures"]


def test_require_artifacts_rejects_unsafe_refusal_without_changes() -> None:
    audit = {
        "final_surface_raw": {
            "local_mirror": {
                "command_executed": False,
                "workspace_file_count": 1,
                "terminal_state": "needs_human_review",
                "investigation": {
                    "refusal_reason": "llm_patch_proposal_unavailable",
                    "read_files": [{"path": "README.md"}],
                },
            }
        }
    }

    check = _artifact_contract_check(audit)

    assert check["ok"] is False
    assert check["safe_refusal_terminal"] is False
    assert "sync_plan_present" in check["failures"]
    assert "actionable_changes_present" in check["failures"]


def test_daemon_rejects_failed_builder_without_waiting_approval(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    db_path = tmp_path / "state.sqlite3"
    run_id = "daemon-artifact-contract-failed-builder"

    audit = run_local_machine_task(
        instruction="build a new AI project",
        source_root=str(source),
        mirror_root=str(mirror_root),
        default_command=[
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "Path('generated').mkdir(); "
                "Path('generated/app.py').write_text('print(\"partial\")\\n', encoding='utf-8'); "
                "raise SystemExit(1)"
            ),
        ],
        allowed_commands=[sys.executable],
        run_id=run_id,
        max_ticks_per_episode=3,
        reset_mirror=True,
        daemon=True,
        supervisor_db=str(db_path),
        allow_empty_exec=True,
        require_artifacts=True,
        required_artifact_paths=["generated/app.py"],
        execution_backend="local",
    )

    check = audit["local_machine_artifact_check"]
    supervisor_state = audit["long_run_supervisor"]
    store = RuntimeStateStore(db_path)
    tasks = store.list_tasks(run_id)

    assert check["ok"] is False
    assert check["latest_command_returncode"] == 1
    assert supervisor_state["run"]["status"] == "FAILED"
    assert supervisor_state["run"]["paused_reason"].startswith("artifact_contract_failed:")
    assert supervisor_state["latest_approval"] == {}
    assert tasks and tasks[0]["status"] == "FAILED"


def test_require_artifacts_checks_required_workspace_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"

    audit = run_local_machine_task(
        instruction="build a complete AI project",
        source_root=str(source),
        mirror_root=str(mirror_root),
        default_command=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path('generated').mkdir(); Path('generated/README.md').write_text('partial\\n', encoding='utf-8')",
        ],
        allowed_commands=[sys.executable],
        run_id="artifact-contract-missing-required-path",
        max_ticks_per_episode=3,
        reset_mirror=True,
        allow_empty_exec=True,
        require_artifacts=True,
        required_artifact_paths=["generated/*/pyproject.toml"],
        execution_backend="local",
    )

    check = audit["local_machine_artifact_check"]
    assert check["ok"] is False
    assert "required_workspace_path:generated/*/pyproject.toml" in check["failures"]


def test_require_artifacts_can_require_internet_research_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"

    audit = run_local_machine_task(
        instruction="research the AI market and build a complete AI project",
        source_root=str(source),
        mirror_root=str(mirror_root),
        default_command=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path('generated').mkdir(); Path('generated/README.md').write_text('ok\\n', encoding='utf-8')",
        ],
        allowed_commands=[sys.executable],
        run_id="artifact-contract-missing-internet-evidence",
        max_ticks_per_episode=3,
        reset_mirror=True,
        allow_empty_exec=True,
        require_artifacts=True,
        require_internet_artifact=True,
        required_artifact_paths=["generated/README.md"],
        execution_backend="local",
    )

    check = audit["local_machine_artifact_check"]
    assert check["ok"] is False
    assert "internet_artifact_present" in check["failures"]


def test_local_machine_planner_compiles_build_plan_from_context() -> None:
    class Goal:
        goal_id = "goal_explore"

    plan = ObjectiveDecomposer().decompose(
        Goal(),
        {
            "task_family": "local_machine",
            "domain": "local_machine",
            "environment_tags": ["local_machine"],
            "available_functions": ["mirror_exec", "mirror_acquire"],
            "default_command_present": True,
            "allow_empty_exec": True,
            "terminal_after_plan": False,
            "workspace_file_count": 0,
            "max_ticks": 6,
        },
    )

    assert plan.plan_id.startswith("local_machine_plan_")
    assert [step.target_function for step in plan.steps] == ["mirror_exec", "mirror_plan", None]
    assert plan.planning_contract["compiler"] == "local_machine_plan_compiler/v1"
    assert plan.verification_contract["require_artifacts_when_default_command_present"] is True


def test_local_machine_planner_inventories_before_mirror_exec_without_candidates() -> None:
    class Goal:
        goal_id = "goal_investigate_codebase"

    plan = ObjectiveDecomposer().decompose(
        Goal(),
        {
            "task_family": "local_machine",
            "domain": "local_machine",
            "environment_tags": ["local_machine"],
            "instruction": "调查 AGI3，发现一个实际可改进点并修复",
            "available_functions": [
                "repo_tree",
                "file_read",
                "repo_grep",
                "note_write",
                "apply_patch",
                "run_lint",
                "mirror_exec",
                "mirror_plan",
            ],
            "default_command_present": False,
            "allow_empty_exec": True,
            "workspace_file_count": 0,
            "max_ticks": 8,
            "local_mirror": {"candidate_files": []},
        },
    )

    targets = [step.target_function for step in plan.steps]
    assert targets[:4] == ["repo_tree", "file_read", "repo_grep", "note_write"]
    assert "mirror_exec" not in targets
    assert "mirror_plan" in targets
    note_step = next(step for step in plan.steps if step.target_function == "note_write")
    patch_step = next(step for step in plan.steps if step.target_function == "apply_patch")
    assert note_step.approval_requirement["required"] is False
    assert note_step.approval_requirement["risk_level"] == "low"
    assert note_step.approval_requirement["allow_high_risk_without_approval"] is True
    assert patch_step.approval_requirement["required"] is False
    assert patch_step.approval_requirement["risk_level"] == "medium"
    assert patch_step.approval_requirement["allow_high_risk_without_approval"] is True
    assert patch_step.constraints["required"] is True
    assert "sync_plan.actionable_change_count > 0" in plan.verification_contract["success_criteria"]


def test_local_machine_mirror_control_write_overrides_high_risk_approval_default() -> None:
    requirement = resolve_effective_task_approval_requirement(
        {
            "goal_id": "goal-1",
            "title": "local machine task",
            "approval": {"require_explicit_approval_for_high_risk": True},
        },
        {
            "node_id": "task-1",
            "title": "Persist the key investigation finding with evidence references",
            "status": "active",
            "goal_id": "goal-1",
            "approval_requirement": {
                "required": False,
                "risk_level": "low",
                "reason": "local_machine_mirror_control_write",
                "allow_high_risk_without_approval": True,
            },
        },
        function_name="note_write",
        capability_class="local_machine_investigation_state",
    )

    assert requirement["required"] is False
    assert requirement["allow_high_risk_without_approval"] is True


def test_local_machine_planner_research_build_upload_flow() -> None:
    class Goal:
        goal_id = "goal_market_ai_product"

    plan = ObjectiveDecomposer().decompose(
        Goal(),
        {
            "task_family": "local_machine",
            "domain": "local_machine",
            "environment_tags": ["local_machine"],
            "instruction": "调研市场，制作产品级 AI tool，然后上传 GitHub",
            "available_functions": ["internet_fetch", "mirror_exec", "mirror_plan"],
            "default_command_present": False,
            "allow_empty_exec": True,
            "workspace_file_count": 0,
            "max_ticks": 8,
        },
    )

    assert [step.target_function for step in plan.steps] == ["internet_fetch", "mirror_exec", "mirror_plan"]


def test_local_machine_planner_does_not_fetch_project_for_negated_clone_instruction() -> None:
    class Goal:
        goal_id = "goal_market_ai_product"

    plan = ObjectiveDecomposer().decompose(
        Goal(),
        {
            "task_family": "local_machine",
            "domain": "local_machine",
            "environment_tags": ["local_machine"],
            "instruction": (
                "Research the AI tools market, then create a product-grade AI tool from scratch. "
                "Choose a useful opportunity, not a clone of an existing repository."
            ),
            "available_functions": ["internet_fetch", "internet_fetch_project", "mirror_exec", "mirror_plan"],
            "default_command_present": False,
            "allow_empty_exec": True,
            "workspace_file_count": 0,
            "max_ticks": 8,
        },
    )

    assert [step.target_function for step in plan.steps] == ["internet_fetch", "mirror_exec", "mirror_plan"]


def test_structured_answer_has_local_machine_market_product_fallbacks() -> None:
    synthesizer = StructuredAnswerSynthesizer()
    obs = {
        "local_mirror": {
            "instruction": "Research the AI tools market and create generated_product/",
            "internet_enabled": True,
            "workspace_root": "/tmp/workspace",
            "diff_summary": {"entry_count": 0, "status_counts": {}, "examples": []},
        },
        "function_signatures": {},
    }

    research_action = synthesizer.maybe_populate_action_kwargs(
        {"kind": "call_tool", "payload": {"tool_args": {"function_name": "internet_fetch", "kwargs": {}}}},
        obs,
    )
    research_kwargs = research_action["payload"]["tool_args"]["kwargs"]
    assert research_kwargs["url"] == "https://github.com/topics/ai-tools"

    exec_action = synthesizer.maybe_populate_action_kwargs(
        {"kind": "call_tool", "payload": {"tool_args": {"function_name": "mirror_exec", "kwargs": {}}}},
        obs,
    )
    exec_kwargs = exec_action["payload"]["tool_args"]["kwargs"]
    assert exec_kwargs["command"][:2] == ["python3", "-c"]
    assert "generated_product" in exec_kwargs["command"][2]
    assert "sys.path.insert" in exec_kwargs["command"][2]
    assert "LICENSE" in exec_kwargs["command"][2]
    assert ".gitignore" in exec_kwargs["command"][2]


def test_structured_answer_prompt_exposes_source_and_workspace_roots() -> None:
    synthesizer = StructuredAnswerSynthesizer()
    prompt = synthesizer._build_llm_prompt(
        "mirror_exec",
        {
            "local_mirror": {
                "instruction": "inspect source",
                "source_root": "/source/project",
                "mirror_root": "/mirror/root",
                "workspace_root": "/mirror/root/workspace",
                "control_root": "/mirror/root/control",
            },
            "function_signatures": {},
        },
    )

    assert "/source/project" in prompt
    assert "/mirror/root/workspace" in prompt


def test_structured_answer_prompt_dedupes_function_signatures_and_tree_context() -> None:
    synthesizer = StructuredAnswerSynthesizer()
    prompt = synthesizer._build_llm_prompt(
        "file_read",
        {
            "local_mirror": {
                "instruction": "read a file",
                "investigation": {
                    "last_tree": {
                        "root": ".",
                        "depth": 2,
                        "entry_count": 200,
                        "entries": [
                            {
                                "path": f"src/file_{idx}.py",
                                "name": f"file_{idx}.py",
                                "kind": "file",
                                "depth": 2,
                                "size_bytes": 123,
                            }
                            for idx in range(100)
                        ],
                    }
                },
            },
            "function_signatures": {
                "file_read": {"description": "read selected file"},
                "mirror_exec": {"description": "UNRELATED LARGE FALLBACK SCHEMA"},
            },
        },
    )

    assert "read selected file" in prompt
    assert "UNRELATED LARGE FALLBACK SCHEMA" not in prompt
    assert "\"entries_truncated\": true" in prompt
    assert "src/file_79.py" in prompt
    assert "src/file_80.py" not in prompt


def test_structured_answer_prompt_compacts_line_based_file_reads() -> None:
    synthesizer = StructuredAnswerSynthesizer()
    prompt = synthesizer._build_llm_prompt(
        "apply_patch",
        {
            "local_mirror": {
                "instruction": "patch the inspected file",
                "investigation": {
                    "last_read": {
                        "path": "core/runtime_budget.py",
                        "start_line": 10,
                        "end_line": 12,
                        "line_count": 3,
                        "lines": [
                            {"line": 10, "text": "def budget():"},
                            {"line": 11, "text": "    return 1"},
                        ],
                    }
                },
            },
            "function_signatures": {"apply_patch": {"description": "apply bounded patch"}},
        },
    )

    assert "core/runtime_budget.py" in prompt
    assert "10: def budget():" in prompt
    assert "11:     return 1" in prompt


def test_structured_answer_can_disable_local_machine_deterministic_fallback() -> None:
    synthesizer = StructuredAnswerSynthesizer()
    obs = {
        "local_mirror": {
            "instruction": "Research the AI tools market and create generated_product/",
            "internet_enabled": True,
            "deterministic_fallback_enabled": False,
        },
        "function_signatures": {},
    }

    action = {"kind": "call_tool", "payload": {"tool_args": {"function_name": "internet_fetch", "kwargs": {}}}}

    updated = synthesizer.maybe_populate_action_kwargs(action, obs)

    assert updated["payload"]["tool_args"]["kwargs"] == {}


def test_structured_answer_records_llm_prompt_response_and_tool_kwargs() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.kwargs = {}

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            assert "Context:" in prompt
            self.kwargs = dict(kwargs)
            return 'KWARGS_JSON: {"command": ["python3", "-c", "print(42)"], "timeout_seconds": 5}'

    llm = FakeLLM()
    synthesizer = StructuredAnswerSynthesizer()
    obs = {
        "local_mirror": {
            "instruction": "Use a model to create a product",
            "internet_enabled": True,
            "deterministic_fallback_enabled": False,
            "require_llm_generation": True,
        },
        "function_signatures": {},
    }

    updated = synthesizer.maybe_populate_action_kwargs(
        {"kind": "call_tool", "payload": {"tool_args": {"function_name": "mirror_exec", "kwargs": {}}}},
        obs,
        llm_client=llm,
    )

    meta = updated["_candidate_meta"]
    trace = meta["structured_answer_llm_trace"][0]
    assert meta["structured_answer_strategy"] == "llm_draft"
    assert trace["prompt"]
    assert trace["system_prompt"]
    assert "KWARGS_JSON" in trace["response"]
    assert trace["parsed_kwargs"]["command"][:2] == ["python3", "-c"]
    assert llm.kwargs["think"] is False
    assert llm.kwargs["max_tokens"] <= 256
    assert llm.kwargs["timeout_sec"] <= 8.0


def test_structured_answer_can_prefer_llm_kwargs_before_fallback() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            return 'KWARGS_JSON: {"command": ["python3", "-c", "print(99)"], "timeout_seconds": 5}'

    llm = FakeLLM()
    synthesizer = StructuredAnswerSynthesizer()
    obs = {
        "local_mirror": {
            "instruction": "Use a model to create an AI product",
            "deterministic_fallback_enabled": True,
            "prefer_llm_kwargs": True,
        },
        "function_signatures": {},
    }

    updated = synthesizer.maybe_populate_action_kwargs(
        {"kind": "call_tool", "payload": {"tool_args": {"function_name": "mirror_exec", "kwargs": {}}}},
        obs,
        llm_client=llm,
    )

    meta = updated["_candidate_meta"]
    kwargs = updated["payload"]["tool_args"]["kwargs"]
    assert llm.calls == 1
    assert meta["structured_answer_strategy"] == "llm_draft"
    assert meta["structured_answer_llm_trace"][0]["parsed_kwargs"]["command"][-1] == "print(99)"
    assert kwargs["command"][-1] == "print(99)"


def test_structured_answer_records_llm_trace_when_preferred_llm_falls_back() -> None:
    class FakeLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            return "not json"

    synthesizer = StructuredAnswerSynthesizer()
    obs = {
        "local_mirror": {
            "instruction": "Use a model to create an AI product",
            "deterministic_fallback_enabled": True,
            "prefer_llm_kwargs": True,
        },
        "function_signatures": {},
    }

    updated = synthesizer.maybe_populate_action_kwargs(
        {"kind": "call_tool", "payload": {"tool_args": {"function_name": "mirror_exec", "kwargs": {}}}},
        obs,
        llm_client=FakeLLM(),
    )

    meta = updated["_candidate_meta"]
    assert meta["structured_answer_strategy"] == "local_machine_fallback"
    assert meta["structured_answer_fallback_used"] is True
    assert meta["structured_answer_llm_trace"][0]["response"] == "not json"
    assert meta["structured_answer_llm_trace"][0]["parsed_kwargs"] == {}


def test_patch_proposal_can_record_bounded_llm_diff(tmp_path: Path) -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            if self.calls == 1:
                return "The helper loses parent path information."
            if self.calls == 2:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["path object was truncated"], '
                    '"hypothesis": {"summary": "path helper returns only name", "target_file": "app/paths.py"}, '
                    '"decision": "patch", "next_action": {"action": "propose_bounded_diff", "target_file": "app/paths.py"}, '
                    '"confidence": 0.84, "failure_boundary": ["full tests must pass"], '
                    '"patch_intent": "return the full path object"}}'
                )
            return (
                'PATCH_JSON: {"unified_diff": "--- a/app/paths.py\\n+++ b/app/paths.py\\n'
                '@@ -1,2 +1,2 @@\\n def normalize(path):\\n-    return path.name\\n+    return path\\n", '
                '"rationale": "preserve the full path object", "expected_tests": ["."], "risk": 0.2}'
            )

    target = tmp_path / "app" / "paths.py"
    target.parent.mkdir(parents=True)
    target.write_text("def normalize(path):\n    return path.name\n", encoding="utf-8")

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix path resolution",
            "investigation_state": {
                "target_binding": {"top_target_file": "app/paths.py", "target_confidence": 0.8},
                "hypotheses": [
                    {
                        "hypothesis_id": "h_downstream",
                        "status": "leading",
                        "summary": "Path helper loses parent path information.",
                        "target_file": "app/paths.py",
                    }
                ],
            },
        },
        top_target_file="app/paths.py",
        llm_client=FakeLLM(),
    )

    assert payload["llm_trace"][0]["error"] == ""
    assert payload["patch_proposals"][0]["proposal_source"] == "bounded_llm_diff"
    assert payload["patch_proposals"][0]["target_file"] == "app/paths.py"


def test_patch_proposal_accepts_llm_diff_with_context_line_offset(tmp_path: Path) -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            if self.calls == 1:
                return "The helper loses parent path information."
            if self.calls == 2:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["path object was truncated"], '
                    '"hypothesis": {"summary": "path helper returns only name", "target_file": "app/paths.py"}, '
                    '"decision": "patch", "next_action": {"action": "propose_bounded_diff", "target_file": "app/paths.py"}, '
                    '"confidence": 0.84, "failure_boundary": ["full tests must pass"], '
                    '"patch_intent": "return the full path object"}}'
                )
            return (
                'PATCH_JSON: {"unified_diff": "--- a/app/paths.py\\n+++ b/app/paths.py\\n'
                '@@ -1,2 +1,2 @@\\n def normalize(path):\\n-    return path.name\\n+    return path\\n", '
                '"rationale": "preserve the full path object", "expected_tests": ["."], "risk": 0.2}'
            )

    target = tmp_path / "app" / "paths.py"
    target.parent.mkdir(parents=True)
    target.write_text("# helpers\n\ndef normalize(path):\n    return path.name\n", encoding="utf-8")

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix path resolution",
            "investigation_state": {
                "target_binding": {"top_target_file": "app/paths.py", "target_confidence": 0.8},
                "hypotheses": [
                    {
                        "hypothesis_id": "h_downstream",
                        "status": "leading",
                        "summary": "Path helper loses parent path information.",
                        "target_file": "app/paths.py",
                    }
                ],
            },
        },
        top_target_file="app/paths.py",
        llm_client=FakeLLM(),
    )

    assert payload["llm_trace"][0]["error"] == ""
    assert payload["patch_proposals"][0]["target_file"] == "app/paths.py"


def test_patch_proposal_accepts_llm_diff_when_hunk_header_starts_too_late(tmp_path: Path) -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            if self.calls == 1:
                return "The helper should use fnmatch for patterns."
            if self.calls == 2:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["exact matching misses wildcard rules"], '
                    '"hypothesis": {"summary": "matcher ignores wildcard semantics", "target_file": "app/matcher.py"}, '
                    '"decision": "patch", "next_action": {"action": "propose_bounded_diff", "target_file": "app/matcher.py"}, '
                    '"confidence": 0.82, "failure_boundary": ["full tests must pass"], '
                    '"patch_intent": "use fnmatch for patterns"}}'
                )
            return (
                'PATCH_JSON: {"unified_diff": "--- a/app/matcher.py\\n+++ b/app/matcher.py\\n'
                '@@ -4,5 +4,6 @@\\n def match(name, patterns):\\n     for pattern in patterns:\\n'
                '-        if name == pattern:\\n+        if fnmatch.fnmatch(name, pattern):\\n'
                '             return True\\n     return False\\n", '
                '"rationale": "support wildcard matching", "expected_tests": ["."], "risk": 0.2}'
            )

    target = tmp_path / "app" / "matcher.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import fnmatch\n\n\n"
        "def match(name, patterns):\n"
        "    for pattern in patterns:\n"
        "        if name == pattern:\n"
        "            return True\n"
        "    return False\n",
        encoding="utf-8",
    )

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Improve wildcard matching",
            "investigation_state": {
                "target_binding": {"top_target_file": "app/matcher.py", "target_confidence": 0.7},
                "hypotheses": [
                    {
                        "hypothesis_id": "h_match",
                        "status": "leading",
                        "summary": "Matcher ignores wildcard semantics.",
                        "target_file": "app/matcher.py",
                    }
                ],
            },
        },
        top_target_file="app/matcher.py",
        llm_client=FakeLLM(),
    )

    assert payload["patch_proposals"]
    assert payload["patch_proposals"][0]["proposal_source"] == "bounded_llm_diff"


def test_patch_proposal_timeout_does_not_fall_back_to_deterministic_patch(tmp_path: Path) -> None:
    class TimeoutLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            raise TimeoutError("remote model timeout")

    target = tmp_path / "app" / "discounts.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "def discount(total, threshold):\n"
        "    if total > threshold:\n"
        "        return 10\n"
        "    return 0\n",
        encoding="utf-8",
    )
    run_output_root = tmp_path / ".runs"
    run_output_root.mkdir()
    (run_output_root / "run_failed.json").write_text(
        json.dumps(
            {
                "run_ref": "run_failed",
                "command": ["pytest", "tests/test_discounts.py"],
                "stdout": "boundary threshold exact equal case failed",
                "stderr": "",
            }
        ),
        encoding="utf-8",
    )
    context = {
        "source_root": str(tmp_path),
        "workspace_root": str(tmp_path),
        "run_output_root": str(run_output_root),
        "investigation_state": {
            "validation_runs": [{"run_ref": "run_failed", "success": False}],
            "target_binding": {"top_target_file": "app/discounts.py", "target_confidence": 0.8},
        },
    }

    deterministic = generate_patch_proposals(context, top_target_file="app/discounts.py")
    timed_out = generate_patch_proposals(
        context,
        top_target_file="app/discounts.py",
        llm_client=TimeoutLLM(),
    )

    assert deterministic["patch_proposals"]
    assert timed_out["patch_proposals"] == []
    assert timed_out["llm_timeout"] is True
    assert timed_out["fallback_patch_enabled"] is False
    assert timed_out["deterministic_fallback_proposal_count"] == 0
    assert timed_out["refusal_reason"] == "timeout"


def test_patch_proposal_budget_exhaustion_is_reported_without_fallback(tmp_path: Path) -> None:
    class BudgetLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            raise RuntimeError("llm_budget_exceeded:max_llm_calls_exceeded")

    target = tmp_path / "app" / "discounts.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "def discount(total, threshold):\n"
        "    if total > threshold:\n"
        "        return 10\n"
        "    return 0\n",
        encoding="utf-8",
    )
    run_output_root = tmp_path / ".runs"
    run_output_root.mkdir()
    context = {
        "source_root": str(tmp_path),
        "workspace_root": str(tmp_path),
        "run_output_root": str(run_output_root),
        "investigation_state": {
            "target_binding": {"top_target_file": "app/discounts.py", "target_confidence": 0.8},
        },
    }

    payload = generate_patch_proposals(
        context,
        top_target_file="app/discounts.py",
        llm_client=BudgetLLM(),
    )

    assert payload["patch_proposals"] == []
    assert payload["llm_timeout"] is False
    assert payload["fallback_patch_enabled"] is False
    assert payload["deterministic_fallback_proposal_count"] == 0
    assert payload["refusal_reason"] == "llm_budget_exceeded"


def test_patch_proposal_empty_target_content_is_safe_evidence_refusal(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "__init__.py"
    target.parent.mkdir(parents=True)
    target.write_text("", encoding="utf-8")

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "investigation_state": {
                "target_binding": {"top_target_file": "pkg/__init__.py", "target_confidence": 0.8},
            },
        },
        top_target_file="pkg/__init__.py",
    )

    assert payload["patch_proposals"] == []
    assert payload["refusal_reason"] == "evidence_insufficient"
    assert payload["rejection_reason"] == "target file content is unavailable"


def test_patch_proposal_uses_think_distill_act_without_storing_raw_thinking(tmp_path: Path) -> None:
    class PipelineLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
            call_number = len(self.calls)
            if call_number == 1:
                return "<think>private chain of thought that must not be stored</think>\nworking notes"
            if call_number == 2:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["failure proves path object was truncated"], '
                    '"hypothesis": {"summary": "path helper returns only name", "target_file": "app/paths.py"}, '
                    '"decision": "patch", '
                    '"next_action": {"action": "propose_bounded_diff", "target_file": "app/paths.py"}, '
                    '"confidence": 0.82, '
                    '"failure_boundary": ["do not modify tests"], '
                    '"patch_intent": "return the original path object"}}'
                )
            return (
                'PATCH_JSON: {"unified_diff": "--- a/app/paths.py\\n+++ b/app/paths.py\\n'
                '@@ -1,2 +1,2 @@\\n def normalize(path):\\n-    return path.name\\n+    return path\\n", '
                '"rationale": "use distilled state to preserve path", "expected_tests": ["."], "risk": 0.2}'
            )

    target = tmp_path / "app" / "paths.py"
    target.parent.mkdir(parents=True)
    target.write_text("def normalize(path):\n    return path.name\n", encoding="utf-8")

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix path resolution",
            "investigation_state": {
                "target_binding": {"top_target_file": "app/paths.py", "target_confidence": 0.8},
                "hypotheses": [
                    {
                        "hypothesis_id": "h_downstream",
                        "status": "leading",
                        "summary": "Path helper loses parent path information.",
                        "target_file": "app/paths.py",
                    }
                ],
            },
        },
        top_target_file="app/paths.py",
        llm_client=PipelineLLM(),
    )

    traces = payload["llm_trace"]
    assert [row["stage"] for row in traces] == ["think_pass", "distill_pass", "act_pass"]
    assert traces[0]["request_kwargs"]["think"] is True
    assert traces[1]["request_kwargs"]["think"] is False
    assert traces[2]["request_kwargs"]["think"] is False
    assert traces[0]["response"] == ""
    assert traces[0]["raw_response_discarded"] is True
    assert traces[1]["raw_thinking_discarded"] is True
    assert "private chain of thought" not in json.dumps(traces, ensure_ascii=False)
    assert payload["patch_proposals"][0]["pipeline"] == "think_distill_act"
    assert payload["patch_proposals"][0]["reasoning_state"]["decision"] == "patch"


def test_patch_proposal_reasoning_gate_can_refuse_without_act_pass(tmp_path: Path) -> None:
    class RefusalLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
            if "Distill the raw thinking" in prompt:
                assert "read_file_paths" in prompt
                assert "tests/test_core.py" in prompt
                assert "successful_validation_count" in prompt
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["overview, tests, and source were inspected"], '
                    '"hypothesis": {"summary": "no safe improvement target is proven", "target_file": "pkg/core.py"}, '
                    '"decision": "requires_further_evidence", '
                    '"next_action": {"action": "needs_human_review", "target_file": "pkg/core.py"}, '
                    '"confidence": 0.42, '
                    '"failure_boundary": ["evidence insufficient for a safe source patch"], '
                    '"patch_intent": "needs_human_review"}}'
                )
            if "Generate one minimal unified diff" in prompt:
                raise AssertionError("act pass should not run after reasoning-state refusal")
            return "No obvious bounded patch is justified."

    target = tmp_path / "pkg" / "core.py"
    target.parent.mkdir(parents=True)
    target.write_text("def value():\n    return 1\n", encoding="utf-8")

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Improve this repository with one small low-risk source patch.",
            "investigation_state": {
                "read_files": [
                    {"path": "README.md"},
                    {"path": "tests/test_core.py"},
                    {"path": "pkg/core.py"},
                ],
                "validation_runs": [{"run_ref": "run_ok", "success": True, "returncode": 0, "target": "tests/test_core.py"}],
                "target_binding": {"top_target_file": "pkg/core.py", "target_confidence": 0.64},
                "hypotheses": [
                    {
                        "hypothesis_id": "h_core",
                        "status": "leading",
                        "summary": "core.py is the best available candidate, but evidence is weak.",
                        "target_file": "pkg/core.py",
                    }
                ],
            },
        },
        top_target_file="pkg/core.py",
        llm_client=RefusalLLM(),
    )

    assert payload["patch_proposals"] == []
    assert payload["refusal_reason"] == "evidence_insufficient"
    assert [row["stage"] for row in payload["llm_trace"]] == ["think_pass", "distill_pass", "reasoning_gate"]
    assert payload["llm_trace"][-1]["needs_human_review"] is True


def test_patch_proposal_accepts_direct_reasoning_state_shape(tmp_path: Path) -> None:
    class DirectStateLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            if "Distill the raw thinking" in prompt:
                return (
                    'REASONING_STATE_JSON: {"evidence": ["boundary test fails"], '
                    '"hypothesis": {"summary": "threshold is exclusive", "target_file": "app/score.py"}, '
                    '"decision": "patch", '
                    '"next_action": {"action": "propose_bounded_diff", "target_file": "app/score.py"}, '
                    '"confidence": 0.86, '
                    '"failure_boundary": ["do not modify tests"], '
                    '"patch_intent": "make threshold inclusive"}'
                )
            if "Generate one minimal unified diff" in prompt:
                return (
                    'PATCH_JSON: {"unified_diff": "--- a/app/score.py\\n+++ b/app/score.py\\n'
                    '@@ -1,4 +1,4 @@\\n def score_label(value):\\n-    if value > 10:\\n+    if value >= 10:\\n'
                    '         return \\"high\\"\\n     return \\"low\\"\\n", '
                    '"rationale": "make threshold inclusive", "expected_tests": ["."], "risk": 0.2}'
                )
            return "brief notes"

    target = tmp_path / "app" / "score.py"
    target.parent.mkdir(parents=True)
    target.write_text('def score_label(value):\n    if value > 10:\n        return "high"\n    return "low"\n', encoding="utf-8")

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix inclusive threshold behavior",
            "investigation_state": {
                "target_binding": {"top_target_file": "app/score.py", "target_confidence": 0.8},
                "hypotheses": [{"hypothesis_id": "h_score", "status": "leading", "target_file": "app/score.py"}],
            },
        },
        top_target_file="app/score.py",
        llm_client=DirectStateLLM(),
    )

    assert payload["patch_proposals"]
    stages = [row["stage"] for row in payload["llm_trace"]]
    assert stages == ["think_pass", "distill_pass", "act_pass"]
    assert payload["llm_trace"][1]["distill_acceptance_override"] == "parsed_reasoning_state"


def test_patch_proposal_accepts_loose_malformed_reasoning_state(tmp_path: Path) -> None:
    class LooseStateLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            if "Distill the raw thinking" in prompt:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state":{'
                    '"evidence":{"finding":{"file":"app/score.py","detail":"threshold is exclusive"}},'
                    '"hypothesis":{"summary":"threshold comparison excludes equality","target_file":"app/score.py"},'
                    '"decision":{"type":"proceed","patch_safe":true,"confidence":0.91},'
                    '"next_action":{"status":"single_patch","action":"change > to >="},'
                    '{"fallback":"rollback if verification fails"},'
                    '"failure_boundary":{"reject_if":"tests fail"},'
                    '{"patch_intent":{"change":"if value >= 10:"}},'
                    '{"need_human_review":false}'
                )
            if "Generate one minimal unified diff" in prompt:
                return (
                    'PATCH_JSON: {"unified_diff": "--- a/app/score.py\\n+++ b/app/score.py\\n'
                    '@@ -1,4 +1,4 @@\\n def score_label(value):\\n-    if value > 10:\\n+    if value >= 10:\\n'
                    '         return \\"high\\"\\n     return \\"low\\"\\n", '
                    '"rationale": "make threshold inclusive", "expected_tests": ["."], "risk": 0.2}'
                )
            return "brief notes"

    target = tmp_path / "app" / "score.py"
    target.parent.mkdir(parents=True)
    target.write_text('def score_label(value):\n    if value > 10:\n        return "high"\n    return "low"\n', encoding="utf-8")

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix inclusive threshold behavior",
            "investigation_state": {
                "target_binding": {"top_target_file": "app/score.py", "target_confidence": 0.8},
                "hypotheses": [{"hypothesis_id": "h_score", "status": "leading", "target_file": "app/score.py"}],
            },
        },
        top_target_file="app/score.py",
        llm_client=LooseStateLLM(),
    )

    assert payload["patch_proposals"]
    assert [row["stage"] for row in payload["llm_trace"]] == ["think_pass", "distill_pass", "act_pass"]
    assert payload["llm_trace"][1]["distill_acceptance_override"] == "loose_reasoning_state"


def test_patch_proposal_pipeline_uses_critical_route_budget_for_all_passes(tmp_path: Path) -> None:
    class PipelineLLM:
        model = "pipeline-test-model"

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
            if "Distill the raw thinking" in prompt:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["boundary test fails"], '
                    '"hypothesis": {"summary": "threshold is exclusive", "target_file": "app/score.py"}, '
                    '"decision": "patch", '
                    '"next_action": {"action": "propose_bounded_diff", "target_file": "app/score.py"}, '
                    '"confidence": 0.82, '
                    '"failure_boundary": ["do not modify tests"], '
                    '"patch_intent": "make threshold inclusive"}}'
                )
            if "Generate one minimal unified diff" in prompt:
                return (
                    'PATCH_JSON: {"unified_diff": "--- a/app/score.py\\n+++ b/app/score.py\\n'
                    '@@ -1,4 +1,4 @@\\n def score_label(value):\\n-    if value > 10:\\n+    if value >= 10:\\n'
                    '         return \\"high\\"\\n     return \\"low\\"\\n", '
                    '"rationale": "make threshold inclusive", "expected_tests": ["."], "risk": 0.2}'
                )
            return "brief notes"

    target = tmp_path / "app" / "score.py"
    target.parent.mkdir(parents=True)
    target.write_text('def score_label(value):\n    if value > 10:\n        return "high"\n    return "low"\n', encoding="utf-8")

    ledger = LLMCostLedger(LLMRuntimeBudget(max_llm_calls=5, critical_route_reserve_calls=3))
    wrapped = wrap_with_budget(PipelineLLM(), ledger)
    assert wrapped.complete_raw("cheap one", capability_route_name="structured_answer") == "brief notes"
    assert wrapped.complete_raw("cheap two", capability_route_name="general") == "brief notes"

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix inclusive threshold behavior",
            "investigation_state": {
                "target_binding": {"top_target_file": "app/score.py", "target_confidence": 0.8},
                "hypotheses": [
                    {
                        "hypothesis_id": "h_threshold",
                        "status": "leading",
                        "summary": "Threshold comparison is exclusive.",
                        "target_file": "app/score.py",
                    }
                ],
            },
        },
        top_target_file="app/score.py",
        llm_client=wrapped,
    )

    assert payload["patch_proposals"]
    assert [row["stage"] for row in payload["llm_trace"]] == ["think_pass", "distill_pass", "act_pass"]
    assert {row["route_name"] for row in payload["llm_trace"]} == {"patch_proposal"}
    summary = ledger.summary()
    assert summary["total_calls"] == 5
    assert summary["by_route"]["patch_proposal"]["calls"] == 3


def test_patch_intent_adapter_compiles_malformed_llm_diff_fragment(tmp_path: Path) -> None:
    class FragmentLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
            if "Distill the raw thinking" in prompt:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["boundary test fails"], '
                    '"hypothesis": {"summary": "score fallback line is wrong", "target_file": "app/score.py"}, '
                    '"decision": "patch", '
                    '"next_action": {"action": "propose_bounded_diff", "target_file": "app/score.py"}, '
                    '"confidence": 0.9, '
                    '"failure_boundary": ["do not modify tests"], '
                    '"patch_intent": "return high for boundary case"}}'
                )
            if "Generate one minimal unified diff" in prompt:
                return (
                    'PATCH_JSON: {"unified_diff": "-    return \\"low\\"\\n'
                    '+    return \\"high\\" if value >= 10 else \\"low\\"}", '
                    '"rationale": "compile nonstandard fragment into a bounded edit", '
                    '"expected_tests": ["."], "risk": 0.2}'
                )
            return "brief notes"

    target = tmp_path / "app" / "score.py"
    target.parent.mkdir(parents=True)
    target.write_text('def score_label(value):\n    if value > 10:\n        return "high"\n    return "low"\n', encoding="utf-8")

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix inclusive threshold behavior",
            "investigation_state": {
                "target_binding": {"top_target_file": "app/score.py", "target_confidence": 0.8},
                "hypotheses": [
                    {
                        "hypothesis_id": "h_threshold",
                        "status": "leading",
                        "summary": "Threshold comparison is exclusive.",
                        "target_file": "app/score.py",
                    }
                ],
            },
        },
        top_target_file="app/score.py",
        llm_client=FragmentLLM(),
    )

    proposal = payload["patch_proposals"][0]
    adapter = proposal["patch_intent_adapter"]
    assert proposal["proposal_source"] == "bounded_llm_intent_diff"
    assert adapter["status"] == "compiled"
    assert adapter["source"] == "diff_fragment"
    assert adapter["match_strategy"] == "exact_snippet"
    assert "--- a/app/score.py" in proposal["unified_diff"]
    assert "+++ b/app/score.py" in proposal["unified_diff"]
    assert '+    return "high" if value >= 10 else "low"' in proposal["unified_diff"]
    assert '"low"}' not in proposal["unified_diff"]


def test_patch_intent_adapter_compiles_unified_diff_with_drifted_blank_context(tmp_path: Path) -> None:
    class DriftedUnifiedDiffLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            if "Distill the raw thinking" in prompt:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["inclusive day count fails"], '
                    '"hypothesis": {"summary": "active_days is exclusive", "target_file": "billing/periods.py"}, '
                    '"decision": "patch active_days", '
                    '"next_action": {"action": "propose_bounded_diff", "target_file": "billing/periods.py"}, '
                    '"confidence": 0.95, '
                    '"failure_boundary": [], '
                    '"patch_intent": "add one day for inclusive range"}}'
                )
            if "Generate one minimal unified diff" in prompt:
                diff = (
                    "--- a/billing/periods.py\n"
                    "+++ b/billing/periods.py\n"
                    "@@ -11,7 +11,7 @@\n"
                    " def active_days(start: date, end: date) -> int:\n"
                    "     \"\"\"Return the number of billable days in an inclusive date window.\"\"\"\n"
                    " \n"
                    "     if end < start:\n"
                    "         return 0\n"
                    "-    return (end - start).days\n"
                    "+    return (end - start).days + 1\n"
                    "\n"
                    " \n"
                    " def same_month(start: date, end: date) -> bool:\n"
                    "     return start.year == end.year and start.month == end.month"
                )
                return "PATCH_JSON: " + json.dumps(
                    {
                        "unified_diff": diff,
                        "rationale": "compile mismatched blank context via bounded intent fallback",
                        "expected_tests": ["tests/test_periods.py"],
                        "risk": 0.1,
                    }
                )
            return "brief notes"

    target = tmp_path / "billing" / "periods.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "from __future__ import annotations\n\n"
        "from calendar import monthrange\n"
        "from datetime import date\n\n\n"
        "def days_in_month(day: date) -> int:\n"
        "    return monthrange(day.year, day.month)[1]\n\n\n"
        "def active_days(start: date, end: date) -> int:\n"
        "    \"\"\"Return the number of billable days in an inclusive date window.\"\"\"\n\n"
        "    if end < start:\n"
        "        return 0\n"
        "    return (end - start).days\n\n\n"
        "def same_month(start: date, end: date) -> bool:\n"
        "    return start.year == end.year and start.month == end.month\n",
        encoding="utf-8",
    )

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix inclusive active day behavior",
            "investigation_state": {
                "target_binding": {"top_target_file": "billing/periods.py", "target_confidence": 0.95},
                "hypotheses": [
                    {
                        "hypothesis_id": "h_periods",
                        "status": "leading",
                        "summary": "active_days is exclusive.",
                        "target_file": "billing/periods.py",
                    }
                ],
            },
        },
        top_target_file="billing/periods.py",
        llm_client=DriftedUnifiedDiffLLM(),
    )

    proposal = payload["patch_proposals"][0]
    adapter = proposal["patch_intent_adapter"]
    assert proposal["proposal_source"] == "bounded_llm_intent_diff"
    assert adapter["status"] == "compiled"
    assert adapter["source"] == "diff_fragment"
    assert "+    return (end - start).days + 1" in proposal["unified_diff"]


def test_repo_grep_query_prefers_issue_code_tokens_over_generic_runtime_words() -> None:
    instruction = (
        "调查真实 GitHub issue：当 self.interval == 1 且 self.unit 为 None 时，"
        "代码尝试 self.unit[:-1] 会崩溃；保持运行测试通过。"
    )

    query, source = choose_repo_grep_query({"instruction": instruction})

    assert query == "unit"
    assert source == "goal code token"


def test_low_value_repo_grep_query_repairs_to_issue_code_token(tmp_path: Path) -> None:
    (tmp_path / "schedule").mkdir()
    (tmp_path / "schedule" / "__init__.py").write_text("self.unit[:-1]\n", encoding="utf-8")
    instruction = "Issue: self.unit 为 None 时 self.unit[:-1] 崩溃；保持运行测试通过。"

    repaired = validate_local_machine_action(
        "repo_grep",
        {"root": ".", "query": "runtime", "globs": ["*.py"], "max_matches": 50},
        {"source_root": str(tmp_path), "workspace_root": str(tmp_path), "instruction": instruction},
    )

    assert repaired["status"] == "repaired"
    assert repaired["function_name"] == "repo_grep"
    assert repaired["kwargs"]["query"] == "unit"
    assert repaired["event"]["low_value_repo_grep_query_repaired"] is True


def test_repeated_low_value_repo_grep_repairs_to_unread_match_window(tmp_path: Path) -> None:
    source_file = tmp_path / "schedule" / "__init__.py"
    source_file.parent.mkdir()
    source_file.write_text("\n".join([f"line {idx}" for idx in range(1, 320)]) + "\nself.unit[:-1]\n", encoding="utf-8")
    instruction = "Issue: self.unit 为 None 时 self.unit[:-1] 崩溃；保持运行测试通过。"

    repaired = validate_local_machine_action(
        "repo_grep",
        {"root": ".", "query": "runtime", "globs": ["*.py"], "max_matches": 50},
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": instruction,
            "investigation_state": {
                "read_files": [{"path": "schedule/__init__.py", "start_line": 1, "end_line": 240}],
                "last_search": {
                    "action": "repo_grep",
                    "root": ".",
                    "query": "unit",
                    "match_count": 2,
                    "matches": [
                        {"path": "docs/conf.py", "line": 104, "text": "# unit titles"},
                        {"path": "schedule/__init__.py", "line": 320, "text": "self.unit[:-1]"},
                    ],
                },
            },
        },
    )

    assert repaired["status"] == "repaired"
    assert repaired["function_name"] == "file_read"
    assert repaired["kwargs"]["path"] == "schedule/__init__.py"
    assert repaired["kwargs"]["start_line"] <= 320 <= repaired["kwargs"]["end_line"]


def test_low_value_repo_grep_repairs_to_patch_when_issue_evidence_is_sufficient(tmp_path: Path) -> None:
    source_file = tmp_path / "schedule" / "__init__.py"
    source_file.parent.mkdir()
    source_file.write_text("class Job:\n    unit = None\n    def __str__(self):\n        return self.unit[:-1]\n", encoding="utf-8")
    (tmp_path / "test_schedule.py").write_text("def test_existing():\n    assert True\n", encoding="utf-8")
    instruction = "Issue: self.unit 为 None 时 self.unit[:-1] 崩溃；保持运行测试通过。"

    repaired = validate_local_machine_action(
        "repo_grep",
        {"root": ".", "query": "runtime", "globs": ["*.py"], "max_matches": 50},
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": instruction,
            "episode_run_test_targets": ["test_schedule.py"],
            "investigation_state": {
                "last_tree": {
                    "entries": [
                        {"kind": "file", "path": "schedule/__init__.py"},
                        {"kind": "file", "path": "test_schedule.py"},
                    ]
                },
                "read_files": [
                    {"path": "test_schedule.py", "start_line": 1, "end_line": 20},
                    {"path": "schedule/__init__.py", "start_line": 1, "end_line": 80},
                ],
                "last_search": {
                    "action": "repo_grep",
                    "root": ".",
                    "query": "unit",
                    "match_count": 1,
                    "matches": [{"path": "schedule/__init__.py", "line": 4, "text": "return self.unit[:-1]"}],
                },
                "target_binding": {
                    "top_target_file": "schedule/__init__.py",
                    "target_confidence": 0.72,
                    "target_file_candidates": [{"target_file": "schedule/__init__.py", "score": 0.72}],
                },
            },
        },
    )

    assert repaired["status"] == "repaired"
    assert repaired["function_name"] == "propose_patch"
    assert repaired["kwargs"]["target_file"] == "schedule/__init__.py"
    assert repaired["event"]["issue_evidence_to_patch_bridge"] is True


def test_patch_intent_adapter_rejects_non_unique_old_snippet(tmp_path: Path) -> None:
    class AmbiguousFragmentLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            if "Distill the raw thinking" in prompt:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["ambiguous duplicated line"], '
                    '"hypothesis": {"summary": "one fallback line is wrong", "target_file": "app/score.py"}, '
                    '"decision": "patch", '
                    '"next_action": {"action": "propose_bounded_diff", "target_file": "app/score.py"}, '
                    '"confidence": 0.7, '
                    '"failure_boundary": ["do not modify tests"], '
                    '"patch_intent": "replace one low fallback"}}'
                )
            if "Generate one minimal unified diff" in prompt:
                return (
                    'PATCH_JSON: {"unified_diff": "-    return \\"low\\"\\n+    return \\"high\\"", '
                    '"rationale": "ambiguous edit", "expected_tests": ["."], "risk": 0.4}'
                )
            return "brief notes"

    target = tmp_path / "app" / "score.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'def first():\n    return "low"\n\n'
        'def second():\n    return "low"\n',
        encoding="utf-8",
    )

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix one ambiguous fallback",
            "investigation_state": {"target_binding": {"top_target_file": "app/score.py", "target_confidence": 0.8}},
        },
        top_target_file="app/score.py",
        llm_client=AmbiguousFragmentLLM(),
    )

    assert payload["patch_proposals"] == []
    act_trace = [row for row in payload["llm_trace"] if row.get("stage") == "act_pass"][-1]
    assert act_trace["error"] == "old_snippet_not_unique"
    assert act_trace["patch_intent_adapter"]["status"] == "rejected"


def test_llm_patch_proposal_with_bare_expected_test_verifies_and_stops(tmp_path: Path) -> None:
    class BareExpectedTestLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            if "Distill the raw thinking" in prompt:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["boundary test fails"], '
                    '"hypothesis": {"summary": "threshold is exclusive", "target_file": "srcpkg/score.py"}, '
                    '"decision": "patch threshold", '
                    '"next_action": {"action": "propose_bounded_diff", "target_file": "srcpkg/score.py"}, '
                    '"confidence": 0.9, '
                    '"failure_boundary": ["do not modify tests"], '
                    '"patch_intent": "make threshold inclusive"}}'
                )
            if "Generate one minimal unified diff" in prompt:
                return (
                    'PATCH_JSON: {"unified_diff": "-    if value > 10:\\n+    if value >= 10:", '
                    '"rationale": "make threshold inclusive", '
                    '"expected_tests": ["test_boundary_score_is_high"], "risk": 0.2}'
                )
            return "brief notes"

    source = tmp_path / "source"
    (source / "srcpkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[tool.pytest.ini_options]\npythonpath=['.']\n", encoding="utf-8")
    (source / "srcpkg" / "__init__.py").write_text("", encoding="utf-8")
    (source / "srcpkg" / "score.py").write_text(
        'def score_label(value: int) -> str:\n'
        '    if value > 10:\n'
        '        return "high"\n'
        '    return "low"\n',
        encoding="utf-8",
    )
    (source / "tests" / "test_score.py").write_text(
        "from srcpkg.score import score_label\n\n"
        "def test_boundary_score_is_high() -> None:\n"
        "    assert score_label(10) == 'high'\n\n"
        "def test_above_boundary_score_is_high() -> None:\n"
        "    assert score_label(11) == 'high'\n",
        encoding="utf-8",
    )
    adapter = LocalMachineSurfaceAdapter(
        instruction="fix the failing score boundary",
        source_root=source,
        mirror_root=tmp_path / "mirror",
        reset_mirror=True,
        allow_empty_exec=True,
        deterministic_fallback_enabled=False,
        prefer_llm_patch_proposals=True,
        llm_client=BareExpectedTestLLM(),
        execution_backend="local",
    )

    assert adapter.act({"action": "repo_tree", "args": {"path": str(source), "depth": 2}}).ok is True
    assert adapter.act({"action": "run_test", "args": {"target": "tests/test_score.py", "timeout_seconds": 30}}).ok is False
    assert adapter.act({"action": "file_read", "args": {"path": "srcpkg/score.py", "start_line": 1, "end_line": 20}}).ok is True
    result = adapter.act({"action": "propose_patch", "args": {"target_file": "srcpkg/score.py"}})

    assert result.ok is True
    assert result.raw["state"] == "PATCH_PROPOSAL_VERIFIED"
    assert result.raw["patch_proposal_verified"] is True
    assert result.raw["terminal"] is True
    assert result.observation.terminal is True
    assert 'if value >= 10' in (tmp_path / "mirror" / "workspace" / "srcpkg" / "score.py").read_text(encoding="utf-8")
    investigation = adapter._load_investigation_state()
    assert investigation["terminal_state"] == "completed_verified"
    assert investigation["verified_completion"] is True


def test_open_improvement_rejected_patch_rolls_back_and_terminally_refuses(tmp_path: Path) -> None:
    class BadPatchLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            if self.calls == 1:
                return "Changing the return value would be a tiny candidate patch."
            if self.calls == 2:
                return (
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["tests currently pass before any patch"], '
                    '"hypothesis": {"summary": "value helper can be improved", "target_file": "srcpkg/value.py"}, '
                    '"decision": "patch", '
                    '"next_action": {"action": "propose_bounded_diff", "target_file": "srcpkg/value.py"}, '
                    '"confidence": 0.6, '
                    '"failure_boundary": [], '
                    '"patch_intent": "change the return value"}}'
                )
            return "PATCH_JSON: " + json.dumps(
                {
                    "unified_diff": (
                        "--- a/srcpkg/value.py\n"
                        "+++ b/srcpkg/value.py\n"
                        "@@ -1,2 +1,2 @@\n"
                        " def value():\n"
                        "-    return 1\n"
                        "+    return 2\n"
                    ),
                    "rationale": "tiny candidate patch that verifier must reject",
                    "expected_tests": ["tests/test_value.py"],
                    "risk": 0.2,
                }
            )

    source = tmp_path / "source"
    (source / "srcpkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / "pyproject.toml").write_text("[tool.pytest.ini_options]\npythonpath=['.']\n", encoding="utf-8")
    (source / "srcpkg" / "__init__.py").write_text("", encoding="utf-8")
    (source / "srcpkg" / "value.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_value.py").write_text(
        "from srcpkg.value import value\n\n\ndef test_value():\n    assert value() == 1\n",
        encoding="utf-8",
    )
    adapter = LocalMachineSurfaceAdapter(
        instruction="Improve this repository with one small low-risk source patch.",
        source_root=source,
        mirror_root=tmp_path / "mirror",
        reset_mirror=True,
        deterministic_fallback_enabled=False,
        prefer_llm_patch_proposals=True,
        llm_client=BadPatchLLM(),
        execution_backend="local",
    )

    assert adapter.act({"function_name": "repo_tree", "kwargs": {"path": ".", "depth": 3, "max_entries": 50}}).ok is True
    assert adapter.act({"function_name": "file_read", "kwargs": {"path": "README.md", "start_line": 1, "end_line": 20}}).ok is True
    assert adapter.act({"function_name": "file_read", "kwargs": {"path": "tests/test_value.py", "start_line": 1, "end_line": 40}}).ok is True
    assert adapter.act({"function_name": "file_read", "kwargs": {"path": "srcpkg/__init__.py", "start_line": 1, "end_line": 20}}).ok is True
    assert adapter.act({"function_name": "file_read", "kwargs": {"path": "srcpkg/value.py", "start_line": 1, "end_line": 20}}).ok is True
    assert adapter.act({"function_name": "run_test", "kwargs": {"target": "tests/test_value.py", "timeout_seconds": 30}}).ok is True
    result = adapter.act({"function_name": "propose_patch", "kwargs": {"target_file": "srcpkg/value.py"}})

    assert result.raw["state"] == "PATCH_PROPOSAL_REJECTED"
    assert result.raw["patch_proposal_rollback_count"] == 1
    assert result.raw["needs_human_review"] is True
    assert result.raw["refusal_reason"] == "verifier_rejected_patch"
    assert result.observation.terminal is True
    workspace_file = tmp_path / "mirror" / "workspace" / "srcpkg" / "value.py"
    assert workspace_file.read_text(encoding="utf-8") == "def value():\n    return 1\n"
    investigation = adapter._load_investigation_state()
    assert investigation["terminal_state"] == "needs_human_review"
    assert investigation["refusal_reason"] == "verifier_rejected_patch"


def test_fast_path_budget_hint_prefers_deterministic_patch_before_llm_proposal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    control = tmp_path / "control"
    run_outputs = control / "run_outputs"
    (source / "app").mkdir(parents=True)
    (workspace / "app").mkdir(parents=True)
    run_outputs.mkdir(parents=True)
    target = "app/amounts.py"
    content = "def parse_amount(value):\n    return float(value.strip().replace(\"$\", \"\"))\n"
    (source / target).write_text(content, encoding="utf-8")
    (workspace / target).write_text(content, encoding="utf-8")
    (run_outputs / "run_failed.json").write_text(
        json.dumps(
            {
                "run_ref": "run_failed",
                "command": [sys.executable, "-m", "pytest", "tests/test_amounts.py"],
                "stdout": "FAILED tests/test_amounts.py::test_parse_grouped_amount expected $1,200",
                "stderr": "",
                "returncode": 1,
            }
        ),
        encoding="utf-8",
    )
    obs = {
        "available_functions": ["apply_patch", "file_read", "run_test", "propose_patch"],
        "local_mirror": {
            "source_root": str(source),
            "workspace_root": str(workspace),
            "control_root": str(control),
            "prefer_llm_patch_proposals": True,
            "investigation": {
                "investigation_phase": "patch",
                "target_binding": {"top_target_file": target, "target_confidence": 0.82},
                "grounding": {"target_file": target},
                "last_read": {"path": target, "start_line": 1, "end_line": 2},
                "read_files": [{"path": target}],
                "validation_runs": [{"run_ref": "run_failed", "success": False}],
            },
            "llm_budget": {
                "selected_path_hint": "fast_path",
                "fast_path": {
                    "eligible": True,
                    "target_file": target,
                    "target_confidence": 0.82,
                    "reasons": ["failing_test_observed", "high_confidence_target_binding", "target_file_read"],
                    "blockers": [],
                },
                "escalation_path": {"recommended": False, "triggers": []},
            },
        },
    }

    candidate = build_local_machine_posterior_action_bridge_candidate(obs, episode_trace=[])

    assert candidate is not None
    assert candidate["function_name"] == "apply_patch"
    assert candidate["kwargs"]["patch"]
    meta = candidate["_candidate_meta"]
    assert meta["budget_path_hint"] == "fast_path"
    assert meta["llm_layer_preference"] == "deterministic"
    assert meta["fast_path_bonus"] > 0


def test_stalled_loop_bridge_leaves_repeated_successful_test_loop(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    control = tmp_path / "control"
    (source / "core" / "runtime").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    (workspace / "core" / "runtime").mkdir(parents=True)
    control.mkdir(parents=True)
    source_file = "core/runtime/runtime_service.py"
    test_file = "tests/test_runtime_service.py"
    (source / source_file).write_text("def status():\n    return 'ok'\n", encoding="utf-8")
    (source / test_file).write_text(
        "from core.runtime.runtime_service import status\n\n\ndef test_status():\n    assert status() == 'ok'\n",
        encoding="utf-8",
    )
    (workspace / source_file).write_text("def status():\n    return 'ok'\n", encoding="utf-8")
    obs = {
        "available_functions": ["repo_grep", "file_read", "run_test", "propose_patch", "investigation_status"],
        "local_mirror": {
            "instruction": "Improve runtime reliability by finding a small useful source change.",
            "source_root": str(source),
            "workspace_root": str(workspace),
            "control_root": str(control),
            "diff_summary": {"entry_count": 0},
            "investigation": {
                "investigation_phase": "inspect",
                "last_tree": {
                    "entries": [
                        {"kind": "file", "path": "pyproject.toml"},
                        {"kind": "file", "path": source_file},
                        {"kind": "file", "path": test_file},
                    ]
                },
                "action_history": [
                    {"function_name": "run_test", "target": test_file, "success": True},
                    {"function_name": "run_test", "target": test_file, "success": True},
                    {"function_name": "run_test", "target": test_file, "success": True},
                ],
                "read_files": [],
                "validation_runs": [],
            },
        },
    }

    candidate = build_local_machine_posterior_action_bridge_candidate(obs, episode_trace=[])

    assert candidate is not None
    assert candidate["function_name"] in {"file_read", "repo_grep", "investigation_status"}
    assert candidate["function_name"] != "run_test"
    meta = candidate["_candidate_meta"]
    assert meta["stalled_loop_recovery_bonus"] > 0
    assert meta["progress_recovery_bonus"] > 0


def test_stalled_loop_ranking_penalizes_repeated_run_test(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    control = tmp_path / "control"
    (source / "core").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    workspace.mkdir(parents=True)
    control.mkdir(parents=True)
    (source / "core" / "worker.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_worker.py").write_text("def test_worker():\n    assert True\n", encoding="utf-8")
    test_target = "tests/test_worker.py"
    obs = {
        "available_functions": ["file_read", "run_test"],
        "local_mirror": {
            "instruction": "Find a small source improvement in the worker runtime.",
            "source_root": str(source),
            "workspace_root": str(workspace),
            "control_root": str(control),
            "diff_summary": {"entry_count": 0},
            "investigation": {
                "investigation_phase": "inspect",
                "last_tree": {
                    "entries": [
                        {"kind": "file", "path": "core/worker.py"},
                        {"kind": "file", "path": test_target},
                    ]
                },
                "action_history": [
                    {"function_name": "run_test", "target": test_target, "success": True},
                    {"function_name": "run_test", "target": test_target, "success": True},
                    {"function_name": "run_test", "target": test_target, "success": True},
                ],
            },
        },
    }
    run_test_action = {
        "kind": "call_tool",
        "function_name": "run_test",
        "kwargs": {"target": test_target},
        "payload": {"tool_args": {"function_name": "run_test", "kwargs": {"target": test_target}}},
    }
    file_read_action = {
        "kind": "call_tool",
        "function_name": "file_read",
        "kwargs": {"path": "core/worker.py"},
        "payload": {"tool_args": {"function_name": "file_read", "kwargs": {"path": "core/worker.py"}}},
    }

    annotated = annotate_local_machine_patch_ranking([run_test_action, file_read_action], obs, episode_trace=[])

    run_test_meta = annotated[0]["_candidate_meta"]
    file_read_meta = annotated[1]["_candidate_meta"]
    assert run_test_meta["stalled_loop_penalty"] > 0
    assert run_test_meta["action_cooldown_recommended"] is True
    assert annotated[0]["final_score"] < annotated[1]["final_score"]
    assert file_read_meta["stalled_loop_recovery_bonus"] > 0


def test_open_task_bridge_collects_project_evidence_before_patch_proposal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    control = tmp_path / "control"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    (workspace / "pkg").mkdir(parents=True)
    control.mkdir(parents=True)
    target = "pkg/core.py"
    related = "pkg/helpers.py"
    test_file = "tests/test_core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("from .helpers import normalize\n\n\ndef run(value):\n    return normalize(value)\n", encoding="utf-8")
    (source / related).write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    (source / test_file).write_text("from pkg.core import run\n\n\ndef test_run():\n    assert run('x') == 'x'\n", encoding="utf-8")
    (workspace / target).write_text((source / target).read_text(encoding="utf-8"), encoding="utf-8")
    obs = {
        "available_functions": ["repo_tree", "file_read", "run_test", "propose_patch"],
        "local_mirror": {
            "instruction": "Improve this repository with one small low-risk source patch.",
            "source_root": str(source),
            "workspace_root": str(workspace),
            "control_root": str(control),
            "prefer_llm_patch_proposals": True,
            "diff_summary": {"entry_count": 0},
            "investigation": {
                "investigation_phase": "patch",
                "last_tree": {
                    "entries": [
                        {"kind": "file", "path": "README.md"},
                        {"kind": "file", "path": target},
                        {"kind": "file", "path": related},
                        {"kind": "file", "path": test_file},
                    ]
                },
                "target_binding": {
                    "top_target_file": target,
                    "target_confidence": 0.66,
                    "target_file_candidates": [
                        {"target_file": target, "score": 0.66},
                        {"target_file": related, "score": 0.58},
                    ],
                },
                "grounding": {"target_file": target},
                "last_read": {"path": target, "start_line": 1, "end_line": 4},
                "read_files": [{"path": target}],
                "validation_runs": [],
            },
        },
    }

    candidate = build_local_machine_posterior_action_bridge_candidate(obs, episode_trace=[])

    assert candidate is not None
    assert candidate["function_name"] == "file_read"
    assert candidate["kwargs"]["path"] == "README.md"
    gate = candidate["_candidate_meta"]["open_task_evidence_gate"]
    assert gate["reason"] == "project_overview_required_before_open_task_patch"


def test_open_task_evidence_gate_requires_tests_and_related_source_before_patch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    target = "pkg/core.py"
    related = "pkg/helpers.py"
    test_file = "tests/test_core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("from .helpers import normalize\n", encoding="utf-8")
    (source / related).write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    (source / test_file).write_text("def test_core():\n    assert True\n", encoding="utf-8")
    base_context = {
        "instruction": "Improve parse reliability with one small low-risk source patch.",
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "README.md"},
                    {"kind": "file", "path": target},
                    {"kind": "file", "path": related},
                    {"kind": "file", "path": test_file},
                ]
            },
            "target_binding": {
                "top_target_file": target,
                "target_confidence": 0.7,
                "target_file_candidates": [
                    {"target_file": target, "score": 0.7},
                    {"target_file": related, "score": 0.61},
                ],
            },
            "read_files": [{"path": "README.md"}, {"path": target}],
        },
    }

    first_gap = open_task_patch_evidence_gap(base_context, target_file=target)

    assert first_gap["sufficient"] is False
    assert first_gap["suggested_action"] == "file_read"
    assert first_gap["suggested_kwargs"]["path"] == test_file

    sufficient_context = {
        **base_context,
        "episode_run_test_targets": [test_file],
        "investigation_state": {
            **base_context["investigation_state"],
            "read_files": [
                {"path": "README.md"},
                {"path": test_file},
                {"path": target},
                {"path": related},
            ],
        },
    }
    final_gap = open_task_patch_evidence_gap(sufficient_context, target_file=target)

    assert final_gap["sufficient"] is True
    assert final_gap["reason"] == "open_task_patch_evidence_sufficient"


def test_validate_file_read_repairs_stale_open_task_read_to_next_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    target = "pkg/core.py"
    test_file = "tests/test_core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / test_file).write_text("def test_core():\n    assert True\n", encoding="utf-8")
    context = {
        "instruction": "Improve parse reliability with one small low-risk source patch.",
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "README.md"},
                    {"kind": "file", "path": target},
                    {"kind": "file", "path": test_file},
                ]
            },
            "read_files": [{"path": "README.md"}],
            "target_binding": {
                "top_target_file": target,
                "target_confidence": 0.7,
                "target_file_candidates": [{"target_file": target, "score": 0.7}],
            },
        },
    }

    repaired = validate_local_machine_action("file_read", {"path": "README.md", "start_line": 1, "end_line": 220}, context)

    assert repaired["status"] == "repaired"
    assert repaired["function_name"] == "file_read"
    assert repaired["kwargs"]["path"] == test_file
    assert repaired["event"]["stale_file_read_repaired"] is True
    assert repaired["event"]["open_task_evidence_gate"]["reason"] == "test_source_required_before_open_task_patch"


def test_validate_propose_patch_repairs_to_next_evidence_action(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    target = "pkg/core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_core.py").write_text("def test_core():\n    assert True\n", encoding="utf-8")
    context = {
        "instruction": "Improve this repository with one small low-risk source patch.",
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "README.md"},
                    {"kind": "file", "path": target},
                    {"kind": "file", "path": "tests/test_core.py"},
                ]
            },
            "read_files": [{"path": "README.md"}, {"path": "tests/test_core.py"}],
            "target_binding": {
                "top_target_file": target,
                "target_confidence": 0.7,
                "target_file_candidates": [{"target_file": target, "score": 0.7}],
            },
        },
        "episode_run_test_targets": ["tests/test_core.py"],
    }

    repaired = validate_local_machine_action("propose_patch", {"target_file": target, "max_changed_lines": 20}, context)

    assert repaired["status"] == "repaired"
    assert repaired["function_name"] == "file_read"
    assert repaired["kwargs"]["path"] == target
    assert repaired["event"]["premature_propose_patch_repaired"] is True
    assert repaired["event"]["open_task_evidence_gate"]["reason"] == "related_source_required_before_open_task_patch"


def test_validate_repeated_successful_run_test_repairs_to_open_task_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    target = "pkg/core.py"
    test_file = "tests/test_core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / test_file).write_text("def test_core():\n    assert True\n", encoding="utf-8")
    context = {
        "instruction": "Improve this repository with one small low-risk source patch.",
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "README.md"},
                    {"kind": "file", "path": target},
                    {"kind": "file", "path": test_file},
                ]
            },
            "action_history": [
                {"function_name": "run_test", "target": test_file, "success": True, "state": "VALIDATION_RUN_COMPLETED"}
            ],
            "target_binding": {
                "top_target_file": target,
                "target_confidence": 0.7,
                "target_file_candidates": [{"target_file": target, "score": 0.7}],
            },
        },
    }

    repaired = validate_local_machine_action("run_test", {"target": test_file, "timeout_seconds": 30}, context)

    assert repaired["status"] == "repaired"
    assert repaired["function_name"] == "file_read"
    assert repaired["kwargs"]["path"] == "README.md"
    assert repaired["event"]["repeated_successful_run_test_repaired"] is True
    assert repaired["event"]["open_task_evidence_gate"]["reason"] == "project_overview_required_before_open_task_patch"


def test_validate_repeated_successful_run_test_repairs_to_patch_when_evidence_complete(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    target = "pkg/core.py"
    test_file = "tests/test_core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "pkg" / "helpers.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (source / test_file).write_text("def test_core():\n    assert True\n", encoding="utf-8")
    context = {
        "instruction": "Improve this repository with one small low-risk source patch.",
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "README.md"},
                    {"kind": "file", "path": target},
                    {"kind": "file", "path": "pkg/helpers.py"},
                    {"kind": "file", "path": test_file},
                ]
            },
            "read_files": [
                {"path": "README.md"},
                {"path": test_file},
                {"path": target},
                {"path": "pkg/helpers.py"},
            ],
            "action_history": [
                {"function_name": "run_test", "target": test_file, "success": True, "state": "VALIDATION_RUN_COMPLETED"}
            ],
            "target_binding": {
                "top_target_file": target,
                "target_confidence": 0.7,
                "target_file_candidates": [{"target_file": target, "score": 0.7}],
            },
        },
        "episode_run_test_targets": [test_file],
    }

    repaired = validate_local_machine_action("run_test", {"target": test_file, "timeout_seconds": 30}, context)

    assert repaired["status"] == "repaired"
    assert repaired["function_name"] == "propose_patch"
    assert repaired["kwargs"]["target_file"] == target
    assert repaired["event"]["successful_test_cooldown"] is True


def test_open_task_evidence_gate_uses_source_root_when_tree_is_shallow(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    target = "pkg/core.py"
    test_file = "tests/test_core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / test_file).write_text("def test_core():\n    assert True\n", encoding="utf-8")
    context = {
        "instruction": "Improve this repository with one small low-risk source patch.",
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "README.md"},
                    {"kind": "dir", "path": "pkg"},
                    {"kind": "dir", "path": "tests"},
                    {"kind": "file", "path": test_file},
                ]
            },
            "read_files": [{"path": "README.md"}, {"path": test_file}],
            "action_history": [
                {"function_name": "run_test", "target": test_file, "success": True, "state": "VALIDATION_RUN_COMPLETED"}
            ],
        },
        "episode_run_test_targets": [test_file],
    }

    gate = open_task_patch_evidence_gap(context)
    repaired = validate_local_machine_action("run_test", {"target": test_file, "timeout_seconds": 30}, context)

    assert gate["sufficient"] is False
    assert gate["reason"] == "related_source_required_before_open_task_patch"
    assert gate["suggested_kwargs"]["path"] == target
    assert repaired["status"] == "repaired"
    assert repaired["function_name"] == "file_read"
    assert repaired["kwargs"]["path"] == target


def test_validate_repeated_empty_repo_grep_repairs_to_open_task_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    target = "pkg/core.py"
    test_file = "tests/test_core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / test_file).write_text("def test_core():\n    assert True\n", encoding="utf-8")
    context = {
        "instruction": "Improve this repository with one small low-risk source patch.",
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "README.md"},
                    {"kind": "file", "path": target},
                    {"kind": "file", "path": test_file},
                ]
            },
            "last_search": {"action": "repo_grep", "query": "runtime", "match_count": 0, "matches": []},
            "action_history": [
                {"function_name": "repo_grep", "query": "runtime", "match_count": 0, "success": True}
            ],
            "target_binding": {
                "top_target_file": target,
                "target_confidence": 0.7,
                "target_file_candidates": [{"target_file": target, "score": 0.7}],
            },
        },
    }

    repaired = validate_local_machine_action(
        "repo_grep",
        {"root": ".", "query": "runtime", "globs": ["*.py"], "max_matches": 50},
        context,
    )

    assert repaired["status"] == "repaired"
    assert repaired["function_name"] == "file_read"
    assert repaired["kwargs"]["path"] == "README.md"
    assert repaired["event"]["stale_repo_grep_repaired"] is True
    assert repaired["event"]["open_task_evidence_gate"]["reason"] == "project_overview_required_before_open_task_patch"


def test_default_open_project_template_does_not_create_runtime_grep_query() -> None:
    query, source = choose_repo_grep_query(
        {
            "instruction": (
                "调查这个仓库，找到一个实际可改进点。\n\n"
                "CONOS_DEFAULT_OPEN_PROJECT_TEMPLATE:\n"
                "- Start with repository inventory before proposing changes.\n"
                "- Run targeted validation first, then full verification before any sync-back or completion.\n"
            )
        }
    )

    assert query == ""
    assert source == ""


def test_propose_patch_action_is_repaired_until_open_task_evidence_is_collected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    target = "pkg/core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_core.py").write_text("from pkg.core import value\n\n\ndef test_value():\n    assert value() == 1\n", encoding="utf-8")
    adapter = LocalMachineSurfaceAdapter(
        instruction="Improve this repository with one small low-risk source patch.",
        source_root=source,
        mirror_root=tmp_path / "mirror",
        reset_mirror=True,
        execution_backend="local",
        prefer_llm_patch_proposals=True,
    )

    assert adapter.act({"function_name": "repo_tree", "kwargs": {"path": ".", "depth": 3, "max_entries": 50}}).ok is True
    assert adapter.act({"function_name": "file_read", "kwargs": {"path": target, "start_line": 1, "end_line": 20}}).ok is True
    result = adapter.act({"function_name": "propose_patch", "kwargs": {"target_file": target}})

    assert result.raw["state"] == "FILE_READ"
    assert result.raw["success"] is True
    assert result.raw["path"] == "README.md"
    grounding = result.raw["local_machine_action_grounding"]
    assert grounding["premature_propose_patch_repaired"] is True
    assert grounding["repaired_action"]["function_name"] == "file_read"
    assert result.observation.terminal is False


def test_propose_patch_execution_guard_blocks_if_validation_is_bypassed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir(parents=True)
    target = "pkg/core.py"
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")
    (source / target).write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_core.py").write_text("from pkg.core import value\n\n\ndef test_value():\n    assert value() == 1\n", encoding="utf-8")
    adapter = LocalMachineSurfaceAdapter(
        instruction="Improve this repository with one small low-risk source patch.",
        source_root=source,
        mirror_root=tmp_path / "mirror",
        reset_mirror=True,
        execution_backend="local",
        prefer_llm_patch_proposals=True,
    )
    assert adapter.act({"function_name": "repo_tree", "kwargs": {"path": ".", "depth": 3, "max_entries": 50}}).ok is True
    assert adapter.act({"function_name": "file_read", "kwargs": {"path": target, "start_line": 1, "end_line": 20}}).ok is True

    result = adapter._act_propose_patch({"target_file": target})

    assert result["state"] == "PATCH_PROPOSAL_NEEDS_EVIDENCE"
    assert result["success"] is False
    assert result["needs_human_review"] is False
    assert result["suggested_action"] == "file_read"
    assert result["suggested_kwargs"]["path"] == "README.md"


def test_open_task_target_binding_prefers_read_goal_relevant_source_over_repeated_search_hits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "core" / "runtime").mkdir(parents=True)
    source_file = "core/runtime/runtime_service.py"
    entry_file = "conos_cli.py"
    (source / source_file).write_text("def runtime_status():\n    return 'idle'\n", encoding="utf-8")
    (source / entry_file).write_text("from core.runtime.runtime_service import runtime_status\n", encoding="utf-8")
    repeated_entry_hits = [{"path": entry_file, "line": index, "text": "runtime"} for index in range(8)]
    context = {
        "instruction": "Improve runtime reliability with one small low-risk source change.",
        "source_root": str(source),
        "run_output_root": str(tmp_path / "run_outputs"),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": source_file},
                    {"kind": "file", "path": entry_file},
                ]
            },
            "last_search": {
                "action": "repo_grep",
                "matches": [*repeated_entry_hits, {"path": source_file, "line": 1, "text": "runtime"}],
            },
            "read_files": [{"path": source_file}],
            "validation_runs": [],
        },
    }

    binding = bind_target(context)

    assert binding["top_target_file"] == source_file
    assert binding["target_confidence"] >= 0.55
    entry_candidates = [
        row for row in binding["target_file_candidates"] if row["target_file"] == entry_file
    ]
    assert entry_candidates
    assert entry_candidates[0]["score"] < binding["target_file_candidates"][0]["score"]


def test_open_task_target_binding_excludes_pytest_files_from_source_targets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src" / "pkg").mkdir(parents=True)
    (source / "testing").mkdir(parents=True)
    (source / "src" / "pkg" / "_parse.py").write_text("def parse(value):\n    return value\n", encoding="utf-8")
    (source / "testing" / "test_pkg.py").write_text("def test_pkg():\n    assert True\n", encoding="utf-8")
    (source / "testing" / "conftest.py").write_text("", encoding="utf-8")
    context = {
        "instruction": "Improve parse reliability with one small low-risk source patch.",
        "source_root": str(source),
        "investigation_state": {
            "last_tree": {
                "entries": [
                    {"kind": "file", "path": "src/pkg/_parse.py"},
                    {"kind": "file", "path": "testing/test_pkg.py"},
                    {"kind": "file", "path": "testing/conftest.py"},
                ]
            },
            "read_files": [{"path": "testing/test_pkg.py"}],
        },
    }

    binding = bind_target(context)

    candidates = [row["target_file"] for row in binding["target_file_candidates"]]
    assert "src/pkg/_parse.py" in candidates
    assert "testing/test_pkg.py" not in candidates
    assert "testing/conftest.py" not in candidates


def test_open_task_patch_phase_can_request_bounded_llm_proposal_without_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    control = tmp_path / "control"
    (source / "core" / "runtime").mkdir(parents=True)
    (workspace / "core" / "runtime").mkdir(parents=True)
    control.mkdir(parents=True)
    target = "core/runtime/runtime_service.py"
    content = "def runtime_status():\n    return {'mode': 'idle'}\n"
    (source / target).write_text(content, encoding="utf-8")
    (workspace / target).write_text(content, encoding="utf-8")
    obs = {
        "available_functions": ["file_read", "propose_patch", "run_test"],
        "local_mirror": {
            "instruction": "Improve runtime reliability with one small low-risk source change.",
            "source_root": str(source),
            "workspace_root": str(workspace),
            "control_root": str(control),
            "prefer_llm_patch_proposals": True,
            "diff_summary": {"entry_count": 0},
            "investigation": {
                "investigation_phase": "patch",
                "target_binding": {
                    "top_target_file": target,
                    "target_confidence": 0.66,
                    "target_file_candidates": [
                        {"target_file": target, "score": 0.66, "reasons": ["source path matches open-task goal tokens"]},
                    ],
                },
                "grounding": {"target_file": target},
                "last_read": {"path": target, "start_line": 1, "end_line": 2},
                "read_files": [{"path": target}],
                "validation_runs": [],
            },
        },
    }

    candidate = build_local_machine_posterior_action_bridge_candidate(obs, episode_trace=[])

    assert candidate is not None
    assert candidate["function_name"] == "propose_patch"
    assert candidate["kwargs"]["target_file"] == target
    assert candidate["_candidate_meta"]["posterior_action_bonus"] > 0


def test_structured_answer_dedupes_identical_llm_kwargs_requests() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = []

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
            return 'KWARGS_JSON: {"path": "README.md", "start_line": 1, "end_line": 20}'

    llm = FakeLLM()
    synthesizer = StructuredAnswerSynthesizer()
    obs = {
        "local_mirror": {
            "instruction": "Read the project README",
            "deterministic_fallback_enabled": False,
        },
        "function_signatures": {},
    }
    action = {"kind": "call_tool", "payload": {"tool_args": {"function_name": "file_read", "kwargs": {}}}}

    first = synthesizer.maybe_populate_action_kwargs(action, obs, llm_client=llm)
    second = synthesizer.maybe_populate_action_kwargs(action, obs, llm_client=llm)

    assert len(llm.calls) == 1
    assert first["payload"]["tool_args"]["kwargs"] == second["payload"]["tool_args"]["kwargs"]
    second_trace = second["_candidate_meta"]["structured_answer_llm_trace"][0]
    assert second_trace["cache_hit"] is True


def test_structured_answer_does_not_spend_llm_budget_on_grounded_bridge_kwargs() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = []

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
            return 'KWARGS_JSON: {"path": "wrong.py"}'

    llm = FakeLLM()
    synthesizer = StructuredAnswerSynthesizer()
    obs = {
        "local_mirror": {
            "instruction": "Read the grounded file",
            "prefer_llm_kwargs": True,
            "deterministic_fallback_enabled": False,
        },
        "function_signatures": {},
    }
    action = {
        "kind": "call_tool",
        "function_name": "file_read",
        "kwargs": {"path": "core/runtime/runtime_service.py", "start_line": 1, "end_line": 120},
        "payload": {
            "tool_args": {
                "function_name": "file_read",
                "kwargs": {"path": "core/runtime/runtime_service.py", "start_line": 1, "end_line": 120},
            }
        },
        "_source": "local_machine_action_grounding_bridge",
        "_candidate_meta": {},
    }

    updated = synthesizer.maybe_populate_action_kwargs(action, obs, llm_client=llm)

    assert llm.calls == []
    assert updated["payload"]["tool_args"]["kwargs"]["path"] == "core/runtime/runtime_service.py"
    assert updated["_candidate_meta"]["structured_answer_skipped"] is True


def test_auto_route_policy_loader_accepts_codex_route_file(tmp_path: Path) -> None:
    route_policy_path = tmp_path / "llm_route_policies.json"
    route_policy_path.write_text(
        json.dumps(
            {
                "codex_cli_gpt_5_5": {
                    "served_routes": ["planner", "patch_proposal"],
                    "provider": "codex-cli",
                    "base_url": "codex-cli://chatgpt-oauth",
                    "model": "gpt-5.5",
                },
                "ollama_fast": {
                    "served_routes": ["retrieval"],
                    "provider": "ollama",
                    "base_url": "http://ollama.test",
                    "model": "fast-small",
                },
            }
        ),
        encoding="utf-8",
    )

    route_policies, report = _resolve_auto_route_policies(
        llm_provider="codex",
        llm_base_url=None,
        llm_model=None,
        llm_timeout=2.0,
        llm_profile_store=None,
        llm_profile_force=False,
        llm_route_policy_file=str(route_policy_path),
    )

    assert sorted(route_policies) == ["codex_cli_gpt_5_5", "ollama_fast"]
    assert report["provider"] == "codex-cli"
    assert report["model_count"] == 2
    assert report["route_policy_source"] == str(route_policy_path)


def test_auto_route_policy_force_all_profiles_visible_catalogs(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_profile_all_configured_models(**kwargs):
        calls.append(dict(kwargs))
        return {
            "schema_version": "conos.model_profile_report/v1",
            "provider": "all",
            "base_url": "",
            "model_count": 2,
            "generated_count": 2,
            "reused_count": 0,
            "store_path": str(tmp_path / "profiles.json"),
            "route_policies": {
                "codex_cli_gpt_5_5": {
                    "served_routes": ["planner"],
                    "provider": "codex-cli",
                    "base_url": "codex-cli://chatgpt-oauth",
                    "model": "gpt-5.5",
                },
                "ollama_fast": {
                    "served_routes": ["retrieval"],
                    "provider": "ollama",
                    "base_url": "http://ollama.test",
                    "model": "fast-small",
                },
            },
        }

    monkeypatch.setattr("integrations.local_machine.runner.profile_all_configured_models", fake_profile_all_configured_models)
    route_policy_path = tmp_path / "routes.json"

    route_policies, report = _resolve_auto_route_policies(
        llm_provider="all",
        llm_base_url="http://ollama.test",
        llm_model=None,
        llm_timeout=2.0,
        llm_profile_store=str(tmp_path / "profiles.json"),
        llm_profile_force=True,
        llm_route_policy_file=str(route_policy_path),
    )

    assert sorted(route_policies) == ["codex_cli_gpt_5_5", "ollama_fast"]
    assert report["provider"] == "all"
    assert calls[0]["include_codex"] is True
    assert calls[0]["catalog_only"] is True
    assert json.loads(route_policy_path.read_text(encoding="utf-8"))["codex_cli_gpt_5_5"]["model"] == "gpt-5.5"


def test_require_llm_generation_rejects_non_llm_generated_product(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"

    audit = run_local_machine_task(
        instruction="build a complete AI project",
        source_root=str(source),
        mirror_root=str(mirror_root),
        default_command=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path('generated').mkdir(); Path('generated/README.md').write_text('ok\\n', encoding='utf-8')",
        ],
        allowed_commands=[sys.executable],
        run_id="artifact-contract-requires-llm",
        max_ticks_per_episode=3,
        reset_mirror=True,
        allow_empty_exec=True,
        require_artifacts=True,
        require_llm_generation=True,
        deterministic_fallback_enabled=False,
        required_artifact_paths=["generated/README.md"],
        execution_backend="local",
    )

    check = audit["local_machine_artifact_check"]
    assert check["ok"] is False
    assert "llm_provider_enabled" in check["failures"]
    assert "mirror_exec_kwargs_from_llm" in check["failures"]


def test_market_evidence_reference_contract_checks_product_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("Built from artifact market123 at https://example.test/ai-tools\n", encoding="utf-8")

    audit = {
        "final_surface_raw": {
            "local_mirror": {
                "workspace_root": str(workspace),
                "command_executed": True,
                "workspace_file_count": 1,
                "audit_events": [{"event_type": "mirror_command_executed", "payload": {"returncode": 0}}],
                "sync_plan": {"plan_id": "p1", "actionable_changes": [{"relative_path": "README.md"}]},
                "internet_ingress": {
                    "artifact_count": 1,
                    "artifacts": [{"artifact_id": "market123", "normalized_url": "https://example.test/ai-tools"}],
                },
            }
        },
        "local_machine_llm_tool_trace": {"llm_call_count": 0, "tool_call_count": 0, "tool_calls": []},
    }

    check = _artifact_contract_check(audit, require_market_evidence_reference=True)

    assert check["ok"] is True
    assert check["market_evidence_reference_report"]["matched_references"]


def test_non_template_product_verifier_rejects_known_builtin_template(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("SignalBrief AI local-first prompt quality and AI-workflow brief analyzer\n", encoding="utf-8")

    audit = {
        "final_surface_raw": {
            "local_mirror": {
                "workspace_root": str(workspace),
                "command_executed": True,
                "workspace_file_count": 1,
                "audit_events": [{"event_type": "mirror_command_executed", "payload": {"returncode": 0}}],
                "sync_plan": {"plan_id": "p1", "actionable_changes": [{"relative_path": "README.md"}]},
            }
        },
        "local_machine_llm_tool_trace": {"llm_call_count": 0, "tool_call_count": 0, "tool_calls": []},
    }

    check = _artifact_contract_check(audit, require_non_template_product=True)

    assert check["ok"] is False
    assert "non_template_product_verifier_passed" in check["failures"]


class _FakeInternetHeaders:
    def get_content_type(self) -> str:
        return "application/json"

    def get(self, name: str, default: str = "") -> str:
        return "application/json" if name.lower() == "content-type" else default


class _FakeInternetResponse:
    headers = _FakeInternetHeaders()

    def __enter__(self) -> "_FakeInternetResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, _: int) -> bytes:
        return b'{"ok": true}'

    def getcode(self) -> int:
        return 200


def test_local_machine_internet_fetch_is_explicit_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"

    disabled = LocalMachineSurfaceAdapter(
        instruction="fetch public docs",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
    )
    enabled = LocalMachineSurfaceAdapter(
        instruction="fetch public docs",
        source_root=source,
        mirror_root=tmp_path / "mirror-enabled",
        reset_mirror=True,
        internet_enabled=True,
    )

    assert "internet_fetch" not in [tool.name for tool in disabled.reset().available_tools]
    enabled_tools = [tool.name for tool in enabled.reset().available_tools]
    assert "internet_fetch" in enabled_tools
    assert "internet_fetch_project" in enabled_tools


def test_local_machine_internet_fetch_writes_audited_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeInternetResponse:
        assert getattr(request, "full_url") == "https://example.com/data.json"
        assert timeout == 5.0
        return _FakeInternetResponse()

    monkeypatch.setattr("modules.internet.ingress.urlopen", fake_urlopen)
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fetch public data",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        internet_enabled=True,
        internet_max_bytes=1024,
        internet_timeout_seconds=5,
    )

    adapter.reset()
    result = adapter.act(
        {
            "function_name": "internet_fetch",
            "kwargs": {"url": "https://example.com/data.json"},
        }
    )

    assert result.ok is True
    artifact = result.raw["internet_artifact"]
    assert artifact["content_type"] == "application/json"
    assert Path(artifact["local_path"]).read_text(encoding="utf-8") == '{"ok": true}'
    assert result.raw["network_policy_audit"]["default_policy"] == "deny_private_by_default"
    assert result.raw["network_policy_audit"]["host"] == "example.com"
    assert result.raw["network_policy_audit"]["credential_values_redacted"] is True
    observed = result.observation.raw["local_mirror"]["internet_ingress"]
    assert observed["artifact_count"] == 1
    assert (mirror_root / "control" / "internet" / "events.jsonl").exists()
    assert result.raw["action_governance"]["status"] == "ALLOWED"
    assert result.raw["side_effect_audit"]["network_policy_audit"]["host"] == "example.com"


def test_local_machine_internet_fetch_respects_host_allowlist(tmp_path: Path, monkeypatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeInternetResponse:
        raise AssertionError("network fetch should be blocked before urlopen")

    monkeypatch.setattr("modules.internet.ingress.urlopen", fake_urlopen)
    source = tmp_path / "source"
    source.mkdir()
    adapter = LocalMachineSurfaceAdapter(
        instruction="fetch public data with host policy",
        source_root=source,
        mirror_root=tmp_path / "mirror",
        reset_mirror=True,
        internet_enabled=True,
        internet_allowed_hosts=["allowed.example"],
    )

    adapter.reset()
    result = adapter.act(
        {
            "function_name": "internet_fetch",
            "kwargs": {"url": "https://example.com/data.json"},
        }
    )

    assert result.ok is False
    assert "url host is not in allowed_hosts: example.com" in result.raw["failure_reason"]
    assert result.raw["action_governance"]["status"] == "ALLOWED"
    assert result.raw["side_effect_audit_enforced"] is True


def test_local_machine_internet_private_network_requires_governance_approval(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    adapter = LocalMachineSurfaceAdapter(
        instruction="fetch private service",
        source_root=source,
        mirror_root=tmp_path / "mirror",
        reset_mirror=True,
        internet_enabled=True,
        internet_allow_private_networks=True,
    )

    adapter.reset()
    result = adapter.act(
        {
            "function_name": "internet_fetch",
            "kwargs": {"url": "http://192.168.0.2/data.json"},
        }
    )

    assert result.ok is False
    assert result.raw["state"] == "WAITING_APPROVAL"
    assert result.raw["approval_state"]["status"] == "WAITING_APPROVAL"
    assert result.raw["action_governance"]["status"] == "WAITING_APPROVAL"
    assert result.raw["action_governance"]["required_approval"] is True
    assert "private_network_access_requires_approval" in result.raw["failure_reason"]


def test_local_machine_capability_approval_enters_runtime_inbox_and_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeInternetResponse:
        assert getattr(request, "full_url") == "https://example.com/data.json"
        return _FakeInternetResponse()

    monkeypatch.setattr("modules.internet.ingress.urlopen", fake_urlopen)
    source = tmp_path / "source"
    source.mkdir()
    db_path = tmp_path / "state.sqlite3"
    run_id = "capability-approval-run"
    supervisor = LongRunSupervisor(db_path=db_path)
    supervisor.create_run("fetch public data", run_id=run_id)
    supervisor.add_task(run_id, "fetch public data", verifier={"kind": "local_machine"})

    waiting_adapter = LocalMachineSurfaceAdapter(
        instruction="fetch public data",
        source_root=source,
        mirror_root=tmp_path / "mirror-waiting",
        reset_mirror=True,
        internet_enabled=True,
        evidence_db_path=db_path,
        task_id=run_id,
        action_governance_policy={"approval_required_capability_layers": ["network"]},
    )
    waiting_adapter.reset()
    waiting = waiting_adapter.act(
        {"function_name": "internet_fetch", "kwargs": {"url": "https://example.com/data.json"}}
    )
    approval_id = waiting.raw["approval_id"]
    inbox = supervisor.state_store.list_approvals(run_id, status="WAITING")

    assert waiting.ok is False
    assert waiting.raw["state"] == "WAITING_APPROVAL"
    assert approval_id
    assert inbox and inbox[0]["approval_id"] == approval_id
    assert inbox[0]["request"]["type"] == "action_capability_approval"
    assert inbox[0]["request"]["required_capability_layers"] == ["network"]
    assert supervisor.state_store.get_run(run_id)["status"] == "WAITING_APPROVAL"

    approved = supervisor.approve(approval_id, approved_by="test")
    assert approved["status"] == "APPROVED"
    assert approved["approval"]["response"]["approved_capability_layers"] == ["network"]

    resumed_adapter = LocalMachineSurfaceAdapter(
        instruction="fetch public data",
        source_root=source,
        mirror_root=tmp_path / "mirror-resumed",
        reset_mirror=True,
        internet_enabled=True,
        evidence_db_path=db_path,
        task_id=run_id,
        action_governance_policy={"approval_required_capability_layers": ["network"]},
    )
    resumed_adapter.reset()
    resumed = resumed_adapter.act(
        {"function_name": "internet_fetch", "kwargs": {"url": "https://example.com/data.json"}}
    )

    assert resumed.ok is True
    assert resumed.raw["action_governance"]["status"] == "ALLOWED"
    assert resumed.observation.raw["local_mirror"]["action_governance"]["approved_capability_layers"] == ["network"]

    supervisor.state_store.close()


def test_local_machine_internet_fetch_blocks_inline_credentials(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    adapter = LocalMachineSurfaceAdapter(
        instruction="fetch public docs",
        source_root=source,
        mirror_root=tmp_path / "mirror",
        reset_mirror=True,
        internet_enabled=True,
    )

    adapter.reset()
    result = adapter.act(
        {
            "function_name": "internet_fetch",
            "kwargs": {
                "url": "https://example.com/data.json",
                "headers": {"Authorization": "Bearer secret"},
            },
        }
    )

    assert result.ok is False
    assert result.raw["action_governance"]["status"] == "BLOCKED"
    assert "inline_credentials_not_allowed" in result.raw["failure_reason"]


def test_local_machine_internet_fetch_project_extracts_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("project/README.md", "ok\n")

    def fake_urlopen(request: object, timeout: float) -> _FakeInternetResponse:
        class ZipResponse(_FakeInternetResponse):
            def read(self, _: int) -> bytes:
                return buffer.getvalue()

        return ZipResponse()

    monkeypatch.setattr("modules.internet.ingress.urlopen", fake_urlopen)
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fetch public project",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        internet_enabled=True,
        internet_max_bytes=4096,
        execution_backend="local",
    )

    adapter.reset()
    result = adapter.act(
        {
            "function_name": "internet_fetch_project",
            "kwargs": {
                "url": "https://example.com/project.zip",
                "source_type": "archive",
            },
        }
    )

    assert result.ok is True
    artifact = result.raw["internet_artifact"]
    assert artifact["fetch_kind"] == "archive_extract"
    assert Path(artifact["local_path"], "project", "README.md").read_text(encoding="utf-8") == "ok\n"
    observed = result.observation.raw["local_mirror"]["internet_ingress"]
    assert observed["artifact_count"] == 2


def test_internet_project_baseline_keeps_raw_diff_out_of_prompt_and_sync_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("project/README.md", "# Demo\n")
        archive.writestr("project/app.py", "print('ok')\n")

    def fake_urlopen(request: object, timeout: float) -> _FakeInternetResponse:
        class ZipResponse(_FakeInternetResponse):
            def read(self, _: int) -> bytes:
                return buffer.getvalue()

        return ZipResponse()

    monkeypatch.setattr("modules.internet.ingress.urlopen", fake_urlopen)
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    adapter = LocalMachineSurfaceAdapter(
        instruction="fetch and improve project",
        source_root=source,
        mirror_root=mirror_root,
        reset_mirror=True,
        internet_enabled=True,
        internet_max_bytes=4096,
        execution_backend="local",
    )

    adapter.reset()
    fetched = adapter.act(
        {
            "function_name": "internet_fetch_project",
            "kwargs": {"url": "https://example.com/project.zip", "source_type": "archive"},
        }
    )
    mirror = fetched.observation.raw["local_mirror"]
    assert mirror["diff_summary"]["entry_count"] == 0
    assert mirror["diff_ref"]["object_type"] == "raw_diff"
    assert mirror["external_baselines"][0]["workspace_relative_path"].startswith("internet_projects/")

    prompt = StructuredAnswerSynthesizer()._build_llm_prompt("mirror_exec", fetched.observation.raw)
    assert "diff_entries" not in prompt
    assert "text_patch" not in prompt
    assert len(prompt) < 12000

    exec_result = adapter.act(
        {
            "function_name": "mirror_exec",
                "kwargs": {
                    "command": [
                        sys.executable,
                        "-c",
                        (
                        "from pathlib import Path; "
                        "Path('internet_projects').glob('*'); "
                        "root=next(Path('internet_projects').glob('*')); "
                            "(root/'CONOS_IMPROVEMENT.md').write_text('improved\\n', encoding='utf-8')"
                        ),
                    ],
                    "allowed_commands": [sys.executable],
                    "purpose": "build",
                    "target": "internet_projects",
                    "timeout_seconds": 10,
                },
            }
        )
    assert exec_result.ok is True
    planned = adapter.act({"function_name": "mirror_plan"})
    assert planned.ok is False
    assert planned.raw["state"] == "MIRROR_PLAN_BLOCKED"
    assert planned.raw["mirror_plan_blocked_reason"] == "no_verified_changes"
