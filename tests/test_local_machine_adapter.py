from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.local_machine.runner import run_local_machine_task
from integrations.local_machine.task_adapter import LocalMachineSurfaceAdapter
from core.runtime.state_store import RuntimeStateStore
from planner.objective_decomposer import ObjectiveDecomposer


def test_local_machine_adapter_keeps_source_unchanged_until_plan_apply(tmp_path: Path) -> None:
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
    assert planned.ok is True
    plan = planned.raw["sync_plan"]
    assert plan["approval"]["status"] == "machine_approved"
    assert plan["actionable_changes"][0]["relative_path"] == "README.md"
    assert (source / "README.md").read_text(encoding="utf-8") == "before\n"

    applied = adapter.act(
        {
            "function_name": "mirror_apply",
            "kwargs": {"plan_id": plan["plan_id"], "approved_by": "machine"},
        }
    )
    assert applied.ok is True
    assert (source / "README.md").read_text(encoding="utf-8") == "after\n"


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
    )

    final_mirror = audit["final_surface_raw"]["local_mirror"]
    assert audit["world_provider_source"] == "integrations.local_machine.runner"
    assert final_mirror["workspace_initial_state"] == "empty"
    assert final_mirror["workspace_file_count"] >= 1
    assert (mirror_root / "workspace" / "README.md").exists()
    assert (source / "README.md").read_text(encoding="utf-8") == "hello\n"


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

    assert planned.raw["state"] == "WAITING_APPROVAL"
    assert planned.raw["waiting_approval"] is True
    assert planned.raw["terminal"] is False
    assert planned.observation.terminal is False
    assert planned.raw["approval_request"]["type"] == "local_mirror_sync_plan"


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
    )

    obs = adapter.reset()

    assert obs.raw["local_mirror"]["workspace_file_count"] == 0
    assert [tool.name for tool in obs.available_tools][0] == "mirror_exec"


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
    )

    adapter.reset()
    executed = adapter.act({"function_name": "mirror_exec"})
    assert executed.ok is True
    assert (mirror_root / "workspace" / "generated" / "app.py").exists()
    assert [tool.name for tool in executed.observation.available_tools] == ["mirror_plan"]

    planned = adapter.act({"function_name": "mirror_plan"})

    assert planned.raw["state"] == "WAITING_APPROVAL"
    assert planned.raw["sync_plan"]["actionable_changes"][0]["relative_path"] == "generated/app.py"


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
    )

    adapter.reset()
    executed = adapter.act({"function_name": "mirror_exec"})

    assert executed.ok is True
    assert executed.raw["mirror_command"]["timeout_seconds"] == 120


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
    )

    check = audit["local_machine_artifact_check"]
    assert check["ok"] is False
    assert check["latest_command_returncode"] == 1
    assert "latest_command_succeeded" in check["failures"]
    assert check["required_workspace_path_matches"]["generated/app.py"] == ["generated/app.py"]


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
    )

    check = audit["local_machine_artifact_check"]
    assert check["ok"] is False
    assert "required_workspace_path:generated/*/pyproject.toml" in check["failures"]


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
