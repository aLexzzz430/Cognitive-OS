from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.local_machine.runner import run_local_machine_task
from integrations.local_machine.task_adapter import LocalMachineSurfaceAdapter


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
