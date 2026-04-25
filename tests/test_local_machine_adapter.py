from __future__ import annotations

import json
import sys
import io
from pathlib import Path
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.local_machine.runner import _artifact_contract_check, run_local_machine_task
from integrations.local_machine.task_adapter import LocalMachineSurfaceAdapter
from core.runtime.state_store import RuntimeStateStore
from core.orchestration.structured_answer import StructuredAnswerSynthesizer
from core.orchestration.goal_task_control import resolve_effective_task_approval_requirement
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
    )

    obs = adapter.reset()
    tools = [tool.name for tool in obs.available_tools]

    assert tools[0] == "repo_tree"
    assert "mirror_exec" in tools
    assert "internet_fetch" not in tools


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
    assert "requires meaningful actionable changes" in planned.raw["failure_reason"]
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


def test_action_governance_requires_validation_before_code_sync(tmp_path: Path) -> None:
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
    assert planned.ok is True

    rejected = adapter.act(
        {
            "action": "mirror_apply",
            "args": {"plan_id": planned.raw["sync_plan"]["plan_id"], "approved_by": "machine"},
        }
    )
    assert rejected.ok is False
    assert rejected.raw["action_governance"]["status"] == "BLOCKED"
    assert "passing_validation_required_before_source_sync" in rejected.raw["failure_reason"]
    assert (source / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 1\n"

    lint = adapter.act({"action": "run_lint", "args": {"target": "app.py", "timeout_seconds": 10}})
    assert lint.ok is True
    applied = adapter.act(
        {
            "action": "mirror_apply",
            "args": {"plan_id": planned.raw["sync_plan"]["plan_id"], "approved_by": "machine"},
        }
    )
    assert applied.ok is True
    assert (source / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 2\n"


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
    observed = result.observation.raw["local_mirror"]["internet_ingress"]
    assert observed["artifact_count"] == 1
    assert (mirror_root / "control" / "internet" / "events.jsonl").exists()


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
    changes = planned.raw["sync_plan"]["actionable_changes"]
    assert [row["relative_path"] for row in changes] == [
        mirror["external_baselines"][0]["workspace_relative_path"] + "/CONOS_IMPROVEMENT.md"
    ]
