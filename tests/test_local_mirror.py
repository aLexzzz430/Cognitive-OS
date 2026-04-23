from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import conos_cli
from modules.local_mirror import (
    LOCAL_MIRROR_VERSION,
    LOCAL_MIRROR_SYNC_PLAN_VERSION,
    MirrorScopeError,
    apply_sync_plan,
    acquire_relevant_files,
    build_sync_plan,
    compute_mirror_diff,
    create_empty_mirror,
    materialize_files,
    run_mirror_command,
)


def test_empty_mirror_has_no_workspace_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("source file", encoding="utf-8")
    mirror_root = tmp_path / "mirror"

    mirror = create_empty_mirror(source, mirror_root)

    assert mirror.workspace_root.exists()
    assert mirror.control_root.exists()
    assert mirror.workspace_is_empty()
    assert mirror.to_manifest()["schema_version"] == LOCAL_MIRROR_VERSION
    assert mirror.to_manifest()["workspace_file_count"] == 0
    assert mirror.manifest_path.exists()


def test_materialize_files_copies_only_explicit_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("readme", encoding="utf-8")
    (source / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)

    mirror = materialize_files(source, mirror_root, ["README.md"])

    assert (mirror.workspace_root / "README.md").read_text(encoding="utf-8") == "readme"
    assert not (mirror.workspace_root / "pyproject.toml").exists()
    manifest = json.loads(mirror.manifest_path.read_text(encoding="utf-8"))
    assert manifest["workspace_file_count"] == 1
    assert manifest["materialized_files"][0]["relative_path"] == "README.md"


def test_instruction_scoped_acquisition_materializes_relevant_candidates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("readme", encoding="utf-8")
    (source / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"

    mirror = acquire_relevant_files(
        source,
        mirror_root,
        instruction="inspect pyproject before changing dependencies",
        candidate_paths=["README.md", "pyproject.toml"],
    )

    assert (mirror.workspace_root / "pyproject.toml").exists()
    assert not (mirror.workspace_root / "README.md").exists()
    event_types = [event["event_type"] for event in mirror.audit_events]
    assert "instruction_scoped_acquisition" in event_types


def test_mirror_rejects_paths_outside_declared_source_scope(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)

    with pytest.raises(MirrorScopeError):
        materialize_files(source, mirror_root, ["../secret.txt"])

    with pytest.raises(MirrorScopeError):
        materialize_files(source, mirror_root, [str((tmp_path / "secret.txt").resolve())])


def test_conos_mirror_cli_init_and_fetch(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("readme", encoding="utf-8")
    mirror_root = tmp_path / "mirror"

    assert conos_cli.main(["mirror", "init", "--source-root", str(source), "--mirror-root", str(mirror_root)]) == 0
    init_payload = json.loads(capsys.readouterr().out)
    assert init_payload["workspace_file_count"] == 0

    assert (
        conos_cli.main(
            [
                "mirror",
                "fetch",
                "--source-root",
                str(source),
                "--mirror-root",
                str(mirror_root),
                "--path",
                "README.md",
            ]
        )
        == 0
    )
    fetch_payload = json.loads(capsys.readouterr().out)
    assert fetch_payload["workspace_file_count"] == 1
    assert (mirror_root / "workspace" / "README.md").exists()


def test_mirror_command_changes_only_workspace_until_sync(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    materialize_files(source, mirror_root, ["README.md"])

    result = run_mirror_command(
        source,
        mirror_root,
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('README.md').write_text('after\\n', encoding='utf-8')",
        ],
        allowed_commands=[sys.executable],
    )

    assert result.returncode == 0
    assert (mirror_root / "workspace" / "README.md").read_text(encoding="utf-8") == "after\n"
    assert (source / "README.md").read_text(encoding="utf-8") == "before\n"


def test_mirror_diff_and_sync_plan_require_review_gate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    materialize_files(source, mirror_root, ["README.md"])
    (mirror_root / "workspace" / "README.md").write_text("after\n", encoding="utf-8")

    diff = compute_mirror_diff(source, mirror_root)
    plan = build_sync_plan(source, mirror_root)

    assert diff[0].status == "modified"
    assert "-before" in diff[0].text_patch
    assert "+after" in diff[0].text_patch
    assert plan["schema_version"] == LOCAL_MIRROR_SYNC_PLAN_VERSION
    assert plan["approval"]["status"] == "machine_approved"
    assert plan["actionable_changes"][0]["relative_path"] == "README.md"

    with pytest.raises(MirrorScopeError):
        apply_sync_plan(source, mirror_root, plan_id="wrong", approved_by="machine")

    result = apply_sync_plan(source, mirror_root, plan_id=plan["plan_id"], approved_by="machine")
    assert result["synced_files"][0]["relative_path"] == "README.md"
    assert (source / "README.md").read_text(encoding="utf-8") == "after\n"


def test_sync_plan_requires_human_review_for_failed_mirror_command(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    materialize_files(source, mirror_root, ["README.md"])

    result = run_mirror_command(
        source,
        mirror_root,
        [sys.executable, "-c", "raise SystemExit(2)"],
        allowed_commands=[sys.executable],
    )
    assert result.returncode == 2
    (mirror_root / "workspace" / "README.md").write_text("after\n", encoding="utf-8")

    plan = build_sync_plan(source, mirror_root)

    assert plan["approval"]["status"] == "human_review_required"
    assert plan["approval"]["human_required"] is True
    with pytest.raises(MirrorScopeError):
        apply_sync_plan(source, mirror_root, plan_id=plan["plan_id"], approved_by="machine")


def test_conos_mirror_exec_plan_and_apply_cli(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    assert (
        conos_cli.main(
            [
                "mirror",
                "fetch",
                "--source-root",
                str(source),
                "--mirror-root",
                str(mirror_root),
                "--path",
                "README.md",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        conos_cli.main(
            [
                "mirror",
                "exec",
                "--source-root",
                str(source),
                "--mirror-root",
                str(mirror_root),
                "--allow-command",
                sys.executable,
                "--",
                sys.executable,
                "-c",
                "from pathlib import Path; Path('README.md').write_text('after\\n', encoding='utf-8')",
            ]
        )
        == 0
    )
    exec_payload = json.loads(capsys.readouterr().out)
    assert exec_payload["returncode"] == 0

    assert conos_cli.main(["mirror", "plan", "--source-root", str(source), "--mirror-root", str(mirror_root)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["approval"]["status"] == "machine_approved"

    assert (
        conos_cli.main(
            [
                "mirror",
                "apply",
                "--source-root",
                str(source),
                "--mirror-root",
                str(mirror_root),
                "--plan-id",
                plan["plan_id"],
                "--approved-by",
                "machine",
            ]
        )
        == 0
    )
    apply_payload = json.loads(capsys.readouterr().out)
    assert apply_payload["synced_files"][0]["relative_path"] == "README.md"
    assert (source / "README.md").read_text(encoding="utf-8") == "after\n"
