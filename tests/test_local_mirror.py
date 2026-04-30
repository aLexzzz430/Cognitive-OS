from __future__ import annotations

import base64
import errno
import gzip
import hashlib
import importlib.util
import json
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import uuid
from io import BytesIO
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
    bootstrap_managed_vm_image,
    boot_managed_vm_instance,
    build_managed_vm_guest_initrd_bundle,
    build_managed_vm_cloud_init_seed,
    build_managed_vm_helper,
    build_managed_vm_linux_base_image,
    build_managed_vm_virtualization_runner,
    create_managed_vm_base_image_bundle,
    build_vm_workspace_command,
    build_sync_plan,
    compute_mirror_diff,
    create_managed_vm_pinned_artifact_recipe,
    create_managed_vm_blank_image,
    create_empty_mirror,
    ensure_managed_vm_instance_running,
    execution_boundary_report,
    init_managed_vm_state,
    install_default_managed_vm_image,
    install_managed_vm_base_image_bundle,
    load_managed_vm_image_manifest,
    load_managed_vm_instance_manifest,
    load_managed_vm_runtime_manifest,
    managed_vm_health_check,
    managed_vm_recovery_drill,
    managed_vm_recovery_soak,
    managed_vm_guest_agent_gate,
    managed_vm_guest_agent_status,
    managed_vm_recipe_report,
    managed_vm_runtime_status,
    managed_vm_report,
    manage_vm_workspace,
    materialize_files,
    prepare_managed_vm_instance,
    pull_workspace_from_vm,
    push_workspace_to_vm,
    register_managed_vm_base_image,
    register_managed_vm_cloud_init_image,
    register_managed_vm_linux_boot_image,
    recover_managed_vm_instance,
    resolve_managed_vm_artifact_recipe,
    rollback_sync_plan,
    run_mirror_command,
    run_managed_vm_agent_command,
    start_managed_vm_instance,
    stop_managed_vm_instance,
    sync_vm_workspace,
    vm_manager_report,
)
import modules.local_mirror.mirror as mirror_module
import modules.local_mirror.managed_vm as managed_vm_module
import modules.local_mirror.vm_backend as vm_backend_module
import modules.local_mirror.vm_workspace_sync as vm_sync_module


def _copy_file_efficient_for_test(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"method": "test_copy2", "status": "COPIED", "source": str(source), "destination": str(destination)}


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


def test_materialize_files_expands_explicit_directories_with_bounds(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = source / "pkg"
    package.mkdir(parents=True)
    (package / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
    (package / "nested").mkdir()
    (package / "nested" / "beta.py").write_text("beta = 2\n", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "ignored.pyc").write_bytes(b"cache")
    (source / "README.md").write_text("readme", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)

    mirror = materialize_files(source, mirror_root, ["pkg"])

    assert (mirror.workspace_root / "pkg" / "alpha.py").exists()
    assert (mirror.workspace_root / "pkg" / "nested" / "beta.py").exists()
    assert not (mirror.workspace_root / "pkg" / "__pycache__" / "ignored.pyc").exists()
    assert not (mirror.workspace_root / "README.md").exists()
    manifest = json.loads(mirror.manifest_path.read_text(encoding="utf-8"))
    assert manifest["workspace_file_count"] == 2
    event = next(event for event in manifest["audit_events"] if event["event_type"] == "directory_materialization_expanded")
    assert event["payload"]["requested_path"] == "pkg"
    assert event["payload"]["file_count"] == 2
    assert event["payload"]["skipped_count"] >= 1


def test_materialize_files_rejects_directory_expansion_past_file_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    package = source / "pkg"
    package.mkdir(parents=True)
    (package / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
    (package / "beta.py").write_text("beta = 2\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)
    monkeypatch.setattr(mirror_module, "MAX_DIRECTORY_MATERIALIZATION_FILES", 1)

    with pytest.raises(MirrorScopeError, match="directory materialization exceeds file limit"):
        materialize_files(source, mirror_root, ["pkg"])


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


def test_conos_mirror_cli_boundary_reports_local_best_effort(capsys) -> None:
    assert conos_cli.main(["mirror", "boundary", "--backend", "local"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "conos.local_mirror.execution_boundary/v1"
    assert payload["status"] == "AVAILABLE"
    assert payload["security_boundary"] == "best_effort_local_process"
    assert payload["real_vm_boundary"] is False
    assert "not_a_vm" in payload["limitations"]


def test_conos_mirror_cli_boundary_defaults_to_managed_vm(capsys) -> None:
    assert conos_cli.main(["mirror", "boundary"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "conos.local_mirror.execution_boundary/v1"
    assert payload["backend"] == "managed-vm"
    assert payload["security_boundary"] == "conos_managed_vm_provider"
    assert "does_not_fall_back_to_host_process" in payload["limitations"]


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
        backend="local",
    )

    assert result.returncode == 0
    assert (mirror_root / "workspace" / "README.md").read_text(encoding="utf-8") == "after\n"
    assert (source / "README.md").read_text(encoding="utf-8") == "before\n"


def test_mirror_command_supports_docker_backend_command_construction(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)
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

    result = run_mirror_command(
        source,
        mirror_root,
        [sys.executable, "-c", "print('ok')"],
        allowed_commands=[sys.executable],
        backend="docker",
        docker_image="python:3.10-slim",
    )

    assert result.backend == "docker"
    assert result.returncode == 0
    assert result.execution_boundary["security_boundary"] == "container_best_effort"
    assert result.execution_boundary["network_boundary"] == "none"
    assert result.execution_boundary["real_vm_boundary"] is False
    assert captured["cmd"][:5] == ["/usr/bin/docker", "run", "--rm", "--network", "none"]
    assert "/workspace" in captured["cmd"]
    assert "python:3.10-slim" in captured["cmd"]
    assert captured["cmd"][-3:] == ["python", "-c", "print('ok')"]


def test_mirror_command_local_backend_uses_sanitized_explicit_env(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)
    captured = {}

    class Completed:
        returncode = 0
        stdout = "visible explicit-env-value\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return Completed()

    monkeypatch.setenv("CONOS_HOST_SECRET_TOKEN", "host-secret-value")
    monkeypatch.setattr(mirror_module.subprocess, "run", fake_run)

    result = run_mirror_command(
        source,
        mirror_root,
        [sys.executable, "-c", "print('done')"],
        allowed_commands=[sys.executable],
        backend="local",
        extra_env={"CONOS_SAFE_HINT": "explicit-env-value"},
    )

    run_env = captured["kwargs"]["env"]
    assert result.returncode == 0
    assert run_env["CONOS_SAFE_HINT"] == "explicit-env-value"
    assert "CONOS_HOST_SECRET_TOKEN" not in run_env
    assert result.env_audit["host_env_passthrough"] is False
    assert result.env_audit["explicit_env_keys"] == ["CONOS_SAFE_HINT"]
    assert result.env_audit["values_redacted_in_audit"] is True

    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    payload = manifest["audit_events"][-1]["payload"]
    assert payload["credential_boundary"] == "sanitized_process_env_explicit_env_only_redacted_in_audit"
    assert payload["host_env_forwarded"] is False
    assert payload["host_env_passthrough"] is False
    assert payload["env_audit"]["explicit_env_keys"] == ["CONOS_SAFE_HINT"]
    assert "explicit-env-value" not in json.dumps(payload)
    assert "<redacted:CONOS_SAFE_HINT>" in payload["stdout_tail"]


def test_mirror_command_supports_lima_vm_backend_command_construction(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)
    captured = {}

    class Completed:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return Completed()

    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: "/usr/bin/limactl" if name == "limactl" else None)
    monkeypatch.setattr(vm_backend_module.subprocess, "run", fake_run)

    result = run_mirror_command(
        source,
        mirror_root,
        [sys.executable, "-c", "print('ok')"],
        allowed_commands=[sys.executable],
        backend="vm",
        vm_provider="lima",
        vm_name="conos-test",
        vm_workdir="/workspace",
    )

    assert result.backend == "vm"
    assert result.returncode == 0
    assert result.vm_provider == "lima"
    assert result.vm_name == "conos-test"
    assert result.real_vm_boundary is True
    assert result.security_boundary == "external_vm_provider"
    assert result.execution_boundary["status"] == "AVAILABLE"
    assert result.execution_boundary["real_vm_boundary"] is True
    assert captured["cmd"][:4] == ["/usr/bin/limactl", "shell", "conos-test", "bash"]
    assert captured["cmd"][4] == "-lc"
    assert captured["cmd"][5].startswith("cd /workspace && exec ")
    assert sys.executable in captured["cmd"][5]
    assert "print(" in captured["cmd"][5]
    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    last_event = manifest["audit_events"][-1]
    assert last_event["event_type"] == "mirror_command_executed"
    assert last_event["payload"]["sandbox_label"] == "external_vm_provider"
    assert last_event["payload"]["not_os_security_sandbox"] is False
    assert last_event["payload"]["execution_boundary"]["real_vm_boundary"] is True


def test_mirror_command_vm_backend_requires_real_provider_config(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)
    monkeypatch.delenv("CONOS_VM_NAME", raising=False)
    monkeypatch.delenv("CONOS_LIMA_INSTANCE", raising=False)
    monkeypatch.delenv("CONOS_VM_SSH_HOST", raising=False)
    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: None)
    unavailable = {
        "status": "UNAVAILABLE",
        "reason": "managed VM provider disabled for provider-auto test",
        "real_vm_boundary": False,
    }
    monkeypatch.setattr(mirror_module, "managed_vm_report", lambda **kwargs: dict(unavailable))
    monkeypatch.setattr(vm_backend_module, "managed_vm_report", lambda **kwargs: dict(unavailable))

    with pytest.raises(MirrorScopeError, match="requires a real VM provider"):
        run_mirror_command(
            source,
            mirror_root,
            [sys.executable, "-c", "print('ok')"],
            allowed_commands=[sys.executable],
            backend="vm",
        )
    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit_events"][-1]["event_type"] == "mirror_vm_backend_unavailable"
    assert manifest["audit_events"][-1]["payload"]["execution_boundary"]["status"] == "UNAVAILABLE"


def test_execution_boundary_report_classifies_backends(monkeypatch) -> None:
    monkeypatch.setattr(mirror_module.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: "/usr/bin/limactl" if name == "limactl" else None)

    local = execution_boundary_report(backend="local")
    docker = execution_boundary_report(backend="docker", docker_image="python:3.10-slim")
    vm = execution_boundary_report(backend="vm", vm_provider="lima", vm_name="conos-test", vm_network_mode="configured_isolated")

    assert local["security_boundary"] == "best_effort_local_process"
    assert local["real_vm_boundary"] is False
    assert docker["security_boundary"] == "container_best_effort"
    assert docker["network_boundary"] == "none"
    assert vm["status"] == "AVAILABLE"
    assert vm["provider"] == "lima"
    assert vm["real_vm_boundary"] is True
    assert vm["vm_network_mode"] == "configured_isolated"
    assert vm["credential_boundary"] == "vm_guest_isolated_explicit_env_only_redacted_in_audit"
    assert vm["host_env_forwarded_to_guest"] is False


def test_managed_vm_report_and_init_use_conos_state_root(tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    state_root = tmp_path / "vm-state"
    monkeypatch.delenv("CONOS_MANAGED_VM_HELPER", raising=False)
    monkeypatch.setattr(managed_vm_module.shutil, "which", lambda name: None)

    unavailable = managed_vm_report(state_root=str(state_root))
    initialized = init_managed_vm_state(
        state_root=str(state_root),
        helper_path=str(helper),
        image_id="conos-test-image",
        instance_id="task-1",
    )
    available = managed_vm_report(state_root=str(state_root), helper_path=str(helper), image_id="conos-test-image", instance_id="task-1")

    assert unavailable["status"] == "UNAVAILABLE"
    assert unavailable["requires_user_configured_vm"] is False
    assert initialized["helper_available"] is True
    assert (state_root / "images").is_dir()
    assert (state_root / "instances").is_dir()
    assert available["status"] == "AVAILABLE"
    assert available["real_vm_boundary"] is False
    assert available["virtualization_runner_available"] is False
    assert available["manifest_present"] is True


def test_execution_boundary_report_classifies_managed_vm(tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    runner = tmp_path / "conos-vz-runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("CONOS_MANAGED_VM_HELPER", str(helper))
    monkeypatch.setenv("CONOS_MANAGED_VM_RUNNER", str(runner))

    report = execution_boundary_report(backend="managed-vm", vm_name="task-1", vm_network_mode="configured_isolated")

    assert report["status"] == "AVAILABLE"
    assert report["backend"] == "managed-vm"
    assert report["provider"] == "managed"
    assert report["real_vm_boundary"] is True
    assert report["managed_vm"]["requires_user_configured_vm"] is False
    assert report["credential_boundary"] == "vm_guest_isolated_explicit_env_only_redacted_in_audit"
    assert report["host_env_forwarded_to_guest"] is False


def test_conos_vm_init_and_report(capsys, tmp_path: Path) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    state_root = tmp_path / "vm-state"

    assert (
        conos_cli.main(
            [
                "vm",
                "init",
                "--state-root",
                str(state_root),
                "--helper-path",
                str(helper),
                "--image-id",
                "conos-test-image",
                "--instance-id",
                "task-1",
            ]
        )
        == 0
    )
    init_payload = json.loads(capsys.readouterr().out)
    assert init_payload["helper_available"] is True
    assert init_payload["image_id"] == "conos-test-image"

    assert (
        conos_cli.main(
            [
                "vm",
                "report",
                "--state-root",
                str(state_root),
                "--helper-path",
                str(helper),
                "--image-id",
                "conos-test-image",
                "--instance-id",
                "task-1",
            ]
        )
        == 0
    )
    report_payload = json.loads(capsys.readouterr().out)
    assert report_payload["status"] == "AVAILABLE"
    assert report_payload["manifest_present"] is True


def test_register_managed_vm_base_image_copies_and_hashes_disk(tmp_path: Path) -> None:
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"conos test disk\n")
    state_root = tmp_path / "vm-state"

    payload = register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    manifest = load_managed_vm_image_manifest(state_root=str(state_root), image_id="unit-image")

    assert payload["status"] == "REGISTERED"
    assert payload["image_id"] == "unit-image"
    assert payload["byte_size"] == len(b"conos test disk\n")
    assert payload["sha256"]
    assert Path(payload["disk_path"]).read_bytes() == b"conos test disk\n"
    assert manifest["sha256"] == payload["sha256"]
    assert manifest["boot_verified"] is False


def test_create_managed_vm_blank_image_is_not_execution_ready(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"

    payload = create_managed_vm_blank_image(
        state_root=str(state_root),
        image_id="blank-image",
        size_mb=1,
    )
    manifest = load_managed_vm_image_manifest(state_root=str(state_root), image_id="blank-image")

    assert payload["status"] == "BLANK_CREATED"
    assert payload["build_method"] == "blank_sparse_disk"
    assert payload["byte_size"] == 1024 * 1024
    assert payload["bootable"] is False
    assert payload["execution_ready"] is False
    assert Path(payload["disk_path"]).stat().st_size == 1024 * 1024
    assert manifest["sha256"] == payload["sha256"]


def test_create_managed_vm_blank_image_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    create_managed_vm_blank_image(state_root=str(state_root), image_id="blank-image", size_mb=1)

    with pytest.raises(FileExistsError):
        create_managed_vm_blank_image(state_root=str(state_root), image_id="blank-image", size_mb=1)

    payload = create_managed_vm_blank_image(
        state_root=str(state_root),
        image_id="blank-image",
        size_mb=2,
        overwrite=True,
    )
    assert payload["byte_size"] == 2 * 1024 * 1024


def test_prepare_managed_vm_instance_requires_registered_image(tmp_path: Path) -> None:
    missing = prepare_managed_vm_instance(
        state_root=str(tmp_path / "vm-state"),
        image_id="missing-image",
        instance_id="task-1",
    )

    assert missing["status"] == "UNAVAILABLE"
    assert missing["reason"] == "managed VM base image is not registered"


def test_prepare_managed_vm_instance_writes_instance_manifest(tmp_path: Path) -> None:
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"disk bytes\n")
    state_root = tmp_path / "vm-state"
    image = register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )

    instance = prepare_managed_vm_instance(
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="task-1",
        network_mode="configured_isolated",
    )
    manifest = load_managed_vm_instance_manifest(state_root=str(state_root), instance_id="task-1")
    report = managed_vm_report(state_root=str(state_root), image_id="unit-image", instance_id="task-1")

    assert instance["status"] == "PREPARED"
    assert instance["base_image_sha256"] == image["sha256"]
    assert instance["execution_ready"] is False
    assert instance["network_mode"] == "configured_isolated"
    assert manifest["instance_id"] == "task-1"
    assert report["image_manifest_present"] is True
    assert report["instance_manifest_present"] is True
    assert report["base_image_present"] is True
    assert report["overlay_present"] is False


def test_register_linux_boot_image_copies_kernel_and_records_agent_contract(tmp_path: Path) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    initrd = tmp_path / "initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    initrd.write_bytes(b"initrd bytes\n")
    state_root = tmp_path / "vm-state"

    payload = register_managed_vm_linux_boot_image(
        state_root=str(state_root),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        initrd_path=str(initrd),
        guest_agent_port=48123,
    )
    manifest = load_managed_vm_image_manifest(state_root=str(state_root), image_id="linux-image")
    instance = prepare_managed_vm_instance(
        state_root=str(state_root),
        image_id="linux-image",
        instance_id="task-linux",
    )

    assert payload["boot_mode"] == "linux_direct"
    assert payload["bootable"] is True
    assert payload["guest_agent_transport"] == "virtio-vsock"
    assert payload["guest_agent_port"] == 48123
    assert Path(payload["kernel_path"]).read_bytes() == b"kernel bytes\n"
    assert Path(payload["initrd_path"]).read_bytes() == b"initrd bytes\n"
    assert "conos.agent=vsock:48123" in payload["kernel_command_line"]
    assert manifest["kernel_sha256"] == payload["kernel_sha256"]
    assert instance["boot_mode"] == "linux_direct"
    assert instance["kernel_path"] == payload["kernel_path"]
    assert instance["guest_agent_port"] == 48123


def test_register_linux_boot_image_detects_partitioned_root_device(tmp_path: Path) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    raw = bytearray(3 * 1024 * 1024)
    raw[512:520] = b"EFI PART"
    raw[512 + 72 : 512 + 80] = (2).to_bytes(8, "little")
    raw[512 + 80 : 512 + 84] = (128).to_bytes(4, "little")
    raw[512 + 84 : 512 + 88] = (128).to_bytes(4, "little")
    entry_offset = 2 * 512
    raw[entry_offset : entry_offset + 16] = b"\x01" * 16
    partition_uuid = uuid.UUID("66e1ad91-06ce-497c-8b46-a705703e882f")
    raw[entry_offset + 16 : entry_offset + 32] = partition_uuid.bytes_le
    raw[entry_offset + 32 : entry_offset + 40] = (2048).to_bytes(8, "little")
    raw[entry_offset + 40 : entry_offset + 48] = (4095).to_bytes(8, "little")
    raw[2048 * 512 + 1024 + 56 : 2048 * 512 + 1024 + 58] = b"\x53\xef"
    source_disk.write_bytes(raw)
    kernel.write_bytes(b"kernel bytes\n")

    payload = register_managed_vm_linux_boot_image(
        state_root=str(tmp_path / "vm-state"),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        guest_agent_port=48123,
    )

    assert payload["detected_root_device"] == "/dev/vda1"
    assert payload["detected_root_partition_uuid"] == str(partition_uuid)
    assert payload["root_boot_spec"] == f"PARTUUID={partition_uuid}"
    assert payload["root_device"] == "/dev/vda1"
    assert f"root=PARTUUID={partition_uuid}" in payload["kernel_command_line"]
    assert "conos.root=/dev/vda1" in payload["kernel_command_line"]


def test_build_guest_initrd_bundle_records_autostart_contract(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    output = tmp_path / "conos-initrd.img"

    payload = build_managed_vm_guest_initrd_bundle(
        state_root=str(state_root),
        output_path=str(output),
        guest_agent_port=48123,
        root_device="/dev/vdb",
    )
    manifest = json.loads(Path(f"{output}.manifest.json").read_text(encoding="utf-8"))

    assert payload["status"] == "BUILT"
    assert output.exists()
    assert output.read_bytes()[:2] == b"\x1f\x8b"
    assert manifest["guest_agent_autostart_configured"] is True
    assert manifest["guest_agent_port"] == 48123
    assert manifest["root_device"] == "/dev/vdb"
    assert "conos/conos_guest_agent.py" in manifest["files"]
    assert "etc/systemd/system/conos-guest-agent.service" in manifest["files"]
    assert "init" in manifest["files"]
    assert manifest["runtime_verified"] is False


def test_build_guest_initrd_bundle_recompresses_zstd_base_initrd(tmp_path: Path) -> None:
    zstd = managed_vm_module._zstd_path()
    if not zstd:
        pytest.skip("zstd is not installed")
    base_raw = managed_vm_module._build_newc_archive([("base-marker.txt", b"base initramfs cpio bytes\n", 0o100644)])
    compressed = subprocess.run(
        [zstd, "-q", "-19", "-c"],
        input=base_raw,
        capture_output=True,
        check=True,
    ).stdout
    base_initrd = tmp_path / "base-initrd.img"
    output = tmp_path / "conos-initrd.img"
    base_initrd.write_bytes(compressed)

    payload = build_managed_vm_guest_initrd_bundle(
        base_initrd_path=str(base_initrd),
        output_path=str(output),
        guest_agent_port=48123,
        include_init_wrapper=False,
    )
    decompressed = subprocess.run(
        [zstd, "-q", "-d", "-c", str(output)],
        capture_output=True,
        check=True,
    ).stdout

    assert payload["status"] == "BUILT"
    assert payload["base_initrd_compression"] == "zstd"
    assert payload["base_initrd_trailer_stripped"] is True
    assert payload["initrd_merge_strategy"] == "decompress_append_cpio_recompress"
    assert payload["initramfs_integration"] == "initramfs_tools_local_bottom_hook"
    assert output.read_bytes().startswith(b"\x28\xb5\x2f\xfd")
    assert b"base-marker.txt" in decompressed
    assert b"base initramfs cpio bytes" in decompressed
    assert b"conos/conos_guest_agent.py" in decompressed
    assert b"scripts/init-top/conos-trace" in decompressed
    assert b"scripts/local-top/conos-trace" in decompressed
    assert b"scripts/init-top/ORDER" in decompressed
    assert b"scripts/local-top/ORDER" in decompressed
    assert b'/scripts/init-top/conos-trace "$@"' in decompressed
    assert b'/scripts/local-top/conos-trace "$@"' in decompressed
    assert b"conos-initramfs-init-top.txt" in decompressed
    assert b"conos-initramfs-local-top.txt" in decompressed
    assert b"mount -t virtiofs conos_host /conos-host" in decompressed
    assert b"scripts/local-bottom/conos-guest-agent" in decompressed
    assert b"scripts/local-bottom/ORDER" in decompressed
    assert b'/scripts/local-bottom/conos-guest-agent "$@"' in decompressed
    assert b"conf/modules" in decompressed
    assert b"virtio_pci" in decompressed
    assert b"virtio_blk" in decompressed
    assert b"virtio_console" in decompressed
    assert "init" not in payload["files"]
    assert "scripts/init-top/conos-trace" in payload["files"]
    assert "scripts/local-top/conos-trace" in payload["files"]
    assert "scripts/local-bottom/ORDER" in payload["files"]
    assert "conf/modules" in payload["files"]
    assert payload["initramfs_local_bottom_order_installed"] is True
    assert payload["initramfs_trace_hooks_installed"] is True
    assert payload["initramfs_trace_stages"] == ["init-top", "local-top", "local-bottom"]
    assert payload["initramfs_local_bottom_order_preserved"] is False
    assert payload["initramfs_modules_installed"] is True
    assert "virtio_blk" in payload["initramfs_modules_added"]
    assert decompressed.count(b"TRAILER!!!") == 1


def test_build_guest_initrd_bundle_replaces_duplicate_base_init(tmp_path: Path) -> None:
    base_raw = managed_vm_module._build_newc_archive(
        [
            ("init", b"#!/bin/sh\necho old init\n", 0o100755),
            ("base-marker.txt", b"base initramfs cpio bytes\n", 0o100644),
        ]
    )
    base_initrd = tmp_path / "base-initrd.img"
    output = tmp_path / "conos-initrd.img"
    base_initrd.write_bytes(gzip.compress(base_raw, compresslevel=9, mtime=0))

    payload = build_managed_vm_guest_initrd_bundle(
        base_initrd_path=str(base_initrd),
        output_path=str(output),
        guest_agent_port=48123,
        include_init_wrapper=True,
    )
    decompressed = gzip.decompress(output.read_bytes())

    assert "init" in payload["base_initrd_entries_replaced"]
    assert b"echo old init" not in decompressed
    assert b"CONOS_INIT_WRAPPER_START" in decompressed
    assert b"root_candidate_paths()" in decompressed
    assert b"PARTUUID=*)" in decompressed
    assert b"/dev/vd[a-z][0-9]*" in decompressed
    assert decompressed.count(b"TRAILER!!!") == 1


def test_build_cloud_init_seed_records_nocloud_contract(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    output = tmp_path / "cloud-init-seed.img"

    payload = build_managed_vm_cloud_init_seed(
        state_root=str(state_root),
        instance_id="task-1",
        output_path=str(output),
        guest_agent_port=48123,
        hostname="conos-test",
    )

    assert payload["status"] == "BUILT"
    assert payload["seed_format"] == "cloud-init_nocloud"
    assert payload["volume_label"] == "CIDATA"
    assert payload["seed_layout"] == "mbr_partitioned_vfat"
    assert payload["partition_table"] == "mbr"
    assert payload["partition_start_lba"] > 0
    assert payload["guest_agent_autostart_configured"] is True
    assert payload["guest_agent_start_target"] == "sysinit.target"
    assert payload["network_wait_online_override"] is True
    assert payload["guest_agent_port"] == 48123
    assert set(payload["files"]) == {"user-data", "meta-data", "network-config"}
    assert payload["cloud_init_marker_files"] == [
        "cloud-init-bootcmd.txt",
        "cloud-init-runcmd.txt",
        "cloud-init-agent-enable.txt",
        "cloud-init-virtiofs.err",
    ]
    assert output.exists()
    assert output.stat().st_size == 4 * 1024 * 1024
    seed_bytes = output.read_bytes()
    assert seed_bytes[510:512] == b"\x55\xaa"
    assert b"CIDATA" in seed_bytes[payload["partition_byte_offset"] : payload["partition_byte_offset"] + 1024]
    assert b"CONOS_CLOUD_INIT_BOOTCMD" in seed_bytes
    assert b"cloud-init-agent-enable.txt" in seed_bytes
    assert b"DefaultDependencies=no" in seed_bytes
    assert b"WantedBy=sysinit.target" in seed_bytes
    assert b"systemd-networkd-wait-online.service.d/conos-fast-boot.conf" in seed_bytes
    assert b"ExecStart=/bin/true" in seed_bytes
    assert Path(f"{output}.manifest.json").exists()


def test_replace_fat_file_in_image_updates_existing_cluster_chain(tmp_path: Path) -> None:
    output = tmp_path / "cloud-init-seed.img"
    payload = build_managed_vm_cloud_init_seed(
        state_root=str(tmp_path / "vm-state"),
        instance_id="seed-fat-replace",
        output_path=str(output),
        overwrite=True,
    )
    partition_offset = int(payload["partition_byte_offset"])
    replacement = b"instance-id: replaced\nlocal-hostname: replaced-host\n"

    result = managed_vm_module._replace_fat_file_in_image(
        output,
        partition_offset=partition_offset,
        file_path="meta-data",
        content=replacement,
    )

    assert result["status"] == "PATCHED"
    with output.open("rb") as handle:
        params = managed_vm_module._fat_parameters(handle, partition_offset)
        entry = managed_vm_module._find_fat_path_entry(handle, params, "meta-data")
        assert entry["file_size"] == len(replacement)
        handle.seek(managed_vm_module._fat_cluster_offset(params, entry["first_cluster"]))
        assert handle.read(len(replacement)) == replacement


def test_create_fat_file_in_image_adds_missing_single_cluster_file(tmp_path: Path) -> None:
    output = tmp_path / "cloud-init-seed.img"
    payload = build_managed_vm_cloud_init_seed(
        state_root=str(tmp_path / "vm-state"),
        instance_id="seed-fat-create",
        output_path=str(output),
        overwrite=True,
    )
    partition_offset = int(payload["partition_byte_offset"])
    content = b"set timeout=0\nboot\n"

    result = managed_vm_module._create_fat_file_in_image(
        output,
        partition_offset=partition_offset,
        file_path="grub.cfg",
        content=content,
    )

    assert result["status"] == "CREATED"
    with output.open("rb") as handle:
        params = managed_vm_module._fat_parameters(handle, partition_offset)
        entry = managed_vm_module._find_fat_path_entry(handle, params, "grub.cfg")
        assert entry["file_size"] == len(content)
        handle.seek(managed_vm_module._fat_cluster_offset(params, entry["first_cluster"]))
        assert handle.read(len(content)) == content


def test_register_cloud_init_image_marks_seed_boot_contract(tmp_path: Path) -> None:
    source_disk = tmp_path / "cloud.img"
    source_disk.write_bytes(b"efi cloud disk bytes\n")
    state_root = tmp_path / "vm-state"

    payload = register_managed_vm_cloud_init_image(
        state_root=str(state_root),
        image_id="cloud-image",
        source_disk_path=str(source_disk),
        guest_agent_port=48123,
    )

    assert payload["status"] == "REGISTERED"
    assert payload["boot_mode"] == "efi_disk"
    assert payload["bootable"] is True
    assert payload["cloud_init_seed_enabled"] is True
    assert payload["guest_agent_autostart_planned"] is True
    assert payload["guest_agent_autostart_configured"] is False
    assert payload["guest_agent_installation_mode"] == "cloud_init_nocloud_seed"
    assert payload["guest_agent_installation_status"] == "BLOCKED_CLOUD_INIT_UNAVAILABLE_IN_GUEST_IMAGE"
    assert payload["guest_agent_port"] == 48123
    assert payload["cloud_init_guest_capability"]["status"] == "UNAVAILABLE"
    assert payload["cloud_init_capability_warning"]
    assert "linux_direct" in payload["next_required_step"]
    assert Path(payload["disk_path"]).read_bytes() == b"efi cloud disk bytes\n"


def test_register_cloud_init_image_enables_efi_agent_initrd_when_boot_artifacts_found(tmp_path: Path) -> None:
    source_disk = tmp_path / "cloud.img"
    source_disk.write_bytes(b"vmlinuz-6.1.0-44-arm64\x00initrd.img-6.1.0-44-arm64\n")
    state_root = tmp_path / "vm-state"

    payload = register_managed_vm_cloud_init_image(
        state_root=str(state_root),
        image_id="cloud-image",
        source_disk_path=str(source_disk),
        guest_agent_port=48123,
    )

    assert payload["status"] == "REGISTERED"
    assert payload["guest_agent_autostart_configured"] is True
    assert payload["guest_agent_installation_mode"] == "efi_initrd_guest_agent_bundle"
    assert payload["guest_agent_installation_status"] == "EFI_INITRD_AGENT_INJECTION_CONFIGURED"
    assert payload["verified_execution_path"] == "efi_disk_observable_boot_agent_initrd"
    assert payload["efi_agent_initrd_injection_enabled"] is True
    assert payload["efi_agent_initrd_boot_artifacts"]["kernel_path"] == "/boot/vmlinuz-6.1.0-44-arm64"
    assert payload["efi_agent_initrd_boot_artifacts"]["initrd_path"] == "/boot/initrd.img-6.1.0-44-arm64"
    assert "guest-agent initrd" in payload["next_required_step"]


def test_register_linux_boot_image_detects_conos_guest_initrd_bundle(tmp_path: Path) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    initrd = tmp_path / "conos-initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    build_managed_vm_guest_initrd_bundle(
        output_path=str(initrd),
        guest_agent_port=48123,
        overwrite=True,
    )
    state_root = tmp_path / "vm-state"

    payload = register_managed_vm_linux_boot_image(
        state_root=str(state_root),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        initrd_path=str(initrd),
        guest_agent_port=48123,
    )
    instance = prepare_managed_vm_instance(
        state_root=str(state_root),
        image_id="linux-image",
        instance_id="task-linux",
    )

    assert payload["guest_agent_autostart_configured"] is True
    assert payload["guest_agent_installation_mode"] == "initrd_autostart_bundle"
    assert payload["guest_agent_installation_status"] == "INITRD_AUTOSTART_BUNDLE_CONFIGURED"
    assert payload["verified_execution_path"] == "linux_direct_initrd_guest_agent_bundle"
    assert payload["guest_initrd_bundle_manifest_present"] is True
    assert payload["guest_initrd_bundle_capability"]["status"] == "VERIFIED"
    assert "conos/conos_guest_agent.py" in payload["guest_agent_bundle_files"]
    assert "conos.root=/dev/vda" in payload["kernel_command_line"]
    assert instance["guest_agent_autostart_configured"] is True
    assert instance["guest_agent_installation_mode"] == "initrd_autostart_bundle"
    assert instance["verified_execution_path"] == "linux_direct_initrd_guest_agent_bundle"


def test_build_managed_vm_linux_base_image_blocks_without_boot_artifacts(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"

    payload = build_managed_vm_linux_base_image(
        state_root=str(state_root),
        image_id="linux-image",
    )

    assert payload["status"] == "BUILD_BLOCKED_MISSING_BOOT_ARTIFACTS"
    assert payload["requires_user_configured_vm"] is False
    assert payload["missing_fields"] == ["source_disk_path", "kernel_path"]
    assert payload["guest_agent_autostart_configured"] is False
    assert payload["no_host_fallback"] is True


def test_build_managed_vm_linux_base_image_registers_guest_agent_image(tmp_path: Path) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    base_initrd = tmp_path / "base-initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    state_root = tmp_path / "vm-state"

    payload = build_managed_vm_linux_base_image(
        state_root=str(state_root),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        base_initrd_path=str(base_initrd),
        guest_agent_port=48123,
        root_device="/dev/vdb",
    )
    manifest = load_managed_vm_image_manifest(state_root=str(state_root), image_id="linux-image")
    instance = prepare_managed_vm_instance(
        state_root=str(state_root),
        image_id="linux-image",
        instance_id="task-linux",
    )

    assert payload["status"] == "BUILT"
    assert payload["guest_agent_autostart_configured"] is True
    assert payload["guest_agent_installation_mode"] == "initrd_autostart_bundle"
    assert payload["guest_agent_installation_status"] == "INITRD_AUTOSTART_BUNDLE_CONFIGURED"
    assert payload["verified_execution_path"] == "linux_direct_initrd_guest_agent_bundle"
    assert payload["execution_ready"] is False
    assert manifest["built_by_conos_base_image_builder"] is True
    assert manifest["boot_mode"] == "linux_direct"
    assert manifest["guest_agent_autostart_configured"] is True
    assert manifest["verified_execution_path"] == "linux_direct_initrd_guest_agent_bundle"
    assert manifest["guest_agent_verified"] is False
    assert "conos.agent=vsock:48123" in manifest["kernel_command_line"]
    assert "conos.root=/dev/vdb" in manifest["kernel_command_line"]
    assert Path(manifest["initrd_path"]).read_bytes().startswith(b"base initrd\n")
    initrd_manifest = json.loads(Path(f"{manifest['initrd_path']}.manifest.json").read_text(encoding="utf-8"))
    assert initrd_manifest["initramfs_integration"] == "initramfs_tools_local_bottom_hook"
    assert initrd_manifest["include_init_wrapper"] is True
    assert "scripts/local-bottom/conos-guest-agent" in initrd_manifest["files"]
    assert "scripts/local-bottom/ORDER" in initrd_manifest["files"]
    assert "conf/modules" in initrd_manifest["files"]
    assert "conos/conos_guest_agent_launcher.sh" in initrd_manifest["files"]
    assert "init" in initrd_manifest["files"]
    assert "init-wrapper" in initrd_manifest["initramfs_trace_stages"]
    assert initrd_manifest["initramfs_local_bottom_order_installed"] is True
    assert initrd_manifest["initramfs_modules_installed"] is True
    assert instance["guest_agent_autostart_configured"] is True


def test_bundle_base_image_blocks_until_bootstrap_verified(tmp_path: Path) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    base_initrd = tmp_path / "base-initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    state_root = tmp_path / "vm-state"

    build_managed_vm_linux_base_image(
        state_root=str(state_root),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        base_initrd_path=str(base_initrd),
        guest_agent_port=48123,
    )

    payload = create_managed_vm_base_image_bundle(
        state_root=str(state_root),
        image_id="linux-image",
        output_dir=str(tmp_path / "bundle"),
    )

    assert payload["status"] == "BUNDLE_BLOCKED_UNVERIFIED_IMAGE"
    assert payload["release_eligible"] is False
    assert payload["required_verified_fields"] == ["boot_verified", "guest_agent_verified", "bootstrap_verified"]


def test_bundle_base_image_exports_relative_recipe_and_bootstraps(tmp_path: Path) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    base_initrd = tmp_path / "base-initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    state_root = tmp_path / "vm-state"

    built = build_managed_vm_linux_base_image(
        state_root=str(state_root),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        base_initrd_path=str(base_initrd),
        guest_agent_port=48123,
    )
    image_manifest_path = Path(built["image_manifest_path"])
    manifest = json.loads(image_manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "boot_verified": True,
            "guest_agent_verified": True,
            "bootstrap_verified": True,
            "bootstrap_verified_at": "2026-04-27T00:00:00+00:00",
        }
    )
    image_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bundle = create_managed_vm_base_image_bundle(
        state_root=str(state_root),
        image_id="linux-image",
        output_dir=str(tmp_path / "bundle"),
    )
    recipe_path = Path(bundle["recipe_path"])
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    resolved = resolve_managed_vm_artifact_recipe(
        state_root=str(tmp_path / "bundle-state"),
        recipe_path=str(recipe_path),
    )
    bootstrapped = bootstrap_managed_vm_image(
        state_root=str(tmp_path / "bootstrap-state"),
        image_id="linux-image-from-bundle",
        recipe_path=str(recipe_path),
        build_runner=False,
        start_instance=False,
    )
    bundled_manifest = load_managed_vm_image_manifest(
        state_root=str(tmp_path / "bootstrap-state"),
        image_id="linux-image-from-bundle",
    )

    assert bundle["status"] == "BUNDLED"
    assert bundle["release_eligible"] is True
    assert Path(bundle["manifest_path"]).exists()
    assert recipe["artifacts"]["source_disk"]["path"] == "artifacts/disk.img"
    assert recipe["artifacts"]["initrd"]["path"] == "artifacts/initrd.img"
    assert (recipe_path.parent / "artifacts" / "initrd.img.manifest.json").exists()
    assert resolved["status"] == "RESOLVED"
    assert resolved["resolved_paths"]["initrd_path"]
    assert Path(f"{resolved['resolved_paths']['initrd_path']}.manifest.json").exists()
    assert bootstrapped["status"] == "BOOTSTRAP_IMAGE_BUILT"
    assert bootstrapped["build_report"]["operation"] == "register_bundled_linux_image"
    assert bundled_manifest["guest_initrd_bundle_capability"]["status"] == "VERIFIED"
    assert bundled_manifest["verified_execution_path"] == "linux_direct_initrd_guest_agent_bundle"


def test_install_base_image_bundle_imports_verified_artifacts(tmp_path: Path) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    base_initrd = tmp_path / "base-initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    source_state = tmp_path / "source-state"

    built = build_managed_vm_linux_base_image(
        state_root=str(source_state),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        base_initrd_path=str(base_initrd),
        guest_agent_port=48123,
    )
    image_manifest_path = Path(built["image_manifest_path"])
    manifest = json.loads(image_manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "boot_verified": True,
            "guest_agent_verified": True,
            "bootstrap_verified": True,
            "bootstrap_verified_at": "2026-04-27T00:00:00+00:00",
        }
    )
    image_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle = create_managed_vm_base_image_bundle(
        state_root=str(source_state),
        image_id="linux-image",
        output_dir=str(tmp_path / "bundle"),
    )

    installed = install_managed_vm_base_image_bundle(
        state_root=str(tmp_path / "installed-state"),
        bundle_dir=str(bundle["output_dir"]),
        image_id="installed-image",
    )
    installed_manifest = load_managed_vm_image_manifest(
        state_root=str(tmp_path / "installed-state"),
        image_id="installed-image",
    )
    instance = prepare_managed_vm_instance(
        state_root=str(tmp_path / "installed-state"),
        image_id="installed-image",
        instance_id="task-1",
    )

    assert installed["status"] == "INSTALLED"
    assert installed["release_eligible"] is True
    assert Path(installed["image_manifest_path"]).exists()
    assert installed_manifest["installed_from_bundle"] is True
    assert installed_manifest["image_id"] == "installed-image"
    assert installed_manifest["sha256"] == bundle["artifacts"]["source_disk"]["sha256"]
    assert installed_manifest["kernel_sha256"] == bundle["artifacts"]["kernel"]["sha256"]
    assert installed_manifest["initrd_sha256"] == bundle["artifacts"]["initrd"]["sha256"]
    assert Path(f"{installed_manifest['initrd_path']}.manifest.json").exists()
    assert installed_manifest["guest_initrd_bundle_capability"]["status"] == "VERIFIED"
    assert installed_manifest["guest_agent_autostart_configured"] is True
    assert installed_manifest["verified_execution_path"] == "linux_direct_initrd_guest_agent_bundle"
    assert instance["guest_agent_autostart_configured"] is True


def test_install_base_image_bundle_blocks_unverified_by_default(tmp_path: Path) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    base_initrd = tmp_path / "base-initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    source_state = tmp_path / "source-state"

    build_managed_vm_linux_base_image(
        state_root=str(source_state),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        base_initrd_path=str(base_initrd),
        guest_agent_port=48123,
    )
    bundle = create_managed_vm_base_image_bundle(
        state_root=str(source_state),
        image_id="linux-image",
        output_dir=str(tmp_path / "bundle"),
        allow_unverified=True,
    )

    blocked = install_managed_vm_base_image_bundle(
        state_root=str(tmp_path / "blocked-state"),
        bundle_dir=str(bundle["output_dir"]),
    )
    installed = install_managed_vm_base_image_bundle(
        state_root=str(tmp_path / "dev-state"),
        bundle_dir=str(bundle["output_dir"]),
        allow_unverified=True,
    )

    assert bundle["status"] == "BUNDLED_UNVERIFIED"
    assert blocked["status"] == "INSTALL_BLOCKED_UNVERIFIED_BUNDLE"
    assert blocked["release_eligible"] is False
    assert installed["status"] == "INSTALLED"
    assert installed["release_eligible"] is False


def test_install_base_image_bundle_blocks_digest_mismatch(tmp_path: Path) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    base_initrd = tmp_path / "base-initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    source_state = tmp_path / "source-state"

    built = build_managed_vm_linux_base_image(
        state_root=str(source_state),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        base_initrd_path=str(base_initrd),
        guest_agent_port=48123,
    )
    image_manifest_path = Path(built["image_manifest_path"])
    manifest = json.loads(image_manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "boot_verified": True,
            "guest_agent_verified": True,
            "bootstrap_verified": True,
            "bootstrap_verified_at": "2026-04-27T00:00:00+00:00",
        }
    )
    image_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle = create_managed_vm_base_image_bundle(
        state_root=str(source_state),
        image_id="linux-image",
        output_dir=str(tmp_path / "bundle"),
    )
    (Path(bundle["output_dir"]) / "artifacts" / "vmlinuz").write_bytes(b"tampered kernel\n")

    installed = install_managed_vm_base_image_bundle(
        state_root=str(tmp_path / "installed-state"),
        bundle_dir=str(bundle["output_dir"]),
    )

    assert installed["status"] == "INSTALL_BLOCKED_DIGEST_MISMATCH"
    assert installed["digest_mismatches"][0]["artifact"] == "kernel"


def test_resolve_artifact_recipe_caches_file_artifacts(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "inputs"
    artifacts_root.mkdir()
    source_disk = artifacts_root / "rootfs.img"
    kernel = artifacts_root / "vmlinuz"
    base_initrd = artifacts_root / "initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema_version": "conos.managed_vm_artifact_recipe/v1",
                "image_id": "linux-recipe",
                "guest_agent_port": 48123,
                "root_device": "/dev/vdb",
                "artifacts": {
                    "source_disk": {
                        "url": source_disk.as_uri(),
                        "sha256": hashlib.sha256(source_disk.read_bytes()).hexdigest(),
                    },
                    "kernel": {
                        "url": kernel.as_uri(),
                        "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
                    },
                    "base_initrd": {
                        "url": base_initrd.as_uri(),
                        "sha256": hashlib.sha256(base_initrd.read_bytes()).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = resolve_managed_vm_artifact_recipe(
        state_root=str(tmp_path / "vm-state"),
        recipe_path=str(recipe),
    )

    assert payload["status"] == "RESOLVED"
    assert payload["guest_agent_port"] == 48123
    assert payload["root_device"] == "/dev/vdb"
    assert Path(payload["resolved_paths"]["source_disk_path"]).read_bytes() == b"disk bytes\n"
    assert Path(payload["resolved_paths"]["kernel_path"]).read_bytes() == b"kernel bytes\n"
    assert Path(payload["resolved_paths"]["base_initrd_path"]).read_bytes() == b"base initrd\n"
    assert payload["artifact_results"]["source_disk"]["status"] == "CACHED"


def test_resolve_artifact_recipe_rejects_sha_mismatch(tmp_path: Path) -> None:
    source_disk = tmp_path / "rootfs.img"
    kernel = tmp_path / "vmlinuz"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema_version": "conos.managed_vm_artifact_recipe/v1",
                "artifacts": {
                    "source_disk": {"url": source_disk.as_uri(), "sha256": "0" * 64},
                    "kernel": {"url": kernel.as_uri(), "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest()},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = resolve_managed_vm_artifact_recipe(
        state_root=str(tmp_path / "vm-state"),
        recipe_path=str(recipe),
    )

    assert payload["status"] == "ARTIFACT_RESOLUTION_FAILED"
    assert payload["failed_artifacts"]["source_disk"]["status"] == "DIGEST_MISMATCH"


def test_resolve_artifact_recipe_accepts_sha512_digest(tmp_path: Path) -> None:
    source_disk = tmp_path / "rootfs.img"
    kernel = tmp_path / "vmlinuz"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema_version": "conos.managed_vm_artifact_recipe/v1",
                "artifacts": {
                    "source_disk": {"url": source_disk.as_uri(), "sha512": hashlib.sha512(source_disk.read_bytes()).hexdigest()},
                    "kernel": {"url": kernel.as_uri(), "sha512": hashlib.sha512(kernel.read_bytes()).hexdigest()},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = resolve_managed_vm_artifact_recipe(
        state_root=str(tmp_path / "vm-state"),
        recipe_path=str(recipe),
    )

    assert payload["status"] == "RESOLVED"
    assert payload["artifact_results"]["source_disk"]["digest_algorithm"] == "sha512"
    assert Path(payload["resolved_paths"]["source_disk_path"]).read_bytes() == b"disk bytes\n"


def test_resolve_efi_cloud_init_recipe_requires_only_source_disk(tmp_path: Path) -> None:
    source_disk = tmp_path / "cloud.img"
    source_disk.write_bytes(b"efi cloud disk bytes\n")
    recipe = tmp_path / "efi-recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema_version": "conos.managed_vm_artifact_recipe/v1",
                "boot_mode": "efi_disk",
                "cloud_init_seed_enabled": True,
                "guest_agent_installation_mode": "cloud_init_nocloud_seed",
                "artifacts": {
                    "source_disk": {
                        "url": source_disk.as_uri(),
                        "sha256": hashlib.sha256(source_disk.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = resolve_managed_vm_artifact_recipe(
        state_root=str(tmp_path / "vm-state"),
        recipe_path=str(recipe),
    )

    assert payload["status"] == "RESOLVED"
    assert payload["boot_mode"] == "efi_disk"
    assert payload["cloud_init_seed_enabled"] is True
    assert payload["resolved_paths"]["kernel_path"] == ""


def test_builtin_vm_recipe_report_lists_digest_pinned_official_candidate() -> None:
    payload = managed_vm_recipe_report()

    assert payload["status"] == "AVAILABLE"
    assert payload["default_recipe_id"] == "debian-genericcloud-arm64"
    recipe = next(recipe for recipe in payload["recipes"] if recipe["id"] == "debian-genericcloud-arm64")
    assert recipe["status"] == "READY"
    assert recipe["source_page"].startswith("https://cloud-image-finder.debian.net/")


def test_resolve_builtin_vm_recipe_refuses_download_when_disabled(tmp_path: Path) -> None:
    payload = resolve_managed_vm_artifact_recipe(
        state_root=str(tmp_path / "vm-state"),
        recipe_path="builtin:debian-nocloud-arm64",
        allow_download=False,
    )

    assert payload["status"] == "ARTIFACT_RESOLUTION_FAILED"
    assert payload["recipe_status"] == "READY"
    assert payload["builtin_recipe"]["recipe_id"] == "debian-nocloud-arm64"
    assert payload["failed_artifacts"]["source_disk"]["status"] == "DOWNLOAD_DISABLED"
    assert payload["allow_download"] is False


def test_resolve_builtin_vm_recipe_uses_digest_pinned_source_disk(tmp_path: Path, monkeypatch) -> None:
    cached_disk = tmp_path / "cached.raw"
    cached_disk.write_bytes(b"cached cloud image\n")
    captured: dict[str, object] = {}

    def fake_copy_or_download_vm_artifact(**kwargs):
        captured.update(kwargs)
        spec = kwargs["spec"]
        return {
            "name": kwargs["name"],
            "status": "CACHED",
            "path": str(cached_disk),
            "digest_algorithm": "sha512",
            "expected_digest": spec["sha512"],
            "source": spec["url"],
            "from_cache": False,
        }

    monkeypatch.setattr(managed_vm_module, "_copy_or_download_vm_artifact", fake_copy_or_download_vm_artifact)

    payload = resolve_managed_vm_artifact_recipe(
        state_root=str(tmp_path / "vm-state"),
        recipe_path="builtin:debian-nocloud-arm64",
    )

    assert payload["status"] == "RESOLVED"
    assert payload["boot_mode"] == "efi_disk"
    assert payload["cloud_init_seed_enabled"] is True
    assert payload["resolved_paths"]["source_disk_path"] == str(cached_disk)
    assert payload["resolved_paths"]["kernel_path"] == ""
    assert captured["allow_download"] is True
    assert captured["name"] == "source_disk"
    assert captured["spec"]["url"].endswith(".raw")
    assert captured["spec"]["sha512"]


def test_bootstrap_image_blocks_on_builtin_recipe_resolution_failure_before_build(tmp_path: Path) -> None:
    payload = bootstrap_managed_vm_image(
        state_root=str(tmp_path / "vm-state"),
        recipe_path="builtin:debian-nocloud-arm64",
        allow_artifact_download=False,
        build_runner=False,
        start_instance=False,
    )

    assert payload["status"] == "BOOTSTRAP_BLOCKED_ARTIFACT_RESOLUTION_FAILED"
    assert payload["recipe_report"]["status"] == "ARTIFACT_RESOLUTION_FAILED"
    assert payload["failed_artifacts"]["source_disk"]["status"] == "DOWNLOAD_DISABLED"
    assert payload["build_report"] == {}
    assert payload["guest_agent_ready"] is False
    assert payload["execution_ready"] is False


def test_resolve_unknown_builtin_vm_recipe_reports_unknown(tmp_path: Path) -> None:
    payload = resolve_managed_vm_artifact_recipe(
        state_root=str(tmp_path / "vm-state"),
        recipe_path="builtin:missing-recipe",
    )

    assert payload["status"] == "RECIPE_UNKNOWN"
    assert payload["builtin_recipe"]["recipe_id"] == "missing-recipe"


def test_pin_builtin_cloud_init_recipe_with_local_source_disk(tmp_path: Path) -> None:
    source_disk = tmp_path / "cloud.img"
    source_disk.write_bytes(b"efi cloud disk bytes\n")
    state_root = tmp_path / "vm-state"

    pinned = create_managed_vm_pinned_artifact_recipe(
        state_root=str(state_root),
        base_recipe_path="builtin:debian-nocloud-arm64",
        source_disk=str(source_disk),
        recipe_id="debian-nocloud-arm64-local",
    )
    recipe_path = Path(pinned["recipe_path"])
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    resolved = resolve_managed_vm_artifact_recipe(
        state_root=str(state_root),
        recipe_path=str(recipe_path),
    )

    assert pinned["status"] == "PINNED"
    assert recipe["status"] == "READY"
    assert recipe["boot_mode"] == "efi_disk"
    assert recipe["cloud_init_seed_enabled"] is True
    assert recipe["artifacts"]["source_disk"]["sha256"] == hashlib.sha256(source_disk.read_bytes()).hexdigest()
    assert resolved["status"] == "RESOLVED"
    assert resolved["boot_mode"] == "efi_disk"
    assert resolved["cloud_init_seed_enabled"] is True
    assert Path(resolved["resolved_paths"]["source_disk_path"]).read_bytes() == b"efi cloud disk bytes\n"


def test_pin_artifact_recipe_blocks_remote_source_without_digest(tmp_path: Path) -> None:
    payload = create_managed_vm_pinned_artifact_recipe(
        state_root=str(tmp_path / "vm-state"),
        base_recipe_path="builtin:debian-nocloud-arm64",
        source_disk="https://example.invalid/cloud.img",
    )

    assert payload["status"] == "PIN_BLOCKED"
    assert payload["failed_artifacts"]["source_disk"]["status"] == "MISSING_DIGEST"


def test_pin_artifact_recipe_rejects_mismatched_local_digest(tmp_path: Path) -> None:
    source_disk = tmp_path / "cloud.img"
    source_disk.write_bytes(b"efi cloud disk bytes\n")

    payload = create_managed_vm_pinned_artifact_recipe(
        state_root=str(tmp_path / "vm-state"),
        base_recipe_path="builtin:debian-nocloud-arm64",
        source_disk=str(source_disk),
        source_disk_sha256="0" * 64,
    )

    assert payload["status"] == "PIN_BLOCKED"
    assert payload["failed_artifacts"]["source_disk"]["status"] == "DIGEST_MISMATCH"


def test_bootstrap_image_uses_artifact_recipe_without_direct_paths(tmp_path: Path) -> None:
    source_disk = tmp_path / "rootfs.img"
    kernel = tmp_path / "vmlinuz"
    base_initrd = tmp_path / "initrd.img"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema_version": "conos.managed_vm_artifact_recipe/v1",
                "guest_agent_port": 48123,
                "root_device": "/dev/vdb",
                "artifacts": {
                    "source_disk": {
                        "url": source_disk.as_uri(),
                        "sha256": hashlib.sha256(source_disk.read_bytes()).hexdigest(),
                    },
                    "kernel": {
                        "url": kernel.as_uri(),
                        "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
                    },
                    "base_initrd": {
                        "url": base_initrd.as_uri(),
                        "sha256": hashlib.sha256(base_initrd.read_bytes()).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    state_root = tmp_path / "vm-state"

    payload = bootstrap_managed_vm_image(
        state_root=str(state_root),
        image_id="linux-image",
        recipe_path=str(recipe),
        build_runner=False,
        start_instance=False,
    )
    manifest = load_managed_vm_image_manifest(state_root=str(state_root), image_id="linux-image")

    assert payload["status"] == "BOOTSTRAP_IMAGE_BUILT"
    assert payload["recipe_report"]["status"] == "RESOLVED"
    assert manifest["guest_agent_port"] == 48123
    assert "conos.root=/dev/vdb" in manifest["kernel_command_line"]


def test_bootstrap_image_registers_efi_cloud_init_recipe_without_kernel(tmp_path: Path) -> None:
    source_disk = tmp_path / "cloud.img"
    source_disk.write_bytes(b"efi cloud disk bytes\n")
    recipe = tmp_path / "efi-recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema_version": "conos.managed_vm_artifact_recipe/v1",
                "boot_mode": "efi_disk",
                "cloud_init_seed_enabled": True,
                "guest_agent_installation_mode": "cloud_init_nocloud_seed",
                "guest_agent_port": 48123,
                "artifacts": {
                    "source_disk": {
                        "url": source_disk.as_uri(),
                        "sha256": hashlib.sha256(source_disk.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    state_root = tmp_path / "vm-state"

    payload = bootstrap_managed_vm_image(
        state_root=str(state_root),
        image_id="cloud-image",
        recipe_path=str(recipe),
        build_runner=False,
        start_instance=False,
    )
    manifest = load_managed_vm_image_manifest(state_root=str(state_root), image_id="cloud-image")

    assert payload["status"] == "BOOTSTRAP_IMAGE_BUILT"
    assert payload["recipe_report"]["status"] == "RESOLVED"
    assert payload["build_report"]["operation"] == "register_cloud_init_image"
    assert manifest["boot_mode"] == "efi_disk"
    assert manifest["cloud_init_seed_enabled"] is True
    assert manifest["guest_agent_installation_mode"] == "cloud_init_nocloud_seed"
    assert manifest["guest_agent_installation_status"] == "BLOCKED_CLOUD_INIT_UNAVAILABLE_IN_GUEST_IMAGE"
    assert manifest["guest_agent_autostart_configured"] is False
    assert manifest["guest_agent_autostart_planned"] is True
    assert manifest["guest_agent_port"] == 48123


def test_install_default_managed_vm_image_uses_builtin_recipe(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_bootstrap_managed_vm_image(**kwargs):
        captured.update(kwargs)
        return {
            "status": "BOOTSTRAP_IMAGE_BUILT",
            "state_root": kwargs["state_root"],
            "image_id": kwargs["image_id"],
            "instance_id": kwargs["instance_id"],
            "runner_available": True,
            "guest_agent_ready": False,
            "execution_ready": False,
            "verified": False,
            "reason": "image built and instance prepared; start verification was not requested",
            "next_required_step": "",
        }

    monkeypatch.setattr(managed_vm_module, "bootstrap_managed_vm_image", fake_bootstrap_managed_vm_image)

    payload = install_default_managed_vm_image(
        state_root=str(tmp_path / "vm-state"),
        start_instance=False,
        allow_artifact_download=False,
    )

    assert payload["operation"] == "install_default_image"
    assert payload["status"] == "BOOTSTRAP_IMAGE_BUILT"
    assert payload["recipe_path"] == "builtin:debian-genericcloud-arm64"
    assert captured["recipe_path"] == "builtin:debian-genericcloud-arm64"
    assert captured["image_id"] == "conos-base"
    assert captured["allow_artifact_download"] is False
    assert captured["start_instance"] is False


def test_bootstrap_image_blocks_without_boot_artifacts(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"

    payload = bootstrap_managed_vm_image(
        state_root=str(state_root),
        image_id="linux-image",
        build_runner=False,
    )

    assert payload["status"] == "BOOTSTRAP_BLOCKED_MISSING_BOOT_ARTIFACTS"
    assert payload["missing_fields"] == ["source_disk_path", "kernel_path"]
    assert payload["requires_user_configured_vm"] is False
    assert payload["verified"] is False
    assert payload["no_host_fallback"] is True


def test_bootstrap_image_verifies_guest_agent_exec_and_marks_image(tmp_path: Path, monkeypatch) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    base_initrd = tmp_path / "base-initrd.img"
    runner = tmp_path / "conos-vz-runner"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    state_root = tmp_path / "vm-state"
    calls: list[str] = []

    def fake_start(**kwargs):
        calls.append("start")
        assert kwargs["runner_path"] == str(runner)
        return {
            "status": "STARTED",
            "virtual_machine_started": True,
            "guest_agent_ready": False,
            "execution_ready": False,
        }

    def fake_wait(**kwargs):
        calls.append("wait")
        return {
            "status": "GUEST_AGENT_READY",
            "ready": True,
            "gate": {
                "ready": True,
                "guest_agent_ready": True,
                "execution_ready": True,
            },
        }

    def fake_exec(command, **kwargs):
        calls.append("exec")
        assert command == ["echo", "ok"]
        return {
            "status": "COMPLETED",
            "returncode": 0,
            "stdout": "ok\n",
            "stderr": "",
        }

    def fake_stop(**kwargs):
        calls.append("stop")
        return {"status": "STOPPED"}

    monkeypatch.setattr(managed_vm_module, "start_managed_vm_instance", fake_start)
    monkeypatch.setattr(managed_vm_module, "wait_managed_vm_guest_agent_ready", fake_wait)
    monkeypatch.setattr(managed_vm_module, "run_managed_vm_agent_command", fake_exec)
    monkeypatch.setattr(managed_vm_module, "stop_managed_vm_instance", fake_stop)

    payload = bootstrap_managed_vm_image(
        state_root=str(state_root),
        image_id="linux-image",
        instance_id="bootstrap-smoke",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        base_initrd_path=str(base_initrd),
        runner_path=str(runner),
        build_runner=False,
        guest_agent_port=48123,
        root_device="/dev/vdb",
    )
    manifest = load_managed_vm_image_manifest(state_root=str(state_root), image_id="linux-image")

    assert payload["status"] == "BOOTSTRAP_VERIFIED"
    assert payload["verified"] is True
    assert payload["guest_agent_ready"] is True
    assert payload["execution_ready"] is True
    assert calls == ["start", "wait", "exec", "stop"]
    assert manifest["bootstrap_verified"] is True
    assert manifest["boot_verified"] is True
    assert manifest["guest_agent_verified"] is True
    assert manifest["bootstrap_agent_exec_stdout"] == "ok\n"


def test_bootstrap_image_recommends_boot_path_after_linux_direct_no_signal(tmp_path: Path, monkeypatch) -> None:
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    base_initrd = tmp_path / "base-initrd.img"
    runner = tmp_path / "conos-vz-runner"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    base_initrd.write_bytes(b"base initrd\n")
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    state_root = tmp_path / "vm-state"
    calls: list[str] = []

    def fake_start(**kwargs):
        calls.append("start")
        return {
            "status": "STARTED",
            "virtual_machine_started": True,
            "guest_agent_ready": False,
            "execution_ready": False,
        }

    def fake_wait(**kwargs):
        calls.append("wait")
        return {
            "status": "GUEST_AGENT_TIMEOUT",
            "ready": False,
            "reason": "guest agent did not become ready before timeout",
            "gate": {
                "ready": False,
                "guest_boot_diagnostic": {
                    "schema_version": "conos.managed_vm_provider/v1",
                    "diagnosis_status": "LINUX_DIRECT_NO_EARLY_GUEST_SIGNAL",
                    "blocked_stage": "guest_boot_or_initramfs",
                    "reason": "VM is running, but no guest console output or initramfs trace marker has been observed",
                    "observed_signals": {"boot_mode": "linux_direct"},
                    "recommended_next_steps": [
                        "verify the initrd contains the Con OS init wrapper and only one /init entry"
                    ],
                },
            },
        }

    def fake_stop(**kwargs):
        calls.append("stop")
        return {"status": "STOPPED"}

    monkeypatch.setattr(managed_vm_module, "start_managed_vm_instance", fake_start)
    monkeypatch.setattr(managed_vm_module, "wait_managed_vm_guest_agent_ready", fake_wait)
    monkeypatch.setattr(managed_vm_module, "stop_managed_vm_instance", fake_stop)

    payload = bootstrap_managed_vm_image(
        state_root=str(state_root),
        image_id="linux-image",
        instance_id="bootstrap-smoke",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        base_initrd_path=str(base_initrd),
        runner_path=str(runner),
        build_runner=False,
    )

    assert payload["status"] == "BOOTSTRAP_GUEST_BOOT_UNOBSERVABLE"
    assert payload["blocker_type"] == "linux_direct_no_early_guest_signal"
    assert payload["verified"] is False
    assert payload["boot_path_recommendation"]["status"] == "BOOT_PATH_NO_EARLY_GUEST_SIGNAL"
    assert payload["boot_path_recommendation"]["retry_same_path"] is False
    assert payload["boot_path_recommendation"]["recommended_boot_mode"] == "efi_disk_cloud_init_or_known_good_linux_direct"
    assert payload["next_required_step"].startswith("prefer the EFI/cloud-init")
    assert calls == ["start", "wait", "stop"]


def test_boot_managed_vm_instance_calls_helper_and_updates_manifest(tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    prepare_managed_vm_instance(
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="task-1",
    )
    overlay = state_root / "instances" / "task-1" / "overlay.img"
    captured = {}

    class Completed:
        returncode = 0
        stdout = json.dumps(
            {
                "status": "BOOT_CONTRACT_READY_EXEC_UNAVAILABLE",
                "reason": "guest agent execution is not implemented",
                "overlay_path": str(overlay),
                "overlay_present": True,
                "virtual_machine_started": False,
                "guest_agent_ready": False,
                "execution_ready": False,
            }
        )
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        overlay.write_bytes(b"overlay\n")
        return Completed()

    monkeypatch.setattr(managed_vm_module.subprocess, "run", fake_run)

    result = boot_managed_vm_instance(
        state_root=str(state_root),
        helper_path=str(helper),
        image_id="unit-image",
        instance_id="task-1",
        network_mode="configured_isolated",
    )
    manifest = load_managed_vm_instance_manifest(state_root=str(state_root), instance_id="task-1")

    assert result["status"] == "BOOT_CONTRACT_READY_EXEC_UNAVAILABLE"
    assert result["execution_ready"] is False
    assert result["guest_agent_ready"] is False
    assert result["overlay_present"] is True
    assert captured["cmd"][:4] == [str(helper), "boot", "--state-root", str(state_root)]
    assert "--network-mode" in captured["cmd"]
    assert "configured_isolated" in captured["cmd"]
    assert manifest["boot_contract_status"] == "BOOT_CONTRACT_READY_EXEC_UNAVAILABLE"
    assert manifest["execution_ready"] is False
    assert manifest["overlay_present"] is True


def test_boot_managed_vm_instance_refuses_missing_helper_or_image(tmp_path: Path) -> None:
    missing_helper = boot_managed_vm_instance(
        state_root=str(tmp_path / "vm-state"),
        helper_path="",
        image_id="unit-image",
        instance_id="task-1",
    )

    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    missing_image = boot_managed_vm_instance(
        state_root=str(tmp_path / "vm-state"),
        helper_path=str(helper),
        image_id="unit-image",
        instance_id="task-1",
    )

    assert missing_helper["status"] == "UNAVAILABLE"
    assert missing_helper["reason"] == "managed VM helper was not found"
    assert missing_image["status"] == "UNAVAILABLE"
    assert missing_image["reason"] == "managed VM base image is not registered"


def test_conos_vm_register_image_prepare_instance_and_reports(capsys, tmp_path: Path) -> None:
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"disk bytes\n")
    state_root = tmp_path / "vm-state"

    assert (
        conos_cli.main(
            [
                "vm",
                "register-image",
                "--state-root",
                str(state_root),
                "--image-id",
                "cli-image",
                "--source-disk",
                str(source_disk),
            ]
        )
        == 0
    )
    image_payload = json.loads(capsys.readouterr().out)
    assert image_payload["status"] == "REGISTERED"

    assert (
        conos_cli.main(
            [
                "vm",
                "prepare-instance",
                "--state-root",
                str(state_root),
                "--image-id",
                "cli-image",
                "--instance-id",
                "cli-task",
            ]
        )
        == 0
    )
    instance_payload = json.loads(capsys.readouterr().out)
    assert instance_payload["status"] == "PREPARED"

    assert conos_cli.main(["vm", "image-report", "--state-root", str(state_root), "--image-id", "cli-image"]) == 0
    image_report = json.loads(capsys.readouterr().out)
    assert image_report["image_id"] == "cli-image"

    assert conos_cli.main(["vm", "instance-report", "--state-root", str(state_root), "--instance-id", "cli-task"]) == 0
    instance_report = json.loads(capsys.readouterr().out)
    assert instance_report["instance_id"] == "cli-task"


def test_conos_vm_create_blank_image_cli(capsys, tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"

    assert (
        conos_cli.main(
            [
                "vm",
                "create-blank-image",
                "--state-root",
                str(state_root),
                "--image-id",
                "cli-blank",
                "--size-mb",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "BLANK_CREATED"
    assert payload["image_id"] == "cli-blank"
    assert payload["execution_ready"] is False


def test_conos_vm_install_default_image_cli(capsys, tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_install_default_managed_vm_image(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": "conos.managed_vm_provider/v1",
            "operation": "install_default_image",
            "status": "BOOTSTRAP_IMAGE_BUILT",
            "state_root": kwargs["state_root"],
            "image_id": kwargs["image_id"] or "conos-base",
            "instance_id": kwargs["instance_id"],
            "recipe_path": kwargs["recipe_path"],
            "allow_artifact_download": kwargs["allow_artifact_download"],
            "verified": False,
            "no_host_fallback": True,
        }

    monkeypatch.setattr(managed_vm_module, "install_default_managed_vm_image", fake_install_default_managed_vm_image)

    assert (
        conos_cli.main(
            [
                "vm",
                "install-default-image",
                "--state-root",
                str(tmp_path / "vm-state"),
                "--no-allow-artifact-download",
                "--no-start-instance",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["operation"] == "install_default_image"
    assert payload["status"] == "BOOTSTRAP_IMAGE_BUILT"
    assert captured["recipe_path"] == "builtin:debian-genericcloud-arm64"
    assert captured["allow_artifact_download"] is False
    assert captured["start_instance"] is False


def test_conos_vm_boot_instance_cli(capsys, tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="cli-image",
        source_disk_path=str(source_disk),
    )

    class Completed:
        returncode = 0
        stdout = json.dumps(
            {
                "status": "BOOT_CONTRACT_READY_EXEC_UNAVAILABLE",
                "reason": "guest agent unavailable",
                "overlay_path": str(state_root / "instances" / "cli-task" / "overlay.img"),
                "overlay_present": True,
                "virtual_machine_started": False,
                "guest_agent_ready": False,
                "execution_ready": False,
            }
        )
        stderr = ""

    monkeypatch.setattr(managed_vm_module.subprocess, "run", lambda cmd, **kwargs: Completed())

    assert (
        conos_cli.main(
            [
                "vm",
                "boot-instance",
                "--state-root",
                str(state_root),
                "--helper-path",
                str(helper),
                "--image-id",
                "cli-image",
                "--instance-id",
                "cli-task",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "BOOT_CONTRACT_READY_EXEC_UNAVAILABLE"
    assert payload["instance_prepared_by_boot"] is True
    assert payload["execution_ready"] is False


def test_start_status_stop_managed_vm_instance_lifecycle(tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    calls = []

    class Completed:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        helper_command = cmd[1]
        if helper_command == "start":
            return Completed(
                json.dumps(
                    {
                        "status": "START_BLOCKED_GUEST_AGENT_OR_BOOT_IMPL_MISSING",
                        "lifecycle_state": "start_blocked",
                        "reason": "guest agent missing",
                        "overlay_path": str(state_root / "instances" / "task-1" / "overlay.img"),
                        "overlay_present": True,
                        "process_pid": "",
                        "virtual_machine_started": False,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                    }
                )
            )
        if helper_command == "runtime-status":
            return Completed(
                json.dumps(
                    {
                        "status": "RUNTIME_MANIFEST_PRESENT",
                        "lifecycle_state": "unknown_from_helper",
                        "runtime_manifest_present": True,
                        "process_pid": "",
                        "virtual_machine_started": False,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                    }
                )
            )
        if helper_command == "stop":
            return Completed(
                json.dumps(
                    {
                        "status": "STOPPED",
                        "lifecycle_state": "stopped",
                        "reason": "no running process",
                        "process_pid": "",
                        "virtual_machine_started": False,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                    }
                )
            )
        raise AssertionError(f"unexpected helper command: {helper_command}")

    monkeypatch.setattr(managed_vm_module.subprocess, "run", fake_run)

    start = start_managed_vm_instance(
        state_root=str(state_root),
        helper_path=str(helper),
        image_id="unit-image",
        instance_id="task-1",
    )
    runtime = load_managed_vm_runtime_manifest(state_root=str(state_root), instance_id="task-1")
    status = managed_vm_runtime_status(
        state_root=str(state_root),
        helper_path=str(helper),
        image_id="unit-image",
        instance_id="task-1",
    )
    stop = stop_managed_vm_instance(
        state_root=str(state_root),
        helper_path=str(helper),
        image_id="unit-image",
        instance_id="task-1",
    )
    stopped_runtime = load_managed_vm_runtime_manifest(state_root=str(state_root), instance_id="task-1")
    instance = load_managed_vm_instance_manifest(state_root=str(state_root), instance_id="task-1")

    assert start["status"] == "START_BLOCKED_GUEST_AGENT_OR_BOOT_IMPL_MISSING"
    assert start["lifecycle_state"] == "start_blocked"
    assert start["execution_ready"] is False
    assert runtime["status"] == "START_BLOCKED_GUEST_AGENT_OR_BOOT_IMPL_MISSING"
    assert runtime["process_alive"] is False
    assert status["runtime_manifest_present"] is True
    assert status["helper_payload"]["status"] == "RUNTIME_MANIFEST_PRESENT"
    assert stop["status"] == "STOPPED"
    assert stopped_runtime["lifecycle_state"] == "stopped"
    assert instance["overlay_present"] is True
    assert instance["lifecycle_state"] == "stopped"
    assert [call[1] for call in calls] == ["start", "runtime-status", "stop"]


def test_start_managed_vm_instance_uses_virtualization_runner_and_records_live_pid(tmp_path: Path, monkeypatch) -> None:
    runner = tmp_path / "conos-vz-runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    captured = {}

    class FakePopen:
        pid = 43210

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["kwargs"] = dict(kwargs)
            runtime_path = Path(cmd[cmd.index("--runtime-manifest") + 1])
            runtime_path.write_text(
                json.dumps(
                    {
                        "schema_version": "conos.managed_vm_provider/v1",
                        "artifact_type": "managed_vm_runtime",
                        "status": "STARTED",
                        "lifecycle_state": "started",
                        "reason": "test runner marked VM started",
                        "state_root": str(state_root),
                        "image_id": "unit-image",
                        "instance_id": "task-1",
                        "process_pid": "43210",
                        "process_alive": True,
                        "virtual_machine_started": True,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                        "launcher_kind": "apple_virtualization_runner",
                        "no_host_fallback": True,
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return None

    monkeypatch.setattr(managed_vm_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "43210")
    monkeypatch.setattr(managed_vm_module, "_copy_file_efficient", _copy_file_efficient_for_test)

    start = start_managed_vm_instance(
        state_root=str(state_root),
        runner_path=str(runner),
        image_id="unit-image",
        instance_id="task-1",
        startup_wait_seconds=0.1,
    )
    runtime = load_managed_vm_runtime_manifest(state_root=str(state_root), instance_id="task-1")
    instance = load_managed_vm_instance_manifest(state_root=str(state_root), instance_id="task-1")

    assert start["status"] == "STARTED"
    assert start["launcher_kind"] == "apple_virtualization_runner"
    assert start["process_pid"] == "43210"
    assert start["process_alive"] is True
    assert runtime["status"] == "STARTED"
    assert runtime["virtual_machine_started"] is True
    assert instance["process_pid"] == "43210"
    assert Path(instance["writable_disk_path"]).exists()
    assert captured["cmd"][:2] == [str(runner.resolve()), "run"]
    assert "--disk-path" in captured["cmd"]
    assert "--console-log" in captured["cmd"]
    assert "--shared-dir" in captured["cmd"]
    assert "--shared-tag" in captured["cmd"]
    assert start["guest_console_log_path"].endswith("guest-console.log")
    assert start["guest_shared_dir_tag"] == "conos_host"
    assert captured["kwargs"]["start_new_session"] is True


def test_start_managed_vm_instance_waits_for_guest_agent_ready_after_vm_start(tmp_path: Path, monkeypatch) -> None:
    runner = tmp_path / "conos-vz-runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    runtime_path_holder: dict[str, Path] = {}
    sleep_calls: list[float] = []

    class FakePopen:
        pid = 43211

        def __init__(self, cmd, **kwargs):
            runtime_path = Path(cmd[cmd.index("--runtime-manifest") + 1])
            runtime_path_holder["path"] = runtime_path
            runtime_path.write_text(
                json.dumps(
                    {
                        "schema_version": "conos.managed_vm_provider/v1",
                        "artifact_type": "managed_vm_runtime",
                        "status": "STARTED",
                        "lifecycle_state": "started",
                        "reason": "test runner marked VM started",
                        "state_root": str(state_root),
                        "image_id": "unit-image",
                        "instance_id": "task-1",
                        "process_pid": "43211",
                        "process_alive": True,
                        "virtual_machine_started": True,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                        "launcher_kind": "apple_virtualization_runner",
                        "no_host_fallback": True,
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return None

    def fake_sleep(seconds):
        sleep_calls.append(float(seconds))
        runtime_path = runtime_path_holder["path"]
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "reason": "guest agent ready",
                "guest_agent_ready": True,
                "execution_ready": True,
                "guest_agent_capabilities": ["ready", "exec"],
            }
        )
        runtime_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(managed_vm_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "43211")
    monkeypatch.setattr(managed_vm_module, "_copy_file_efficient", _copy_file_efficient_for_test)
    monkeypatch.setattr(managed_vm_module.time, "sleep", fake_sleep)

    start = start_managed_vm_instance(
        state_root=str(state_root),
        runner_path=str(runner),
        image_id="unit-image",
        instance_id="task-1",
        startup_wait_seconds=1.0,
    )

    assert sleep_calls
    assert start["status"] == "STARTED"
    assert start["guest_agent_ready"] is True
    assert start["execution_ready"] is True
    assert start["runtime_manifest"]["guest_agent_capabilities"] == ["ready", "exec"]


def test_start_managed_vm_instance_auto_builds_runner_before_start(tmp_path: Path, monkeypatch) -> None:
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    built_runner = Path(managed_vm_module.managed_vm_runner_build_output_path(str(state_root)))
    calls = {}

    def fake_build_runner(**kwargs):
        calls["build_kwargs"] = dict(kwargs)
        built_runner.parent.mkdir(parents=True, exist_ok=True)
        built_runner.write_text("#!/bin/sh\n", encoding="utf-8")
        return {
            "schema_version": "conos.managed_vm_provider/v1",
            "operation": "build_virtualization_runner",
            "status": "BUILT",
            "output_path": str(built_runner),
            "output_present": True,
        }

    class FakePopen:
        pid = 43212

        def __init__(self, cmd, **kwargs):
            calls["cmd"] = list(cmd)
            runtime_path = Path(cmd[cmd.index("--runtime-manifest") + 1])
            runtime_path.write_text(
                json.dumps(
                    {
                        "schema_version": "conos.managed_vm_provider/v1",
                        "artifact_type": "managed_vm_runtime",
                        "status": "STARTED",
                        "lifecycle_state": "started",
                        "reason": "test auto-built runner marked VM started",
                        "state_root": str(state_root),
                        "image_id": "unit-image",
                        "instance_id": "task-1",
                        "process_pid": "43212",
                        "process_alive": True,
                        "virtual_machine_started": True,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                        "launcher_kind": "apple_virtualization_runner",
                        "no_host_fallback": True,
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return None

    def fake_runner_path(explicit="", state_root=""):
        explicit_path = str(explicit or "")
        if explicit_path and Path(explicit_path).exists():
            return explicit_path
        if built_runner.exists():
            return str(built_runner)
        return ""

    monkeypatch.setattr(managed_vm_module, "managed_vm_runner_path", fake_runner_path)
    monkeypatch.setattr(managed_vm_module, "build_managed_vm_virtualization_runner", fake_build_runner)
    monkeypatch.setattr(managed_vm_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "43212")
    monkeypatch.setattr(managed_vm_module, "_copy_file_efficient", _copy_file_efficient_for_test)

    start = start_managed_vm_instance(
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="task-1",
        startup_wait_seconds=0.1,
    )

    assert start["status"] == "STARTED"
    assert start["runner_path"] == str(built_runner)
    assert start["process_pid"] == "43212"
    assert calls["build_kwargs"]["state_root"] == str(state_root)
    assert calls["cmd"][:2] == [str(built_runner), "run"]


def test_start_managed_vm_instance_blocks_without_runner_when_auto_build_disabled(tmp_path: Path, monkeypatch) -> None:
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    monkeypatch.setattr(managed_vm_module, "managed_vm_runner_path", lambda explicit="", state_root="": "")
    monkeypatch.setattr(managed_vm_module, "managed_vm_helper_path", lambda explicit="": "")

    start = start_managed_vm_instance(
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="task-1",
        auto_build_runner=False,
    )

    assert start["status"] == "START_BLOCKED_RUNNER_UNAVAILABLE"
    assert start["blocker_type"] == "runner_unavailable"
    assert start["no_host_fallback"] is True


def test_ensure_managed_vm_instance_running_starts_missing_default_runtime(tmp_path: Path, monkeypatch) -> None:
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    calls: list[str] = []

    def fake_start(**kwargs):
        calls.append("start")
        runtime_path = managed_vm_module.managed_vm_runtime_manifest_path(str(state_root), "default")
        runtime_payload = {
            "schema_version": managed_vm_module.MANAGED_VM_PROVIDER_VERSION,
            "artifact_type": "managed_vm_runtime",
            "status": "STARTED",
            "lifecycle_state": "started",
            "state_root": str(state_root),
            "image_id": "unit-image",
            "instance_id": "default",
            "process_pid": "4242",
            "process_alive": True,
            "virtual_machine_started": True,
            "guest_agent_ready": True,
            "execution_ready": True,
            "launcher_kind": "apple_virtualization_runner",
            "no_host_fallback": True,
        }
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
        return {"status": "STARTED", "runtime_manifest": runtime_payload, "process_pid": "4242"}

    monkeypatch.setattr(managed_vm_module, "start_managed_vm_instance", fake_start)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "4242")

    result = ensure_managed_vm_instance_running(
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="default",
        guest_wait_seconds=0,
    )

    assert result["status"] == "READY"
    assert result["ready"] is True
    assert result["start_attempted"] is True
    assert result["process_pid"] == "4242"
    assert calls == ["start"]


def test_ensure_managed_vm_instance_running_is_idempotent_when_ready(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "vm-state"
    runtime_path = managed_vm_module.managed_vm_runtime_manifest_path(str(state_root), "default")
    runtime_payload = {
        "schema_version": managed_vm_module.MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_runtime",
        "status": "STARTED",
        "lifecycle_state": "started",
        "state_root": str(state_root),
        "image_id": "unit-image",
        "instance_id": "default",
        "process_pid": "4242",
        "process_alive": True,
        "virtual_machine_started": True,
        "guest_agent_ready": True,
        "execution_ready": True,
        "launcher_kind": "apple_virtualization_runner",
        "no_host_fallback": True,
    }
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")

    def fail_start(**kwargs):
        raise AssertionError("ensure-running should not restart a ready VM")

    monkeypatch.setattr(managed_vm_module, "start_managed_vm_instance", fail_start)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "4242")

    result = ensure_managed_vm_instance_running(
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="default",
        guest_wait_seconds=0,
    )

    assert result["status"] == "READY"
    assert result["already_ready"] is True
    assert result["start_attempted"] is False


def test_managed_vm_health_check_syncs_ready_runtime_to_instance_manifest(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "vm-state"
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    prepare_managed_vm_instance(state_root=str(state_root), image_id="unit-image", instance_id="task-1")
    runtime_path = managed_vm_module.managed_vm_runtime_manifest_path(str(state_root), "task-1")
    runtime_payload = {
        "schema_version": managed_vm_module.MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_runtime",
        "status": "STARTED",
        "lifecycle_state": "started",
        "state_root": str(state_root),
        "image_id": "unit-image",
        "instance_id": "task-1",
        "process_pid": "4242",
        "process_alive": True,
        "virtual_machine_started": True,
        "guest_agent_ready": True,
        "execution_ready": True,
        "launcher_kind": "apple_virtualization_runner",
        "no_host_fallback": True,
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "4242")

    health = managed_vm_health_check(state_root=str(state_root), image_id="unit-image", instance_id="task-1")
    instance = load_managed_vm_instance_manifest(state_root=str(state_root), instance_id="task-1")

    assert health["status"] == "HEALTHY"
    assert health["healthy"] is True
    assert "instance_manifest_synced" in health["repairs"]
    assert instance["guest_agent_ready"] is True
    assert instance["execution_ready"] is True


def test_recover_managed_vm_instance_starts_missing_runtime(tmp_path: Path, monkeypatch) -> None:
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    calls: list[str] = []

    def fake_start(**kwargs):
        calls.append("start")
        runtime_path = managed_vm_module.managed_vm_runtime_manifest_path(str(state_root), "task-1")
        runtime_payload = {
            "schema_version": managed_vm_module.MANAGED_VM_PROVIDER_VERSION,
            "artifact_type": "managed_vm_runtime",
            "status": "STARTED",
            "lifecycle_state": "started",
            "state_root": str(state_root),
            "image_id": "unit-image",
            "instance_id": "task-1",
            "process_pid": "4242",
            "process_alive": True,
            "virtual_machine_started": True,
            "guest_agent_ready": True,
            "execution_ready": True,
            "launcher_kind": "apple_virtualization_runner",
            "no_host_fallback": True,
        }
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
        return {"status": "STARTED", "runtime_manifest": runtime_payload, "process_pid": "4242"}

    monkeypatch.setattr(managed_vm_module, "start_managed_vm_instance", fake_start)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "4242")

    result = recover_managed_vm_instance(
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="task-1",
        guest_wait_seconds=0,
    )

    assert result["status"] == "RECOVERED"
    assert result["recovered"] is True
    assert result["final_health"]["status"] == "HEALTHY"
    assert calls == ["start"]


def test_managed_vm_recovery_drill_kills_recovers_and_verifies_agent(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "vm-state"
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    prepare_managed_vm_instance(state_root=str(state_root), image_id="unit-image", instance_id="task-1")
    runtime_path = managed_vm_module.managed_vm_runtime_manifest_path(str(state_root), "task-1")
    runtime_payload = {
        "schema_version": managed_vm_module.MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_runtime",
        "status": "STARTED",
        "lifecycle_state": "started",
        "state_root": str(state_root),
        "image_id": "unit-image",
        "instance_id": "task-1",
        "process_pid": "4242",
        "process_alive": True,
        "virtual_machine_started": True,
        "guest_agent_ready": True,
        "execution_ready": True,
        "launcher_kind": "apple_virtualization_runner",
        "no_host_fallback": True,
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    alive_pids = {"4242"}
    started_pids: list[str] = []

    def fake_kill(pid, sig):
        if sig == 0:
            if str(pid) not in alive_pids:
                raise OSError(errno.ESRCH, "no such process")
            return None
        alive_pids.discard(str(pid))
        return None

    def fake_start(**kwargs):
        started_pids.append("5678")
        alive_pids.add("5678")
        next_runtime = dict(runtime_payload)
        next_runtime.update(
            {
                "process_pid": "5678",
                "process_alive": True,
                "status": "STARTED",
                "lifecycle_state": "started",
                "virtual_machine_started": True,
                "guest_agent_ready": True,
                "execution_ready": True,
            }
        )
        runtime_path.write_text(json.dumps(next_runtime, indent=2), encoding="utf-8")
        return {"status": "STARTED", "runtime_manifest": next_runtime, "process_pid": "5678"}

    def fake_agent_exec(command, **kwargs):
        return {"status": "COMPLETED", "returncode": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(managed_vm_module.os, "kill", fake_kill)
    monkeypatch.setattr(managed_vm_module, "start_managed_vm_instance", fake_start)
    monkeypatch.setattr(managed_vm_module, "run_managed_vm_agent_command", fake_agent_exec)

    result = managed_vm_recovery_drill(
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="task-1",
        guest_wait_seconds=0,
        agent_timeout_seconds=1,
    )

    assert result["status"] == "DRILL_PASSED"
    assert result["passed"] is True
    assert result["crash_report"]["signal"] == "SIGKILL"
    assert result["post_crash_health"]["status"] == "STOPPED"
    assert result["recovery"]["status"] == "RECOVERED"
    assert result["final_health"]["status"] == "HEALTHY"
    assert result["initial_pid"] == "4242"
    assert result["final_pid"] == "5678"
    assert result["pid_changed"] is True
    assert result["agent_exec_verified"] is True
    assert started_pids == ["5678"]


def test_managed_vm_recovery_soak_summarizes_rounds_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "vm-state"
    report_path = tmp_path / "recovery-soak.json"
    calls: list[int] = []

    def fake_drill(**kwargs):
        round_index = len(calls) + 1
        calls.append(round_index)
        initial_pid = str(4000 + round_index)
        final_pid = str(5000 + round_index)
        return {
            "status": "DRILL_PASSED",
            "passed": True,
            "state_root": str(state_root),
            "image_id": "unit-image",
            "instance_id": "task-1",
            "initial_pid": initial_pid,
            "final_pid": final_pid,
            "pid_changed": True,
            "agent_exec_verified": True,
            "recovery_seconds": float(round_index),
            "duration_seconds": float(round_index) + 0.5,
            "final_health": {"status": "HEALTHY", "healthy": True, "process_pid": final_pid},
            "reason": "",
        }

    def fake_agent_exec(command, **kwargs):
        return {"status": "COMPLETED", "returncode": 0, "stdout": "conos-recovery-soak-ok", "stderr": ""}

    def fake_health(**kwargs):
        return {"status": "HEALTHY", "healthy": True, "process_pid": "5002", "reason": ""}

    monkeypatch.setattr(managed_vm_module, "managed_vm_recovery_drill", fake_drill)
    monkeypatch.setattr(managed_vm_module, "run_managed_vm_agent_command", fake_agent_exec)
    monkeypatch.setattr(managed_vm_module, "managed_vm_health_check", fake_health)

    result = managed_vm_recovery_soak(
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="task-1",
        rounds=2,
        cooldown_seconds=0,
        report_path=str(report_path),
    )

    assert result["status"] == "SOAK_PASSED"
    assert result["passed"] is True
    assert result["rounds_requested"] == 2
    assert result["rounds_completed"] == 2
    assert result["success_count"] == 2
    assert result["failure_count"] == 0
    assert result["success_rate"] == 1.0
    assert result["agent_exec_success_count"] == 2
    assert result["disk_probe_success_count"] == 2
    assert result["pid_changed_count"] == 2
    assert result["recovery_seconds"]["values"] == [1.0, 2.0]
    assert result["recovery_seconds"]["p95"] == 2.0
    assert report_path.exists()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "SOAK_PASSED"
    assert calls == [1, 2]


def test_runtime_status_marks_dead_apple_runner_as_stopped(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "vm-state"
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    prepare_managed_vm_instance(state_root=str(state_root), image_id="unit-image", instance_id="task-1")
    runtime_path = state_root / "instances" / "task-1" / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "status": "STARTED",
                "lifecycle_state": "started",
                "reason": "start callback succeeded",
                "process_pid": "999999",
                "process_alive": True,
                "virtual_machine_started": True,
                "guest_agent_listener_ready": True,
                "guest_agent_ready": False,
                "execution_ready": False,
                "launcher_kind": "apple_virtualization_runner",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: False)

    status = managed_vm_runtime_status(state_root=str(state_root), image_id="unit-image", instance_id="task-1")
    runtime = load_managed_vm_runtime_manifest(state_root=str(state_root), instance_id="task-1")
    instance = load_managed_vm_instance_manifest(state_root=str(state_root), instance_id="task-1")

    assert status["status"] == "STOPPED"
    assert runtime["status"] == "STOPPED"
    assert runtime["process_alive"] is False
    assert runtime["last_guest_boot_diagnostic"]["diagnosis_status"] == "VM_PROCESS_EXITED_BEFORE_AGENT"
    assert instance["status"] == "STOPPED"


def test_process_alive_treats_permission_error_as_alive(monkeypatch) -> None:
    def fake_kill(pid, sig):
        raise PermissionError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(managed_vm_module.os, "kill", fake_kill)

    assert managed_vm_module._process_alive(12345) is True


def test_runtime_status_repairs_stale_stopped_apple_runner(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "vm-state"
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )
    prepare_managed_vm_instance(state_root=str(state_root), image_id="unit-image", instance_id="task-1")
    runtime_path = state_root / "instances" / "task-1" / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "status": "STOPPED",
                "lifecycle_state": "stopped",
                "reason": "previous sandbox status check misclassified the process",
                "process_pid": "999999",
                "process_alive": False,
                "virtual_machine_started": True,
                "guest_agent_listener_ready": True,
                "guest_agent_ready": False,
                "execution_ready": False,
                "launcher_kind": "apple_virtualization_runner",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: True)

    status = managed_vm_runtime_status(state_root=str(state_root), image_id="unit-image", instance_id="task-1")
    runtime = load_managed_vm_runtime_manifest(state_root=str(state_root), instance_id="task-1")

    assert status["status"] == "STARTED"
    assert runtime["status"] == "STARTED"
    assert runtime["process_alive"] is True
    assert "restored stale stopped runtime status" in runtime["reason"]


def test_start_managed_vm_instance_classifies_host_virtualization_blocker(tmp_path: Path, monkeypatch) -> None:
    runner = tmp_path / "conos-vz-runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"bootable-ish disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="unit-image",
        source_disk_path=str(source_disk),
    )

    class FakePopen:
        pid = 43211

        def __init__(self, cmd, **kwargs):
            runtime_path = Path(cmd[cmd.index("--runtime-manifest") + 1])
            runtime_path.write_text(
                json.dumps(
                    {
                        "schema_version": "conos.managed_vm_provider/v1",
                        "artifact_type": "managed_vm_runtime",
                        "status": "START_FAILED",
                        "lifecycle_state": "failed",
                        "reason": "Invalid virtual machine configuration. Virtualization is not available on this hardware.",
                        "state_root": str(state_root),
                        "image_id": "unit-image",
                        "instance_id": "task-1",
                        "process_pid": "43211",
                        "process_alive": False,
                        "virtual_machine_started": False,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                        "launcher_kind": "apple_virtualization_runner",
                        "no_host_fallback": True,
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return 70

    monkeypatch.setattr(managed_vm_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: False)
    monkeypatch.setattr(managed_vm_module, "_copy_file_efficient", _copy_file_efficient_for_test)
    monkeypatch.setattr(
        managed_vm_module,
        "_managed_vm_host_virtualization_capability",
        lambda: {"platform": "darwin", "kern_hv_support": "1", "probe_status": "PROBED"},
    )

    start = start_managed_vm_instance(
        state_root=str(state_root),
        runner_path=str(runner),
        image_id="unit-image",
        instance_id="task-1",
        startup_wait_seconds=0.1,
    )
    runtime = load_managed_vm_runtime_manifest(state_root=str(state_root), instance_id="task-1")
    instance = load_managed_vm_instance_manifest(state_root=str(state_root), instance_id="task-1")

    assert start["status"] == "START_BLOCKED_HOST_VIRTUALIZATION_UNAVAILABLE"
    assert start["lifecycle_state"] == "blocked"
    assert start["blocker_type"] == "host_virtualization_unavailable"
    assert start["virtual_machine_started"] is False
    assert start["host_virtualization_capability"]["kern_hv_support"] == "1"
    assert runtime["status"] == "START_BLOCKED_HOST_VIRTUALIZATION_UNAVAILABLE"
    assert runtime["execution_ready"] is False
    assert instance["status"] == "START_BLOCKED_HOST_VIRTUALIZATION_UNAVAILABLE"


def test_start_blocker_classifies_unsupported_linux_direct_kernel(monkeypatch) -> None:
    monkeypatch.setattr(
        managed_vm_module,
        "_managed_vm_host_virtualization_capability",
        lambda: {"platform": "darwin", "kern_hv_support": "1", "probe_status": "PROBED"},
    )

    payload = managed_vm_module._managed_vm_start_blocker_payload(
        "linux_direct kernel artifact is EFI/PE-wrapped, not a raw Linux kernel Image"
    )

    assert payload["status"] == "START_BLOCKED_UNSUPPORTED_BOOT_ARTIFACT"
    assert payload["lifecycle_state"] == "blocked"
    assert payload["blocker_type"] == "unsupported_boot_artifact"
    assert "raw Linux kernel Image" in payload["next_required_step"]
    assert payload["host_virtualization_capability"]["kern_hv_support"] == "1"


def test_terminate_runner_process_forces_after_sigterm_timeout(monkeypatch) -> None:
    calls = []
    alive_checks = {"count": 0}

    def fake_kill(pid, sig):
        calls.append((pid, sig))

    def fake_process_alive(pid):
        alive_checks["count"] += 1
        if calls and calls[-1][1] == signal.SIGKILL and alive_checks["count"] > 4:
            return False
        return True

    monkeypatch.setattr(managed_vm_module.os, "kill", fake_kill)
    monkeypatch.setattr(managed_vm_module, "_process_alive", fake_process_alive)

    result = managed_vm_module._terminate_runner_process(12345, timeout_seconds=0.1)

    assert result["terminated"] is True
    assert result["force_signal"] == "SIGKILL"
    assert calls[0] == (12345, signal.SIGTERM)
    assert calls[-1] == (12345, signal.SIGKILL)


def test_start_managed_vm_instance_passes_linux_boot_artifacts_to_runner(tmp_path: Path, monkeypatch) -> None:
    runner = tmp_path / "conos-vz-runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "disk.img"
    kernel = tmp_path / "vmlinuz"
    source_disk.write_bytes(b"disk bytes\n")
    kernel.write_bytes(b"kernel bytes\n")
    state_root = tmp_path / "vm-state"
    image = register_managed_vm_linux_boot_image(
        state_root=str(state_root),
        image_id="linux-image",
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        kernel_command_line="console=hvc0 root=/dev/vda rw",
        guest_agent_port=48123,
    )
    captured = {}

    class FakePopen:
        pid = 54321

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            runtime_path = Path(cmd[cmd.index("--runtime-manifest") + 1])
            runtime_path.write_text(
                json.dumps(
                    {
                        "schema_version": "conos.managed_vm_provider/v1",
                        "artifact_type": "managed_vm_runtime",
                        "status": "STARTED",
                        "lifecycle_state": "started",
                        "state_root": str(state_root),
                        "image_id": "linux-image",
                        "instance_id": "task-linux",
                        "process_pid": "54321",
                        "process_alive": True,
                        "virtual_machine_started": True,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                        "launcher_kind": "apple_virtualization_runner",
                        "boot_mode": "linux_direct",
                        "guest_agent_transport": "virtio-vsock",
                        "guest_agent_port": 48123,
                        "no_host_fallback": True,
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return None

    monkeypatch.setattr(managed_vm_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "54321")
    monkeypatch.setattr(managed_vm_module, "_copy_file_efficient", _copy_file_efficient_for_test)

    start = start_managed_vm_instance(
        state_root=str(state_root),
        runner_path=str(runner),
        image_id="linux-image",
        instance_id="task-linux",
        startup_wait_seconds=0.1,
    )

    assert start["status"] == "STARTED"
    assert "--boot-mode" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--boot-mode") + 1] == "linux_direct"
    assert captured["cmd"][captured["cmd"].index("--kernel-path") + 1] == image["kernel_path"]
    assert captured["cmd"][captured["cmd"].index("--kernel-command-line") + 1] == "console=hvc0 root=/dev/vda rw"
    assert captured["cmd"][captured["cmd"].index("--guest-agent-port") + 1] == "48123"
    assert start["runtime_manifest"]["guest_agent_transport"] == "virtio-vsock"


def test_start_managed_vm_instance_attaches_cloud_init_seed_to_runner(tmp_path: Path, monkeypatch) -> None:
    runner = tmp_path / "conos-vz-runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "cloud.img"
    source_disk.write_bytes(b"efi cloud disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_cloud_init_image(
        state_root=str(state_root),
        image_id="cloud-image",
        source_disk_path=str(source_disk),
        guest_agent_port=48123,
    )
    captured = {}

    class FakePopen:
        pid = 65432

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            runtime_path = Path(cmd[cmd.index("--runtime-manifest") + 1])
            seed_path = cmd[cmd.index("--cloud-init-seed") + 1]
            runtime_path.write_text(
                json.dumps(
                    {
                        "schema_version": "conos.managed_vm_provider/v1",
                        "artifact_type": "managed_vm_runtime",
                        "status": "STARTED",
                        "lifecycle_state": "started",
                        "state_root": str(state_root),
                        "image_id": "cloud-image",
                        "instance_id": "task-cloud",
                        "process_pid": "65432",
                        "process_alive": True,
                        "virtual_machine_started": True,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                        "launcher_kind": "apple_virtualization_runner",
                        "boot_mode": "efi_disk",
                        "guest_agent_transport": "virtio-vsock",
                        "guest_agent_port": 48123,
                        "cloud_init_seed_path": seed_path,
                        "cloud_init_seed_present": Path(seed_path).exists(),
                        "cloud_init_seed_read_only": True,
                        "no_host_fallback": True,
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return None

    monkeypatch.setattr(managed_vm_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "65432")
    monkeypatch.setattr(managed_vm_module, "_copy_file_efficient", _copy_file_efficient_for_test)
    monkeypatch.setattr(
        managed_vm_module,
        "_apply_efi_observable_boot_patch",
        lambda disk_path, **kwargs: {
            "status": "PATCHED",
            "disk_path": str(disk_path),
            "patched_paths": ["EFI/debian/grub.cfg"],
            "purpose": "unit-test observable boot patch",
        },
    )

    start = start_managed_vm_instance(
        state_root=str(state_root),
        runner_path=str(runner),
        image_id="cloud-image",
        instance_id="task-cloud",
        startup_wait_seconds=0.1,
    )
    instance = load_managed_vm_instance_manifest(state_root=str(state_root), instance_id="task-cloud")

    assert start["status"] == "STARTED"
    assert "--cloud-init-seed" in captured["cmd"]
    seed_path = Path(captured["cmd"][captured["cmd"].index("--cloud-init-seed") + 1])
    assert seed_path.exists()
    assert start["runtime_manifest"]["cloud_init_seed_present"] is True
    assert start["runtime_manifest"]["efi_observable_boot_patch"]["status"] == "PATCHED"
    assert instance["cloud_init_seed_present"] is True
    assert instance["cloud_init_seed_read_only"] is True


def test_start_managed_vm_instance_enables_efi_agent_initrd_patch_from_manifest(tmp_path: Path, monkeypatch) -> None:
    runner = tmp_path / "conos-vz-runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "cloud.img"
    source_disk.write_bytes(b"vmlinuz-6.1.0-44-arm64\x00initrd.img-6.1.0-44-arm64\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_cloud_init_image(
        state_root=str(state_root),
        image_id="cloud-image",
        source_disk_path=str(source_disk),
        guest_agent_port=48123,
    )
    captured: dict[str, object] = {}

    class FakePopen:
        pid = 65433

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            runtime_path = Path(cmd[cmd.index("--runtime-manifest") + 1])
            runtime_path.write_text(
                json.dumps(
                    {
                        "schema_version": "conos.managed_vm_provider/v1",
                        "artifact_type": "managed_vm_runtime",
                        "status": "STARTED",
                        "lifecycle_state": "started",
                        "state_root": str(state_root),
                        "image_id": "cloud-image",
                        "instance_id": "task-cloud",
                        "process_pid": "65433",
                        "process_alive": True,
                        "virtual_machine_started": True,
                        "guest_agent_ready": False,
                        "execution_ready": False,
                        "launcher_kind": "apple_virtualization_runner",
                        "boot_mode": "efi_disk",
                        "guest_agent_transport": "virtio-vsock",
                        "guest_agent_port": 48123,
                        "no_host_fallback": True,
                    }
                ),
                encoding="utf-8",
            )

        def poll(self):
            return None

    def fake_apply_efi_observable_boot_patch(disk_path, **kwargs):
        captured["patch_kwargs"] = dict(kwargs)
        return {
            "status": "PATCHED",
            "disk_path": str(disk_path),
            "patched_paths": ["EFI/debian/grub.cfg"],
            "agent_initrd": {"status": "CREATED", "path": "CONOSAGT.IMG"},
            "agent_initrd_injection_enabled": bool(kwargs.get("inject_agent_initrd")),
        }

    monkeypatch.setattr(managed_vm_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(managed_vm_module, "_process_alive", lambda pid: str(pid) == "65433")
    monkeypatch.setattr(managed_vm_module, "_copy_file_efficient", _copy_file_efficient_for_test)
    monkeypatch.setattr(managed_vm_module, "_apply_efi_observable_boot_patch", fake_apply_efi_observable_boot_patch)

    start = start_managed_vm_instance(
        state_root=str(state_root),
        runner_path=str(runner),
        image_id="cloud-image",
        instance_id="task-cloud",
        startup_wait_seconds=0.1,
    )

    assert start["status"] == "STARTED"
    assert captured["patch_kwargs"]["inject_agent_initrd"] is True
    assert start["runtime_manifest"]["efi_observable_boot_patch"]["agent_initrd_injection_enabled"] is True


def test_existing_observable_efi_boot_patch_report_reuses_matching_grub(monkeypatch, tmp_path: Path) -> None:
    disk = tmp_path / "disk.img"
    disk.write_bytes(b"fake disk")
    config = managed_vm_module._observable_efi_grub_config(
        root_uuid="11111111-2222-3333-4444-555555555555",
        kernel_path="/boot/vmlinuz-6.1.0-44-arm64",
        initrd_path="/boot/initrd.img-6.1.0-44-arm64",
    )

    def fake_read_fat_file(image_path, *, partition_offset, file_path):
        assert image_path == disk
        assert partition_offset == 4096
        assert file_path in managed_vm_module.MANAGED_VM_OBSERVABLE_GRUB_CONFIG_PATHS
        return config

    monkeypatch.setattr(managed_vm_module, "_read_fat_file_in_image", fake_read_fat_file)
    monkeypatch.setattr(
        managed_vm_module,
        "_efi_boot_fallback_loaders_present",
        lambda image_path, *, partition_offset: {
            "status": "READY",
            "copies": [
                {"target": "EFI/BOOT/grubaa64.efi", "status": "ALREADY_PRESENT"},
                {"target": "EFI/BOOT/mmaa64.efi", "status": "ALREADY_PRESENT"},
            ],
            "created_count": 0,
            "skipped_count": 0,
        },
    )

    report = managed_vm_module._existing_observable_efi_boot_patch_report(disk, partition_offset=4096)

    assert report["status"] == "UNCHANGED"
    assert report["boot_artifacts_source"] == "existing_observable_grub_config"
    assert report["root_uuid"] == "11111111-2222-3333-4444-555555555555"
    assert report["boot_artifacts"]["kernel_path"] == "/boot/vmlinuz-6.1.0-44-arm64"
    assert report["boot_artifacts"]["initrd_path"] == "/boot/initrd.img-6.1.0-44-arm64"
    assert report["idempotent_skip_count"] == len(managed_vm_module.MANAGED_VM_OBSERVABLE_GRUB_CONFIG_PATHS)


def test_apply_efi_observable_boot_patch_fast_path_skips_root_disk_scan(monkeypatch, tmp_path: Path) -> None:
    disk = tmp_path / "disk.img"
    disk.write_bytes(b"fake disk")
    esp = {
        "index": 1,
        "type_guid": str(managed_vm_module.EFI_SYSTEM_PARTITION_GUID),
        "byte_offset": 4096,
    }

    monkeypatch.setattr(managed_vm_module, "_gpt_partitions", lambda image_path: [esp])
    monkeypatch.setattr(
        managed_vm_module,
        "_existing_observable_efi_boot_patch_report",
        lambda image_path, *, partition_offset: {
            "status": "UNCHANGED",
            "disk_path": str(image_path),
            "patched_paths": [],
            "unchanged_paths": list(managed_vm_module.MANAGED_VM_OBSERVABLE_GRUB_CONFIG_PATHS),
        },
    )
    monkeypatch.setattr(
        managed_vm_module,
        "_managed_vm_root_ext_uuid",
        lambda *_args, **_kwargs: pytest.fail("root UUID scan should be skipped on the observable EFI fast path"),
    )
    monkeypatch.setattr(
        managed_vm_module,
        "_scan_linux_boot_artifact_paths",
        lambda *_args, **_kwargs: pytest.fail("full disk boot artifact scan should be skipped on the observable EFI fast path"),
    )

    report = managed_vm_module._apply_efi_observable_boot_patch(disk)

    assert report["status"] == "UNCHANGED"
    assert report["esp_partition_index"] == 1
    assert report["esp_partition_offset"] == 4096


def test_managed_vm_guest_agent_gate_requires_ready_runtime(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"

    missing = managed_vm_guest_agent_gate(state_root=str(state_root), image_id="unit-image", instance_id="task-1")
    runtime_path = state_root / "instances" / "task-1" / "runtime.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "virtual_machine_started": True,
                "guest_agent_ready": True,
                "execution_ready": True,
            }
        ),
        encoding="utf-8",
    )
    ready = managed_vm_guest_agent_gate(state_root=str(state_root), image_id="unit-image", instance_id="task-1")

    assert missing["ready"] is False
    assert "runtime_manifest_missing" in missing["blocked_reasons"]
    assert ready["ready"] is True
    assert ready["status"] == "GUEST_AGENT_READY"
    assert ready["guest_boot_diagnostic"]["diagnosis_status"] == "GUEST_AGENT_READY"


def test_managed_vm_guest_agent_gate_diagnoses_started_vm_without_guest_observability(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    instance_root = state_root / "instances" / "task-1"
    logs_root = instance_root / "logs"
    shared_root = instance_root / "guest-share"
    logs_root.mkdir(parents=True)
    shared_root.mkdir(parents=True)
    console_log = logs_root / "guest-console.log"
    console_log.write_text("", encoding="utf-8")
    runtime_path = instance_root / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "status": "STARTED",
                "virtual_machine_started": True,
                "guest_agent_listener_ready": True,
                "guest_agent_ready": False,
                "execution_ready": False,
                "boot_mode": "efi_disk",
                "cloud_init_seed_enabled": True,
                "cloud_init_seed_present": True,
                "guest_console_log_path": str(console_log),
                "guest_shared_dir_path": str(shared_root),
                "guest_shared_dir_present": True,
            }
        ),
        encoding="utf-8",
    )

    gate = managed_vm_guest_agent_gate(state_root=str(state_root), image_id="unit-image", instance_id="task-1")

    assert gate["ready"] is False
    assert gate["guest_shared_runcmd_marker_present"] is False
    diagnostic = gate["guest_boot_diagnostic"]
    assert diagnostic["diagnosis_status"] == "GUEST_BOOT_NO_OBSERVABILITY"
    assert diagnostic["blocked_stage"] == "guest_boot_or_cloud_init"
    assert diagnostic["observed_signals"]["virtual_machine_started"] is True
    assert diagnostic["observed_signals"]["guest_console_output_observed"] is False
    assert diagnostic["observed_signals"]["guest_shared_runcmd_marker_present"] is False

    stopped = stop_managed_vm_instance(state_root=str(state_root), image_id="unit-image", instance_id="task-1")
    assert stopped["last_guest_boot_diagnostic"]["diagnosis_status"] == "GUEST_BOOT_NO_OBSERVABILITY"
    stopped_runtime = load_managed_vm_runtime_manifest(state_root=str(state_root), instance_id="task-1")
    assert stopped_runtime["last_guest_boot_diagnostic"]["blocked_stage"] == "guest_boot_or_cloud_init"


def test_managed_vm_guest_agent_gate_diagnoses_cloud_init_partial_progress(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    instance_root = state_root / "instances" / "task-1"
    logs_root = instance_root / "logs"
    shared_root = instance_root / "guest-share"
    logs_root.mkdir(parents=True)
    shared_root.mkdir(parents=True)
    console_log = logs_root / "guest-console.log"
    console_log.write_text("", encoding="utf-8")
    (shared_root / "cloud-init-bootcmd.txt").write_text("CONOS_CLOUD_INIT_BOOTCMD\n", encoding="utf-8")
    runtime_path = instance_root / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "status": "STARTED",
                "virtual_machine_started": True,
                "guest_agent_listener_ready": True,
                "guest_agent_ready": False,
                "execution_ready": False,
                "boot_mode": "efi_disk",
                "cloud_init_seed_enabled": True,
                "cloud_init_seed_present": True,
                "guest_console_log_path": str(console_log),
                "guest_shared_dir_path": str(shared_root),
                "guest_shared_dir_present": True,
            }
        ),
        encoding="utf-8",
    )

    gate = managed_vm_guest_agent_gate(state_root=str(state_root), image_id="unit-image", instance_id="task-1")

    assert gate["guest_cloud_init_marker_present"] is True
    assert gate["guest_cloud_init_markers"] == ["cloud-init-bootcmd.txt"]
    diagnostic = gate["guest_boot_diagnostic"]
    assert diagnostic["diagnosis_status"] == "CLOUD_INIT_BOOTCMD_OBSERVED_RUNCMD_NOT_OBSERVED"
    assert diagnostic["blocked_stage"] == "cloud_init"
    assert diagnostic["observed_signals"]["guest_cloud_init_marker_present"] is True


def test_managed_vm_guest_agent_gate_diagnoses_cloud_init_unavailable(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    instance_root = state_root / "instances" / "task-1"
    logs_root = instance_root / "logs"
    shared_root = instance_root / "guest-share"
    logs_root.mkdir(parents=True)
    shared_root.mkdir(parents=True)
    console_log = logs_root / "guest-console.log"
    console_log.write_text("[ 1.0 ] Linux booted on hvc0\n", encoding="utf-8")
    runtime_path = instance_root / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "status": "STARTED",
                "virtual_machine_started": True,
                "guest_agent_listener_ready": True,
                "guest_agent_ready": False,
                "execution_ready": False,
                "boot_mode": "efi_disk",
                "cloud_init_seed_enabled": True,
                "cloud_init_seed_present": True,
                "cloud_init_guest_capability": {
                    "status": "UNAVAILABLE",
                    "cloud_init_likely_available": False,
                    "method": "raw_disk_token_scan",
                },
                "guest_console_log_path": str(console_log),
                "guest_shared_dir_path": str(shared_root),
                "guest_shared_dir_present": True,
            }
        ),
        encoding="utf-8",
    )

    gate = managed_vm_guest_agent_gate(state_root=str(state_root), image_id="unit-image", instance_id="task-1")

    diagnostic = gate["guest_boot_diagnostic"]
    assert diagnostic["diagnosis_status"] == "CLOUD_INIT_UNAVAILABLE_IN_GUEST_IMAGE"
    assert diagnostic["blocked_stage"] == "guest_agent_installation"
    assert diagnostic["observed_signals"]["cloud_init_guest_capability"]["status"] == "UNAVAILABLE"
    recommendation = managed_vm_module._managed_vm_boot_path_recommendation(
        boot_mode="efi_disk",
        guest_boot_diagnostic=diagnostic,
        start_status="STARTED",
    )
    assert recommendation["status"] == "BOOT_PATH_CLOUD_INIT_UNAVAILABLE"
    assert recommendation["recommended_boot_mode"] == "linux_direct_with_verified_initrd_bundle_or_preinstalled_agent"
    assert recommendation["retry_same_path"] is False


def test_managed_vm_guest_agent_gate_diagnoses_agent_enable_without_handshake(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    instance_root = state_root / "instances" / "task-1"
    shared_root = instance_root / "guest-share"
    shared_root.mkdir(parents=True)
    (shared_root / "cloud-init-runcmd.txt").write_text("CONOS_CLOUD_INIT_RUNCMD\n", encoding="utf-8")
    (shared_root / "cloud-init-agent-enable.txt").write_text("CONOS_CLOUD_INIT_AGENT_ENABLE rc=1\n", encoding="utf-8")
    runtime_path = instance_root / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "status": "STARTED",
                "virtual_machine_started": True,
                "guest_agent_listener_ready": True,
                "guest_agent_ready": False,
                "execution_ready": False,
                "boot_mode": "efi_disk",
                "cloud_init_seed_enabled": True,
                "cloud_init_seed_present": True,
                "guest_shared_dir_path": str(shared_root),
                "guest_shared_dir_present": True,
            }
        ),
        encoding="utf-8",
    )

    gate = managed_vm_guest_agent_gate(state_root=str(state_root), image_id="unit-image", instance_id="task-1")

    assert gate["guest_shared_runcmd_marker_present"] is True
    assert gate["guest_cloud_init_markers"] == ["cloud-init-runcmd.txt", "cloud-init-agent-enable.txt"]
    assert "rc=1" in gate["guest_cloud_init_marker_tails"]["cloud-init-agent-enable.txt"]
    diagnostic = gate["guest_boot_diagnostic"]
    assert diagnostic["diagnosis_status"] == "CLOUD_INIT_AGENT_ENABLE_OBSERVED_HANDSHAKE_NOT_OBSERVED"
    assert diagnostic["blocked_stage"] == "guest_agent"


def test_managed_vm_guest_agent_gate_diagnoses_linux_direct_without_early_signal(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    instance_root = state_root / "instances" / "task-1"
    logs_root = instance_root / "logs"
    shared_root = instance_root / "guest-share"
    logs_root.mkdir(parents=True)
    shared_root.mkdir(parents=True)
    console_log = logs_root / "guest-console.log"
    console_log.write_text("", encoding="utf-8")
    runtime_path = instance_root / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "status": "STARTED",
                "virtual_machine_started": True,
                "guest_agent_listener_ready": True,
                "guest_agent_ready": False,
                "execution_ready": False,
                "boot_mode": "linux_direct",
                "cloud_init_seed_enabled": False,
                "guest_console_log_path": str(console_log),
                "guest_shared_dir_path": str(shared_root),
                "guest_shared_dir_present": True,
            }
        ),
        encoding="utf-8",
    )

    gate = managed_vm_guest_agent_gate(state_root=str(state_root), image_id="unit-image", instance_id="task-1")

    diagnostic = gate["guest_boot_diagnostic"]
    assert diagnostic["diagnosis_status"] == "LINUX_DIRECT_NO_EARLY_GUEST_SIGNAL"
    assert diagnostic["blocked_stage"] == "guest_boot_or_initramfs"
    assert diagnostic["observed_signals"]["boot_mode"] == "linux_direct"
    assert "only one /init entry" in diagnostic["recommended_next_steps"][0]


def test_managed_vm_guest_agent_gate_diagnoses_unsupported_boot_artifact(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    runtime_path = state_root / "instances" / "task-1" / "runtime.json"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text(
        json.dumps(
            {
                "process_pid": "",
                "status": "START_BLOCKED_UNSUPPORTED_BOOT_ARTIFACT",
                "lifecycle_state": "blocked",
                "blocker_type": "unsupported_boot_artifact",
                "reason": "linux_direct kernel artifact is EFI/PE-wrapped, not a raw Linux kernel Image",
                "virtual_machine_started": False,
                "guest_agent_ready": False,
                "execution_ready": False,
                "boot_mode": "linux_direct",
            }
        ),
        encoding="utf-8",
    )

    gate = managed_vm_guest_agent_gate(state_root=str(state_root), image_id="unit-image", instance_id="task-1")

    assert gate["ready"] is False
    assert gate["guest_boot_diagnostic"]["diagnosis_status"] == "UNSUPPORTED_BOOT_ARTIFACT"
    assert gate["guest_boot_diagnostic"]["blocked_stage"] == "host_start_preflight"
    assert "raw Linux kernel Image" in gate["guest_boot_diagnostic"]["recommended_next_steps"][0]


def test_managed_vm_guest_agent_gate_surfaces_initramfs_trace_markers(tmp_path: Path) -> None:
    state_root = tmp_path / "vm-state"
    instance_root = state_root / "instances" / "task-1"
    shared_root = instance_root / "guest-share"
    shared_root.mkdir(parents=True)
    marker = shared_root / "conos-initramfs-local-top.txt"
    marker.write_text("CONOS_INITRAMFS_LOCAL_TOP\n", encoding="utf-8")
    runtime_path = instance_root / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "status": "STARTED",
                "virtual_machine_started": True,
                "guest_agent_listener_ready": True,
                "guest_agent_ready": False,
                "execution_ready": False,
                "boot_mode": "linux_direct",
                "guest_shared_dir_path": str(shared_root),
                "guest_shared_dir_present": True,
            }
        ),
        encoding="utf-8",
    )

    gate = managed_vm_guest_agent_gate(state_root=str(state_root), image_id="unit-image", instance_id="task-1")

    assert gate["ready"] is False
    assert gate["guest_initramfs_trace_marker_present"] is True
    assert gate["guest_initramfs_trace_markers"] == ["conos-initramfs-local-top.txt"]
    assert gate["guest_boot_diagnostic"]["diagnosis_status"] == "INITRAMFS_TRACE_OBSERVED_AGENT_NOT_READY"
    assert gate["guest_boot_diagnostic"]["blocked_stage"] == "guest_root_mount_or_initramfs"


def test_managed_vm_agent_status_and_exec_are_gated(tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    state_root = tmp_path / "vm-state"
    calls = []

    class Completed:
        returncode = 78
        stdout = json.dumps(
            {
                "status": "GUEST_AGENT_NOT_READY",
                "reason": "guest transport missing",
                "guest_agent_ready": False,
                "execution_ready": False,
            }
        )
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return Completed()

    monkeypatch.setattr(managed_vm_module.subprocess, "run", fake_run)

    status = managed_vm_guest_agent_status(
        state_root=str(state_root),
        helper_path=str(helper),
        image_id="unit-image",
        instance_id="task-1",
    )
    blocked = run_managed_vm_agent_command(
        ["python3", "-c", "print('ok')"],
        state_root=str(state_root),
        helper_path=str(helper),
        image_id="unit-image",
        instance_id="task-1",
    )

    assert status["ready"] is False
    assert status["helper_payload"]["status"] == "GUEST_AGENT_NOT_READY"
    assert blocked["status"] == "EXEC_BLOCKED_GUEST_AGENT_NOT_READY"
    assert blocked["returncode"] == 78
    assert [call[1] for call in calls] == ["agent-status"]


def test_managed_vm_agent_exec_uses_runner_request_spool_when_ready(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "vm-state"
    runtime = state_root / "instances" / "task-1" / "runtime.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "virtual_machine_started": True,
                "guest_agent_ready": True,
                "execution_ready": True,
                "launcher_kind": "apple_virtualization_runner",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(managed_vm_module, "managed_vm_helper_path", lambda explicit="": "")

    def fake_wait(result_path: Path, *, timeout_seconds: int):
        request_files = sorted((state_root / "instances" / "task-1" / "agent-requests").glob("*.request.json"))
        assert request_files
        request = json.loads(request_files[0].read_text(encoding="utf-8"))
        assert request["event_type"] == "exec"
        assert request["command"] == ["python3", "-c", "print('ok')"]
        assert str(result_path).endswith(".result.json")
        assert timeout_seconds == 7
        return {
            "event_type": "exec_result",
            "status": "COMPLETED",
            "returncode": 0,
            "stdout": "ok\n",
            "stderr": "",
        }

    monkeypatch.setattr(managed_vm_module, "_wait_for_managed_vm_agent_result", fake_wait)

    result = run_managed_vm_agent_command(
        ["python3", "-c", "print('ok')"],
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="task-1",
        timeout_seconds=7,
    )

    assert result["status"] == "COMPLETED"
    assert result["returncode"] == 0
    assert result["stdout"] == "ok\n"
    assert result["agent_transport"] == "apple_virtualization_request_spool"
    assert result["helper_path"] == ""


def test_managed_vm_agent_exec_spool_carries_binary_stdin(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "vm-state"
    runtime = state_root / "instances" / "task-1" / "runtime.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "virtual_machine_started": True,
                "guest_agent_ready": True,
                "execution_ready": True,
                "launcher_kind": "apple_virtualization_runner",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(managed_vm_module, "managed_vm_helper_path", lambda explicit="": "")
    stdin_bytes = b"\x00tar-like-bytes\n"

    def fake_wait(result_path: Path, *, timeout_seconds: int):
        request_files = sorted((state_root / "instances" / "task-1" / "agent-requests").glob("*.request.json"))
        request = json.loads(request_files[0].read_text(encoding="utf-8"))
        assert base64.b64decode(request["stdin_b64"].encode("ascii")) == stdin_bytes
        assert request["cwd"] == "/workspace"
        assert str(result_path).endswith(".result.json")
        return {
            "event_type": "exec_result",
            "status": "COMPLETED",
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "stdout_b64": base64.b64encode(b"\xffbinary-out").decode("ascii"),
            "stderr_b64": "",
        }

    monkeypatch.setattr(managed_vm_module, "_wait_for_managed_vm_agent_result", fake_wait)

    result = run_managed_vm_agent_command(
        ["cat"],
        state_root=str(state_root),
        image_id="unit-image",
        instance_id="task-1",
        timeout_seconds=7,
        cwd="/workspace",
        stdin_bytes=stdin_bytes,
    )

    assert result["status"] == "COMPLETED"
    assert result["stdout_b64"] == base64.b64encode(b"\xffbinary-out").decode("ascii")
    assert result["stdin_bytes_present"] is True


def test_conos_vm_lifecycle_cli(capsys, tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    source_disk = tmp_path / "seed.img"
    source_disk.write_bytes(b"disk bytes\n")
    state_root = tmp_path / "vm-state"
    register_managed_vm_base_image(
        state_root=str(state_root),
        image_id="cli-image",
        source_disk_path=str(source_disk),
    )

    class Completed:
        returncode = 0
        stderr = ""

        def __init__(self, status: str, lifecycle_state: str):
            self.stdout = json.dumps(
                {
                    "status": status,
                    "lifecycle_state": lifecycle_state,
                    "reason": "",
                    "overlay_path": str(state_root / "instances" / "cli-task" / "overlay.img"),
                    "overlay_present": True,
                    "process_pid": "",
                    "virtual_machine_started": False,
                    "guest_agent_ready": False,
                    "execution_ready": False,
                }
            )

    def fake_run(cmd, **kwargs):
        if cmd[1] == "start":
            return Completed("START_BLOCKED_GUEST_AGENT_OR_BOOT_IMPL_MISSING", "start_blocked")
        if cmd[1] == "runtime-status":
            return Completed("RUNTIME_MANIFEST_PRESENT", "unknown_from_helper")
        if cmd[1] == "stop":
            return Completed("STOPPED", "stopped")
        raise AssertionError(cmd)

    monkeypatch.setattr(managed_vm_module.subprocess, "run", fake_run)

    assert (
        conos_cli.main(
            [
                "vm",
                "start-instance",
                "--state-root",
                str(state_root),
                "--helper-path",
                str(helper),
                "--image-id",
                "cli-image",
                "--instance-id",
                "cli-task",
            ]
        )
        == 0
    )
    start_payload = json.loads(capsys.readouterr().out)
    assert start_payload["lifecycle_state"] == "start_blocked"

    assert (
        conos_cli.main(
            [
                "vm",
                "runtime-status",
                "--state-root",
                str(state_root),
                "--helper-path",
                str(helper),
                "--image-id",
                "cli-image",
                "--instance-id",
                "cli-task",
            ]
        )
        == 0
    )
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["runtime_manifest_present"] is True

    assert (
        conos_cli.main(
            [
                "vm",
                "stop-instance",
                "--state-root",
                str(state_root),
                "--helper-path",
                str(helper),
                "--image-id",
                "cli-image",
                "--instance-id",
                "cli-task",
            ]
        )
        == 0
    )
    stop_payload = json.loads(capsys.readouterr().out)
    assert stop_payload["lifecycle_state"] == "stopped"


def test_conos_vm_agent_status_and_exec_cli(capsys, tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    state_root = tmp_path / "vm-state"

    class Completed:
        returncode = 78
        stdout = json.dumps({"status": "GUEST_AGENT_NOT_READY", "reason": "guest transport missing"})
        stderr = ""

    monkeypatch.setattr(managed_vm_module.subprocess, "run", lambda cmd, **kwargs: Completed())

    assert (
        conos_cli.main(
            [
                "vm",
                "agent-status",
                "--state-root",
                str(state_root),
                "--helper-path",
                str(helper),
                "--image-id",
                "cli-image",
                "--instance-id",
                "cli-task",
            ]
        )
        == 0
    )
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["ready"] is False

    assert (
        conos_cli.main(
            [
                "vm",
                "agent-exec",
                "--state-root",
                str(state_root),
                "--helper-path",
                str(helper),
                "--image-id",
                "cli-image",
                "--instance-id",
                "cli-task",
                "--",
                "python3",
                "-c",
                "print('ok')",
            ]
        )
        == 0
    )
    exec_payload = json.loads(capsys.readouterr().out)
    assert exec_payload["status"] == "EXEC_BLOCKED_GUEST_AGENT_NOT_READY"
    assert exec_payload["returncode"] == 78


def test_managed_vm_guest_agent_prints_ready_handshake() -> None:
    agent = REPO_ROOT / "tools" / "managed_vm" / "guest_agent" / "conos_guest_agent.py"

    completed = subprocess.run(
        [sys.executable, str(agent), "--print-handshake", "--port", "48123"],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert payload["event_type"] == "guest_agent_ready"
    assert payload["protocol_version"] == "conos.guest_agent.protocol/v0.1"
    assert payload["execution_ready"] is True
    assert payload["port"] == 48123
    assert "exec" in payload["capabilities"]


def test_managed_vm_guest_agent_exec_supports_binary_stdin_stdout(tmp_path: Path) -> None:
    agent = REPO_ROOT / "tools" / "managed_vm" / "guest_agent" / "conos_guest_agent.py"
    spec = importlib.util.spec_from_file_location("conos_guest_agent_for_test", agent)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    stdin_bytes = b"\x00hello\n"
    result = module._run_exec_command(
        {
            "command": [
                sys.executable,
                "-c",
                "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data[::-1])",
            ],
            "stdin_b64": base64.b64encode(stdin_bytes).decode("ascii"),
            "timeout_seconds": 5,
            "cwd": str(tmp_path),
        }
    )

    assert result["status"] == "COMPLETED"
    assert result["returncode"] == 0
    assert base64.b64decode(result["stdout_b64"].encode("ascii")) == stdin_bytes[::-1]


def test_managed_vm_build_helper_reports_missing_swiftc(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "helper.swift"
    source.write_text("print(\"ok\")\n", encoding="utf-8")
    monkeypatch.setattr(managed_vm_module.sys, "platform", "darwin")
    monkeypatch.setattr(managed_vm_module.shutil, "which", lambda name: None)

    report = build_managed_vm_helper(state_root=str(tmp_path / "state"), source_path=str(source))

    assert report["status"] == "UNAVAILABLE"
    assert report["reason"] == "swiftc was not found"
    assert report["source_present"] is True


def test_managed_vm_build_helper_invokes_swiftc(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "helper.swift"
    source.write_text("print(\"ok\")\n", encoding="utf-8")
    output = tmp_path / "bin" / "conos-managed-vm"
    captured = {}

    class Completed:
        returncode = 0
        stdout = "built\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("#!/bin/sh\n", encoding="utf-8")
        return Completed()

    monkeypatch.setattr(managed_vm_module.sys, "platform", "darwin")
    monkeypatch.setattr(managed_vm_module.shutil, "which", lambda name: "/usr/bin/swiftc" if name == "swiftc" else None)
    monkeypatch.setattr(managed_vm_module.subprocess, "run", fake_run)

    report = build_managed_vm_helper(source_path=str(source), output_path=str(output))

    assert report["status"] == "BUILT"
    assert report["output_present"] is True
    assert captured["cmd"] == ["/usr/bin/swiftc", str(source), "-O", "-o", str(output)]


def test_managed_vm_build_runner_invokes_clang_for_objc_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "runner.m"
    source.write_text("print(\"ok\")\n", encoding="utf-8")
    output = tmp_path / "bin" / "conos-vz-runner"
    captured = {}

    class Completed:
        def __init__(self, stdout: str = "built\n"):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        captured.setdefault("cmds", []).append(list(cmd))
        if "clang" in str(cmd[0]):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("#!/bin/sh\n", encoding="utf-8")
            return Completed()
        return Completed(stdout="signed\n")

    monkeypatch.setattr(managed_vm_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        managed_vm_module.shutil,
        "which",
        lambda name: (
            "/usr/bin/clang"
            if name == "clang"
            else ("/usr/bin/swiftc" if name == "swiftc" else ("/usr/bin/codesign" if name == "codesign" else None))
        ),
    )
    monkeypatch.setattr(managed_vm_module.subprocess, "run", fake_run)

    report = build_managed_vm_virtualization_runner(source_path=str(source), output_path=str(output))

    assert report["status"] == "BUILT"
    assert report["output_present"] is True
    assert report["codesigned"] is True
    assert captured["cmds"][0] == [
        "/usr/bin/clang",
        "-fobjc-arc",
        "-framework",
        "Foundation",
        "-framework",
        "Virtualization",
        str(source),
        "-o",
        str(output),
    ]
    assert captured["cmds"][1][:4] == ["/usr/bin/codesign", "--force", "--sign", "-"]


def test_conos_vm_build_helper_cli(capsys, tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "helper.swift"
    source.write_text("print(\"ok\")\n", encoding="utf-8")
    output = tmp_path / "conos-managed-vm"

    class Completed:
        def __init__(self):
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        if "clang" in str(cmd[0]):
            output.write_text("#!/bin/sh\n", encoding="utf-8")
        return Completed()

    monkeypatch.setattr(managed_vm_module.sys, "platform", "darwin")
    monkeypatch.setattr(managed_vm_module.shutil, "which", lambda name: "/usr/bin/swiftc" if name == "swiftc" else None)
    monkeypatch.setattr(managed_vm_module.subprocess, "run", fake_run)

    assert (
        conos_cli.main(
            [
                "vm",
                "build-helper",
                "--source-path",
                str(source),
                "--output-path",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BUILT"
    assert payload["output_path"] == str(output.resolve())


def test_conos_vm_build_runner_cli(capsys, tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "runner.m"
    source.write_text("print(\"ok\")\n", encoding="utf-8")
    output = tmp_path / "conos-vz-runner"

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        output.write_text("#!/bin/sh\n", encoding="utf-8")
        return Completed()

    monkeypatch.setattr(managed_vm_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        managed_vm_module.shutil,
        "which",
        lambda name: (
            "/usr/bin/clang"
            if name == "clang"
            else ("/usr/bin/swiftc" if name == "swiftc" else ("/usr/bin/codesign" if name == "codesign" else None))
        ),
    )
    monkeypatch.setattr(managed_vm_module.subprocess, "run", fake_run)

    assert (
        conos_cli.main(
            [
                "vm",
                "build-runner",
                "--source-path",
                str(source),
                "--output-path",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BUILT"
    assert payload["output_path"] == str(output.resolve())


def test_vm_manager_report_requires_real_provider(monkeypatch) -> None:
    monkeypatch.delenv("CONOS_VM_NAME", raising=False)
    monkeypatch.delenv("CONOS_LIMA_INSTANCE", raising=False)
    monkeypatch.delenv("CONOS_VM_SSH_HOST", raising=False)
    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: None)

    report = vm_manager_report()

    assert report["schema_version"] == "conos.local_mirror.vm_manager/v1"
    assert report["status"] == "UNAVAILABLE"
    assert report["real_vm_boundary"] is False
    assert report["requires_real_provider"] is True


def test_vm_workspace_command_builds_bounded_checkpoint_and_restore() -> None:
    checkpoint_command, checkpoint_id, checkpoint_path = build_vm_workspace_command(
        "checkpoint",
        vm_workdir="/workspace",
        checkpoint_id="plan:abc",
    )
    restore_command, restore_id, restore_path = build_vm_workspace_command(
        "restore",
        vm_workdir="/workspace",
        checkpoint_id="plan:abc",
    )

    assert checkpoint_id == "plan_abc"
    assert restore_id == "plan_abc"
    assert checkpoint_path.endswith("/.conos_vm_checkpoints/plan_abc")
    assert restore_path == checkpoint_path
    assert checkpoint_command[:2] == ["sh", "-lc"]
    assert "tar -C /workspace" in checkpoint_command[-1]
    assert "workspace.tar" in checkpoint_command[-1]
    assert "find /workspace" in restore_command[-1]
    assert "tar -C /workspace -xf" in restore_command[-1]


def test_manage_vm_workspace_records_prepare_audit_event(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)
    captured = {}

    class Completed:
        returncode = 0
        stdout = "prepared\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return Completed()

    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: "/usr/bin/limactl" if name == "limactl" else None)
    monkeypatch.setattr(vm_backend_module.subprocess, "run", fake_run)

    result = manage_vm_workspace(
        source,
        mirror_root,
        operation="prepare",
        vm_provider="lima",
        vm_name="conos-test",
        vm_workdir="/workspace",
    )

    assert result["status"] == "COMPLETED"
    assert result["operation"] == "prepare"
    assert result["real_vm_boundary"] is True
    assert captured["cmd"][:4] == ["/usr/bin/limactl", "shell", "conos-test", "bash"]
    assert "mkdir -p /workspace" in captured["cmd"][-1]
    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    last_event = manifest["audit_events"][-1]
    assert last_event["event_type"] == "mirror_vm_workspace_operation"
    assert last_event["payload"]["operation"] == "prepare"
    assert last_event["payload"]["real_vm_boundary"] is True


def test_conos_mirror_cli_vm_report(capsys, monkeypatch) -> None:
    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: "/usr/bin/ssh" if name == "ssh" else None)

    assert conos_cli.main(["mirror", "vm", "--operation", "report", "--vm-provider", "ssh", "--vm-host", "vm.local"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "conos.local_mirror.vm_manager/v1"
    assert payload["status"] == "AVAILABLE"
    assert payload["provider"] == "ssh"
    assert payload["real_vm_boundary"] is True


def _tar_bytes(files: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in files.items():
            body = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, BytesIO(body))
    return buffer.getvalue()


def test_push_workspace_to_vm_streams_workspace_tar(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    captured = {}

    class Completed:
        returncode = 0
        stdout = b"workspace pushed\n"
        stderr = b""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return Completed()

    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: "/usr/bin/ssh" if name == "ssh" else None)
    monkeypatch.setattr(vm_sync_module.subprocess, "run", fake_run)

    result = push_workspace_to_vm(
        workspace,
        vm_provider="ssh",
        vm_host="vm.local",
        vm_workdir="/workspace",
    )

    assert result.status == "COMPLETED"
    assert result.direction == "push"
    assert result.file_count == 1
    assert result.byte_count > 0
    assert captured["cmd"][:4] == ["/usr/bin/ssh", "vm.local", "bash", "-lc"]
    assert "tar -C /workspace -xf -" in captured["cmd"][-1]
    assert isinstance(captured["kwargs"]["input"], bytes)


def test_pull_workspace_from_vm_safely_extracts_tar(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "old.txt").write_text("old\n", encoding="utf-8")
    remote_tar = _tar_bytes({"result.txt": "from vm\n"})

    class Completed:
        returncode = 0
        stdout = remote_tar
        stderr = b""

    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: "/usr/bin/ssh" if name == "ssh" else None)
    monkeypatch.setattr(vm_sync_module.subprocess, "run", lambda cmd, **kwargs: Completed())

    result = pull_workspace_from_vm(
        workspace,
        vm_provider="ssh",
        vm_host="vm.local",
        vm_workdir="/workspace",
    )

    assert result.status == "COMPLETED"
    assert result.direction == "pull"
    assert result.file_count == 1
    assert not (workspace / "old.txt").exists()
    assert (workspace / "result.txt").read_text(encoding="utf-8") == "from vm\n"


def test_sync_vm_workspace_records_pull_audit_event(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)
    remote_tar = _tar_bytes({"vm.txt": "ok\n"})

    class Completed:
        returncode = 0
        stdout = remote_tar
        stderr = b""

    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: "/usr/bin/ssh" if name == "ssh" else None)
    monkeypatch.setattr(vm_sync_module.subprocess, "run", lambda cmd, **kwargs: Completed())

    result = sync_vm_workspace(
        source,
        mirror_root,
        direction="pull",
        vm_provider="ssh",
        vm_host="vm.local",
        vm_workdir="/workspace",
    )

    assert result["status"] == "COMPLETED"
    assert (mirror_root / "workspace" / "vm.txt").read_text(encoding="utf-8") == "ok\n"
    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    last_event = manifest["audit_events"][-1]
    assert last_event["event_type"] == "mirror_vm_workspace_synced"
    assert last_event["payload"]["direction"] == "pull"
    assert last_event["payload"]["real_vm_boundary"] is True


def test_vm_exec_push_pull_syncs_workspace_around_command(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    materialize_files(source, mirror_root, ["README.md"])
    remote_tar = _tar_bytes({"README.md": "after\n"})
    calls = []

    class Completed:
        def __init__(self, *, returncode=0, stdout=b"ok\n", stderr=b""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        if kwargs.get("text"):
            calls.append(("exec", list(cmd), dict(kwargs)))
            return Completed(stdout="command ok\n", stderr="")
        calls.append(("sync", list(cmd), dict(kwargs)))
        if kwargs.get("input") is None:
            return Completed(stdout=remote_tar)
        return Completed(stdout=b"workspace pushed\n")

    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: "/usr/bin/ssh" if name == "ssh" else None)
    monkeypatch.setattr(vm_sync_module.subprocess, "run", fake_run)

    result = run_mirror_command(
        source,
        mirror_root,
        [sys.executable, "-c", "print('ok')"],
        allowed_commands=[sys.executable],
        backend="vm",
        vm_provider="ssh",
        vm_host="vm.local",
        vm_workdir="/workspace",
        vm_sync_mode="push-pull",
    )

    assert result.returncode == 0
    assert result.vm_sync_mode == "push-pull"
    assert [row["direction"] for row in result.vm_workspace_sync] == ["push", "pull"]
    assert (mirror_root / "workspace" / "README.md").read_text(encoding="utf-8") == "after\n"
    assert [kind for kind, _, _ in calls] == ["sync", "exec", "sync"]
    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    event_types = [event["event_type"] for event in manifest["audit_events"]]
    assert event_types[-3:] == [
        "mirror_vm_workspace_synced",
        "mirror_vm_workspace_synced",
        "mirror_command_executed",
    ]


def test_managed_vm_exec_uses_conos_helper_and_default_push_pull(tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    state_root = tmp_path / "vm-state"
    runtime = state_root / "instances" / "task-1" / "runtime.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "virtual_machine_started": True,
                "guest_agent_ready": True,
                "execution_ready": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONOS_MANAGED_VM_HELPER", str(helper))
    monkeypatch.setenv("CONOS_MANAGED_VM_STATE_ROOT", str(state_root))
    monkeypatch.setenv("CONOS_MANAGED_VM_IMAGE_ID", "conos-test-image")
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    materialize_files(source, mirror_root, ["README.md"])
    remote_tar = _tar_bytes({"README.md": "after managed\n"})
    calls = []

    class Completed:
        def __init__(self, *, returncode=0, stdout=b"ok\n", stderr=b""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        if kwargs.get("input") is None and not kwargs.get("text"):
            return Completed(stdout=remote_tar)
        if kwargs.get("text"):
            return Completed(stdout="managed command ok\n", stderr="")
        return Completed(stdout=b"workspace pushed\n")

    monkeypatch.setattr(vm_sync_module.subprocess, "run", fake_run)

    result = run_mirror_command(
        source,
        mirror_root,
        [sys.executable, "-c", "print('ok')"],
        allowed_commands=[sys.executable],
        backend="managed-vm",
        vm_name="task-1",
        vm_workdir="/workspace",
    )

    assert result.backend == "managed-vm"
    assert result.vm_provider == "managed"
    assert result.real_vm_boundary is True
    assert result.vm_sync_mode == "push-pull"
    assert [row["direction"] for row in result.vm_workspace_sync] == ["push", "pull"]
    assert calls[1][0][0] == str(helper)
    assert calls[1][0][1:4] == ["agent-exec", "--state-root", str(state_root)]
    assert "--image-id" in calls[1][0]
    assert "conos-test-image" in calls[1][0]
    assert (mirror_root / "workspace" / "README.md").read_text(encoding="utf-8") == "after managed\n"
    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    last_event = manifest["audit_events"][-1]
    assert last_event["payload"]["backend"] == "managed-vm"
    assert last_event["payload"]["vm_provider"] == "managed"
    assert last_event["payload"]["credential_boundary"] == "vm_guest_isolated_explicit_env_only_redacted_in_audit"
    assert last_event["payload"]["host_env_forwarded_to_guest"] is False
    assert last_event["payload"]["source_sync_allowed"] is False
    assert last_event["payload"]["source_sync_requires_patch_gate"] is True


def test_managed_vm_exec_uses_runner_spool_without_legacy_helper(tmp_path: Path, monkeypatch) -> None:
    runner = tmp_path / "conos-vz-runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    state_root = tmp_path / "vm-state"
    runtime = state_root / "instances" / "task-1" / "runtime.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        json.dumps(
            {
                "process_pid": str(os.getpid()),
                "virtual_machine_started": True,
                "guest_agent_ready": True,
                "execution_ready": True,
                "launcher_kind": "apple_virtualization_runner",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("CONOS_MANAGED_VM_HELPER", raising=False)
    monkeypatch.setenv("CONOS_MANAGED_VM_RUNNER", str(runner))
    monkeypatch.setenv("CONOS_MANAGED_VM_STATE_ROOT", str(state_root))
    monkeypatch.setenv("CONOS_MANAGED_VM_IMAGE_ID", "conos-test-image")
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    materialize_files(source, mirror_root, ["README.md"])
    remote_tar = _tar_bytes({"README.md": "after spool\n"})
    calls = []

    def fake_agent(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        if kwargs.get("stdin_bytes") is not None:
            return {"status": "COMPLETED", "returncode": 0, "stdout": "workspace pushed\n", "stderr": ""}
        if command[:2] == ["sh", "-lc"] and "tar -C" in command[2]:
            return {
                "status": "COMPLETED",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "stdout_b64": base64.b64encode(remote_tar).decode("ascii"),
            }
        return {"status": "COMPLETED", "returncode": 0, "stdout": "managed command ok\n", "stderr": ""}

    monkeypatch.setattr(vm_sync_module, "run_managed_vm_agent_command", fake_agent)
    monkeypatch.setattr(vm_backend_module, "run_managed_vm_agent_command", fake_agent)

    result = run_mirror_command(
        source,
        mirror_root,
        [sys.executable, "-c", "print('ok')"],
        allowed_commands=[sys.executable],
        backend="managed-vm",
        vm_name="task-1",
        vm_workdir="/workspace",
    )

    assert result.backend == "managed-vm"
    assert result.vm_provider == "managed"
    assert result.real_vm_boundary is True
    assert result.vm_sync_mode == "push-pull"
    assert [row["direction"] for row in result.vm_workspace_sync] == ["push", "pull"]
    assert calls[0][0][:2] == ["sh", "-lc"]
    assert calls[1][0][:2] == ["bash", "-lc"]
    assert calls[2][0][:2] == ["sh", "-lc"]
    assert (mirror_root / "workspace" / "README.md").read_text(encoding="utf-8") == "after spool\n"
    assert result.provider_command[0] == "managed-vm-agent-spool"


def test_managed_vm_exec_blocks_until_guest_agent_ready(tmp_path: Path, monkeypatch) -> None:
    helper = tmp_path / "conos-managed-vm"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("CONOS_MANAGED_VM_HELPER", str(helper))
    monkeypatch.setenv("CONOS_MANAGED_VM_STATE_ROOT", str(tmp_path / "vm-state"))
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    materialize_files(source, mirror_root, ["README.md"])

    with pytest.raises(MirrorScopeError, match="managed VM guest agent is not ready"):
        run_mirror_command(
            source,
            mirror_root,
            [sys.executable, "-c", "print('ok')"],
            allowed_commands=[sys.executable],
            backend="managed-vm",
            vm_name="task-1",
            vm_workdir="/workspace",
        )

    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    last_event = manifest["audit_events"][-1]
    assert last_event["event_type"] == "mirror_vm_backend_unavailable"
    assert "guest agent is not ready" in last_event["payload"]["reason"]


def test_mirror_command_supports_ssh_vm_backend_command_construction(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)
    captured = {}

    class Completed:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return Completed()

    monkeypatch.setattr(vm_backend_module.shutil, "which", lambda name: "/usr/bin/ssh" if name == "ssh" else None)
    monkeypatch.setattr(vm_backend_module.subprocess, "run", fake_run)

    result = run_mirror_command(
        source,
        mirror_root,
        [sys.executable, "-c", "print('ok')"],
        allowed_commands=[sys.executable],
        backend="vm",
        vm_provider="ssh",
        vm_host="conos-vm.local",
        vm_workdir="/workspace",
        extra_env={"TOKEN": "secret"},
    )

    assert result.backend == "vm"
    assert result.vm_provider == "ssh"
    assert result.vm_host == "conos-vm.local"
    assert result.real_vm_boundary is True
    assert captured["cmd"][:4] == ["/usr/bin/ssh", "conos-vm.local", "bash", "-lc"]
    assert "TOKEN=secret" in captured["cmd"][-1]
    assert "TOKEN=<redacted>" in result.provider_command[-1]


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
    assert plan["apply_scope"]["mode"] == "patch_gate_added_or_modified_files_only"
    assert plan["apply_scope"]["apply_method"] == "unified_text_patch"
    assert plan["apply_scope"]["copy_back_allowed"] is False
    assert plan["apply_scope"]["requires_source_hash_match"] is True
    assert plan["apply_scope"]["creates_rollback_checkpoint"] is True
    assert plan["actionable_changes"][0]["relative_path"] == "README.md"

    with pytest.raises(MirrorScopeError):
        apply_sync_plan(source, mirror_root, plan_id="wrong", approved_by="machine")

    result = apply_sync_plan(source, mirror_root, plan_id=plan["plan_id"], approved_by="machine")
    assert result["apply_method"] == "unified_text_patch"
    assert result["sync_gate_mode"] == "patch_gate_added_or_modified_files_only"
    assert result["copy_back_allowed"] is False
    assert result["checkpoint_path"]
    assert result["source_hash_checks"][0]["matched"] is True
    assert result["mirror_hash_checks"][0]["matched"] is True
    assert result["synced_files"][0]["relative_path"] == "README.md"
    assert (source / "README.md").read_text(encoding="utf-8") == "after\n"


def test_sync_plan_ignores_generated_runtime_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    mirror = create_empty_mirror(source, mirror_root)
    cache_file = mirror.workspace_root / ".pytest_cache" / "v" / "cache" / "nodeids"
    pycache_file = mirror.workspace_root / "core" / "__pycache__" / "runtime_budget.cpython-310.pyc"
    cache_file.parent.mkdir(parents=True)
    pycache_file.parent.mkdir(parents=True)
    cache_file.write_text("[]", encoding="utf-8")
    pycache_file.write_bytes(b"cache")

    mirror = create_empty_mirror(source, mirror_root)
    diff = compute_mirror_diff(source, mirror_root)
    plan = build_sync_plan(source, mirror_root)

    assert mirror.to_manifest()["workspace_file_count"] == 0
    assert diff == []
    assert plan["actionable_changes"] == []


def test_sync_plan_apply_creates_checkpoint_and_rollback_restores_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    materialize_files(source, mirror_root, ["README.md"])
    (mirror_root / "workspace" / "README.md").write_text("after\n", encoding="utf-8")
    plan = build_sync_plan(source, mirror_root)

    applied = apply_sync_plan(source, mirror_root, plan_id=plan["plan_id"], approved_by="machine")
    checkpoint_path = Path(applied["checkpoint_path"])
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["files"][0]["original_sha256"] == plan["actionable_changes"][0]["source_sha256"]
    assert checkpoint["files"][0]["reverse_patch"]
    assert (source / "README.md").read_text(encoding="utf-8") == "after\n"

    rolled_back = rollback_sync_plan(source, mirror_root, plan_id=plan["plan_id"])

    assert rolled_back["restored_files"][0]["relative_path"] == "README.md"
    assert rolled_back["restored_files"][0]["rollback_action"] == "reverse_patch_applied"
    assert (source / "README.md").read_text(encoding="utf-8") == "before\n"
    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    event_types = [event["event_type"] for event in manifest["audit_events"]]
    assert "sync_plan_checkpoint_created" in event_types
    assert "sync_plan_rolled_back" in event_types


def test_rollback_removes_file_added_by_sync_plan(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mirror_root = tmp_path / "mirror"
    create_empty_mirror(source, mirror_root)
    (mirror_root / "workspace" / "NEW.md").write_text("new file\n", encoding="utf-8")
    plan = build_sync_plan(source, mirror_root)

    apply_sync_plan(source, mirror_root, plan_id=plan["plan_id"], approved_by="machine")
    assert (source / "NEW.md").exists()

    rolled_back = rollback_sync_plan(source, mirror_root, plan_id=plan["plan_id"])

    assert rolled_back["restored_files"][0]["rollback_action"] == "removed_added_file"
    assert not (source / "NEW.md").exists()


def test_apply_sync_plan_rejects_stale_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("before\n", encoding="utf-8")
    mirror_root = tmp_path / "mirror"
    materialize_files(source, mirror_root, ["README.md"])
    (mirror_root / "workspace" / "README.md").write_text("mirror update\n", encoding="utf-8")
    plan = build_sync_plan(source, mirror_root)

    (source / "README.md").write_text("concurrent source update\n", encoding="utf-8")

    with pytest.raises(MirrorScopeError, match="source hash mismatch"):
        apply_sync_plan(source, mirror_root, plan_id=plan["plan_id"], approved_by="machine")

    assert (source / "README.md").read_text(encoding="utf-8") == "concurrent source update\n"
    manifest = json.loads((mirror_root / "control" / "manifest.json").read_text(encoding="utf-8"))
    event_types = [event["event_type"] for event in manifest["audit_events"]]
    assert "sync_plan_rejected_source_hash_mismatch" in event_types


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
        backend="local",
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
                "--backend",
                "local",
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

    assert (
        conos_cli.main(
            [
                "mirror",
                "rollback",
                "--source-root",
                str(source),
                "--mirror-root",
                str(mirror_root),
                "--plan-id",
                plan["plan_id"],
            ]
        )
        == 0
    )
    rollback_payload = json.loads(capsys.readouterr().out)
    assert rollback_payload["restored_files"][0]["relative_path"] == "README.md"
    assert (source / "README.md").read_text(encoding="utf-8") == "before\n"
