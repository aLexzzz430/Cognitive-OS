from __future__ import annotations

import json
from pathlib import Path

import conos_cli
import modules.local_mirror.managed_vm as managed_vm
from modules.local_mirror.managed_vm import _managed_vm_setup_plan_from_report, managed_vm_prepare_default_boundary


def _base_report(tmp_path: Path) -> dict:
    return {
        "schema_version": "conos.managed_vm_provider/v1",
        "status": "UNAVAILABLE",
        "state_root": str(tmp_path / "vm"),
        "image_id": "conos-base",
        "instance_id": "default",
        "real_vm_boundary": False,
        "virtualization_runner_available": False,
        "base_image_present": False,
        "image_manifest_present": False,
        "instance_manifest_present": False,
        "runtime_manifest_present": False,
        "runtime_process_alive": False,
        "guest_agent_gate": {"status": "BLOCKED", "reason": "runtime_not_started"},
    }


def test_setup_plan_blocks_on_missing_runner_and_image(tmp_path: Path) -> None:
    plan = _managed_vm_setup_plan_from_report(_base_report(tmp_path))

    assert plan["status"] == "NEEDS_SETUP"
    assert plan["safe_to_run_tasks"] is False
    assert plan["default_execution_boundary_ready"] is False
    assert plan["no_host_fallback"] is True
    stage_status = {stage["name"]: stage["status"] for stage in plan["stages"]}
    assert stage_status["virtualization_runner"] == "MISSING"
    assert stage_status["base_image"] == "MISSING"
    assert plan["next_actions"][0]["command"][:3] == ["conos", "vm", "build-runner"]


def test_setup_plan_recovers_stale_runtime_before_guest_agent(tmp_path: Path) -> None:
    report = _base_report(tmp_path)
    report.update(
        {
            "status": "AVAILABLE",
            "real_vm_boundary": True,
            "virtualization_runner_available": True,
            "base_image_present": True,
            "image_manifest_present": True,
            "instance_manifest_present": True,
            "runtime_manifest_present": True,
            "runtime_process_alive": False,
        }
    )

    plan = _managed_vm_setup_plan_from_report(report)

    assert plan["status"] == "NEEDS_RECOVERY"
    assert plan["safe_to_run_tasks"] is False
    assert plan["next_actions"][0]["command"][:3] == ["conos", "vm", "recover-instance"]


def test_setup_plan_reports_ready_only_when_guest_execution_is_ready(tmp_path: Path) -> None:
    report = _base_report(tmp_path)
    report.update(
        {
            "status": "AVAILABLE",
            "real_vm_boundary": True,
            "virtualization_runner_available": True,
            "base_image_present": True,
            "image_manifest_present": True,
            "instance_manifest_present": True,
            "runtime_manifest_present": True,
            "runtime_process_alive": True,
            "guest_agent_gate": {"status": "READY", "ready": True, "reason": ""},
        }
    )

    plan = _managed_vm_setup_plan_from_report(report)

    assert plan["status"] == "READY"
    assert plan["safe_to_run_tasks"] is True
    assert plan["default_execution_boundary_ready"] is True
    assert plan["next_actions"] == []
    assert all(stage["ready"] for stage in plan["stages"])


def test_product_cli_vm_setup_plan_outputs_readiness_schema(tmp_path: Path, capsys) -> None:
    code = conos_cli.main(["vm", "setup-plan", "--state-root", str(tmp_path / "vm")])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["schema_version"] == "conos.managed_vm_setup_plan/v1"
    assert payload["safe_to_run_tasks"] is False
    assert payload["no_host_fallback"] is True
    assert payload["stages"]
    assert payload["next_actions"]


def test_prepare_default_boundary_is_dry_run_by_default(monkeypatch, tmp_path: Path) -> None:
    def fail_if_called(**kwargs):
        raise AssertionError("dry-run must not execute setup stages")

    monkeypatch.setattr(managed_vm, "build_managed_vm_virtualization_runner", fail_if_called)

    payload = managed_vm_prepare_default_boundary(state_root=str(tmp_path / "vm"))

    assert payload["status"] == "DRY_RUN"
    assert payload["execute"] is False
    assert payload["dry_run"] is True
    assert payload["stage_results"] == []
    assert payload["next_actions"][0]["command"][:3] == ["conos", "vm", "build-runner"]


def test_prepare_default_boundary_executes_next_stage_and_writes_audit(monkeypatch, tmp_path: Path) -> None:
    missing_runner = _base_report(tmp_path)
    ready_report = _base_report(tmp_path)
    ready_report.update(
        {
            "status": "AVAILABLE",
            "real_vm_boundary": True,
            "virtualization_runner_available": True,
            "base_image_present": True,
            "image_manifest_present": True,
            "instance_manifest_present": True,
            "runtime_manifest_present": True,
            "runtime_process_alive": True,
            "guest_agent_gate": {"status": "READY", "ready": True, "reason": ""},
        }
    )
    calls = {"plan": 0, "build": 0}

    def fake_plan(**kwargs):
        calls["plan"] += 1
        if calls["plan"] == 1:
            return _managed_vm_setup_plan_from_report(missing_runner)
        return _managed_vm_setup_plan_from_report(ready_report)

    def fake_build(**kwargs):
        calls["build"] += 1
        return {"status": "BUILT", "output_present": True}

    def fake_agent_exec(command, **kwargs):
        return {"status": "OK", "returncode": 0, "stdout": "conos-vm-ready", "stderr": ""}

    monkeypatch.setattr(managed_vm, "managed_vm_setup_plan", fake_plan)
    monkeypatch.setattr(managed_vm, "build_managed_vm_virtualization_runner", fake_build)
    monkeypatch.setattr(managed_vm, "run_managed_vm_agent_command", fake_agent_exec)

    payload = managed_vm_prepare_default_boundary(state_root=str(tmp_path / "vm"), execute=True)

    assert payload["status"] == "READY"
    assert payload["safe_to_run_tasks"] is True
    assert calls["build"] == 1
    assert payload["stage_results"][0]["stage"] == "virtualization_runner"
    assert payload["stage_results"][-1]["stage"] == "agent_exec_smoke"
    audit_path = Path(payload["audit_path"])
    assert audit_path.exists()
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["stage"] == "virtualization_runner"
    assert events[-1]["stage"] == "agent_exec_smoke"


def test_product_cli_vm_setup_default_is_dry_run(tmp_path: Path, capsys) -> None:
    code = conos_cli.main(["vm", "setup-default", "--state-root", str(tmp_path / "vm")])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["schema_version"] == "conos.managed_vm_default_boundary_setup/v1"
    assert payload["dry_run"] is True
    assert payload["stage_results"] == []
