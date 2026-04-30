from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


RUNTIME_RECOVERY_GUIDANCE_VERSION = "conos.runtime.recovery_guidance/v1"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _guidance(issue: str, message: str, next_actions: Iterable[str], *, severity: str = "action_needed") -> Dict[str, Any]:
    return {
        "schema_version": RUNTIME_RECOVERY_GUIDANCE_VERSION,
        "issue": issue,
        "severity": severity,
        "message": message,
        "next_actions": list(next_actions),
    }


def guidance_for_check(check: Mapping[str, Any]) -> Dict[str, Any]:
    name = str(check.get("name") or "")
    detail = str(check.get("detail") or "")
    if bool(check.get("ok", False)):
        return {}
    lowered = detail.lower()
    if name == "python_version":
        return _guidance(
            "python_version_too_old",
            "当前 Python 版本太低，Con OS 需要 Python 3.10 或更高版本。",
            ["安装 Python 3.10+", "重新创建 .venv", "重新运行 conos doctor"],
        )
    if name == "core_import":
        return _guidance(
            "core_import_failed",
            "核心模块无法导入，通常是依赖未安装或命令不在仓库根目录运行。",
            ["python -m pip install -e .", "确认当前目录是 Cognitive OS 仓库根目录", "重新运行 conos doctor"],
        )
    if name == "repo_layout":
        return _guidance(
            "repo_layout_invalid",
            "仓库边界检查失败，可能存在 public/private import 或 adapter 注册问题。",
            ["conos layout", "修复报告中的边界违规后重试"],
        )
    if name == "dev_dependency:pytest":
        return _guidance(
            "pytest_missing",
            "开发验证依赖缺失，测试和开放任务 verifier 可能无法运行。",
            ["python -m pip install -r requirements-dev.txt", "重新运行 pytest -q"],
            severity="warning",
        )
    if name in {"runtime_home", "state_db_parent", "runtime_status_command"}:
        return _guidance(
            "runtime_not_initialized",
            "运行时目录或状态库还没有准备好。",
            ["conos setup", "conos status", "conos logs --tail 120"],
        )
    if name == "managed_vm_provider":
        return _guidance(
            "managed_vm_unavailable",
            "内置 VM provider 当前不可用，通常是缺少 runner、镜像或启动边界未通过。",
            ["conos vm status", "conos vm build-runner", "conos vm image-report", "conos vm bootstrap-image"],
            severity="warning",
        )
    if name == "llm_policy_contracts":
        return _guidance(
            "llm_policy_unavailable",
            "模型策略契约不可用，Provider/Auth/Cost/Context/Verifier 路由可能不能稳定工作。",
            ["conos llm policy --provider all", "conos llm profile-models", "conos doctor --live-llm"],
        )
    if name == "llm_live_check":
        return _guidance(
            "model_unavailable",
            "模型连接失败或超时，系统应该降级、暂停或等待人工处理，而不是偷偷 fallback patch。",
            ["conos llm check", "检查 Ollama/OpenAI/Codex 登录状态", "确认网络和模型 endpoint 可用"],
            severity="warning",
        )
    if "permission" in lowered or "denied" in lowered:
        return _guidance(
            "permission_denied",
            "当前操作被权限拒绝。",
            ["检查文件/目录权限", "确认审批策略是否允许该能力", "在 approvals 中处理等待项"],
        )
    return _guidance(
        "check_failed",
        f"{name or '检查项'} 未通过。",
        ["查看该检查项 detail", "修复后重新运行 conos doctor"],
        severity="warning" if not bool(check.get("required", True)) else "action_needed",
    )


def guidance_for_vm_report(report: Mapping[str, Any]) -> list[Dict[str, Any]]:
    payload = _as_dict(report)
    guidance: list[Dict[str, Any]] = []
    if not bool(payload.get("virtualization_runner_available", False)):
        guidance.append(
            _guidance(
                "missing_vm_runner",
                "Apple Virtualization runner 还不可用，VM 不能真正启动。",
                ["conos vm build-runner", "conos vm status"],
            )
        )
    if not bool(payload.get("image_manifest_present", False)) or not bool(payload.get("base_image_present", False)):
        guidance.append(
            _guidance(
                "missing_vm_image",
                "VM 基础镜像还没有注册或镜像文件缺失。",
                ["conos vm recipe-report", "conos vm bootstrap-image", "conos vm image-report"],
            )
        )
    gate = _as_dict(payload.get("guest_agent_gate"))
    runtime = _as_dict(payload.get("runtime_manifest"))
    runtime_status = str(runtime.get("status") or "")
    if runtime_status.startswith("START_BLOCKED") or str(payload.get("status") or "") == "UNAVAILABLE":
        guidance.append(
            _guidance(
                "vm_start_blocked",
                "VM 启动被阻止或 provider 不可用。",
                ["conos vm status", "conos vm start-instance", "conos vm logs 或查看 runner stderr"],
            )
        )
    if payload.get("runtime_process_alive") is False and bool(payload.get("runtime_manifest_present", False)):
        guidance.append(
            _guidance(
                "vm_process_not_alive",
                "VM runtime 记录存在，但进程已经不在运行。",
                ["conos vm recover-instance", "conos vm runtime-status"],
            )
        )
    if gate and not bool(gate.get("ready", False)):
        guidance.append(
            _guidance(
                "guest_agent_not_ready",
                "VM 已有状态记录，但 guest agent 尚未 ready，不能安全执行任务。",
                ["conos vm agent-status", "conos vm recover-instance", "conos vm recovery-drill"],
            )
        )
    return guidance


def guidance_for_runtime_status(status_payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    payload = _as_dict(status_payload)
    guidance: list[Dict[str, Any]] = []
    if str(payload.get("status") or "") == "DEGRADED":
        watchdog = _as_dict(payload.get("watchdog"))
        reasons = [str(item) for item in _as_list(watchdog.get("degraded_reasons"))]
        if any("ollama" in item.lower() or "model" in item.lower() for item in reasons):
            guidance.append(
                _guidance(
                    "model_unavailable",
                    "模型服务不可用或延迟异常，长任务应进入降级状态，必要时升级到云端模型。",
                    ["conos llm check", "conos status", "检查模型 endpoint 和网络"],
                    severity="warning",
                )
            )
        else:
            guidance.append(
                _guidance(
                    "runtime_degraded",
                    "运行时处于降级状态，需要查看 watchdog 原因。",
                    ["conos status", "conos logs --tail 120", "conos doctor"],
                    severity="warning",
                )
            )
    approvals = _as_list(payload.get("waiting_approvals"))
    if approvals:
        guidance.append(
            _guidance(
                "waiting_for_approval",
                "系统正在等待用户审批，side-effect 动作不会继续执行。",
                ["conos approvals", "conos approve <approval_id>"],
            )
        )
    for run in _as_list(payload.get("runs")):
        run_status = str(_as_dict(run).get("status") or "").upper()
        if run_status in {"FAILED", "DEGRADED"}:
            guidance.append(
                _guidance(
                    "run_failed_or_degraded",
                    "存在失败或降级的 run，需要查看日志、证据和最近测试输出。",
                    ["conos logs --tail 200", "conos status", "必要时 pause/resume 或重新下达任务"],
                )
            )
            break
    return guidance


def attach_guidance_to_checks(checks: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    enriched: list[Dict[str, Any]] = []
    for check in checks:
        row = dict(check)
        guidance = guidance_for_check(row)
        if guidance:
            row["operator_guidance"] = guidance
        enriched.append(row)
    return enriched

