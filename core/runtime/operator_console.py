from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


OPERATOR_CONSOLE_VERSION = "conos.operator_console/v1"


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _upper(value: Any) -> str:
    return _text(value).upper()


def _guidance_rows(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for row in _as_list(payload.get("recovery_guidance")):
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _dedup_actions(guidance: Iterable[Mapping[str, Any]], *, limit: int = 6) -> list[str]:
    actions: list[str] = []
    for row in guidance:
        for action in _as_list(row.get("next_actions")):
            clean = _text(action)
            if clean and clean not in actions:
                actions.append(clean)
            if len(actions) >= limit:
                return actions
    return actions


def _severity(guidance: Iterable[Mapping[str, Any]]) -> str:
    severities = {_text(row.get("severity")) for row in guidance}
    if "action_needed" in severities:
        return "action_needed"
    if "warning" in severities:
        return "warning"
    return "none"


def _health(status: str, severity: str) -> str:
    normalized = _upper(status)
    if normalized in {"FAILED", "BLOCKED", "DEGRADED"} or severity == "action_needed":
        return "needs_action"
    if normalized in {"WARN", "NEEDS_VALIDATION", "NEEDS_ACTION"} or severity == "warning":
        return "warning"
    return "healthy"


def _message_for(kind: str, health: str, payload: Mapping[str, Any]) -> str:
    if kind == "setup":
        if bool(payload.get("dry_run")):
            if bool(payload.get("one_click")):
                return "一键安装预演完成，未写入文件；请按 next_actions 决定是否执行。"
            return "安装预演完成，未写入文件。"
        if health == "healthy":
            return "安装准备完成。"
        if health == "warning":
            return "安装基础步骤完成，但仍有后续验证或 VM 边界动作。"
        return "安装流程需要处理失败步骤或阻塞项。"
    if kind == "status":
        waiting = len(_as_list(payload.get("waiting_approvals")))
        if health == "healthy" and waiting == 0:
            return "Con OS 正常在线，没有等待审批。"
        if waiting:
            return f"Con OS 在线，但有 {waiting} 个审批等待处理。"
        return "Con OS 需要处理降级、失败或运行时告警。"
    if kind == "doctor":
        summary = _as_dict(payload.get("summary"))
        required_failed = int(summary.get("required_failed", 0) or 0)
        warnings = int(summary.get("warnings", 0) or 0)
        if required_failed:
            return f"Doctor 发现 {required_failed} 个必需问题，先按 next_actions 修复。"
        if warnings:
            return f"Doctor 发现 {warnings} 个警告，系统可运行但还不是干净状态。"
        return "Doctor 检查通过。"
    if kind == "logs":
        stderr = _text(payload.get("stderr"))
        stdout = _text(payload.get("stdout"))
        if stderr:
            return "日志中 stderr 有内容，建议先查看 error_signals。"
        if stdout:
            return "日志可读，未发现 stderr。"
        return "还没有可读日志，可能服务未启动或尚未写入输出。"
    if kind == "approvals":
        count = int(payload.get("count", 0) or 0)
        if count:
            return f"当前有 {count} 个审批项，side-effect 动作会等待你处理。"
        return "当前没有等待审批。"
    if kind == "validate-install":
        if _upper(payload.get("status")) == "READY":
            return "安装验证通过。"
        return "安装验证还没完全通过，请按 next_actions 继续。"
    return "Con OS 状态已汇总。"


def summarize_runtime_status(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = _as_dict(payload)
    guidance = _guidance_rows(data)
    severity = _severity(guidance)
    health = _health(_text(data.get("status")), severity)
    metrics = _as_dict(data.get("metrics"))
    runtime_mode = _as_dict(data.get("runtime_mode"))
    watchdog = _as_dict(data.get("watchdog"))
    return {
        "schema_version": OPERATOR_CONSOLE_VERSION,
        "surface": "status",
        "health": health,
        "status": _text(data.get("status")) or "UNKNOWN",
        "runtime_mode": _text(runtime_mode.get("mode")) or "",
        "message": _message_for("status", health, data),
        "waiting_approval_count": len(_as_list(data.get("waiting_approvals"))),
        "run_count": int(metrics.get("run_count", 0) or 0),
        "degraded_reasons": _as_list(watchdog.get("degraded_reasons")),
        "top_issues": [_text(row.get("issue")) for row in guidance[:5] if _text(row.get("issue"))],
        "next_actions": _dedup_actions(guidance),
    }


def summarize_setup(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = _as_dict(payload)
    one_click = _as_dict(data.get("one_click_report"))
    steps = [dict(row) for row in _as_list(one_click.get("steps")) if isinstance(row, Mapping)]
    failed_steps = [_text(row.get("name")) for row in steps if not bool(row.get("ok", True))]
    pending_steps = [
        _text(row.get("name"))
        for row in steps
        if _upper(row.get("status")) in {"NEEDS_ACTION", "MISSING", "STOPPED", "BLOCKED", "NOT_READY"}
        and _text(row.get("name"))
    ]
    action_needed = _as_list(one_click.get("action_needed"))
    guidance: list[Dict[str, Any]] = []
    top_issues: list[str] = []
    if failed_steps:
        top_issues.extend(failed_steps)
        guidance.append(
            {
                "issue": failed_steps[0],
                "severity": "action_needed",
                "message": "one-click setup step failed",
                "next_actions": ["conos validate-install", "conos doctor"],
            }
        )
    if action_needed:
        issue = "vm_default_boundary" if bool(data.get("one_click")) else "setup_action_needed"
        top_issues.append(issue)
        actions: list[str] = []
        for item in action_needed:
            if isinstance(item, Mapping):
                display = _text(item.get("display")) or " ".join(_as_list(item.get("command")))
                if display:
                    actions.append(display)
            else:
                clean = _text(item)
                if clean:
                    actions.append(clean)
        guidance.append(
            {
                "issue": issue,
                "severity": "warning" if bool(data.get("dry_run")) else "action_needed",
                "message": "setup has remaining operator actions",
                "next_actions": actions or ["conos validate-install", "conos doctor"],
            }
        )
    severity = _severity(guidance)
    health = _health(_text(data.get("status")), severity)
    return {
        "schema_version": OPERATOR_CONSOLE_VERSION,
        "surface": "setup",
        "health": health,
        "status": _text(data.get("status")) or "UNKNOWN",
        "message": _message_for("setup", health, data),
        "dry_run": bool(data.get("dry_run")),
        "one_click": bool(data.get("one_click")),
        "written": bool(data.get("written")),
        "failed_steps": [item for item in failed_steps if item],
        "pending_steps": [item for item in pending_steps if item],
        "top_issues": [item for item in top_issues if item][:8],
        "next_actions": _dedup_actions(guidance, limit=8),
    }


def summarize_doctor(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = _as_dict(payload)
    guidance = _guidance_rows(data)
    severity = _severity(guidance)
    health = _health(_text(data.get("status")), severity)
    summary = _as_dict(data.get("summary"))
    return {
        "schema_version": OPERATOR_CONSOLE_VERSION,
        "surface": "doctor",
        "health": health,
        "status": _text(data.get("status")) or "UNKNOWN",
        "message": _message_for("doctor", health, data),
        "required_failed": int(summary.get("required_failed", 0) or 0),
        "warnings": int(summary.get("warnings", 0) or 0),
        "total_checks": int(summary.get("total", 0) or 0),
        "top_issues": [_text(row.get("issue")) for row in guidance[:8] if _text(row.get("issue"))],
        "next_actions": _dedup_actions(guidance, limit=8),
    }


def summarize_logs(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = _as_dict(payload)
    stderr = _text(data.get("stderr"))
    stdout = _text(data.get("stdout"))
    error_tokens = ("traceback", "error", "exception", "failed", "timeout")
    stderr_lower = stderr.lower()
    stdout_lower = stdout.lower()
    error_signals = [token for token in error_tokens if token in stderr_lower or token in stdout_lower]
    guidance: list[Dict[str, Any]] = []
    if stderr:
        guidance.append(
            {
                "issue": "stderr_not_empty",
                "severity": "warning",
                "message": "stderr log has recent output",
                "next_actions": ["conos doctor", "conos status", "conos logs --tail 200"],
            }
        )
    if error_signals:
        guidance.append(
            {
                "issue": "error_signals_in_logs",
                "severity": "warning",
                "message": "recent logs contain error-like tokens",
                "next_actions": ["conos doctor", "conos status"],
            }
        )
    if not stdout and not stderr:
        guidance.append(
            {
                "issue": "logs_empty",
                "severity": "warning",
                "message": "no recent stdout/stderr output",
                "next_actions": ["conos status", "conos start", "conos doctor"],
            }
        )
    severity = _severity(guidance)
    health = _health("WARN" if guidance else "OK", severity)
    return {
        "schema_version": OPERATOR_CONSOLE_VERSION,
        "surface": "logs",
        "health": health,
        "status": "WARN" if guidance else "OK",
        "message": _message_for("logs", health, data),
        "stdout_path": _text(data.get("stdout_path")),
        "stderr_path": _text(data.get("stderr_path")),
        "tail": int(data.get("tail", 0) or 0),
        "error_signals": error_signals,
        "top_issues": [_text(row.get("issue")) for row in guidance],
        "next_actions": _dedup_actions(guidance),
    }


def summarize_approvals(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = _as_dict(payload)
    approvals = [dict(row) for row in _as_list(data.get("approvals")) if isinstance(row, Mapping)]
    waiting = [
        row
        for row in approvals
        if _upper(row.get("status") or "WAITING") in {"WAITING", "WAITING_APPROVAL"}
    ]
    guidance = []
    if waiting:
        guidance.append(
            {
                "issue": "waiting_for_approval",
                "severity": "action_needed",
                "message": "approval items are waiting",
                "next_actions": ["conos approvals", "conos approve <approval_id>"],
            }
        )
    health = "needs_action" if waiting else "healthy"
    return {
        "schema_version": OPERATOR_CONSOLE_VERSION,
        "surface": "approvals",
        "health": health,
        "status": "WAITING" if waiting else "OK",
        "message": _message_for("approvals", health, data),
        "approval_count": len(approvals),
        "waiting_count": len(waiting),
        "approval_ids": [_text(row.get("approval_id")) for row in waiting if _text(row.get("approval_id"))],
        "top_issues": [_text(row.get("issue")) for row in guidance],
        "next_actions": _dedup_actions(guidance),
    }


def summarize_validate_install(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = _as_dict(payload)
    checks = [dict(row) for row in _as_list(data.get("checks")) if isinstance(row, Mapping)]
    guidance = []
    for row in checks:
        operator_guidance = _as_dict(row.get("operator_guidance"))
        if operator_guidance:
            guidance.append(operator_guidance)
    for action in _as_list(data.get("setup_actions")) + _as_list(data.get("validation_remaining")):
        if isinstance(action, Mapping):
            display = _text(action.get("display")) or " ".join(_as_list(action.get("command")))
            if display:
                guidance.append(
                    {
                        "issue": _text(action.get("check")) or "validation_remaining",
                        "severity": "action_needed",
                        "message": _text(action.get("reason")) or "validation step remains",
                        "next_actions": [display],
                    }
                )
    severity = _severity(guidance)
    health = _health(_text(data.get("status")), severity)
    return {
        "schema_version": OPERATOR_CONSOLE_VERSION,
        "surface": "validate-install",
        "health": health,
        "status": _text(data.get("status")) or "UNKNOWN",
        "message": _message_for("validate-install", health, data),
        "failed_checks": [_text(row.get("name")) for row in checks if not bool(row.get("ok"))],
        "top_issues": [_text(row.get("issue")) for row in guidance[:8] if _text(row.get("issue"))],
        "next_actions": _dedup_actions(guidance, limit=8),
    }


def attach_operator_panel(payload: Mapping[str, Any], *, surface: str) -> Dict[str, Any]:
    data = dict(payload or {})
    if surface == "setup":
        panel = summarize_setup(data)
    elif surface == "status":
        panel = summarize_runtime_status(data)
    elif surface == "doctor":
        panel = summarize_doctor(data)
    elif surface == "logs":
        panel = summarize_logs(data)
    elif surface == "approvals":
        panel = summarize_approvals(data)
    elif surface == "validate-install":
        panel = summarize_validate_install(data)
    else:
        panel = {
            "schema_version": OPERATOR_CONSOLE_VERSION,
            "surface": str(surface or "unknown"),
            "health": _health(_text(data.get("status")), _severity(_guidance_rows(data))),
            "status": _text(data.get("status")) or "UNKNOWN",
            "message": _message_for(str(surface or "unknown"), "healthy", data),
            "next_actions": _dedup_actions(_guidance_rows(data)),
        }
    data["operator_panel"] = panel
    data["operator_summary"] = panel["message"]
    if "recovery_guidance" not in data and panel.get("next_actions"):
        data["recovery_guidance"] = [
            {
                "schema_version": OPERATOR_CONSOLE_VERSION,
                "issue": issue,
                "severity": "warning",
                "message": panel["message"],
                "next_actions": panel.get("next_actions", []),
            }
            for issue in list(panel.get("top_issues") or [])[:1]
        ]
    return data
