from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


RUNTIME_RECOVERY_PLAYBOOK_VERSION = "conos.runtime.recovery_playbook/v1"


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _dedup(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        clean = _text(value)
        if clean and clean not in result:
            result.append(clean)
    return result


_CATEGORY_ORDER = [
    "install_runtime",
    "vm_boundary",
    "model_runtime",
    "approval_permission",
    "verifier_tests",
    "logs_runtime",
]


_CATEGORY_SPECS: Dict[str, Dict[str, Any]] = {
    "install_runtime": {
        "title": "安装与运行时基础",
        "issues": {
            "python_version_too_old",
            "core_import_failed",
            "repo_layout_invalid",
            "runtime_not_initialized",
            "setup_manifest",
            "runtime_directories",
            "launchd_plist",
            "launchd_loaded",
            "llm_policy_unavailable",
            "check_failed",
        },
        "summary": "基础安装或运行时目录还没稳定。",
        "likely_cause": "依赖、仓库根目录、setup manifest、launchd 或 runtime home 仍缺一项。",
        "recovery_path": ["conos setup --one-click", "conos validate-install", "conos doctor"],
        "blocked_until": "必需安装检查通过。",
        "escalation_condition": "setup 后仍失败，检查具体 failed check 的 detail。",
    },
    "vm_boundary": {
        "title": "VM 默认执行边界",
        "issues": {
            "managed_vm_unavailable",
            "missing_vm_runner",
            "missing_vm_image",
            "vm_start_blocked",
            "vm_process_not_alive",
            "guest_agent_not_ready",
            "vm_default_boundary",
            "vm_real_boundary",
        },
        "summary": "内置 VM 还不能作为默认 side-effect 边界。",
        "likely_cause": "runner、base image、runtime process 或 guest agent gate 没有就绪。",
        "recovery_path": [
            "conos vm setup-plan",
            "conos vm build-runner",
            "conos vm bootstrap-image",
            "conos vm start-instance",
            "conos vm agent-status",
            "conos vm recover-instance",
        ],
        "blocked_until": "conos vm setup-plan 返回 safe_to_run_tasks=true。",
        "escalation_condition": "runner 已构建且镜像存在但 guest agent 仍不 ready，运行 recovery-drill。",
    },
    "model_runtime": {
        "title": "模型与路由可靠性",
        "issues": {
            "model_unavailable",
            "llm_policy_unavailable",
            "llm_timeout",
            "llm_format_error",
            "llm_budget_exceeded",
        },
        "summary": "模型 provider、路由策略或 endpoint 不稳定。",
        "likely_cause": "Ollama/OpenAI/Codex 登录、网络、模型超时、输出格式或预算策略异常。",
        "recovery_path": [
            "conos llm check",
            "conos llm policy --provider all",
            "conos llm profile-models",
            "conos doctor --live-llm",
        ],
        "blocked_until": "模型检查通过，或任务进入可审计降级/升级路径。",
        "escalation_condition": "本地模型持续超时或格式错误，按 cost policy 升级到强模型。",
    },
    "approval_permission": {
        "title": "审批与权限治理",
        "issues": {
            "waiting_for_approval",
            "permission_denied",
            "side_effect_after_verified_completion",
            "capability_layer_requires_approval",
        },
        "summary": "系统正在等待审批，或当前能力层不允许继续。",
        "likely_cause": "side-effect、credential、network、sync-back 或写路径需要明确授权。",
        "recovery_path": ["conos approvals", "conos approve <approval_id>", "conos status"],
        "blocked_until": "等待项被批准、拒绝或任务被改写到更低权限路径。",
        "escalation_condition": "审批项含 credential/network/sync-back 时必须人工复核。",
    },
    "verifier_tests": {
        "title": "验证与测试失败",
        "issues": {
            "run_failed_or_degraded",
            "pytest_missing",
            "test_failed",
            "verifier_failed",
            "run_failed",
        },
        "summary": "任务或测试验证没有通过。",
        "likely_cause": "测试依赖缺失、verifier 失败、patch 未通过 targeted/full tests。",
        "recovery_path": ["conos logs --tail 200", "conos status", "pytest -q"],
        "blocked_until": "失败证据被读取，修复重新验证通过，或进入 needs_human_review。",
        "escalation_condition": "同一 verifier 连续失败时暂停同步，进入 DEEP_THINK 或人工复核。",
    },
    "logs_runtime": {
        "title": "日志与服务运行",
        "issues": {
            "stderr_not_empty",
            "error_signals_in_logs",
            "logs_empty",
            "runtime_degraded",
        },
        "summary": "服务日志或 watchdog 显示运行态需要检查。",
        "likely_cause": "服务刚启动未写日志、stderr 有异常、watchdog 进入 DEGRADED。",
        "recovery_path": ["conos logs --tail 200", "conos status", "conos doctor"],
        "blocked_until": "日志无新增错误，watchdog 恢复 OK。",
        "escalation_condition": "日志持续出现 timeout/error/traceback，暂停 run 并收集诊断。",
    },
}


def _guidance_issue_rows(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for row in _as_list(payload.get("recovery_guidance")):
        if isinstance(row, Mapping):
            rows.append(dict(row))
    operator_panel = _as_dict(payload.get("operator_panel"))
    for issue in _as_list(operator_panel.get("top_issues")):
        clean = _text(issue)
        if clean:
            rows.append(
                {
                    "issue": clean,
                    "severity": "warning" if operator_panel.get("health") == "warning" else "action_needed",
                    "message": _text(operator_panel.get("message")),
                    "next_actions": _as_list(operator_panel.get("next_actions")),
                }
            )
    for row in _as_list(payload.get("checks")):
        if not isinstance(row, Mapping) or bool(row.get("ok", False)):
            continue
        guidance = _as_dict(row.get("operator_guidance"))
        if guidance:
            rows.append(guidance)
        else:
            rows.append(
                {
                    "issue": _text(row.get("name")) or "check_failed",
                    "severity": "action_needed" if bool(row.get("required", False)) else "warning",
                    "message": _text(row.get("detail")),
                    "next_actions": ["conos doctor"],
                }
            )
    product_gate = _as_dict(payload.get("product_deployment_gate"))
    for blocker in _as_list(product_gate.get("blockers")):
        if isinstance(blocker, Mapping):
            rows.append(
                {
                    "issue": _text(blocker.get("check")) or "product_gate_blocker",
                    "severity": "action_needed",
                    "message": _text(blocker.get("reason")),
                    "next_actions": ["conos validate-install --product", "conos doctor"],
                }
            )
    return rows


def build_recovery_diagnosis_tree(payload: Mapping[str, Any], *, surface: str = "") -> Dict[str, Any]:
    data = _as_dict(payload)
    rows = _guidance_issue_rows(data)
    issue_set = {_text(row.get("issue")) for row in rows if _text(row.get("issue"))}
    category_reports: list[Dict[str, Any]] = []
    for category in _CATEGORY_ORDER:
        spec = _CATEGORY_SPECS[category]
        matched = sorted(issue_set.intersection(set(spec["issues"])))
        if not matched:
            category_reports.append(
                {
                    "category": category,
                    "title": spec["title"],
                    "status": "OK",
                    "matched_issues": [],
                    "summary": "",
                    "likely_cause": "",
                    "recovery_path": [],
                    "blocked_until": "",
                    "escalation_condition": "",
                }
            )
            continue
        severities = {
            _text(row.get("severity"))
            for row in rows
            if _text(row.get("issue")) in set(matched)
        }
        status = "ACTION_NEEDED" if "action_needed" in severities else "WARN"
        category_actions: list[str] = []
        for row in rows:
            if _text(row.get("issue")) in set(matched):
                category_actions.extend(_as_list(row.get("next_actions")))
        if not category_actions:
            category_actions.extend(_as_list(spec.get("recovery_path")))
        category_reports.append(
            {
                "category": category,
                "title": spec["title"],
                "status": status,
                "matched_issues": matched,
                "summary": spec["summary"],
                "likely_cause": spec["likely_cause"],
                "recovery_path": _dedup([*category_actions, *_as_list(spec.get("recovery_path"))])[:8],
                "blocked_until": spec["blocked_until"],
                "escalation_condition": spec["escalation_condition"],
            }
        )
    active = [row for row in category_reports if row["status"] != "OK"]
    if any(row["status"] == "ACTION_NEEDED" for row in active):
        status = "ACTION_NEEDED"
    elif active:
        status = "WARN"
    else:
        status = "OK"
    return {
        "schema_version": RUNTIME_RECOVERY_PLAYBOOK_VERSION,
        "surface": str(surface or data.get("action") or "runtime"),
        "status": status,
        "active_category_count": len(active),
        "active_categories": [row["category"] for row in active],
        "issue_count": len(issue_set),
        "issues": sorted(issue_set),
        "categories": category_reports,
        "operator_summary": (
            "没有恢复阻塞项。"
            if status == "OK"
            else "恢复路径已归类：先处理 ACTION_NEEDED，再处理 WARN。"
        ),
    }


def attach_recovery_diagnosis(payload: Mapping[str, Any], *, surface: str = "") -> Dict[str, Any]:
    data = dict(payload or {})
    data["recovery_diagnosis_tree"] = build_recovery_diagnosis_tree(data, surface=surface)
    return data
