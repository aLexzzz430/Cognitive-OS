"""
planner/plan_state.py

Sprint 3: 正式规划器官

维护当前计划状态.

Rules:
- 第一版只维护单个活动计划
- 不做计划栈或多计划管理
"""

from __future__ import annotations
import hashlib
import json
from typing import Any, Dict, List, Optional

from core.conos_kernel import build_task_contract
from core.orchestration.goal_task_control import (
    derive_verifier_authority_snapshot,
    resolve_effective_task_approval_requirement,
    resolve_effective_task_verification_gate,
    resolve_goal_contract_authority,
    resolve_task_graph_active_task_payload,
)
from core.runtime_budget import (
    merge_llm_capability_specs,
    resolve_llm_capability_policies,
    resolve_llm_capability_policy_entries,
    resolve_llm_route_policies,
)
from planner.plan_schema import Plan, PlanStatus, PlanStep, StepStatus


_HIGH_RISK_STEP_INTENTS = frozenset(
    {
        "submit",
        "commit",
        "publish",
        "deploy",
        "delete",
        "write",
        "checkout",
    }
)

_HIGH_RISK_STEP_TOKENS = (
    "submit",
    "checkout",
    "purchase",
    "delete",
    "remove",
    "write",
    "patch",
    "commit",
    "publish",
    "deploy",
    "finalize",
)

_VERIFICATION_STEP_INTENTS = frozenset({"verify", "test", "inspect", "probe", "measure", "check"})


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _unique_strings(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


_GOAL_RISK_LEVEL_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


def _normalize_goal_risk_level(value: Any, *, default: str = "low") -> str:
    clean = str(value or "").strip().lower()
    if clean in _GOAL_RISK_LEVEL_RANK:
        return clean
    return default


def _max_goal_risk_level(*values: Any, default: str = "low") -> str:
    best = _normalize_goal_risk_level(default, default="low")
    best_rank = _GOAL_RISK_LEVEL_RANK.get(best, 0)
    for value in values:
        clean = str(value or "").strip().lower()
        if clean not in _GOAL_RISK_LEVEL_RANK:
            continue
        rank = _GOAL_RISK_LEVEL_RANK[clean]
        if rank > best_rank:
            best = clean
            best_rank = rank
    return best


def _payload_signature(value: Any) -> str:
    try:
        return json.dumps(_dict_or_empty(value), sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return json.dumps({}, sort_keys=True, separators=(",", ":"), default=str)


def _task_contract_verifier_authority(task_contract: Any) -> Dict[str, Any]:
    payload = _dict_or_empty(task_contract)
    verification_requirement = _dict_or_empty(payload.get("verification_requirement", {}))
    return _dict_or_empty(verification_requirement.get("verifier_authority", {}))


def _step_is_high_risk(step: PlanStep) -> bool:
    intent = str(getattr(step, "intent", "") or "").strip().lower()
    if intent in _HIGH_RISK_STEP_INTENTS:
        return True
    target = str(getattr(step, "target_function", "") or "").strip().lower()
    return bool(target) and any(token in target for token in _HIGH_RISK_STEP_TOKENS)


def _step_verification_gate(step: PlanStep) -> Dict[str, Any]:
    explicit = _dict_or_empty(getattr(step, "verification_gate", {}))
    success_criteria = _string_list(explicit.get("success_criteria", []))
    if not success_criteria:
        target_function = str(getattr(step, "target_function", "") or "").strip()
        target_state = str(getattr(step, "target_state", "") or "").strip()
        if target_function:
            success_criteria.append(f"invoke:{target_function}")
        if target_state:
            success_criteria.append(f"reach_state:{target_state}")
    required = bool(explicit.get("required", False))
    default_mode = "before_completion" if required else (
        "observation" if str(getattr(step, "intent", "") or "").strip().lower() in _VERIFICATION_STEP_INTENTS else "none"
    )
    return {
        "required": required,
        "mode": str(explicit.get("mode", default_mode) or default_mode),
        "verifier_function": str(explicit.get("verifier_function", "") or ""),
        "success_criteria": success_criteria,
        "failure_mode": str(explicit.get("failure_mode", "block") or "block"),
        "last_verified": bool(explicit.get("last_verified", False)),
        "last_verdict": str(
            explicit.get("last_verdict", "pending" if required else "not_required")
            or ("pending" if required else "not_required")
        ),
        "evidence": _dict_or_empty(explicit.get("evidence", {})),
    }


def _step_retry_policy(step: PlanStep) -> Dict[str, Any]:
    explicit = _dict_or_empty(getattr(step, "retry_policy", {}))
    constraints = _dict_or_empty(getattr(step, "constraints", {}))
    max_attempts = explicit.get("max_attempts", constraints.get("max_attempts"))
    if max_attempts in (None, ""):
        max_attempts = 2 if _step_is_high_risk(step) else 1
    try:
        max_attempts = max(1, int(max_attempts))
    except (TypeError, ValueError):
        max_attempts = 1
    return {
        "max_attempts": max_attempts,
        "on_failure": str(
            explicit.get("on_failure", "block" if _step_is_high_risk(step) else "revise")
            or ("block" if _step_is_high_risk(step) else "revise")
        ),
        "backoff": str(explicit.get("backoff", "none") or "none"),
    }


def _step_assigned_worker(step: PlanStep) -> Dict[str, Any]:
    explicit = _dict_or_empty(getattr(step, "assigned_worker", {}))
    constraints = _dict_or_empty(getattr(step, "constraints", {}))
    worker_type = explicit.get("worker_type", constraints.get("worker_type"))
    source = "explicit" if explicit else ("constraint" if constraints.get("worker_type") or constraints.get("assigned_worker") else "derived")
    if not worker_type:
        worker_type = "verifier" if str(getattr(step, "intent", "") or "").strip().lower() in _VERIFICATION_STEP_INTENTS else "executor"
    return {
        "worker_id": str(explicit.get("worker_id", constraints.get("assigned_worker", "primary")) or "primary"),
        "worker_type": str(worker_type or "executor"),
        "ownership": str(explicit.get("ownership", "exclusive") or "exclusive"),
        "source": source,
    }


def _step_approval_requirement(step: PlanStep) -> Dict[str, Any]:
    explicit = _dict_or_empty(getattr(step, "approval_requirement", {}))
    required = bool(explicit.get("required", _step_is_high_risk(step)))
    target_function = str(getattr(step, "target_function", "") or "").strip()
    capability_class = str(explicit.get("capability_class", "") or "")
    if not capability_class and required:
        capability_class = "high_risk_action"
    reason = str(explicit.get("reason", "") or "")
    if not reason and required:
        reason = f"target_function:{target_function}" if target_function else f"intent:{getattr(step, 'intent', '')}"
    return {
        "required": required,
        "risk_level": str(explicit.get("risk_level", "high" if required else "low") or ("high" if required else "low")),
        "capability_class": capability_class,
        "reason": reason,
    }


def _step_is_verification_step(step: PlanStep, verification_functions: List[str]) -> bool:
    intent = str(getattr(step, "intent", "") or "").strip().lower()
    if intent in _VERIFICATION_STEP_INTENTS:
        return True
    verifier_names = {str(item or "").strip().lower() for item in verification_functions if str(item or "").strip()}
    target_function = str(getattr(step, "target_function", "") or "").strip().lower()
    return bool(target_function) and target_function in verifier_names


def _step_approval_state(step: PlanStep) -> Dict[str, Any]:
    explicit = _dict_or_empty(getattr(step, "approval_state", {}))
    approved = bool(explicit.get("approved", False))
    return {
        "approved": approved,
        "approval_grant_id": str(explicit.get("approval_grant_id", "") or ""),
        "approval_sources": _string_list(explicit.get("approval_sources", [])),
        "approved_by": str(explicit.get("approved_by", "") or ""),
        "evidence": _dict_or_empty(explicit.get("evidence", {})),
        "pending": bool(explicit.get("pending", not approved)),
    }


def _step_branch_targets(step: PlanStep) -> List[Dict[str, Any]]:
    constraints = _dict_or_empty(getattr(step, "constraints", {}))
    rows = _dict_list(getattr(step, "branch_targets", []))
    if not rows:
        rows = _dict_list(constraints.get("branch_targets", []))
    branch_targets: List[Dict[str, Any]] = []
    for row in rows:
        target_step_id = str(
            row.get("target_step_id", "")
            or row.get("step_id", "")
            or row.get("target", "")
            or ""
        ).strip()
        if not target_step_id:
            continue
        branch_key = str(
            row.get("branch_key", "")
            or row.get("key", "")
            or row.get("when", "")
            or row.get("result", "")
            or "default"
        ).strip()
        branch_targets.append(
            {
                "branch_key": branch_key or "default",
                "target_step_id": target_step_id,
                "condition": str(row.get("condition", row.get("when", "")) or ""),
                "reason": str(row.get("reason", "") or ""),
            }
        )
    return branch_targets


def _step_rollback_edge(step: PlanStep) -> Dict[str, Any]:
    explicit = _dict_or_empty(getattr(step, "rollback_edge", {}))
    constraints = _dict_or_empty(getattr(step, "constraints", {}))
    target_step_id = str(
        explicit.get("target_step_id", "")
        or explicit.get("step_id", "")
        or constraints.get("rollback_to_step_id", "")
        or ""
    ).strip()
    return {
        "target_step_id": target_step_id,
        "reason": str(explicit.get("reason", constraints.get("rollback_reason", "")) or ""),
        "mode": str(explicit.get("mode", "rollback" if target_step_id else "none") or ("rollback" if target_step_id else "none")),
    }


def _step_retry_state(step: PlanStep) -> Dict[str, Any]:
    explicit = _dict_or_empty(getattr(step, "retry_state", {}))
    policy = _step_retry_policy(step)
    attempts = int(getattr(step, "execution_attempts", 0) or 0)
    max_attempts = int(policy.get("max_attempts", 1) or 1)
    remaining_attempts = max(0, max_attempts - attempts)
    last_failure_reason = str(explicit.get("last_failure_reason", "") or "")
    if not last_failure_reason and getattr(step, "status", None) == StepStatus.FAILED:
        last_failure_reason = str(getattr(step, "execution_result", "") or "")
    return {
        "attempts": attempts,
        "max_attempts": max_attempts,
        "remaining_attempts": remaining_attempts,
        "retry_pending": bool(explicit.get("retry_pending", False)),
        "last_failure_reason": last_failure_reason,
        "last_transition": str(explicit.get("last_transition", "") or ""),
        "rollback_ready": bool(
            _step_rollback_edge(step).get("target_step_id")
            and str(policy.get("on_failure", "") or "") == "rollback"
        ),
    }


def _governance_event_rows(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _step_governance_memory(step: PlanStep) -> Dict[str, Any]:
    explicit = _dict_or_empty(getattr(step, "governance_memory", {}))
    retry_state = _step_retry_state(step)
    verification_gate = _step_verification_gate(step)
    approval_state = _step_approval_state(step)
    rollback_edge = _step_rollback_edge(step)
    return {
        "approval_events": _governance_event_rows(explicit.get("approval_events", [])),
        "verification_events": _governance_event_rows(explicit.get("verification_events", [])),
        "failure_events": _governance_event_rows(explicit.get("failure_events", [])),
        "rollback_events": _governance_event_rows(explicit.get("rollback_events", [])),
        "last_transition": str(
            explicit.get("last_transition", retry_state.get("last_transition", "")) or retry_state.get("last_transition", "")
        ),
        "last_failure_reason": str(
            explicit.get("last_failure_reason", retry_state.get("last_failure_reason", "")) or retry_state.get("last_failure_reason", "")
        ),
        "retry_pending": bool(explicit.get("retry_pending", retry_state.get("retry_pending", False))),
        "approval_pending": bool(explicit.get("approval_pending", approval_state.get("pending", False))),
        "last_approval_required": bool(explicit.get("last_approval_required", _step_approval_requirement(step).get("required", False))),
        "last_approval_grant_id": str(
            explicit.get("last_approval_grant_id", approval_state.get("approval_grant_id", "")) or approval_state.get("approval_grant_id", "")
        ),
        "last_verification_required": bool(explicit.get("last_verification_required", verification_gate.get("required", False))),
        "last_verification_verdict": str(
            explicit.get("last_verification_verdict", verification_gate.get("last_verdict", "not_required"))
            or verification_gate.get("last_verdict", "not_required")
        ),
        "rollback_ready": bool(explicit.get("rollback_ready", retry_state.get("rollback_ready", False))),
        "rollback_target_step_id": str(
            explicit.get("rollback_target_step_id", rollback_edge.get("target_step_id", "")) or rollback_edge.get("target_step_id", "")
        ),
    }


def _memory_event(
    event_type: str,
    **payload: Any,
) -> Dict[str, Any]:
    row = {"event_type": str(event_type or "").strip() or "unknown"}
    for key, value in payload.items():
        if isinstance(value, dict):
            row[str(key)] = dict(value)
        elif isinstance(value, list):
            row[str(key)] = [dict(item) if isinstance(item, dict) else item for item in value]
        else:
            row[str(key)] = value
    return row


def _step_task_node_payload(
    *,
    plan: Plan,
    step: PlanStep,
    goal_id: str,
    node_id: str,
    status: str,
    verification_gate: Optional[Dict[str, Any]] = None,
    approval_requirement: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    success_criteria: List[str] = []
    if step.target_function:
        success_criteria.append(f"invoke:{step.target_function}")
    if step.target_state:
        success_criteria.append(f"reach_state:{step.target_state}")
    capability_specs = resolve_llm_capability_policies(getattr(step, "llm_capability_policies", {}))
    return {
        "node_id": node_id,
        "title": str(step.description or step.intent or step.step_id or ""),
        "status": status,
        "goal_id": goal_id,
        "success_criteria": success_criteria,
        "verification_gate": dict(verification_gate or _step_verification_gate(step)),
        "retry_policy": _step_retry_policy(step),
        "retry_state": _step_retry_state(step),
        "assigned_worker": _step_assigned_worker(step),
        "approval_requirement": dict(approval_requirement or _step_approval_requirement(step)),
        "approval_state": _step_approval_state(step),
        "governance_memory": _step_governance_memory(step),
        "branch_targets": _step_branch_targets(step),
        "rollback_edge": _step_rollback_edge(step),
        "verifier_node_id": "",
        "approval_node_id": "",
        "llm_route_policies": resolve_llm_route_policies(getattr(step, "llm_route_policies", {})),
        "llm_capability_policies": capability_specs,
        "llm_capability_policy_entries": resolve_llm_capability_policy_entries(capability_specs),
        "provenance": {
            "source": "planner.plan_state",
            "plan_id": str(plan.plan_id or ""),
            "step_id": str(step.step_id or ""),
            "revision_count": int(plan.revision_count or 0),
        },
        "metadata": {
            "intent": str(step.intent or ""),
            "target_function": str(step.target_function or ""),
            "target_state": str(step.target_state or ""),
            "constraints": dict(step.constraints or {}),
            "execution_attempts": int(step.execution_attempts or 0),
            "execution_result": step.execution_result,
        },
    }


class PlanState:
    """
    计划状态管理器.
    
    第一版职责:
    1. 维护当前活动计划
    2. 提供步骤查询
    3. 推进步骤
    4. 检查退出条件
    
    不做:
    - 多计划栈
    - 计划历史
    - 复杂回滚
    """
    
    def __init__(self):
        self._current_plan: Optional[Plan] = None
        self._tick_count: int = 0
        self._episode_reward: float = 0.0
        self._discovered_functions: List[str] = []
    
    @property
    def current_plan(self) -> Optional[Plan]:
        """获取当前计划"""
        return self._current_plan
    
    @property
    def has_plan(self) -> bool:
        """是否有活动计划"""
        return self._current_plan is not None
    
    @property
    def current_step(self) -> Optional[PlanStep]:
        """获取当前步骤"""
        if self._current_plan:
            return self._current_plan.current_step
        return None
    
    def set_plan(self, plan: Plan):
        """设置新计划（替换当前计划）"""
        self._current_plan = plan
    
    def clear_plan(self):
        """清除当前计划"""
        self._current_plan = None
    
    def update_context(
        self,
        tick: int,
        reward: float,
        discovered_functions: List[str],
    ):
        """更新上下文（用于退出条件检查）"""
        self._tick_count = tick
        self._episode_reward = reward
        self._discovered_functions = discovered_functions

    @staticmethod
    def _stable_id(prefix: str, *parts: Any) -> str:
        blob = "||".join(str(part or "").strip() for part in parts if str(part or "").strip())
        if not blob:
            blob = prefix
        digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}:{digest}"

    def _step_index_for_id(self, plan: Plan, step_id: str) -> Optional[int]:
        clean = str(step_id or "").strip()
        if not clean:
            return None
        for index, step in enumerate(list(plan.steps or [])):
            if str(getattr(step, "step_id", "") or "").strip() == clean:
                return index
        return None

    def _task_node_id(self, plan: Plan, step: PlanStep) -> str:
        return self._stable_id('task', plan.plan_id, step.step_id, step.description)

    def _approval_node_id(self, plan: Plan, step: PlanStep) -> str:
        return self._stable_id('approval', plan.plan_id, step.step_id, step.description)

    def _verifier_node_id(self, plan: Plan, step: PlanStep) -> str:
        return self._stable_id('verify', plan.plan_id, step.step_id, step.description)

    def _goal_contract_summary(self, plan: Plan) -> Dict[str, Any]:
        exit_criteria = getattr(plan, "exit_criteria", None)
        success_indicator = getattr(exit_criteria, "success_indicator", None)
        max_steps = int(getattr(exit_criteria, "max_steps", 0) or 0)
        max_ticks = int(getattr(exit_criteria, "max_ticks", 0) or 0)
        target_reward = getattr(exit_criteria, "target_reward", None)
        success_criteria: List[str] = []
        if success_indicator:
            success_criteria.append(str(success_indicator))
        abort_criteria = [
            f"max_steps:{max_steps}",
            f"max_ticks:{max_ticks}",
        ]
        if target_reward is not None:
            abort_criteria.append(f"target_reward:{float(target_reward)}")
        planning_contract = _dict_or_empty(getattr(plan, "planning_contract", {}))
        approval_contract = _dict_or_empty(getattr(plan, "approval_contract", {}))
        verification_contract = _dict_or_empty(getattr(plan, "verification_contract", {}))
        completion_contract = _dict_or_empty(getattr(plan, "completion_contract", {}))
        step_rows = list(plan.steps or [])
        allowed_step_intents = _unique_strings(
            _string_list(planning_contract.get("allowed_step_intents", []))
            + [
                str(getattr(step, "intent", "") or "").strip()
                for step in step_rows
                if str(getattr(step, "intent", "") or "").strip()
            ]
        )
        approval_required_functions = _unique_strings(
            _string_list(approval_contract.get("required_functions", []))
            + [
                str(getattr(step, "target_function", "") or "").strip()
                for step in step_rows
                if _step_approval_requirement(step).get("required", False)
                and str(getattr(step, "target_function", "") or "").strip()
            ]
        )
        verification_functions = _unique_strings(
            _string_list(verification_contract.get("verification_functions", []))
            + [
                str(_step_verification_gate(step).get("verifier_function", "") or "").strip()
                for step in step_rows
                if str(_step_verification_gate(step).get("verifier_function", "") or "").strip()
            ]
        )
        requires_verification = bool(
            verification_contract.get("required_for_completion", False)
            or completion_contract.get("requires_verification", False)
            or any(_step_verification_gate(step).get("required", False) for step in step_rows)
        )
        deliverables = _unique_strings(
            _string_list(planning_contract.get("deliverables", []))
            + _string_list(completion_contract.get("deliverables", []))
            + [
                str(getattr(step, "target_state", "") or "").strip()
                for step in step_rows
                if str(getattr(step, "target_state", "") or "").strip()
            ]
        )
        if not deliverables:
            deliverables = _unique_strings(
                list(success_criteria)
                + [
                    str(getattr(step_rows[-1], "description", "") or "").strip()
                ]
                if step_rows
                else list(success_criteria)
            )
        step_constraints = {
            str(getattr(step, "step_id", "") or ""): _dict_or_empty(getattr(step, "constraints", {}))
            for step in step_rows
            if str(getattr(step, "step_id", "") or "").strip() and _dict_or_empty(getattr(step, "constraints", {}))
        }
        constraints = dict(_dict_or_empty(planning_contract.get("constraints", {})))
        execution_limits: Dict[str, Any] = {}
        if int(planning_contract.get("max_steps", max_steps) or max_steps) > 0:
            execution_limits["max_steps"] = int(planning_contract.get("max_steps", max_steps) or max_steps)
        if int(planning_contract.get("max_ticks", max_ticks) or max_ticks) > 0:
            execution_limits["max_ticks"] = int(planning_contract.get("max_ticks", max_ticks) or max_ticks)
        if planning_contract.get("target_reward", target_reward) is not None:
            execution_limits["target_reward"] = planning_contract.get("target_reward", target_reward)
        if execution_limits:
            constraints.setdefault("execution_limits", execution_limits)
        if step_constraints:
            constraints.setdefault("step_constraints", step_constraints)
        step_risk_levels = [
            str(_step_approval_requirement(step).get("risk_level", "") or "").strip().lower()
            for step in step_rows
            if str(_step_approval_requirement(step).get("risk_level", "") or "").strip()
        ]
        risk_level = _max_goal_risk_level(
            planning_contract.get("risk_level", ""),
            approval_contract.get("risk_level", ""),
            *step_risk_levels,
            default="high" if approval_required_functions else "low",
        )
        allowed_tools = _unique_strings(
            _string_list(planning_contract.get("allowed_tools", []))
            + _string_list(approval_contract.get("allowed_tools", []))
            + [
                str(getattr(step, "target_function", "") or "").strip()
                for step in step_rows
                if str(getattr(step, "target_function", "") or "").strip()
            ]
        )
        forbidden_actions = _unique_strings(
            _string_list(planning_contract.get("forbidden_actions", []))
            + _string_list(approval_contract.get("forbidden_actions", []))
            + _string_list(planning_contract.get("blocked_functions", []))
            + _string_list(approval_contract.get("blocked_functions", []))
        )
        approval_points = _dict_list(approval_contract.get("approval_points", []))
        derived_approval_points = [
            {
                "step_id": str(getattr(step, "step_id", "") or ""),
                "title": str(getattr(step, "description", "") or ""),
                "intent": str(getattr(step, "intent", "") or ""),
                "target_function": str(getattr(step, "target_function", "") or ""),
                "reason": str(_step_approval_requirement(step).get("reason", "") or ""),
                "risk_level": str(_step_approval_requirement(step).get("risk_level", "low") or "low"),
            }
            for step in step_rows
            if bool(_step_approval_requirement(step).get("required", False))
        ]
        approval_points = approval_points + [
            point
            for point in derived_approval_points
            if point not in approval_points
        ]
        verification_success_criteria = (
            _string_list(verification_contract.get("success_criteria", success_criteria)) or success_criteria
        )
        verification_plan = _dict_or_empty(verification_contract.get("verification_plan", {}))
        if not verification_plan:
            verification_plan = {
                "required_for_completion": requires_verification,
                "verification_scope": str(
                    verification_contract.get("verification_scope", "task_graph") or "task_graph"
                ),
                "failure_mode": str(
                    verification_contract.get("failure_mode", "block_completion") or "block_completion"
                ),
                "verification_functions": verification_functions,
                "success_criteria": verification_success_criteria,
            }
            step_checks = [
                {
                    "step_id": str(getattr(step, "step_id", "") or ""),
                    "title": str(getattr(step, "description", "") or ""),
                    "required": bool(_step_verification_gate(step).get("required", False)),
                    "mode": str(_step_verification_gate(step).get("mode", "") or ""),
                    "verifier_function": str(_step_verification_gate(step).get("verifier_function", "") or ""),
                    "success_criteria": _string_list(_step_verification_gate(step).get("success_criteria", [])),
                }
                for step in step_rows
                if bool(_step_verification_gate(step).get("required", False))
                or str(_step_verification_gate(step).get("verifier_function", "") or "").strip()
            ]
            if step_checks:
                verification_plan["step_checks"] = step_checks
        assumptions = _unique_strings(
            _string_list(planning_contract.get("assumptions", []))
            + _string_list(verification_contract.get("assumptions", []))
        )
        unknowns = _unique_strings(
            _string_list(planning_contract.get("unknowns", []))
            + _string_list(verification_contract.get("unknowns", []))
        )
        planning_capability_specs = merge_llm_capability_specs(
            planning_contract.get("llm_capability_policies", {}),
            planning_contract.get("llm_capability_policy_entries", []),
        )
        return {
            'goal_id': self._stable_id('goal', plan.goal),
            'title': str(plan.goal or ''),
            'objective': str(plan.goal or ''),
            'success_criteria': success_criteria,
            'abort_criteria': abort_criteria,
            'deliverables': deliverables,
            'constraints': constraints,
            'risk_level': risk_level,
            'allowed_tools': allowed_tools,
            'forbidden_actions': forbidden_actions,
            'approval_points': approval_points,
            'verification_plan': verification_plan,
            'assumptions': assumptions,
            'unknowns': unknowns,
            'priority': 'normal',
            'planning': {
                'max_steps': int(planning_contract.get('max_steps', max_steps) or max_steps),
                'max_ticks': int(planning_contract.get('max_ticks', max_ticks) or max_ticks),
                'target_reward': planning_contract.get('target_reward', target_reward),
                'success_indicator': str(planning_contract.get('success_indicator', success_indicator or '') or ''),
                'allowed_step_intents': allowed_step_intents,
                'blocked_functions': _string_list(planning_contract.get('blocked_functions', [])),
                'replanning_allowed': bool(planning_contract.get('replanning_allowed', True)),
                'llm_route_policies': resolve_llm_route_policies(planning_contract.get('llm_route_policies', {})),
                'llm_capability_policies': planning_capability_specs,
                'llm_capability_policy_entries': resolve_llm_capability_policy_entries(planning_capability_specs),
            },
            'approval': {
                'require_explicit_approval_for_high_risk': bool(
                    approval_contract.get('require_explicit_approval_for_high_risk', True)
                ),
                'required_functions': approval_required_functions,
                'blocked_functions': _string_list(approval_contract.get('blocked_functions', [])),
                'approval_scope': str(approval_contract.get('approval_scope', 'goal_task_binding') or 'goal_task_binding'),
            },
            'verification': {
                'required_for_completion': requires_verification,
                'verification_functions': verification_functions,
                'success_criteria': verification_success_criteria,
                'failure_mode': str(verification_contract.get('failure_mode', 'block_completion') or 'block_completion'),
                'verification_scope': str(verification_contract.get('verification_scope', 'task_graph') or 'task_graph'),
            },
            'completion': {
                'completion_mode': str(
                    completion_contract.get(
                        'completion_mode',
                        'verified_terminal_state' if requires_verification else 'terminal_state',
                    )
                    or ('verified_terminal_state' if requires_verification else 'terminal_state')
                ),
                'requires_all_nodes_terminal': bool(completion_contract.get('requires_all_nodes_terminal', True)),
                'terminal_statuses': _string_list(completion_contract.get('terminal_statuses', ['completed', 'skipped'])) or ['completed', 'skipped'],
                'requires_goal_success_criteria': bool(
                    completion_contract.get('requires_goal_success_criteria', bool(success_criteria))
                ),
                'requires_verification': bool(completion_contract.get('requires_verification', requires_verification)),
            },
            'provenance': {
                'source': 'planner.plan_state',
                'plan_id': str(plan.plan_id or ''),
            },
            'metadata': {
                'revision_count': int(plan.revision_count or 0),
                'created_episode': int(plan.created_episode or 0),
                'created_tick': int(plan.created_tick or 0),
            },
            'contract_version': 'goal_contract/v2',
        }

    def _task_graph_summary(self, plan: Plan, goal_contract: Dict[str, Any]) -> Dict[str, Any]:
        goal_id = str(goal_contract.get('goal_id', '') or '')
        nodes: List[Dict[str, Any]] = []
        approval_nodes: List[Dict[str, Any]] = []
        verifier_nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        active_node_id = ''
        active_control_node_id = ''
        active_plan_statuses = {PlanStatus.ACTIVE, PlanStatus.BLOCKED}
        previous_node_id = ''
        for index, step in enumerate(list(plan.steps or [])):
            node_id = self._task_node_id(plan, step)
            if plan.status in active_plan_statuses and index == int(plan.current_step_index or 0):
                active_node_id = node_id
            if plan.status in active_plan_statuses and index == int(plan.current_step_index or 0):
                node_status = 'active' if step.status == StepStatus.PENDING else step.status.value
            elif (
                plan.status in active_plan_statuses
                and step.status == StepStatus.PENDING
                and index > int(plan.current_step_index or 0)
            ):
                node_status = 'queued'
            else:
                node_status = step.status.value
            node_payload = _step_task_node_payload(
                plan=plan,
                step=step,
                goal_id=goal_id,
                node_id=node_id,
                status=node_status,
            )
            node_payload['dependencies'] = [previous_node_id] if previous_node_id else []
            node_payload['verification_gate'] = resolve_effective_task_verification_gate(goal_contract, node_payload)
            node_payload['approval_requirement'] = resolve_effective_task_approval_requirement(goal_contract, node_payload)
            node_payload['approval_state'] = _step_approval_state(step)
            node_payload['retry_state'] = _step_retry_state(step)

            approval_required = bool(node_payload['approval_requirement'].get('required', False))
            if approval_required:
                approval_node_id = self._approval_node_id(plan, step)
                approval_pending = not bool(node_payload['approval_state'].get('approved', False))
                node_payload['approval_node_id'] = approval_node_id
                approval_nodes.append(
                    {
                        'node_id': approval_node_id,
                        'title': f"approve:{node_payload['title']}",
                        'status': 'completed' if not approval_pending else ('active' if node_id == active_node_id else 'queued'),
                        'goal_id': goal_id,
                        'owner_node_id': node_id,
                        'worker': {
                            'worker_id': 'approval-controller',
                            'worker_type': 'approver',
                            'ownership': 'exclusive',
                        },
                        'approval_state': dict(node_payload['approval_state']),
                        'reason': str(node_payload['approval_requirement'].get('reason', '') or ''),
                    }
                )
                edges.append(
                    {
                        'source_node_id': approval_node_id,
                        'target_node_id': node_id,
                        'edge_type': 'approval_gate',
                    }
                )
                if node_id == active_node_id and approval_pending and not active_control_node_id:
                    active_control_node_id = approval_node_id

            if node_payload['dependencies']:
                edges.append(
                    {
                        'source_node_id': str(node_payload['dependencies'][0] or ''),
                        'target_node_id': node_id,
                        'edge_type': 'sequence',
                    }
                )

            branch_targets: List[Dict[str, Any]] = []
            for branch in list(node_payload.get('branch_targets', []) or []):
                target_step_id = str(branch.get('target_step_id', '') or '').strip()
                branch_target_index = self._step_index_for_id(plan, target_step_id)
                if branch_target_index is None:
                    continue
                target_step = list(plan.steps or [])[branch_target_index]
                target_node_id = self._task_node_id(plan, target_step)
                branch_payload = dict(branch)
                branch_payload['target_node_id'] = target_node_id
                branch_targets.append(branch_payload)
                edges.append(
                    {
                        'source_node_id': node_id,
                        'target_node_id': target_node_id,
                        'edge_type': 'branch',
                        'branch_key': str(branch_payload.get('branch_key', 'default') or 'default'),
                        'condition': str(branch_payload.get('condition', '') or ''),
                    }
                )
            node_payload['branch_targets'] = branch_targets

            rollback_edge = dict(node_payload.get('rollback_edge', {}) or {})
            rollback_target_step_id = str(rollback_edge.get('target_step_id', '') or '').strip()
            if rollback_target_step_id:
                rollback_index = self._step_index_for_id(plan, rollback_target_step_id)
                if rollback_index is not None:
                    rollback_target_step = list(plan.steps or [])[rollback_index]
                    rollback_edge['target_node_id'] = self._task_node_id(plan, rollback_target_step)
                    edges.append(
                        {
                            'source_node_id': node_id,
                            'target_node_id': str(rollback_edge.get('target_node_id', '') or ''),
                            'edge_type': 'rollback',
                            'reason': str(rollback_edge.get('reason', '') or ''),
                        }
                    )
            node_payload['rollback_edge'] = rollback_edge

            verification_required = bool(node_payload['verification_gate'].get('required', False))
            if verification_required:
                verifier_node_id = self._verifier_node_id(plan, step)
                verification_pending = not bool(node_payload['verification_gate'].get('last_verified', False))
                node_payload['verifier_node_id'] = verifier_node_id
                verifier_nodes.append(
                    {
                        'node_id': verifier_node_id,
                        'title': f"verify:{node_payload['title']}",
                        'status': 'completed' if not verification_pending else ('active' if node_id == active_node_id and not active_control_node_id else 'queued'),
                        'goal_id': goal_id,
                        'owner_node_id': node_id,
                        'worker': {
                            'worker_id': 'verification-controller',
                            'worker_type': 'verifier',
                            'ownership': 'exclusive',
                        },
                        'verification_gate': dict(node_payload['verification_gate']),
                    }
                )
                edges.append(
                    {
                        'source_node_id': node_id,
                        'target_node_id': verifier_node_id,
                        'edge_type': 'verification_gate',
                    }
                )
                if (
                    node_id == active_node_id
                    and verification_pending
                    and str(step.status.value) == StepStatus.IN_PROGRESS.value
                    and not active_control_node_id
                ):
                    active_control_node_id = verifier_node_id
            nodes.append(node_payload)
            previous_node_id = node_id
        if not active_node_id and nodes:
            active_node_id = str(nodes[0].get('node_id', '') or '')
        completion_gate = self._evaluate_completion_gate(plan, goal_contract=goal_contract)
        return {
            'graph_id': self._stable_id('task_graph', plan.plan_id, plan.goal),
            'goal_id': goal_id,
            'status': plan.status.value,
            'active_node_id': active_node_id,
            'nodes': nodes,
            'approval_nodes': approval_nodes,
            'verifier_nodes': verifier_nodes,
            'edges': edges,
            'active_control_node_id': active_control_node_id,
            'completion_gate': completion_gate,
            'metadata': {
                'plan_id': str(plan.plan_id or ''),
                'plan_status': plan.status.value,
                'current_step_index': int(plan.current_step_index or 0),
                'revision_count': int(plan.revision_count or 0),
                'graph_revision': int(plan.revision_count or 0),
                'created_episode': int(plan.created_episode or 0),
                'created_tick': int(plan.created_tick or 0),
            },
            'graph_version': 'task_graph/v2',
        }

    def _goal_success_signal_satisfied(self, plan: Plan, goal_contract: Dict[str, Any]) -> bool:
        success_criteria = _string_list(goal_contract.get('success_criteria', []))
        if not success_criteria:
            return True
        exit_criteria = getattr(plan, 'exit_criteria', None)
        if exit_criteria is not None:
            target_reward = getattr(exit_criteria, 'target_reward', None)
            if target_reward is not None:
                try:
                    if float(self._episode_reward) >= float(target_reward):
                        return True
                except (TypeError, ValueError):
                    pass
            success_indicator = str(getattr(exit_criteria, 'success_indicator', '') or '').strip()
            if success_indicator:
                clauses = [
                    str(part or '').strip()
                    for part in success_indicator.split(' or ')
                    if str(part or '').strip()
                ]
                context = {'discovered_functions': list(self._discovered_functions)}
                for clause in clauses or [success_indicator]:
                    try:
                        if exit_criteria._clause_satisfied(clause, context):  # type: ignore[attr-defined]
                            return True
                    except AttributeError:
                        continue
        discovered = {str(item or '').strip().lower() for item in list(self._discovered_functions or []) if str(item or '').strip()}
        return any(str(item or '').strip().lower() in discovered for item in success_criteria)

    def _evaluate_completion_gate(
        self,
        plan: Plan,
        *,
        goal_contract: Optional[Dict[str, Any]] = None,
        candidate_step_index: Optional[int] = None,
        candidate_status: Optional[str] = None,
        candidate_verification_gate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        contract = dict(goal_contract or self._goal_contract_summary(plan))
        completion_contract = _dict_or_empty(contract.get('completion', {}))
        verification_contract = _dict_or_empty(contract.get('verification', {}))
        terminal_statuses = set(
            _string_list(completion_contract.get('terminal_statuses', ['completed', 'skipped']))
            or ['completed', 'skipped']
        )
        verification_functions = _string_list(verification_contract.get('verification_functions', []))
        current_index = int(plan.current_step_index or 0)
        required_verification_node_ids: List[str] = []
        pending_verification_node_ids: List[str] = []
        failed_verification_node_ids: List[str] = []
        verified_node_ids: List[str] = []
        completed_verification_node_ids: List[str] = []
        active_task_completion_ready = True
        all_nodes_terminal = True
        active_task_payload: Optional[Dict[str, Any]] = None

        for index, step in enumerate(list(plan.steps or [])):
            node_id = self._stable_id('task', plan.plan_id, step.step_id, step.description)
            status = str(step.status.value)
            if candidate_step_index is not None and index == int(candidate_step_index):
                status = str(candidate_status or status)
            node_payload = _step_task_node_payload(
                plan=plan,
                step=step,
                goal_id=str(contract.get('goal_id', '') or ''),
                node_id=node_id,
                status=status,
                verification_gate=(
                    candidate_verification_gate
                    if candidate_step_index is not None and index == int(candidate_step_index) and isinstance(candidate_verification_gate, dict)
                    else None
                ),
            )
            verification_gate = resolve_effective_task_verification_gate(contract, node_payload)
            if candidate_step_index is not None and index == int(candidate_step_index) and isinstance(candidate_verification_gate, dict):
                verification_gate = resolve_effective_task_verification_gate(contract, node_payload)
            if index == current_index:
                active_task_payload = dict(node_payload)
            terminal = status in terminal_statuses
            all_nodes_terminal = all_nodes_terminal and terminal
            if verification_gate.get('required', False):
                required_verification_node_ids.append(node_id)
                verdict = str(verification_gate.get('last_verdict', 'pending') or 'pending')
                if verdict == 'failed':
                    failed_verification_node_ids.append(node_id)
                elif bool(verification_gate.get('last_verified', False)):
                    verified_node_ids.append(node_id)
                else:
                    pending_verification_node_ids.append(node_id)
            if terminal and _step_is_verification_step(step, verification_functions):
                if not verification_gate.get('required', False) or bool(verification_gate.get('last_verified', False)):
                    completed_verification_node_ids.append(node_id)
            if index == current_index:
                active_task_completion_ready = (
                    (not bool(verification_gate.get('required', False)))
                    or bool(verification_gate.get('last_verified', False))
                )

        requires_verification = bool(
            completion_contract.get('requires_verification', False)
            or verification_contract.get('required_for_completion', False)
        )
        verification_ready = True
        if requires_verification:
            verification_ready = (
                not pending_verification_node_ids
                and not failed_verification_node_ids
                and (bool(required_verification_node_ids) or bool(completed_verification_node_ids))
            )
        requires_all_nodes_terminal = bool(completion_contract.get('requires_all_nodes_terminal', True))
        requires_goal_success_criteria = bool(completion_contract.get('requires_goal_success_criteria', False))
        goal_success_satisfied = self._goal_success_signal_satisfied(plan, contract)
        blocked_reasons: List[str] = []
        if requires_all_nodes_terminal and not all_nodes_terminal:
            blocked_reasons.append('awaiting_terminal_nodes')
        if requires_verification and failed_verification_node_ids:
            blocked_reasons.append('verification_failed')
        elif requires_verification and not verification_ready:
            blocked_reasons.append('verification_incomplete')
        if requires_goal_success_criteria and not goal_success_satisfied:
            blocked_reasons.append('goal_success_criteria_unsatisfied')
        goal_completion_ready = not blocked_reasons
        if current_index >= len(list(plan.steps or [])):
            active_task_completion_ready = goal_completion_ready
        gate = {
            'active_task_completion_ready': active_task_completion_ready,
            'goal_completion_ready': goal_completion_ready,
            'verification_ready': verification_ready,
            'goal_success_satisfied': goal_success_satisfied,
            'all_nodes_terminal': all_nodes_terminal,
            'requires_verification': requires_verification,
            'requires_goal_success_criteria': requires_goal_success_criteria,
            'requires_all_nodes_terminal': requires_all_nodes_terminal,
            'required_verification_node_ids': required_verification_node_ids,
            'pending_verification_node_ids': pending_verification_node_ids,
            'failed_verification_node_ids': failed_verification_node_ids,
            'verified_node_ids': verified_node_ids,
            'completed_verification_node_ids': completed_verification_node_ids,
            'blocked_reasons': blocked_reasons,
        }
        gate['verifier_authority'] = derive_verifier_authority_snapshot(
            goal_contract=contract,
            active_task=active_task_payload,
            completion_gate=gate,
        )
        return gate
    
    def advance_step(self) -> bool:
        """
        推进到下一步.
        
        Returns:
            True if advanced, False if no more steps
        """
        if not self.complete_current_step():
            return False
        return bool(self._current_plan and self._current_plan.status != PlanStatus.COMPLETED)

    def _update_step_retry_state(self, step: PlanStep, **updates: Any) -> None:
        retry_state = _step_retry_state(step)
        explicit = _dict_or_empty(getattr(step, 'retry_state', {}))
        explicit.update(retry_state)
        explicit.update(updates)
        step.retry_state = explicit

    def _update_step_approval_state(self, step: PlanStep, **updates: Any) -> None:
        approval_state = _step_approval_state(step)
        explicit = _dict_or_empty(getattr(step, 'approval_state', {}))
        explicit.update(approval_state)
        explicit.update(updates)
        step.approval_state = explicit

    def _update_step_governance_memory(self, step: PlanStep, **updates: Any) -> None:
        memory = _step_governance_memory(step)
        explicit = _dict_or_empty(getattr(step, 'governance_memory', {}))
        explicit.update(memory)
        for key, value in updates.items():
            if key.endswith("_events"):
                explicit[key] = _governance_event_rows(value)
            elif isinstance(value, dict):
                explicit[key] = dict(value)
            else:
                explicit[key] = value
        step.governance_memory = explicit

    def _append_step_governance_event(
        self,
        step: PlanStep,
        *,
        key: str,
        event: Dict[str, Any],
        max_events: int = 6,
        **updates: Any,
    ) -> None:
        memory = _step_governance_memory(step)
        rows = _governance_event_rows(memory.get(key, []))
        rows.append(dict(event))
        trimmed = rows[-max(1, int(max_events)) :]
        payload = {str(key): trimmed}
        payload.update(updates)
        self._update_step_governance_memory(step, **payload)

    def _resolve_branch_target_index(self, step: PlanStep, branch_key: str = '') -> Optional[int]:
        if not self._current_plan:
            return None
        clean_key = str(branch_key or '').strip()
        branch_targets = _step_branch_targets(step)
        if not branch_targets:
            return None
        selected: Optional[Dict[str, Any]] = None
        if clean_key:
            for branch in branch_targets:
                if str(branch.get('branch_key', '') or '').strip() == clean_key:
                    selected = branch
                    break
        if selected is None:
            for branch in branch_targets:
                if str(branch.get('branch_key', 'default') or 'default').strip() == 'default':
                    selected = branch
                    break
        if selected is None:
            selected = branch_targets[0]
        return self._step_index_for_id(self._current_plan, str(selected.get('target_step_id', '') or ''))

    def _set_active_step_index(self, target_index: int) -> None:
        if not self._current_plan:
            return
        self._current_plan.current_step_index = max(0, int(target_index))
        if self._current_plan.current_step_index >= len(self._current_plan.steps):
            self._current_plan.status = PlanStatus.COMPLETED
        else:
            self._current_plan.status = PlanStatus.ACTIVE

    def record_current_step_approval(
        self,
        *,
        step_id: Optional[str] = None,
        approved: bool = True,
        approval_grant_id: str = '',
        approval_sources: Optional[List[str]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        approved_by: str = '',
    ) -> bool:
        if not self._current_plan or not self.current_step:
            return False
        if step_id and str(self.current_step.step_id or '') != str(step_id or ''):
            return False
        self._update_step_approval_state(
            self.current_step,
            approved=bool(approved),
            pending=not bool(approved),
            approval_grant_id=str(approval_grant_id or ''),
            approval_sources=[str(item) for item in list(approval_sources or []) if str(item)],
            approved_by=str(approved_by or ''),
            evidence=dict(evidence) if isinstance(evidence, dict) else {},
        )
        self._append_step_governance_event(
            self.current_step,
            key='approval_events',
            event=_memory_event(
                'approval',
                approved=bool(approved),
                approval_grant_id=str(approval_grant_id or ''),
                approval_sources=[str(item) for item in list(approval_sources or []) if str(item)],
                approved_by=str(approved_by or ''),
                evidence_keys=sorted((dict(evidence) if isinstance(evidence, dict) else {}).keys()),
            ),
            last_transition='approve',
            approval_pending=not bool(approved),
            last_approval_required=bool(_step_approval_requirement(self.current_step).get('required', False)),
            last_approval_grant_id=str(approval_grant_id or ''),
        )
        if self._current_plan.status == PlanStatus.BLOCKED and bool(approved):
            self._current_plan.status = PlanStatus.ACTIVE
        return True

    def mark_current_step_in_progress(self, *, step_id: Optional[str] = None) -> bool:
        """正式标记当前步骤进入执行态，并增加尝试次数。"""
        if not self._current_plan or not self.current_step:
            return False
        step = self.current_step
        if step_id and str(step.step_id or '') != str(step_id or ''):
            return False
        if step.status == StepStatus.PENDING:
            step.mark_in_progress()
        elif step.status != StepStatus.IN_PROGRESS:
            return False
        self._update_step_retry_state(step, retry_pending=False, last_transition='start')
        self._update_step_governance_memory(step, last_transition='start', retry_pending=False)
        if self._current_plan.status != PlanStatus.COMPLETED:
            self._current_plan.status = PlanStatus.ACTIVE
        return True

    def record_current_step_verification(
        self,
        *,
        step_id: Optional[str] = None,
        verified: bool = True,
        evidence: Optional[Dict[str, Any]] = None,
        verifier_function: str = '',
    ) -> bool:
        """记录当前步骤的验证结果，不直接推进完成态。"""
        if not self._current_plan or not self.current_step:
            return False
        if step_id and str(self.current_step.step_id or '') != str(step_id or ''):
            return False
        goal_contract = self._goal_contract_summary(self._current_plan)
        explicit_gate = _dict_or_empty(getattr(self.current_step, 'verification_gate', {}))
        node_payload = _step_task_node_payload(
            plan=self._current_plan,
            step=self.current_step,
            goal_id=str(goal_contract.get('goal_id', '') or ''),
            node_id=self._stable_id('task', self._current_plan.plan_id, self.current_step.step_id, self.current_step.description),
            status=str(self.current_step.status.value),
            verification_gate=explicit_gate,
        )
        gate = resolve_effective_task_verification_gate(goal_contract, node_payload)
        explicit_gate.update(gate)
        explicit_gate['last_verified'] = bool(verified)
        explicit_gate['last_verdict'] = 'passed' if bool(verified) else 'failed'
        if isinstance(evidence, dict) and evidence:
            explicit_gate['evidence'] = dict(evidence)
        if verifier_function:
            explicit_gate['verifier_function'] = str(verifier_function)
        self.current_step.verification_gate = explicit_gate
        self._update_step_retry_state(self.current_step, last_transition='verify')
        self._append_step_governance_event(
            self.current_step,
            key='verification_events',
            event=_memory_event(
                'verification',
                verified=bool(verified),
                verdict='passed' if bool(verified) else 'failed',
                verifier_function=str(verifier_function or explicit_gate.get('verifier_function', '') or ''),
                evidence_keys=sorted((dict(evidence) if isinstance(evidence, dict) else {}).keys()),
            ),
            last_transition='verify',
            last_verification_required=bool(explicit_gate.get('required', False)),
            last_verification_verdict='passed' if bool(verified) else 'failed',
        )
        if self._current_plan.status == PlanStatus.BLOCKED and bool(verified):
            self._current_plan.status = PlanStatus.ACTIVE
        return True

    def complete_current_step(
        self,
        *,
        result: str = 'success',
        step_id: Optional[str] = None,
        verification_passed: Optional[bool] = None,
        verification_evidence: Optional[Dict[str, Any]] = None,
        branch_key: str = '',
    ) -> bool:
        """正式完成当前步骤，并推进 current_step_index。"""
        if not self._current_plan or not self.current_step:
            return False
        if step_id and str(self.current_step.step_id or '') != str(step_id or ''):
            return False
        goal_contract = self._goal_contract_summary(self._current_plan)
        explicit_gate = _dict_or_empty(getattr(self.current_step, 'verification_gate', {}))
        if verification_passed is not None or verification_evidence:
            node_payload = _step_task_node_payload(
                plan=self._current_plan,
                step=self.current_step,
                goal_id=str(goal_contract.get('goal_id', '') or ''),
                node_id=self._stable_id('task', self._current_plan.plan_id, self.current_step.step_id, self.current_step.description),
                status=str(self.current_step.status.value),
                verification_gate=explicit_gate,
            )
            verification_gate = resolve_effective_task_verification_gate(goal_contract, node_payload)
            explicit_gate.update(verification_gate)
            if verification_passed is not None:
                explicit_gate['last_verified'] = bool(verification_passed)
                explicit_gate['last_verdict'] = 'passed' if bool(verification_passed) else 'failed'
            if isinstance(verification_evidence, dict) and verification_evidence:
                explicit_gate['evidence'] = dict(verification_evidence)
            self.current_step.verification_gate = explicit_gate
        verification_gate = resolve_effective_task_verification_gate(
            goal_contract,
            _step_task_node_payload(
                plan=self._current_plan,
                step=self.current_step,
                goal_id=str(goal_contract.get('goal_id', '') or ''),
                node_id=self._stable_id('task', self._current_plan.plan_id, self.current_step.step_id, self.current_step.description),
                status=str(self.current_step.status.value),
                verification_gate=_dict_or_empty(getattr(self.current_step, 'verification_gate', {})),
            ),
        )
        if verification_gate.get('required', False) and not bool(verification_gate.get('last_verified', False)):
            return False
        if self.current_step.status == StepStatus.PENDING:
            self.current_step.mark_in_progress()
        elif self.current_step.status in {StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED}:
            return False
        is_final_step = int(self._current_plan.current_step_index or 0) >= (len(self._current_plan.steps) - 1)
        if is_final_step:
            completion_gate = self._evaluate_completion_gate(
                self._current_plan,
                goal_contract=goal_contract,
                candidate_step_index=int(self._current_plan.current_step_index or 0),
                candidate_status=StepStatus.COMPLETED.value,
                candidate_verification_gate=verification_gate,
            )
            if not completion_gate.get('goal_completion_ready', False):
                self._current_plan.status = PlanStatus.BLOCKED
                return False
        self.current_step.mark_completed(result=result)
        self._update_step_retry_state(self.current_step, retry_pending=False, last_transition='complete')
        self._update_step_governance_memory(
            self.current_step,
            last_transition='complete',
            retry_pending=False,
            last_verification_required=bool(verification_gate.get('required', False)),
            last_verification_verdict=str(verification_gate.get('last_verdict', 'not_required') or 'not_required'),
        )
        branch_target_index = self._resolve_branch_target_index(self.current_step, branch_key=branch_key)
        if branch_target_index is None:
            branch_target_index = int(self._current_plan.current_step_index or 0) + 1
        self._set_active_step_index(branch_target_index)
        return True

    def fail_current_step(
        self,
        *,
        reason: str = '',
        step_id: Optional[str] = None,
        block_plan: bool = False,
    ) -> bool:
        """正式标记当前步骤失败；可选地把计划置为 blocked。"""
        if not self._current_plan or not self.current_step:
            return False
        step = self.current_step
        if step_id and str(step.step_id or '') != str(step_id or ''):
            return False
        if step.status == StepStatus.PENDING:
            step.mark_in_progress()
        elif step.status in {StepStatus.COMPLETED, StepStatus.SKIPPED}:
            return False
        step.mark_failed(reason=reason)
        retry_policy = _step_retry_policy(step)
        rollback_edge = _step_rollback_edge(step)
        remaining_attempts = max(0, int(retry_policy.get('max_attempts', 1) or 1) - int(step.execution_attempts or 0))
        self._update_step_retry_state(
            step,
            retry_pending=bool(remaining_attempts > 0 and not block_plan),
            last_failure_reason=str(reason or ''),
            last_transition='fail',
        )
        rollback_target_step_id = str(rollback_edge.get('target_step_id', '') or '').strip()
        rollback_triggered = False
        if block_plan:
            self._current_plan.mark_blocked(reason=reason)
        elif (
            str(retry_policy.get('on_failure', '') or '') == 'rollback'
            and str(rollback_edge.get('target_step_id', '') or '').strip()
        ):
            rollback_index = self._step_index_for_id(self._current_plan, str(rollback_edge.get('target_step_id', '') or ''))
            if rollback_index is not None:
                rollback_step = self._current_plan.steps[rollback_index]
                if rollback_step.status == StepStatus.COMPLETED:
                    rollback_step.status = StepStatus.PENDING
                self._set_active_step_index(rollback_index)
                rollback_triggered = True
            else:
                self._current_plan.mark_blocked(reason=reason or 'rollback_target_missing')
        elif remaining_attempts > 0 and str(retry_policy.get('on_failure', '') or '') in {'retry', 'revise', 'rollback'}:
            step.status = StepStatus.PENDING
            self._current_plan.status = PlanStatus.ACTIVE
        elif self._current_plan.status != PlanStatus.COMPLETED:
            if str(retry_policy.get('on_failure', '') or '') == 'block':
                self._current_plan.mark_blocked(reason=reason)
            else:
                self._current_plan.status = PlanStatus.ACTIVE
        self._append_step_governance_event(
            step,
            key='failure_events',
            event=_memory_event(
                'failure',
                reason=str(reason or ''),
                block_plan=bool(block_plan),
                retry_mode=str(retry_policy.get('on_failure', '') or ''),
                remaining_attempts=int(remaining_attempts),
                rollback_target_step_id=rollback_target_step_id,
                rollback_triggered=bool(rollback_triggered),
            ),
            last_transition='fail',
            last_failure_reason=str(reason or ''),
            retry_pending=bool(remaining_attempts > 0 and not block_plan),
            rollback_ready=bool(
                rollback_target_step_id and str(retry_policy.get('on_failure', '') or '') == 'rollback'
            ),
            rollback_target_step_id=rollback_target_step_id,
        )
        if rollback_target_step_id:
            self._append_step_governance_event(
                step,
                key='rollback_events',
                event=_memory_event(
                    'rollback',
                    reason=str(rollback_edge.get('reason', '') or ''),
                    target_step_id=rollback_target_step_id,
                    triggered=bool(rollback_triggered),
                    failure_reason=str(reason or ''),
                ),
                last_transition='rollback' if rollback_triggered else 'fail',
                rollback_ready=bool(
                    rollback_target_step_id and str(retry_policy.get('on_failure', '') or '') == 'rollback'
                ),
                rollback_target_step_id=rollback_target_step_id,
            )
        return True

    def skip_current_step(self, *, reason: str = '', step_id: Optional[str] = None) -> bool:
        """正式跳过当前步骤并推进到下一步。"""
        if not self._current_plan or not self.current_step:
            return False
        if step_id and str(self.current_step.step_id or '') != str(step_id or ''):
            return False
        is_final_step = int(self._current_plan.current_step_index or 0) >= (len(self._current_plan.steps) - 1)
        if is_final_step:
            goal_contract = self._goal_contract_summary(self._current_plan)
            verification_gate = resolve_effective_task_verification_gate(
                goal_contract,
                _step_task_node_payload(
                    plan=self._current_plan,
                    step=self.current_step,
                    goal_id=str(goal_contract.get('goal_id', '') or ''),
                    node_id=self._stable_id('task', self._current_plan.plan_id, self.current_step.step_id, self.current_step.description),
                    status=str(self.current_step.status.value),
                    verification_gate=_dict_or_empty(getattr(self.current_step, 'verification_gate', {})),
                ),
            )
            completion_gate = self._evaluate_completion_gate(
                self._current_plan,
                goal_contract=goal_contract,
                candidate_step_index=int(self._current_plan.current_step_index or 0),
                candidate_status=StepStatus.SKIPPED.value,
                candidate_verification_gate=verification_gate,
            )
            if not completion_gate.get('goal_completion_ready', False):
                self._current_plan.status = PlanStatus.BLOCKED
                return False
        self.current_step.mark_skipped(reason=reason)
        self._update_step_retry_state(self.current_step, retry_pending=False, last_transition='skip')
        self._update_step_governance_memory(
            self.current_step,
            last_transition='skip',
            retry_pending=False,
        )
        self._set_active_step_index(int(self._current_plan.current_step_index or 0) + 1)
        return True

    def apply_step_transition(self, transition: Dict[str, Any]) -> bool:
        """应用结构化 step transition，避免外部直接改写 current_step。"""
        if not isinstance(transition, dict):
            return False
        event = str(transition.get('event') or transition.get('kind') or '').strip().lower()
        step_id = str(transition.get('step_id', '') or '') or None
        if event in {'start', 'attempt', 'mark_in_progress'}:
            return self.mark_current_step_in_progress(step_id=step_id)
        if event in {'verify', 'mark_verified', 'verification_result'}:
            verified = transition.get('verified')
            if verified is None:
                verified = transition.get('verification_passed', True)
            evidence = transition.get('verification_evidence', {})
            applied = self.record_current_step_verification(
                step_id=step_id,
                verified=bool(verified),
                evidence=dict(evidence) if isinstance(evidence, dict) else None,
                verifier_function=str(transition.get('verifier_function', '') or ''),
            )
            if applied and bool(transition.get('consume_verifier_authority', False)):
                self._consume_verifier_authority_transition(
                    step_id=step_id,
                    verified=bool(verified),
                    reason=str(
                        transition.get('verification_failure_reason', '')
                        or transition.get('reason', '')
                        or ''
                    ),
                    block_plan=bool(transition.get('block_plan', False)),
                )
            return applied
        if event in {'approve', 'mark_approved', 'approval_result'}:
            return self.record_current_step_approval(
                step_id=step_id,
                approved=bool(transition.get('approved', True)),
                approval_grant_id=str(transition.get('approval_grant_id', '') or ''),
                approval_sources=[
                    str(item) for item in list(transition.get('approval_sources', []) or []) if str(item)
                ],
                evidence=dict(transition.get('approval_evidence', {})) if isinstance(transition.get('approval_evidence', {}), dict) else None,
                approved_by=str(transition.get('approved_by', '') or ''),
            )
        if event in {'complete', 'advance', 'mark_completed'}:
            verification_passed = transition.get('verification_passed')
            if verification_passed is None and 'verified' in transition:
                verification_passed = transition.get('verified')
            if verification_passed is None and 'verification_ok' in transition:
                verification_passed = transition.get('verification_ok')
            verification_evidence = transition.get('verification_evidence', {})
            return self.complete_current_step(
                step_id=step_id,
                result=str(transition.get('result', 'success') or 'success'),
                verification_passed=verification_passed if verification_passed is None else bool(verification_passed),
                verification_evidence=dict(verification_evidence) if isinstance(verification_evidence, dict) else None,
                branch_key=str(transition.get('branch_key', transition.get('branch', '')) or ''),
            )
        if event in {'fail', 'mark_failed'}:
            return self.fail_current_step(
                step_id=step_id,
                reason=str(transition.get('reason', '') or ''),
                block_plan=bool(transition.get('block_plan', False)),
            )
        if event in {'skip', 'mark_skipped'}:
            return self.skip_current_step(
                step_id=step_id,
                reason=str(transition.get('reason', '') or ''),
            )
        return False

    def _consume_verifier_authority_transition(
        self,
        *,
        step_id: Optional[str],
        verified: bool,
        reason: str = '',
        block_plan: bool = False,
    ) -> bool:
        if verified or not self._current_plan or not self.current_step:
            return False
        if step_id and str(self.current_step.step_id or '') != str(step_id or ''):
            return False
        summary = self.get_plan_summary()
        verifier_authority = _task_contract_verifier_authority(summary.get('task_contract', {}))
        if not verifier_authority:
            completion_gate = (
                dict(summary.get('completion_gate', {}) or {})
                if isinstance(summary.get('completion_gate', {}), dict)
                else {}
            )
            verifier_authority = (
                dict(completion_gate.get('verifier_authority', {}) or {})
                if isinstance(completion_gate.get('verifier_authority', {}), dict)
                else {}
            )
        if str(verifier_authority.get('decision', '') or '') != 'block_completion':
            return False
        failure_reason = str(
            reason
            or verifier_authority.get('rollback_reason', '')
            or verifier_authority.get('blocked_reason', '')
            or 'verification_failed'
        )
        return self.fail_current_step(
            step_id=step_id,
            reason=failure_reason,
            block_plan=block_plan,
        )

    def apply_step_transitions(self, transitions: List[Dict[str, Any]]) -> int:
        """批量应用 transition，返回成功应用的数量。"""
        applied = 0
        for transition in list(transitions or []):
            if self.apply_step_transition(transition):
                applied += 1
        return applied
    
    def check_exit(self) -> bool:
        """
        检查退出条件.
        
        Returns:
            True if exit criteria met
        """
        if not self._current_plan:
            return True  # 无计划 = 退出
        
        return self._current_plan.exit_criteria.is_satisfied(
            current_step=self._current_plan.current_step_index,
            current_ticks=self._tick_count,
            current_reward=self._episode_reward,
            context={
                'discovered_functions': self._discovered_functions,
            },
        )
    
    def get_plan_summary(self) -> Dict[str, Any]:
        """获取计划摘要"""
        if not self._current_plan:
            return {
                'has_plan': False,
                'status': 'no_plan',
            }
        
        plan = self._current_plan
        goal_contract = self._goal_contract_summary(plan)
        task_graph = self._task_graph_summary(plan, goal_contract)
        completion_gate = dict(task_graph.get('completion_gate', {}) or {})
        active_task_node = resolve_task_graph_active_task_payload(task_graph)
        canonical_task_intent = str(
            _dict_or_empty(active_task_node.get('metadata', {})).get('intent', '')
            or active_task_node.get('intent', '')
            or ''
        )
        canonical_task_description = str(active_task_node.get('title', '') or '')
        current_step_node_id = (
            self._task_node_id(plan, plan.current_step)
            if plan.current_step is not None
            else ''
        )
        current_step_description_mirror = plan.current_step.description if plan.current_step else None
        current_step_intent_mirror = plan.current_step.intent if plan.current_step else None
        current_step_description = canonical_task_description or current_step_description_mirror
        current_step_intent = canonical_task_intent or current_step_intent_mirror
        active_task_pre_aligned = (
            str(active_task_node.get('node_id', '') or '') == str(current_step_node_id or '')
            and canonical_task_description == str(current_step_description_mirror or '')
            and canonical_task_intent == str(current_step_intent_mirror or '')
        )
        authority_snapshot = {
            'source': 'planner_native',
            'integrity': 'complete' if goal_contract and task_graph else 'incomplete',
            'warnings': [] if goal_contract and task_graph else ['planner_native_goal_task_authority_incomplete'],
        }
        task_contract = build_task_contract(
            goal_contract=goal_contract,
            task_graph=task_graph,
            task_node=active_task_node,
            completion_gate=completion_gate,
            authority_snapshot=authority_snapshot,
        ).to_dict()
        canonical_verifier_authority = _task_contract_verifier_authority(task_contract)
        completion_gate_verifier_authority = _dict_or_empty(completion_gate.get('verifier_authority', {}))
        completion_gate_pre_aligned = (
            _payload_signature(completion_gate_verifier_authority) == _payload_signature(canonical_verifier_authority)
        )
        if canonical_verifier_authority:
            completion_gate['verifier_authority'] = dict(canonical_verifier_authority)
        execution_authority = resolve_goal_contract_authority(
            goal_contract=goal_contract,
            task_graph=task_graph,
            active_task=active_task_node,
            completion_gate=completion_gate,
        )
        execution_authority_verifier_authority = _dict_or_empty(execution_authority.get('verifier_authority', {}))
        execution_authority_pre_aligned = (
            _payload_signature(execution_authority_verifier_authority) == _payload_signature(canonical_verifier_authority)
        )
        if canonical_verifier_authority:
            execution_authority['verifier_authority'] = dict(canonical_verifier_authority)
            completion_snapshot = _dict_or_empty(execution_authority.get('completion', {}))
            execution_completion_gate = _dict_or_empty(completion_snapshot.get('completion_gate', {}))
            execution_completion_gate['verifier_authority'] = dict(canonical_verifier_authority)
            completion_snapshot['completion_gate'] = execution_completion_gate
            execution_authority['completion'] = completion_snapshot
        completion_gate_aligned = (
            _payload_signature(_dict_or_empty(completion_gate.get('verifier_authority', {})))
            == _payload_signature(canonical_verifier_authority)
        )
        execution_authority_aligned = (
            _payload_signature(_dict_or_empty(execution_authority.get('verifier_authority', {})))
            == _payload_signature(canonical_verifier_authority)
        )
        authority_views = {
            'task_contract': {
                'role': 'canonical',
                'source': str(authority_snapshot.get('source', '') or ''),
                'integrity': str(authority_snapshot.get('integrity', '') or ''),
                'warnings': list(authority_snapshot.get('warnings', []) or []),
            },
            'completion_gate.verifier_authority': {
                'role': 'mirror',
                'source': 'task_contract',
                'aligned': bool(completion_gate_aligned),
                'warnings': [] if completion_gate_pre_aligned else ['verifier_authority_mirror_normalized'],
            },
            'execution_authority.verifier_authority': {
                'role': 'mirror',
                'source': 'task_contract',
                'aligned': bool(execution_authority_aligned),
                'warnings': [] if execution_authority_pre_aligned else ['verifier_authority_mirror_normalized'],
            },
        }
        task_graph_snapshot = {
            'source': 'task_graph',
            'integrity': 'complete' if task_graph else 'incomplete',
            'graph_ref': str(task_graph.get('graph_id', '') or ''),
            'active_node_ref': str(task_graph.get('active_node_id', '') or ''),
            'active_control_node_ref': str(task_graph.get('active_control_node_id', '') or ''),
            'status': str(task_graph.get('status', '') or ''),
            'warnings': [] if active_task_pre_aligned else ['active_task_mirror_normalized'],
        }
        task_graph_views = {
            'task_graph': {
                'role': 'canonical',
                'source': 'planner.plan_state',
                'integrity': str(task_graph_snapshot.get('integrity', '') or ''),
                'warnings': list(task_graph_snapshot.get('warnings', []) or []),
            },
            'active_task_node': {
                'role': 'mirror',
                'source': 'task_graph.active_node_id',
                'aligned': True,
                'warnings': [],
            },
            'current_step_description': {
                'role': 'mirror',
                'source': 'task_graph.active_node_id',
                'aligned': bool(canonical_task_description == str(current_step_description or '')),
                'warnings': []
                if canonical_task_description == str(current_step_description or '')
                else ['current_step_description_normalized'],
            },
            'current_step_intent': {
                'role': 'mirror',
                'source': 'task_graph.active_node_id',
                'aligned': bool(canonical_task_intent == str(current_step_intent or '')),
                'warnings': []
                if canonical_task_intent == str(current_step_intent or '')
                else ['current_step_intent_normalized'],
            },
        }
        return {
            'has_plan': True,
            'plan_id': plan.plan_id,
            'goal': plan.goal,
            'status': plan.status.value,
            'current_step_index': plan.current_step_index,
            'total_steps': len(plan.steps),
            'remaining_steps': plan.remaining_steps,
            'revision_count': plan.revision_count,
            'current_step_description': current_step_description,
            'current_step_intent': current_step_intent,
            'goal_contract': goal_contract,
            'task_graph': task_graph,
            'active_task_node': active_task_node,
            'task_graph_snapshot': task_graph_snapshot,
            'task_graph_views': task_graph_views,
            'task_contract': task_contract,
            'authority_snapshot': authority_snapshot,
            'authority_views': authority_views,
            'completion_gate': completion_gate,
            'execution_authority': execution_authority,
        }
    
    def get_intent_for_step(self) -> Optional[str]:
        """
        获取当前步骤的意图.
        
        Returns:
            intent string or None
        """
        if self._current_plan and self._current_plan.current_step:
            return self._current_plan.current_step.intent
        return None
    
    def get_target_function_for_step(self) -> Optional[str]:
        """
        获取当前步骤的目标函数.
        
        Returns:
            function name or None
        """
        if self._current_plan and self._current_plan.current_step:
            return self._current_plan.current_step.target_function
        return None
