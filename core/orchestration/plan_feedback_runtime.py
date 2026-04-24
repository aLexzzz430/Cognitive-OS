from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from core.conos_kernel import build_verification_result
from core.orchestration.goal_task_control import resolve_plan_summary_active_task
from core.orchestration.verifier_runtime import build_verifier_runtime


def plan_step_feedback_reference(plan_state: Any, step_id: Optional[str] = None) -> Dict[str, Any]:
    if plan_state is None or not bool(getattr(plan_state, "has_plan", False)):
        return {}

    summary_payload = {}
    if callable(getattr(plan_state, "get_plan_summary", None)):
        summary = plan_state.get_plan_summary()
        summary_payload = dict(summary or {}) if isinstance(summary, dict) else {}
    if summary_payload:
        task_graph = (
            dict(summary_payload.get("task_graph", {}) or {})
            if isinstance(summary_payload.get("task_graph", {}), dict)
            else {}
        )
        goal_contract = (
            dict(summary_payload.get("goal_contract", {}) or {})
            if isinstance(summary_payload.get("goal_contract", {}), dict)
            else {}
        )
        nodes = [
            dict(node)
            for node in list(task_graph.get("nodes", []) or [])
            if isinstance(node, dict)
        ]
        clean_step_id = str(step_id or "").strip()
        target_task_node: Dict[str, Any] = {}
        if clean_step_id:
            target_task_node = next(
                (
                    dict(node)
                    for node in nodes
                    if str(dict(node.get("provenance", {}) or {}).get("step_id", "") or "").strip() == clean_step_id
                ),
                {},
            )
        if not target_task_node:
            target_task_node = resolve_plan_summary_active_task(summary_payload)
        if target_task_node:
            verification_gate = (
                dict(target_task_node.get("verification_gate", {}) or {})
                if isinstance(target_task_node.get("verification_gate", {}), dict)
                else {}
            )
            provenance = (
                dict(target_task_node.get("provenance", {}) or {})
                if isinstance(target_task_node.get("provenance", {}), dict)
                else {}
            )
            return {
                "goal_id": str(
                    goal_contract.get("goal_id", "")
                    or task_graph.get("goal_id", "")
                    or target_task_node.get("goal_id", "")
                    or ""
                ),
                "step_id": str(provenance.get("step_id", "") or clean_step_id),
                "task_node_id": str(target_task_node.get("node_id", "") or ""),
                "step_title": str(target_task_node.get("title", "") or ""),
                "verifier_function": str(verification_gate.get("verifier_function", "") or ""),
                "verification_required": bool(verification_gate.get("required", False)),
            }
    plan = getattr(plan_state, "current_plan", None)
    if plan is None:
        return {}
    current_step = getattr(plan_state, "current_step", None)
    target_step = current_step
    clean_step_id = str(step_id or "").strip()
    if clean_step_id and callable(getattr(plan_state, "_step_index_for_id", None)):
        step_index = plan_state._step_index_for_id(plan, clean_step_id)  # type: ignore[attr-defined]
        if step_index is not None:
            steps = list(getattr(plan, "steps", []) or [])
            if 0 <= int(step_index) < len(steps):
                target_step = steps[int(step_index)]
    if target_step is None:
        return {}
    goal_contract = {}
    if callable(getattr(plan_state, "_goal_contract_summary", None)):
        goal_contract = dict(plan_state._goal_contract_summary(plan) or {})  # type: ignore[attr-defined]
    task_node_id = ""
    if callable(getattr(plan_state, "_task_node_id", None)):
        task_node_id = str(plan_state._task_node_id(plan, target_step) or "")  # type: ignore[attr-defined]
    verification_gate = (
        dict(getattr(target_step, "verification_gate", {}) or {})
        if isinstance(getattr(target_step, "verification_gate", {}), dict)
        else {}
    )
    return {
        "goal_id": str(goal_contract.get("goal_id", "") or ""),
        "step_id": str(getattr(target_step, "step_id", "") or ""),
        "task_node_id": task_node_id,
        "step_title": str(getattr(target_step, "description", "") or ""),
        "verifier_function": str(verification_gate.get("verifier_function", "") or ""),
        "verification_required": bool(verification_gate.get("required", False)),
    }


def verification_feedback_from_transition(transition: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(transition, dict):
        return None
    event = str(transition.get("event") or transition.get("kind") or "").strip().lower()
    feedback_kind = ""
    verified: Optional[bool] = None
    if event in {"verify", "mark_verified", "verification_result"}:
        verified = transition.get("verified")
        if verified is None:
            verified = transition.get("verification_passed", True)
        feedback_kind = "verification_result"
    elif event in {"complete", "advance", "mark_completed"}:
        if "verification_passed" in transition or "verified" in transition or "verification_ok" in transition:
            verified = transition.get("verification_passed")
            if verified is None:
                verified = transition.get("verified")
            if verified is None:
                verified = transition.get("verification_ok")
            feedback_kind = "completion_verification"
        elif isinstance(transition.get("verification_evidence", {}), dict) and transition.get("verification_evidence", {}):
            verified = True
            feedback_kind = "completion_verification"
    if verified is None:
        return None
    evidence = transition.get("verification_evidence", {})
    return {
        "verified": bool(verified),
        "feedback_kind": feedback_kind or event,
        "verifier_function": str(transition.get("verifier_function", "") or ""),
        "evidence": dict(evidence) if isinstance(evidence, dict) else {},
    }


def should_auto_consume_verifier_authority(
    transition: Any,
    *,
    parsed_feedback: Optional[Dict[str, Any]],
    step_ref: Optional[Dict[str, Any]] = None,
    plan_state: Any = None,
) -> bool:
    if not isinstance(transition, dict) or not isinstance(parsed_feedback, dict):
        return False
    if "consume_verifier_authority" in transition:
        return False
    if bool(parsed_feedback.get("verified", False)):
        return False
    if str(parsed_feedback.get("feedback_kind", "") or "").strip().lower() != "verification_result":
        return False
    reference = dict(step_ref or {})
    if bool(reference.get("verification_required", False)):
        return True
    if plan_state is None or not callable(getattr(plan_state, "get_plan_summary", None)):
        return False

    summary = plan_state.get_plan_summary()
    summary_payload = dict(summary or {}) if isinstance(summary, dict) else {}
    task_contract = (
        dict(summary_payload.get("task_contract", {}) or {})
        if isinstance(summary_payload.get("task_contract", {}), dict)
        else {}
    )
    completion_gate = (
        dict(summary_payload.get("completion_gate", {}) or {})
        if isinstance(summary_payload.get("completion_gate", {}), dict)
        else {}
    )
    execution_authority = (
        dict(summary_payload.get("execution_authority", {}) or {})
        if isinstance(summary_payload.get("execution_authority", {}), dict)
        else {}
    )
    verifier_runtime = build_verifier_runtime(
        task_contract=task_contract,
        completion_gate=completion_gate,
        execution_authority=execution_authority,
        context=summary_payload,
    )
    verifier_authority = dict(verifier_runtime.verifier_authority)
    if bool(verifier_authority.get("required", False)):
        return True
    active_task_node = resolve_plan_summary_active_task(summary_payload)
    active_verification_gate = (
        dict(active_task_node.get("verification_gate", {}) or {})
        if isinstance(active_task_node.get("verification_gate", {}), dict)
        else {}
    )
    return bool(active_verification_gate.get("required", False))


def recent_llm_route_usage_for_task(
    llm_route_usage_log: Any,
    *,
    task_node_id: str,
    goal_id: str,
    current_episode: int,
    current_tick: int,
    tick_window: int = 12,
) -> List[Dict[str, Any]]:
    if not isinstance(llm_route_usage_log, list):
        return []
    matched: List[Dict[str, Any]] = []
    for row in reversed(llm_route_usage_log):
        if not isinstance(row, dict) or str(row.get("event", "") or "") != "request":
            continue
        if int(row.get("episode", -1) or -1) != int(current_episode or 0):
            continue
        row_tick = int(row.get("tick", -10_000) or -10_000)
        if row_tick < int(current_tick or 0) - max(1, int(tick_window or 0)):
            continue
        row_task = str(row.get("active_task_id", "") or "")
        row_goal = str(row.get("goal_id", "") or "")
        if task_node_id and row_task == task_node_id:
            matched.append(dict(row))
            continue
        if goal_id and row_goal == goal_id:
            matched.append(dict(row))
    return matched


def record_verification_feedback_for_transition(
    transition: Any,
    *,
    step_ref: Optional[Dict[str, Any]] = None,
    plan_state: Any = None,
    route_usage_log: Any = None,
    current_episode: int = 0,
    current_tick: int = 0,
    record_route_feedback: Optional[Callable[..., Any]] = None,
) -> Optional[Dict[str, Any]]:
    parsed = verification_feedback_from_transition(transition)
    if not isinstance(parsed, dict):
        return None
    reference = dict(step_ref or {})
    if not reference:
        reference = plan_step_feedback_reference(
            plan_state,
            step_id=str((transition or {}).get("step_id", "") or ""),
        )
    task_node_id = str(reference.get("task_node_id", "") or "")
    goal_id = str(reference.get("goal_id", "") or "")
    usage_rows = recent_llm_route_usage_for_task(
        route_usage_log,
        task_node_id=task_node_id,
        goal_id=goal_id,
        current_episode=current_episode,
        current_tick=current_tick,
    )
    if not usage_rows:
        return None
    verified = bool(parsed.get("verified", False))
    feedback_score = 0.75 if verified else -1.0
    seen_routes: set[str] = set()
    reason = f"{str(parsed.get('feedback_kind', 'verification') or 'verification')}:{'pass' if verified else 'fail'}"
    verifier_function = str(parsed.get("verifier_function", "") or reference.get("verifier_function", "") or "")
    evidence = dict(parsed.get("evidence", {}) or {}) if isinstance(parsed.get("evidence", {}), dict) else {}
    verification_result = build_verification_result(
        goal_ref=goal_id,
        task_ref=task_node_id,
        verifier_function=verifier_function,
        passed=verified,
        requirement={
            "feedback_kind": str(parsed.get("feedback_kind", "") or ""),
            "step_id": str(reference.get("step_id", "") or ""),
            "step_title": str(reference.get("step_title", "") or ""),
        },
        evidence=evidence,
        failure_mode="block" if not verified else "pass",
        source=str(parsed.get("feedback_kind", "verification") or "verification"),
        metadata={
            "goal_id": goal_id,
            "task_node_id": task_node_id,
            "step_id": str(reference.get("step_id", "") or ""),
        },
    )
    verification_payload = verification_result.to_dict()
    if not callable(record_route_feedback):
        return verification_payload
    for row in usage_rows:
        selected_route = str(row.get("selected_route", row.get("route_name", "")) or "")
        if not selected_route or selected_route in seen_routes:
            continue
        seen_routes.add(selected_route)
        record_route_feedback(
            selected_route,
            score=feedback_score,
            source="verification_result",
            reason=reason,
            metadata={
                "goal_id": goal_id,
                "task_node_id": task_node_id,
                "step_id": str(reference.get("step_id", "") or ""),
                "step_title": str(reference.get("step_title", "") or ""),
                "usage_tick": int(row.get("tick", -1) or -1),
                "requested_route": str(row.get("requested_route", row.get("route_name", "")) or ""),
                "selected_route": selected_route,
                "verifier_function": verifier_function,
                "verified": verified,
                "evidence_keys": sorted(evidence.keys()),
                "verification_result_id": verification_result.result_id,
                "verification_result": verification_payload,
            },
        )
    return verification_payload


def apply_step_transitions_with_feedback(
    *,
    plan_state: Any,
    transitions: List[Dict[str, Any]],
    route_usage_log: Any = None,
    current_episode: int = 0,
    current_tick: int = 0,
    record_route_feedback: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    if plan_state is None:
        return {"applied": 0, "last_verification_result": None}
    applied = 0
    last_verification_result = None
    for transition in list(transitions or []):
        effective_transition = dict(transition) if isinstance(transition, dict) else {}
        step_id = ""
        if effective_transition:
            step_id = str(effective_transition.get("step_id", "") or "")
        step_ref = plan_step_feedback_reference(plan_state, step_id=step_id)
        parsed_feedback = verification_feedback_from_transition(effective_transition)
        if should_auto_consume_verifier_authority(
            effective_transition,
            parsed_feedback=parsed_feedback,
            step_ref=step_ref,
            plan_state=plan_state,
        ):
            effective_transition["consume_verifier_authority"] = True
        if plan_state.apply_step_transition(effective_transition):
            applied += 1
            verification_result = record_verification_feedback_for_transition(
                effective_transition,
                step_ref=step_ref,
                plan_state=plan_state,
                route_usage_log=route_usage_log,
                current_episode=current_episode,
                current_tick=current_tick,
                record_route_feedback=record_route_feedback,
            )
            if verification_result is not None:
                last_verification_result = verification_result
    return {
        "applied": applied,
        "last_verification_result": last_verification_result,
    }
