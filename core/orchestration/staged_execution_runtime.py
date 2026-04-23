from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, List

from core.cognition.unified_context import UnifiedCognitiveContext
from core.conos_kernel import build_audit_event
from core.objects import OBJECT_TYPE_HYPOTHESIS, proposal_to_object_record
from core.orchestration.execution_control import (
    ApprovalPolicy,
    ToolCapabilityRegistry,
    attach_execution_ticket,
    build_policy_block_result,
    consume_approval_grant,
    infer_available_tools_from_observation,
    issue_execution_ticket,
)
from core.orchestration.verifier_runtime import build_verifier_runtime
from core.orchestration.state_abstraction import summarize_cognitive_object_records
from core.orchestration.prediction_feedback import PredictionFeedbackInput
from core.orchestration.runtime_stage_contracts import Stage3ExecutionInput
from core.reasoning.posterior_update import update_hypothesis_posteriors
from modules.world_model.events import EventType, WorldModelEvent
from modules.world_model.learned_dynamics import (
    build_learned_dynamics_state_snapshot,
    build_transition_target,
    compare_transition_prediction,
)


@dataclass(frozen=True)
class PreActionContext:
    world_model_summary: Dict[str, Any]
    task_frame_summary: Dict[str, Any]
    object_bindings_summary: Dict[str, Any]


@dataclass(frozen=True)
class ActionExecutionArtifacts:
    function_name: str
    hypotheses_before: List[Any]
    result: Dict[str, Any]
    reward: float


@dataclass(frozen=True)
class TraceArtifacts:
    information_gain: float
    progress_markers: List[str]
    predicted_transition: Dict[str, Any]
    actual_transition: Dict[str, Any]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _hypothesis_object_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(list(rows or [])):
        if not isinstance(row, dict):
            continue
        object_id = str(
            row.get("object_id", "")
            or row.get("hypothesis_id", "")
            or f"posterior_hypothesis_{index + 1}"
        ).strip()
        proposal = dict(row)
        proposal.setdefault("object_type", OBJECT_TYPE_HYPOTHESIS)
        proposal.setdefault("family", str(row.get("hypothesis_type", row.get("family", "generic")) or "generic"))
        proposal.setdefault(
            "summary",
            str(row.get("summary", row.get("description", row.get("hypothesis_id", object_id))) or object_id),
        )
        proposal.setdefault("confidence", row.get("posterior", row.get("confidence", 0.0)))
        proposal.setdefault("source_stage", "post_execution")
        records.append(proposal_to_object_record(proposal, object_id=object_id))
    return records


def _posterior_event_evidence_ids(hypothesis_id: str, posterior_events: List[Dict[str, Any]]) -> List[str]:
    evidence_ids: List[str] = []
    clean_hypothesis_id = str(hypothesis_id or "").strip()
    if not clean_hypothesis_id:
        return evidence_ids
    for index, event in enumerate(list(posterior_events or []), start=1):
        if not isinstance(event, dict):
            continue
        if str(event.get("hypothesis_id", "") or "").strip() != clean_hypothesis_id:
            continue
        event_type = str(event.get("event_type", "unresolved") or "unresolved").strip() or "unresolved"
        evidence_ids.append(f"ev_posterior::{clean_hypothesis_id}::{event_type}::{index}")
    return evidence_ids


def _hypothesis_additional_content(record: Dict[str, Any]) -> Dict[str, Any]:
    structured_payload = (
        dict(record.get("structured_payload", {}) or {})
        if isinstance(record.get("structured_payload", {}), dict)
        else {}
    )
    if not structured_payload:
        structured_payload = (
            dict(record.get("content", {}) or {})
            if isinstance(record.get("content", {}), dict)
            else {}
        )
    additional_content = dict(structured_payload)
    for field_name in (
        "summary",
        "family",
        "source_stage",
        "surface_priority",
        "supporting_evidence",
        "contradicting_evidence",
        "hypothesis_type",
        "posterior",
        "support_count",
        "contradiction_count",
        "scope",
        "source",
        "predictions",
        "falsifiers",
        "conflicts_with",
        "supporting_evidence_rows",
        "contradicting_evidence_rows",
        "tags",
        "hypothesis_metadata",
    ):
        if field_name in record:
            additional_content[field_name] = deepcopy(record.get(field_name))
    return additional_content


def _sync_hypothesis_objects_to_store(
    loop: Any,
    *,
    hypothesis_objects: List[Dict[str, Any]],
    posterior_events: List[Dict[str, Any]],
) -> List[str]:
    store = getattr(loop, "_shared_store", None)
    if store is None:
        store = getattr(loop, "_store", None)
    if store is None or not callable(getattr(store, "get", None)):
        return []

    synced_ids: List[str] = []
    for record in list(hypothesis_objects or []):
        if not isinstance(record, dict):
            continue
        object_id = str(record.get("object_id", "") or "").strip()
        if not object_id:
            continue
        evidence_ids = _posterior_event_evidence_ids(object_id, posterior_events)
        existing = store.get(object_id)
        if isinstance(existing, dict) and existing and callable(getattr(store, "merge_update", None)):
            store.merge_update(
                object_id,
                evidence_ids,
                _hypothesis_additional_content(record),
            )
            synced_ids.append(object_id)
            continue
        if callable(getattr(store, "restore_records", None)):
            seeded_record = deepcopy(record)
            existing_evidence_ids = [
                str(item)
                for item in list(seeded_record.get("evidence_ids", []) or [])
                if str(item or "")
            ]
            seeded_record["evidence_ids"] = list(dict.fromkeys(existing_evidence_ids + evidence_ids))
            store.restore_records([seeded_record], replace=False)
            synced_ids.append(object_id)
    return synced_ids


def _execution_audit_from_ticket(
    *,
    ticket_payload: Dict[str, Any],
    event_type: str,
    source_stage: str,
    payload: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not isinstance(ticket_payload, dict) or not ticket_payload:
        return {}
    audit_event = build_audit_event(
        event_type,
        goal_ref=str(ticket_payload.get("goal_ref", "") or ""),
        task_ref=str(ticket_payload.get("task_ref", "") or ""),
        graph_ref=str(ticket_payload.get("graph_ref", "") or ""),
        tool_id=str(ticket_payload.get("tool_id", "") or ticket_payload.get("function_name", "") or ""),
        execution_ticket_id=str(ticket_payload.get("ticket_id", "") or ""),
        source_stage=source_stage,
        payload=dict(payload or {}),
        metadata=dict(metadata or {}),
        audit_event_id=str(ticket_payload.get("audit_event_id", "") or ""),
    )
    return audit_event.to_dict()


def _capture_pre_action_context(loop: Any) -> PreActionContext:
    active_frame = getattr(loop, "_active_tick_context_frame", None)
    world_model_summary = (
        getattr(active_frame, "world_model_summary", {})
        if active_frame is not None
        else {}
    )
    unified_context = (
        getattr(active_frame, "unified_context", None)
        if active_frame is not None
        else None
    )
    task_frame_summary = (
        getattr(unified_context, "task_frame_summary", {})
        if isinstance(unified_context, UnifiedCognitiveContext)
        else {}
    )
    object_bindings_summary = (
        getattr(unified_context, "object_bindings_summary", {})
        if isinstance(unified_context, UnifiedCognitiveContext)
        else {}
    )
    return PreActionContext(
        world_model_summary=world_model_summary if isinstance(world_model_summary, dict) else {},
        task_frame_summary=task_frame_summary if isinstance(task_frame_summary, dict) else {},
        object_bindings_summary=(
            object_bindings_summary if isinstance(object_bindings_summary, dict) else {}
        ),
    )


def _resolve_execution_policy_context(loop: Any) -> Dict[str, Any]:
    active_frame = getattr(loop, "_active_tick_context_frame", None)
    unified_context = getattr(active_frame, "unified_context", None) if active_frame is not None else None
    state_mgr = getattr(loop, "_state_mgr", None)
    runtime = getattr(loop, "_goal_task_runtime", None)
    if runtime is not None and hasattr(runtime, "refresh"):
        binding = runtime.refresh(
            unified_context=unified_context,
            state_mgr=state_mgr,
            run_id=str(getattr(loop, "run_id", "") or ""),
            episode=int(getattr(loop, "_episode", 0) or 0),
            tick=int(getattr(loop, "_tick", 0) or 0),
        )
        payload = binding.to_context() if hasattr(binding, "to_context") else {}
    else:
        payload = {}
    approval_context = getattr(loop, "_execution_approval_context", None)
    if isinstance(approval_context, dict):
        approved_ticket_scopes = [
            dict(item)
            for item in list(approval_context.get("approved_ticket_scopes", []) or [])
            if isinstance(item, dict)
        ]
        approved_functions = [
            str(item).strip()
            for item in list(approval_context.get("approved_functions", []) or [])
            if str(item).strip()
        ]
        approved_capabilities = [
            str(item).strip()
            for item in list(approval_context.get("approved_capabilities", []) or [])
            if str(item).strip()
        ]
        approved_secret_leases = [
            dict(item)
            for item in list(approval_context.get("approved_secret_leases", []) or [])
            if isinstance(item, dict)
        ]
        if approved_ticket_scopes:
            payload["approved_ticket_scopes"] = approved_ticket_scopes
        if approved_functions:
            payload["approved_functions"] = approved_functions
        if approved_capabilities:
            payload["approved_capabilities"] = approved_capabilities
        if approved_secret_leases:
            payload["approved_secret_leases"] = approved_secret_leases
    payload["run_id"] = str(getattr(loop, "run_id", "") or "")
    payload["episode"] = int(getattr(loop, "_episode", 0) or 0)
    payload["tick"] = int(getattr(loop, "_tick", 0) or 0)
    return payload


def _worker_allows_action(policy_context: Dict[str, Any], function_name: str) -> tuple[bool, str]:
    execution_authority = _as_dict(policy_context.get("execution_authority", {}))
    task_contract = _as_dict(policy_context.get("task_contract", {}))
    active_task = _as_dict(execution_authority.get("active_task", {}))
    assigned_worker = _as_dict(active_task.get("assigned_worker", {}))
    verifier_runtime = build_verifier_runtime(
        task_contract=task_contract,
        execution_authority=execution_authority,
        context=policy_context,
    )
    verifier_authority = dict(verifier_runtime.verifier_authority)
    worker_type = str(assigned_worker.get("worker_type", "executor") or "executor").strip().lower()
    ownership = str(assigned_worker.get("ownership", "exclusive") or "exclusive").strip().lower()
    assignment_source = str(assigned_worker.get("source", "derived") or "derived").strip().lower()
    if (
        function_name in {"", "wait"}
        or ownership != "exclusive"
        or worker_type in {"", "executor"}
        or assignment_source == "derived"
    ):
        return True, ""
    verification_gate = _as_dict(active_task.get("verification_gate", {}))
    verifier_function = str(
        verifier_authority.get("verifier_function", "")
        or verification_gate.get("verifier_function", "")
        or ""
    ).strip()
    if worker_type == "verifier":
        if function_name == verifier_function or function_name in {"inspect", "probe", "verify", "check", "measure"}:
            return True, ""
    return False, f"worker_ownership_mismatch:{worker_type}:{function_name}"


def _build_worker_block_result(*, ticket: Dict[str, Any], reason: str, worker_type: str) -> Dict[str, Any]:
    return {
        "success": False,
        "ok": False,
        "blocked_by_policy": False,
        "state": "WORKER_OWNERSHIP_BLOCKED",
        "failure_reason": reason,
        "reward": 0.0,
        "terminal": False,
        "done": False,
        "execution_ticket": dict(ticket),
        "events": [
            {
                "type": "worker_ownership_blocked",
                "reason": reason,
                "worker_type": worker_type,
                "ticket_id": str(ticket.get("ticket_id", "") or ""),
            }
        ],
    }


def _maybe_mark_selected_plan_step_in_progress(
    loop: Any,
    *,
    action_to_use: Dict[str, Any],
    function_name: str,
    policy_context: Dict[str, Any],
) -> bool:
    if str(function_name or "").strip().lower() == "wait":
        return False
    plan_state = getattr(loop, "_plan_state", None)
    if plan_state is None or not bool(getattr(plan_state, "has_plan", False)):
        return False
    current_step = getattr(plan_state, "current_step", None)
    if current_step is None:
        return False
    step_id = str(getattr(current_step, "step_id", "") or "").strip()
    execution_authority = _as_dict(policy_context.get("execution_authority", {}))
    authority_active_task = _as_dict(execution_authority.get("active_task", {}))
    step_target = str(
        authority_active_task.get("target_function", "")
        or getattr(current_step, "target_function", "")
        or ""
    ).strip()
    planning = _as_dict(execution_authority.get("planning", {}))
    if not bool(authority_active_task.get("allowed_by_planning", planning.get("active_task_allowed", True))):
        return False
    action_matches = bool(function_name) and (
        (bool(step_target) and (step_target == "combine" or function_name == step_target))
        or (not step_target)
    )
    if not action_matches:
        return False
    transition = {
        "event": "start",
        "step_id": step_id,
        "action_function": function_name,
        "source": "stage3_pre_execution",
    }
    applied = False
    if hasattr(plan_state, "apply_step_transition"):
        applied = bool(plan_state.apply_step_transition(transition))
    elif hasattr(plan_state, "mark_current_step_in_progress"):
        applied = bool(plan_state.mark_current_step_in_progress(step_id=step_id or None))
    if not applied:
        return False
    if isinstance(action_to_use, dict):
        meta = action_to_use.setdefault("_candidate_meta", {})
        if isinstance(meta, dict):
            meta["pre_execution_step_transition"] = dict(transition)
    setattr(loop, "_last_pre_execution_step_transition", dict(transition))
    return True


def _execute_action(loop: Any, action_to_use: Dict[str, Any], obs_before: Dict[str, Any]) -> ActionExecutionArtifacts:
    function_name = loop._extract_action_function_name(action_to_use, default="wait")
    hypotheses_before = list(loop._hypotheses.get_active())
    runtime_raw: Dict[str, Any]
    observed_tools = infer_available_tools_from_observation(obs_before)
    registry = ToolCapabilityRegistry.from_available_tools(
        observed_tools
        or list(getattr(loop, "_last_available_tools", []) or [])
    )
    policy_context: Dict[str, Any] = {}
    if observed_tools:
        loop._last_available_tools = [dict(item) for item in observed_tools]
    loop._tool_capability_registry = registry
    approval_policy = getattr(loop, "_execution_approval_policy", None)
    if approval_policy is None:
        approval_policy = ApprovalPolicy()
        loop._execution_approval_policy = approval_policy
    policy_context = _resolve_execution_policy_context(loop)
    decision = approval_policy.evaluate(action_to_use, registry, context=policy_context)
    ticket = issue_execution_ticket(
        action=action_to_use,
        decision=decision,
        context=policy_context,
    )
    ticket_payload = ticket.to_dict()
    loop._last_execution_ticket = ticket_payload
    attach_execution_ticket(action_to_use, ticket, decision)
    if not decision.allowed:
        runtime_raw = build_policy_block_result(ticket=ticket, decision=decision)
        result = loop._maybe_enrich_observation(runtime_raw or {})
        reward = float(result.get("reward", 0.0) or 0.0)
        return ActionExecutionArtifacts(
            function_name=function_name,
            hypotheses_before=hypotheses_before,
            result=result,
            reward=reward,
        )
    plan_state = getattr(loop, "_plan_state", None)
    if (
        decision.approval_required
        and decision.approval_granted
        and plan_state is not None
        and hasattr(plan_state, "apply_step_transition")
    ):
        current_step = getattr(plan_state, "current_step", None)
        step_id = str(getattr(current_step, "step_id", "") or "")
        if step_id:
            plan_state.apply_step_transition(
                {
                    "event": "approve",
                    "step_id": step_id,
                    "approval_grant_id": str(decision.approval_grant_id or ""),
                    "approval_sources": [str(item) for item in list(decision.approval_sources or []) if str(item)],
                    "approved_by": "execution_control",
                }
            )
    approval_context = getattr(loop, "_execution_approval_context", None)
    consume_approval_grant(approval_context=approval_context, decision=decision)
    worker_allowed, worker_reason = _worker_allows_action(policy_context, function_name)
    if not worker_allowed:
        result = loop._maybe_enrich_observation(
            _build_worker_block_result(
                ticket=ticket_payload,
                reason=worker_reason,
                worker_type=str(
                    _as_dict(_as_dict(policy_context.get("execution_authority", {})).get("active_task", {}))
                    .get("assigned_worker", {})
                    .get("worker_type", "")
                    or ""
                ),
            )
        )
        result["policy_decision"] = decision.to_dict()
        result["audit_event"] = _execution_audit_from_ticket(
            ticket_payload=ticket_payload,
            event_type="execution_worker_blocked",
            source_stage="execution",
            payload={
                "reason": worker_reason,
                "function_name": function_name,
            },
            metadata={
                "worker_type": str(
                    _as_dict(_as_dict(policy_context.get("execution_authority", {})).get("active_task", {}))
                    .get("assigned_worker", {})
                    .get("worker_type", "")
                    or ""
                ),
            },
        )
        return ActionExecutionArtifacts(
            function_name=function_name,
            hypotheses_before=hypotheses_before,
            result=result,
            reward=0.0,
        )
    _maybe_mark_selected_plan_step_in_progress(
        loop,
        action_to_use=action_to_use,
        function_name=function_name,
        policy_context=policy_context,
    )
    if function_name == "wait":
        runtime_raw = {"success": True, "wait": True, "reward": 0.0}
    else:
        runtime_raw = loop._world.act(action_to_use)
    result = loop._maybe_enrich_observation(runtime_raw or {})
    result["execution_ticket"] = ticket_payload
    result["policy_decision"] = decision.to_dict()
    result["audit_event"] = _execution_audit_from_ticket(
        ticket_payload=ticket_payload,
        event_type="execution_completed",
        source_stage="execution",
        payload={
            "function_name": function_name,
            "success": bool(result.get("success", True)),
            "blocked_by_policy": bool(result.get("blocked_by_policy", False)),
            "reward": float(result.get("reward", 0.0) or 0.0),
        },
        metadata={
            "terminal": bool(result.get("terminal", False)),
            "done": bool(result.get("done", False)),
        },
    )
    reward = float(result.get("reward", 0.0) or 0.0)
    return ActionExecutionArtifacts(
        function_name=function_name,
        hypotheses_before=hypotheses_before,
        result=result,
        reward=reward,
    )


def _append_execution_trace(
    loop: Any,
    *,
    action_to_use: Dict[str, Any],
    query: Any,
    obs_before: Dict[str, Any],
    pre_action_context: PreActionContext,
    execution: ActionExecutionArtifacts,
) -> TraceArtifacts:
    information_gain = loop._estimate_information_gain(
        obs_before,
        execution.result,
        execution.function_name,
    )
    progress_markers = loop._build_progress_markers(
        execution.result,
        execution.reward,
        execution.function_name,
    )
    step_success = bool(execution.result.get("success", True))
    task_solved = bool(execution.result.get("solved", False))
    solve_proximity = (
        1.0
        if task_solved
        else (0.6 if execution.reward > 0.0 or step_success else 0.0)
    )
    loop._episode_reward += execution.reward
    loop._retriever.consume(action_to_use, execution.result, query)
    loop._record_memory_consumption_proof(action_to_use, query, execution.result)
    effect_trace = loop._derive_action_effect_signature(
        obs_before=obs_before,
        result=execution.result,
        action=action_to_use,
        information_gain=information_gain,
        progress_markers=progress_markers,
    )
    inferred_level_goal = loop._infer_level_goal_summary(
        obs_before=obs_before if isinstance(obs_before, dict) else {},
        world_model_summary=pre_action_context.world_model_summary,
        task_frame_summary=pre_action_context.task_frame_summary,
        object_bindings_summary=pre_action_context.object_bindings_summary,
    )
    goal_progress_assessment = loop._derive_goal_progress_assessment(
        goal_summary=inferred_level_goal,
        effect_trace=effect_trace if isinstance(effect_trace, dict) else {},
        information_gain=information_gain,
        progress_markers=progress_markers,
    )
    goal_bundle_state = loop._derive_goal_bundle_state(
        goal_summary=inferred_level_goal,
        goal_progress_assessment=goal_progress_assessment,
        recent_state=loop._recent_goal_progress_state(list(loop._episode_trace), limit=10),
    )
    hidden_before = (
        loop._hidden_state_tracker.summary()
        if getattr(loop, "_hidden_state_tracker", None) is not None
        else {}
    )
    post_snapshot = build_learned_dynamics_state_snapshot(
        execution.result,
        world_model_summary={"inferred_level_goal": inferred_level_goal},
        hidden_state_summary={},
        belief_summary={
            "inferred_level_goal": inferred_level_goal,
            "goal_progress_assessment": goal_progress_assessment,
            "goal_bundle_state": goal_bundle_state,
        },
        identity_tracker=getattr(loop, "_persistent_object_identity_tracker", None),
        tick=int(loop._tick) * 2 + 1,
    )
    learned_dynamics_shadow = {}
    predictor = getattr(loop, "_learned_dynamics_shadow_predictor", None)
    if predictor is not None:
        pre_snapshot = build_learned_dynamics_state_snapshot(
            obs_before,
            world_model_summary=pre_action_context.world_model_summary,
            hidden_state_summary=hidden_before,
            belief_summary=pre_action_context.world_model_summary,
            identity_tracker=getattr(loop, "_persistent_object_identity_tracker", None),
            tick=int(loop._tick) * 2,
        )
        predicted_transition = predictor.predict(pre_snapshot, action_to_use)
        if isinstance(predicted_transition, dict) and predicted_transition:
            actual_transition = build_transition_target(
                pre_snapshot,
                post_snapshot,
                result=execution.result,
                reward=execution.reward,
                information_gain=information_gain,
                goal_progress_assessment=goal_progress_assessment,
            )
            learned_dynamics_shadow = {
                "tick": loop._tick,
                "function_name": execution.function_name,
                "prediction": predicted_transition,
                "actual": actual_transition,
                "error": compare_transition_prediction(predicted_transition, actual_transition),
            }
            loop._learned_dynamics_shadow_log.append(learned_dynamics_shadow)
            del loop._learned_dynamics_shadow_log[:-200]
    progress_markers = list(progress_markers) + loop._build_goal_progress_markers(
        goal_progress_assessment,
        execution.function_name,
    )
    if bool(goal_progress_assessment.get("progressed", False)):
        solve_proximity = max(
            solve_proximity,
            min(
                0.92,
                0.18
                + float(goal_progress_assessment.get("goal_progress_score", 0.0) or 0.0)
                * 0.42,
            ),
        )
    payload = action_to_use.get("payload")
    kwargs = {}
    if isinstance(payload, dict):
        tool_args = payload.get("tool_args", {})
        if isinstance(tool_args, dict):
            kwargs = tool_args.get("kwargs", {})
    action_snapshot = {
        "kind": "wait" if execution.function_name == "wait" else "action",
        "function_name": execution.function_name,
        "kwargs": kwargs if isinstance(kwargs, dict) else {},
    }
    loop._episode_trace.append(
        {
            "tick": loop._tick,
            "observation": obs_before,
            "action": action_to_use,
            "action_snapshot": action_snapshot,
            "outcome": execution.result,
            "reward": execution.reward,
            "progress_markers": progress_markers,
            "information_gain": information_gain,
            "solve_proximity": solve_proximity,
            "task_progress": {
                "progressed": bool(
                    goal_progress_assessment.get("progressed", False)
                    or loop._progress_markers_show_positive_progress(progress_markers)
                ),
                "solved": bool(task_solved),
                "goal_progress_score": float(
                    goal_progress_assessment.get("goal_progress_score", 0.0) or 0.0
                ),
                "goal_distance_estimate": float(
                    goal_progress_assessment.get("goal_distance_estimate", 1.0) or 1.0
                ),
            },
            "inferred_level_goal": (
                inferred_level_goal if isinstance(inferred_level_goal, dict) else {}
            ),
            "goal_progress_assessment": (
                goal_progress_assessment
                if isinstance(goal_progress_assessment, dict)
                else {}
            ),
            "goal_bundle_state": (
                goal_bundle_state if isinstance(goal_bundle_state, dict) else {}
            ),
            "clicked_family": (
                effect_trace.get("clicked_family", {})
                if isinstance(effect_trace, dict)
                else {}
            ),
            "action_effect_signature": (
                effect_trace.get("effect_signature", {})
                if isinstance(effect_trace, dict)
                else {}
            ),
            "family_effect_attribution": (
                effect_trace.get("family_effect_attribution", {})
                if isinstance(effect_trace, dict)
                else {}
            ),
            "persistent_object_identity_summary": _as_dict(_as_dict(post_snapshot).get("object_graph", {})).get("identity_summary", {}),
            "learned_dynamics_shadow": (
                learned_dynamics_shadow if isinstance(learned_dynamics_shadow, dict) else {}
            ),
            "unified_context_mode": loop._ablation_flags_snapshot().get(
                "unified_context_mode",
                "full",
            ),
        }
    )
    return TraceArtifacts(
        information_gain=information_gain,
        progress_markers=progress_markers,
        predicted_transition=(
            _as_dict(learned_dynamics_shadow.get("prediction", {}))
            if isinstance(learned_dynamics_shadow, dict)
            else {}
        ),
        actual_transition=(
            _as_dict(learned_dynamics_shadow.get("actual", {}))
            if isinstance(learned_dynamics_shadow, dict)
            else {}
        ),
    )


def _update_hypothesis_posteriors_after_action(
    loop: Any,
    *,
    action_to_use: Dict[str, Any],
    execution: ActionExecutionArtifacts,
    trace: TraceArtifacts,
) -> None:
    active_frame = getattr(loop, "_active_tick_context_frame", None)
    unified_context = getattr(active_frame, "unified_context", None)
    if not isinstance(unified_context, UnifiedCognitiveContext):
        return
    plan_state_summary = (
        dict(getattr(unified_context, "plan_state_summary", {}) or {})
        if isinstance(getattr(unified_context, "plan_state_summary", {}), dict)
        else {}
    )
    verifier_runtime = build_verifier_runtime(
        task_contract=_as_dict(plan_state_summary.get("task_contract", {})),
        completion_gate=_as_dict(plan_state_summary.get("completion_gate", {})),
        execution_authority=_as_dict(plan_state_summary.get("execution_authority", {})),
        context=plan_state_summary,
    )
    verifier_teaching = dict(verifier_runtime.posterior_teaching)
    hypotheses = list(unified_context.competing_hypotheses or [])
    if not hypotheses:
        return
    posterior_update = update_hypothesis_posteriors(
        hypotheses,
        action=action_to_use if isinstance(action_to_use, dict) else {},
        result=execution.result if isinstance(execution.result, dict) else {},
        predicted_transition=trace.predicted_transition,
        actual_transition=trace.actual_transition,
        reward=execution.reward,
        information_gain=trace.information_gain,
        obs_before=getattr(loop, "_last_obs", None) if isinstance(getattr(loop, "_last_obs", None), dict) else None,
        world_model_summary=(
            unified_context.active_beliefs_summary
            if isinstance(getattr(unified_context, "active_beliefs_summary", {}), dict)
            else None
        ),
        verifier_teaching=verifier_teaching,
    )
    updated_hypotheses = [
        dict(item)
        for item in list(posterior_update.get("updated_hypotheses", []) or [])
        if isinstance(item, dict)
    ]
    previous_summary = (
        dict(unified_context.posterior_summary or {})
        if isinstance(getattr(unified_context, "posterior_summary", {}), dict)
        else {}
    )
    posterior_summary = dict(posterior_update.get("posterior_summary", {}) or {})
    posterior_events = [
        dict(item)
        for item in list(posterior_update.get("posterior_events", []) or [])
        if isinstance(item, dict)
    ]
    posterior_debug = dict(posterior_update.get("posterior_debug", {}) or {})
    if posterior_summary:
        deliberation_snapshot = (
            dict(previous_summary.get("deliberation_snapshot", {}) or {})
            if isinstance(previous_summary.get("deliberation_snapshot", {}), dict)
            else {}
        )
        execution_snapshot = (
            dict(previous_summary.get("execution_snapshot", {}) or {})
            if isinstance(previous_summary.get("execution_snapshot", {}), dict)
            else {}
        )
        execution_snapshot.update(
            {
                "action_function_name": str(execution.function_name or ""),
                "reward": round(float(execution.reward or 0.0), 6),
                "information_gain": round(float(trace.information_gain or 0.0), 6),
                "posterior_event_count": len(posterior_events),
                "predicted_transition_available": bool(trace.predicted_transition),
                "actual_transition_available": bool(trace.actual_transition),
                "leading_hypothesis_id_after_action": str(posterior_summary.get("leading_hypothesis_id", "") or ""),
                "support_events": int(posterior_summary.get("support_events", 0) or 0),
                "contradiction_events": int(posterior_summary.get("contradiction_events", 0) or 0),
                "unresolved_events": int(posterior_summary.get("unresolved_events", 0) or 0),
                "verifier_teaching": verifier_teaching,
                "verifier_runtime_version": str(verifier_runtime.runtime_version),
                "verifier_authority_decision": str(
                    verifier_runtime.verifier_authority.get("decision", "") or ""
                ),
            }
        )
        merged_summary = dict(previous_summary)
        merged_summary.update(posterior_summary)
        merged_summary.pop("runtime_object_graph", None)
        merged_summary["summary_stage"] = "post_execution"
        merged_summary["last_update_source"] = "execution"
        if deliberation_snapshot:
            merged_summary["deliberation_snapshot"] = deliberation_snapshot
        merged_summary["execution_snapshot"] = execution_snapshot
        posterior_summary = merged_summary
    if updated_hypotheses:
        unified_context.competing_hypotheses = updated_hypotheses
        hypothesis_objects = _hypothesis_object_records(updated_hypotheses)
        active_hypotheses_summary = summarize_cognitive_object_records(
            hypothesis_objects,
            limit=8,
        )
        unified_context.active_hypotheses_summary = active_hypotheses_summary
        synced_hypothesis_object_ids = _sync_hypothesis_objects_to_store(
            loop,
            hypothesis_objects=hypothesis_objects,
            posterior_events=posterior_events,
        )
    if posterior_summary:
        unified_context.posterior_summary = posterior_summary
    if getattr(loop, "_episode_trace", None) and isinstance(loop._episode_trace[-1], dict):
        loop._episode_trace[-1]["hypothesis_posterior_events"] = posterior_events
        if updated_hypotheses:
            loop._episode_trace[-1]["durable_hypothesis_object_ids"] = synced_hypothesis_object_ids
        loop._episode_trace[-1]["posterior_summary"] = posterior_summary
        if posterior_debug:
            loop._episode_trace[-1]["posterior_debug"] = posterior_debug
    if hasattr(loop, "_state_mgr"):
        patch: Dict[str, Any] = {}
        if updated_hypotheses:
            patch["object_workspace.competing_hypotheses"] = updated_hypotheses
            patch["object_workspace.competing_hypothesis_objects"] = hypothesis_objects
            patch["object_workspace.active_hypotheses_summary"] = active_hypotheses_summary
        if posterior_summary:
            patch["object_workspace.posterior_summary"] = posterior_summary
        if patch:
            loop._state_mgr.update_state(
                patch,
                reason="reasoning:posterior_update",
                module="core.reasoning",
            )


def _emit_execution_events(loop: Any, execution: ActionExecutionArtifacts) -> None:
    loop._event_bus.emit(
        WorldModelEvent(
            event_type=EventType.ACTION_EXECUTED,
            episode=loop._episode,
            tick=loop._tick,
            data={
                "function_name": execution.function_name,
                "terminal": execution.result.get("terminal", False),
                "done": execution.result.get("done", False),
                "blocked_by_policy": execution.result.get("blocked_by_policy", False),
                "execution_ticket_id": (
                    execution.result.get("execution_ticket", {}) if isinstance(execution.result.get("execution_ticket", {}), dict) else {}
                ).get("ticket_id", ""),
            },
            source_stage="execution",
        )
    )
    loop._event_bus.emit(
        WorldModelEvent(
            event_type=EventType.REWARD_OBSERVED,
            episode=loop._episode,
            tick=loop._tick,
            data={
                "reward": execution.reward,
                "cumulative_episode_reward": loop._episode_reward,
            },
            source_stage="execution",
        )
    )
    loop._event_log.append(
        loop._event_log_builder.action_executed(
            episode=loop._episode,
            tick=loop._tick,
            data={
                "function_name": execution.function_name,
                "reward": execution.reward,
                "terminal": execution.result.get("terminal", False),
                "blocked_by_policy": execution.result.get("blocked_by_policy", False),
            },
            source_stage="execution",
        )
    )
    loop._event_log.append(
        {
            "event_type": "reward_observed",
            "episode": loop._episode,
            "tick": loop._tick,
            "data": {
                "reward": execution.reward,
                "cumulative": loop._episode_reward,
            },
            "source_module": "core",
            "source_stage": "execution",
        }
    )
    audit_event = execution.result.get("audit_event", {})
    if isinstance(audit_event, dict) and audit_event:
        loop._event_log.append(dict(audit_event))


def _apply_prediction_feedback(
    loop: Any,
    *,
    action_to_use: Dict[str, Any],
    obs_before: Dict[str, Any],
    execution: ActionExecutionArtifacts,
) -> Any:
    action_id = loop._build_action_id(action_to_use)
    feedback_output = loop._prediction_feedback.apply_after_action(
        PredictionFeedbackInput(
            episode=loop._episode,
            tick=loop._tick,
            action_id=action_id,
            function_name=execution.function_name,
            result=execution.result,
            reward=execution.reward,
            obs_before=obs_before,
            hypotheses_before=execution.hypotheses_before,
            hypotheses_after=list(loop._hypotheses.get_active()),
            prediction_bundle=loop._last_prediction_bundle_by_action_id.get(action_id),
            prediction_enabled=bool(loop._prediction_enabled),
        ),
        prediction_adjudicator=loop._prediction_adjudicator,
        prediction_registry=loop._prediction_registry,
        prediction_engine=loop._prediction_engine,
        reliability_tracker=loop._reliability_tracker,
        meta_control=loop._meta_control,
        governance_log=loop._governance_log,
        prediction_trace_log=loop._prediction_trace_log,
        prediction_positive_miss_streak=loop._prediction_positive_miss_streak,
        world_model_feedback_port=loop._prediction_miss_feedback,
    )
    loop._prediction_positive_miss_streak = (
        feedback_output.prediction_positive_miss_streak
    )
    if feedback_output.pending_replan_patch is not None:
        loop._pending_replan = feedback_output.pending_replan_patch
    return feedback_output


def _update_hidden_state(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    pre_action_context: PreActionContext,
    execution: ActionExecutionArtifacts,
    feedback_output: Any,
) -> None:
    if loop._hidden_state_tracker is None:
        return
    loop._hidden_state_tracker.update(
        episode=int(loop._episode),
        tick=int(loop._tick),
        obs_before=obs_before if isinstance(obs_before, dict) else {},
        result=execution.result if isinstance(execution.result, dict) else {},
        reward=execution.reward,
        function_name=execution.function_name,
        world_model_summary=pre_action_context.world_model_summary,
    )
    error_payload = (
        feedback_output.trace_entry.get("error", {})
        if isinstance(feedback_output.trace_entry, dict)
        else {}
    )
    total_error = float(error_payload.get("total_error", 0.0) or 0.0)
    if total_error > 0.0:
        loop._hidden_state_tracker.record_prediction_error(total_error)
    if isinstance(execution.result, dict):
        execution.result["world_model"] = loop._build_world_model_context(
            execution.result.get("perception")
        )


def _mechanism_rows(loop: Any) -> List[Dict[str, Any]]:
    active_frame = getattr(loop, "_active_tick_context_frame", None)
    unified_context = getattr(active_frame, "unified_context", None)
    if not isinstance(unified_context, UnifiedCognitiveContext):
        return []
    return list(unified_context.mechanism_hypotheses_summary or [])


def _update_mechanism_runtime(
    loop: Any,
    *,
    action_to_use: Dict[str, Any],
    obs_before: Dict[str, Any],
    execution: ActionExecutionArtifacts,
    trace: TraceArtifacts,
) -> None:
    mechanism_rows = _mechanism_rows(loop)
    if not mechanism_rows:
        return
    update_out = loop._mechanism_posterior_updater.update_after_action(
        loop._mechanism_runtime_state,
        mechanism_rows,
        action=action_to_use if isinstance(action_to_use, dict) else {},
        result=execution.result if isinstance(execution.result, dict) else {},
        reward=execution.reward,
        information_gain=trace.information_gain,
        progress_markers=trace.progress_markers,
        obs_before=obs_before if isinstance(obs_before, dict) else {},
        tick=loop._tick,
        object_store=getattr(loop, "_shared_store", None) or getattr(loop, "_store", None),
    )
    loop._mechanism_runtime_state = dict(update_out.state or {})
    updated_mechanism_rows = list(update_out.mechanisms or mechanism_rows)
    loop._last_mechanism_runtime_view = {
        "mechanism_hypotheses_summary": updated_mechanism_rows,
        "mechanism_control_summary": dict(update_out.control_summary or {}),
    }
    active_frame = getattr(loop, "_active_tick_context_frame", None)
    unified_context = getattr(active_frame, "unified_context", None)
    if isinstance(unified_context, UnifiedCognitiveContext):
        unified_context.mechanism_hypotheses_summary = list(updated_mechanism_rows)
        unified_context.mechanism_control_summary = dict(update_out.control_summary or {})
        beliefs_summary = dict(unified_context.active_beliefs_summary or {})
        beliefs_summary["mechanism_hypotheses_summary"] = list(updated_mechanism_rows)
        beliefs_summary["mechanism_hypothesis_objects"] = list(updated_mechanism_rows)
        beliefs_summary["mechanism_control_summary"] = dict(update_out.control_summary or {})
        unified_context.active_beliefs_summary = beliefs_summary
    if isinstance(getattr(active_frame, "world_model_summary", None), dict):
        active_frame.world_model_summary["mechanism_hypotheses_summary"] = list(updated_mechanism_rows)
        active_frame.world_model_summary["mechanism_hypothesis_objects"] = list(updated_mechanism_rows)
        active_frame.world_model_summary["mechanism_control_summary"] = dict(update_out.control_summary or {})
    if getattr(loop, "_episode_trace", None):
        loop._episode_trace[-1]["durable_mechanism_object_ids"] = list(update_out.durable_object_ids or [])
    loop._mechanism_control_audit_log.append(
        {
            "episode": loop._episode,
            "tick": loop._tick,
            "event": "post_action_mechanism_update",
            "selected_function": execution.function_name,
            "reward": execution.reward,
            "information_gain": trace.information_gain,
            "diagnostics": loop._json_safe(update_out.diagnostics),
        }
    )


def run_stage3_execution(loop: Any, stage_input: Stage3ExecutionInput) -> Dict[str, Any]:
    pre_action_context = _capture_pre_action_context(loop)
    execution = _execute_action(loop, stage_input.action_to_use, stage_input.obs_before)
    trace = _append_execution_trace(
        loop,
        action_to_use=stage_input.action_to_use,
        query=stage_input.query,
        obs_before=stage_input.obs_before,
        pre_action_context=pre_action_context,
        execution=execution,
    )
    _emit_execution_events(loop, execution)
    _update_hypothesis_posteriors_after_action(
        loop,
        action_to_use=stage_input.action_to_use,
        execution=execution,
        trace=trace,
    )
    feedback_output = _apply_prediction_feedback(
        loop,
        action_to_use=stage_input.action_to_use,
        obs_before=stage_input.obs_before,
        execution=execution,
    )
    _update_hidden_state(
        loop,
        obs_before=stage_input.obs_before,
        pre_action_context=pre_action_context,
        execution=execution,
        feedback_output=feedback_output,
    )
    _update_mechanism_runtime(
        loop,
        action_to_use=stage_input.action_to_use,
        obs_before=stage_input.obs_before,
        execution=execution,
        trace=trace,
    )
    return {"result": execution.result, "reward": execution.reward}
