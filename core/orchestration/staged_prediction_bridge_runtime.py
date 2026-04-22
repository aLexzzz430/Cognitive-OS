from __future__ import annotations

from typing import Any, Dict, List

from core.main_loop_components import ARM_MODE_FULL
from core.orchestration.self_model_stage import SelfModelRefreshInput, SelfModelStage
from modules.world_model.learned_dynamics import build_learned_dynamics_state_snapshot


def _build_capability_switches(loop: Any, ablation_flags: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "teacher_protocol": True,
        "world_model": True,
        "episodic_memory": True,
        "planner": True,
        "prediction": bool(getattr(loop, "_prediction_enabled", False)),
        "procedure_learning": bool(getattr(loop, "_procedure_enabled", False)),
        "unified_context": bool(ablation_flags.get("unified_context_mode") == "full"),
    }


def _refresh_self_model(
    loop: Any,
    *,
    continuity_snapshot: Dict[str, Any],
    ablation_flags: Dict[str, Any],
    enabled_modules: List[str],
    capability_switches: Dict[str, bool],
) -> None:
    SelfModelStage.refresh(
        SelfModelRefreshInput(
            continuity_snapshot=continuity_snapshot,
            resource_state=getattr(loop, "_resource_state", None),
            self_model_facade=getattr(loop, "_self_model_facade", None),
            agent_id=str(getattr(loop, "agent_id", "agent")),
            arm_mode=str(getattr(loop, "arm_mode", ARM_MODE_FULL)),
            teacher_present=bool(loop._teacher_allows_intervention()),
            runtime_source=str(getattr(loop, "run_id", "unknown")),
            world_provider=str(
                getattr(loop, "_world_provider_meta", {}).get(
                    "world_provider_source",
                    "unknown",
                )
            ),
            ablation_mode=str(ablation_flags.get("unified_context_mode", "full")),
            enabled_modules=enabled_modules,
            capability_switches=capability_switches,
            module_registry={
                "teacher_protocol": "teacher_protocol",
                "world_model": "world_model",
                "episodic_memory": "episodic_memory",
                "planner": "planner",
                "prediction": "prediction",
                "procedure_learning": "procedure_learning",
                "unified_context": "unified_context",
            },
        )
    )


def _maybe_attach_prediction_bundles(
    loop: Any,
    *,
    candidate_actions: List[Dict[str, Any]],
    obs_before: Dict[str, Any],
    surfaced: List[Any],
    frame: Any,
) -> None:
    if not loop._prediction_runtime_active() or not candidate_actions:
        return
    bundles = loop._prediction_engine.predict_candidates(
        candidate_actions=candidate_actions[:6],
        episode=loop._episode,
        tick=loop._tick,
        obs=obs_before,
        surfaced=surfaced,
        hypotheses=loop._hypotheses.get_active(),
        belief_summary=frame.world_model_summary,
        plan_summary=loop._plan_state.get_plan_summary(),
        step_intent=loop._plan_state.get_intent_for_step(),
        recent_trace=loop._episode_trace[-5:],
        self_model_summary=frame.self_model_summary,
        policy_profile=loop._get_policy_profile(),
        recovery_context=loop._build_recovery_prediction_context(),
    )
    loop._last_prediction_bundle_by_action_id = bundles
    for action in candidate_actions:
        action_id = loop._build_action_id(action)
        bundle = bundles.get(action_id)
        if not bundle:
            continue
        meta = action.setdefault("_candidate_meta", {})
        if isinstance(meta, dict):
            meta["prediction"] = loop._prediction_bundle_to_dict(bundle)


def _maybe_attach_learned_dynamics_predictions(
    loop: Any,
    *,
    candidate_actions: List[Dict[str, Any]],
    obs_before: Dict[str, Any],
    frame: Any,
) -> None:
    predictor = getattr(loop, "_learned_dynamics_shadow_predictor", None)
    if predictor is None or not candidate_actions:
        return
    hidden_state_summary = (
        loop._hidden_state_tracker.summary()
        if getattr(loop, "_hidden_state_tracker", None) is not None
        else {}
    )
    state_snapshot = build_learned_dynamics_state_snapshot(
        obs_before,
        world_model_summary=getattr(frame, "world_model_summary", {}) or {},
        hidden_state_summary=hidden_state_summary,
        belief_summary=getattr(frame, "world_model_summary", {}) or {},
        identity_tracker=getattr(loop, "_persistent_object_identity_tracker", None),
        tick=int(getattr(loop, "_tick", 0) or 0) * 2,
    )
    deployment_mode = str(
        getattr(loop, "_learned_dynamics_deployment_mode", "shadow") or "shadow"
    ).strip().lower()
    for action in candidate_actions[:8]:
        prediction = predictor.predict(state_snapshot, action)
        if not isinstance(prediction, dict) or not prediction:
            continue
        meta = action.setdefault("_candidate_meta", {})
        if not isinstance(meta, dict):
            continue
        confidence = float(prediction.get("confidence", 0.0) or 0.0)
        support = int(prediction.get("support", 0) or 0)
        fusion_support = float(prediction.get("fusion_support", 0.0) or 0.0)
        valid_state_change = bool(prediction.get("valid_state_change", False))
        reward_sign = str(prediction.get("reward_sign", "") or "")
        information_gain = float(prediction.get("information_gain", 0.0) or 0.0)
        high_confidence = bool(
            confidence >= 0.78
            and support >= 2
            and valid_state_change
            and reward_sign in {"positive", "zero"}
        )
        selective_ready = bool(
            valid_state_change
            and reward_sign in {"positive", "zero"}
            and (
                (
                    confidence >= 0.68
                    and support >= 2
                    and fusion_support >= 2.4
                )
                or (
                    confidence >= 0.62
                    and support >= 3
                    and fusion_support >= 3.2
                    and information_gain >= 0.12
                )
            )
        )
        high_confidence_negative = bool(
            confidence >= 0.84
            and support >= 3
            and (not valid_state_change)
            and reward_sign == "negative"
        )
        governance_bonus = 0.0
        routing_active = False
        veto_signal = False
        promotion_signal = False
        if deployment_mode == "selective_routing":
            routing_active = selective_ready
            if selective_ready:
                if reward_sign == "positive":
                    governance_bonus = 0.08
                    promotion_signal = True
                elif information_gain >= 0.12:
                    governance_bonus = 0.06
        elif deployment_mode in {"limited_veto_promotion", "planner_rollout_dependence"}:
            routing_active = high_confidence or high_confidence_negative
            if high_confidence:
                if reward_sign == "positive":
                    governance_bonus = 0.06 if deployment_mode == "limited_veto_promotion" else 0.075
                    promotion_signal = True
                elif information_gain >= 0.18:
                    governance_bonus = 0.035 if deployment_mode == "limited_veto_promotion" else 0.045
            elif high_confidence_negative:
                governance_bonus = -0.04 if deployment_mode == "limited_veto_promotion" else -0.05
                veto_signal = True
        meta["learned_dynamics_prediction"] = prediction
        meta["learned_dynamics_confidence"] = confidence
        meta["learned_dynamics_support"] = support
        meta["learned_dynamics_fusion_support"] = float(prediction.get("fusion_support", 0.0) or 0.0)
        meta["learned_dynamics_high_confidence"] = high_confidence
        meta["learned_dynamics_selective_ready"] = selective_ready
        meta["learned_dynamics_high_confidence_negative"] = high_confidence_negative
        meta["learned_dynamics_routing_active"] = bool(routing_active)
        meta["learned_dynamics_deployment_mode"] = deployment_mode
        meta["learned_dynamics_veto_signal"] = bool(veto_signal)
        meta["learned_dynamics_promotion_signal"] = bool(promotion_signal)
        meta["learned_dynamics_governance_bonus"] = round(governance_bonus, 4)
        if deployment_mode == "planner_rollout_dependence" and routing_active:
            meta["learned_dynamics_planner_rollout_hint"] = True
        action["_candidate_meta"] = meta


def _apply_deliberation_context(
    decision_context: Dict[str, Any],
    deliberation_result: Dict[str, Any],
) -> None:
    if not deliberation_result:
        return
    decision_context["rollout_predictions"] = {
        str(key): dict(value)
        for key, value in deliberation_result.get("rollout_predictions", {}).items()
        if isinstance(value, dict)
    }
    budget = (
        deliberation_result.get("budget", {})
        if isinstance(deliberation_result.get("budget", {}), dict)
        else {}
    )
    decision_context["deliberation_depth"] = int(budget.get("depth", 1) or 1)
    decision_context["deliberation_budget"] = dict(budget)
    decision_context["deliberation_mode"] = str(
        deliberation_result.get("mode", "") or ""
    )
    decision_context["deliberation_trace"] = [
        dict(item)
        for item in deliberation_result.get("deliberation_trace", [])
        if isinstance(item, dict)
    ]
    decision_context["deliberation_candidate_tests"] = [
        dict(item)
        for item in deliberation_result.get("ranked_candidate_tests", [])
        if isinstance(item, dict)
    ]
    decision_context["deliberation_candidate_programs"] = [
        dict(item)
        for item in deliberation_result.get("ranked_candidate_programs", [])
        if isinstance(item, dict)
    ]
    decision_context["deliberation_candidate_outputs"] = [
        dict(item)
        for item in deliberation_result.get("ranked_candidate_outputs", [])
        if isinstance(item, dict)
    ]
    decision_context["deliberation_rejected_candidates"] = [
        dict(item)
        for item in deliberation_result.get("rejected_candidates", [])
        if isinstance(item, dict)
    ]
    decision_context["probe_before_commit"] = bool(
        deliberation_result.get("probe_before_commit", False)
    )


def run_stage2_prediction_bridge(loop: Any, stage_input: Any) -> Dict[str, Any]:
    bridge = stage_input.bridge
    candidate_actions = bridge.candidate_actions
    obs_before = bridge.obs_before
    surfaced = bridge.surfaced
    continuity_snapshot = bridge.continuity_snapshot
    frame = bridge.frame
    plan_tick_meta = bridge.plan_tick_meta
    deliberation_result = loop._extract_bridge_deliberation_result(bridge)

    for action in candidate_actions:
        loop._build_action_id(action)
    ablation_flags = loop._ablation_flags_snapshot()
    capability_switches = _build_capability_switches(loop, ablation_flags)
    enabled_modules = [
        name for name, enabled in capability_switches.items() if enabled
    ]
    _refresh_self_model(
        loop,
        continuity_snapshot=continuity_snapshot,
        ablation_flags=ablation_flags,
        enabled_modules=enabled_modules,
        capability_switches=capability_switches,
    )
    self_model_summary = frame.self_model_summary
    _maybe_attach_prediction_bundles(
        loop,
        candidate_actions=candidate_actions,
        obs_before=obs_before,
        surfaced=surfaced,
        frame=frame,
    )
    _maybe_attach_learned_dynamics_predictions(
        loop,
        candidate_actions=candidate_actions,
        obs_before=obs_before,
        frame=frame,
    )
    loop._annotate_candidates_with_counterfactual(
        candidate_actions,
        continuity_snapshot=continuity_snapshot,
        obs_before=obs_before,
    )
    recent_failures = sum(
        1
        for entry in loop._episode_trace[-5:]
        if float(entry.get("reward", 0.0) or 0.0) < 0.0
    )
    unified_context = frame.unified_context
    legacy_hard_off = ablation_flags.get("unified_context_mode") == "hard_off"
    decision_context = loop._build_legacy_decision_context(
        frame=frame,
        unified_context=None if legacy_hard_off else unified_context,
        obs_before=obs_before,
        continuity_snapshot=continuity_snapshot,
        surfaced=surfaced,
        plan_tick_meta=plan_tick_meta,
        recent_failures=recent_failures,
    )
    decision_context["learning_policy"] = loop._learning_context_payload()
    _apply_deliberation_context(decision_context, deliberation_result)
    return {
        "decision_context": decision_context,
        "self_model_summary": self_model_summary,
    }
