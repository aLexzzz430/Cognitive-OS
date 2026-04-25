from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from core.orchestration.runtime_stage_contracts import Stage2CandidateGenerationInput
from core.orchestration.stage_types import PlannerStageOutput


def _candidate_kwargs_request_key(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args"), dict) else {}
    if tool_args:
        function_name = str(tool_args.get("function_name", "") or "").strip()
        kwargs = tool_args.get("kwargs", {}) if isinstance(tool_args.get("kwargs", {}), dict) else {}
    else:
        function_name = str(action.get("function") or action.get("function_name") or "").strip()
        kwargs = action.get("kwargs", {}) if isinstance(action.get("kwargs", {}), dict) else {}
    if not function_name:
        return ""
    try:
        kwargs_key = json.dumps(kwargs, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    except TypeError:
        kwargs_key = repr(kwargs)
    return f"{function_name}:{kwargs_key}"


def _merge_cached_kwargs_result(action: Any, cached_result: Any) -> Any:
    if not isinstance(action, dict) or not isinstance(cached_result, dict):
        return deepcopy(cached_result)
    merged = deepcopy(action)

    cached_payload = cached_result.get("payload", {}) if isinstance(cached_result.get("payload"), dict) else {}
    cached_tool_args = cached_payload.get("tool_args", {}) if isinstance(cached_payload.get("tool_args"), dict) else {}
    if isinstance(cached_tool_args.get("kwargs"), dict):
        payload = merged.setdefault("payload", {})
        if isinstance(payload, dict):
            tool_args = payload.setdefault("tool_args", {})
            if isinstance(tool_args, dict):
                tool_args["kwargs"] = deepcopy(cached_tool_args["kwargs"])

    if isinstance(cached_result.get("kwargs"), dict):
        merged["kwargs"] = deepcopy(cached_result["kwargs"])

    cached_meta = cached_result.get("_candidate_meta", {}) if isinstance(cached_result.get("_candidate_meta"), dict) else {}
    if cached_meta:
        meta = merged.setdefault("_candidate_meta", {})
        if isinstance(meta, dict):
            for key, value in cached_meta.items():
                if str(key).startswith("structured_answer_"):
                    meta[key] = deepcopy(value)
    return merged


def run_stage2_candidate_generation(loop: Any, stage_input: Stage2CandidateGenerationInput) -> PlannerStageOutput:
    """
    Stage 2 action generation without governance side effects.

    This runtime owns the generation-side wiring; CoreMainLoop remains the
    authority that supplies ports, state, and downstream stage orchestration.
    """
    obs_before = stage_input.obs_before
    surfaced = stage_input.surfaced
    continuity_snapshot = stage_input.continuity_snapshot
    frame = stage_input.frame
    runtime_out = loop._planner_runtime.tick(
        phase="control",
        obs=obs_before,
        continuity_snapshot=continuity_snapshot,
        frame=frame,
    )
    runtime_payload = loop._consume_planner_runtime_result(runtime_out)
    plan_tick_meta = {
        "events": runtime_payload["decision_flags"].get("events", []),
        "has_plan": loop._plan_state.has_plan,
        "plan_summary": loop._plan_state.get_plan_summary(),
        "policy_profile": runtime_payload["decision_flags"].get("policy_profile"),
        "representation_profile": runtime_payload["decision_flags"].get("representation_profile"),
        "meta_control_snapshot_id": runtime_payload["decision_flags"].get("meta_control_snapshot_id"),
        "meta_control_inputs_hash": runtime_payload["decision_flags"].get("meta_control_inputs_hash"),
    }
    raw_base_action = loop._generate_action(obs_before, continuity_snapshot)
    api_raw = obs_before.get("novel_api", {}) if isinstance(obs_before.get("novel_api"), dict) else {}
    visible_functions = list(api_raw.get("visible_functions", []) or [])
    discovered_functions = list(api_raw.get("discovered_functions", []) or [])

    base_action = raw_base_action
    active_hypotheses = loop._hypotheses.get_active()
    active_hypothesis = active_hypotheses[0] if active_hypotheses else None
    if active_hypothesis:
        skills = loop._skill_rewriter.retrieve_skills(active_hypothesis, top_k=3)
        base_action = loop._skill_rewriter.rewrite(base_action, skills, active_hypothesis)
    base_action = loop._skill_frontend.rewrite_with_llm(
        base_action=base_action,
        hypotheses=loop._hypotheses.get_active(),
        obs=obs_before,
        episode=loop._episode,
        tick=loop._tick,
    )
    structured_llm_client = loop._resolve_structured_answer_llm_client()
    structured_kwargs_cache: dict[str, Any] = {}

    def _populate_action_kwargs(action: Any) -> Any:
        request_key = _candidate_kwargs_request_key(action)
        if request_key and request_key in structured_kwargs_cache:
            return _merge_cached_kwargs_result(action, structured_kwargs_cache[request_key])
        populated = loop._structured_answer_synthesizer.maybe_populate_action_kwargs(
            action,
            obs_before,
            llm_client=structured_llm_client,
        )
        if request_key:
            structured_kwargs_cache[request_key] = deepcopy(populated)
        return populated

    base_action = _populate_action_kwargs(base_action)
    arm_action, arm_meta = loop._retriever.arm_evaluate(surfaced, base_action, obs_before)
    arm_action = _populate_action_kwargs(arm_action)

    candidate_actions = loop._candidate_generator.generate(
        obs=obs_before,
        surfaced=surfaced,
        continuity_snapshot=continuity_snapshot,
        base_action=base_action,
        arm_action=arm_action,
        plan_state=loop._plan_state,
        capability_profile=loop._capability_profile,
        reliability_tracker=loop._reliability_tracker,
        episode_trace=loop._episode_trace,
        perception_summary=frame.perception_summary,
        world_model_summary=frame.world_model_summary,
        procedure_objects=loop._load_procedure_objects(obs_before),
    )
    deliberation_result = loop._run_deliberation_engine(
        obs_before=obs_before,
        surfaced=surfaced,
        continuity_snapshot=continuity_snapshot,
        frame=frame,
        candidate_actions=candidate_actions,
    )
    ranked_actions = (
        deliberation_result.get("_ranked_candidate_actions", [])
        if isinstance(deliberation_result, dict)
        else []
    )
    if isinstance(ranked_actions, list) and ranked_actions:
        candidate_actions = [dict(action) for action in ranked_actions if isinstance(action, dict)]
    plan_tick_meta["deliberation_mode"] = str(deliberation_result.get("mode", "") or "")
    plan_tick_meta["deliberation_backend"] = str(deliberation_result.get("backend", "") or "")
    deliberation_trace = deliberation_result.get("deliberation_trace", [])
    plan_tick_meta["deliberation_trace_length"] = (
        len(deliberation_trace) if isinstance(deliberation_trace, list) else 0
    )
    plan_tick_meta["probe_before_commit"] = bool(deliberation_result.get("probe_before_commit", False))

    return PlannerStageOutput(
        raw_base_action=raw_base_action,
        base_action=base_action,
        arm_action=arm_action,
        arm_meta=arm_meta,
        plan_tick_meta=plan_tick_meta,
        candidate_actions=candidate_actions,
        visible_functions=visible_functions,
        discovered_functions=discovered_functions,
        raw_candidates_snapshot=loop._snapshot_candidate_list(candidate_actions),
        decision_context={},
        stage_metrics={},
        deliberation_result=deliberation_result,
    )
