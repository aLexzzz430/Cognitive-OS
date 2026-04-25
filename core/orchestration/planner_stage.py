from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set

from core.orchestration.stage_types import PlannerStageInput, PlannerStageOutput


_KWARGS_MATERIALIZATION_FUNCTIONS: Set[str] = {
    "file_read",
    "repo_grep",
    "note_write",
    "hypothesis_add",
    "hypothesis_update",
    "hypothesis_compete",
    "discriminating_test_add",
    "candidate_files_set",
    "apply_patch",
    "edit_replace_range",
    "edit_insert_after",
    "create_file",
    "delete_file",
    "run_test",
    "run_lint",
    "run_typecheck",
    "run_build",
    "read_run_output",
    "read_test_failure",
}


def _candidate_payload_tool_args(action: Any) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args"), dict) else {}
    return tool_args if isinstance(tool_args, dict) else {}


def _candidate_function_name(loop: Any, action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    extractor = getattr(loop, "_extract_action_function_name", None)
    if callable(extractor):
        try:
            fn = str(extractor(action, default="") or "").strip()
            if fn:
                return fn
        except TypeError:
            fn = str(extractor(action) or "").strip()
            if fn:
                return fn
    tool_args = _candidate_payload_tool_args(action)
    for value in (
        tool_args.get("function_name"),
        action.get("function_name"),
        action.get("function"),
    ):
        fn = str(value or "").strip()
        if fn:
            return fn
    return ""


def _candidate_kwargs(action: Any) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    tool_args = _candidate_payload_tool_args(action)
    kwargs = tool_args.get("kwargs")
    if isinstance(kwargs, dict):
        return kwargs
    top_level = action.get("kwargs")
    return top_level if isinstance(top_level, dict) else {}


def _candidate_missing_required_kwargs(action: Any) -> List[str]:
    if not isinstance(action, dict):
        return []
    meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta"), dict) else {}
    values = meta.get("missing_required_kwargs", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value or "").strip()]


def _needs_kwargs_materialization(loop: Any, action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    if _candidate_kwargs(action):
        return False
    fn = _candidate_function_name(loop, action)
    if not fn:
        return False
    meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta"), dict) else {}
    kwargs_status = str(meta.get("kwargs_status", "") or "").strip()
    return (
        fn in _KWARGS_MATERIALIZATION_FUNCTIONS
        or bool(_candidate_missing_required_kwargs(action))
        or kwargs_status in {"pending_completion", "insufficient_required_kwargs"}
    )


def _current_plan_target(loop: Any) -> str:
    plan_state = getattr(loop, "_plan_state", None)
    current_step = getattr(plan_state, "current_step", None)
    target = str(getattr(current_step, "target_function", "") or "").strip()
    if target:
        return target
    getter = getattr(plan_state, "get_target_function_for_step", None)
    if callable(getter):
        try:
            return str(getter() or "").strip()
        except TypeError:
            return ""
    return ""


def _normalize_materialized_action(action: Dict[str, Any], function_name: str) -> Dict[str, Any]:
    updated = deepcopy(action)
    kwargs = dict(_candidate_kwargs(updated))
    if not kwargs:
        return updated

    updated["function_name"] = updated.get("function_name") or function_name
    updated["function"] = updated.get("function") or function_name
    updated["kwargs"] = deepcopy(kwargs)
    if kwargs.get("x") is not None:
        updated["x"] = kwargs.get("x")
    if kwargs.get("y") is not None:
        updated["y"] = kwargs.get("y")

    payload = updated.setdefault("payload", {})
    if isinstance(payload, dict):
        tool_args = payload.setdefault("tool_args", {})
        if isinstance(tool_args, dict):
            tool_args["function_name"] = function_name
            tool_args["kwargs"] = deepcopy(kwargs)

    meta = updated.setdefault("_candidate_meta", {})
    if isinstance(meta, dict):
        missing = [key for key in _candidate_missing_required_kwargs(updated) if key not in kwargs]
        meta["structured_answer_materialized_stage"] = "planner_stage"
        meta["structured_answer_materialized_before_governance"] = True
        meta["missing_required_kwargs"] = missing
        if not missing:
            meta["kwargs_status"] = "ready_from_structured_answer"
            meta["executable"] = True
            meta.pop("non_executable_reason", None)
            if updated.get("kind") == "non_executable_call":
                updated["kind"] = "call_tool"
    return updated


def _materialize_candidate_kwargs(
    loop: Any,
    candidate_actions: Sequence[Dict[str, Any]],
    obs_before: Dict[str, Any],
    *,
    limit: int = 3,
    include_plan_target: bool = True,
) -> List[Dict[str, Any]]:
    synthesizer = getattr(loop, "_structured_answer_synthesizer", None)
    populate = getattr(synthesizer, "maybe_populate_action_kwargs", None)
    if not callable(populate):
        return [dict(action) if isinstance(action, dict) else action for action in list(candidate_actions or [])]

    actions: List[Dict[str, Any]] = [
        dict(action) if isinstance(action, dict) else action
        for action in list(candidate_actions or [])
    ]
    if not actions:
        return actions

    plan_target = _current_plan_target(loop) if include_plan_target else ""
    ranked_indexes = []
    for index, action in enumerate(actions):
        if not _needs_kwargs_materialization(loop, action):
            continue
        fn = _candidate_function_name(loop, action)
        priority = 0 if plan_target and fn == plan_target else 1
        missing_count = len(_candidate_missing_required_kwargs(action))
        ranked_indexes.append((priority, -missing_count, index, fn))

    llm_client = None
    resolver = getattr(loop, "_resolve_structured_answer_llm_client", None)
    if callable(resolver):
        llm_client = resolver()

    for _priority, _missing_count, index, fn in sorted(ranked_indexes)[: max(0, int(limit))]:
        action = actions[index]
        populated = populate(
            action,
            obs_before if isinstance(obs_before, dict) else {},
            llm_client=llm_client,
        )
        if isinstance(populated, dict):
            actions[index] = _normalize_materialized_action(populated, fn)
    return actions


class PlannerStage:
    """Stage-2 planner/candidate generation without governance side effects."""

    def run(self, loop, stage_input: PlannerStageInput) -> PlannerStageOutput:
        planner_out = loop._stage2_candidate_generation_substage(
            stage_input.obs_before,
            stage_input.surfaced,
            stage_input.continuity_snapshot,
            stage_input.frame,
        )
        candidate_actions = planner_out.candidate_actions
        n_raw_candidates, n_non_wait_raw_candidates = loop._candidate_counts(candidate_actions)
        candidate_actions = _materialize_candidate_kwargs(
            loop,
            candidate_actions,
            stage_input.obs_before,
            limit=3,
            include_plan_target=True,
        )
        after_kwargs_materialization_snapshot = loop._snapshot_candidate_list(candidate_actions)

        candidate_actions = loop._stage2_plan_constraints_substage(stage_input.obs_before, candidate_actions)
        after_plan_constraints_snapshot = loop._snapshot_candidate_list(candidate_actions)
        n_after_plan_constraints, n_non_wait_after_plan_constraints = loop._candidate_counts(candidate_actions)

        candidate_actions = loop._stage2_self_model_suppression_substage(
            candidate_actions,
            stage_input.continuity_snapshot,
            stage_input.obs_before,
        )
        after_self_model_snapshot = loop._snapshot_candidate_list(candidate_actions)
        n_after_self_model, n_non_wait_after_self_model = loop._candidate_counts(candidate_actions)
        candidate_actions = loop._annotate_candidates_with_learning_updates(candidate_actions, stage_input.continuity_snapshot)

        loop._annotate_candidates_with_counterfactual(candidate_actions, stage_input.continuity_snapshot)
        candidate_actions = loop._counterfactual_rank_candidates(candidate_actions)
        after_counterfactual_rank_snapshot = loop._snapshot_candidate_list(candidate_actions)

        after_procedure_annotation_snapshot = after_counterfactual_rank_snapshot
        if loop._procedure_enabled and candidate_actions:
            available_functions = stage_input.obs_before.get('novel_api', {}).get('visible_functions', []) if isinstance(stage_input.obs_before.get('novel_api', {}), dict) else []
            completed_functions = []
            for row in loop._episode_trace[-5:]:
                action = row.get('action', {}) if isinstance(row, dict) else {}
                fn = loop._extract_action_function_name(action, default='')
                if fn:
                    completed_functions.append(fn)
            proc_context = {
                'available_functions': list(available_functions),
                'completed_functions': completed_functions,
                'task_family': loop._infer_task_family(stage_input.obs_before),
            }
            procedure_matches = loop._procedure_matcher.match(context=proc_context, registry=loop._procedure_registry, top_k=5)
            loop._last_procedure_matches = procedure_matches
            candidate_actions = loop._procedure_executor.annotate_candidates(candidate_actions, procedure_matches=procedure_matches)
            candidate_actions = _materialize_candidate_kwargs(
                loop,
                candidate_actions,
                stage_input.obs_before,
                limit=3,
                include_plan_target=True,
            )
            after_procedure_annotation_snapshot = loop._snapshot_candidate_list(candidate_actions)

        loop._stage2_prediction_runtime_substage(candidate_actions)
        bridge_out = loop._stage2_prediction_context_bridge_substage(
            loop._decision_bridge_input_cls(
                obs_before=stage_input.obs_before,
                surfaced=stage_input.surfaced,
                continuity_snapshot=stage_input.continuity_snapshot,
                plan_tick_meta=planner_out.plan_tick_meta,
                candidate_actions=candidate_actions,
                frame=stage_input.frame,
                deliberation_result=planner_out.deliberation_result,
            )
        )

        return PlannerStageOutput(
            raw_base_action=planner_out.raw_base_action,
            base_action=planner_out.base_action,
            arm_action=planner_out.arm_action,
            arm_meta=planner_out.arm_meta,
            plan_tick_meta=planner_out.plan_tick_meta,
            candidate_actions=candidate_actions,
            visible_functions=planner_out.visible_functions,
            discovered_functions=planner_out.discovered_functions,
            raw_candidates_snapshot=planner_out.raw_candidates_snapshot,
            decision_context=bridge_out.get('decision_context', {}),
            stage_metrics={
                'n_raw_candidates': n_raw_candidates,
                'n_non_wait_raw_candidates': n_non_wait_raw_candidates,
                'after_kwargs_materialization': after_kwargs_materialization_snapshot,
                'after_plan_constraints': after_plan_constraints_snapshot,
                'n_after_plan_constraints': n_after_plan_constraints,
                'n_non_wait_after_plan_constraints': n_non_wait_after_plan_constraints,
                'after_self_model': after_self_model_snapshot,
                'n_after_self_model': n_after_self_model,
                'n_non_wait_after_self_model': n_non_wait_after_self_model,
                'after_counterfactual_rank': after_counterfactual_rank_snapshot,
                'after_procedure_annotation': after_procedure_annotation_snapshot,
            },
            deliberation_result=dict(planner_out.deliberation_result or {}),
        )
