from __future__ import annotations

from core.orchestration.stage_types import PlannerStageInput, PlannerStageOutput


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
