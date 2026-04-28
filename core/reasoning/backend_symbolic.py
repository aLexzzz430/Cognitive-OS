from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.orchestration.action_utils import (
    action_semantic_signature,
    extract_action_function_name,
    extract_action_kind,
)
from core.reasoning.answer_critic import rank_candidate_outputs
from core.reasoning.backend import DeliberationBudget, ReasoningRequest, ReasoningResult
from core.reasoning.candidate_output_search import search_candidate_outputs
from core.reasoning.candidate_program_search import search_candidate_programs
from core.reasoning.discriminating_experiment import build_discriminating_experiments
from core.reasoning.hypothesis_schema import hypothesis_action_prediction, normalize_hypothesis_rows
from core.reasoning.hypothesis_competition import rank_hypotheses
from core.reasoning.test_designer import design_candidate_tests
from modules.world_model.mechanism_runtime import (
    build_mechanism_runtime_state,
    evaluate_mechanism_preconditions,
)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _string_tokens(*values: Any, limit: int = 12) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for token in _string_tokens(*list(value), limit=limit):
                if token not in seen:
                    seen.add(token)
                    ordered.append(token)
                    if len(ordered) >= limit:
                        return ordered[:limit]
            continue
        text = str(value or "").strip().lower()
        if not text:
            continue
        canonical = text.replace("::", "_").replace("-", "_").replace(" ", "_")
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
            if len(ordered) >= limit:
                return ordered[:limit]
        for raw in canonical.split("_"):
            token = str(raw or "").strip().lower()
            if token and token not in seen:
                seen.add(token)
                ordered.append(token)
                if len(ordered) >= limit:
                    return ordered[:limit]
    return ordered[:limit]


def _normalize_action_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text in {
        "pointer_interaction",
        "confirm_interaction",
        "navigation_interaction",
        "state_transform_interaction",
        "probe_interaction",
        "wait",
    }:
        return text
    upper = text.upper()
    if upper in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
        return "navigation_interaction"
    if upper in {"ACTION5", "CONFIRM", "INTERACT", "SUBMIT", "ENTER", "APPLY"}:
        return "confirm_interaction"
    if upper in {"ACTION6", "CLICK", "TAP", "POINTER_CLICK", "POINTER_SELECT", "POINTER_ACTIVATE", "SELECT"}:
        return "pointer_interaction"
    if upper in {"ACTION7", "PROBE", "PROBE_STATE_CHANGE", "PROBE_RELATION", "DRAG", "TOGGLE", "TRANSFORM"}:
        return "state_transform_interaction"
    if "nav" in text or text in {"move", "left", "right", "up", "down", "focus"}:
        return "navigation_interaction"
    if "confirm" in text or "submit" in text or "interact" in text:
        return "confirm_interaction"
    if "pointer" in text or "click" in text or "tap" in text or "select" in text:
        return "pointer_interaction"
    if "probe" in text or "transform" in text or "toggle" in text:
        return "state_transform_interaction"
    return text


def _action_family(action: Dict[str, Any], fn_name: str, kind: str) -> str:
    meta = _as_dict(action.get("_candidate_meta", {}))
    for key in ("action_family", "runtime_action_family", "solver_dominant_interaction_mode"):
        family = _normalize_action_family(meta.get(key))
        if family:
            return family
    if kind == "wait" or fn_name == "wait":
        return "wait"
    return _normalize_action_family(fn_name)


def _token_overlap(expected: Sequence[str], actual: Sequence[str]) -> float:
    expected_set = {str(item or "").strip().lower() for item in list(expected or []) if str(item or "").strip()}
    actual_set = {str(item or "").strip().lower() for item in list(actual or []) if str(item or "").strip()}
    if not expected_set or not actual_set:
        return 0.0
    return len(expected_set & actual_set) / float(max(1, len(expected_set)))


def _workspace_world_model_summary(workspace: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("active_beliefs_summary", "world_model_summary"):
        raw = workspace.get(key, {})
        if isinstance(raw, dict):
            return raw
    return {}


def _hypothesis_disagreement_pressure(
    ranked_hypotheses: Sequence[Dict[str, Any]],
    ranked_experiments: Sequence[Dict[str, Any]],
) -> float:
    posterior_values = sorted(
        (
            _clamp01(row.get("posterior", row.get("confidence", 0.0)))
            for row in list(ranked_hypotheses or [])
            if isinstance(row, dict)
        ),
        reverse=True,
    )
    top_experiment_score = _clamp01(
        ranked_experiments[0].get("score", 0.0),
    ) if ranked_experiments and isinstance(ranked_experiments[0], dict) else 0.0
    if len(posterior_values) < 2:
        return top_experiment_score
    leading_gap = max(0.0, posterior_values[0] - posterior_values[1])
    gap_pressure = max(0.0, min(1.0, 1.0 - (leading_gap / 0.35)))
    return max(top_experiment_score, gap_pressure)


def _deliberation_control_policy(
    *,
    workspace: Dict[str, Any],
    ranked_hypotheses: Sequence[Dict[str, Any]],
    ranked_experiments: Sequence[Dict[str, Any]],
    probe_before_commit: bool,
) -> Dict[str, Any]:
    uncertainty = _clamp01(
        (workspace.get("uncertainty_vector", {}) if isinstance(workspace.get("uncertainty_vector", {}), dict) else {}).get("overall", 0.0)
    )
    world_shift_risk = _clamp01(workspace.get("world_shift_risk", 0.0))
    hypothesis_disagreement = _hypothesis_disagreement_pressure(
        ranked_hypotheses,
        ranked_experiments,
    )
    commit_risk_pressure = max(
        world_shift_risk,
        0.82 if probe_before_commit else 0.0,
    )
    experiment_priority_pressure = (
        0.28
        + (hypothesis_disagreement * 0.34)
        + (commit_risk_pressure * 0.24)
    ) if (
        uncertainty >= 0.72
        and hypothesis_disagreement >= 0.55
        and commit_risk_pressure >= 0.45
    ) else 0.0
    switch_reasons: List[str] = []
    if uncertainty >= 0.72:
        switch_reasons.append("high_uncertainty")
    if hypothesis_disagreement >= 0.55:
        switch_reasons.append("high_hypothesis_disagreement")
    if commit_risk_pressure >= 0.45:
        switch_reasons.append("high_commit_risk")
    if probe_before_commit:
        switch_reasons.append("probe_before_commit")
    top_experiment = dict(ranked_experiments[0]) if ranked_experiments else {}
    return {
        "strategy": "experiment_first" if experiment_priority_pressure > 0.0 else "reward_first",
        "uncertainty": round(uncertainty, 6),
        "world_shift_risk": round(world_shift_risk, 6),
        "hypothesis_disagreement": round(hypothesis_disagreement, 6),
        "commit_risk_pressure": round(commit_risk_pressure, 6),
        "experiment_priority_pressure": round(experiment_priority_pressure, 6),
        "switch_reasons": switch_reasons,
        "probe_before_commit": bool(probe_before_commit),
        "top_experiment_function": str(top_experiment.get("function_name", "") or ""),
        "top_experiment_score": round(float(top_experiment.get("score", 0.0) or 0.0), 6),
    }


def _experiment_hypothesis_pool(
    workspace: Dict[str, Any],
    ranked_hypotheses: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    runtime_pool = normalize_hypothesis_rows(
        [
            dict(row)
            for row in _as_list(workspace.get("competing_hypotheses", []))
            if isinstance(row, dict)
        ],
        fallback_id_prefix="exp",
    )
    if runtime_pool:
        return runtime_pool
    return normalize_hypothesis_rows(
        [dict(row) for row in list(ranked_hypotheses or []) if isinstance(row, dict)],
        fallback_id_prefix="exp",
    )


def _experiment_builder_limits(
    *,
    workspace: Dict[str, Any],
    budget: DeliberationBudget,
    hypothesis_pool_count: int,
    action_pool_count: int,
) -> Dict[str, int]:
    uncertainty = _clamp01(
        _as_dict(workspace.get("uncertainty_vector", {})).get("overall", 0.0),
    )
    uncertainty_bonus = 2 if uncertainty >= 0.72 else 0
    hypothesis_limit = min(
        max(0, int(hypothesis_pool_count)),
        max(
            int(budget.hypothesis_limit) + int(budget.test_limit) + uncertainty_bonus,
            int(budget.hypothesis_limit) * 2,
            6,
        ),
    )
    action_limit = min(
        max(0, int(action_pool_count)),
        max(
            int(budget.test_limit) * 3 + uncertainty_bonus,
            int(budget.branch_budget) + int(budget.test_limit),
            4,
        ),
    )
    pair_budget = max(
        int(budget.test_limit) * 4 + uncertainty_bonus,
        int(budget.hypothesis_limit) * 2,
        int(budget.branch_budget) * 2,
        4,
    )
    return {
        "hypothesis_limit": max(0, int(hypothesis_limit)),
        "action_limit": max(0, int(action_limit)),
        "pair_budget": max(1, int(pair_budget)),
    }


def _experiment_builder_trace(
    experiments: Sequence[Dict[str, Any]],
    *,
    hypothesis_pool_count: int,
    action_pool_count: int,
    previous_experiment_count: int,
    limits: Dict[str, int],
) -> Dict[str, Any]:
    sample = next((dict(row) for row in list(experiments or []) if isinstance(row, dict)), {})
    budget_audit = _as_dict(sample.get("_builder_budget_audit", {}))
    hypothesis_selection = _as_dict(budget_audit.get("hypothesis_selection", {}))
    action_selection = _as_dict(budget_audit.get("action_selection", {}))
    pair_selection = _as_dict(budget_audit.get("pair_selection", {}))
    loss_estimate = _as_dict(budget_audit.get("budget_loss_estimate", {}))
    return {
        "hypothesis_pool_count": int(hypothesis_pool_count),
        "action_pool_count": int(action_pool_count),
        "previous_experiment_count": int(previous_experiment_count),
        "prefilter_hypothesis_limit": int(limits.get("hypothesis_limit", 0) or 0),
        "prefilter_action_limit": int(limits.get("action_limit", 0) or 0),
        "pair_budget": int(limits.get("pair_budget", 0) or 0),
        "prefiltered_hypothesis_count": int(sample.get("_builder_hypothesis_count", 0) or 0),
        "prefiltered_action_count": int(sample.get("_builder_action_count", 0) or 0),
        "evaluated_pair_count": int(sample.get("_builder_pair_count", 0) or 0),
        "pruned_hypothesis_count": int(hypothesis_selection.get("pruned_count", 0) or 0),
        "pruned_action_count": int(action_selection.get("pruned_count", 0) or 0),
        "pruned_pair_count": int(pair_selection.get("pruned_count", 0) or 0),
        "pair_budget_used_count": int(pair_selection.get("kept_count", 0) or 0),
        "budget_loss_estimate_total": round(float(loss_estimate.get("total", 0.0) or 0.0), 6),
        "pair_information_loss_estimate": round(
            float(loss_estimate.get("pair_information_gain_pruned", 0.0) or 0.0),
            6,
        ),
        "cache_hit_count": sum(
            1
            for row in list(experiments or [])
            if isinstance(row, dict) and bool(row.get("_builder_cache_hit", False))
        ),
    }


def _mechanism_control_summary(workspace: Dict[str, Any]) -> Dict[str, Any]:
    summary = _workspace_world_model_summary(workspace)
    control = summary.get("mechanism_control_summary", workspace.get("mechanism_control_summary", {}))
    return dict(control) if isinstance(control, dict) else {}


def _is_mechanism_row(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    metadata = _as_dict(row.get("metadata", {}))
    source = str(row.get("source", "") or "").strip().lower()
    hypothesis_id = str(row.get("hypothesis_id", row.get("object_id", "")) or "")
    return bool(
        metadata.get("mechanism_hypothesis", False)
        or source == "world_model_mechanism"
        or hypothesis_id.startswith("mech_")
    )


def _mechanism_rows(
    workspace: Dict[str, Any],
    ranked_hypotheses: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summary = _workspace_world_model_summary(workspace)
    rows: List[Dict[str, Any]] = []
    for row in _as_list(summary.get("mechanism_hypotheses", summary.get("mechanism_hypotheses_summary", []))):
        if isinstance(row, dict):
            rows.append(dict(row))
    for row in list(ranked_hypotheses or []):
        if _is_mechanism_row(row):
            rows.append(dict(row))
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        hypothesis_id = str(row.get("hypothesis_id", row.get("object_id", "")) or "")
        dedupe_key = hypothesis_id or str(row.get("summary", "") or "")
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(row)
    deduped.sort(
        key=lambda item: (
            -float(item.get("posterior", item.get("confidence", 0.0)) or 0.0),
            str(item.get("hypothesis_id", item.get("object_id", "")) or ""),
        )
    )
    return deduped


def _action_binding_tokens(action: Dict[str, Any], fn_name: str, action_family: str) -> List[str]:
    meta = _as_dict(action.get("_candidate_meta", {}))
    intervention_target = _as_dict(meta.get("intervention_target", {}))
    return _string_tokens(
        meta.get("grounded_binding_tokens", []),
        meta.get("anchor_ref", ""),
        intervention_target.get("anchor_ref", ""),
        intervention_target.get("target_kind", ""),
        fn_name,
        action_family,
        limit=12,
    )


def _build_deliberation_posterior_summary(
    *,
    workspace: Dict[str, Any],
    ranked_hypotheses: Sequence[Dict[str, Any]],
    rejected_hypotheses: Sequence[Dict[str, Any]],
    ranked_experiments: Sequence[Dict[str, Any]],
    ranked_tests: Sequence[Dict[str, Any]],
    ranked_actions: Sequence[Dict[str, Any]],
    budget_mode: str,
    probe_before_commit: bool,
) -> Dict[str, Any]:
    previous_summary = (
        dict(workspace.get("posterior_summary", {}) or {})
        if isinstance(workspace.get("posterior_summary", {}), dict)
        else {}
    )
    deliberation_snapshot = (
        dict(previous_summary.get("deliberation_snapshot", {}) or {})
        if isinstance(previous_summary.get("deliberation_snapshot", {}), dict)
        else {}
    )
    top_experiment = dict(ranked_experiments[0]) if ranked_experiments else {}
    top_test = dict(ranked_tests[0]) if ranked_tests else {}
    top_action = dict(ranked_actions[0]) if ranked_actions else {}
    leading_hypothesis = dict(ranked_hypotheses[0]) if ranked_hypotheses else {}
    deliberation_snapshot.update(
        {
            "mode": str(budget_mode or ""),
            "probe_before_commit": bool(probe_before_commit),
            "ranked_hypothesis_count": len([row for row in ranked_hypotheses if isinstance(row, dict)]),
            "ranked_experiment_count": len([row for row in ranked_experiments if isinstance(row, dict)]),
            "ranked_test_count": len([row for row in ranked_tests if isinstance(row, dict)]),
            "ranked_action_count": len([row for row in ranked_actions if isinstance(row, dict)]),
            "top_experiment_function": str(top_experiment.get("function_name", "") or ""),
            "top_test_function": str(top_test.get("function_name", "") or ""),
            "top_action_function": extract_action_function_name(top_action, default=""),
            "leading_hypothesis_status": str(leading_hypothesis.get("status", "") or ""),
        }
    )
    posterior_summary = dict(previous_summary)
    posterior_summary.pop("runtime_object_graph", None)
    posterior_summary.update(
        {
            "leading_hypothesis_id": str(leading_hypothesis.get("hypothesis_id", "") or ""),
            "leading_posterior": round(float(leading_hypothesis.get("posterior", 0.0) or 0.0), 6) if leading_hypothesis else 0.0,
            "rejected_count": len([row for row in rejected_hypotheses if isinstance(row, dict)]),
            "updated_count": len([row for row in ranked_hypotheses if isinstance(row, dict)]),
            "support_events": 0,
            "contradiction_events": 0,
            "unresolved_events": len(
                [
                    row
                    for row in ranked_hypotheses
                    if isinstance(row, dict) and str(row.get("status", "") or "") == "unresolved"
                ]
            ),
            "summary_stage": "deliberation",
            "last_update_source": "deliberation",
            "deliberation_snapshot": deliberation_snapshot,
        }
    )
    return posterior_summary


class SymbolicReasoningBackend:
    name = "symbolic"

    def deliberate(
        self,
        request: ReasoningRequest,
        budget: DeliberationBudget,
    ) -> ReasoningResult:
        workspace = dict(request.workspace or {})
        candidate_actions = [deepcopy(action) for action in request.candidate_actions if isinstance(action, dict)]
        previous_experiments = [
            dict(row)
            for row in _as_list(workspace.get("ranked_discriminating_experiments", []))
            if isinstance(row, dict)
        ]
        ranked_hypotheses, rejected_hypotheses = rank_hypotheses(workspace, limit=budget.hypothesis_limit)
        experiment_hypothesis_pool = _experiment_hypothesis_pool(workspace, ranked_hypotheses)
        experiment_builder_limits = _experiment_builder_limits(
            workspace=workspace,
            budget=budget,
            hypothesis_pool_count=len(experiment_hypothesis_pool),
            action_pool_count=len(candidate_actions),
        )
        ranked_discriminating_experiments = build_discriminating_experiments(
            experiment_hypothesis_pool,
            candidate_actions,
            limit=budget.test_limit,
            hypothesis_limit=experiment_builder_limits["hypothesis_limit"],
            action_limit=experiment_builder_limits["action_limit"],
            pair_budget=experiment_builder_limits["pair_budget"],
            previous_experiments=previous_experiments,
        )
        experiment_builder_trace = _experiment_builder_trace(
            ranked_discriminating_experiments,
            hypothesis_pool_count=len(experiment_hypothesis_pool),
            action_pool_count=len(candidate_actions),
            previous_experiment_count=len(previous_experiments),
            limits=experiment_builder_limits,
        )
        workspace["competing_hypotheses"] = [dict(item) for item in ranked_hypotheses]
        workspace["ranked_discriminating_experiments"] = [dict(item) for item in ranked_discriminating_experiments]
        ranked_tests, injected_probe_actions, probe_before_commit, active_test_ids = design_candidate_tests(
            workspace,
            request.candidate_actions,
            available_functions=request.available_functions,
            limit=budget.test_limit,
        )
        control_policy = _deliberation_control_policy(
            workspace=workspace,
            ranked_hypotheses=ranked_hypotheses,
            ranked_experiments=ranked_discriminating_experiments,
            probe_before_commit=probe_before_commit,
        )
        candidate_programs = search_candidate_programs(
            workspace=workspace,
            obs=request.obs,
            synthesizer=request.structured_answer_synthesizer,
            limit=budget.program_limit,
        )
        candidate_outputs = rank_candidate_outputs(
            search_candidate_outputs(
                workspace=workspace,
                obs=request.obs,
                candidate_programs=candidate_programs,
                synthesizer=request.structured_answer_synthesizer,
                limit=budget.output_limit,
            )
        )

        actions = [deepcopy(action) for action in candidate_actions]
        actions.extend(injected_probe_actions)
        ranked_actions, rollout_predictions, rejected_actions = self._rank_actions(
            actions,
            workspace=workspace,
            ranked_hypotheses=ranked_hypotheses,
            ranked_experiments=ranked_discriminating_experiments,
            ranked_tests=ranked_tests,
            probe_before_commit=probe_before_commit,
            control_policy=control_policy,
            depth=budget.depth,
        )
        posterior_summary = _build_deliberation_posterior_summary(
            workspace=workspace,
            ranked_hypotheses=ranked_hypotheses,
            rejected_hypotheses=rejected_hypotheses,
            ranked_experiments=ranked_discriminating_experiments,
            ranked_tests=ranked_tests,
            ranked_actions=ranked_actions,
            budget_mode=budget.mode,
            probe_before_commit=probe_before_commit,
        )

        trace = [
            {
                "stage": "budget",
                "mode": budget.mode,
                "depth": budget.depth,
                "verification_budget": budget.verification_budget,
            },
            {
                "stage": "hypothesis_competition",
                "ranked_count": len(ranked_hypotheses),
                "rejected_count": len(rejected_hypotheses),
            },
            {
                "stage": "test_design",
                "ranked_count": len(ranked_tests),
                "injected_actions": len(injected_probe_actions),
                "active_tests": len(active_test_ids),
                "probe_before_commit": probe_before_commit,
            },
            {
                "stage": "deliberation_control_policy",
                **dict(control_policy),
            },
            {
                "stage": "discriminating_experiment_selection",
                "ranked_count": len(ranked_discriminating_experiments),
                **dict(experiment_builder_trace),
            },
            {
                "stage": "program_search",
                "ranked_count": len(candidate_programs),
            },
            {
                "stage": "output_search",
                "ranked_count": len(candidate_outputs),
            },
        ]

        rejected = list(rejected_hypotheses)
        rejected.extend(rejected_actions)

        return ReasoningResult(
            ranked_candidate_actions=ranked_actions,
            ranked_candidate_hypotheses=ranked_hypotheses,
            ranked_discriminating_experiments=ranked_discriminating_experiments,
            ranked_candidate_tests=ranked_tests,
            ranked_candidate_programs=candidate_programs,
            ranked_candidate_outputs=candidate_outputs,
            active_test_ids=active_test_ids,
            deliberation_trace=trace,
            rejected_candidates=rejected,
            rollout_predictions=rollout_predictions,
            posterior_summary=posterior_summary,
            control_policy=control_policy,
            budget=budget.to_dict(),
            backend=self.name,
            mode=budget.mode,
            probe_before_commit=probe_before_commit,
        )

    def _rank_actions(
        self,
        actions: Sequence[Dict[str, Any]],
        *,
        workspace: Dict[str, Any],
        ranked_hypotheses: Sequence[Dict[str, Any]],
        ranked_experiments: Sequence[Dict[str, Any]],
        ranked_tests: Sequence[Dict[str, Any]],
        probe_before_commit: bool,
        control_policy: Optional[Dict[str, Any]] = None,
        depth: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        if not isinstance(control_policy, dict):
            control_policy = _deliberation_control_policy(
                workspace=workspace,
                ranked_hypotheses=ranked_hypotheses,
                ranked_experiments=ranked_experiments,
                probe_before_commit=probe_before_commit,
            )
        world_shift_risk = _clamp01(workspace.get("world_shift_risk", 0.0))
        uncertainty = _clamp01(
            (workspace.get("uncertainty_vector", {}) if isinstance(workspace.get("uncertainty_vector", {}), dict) else {}).get("overall", 0.0)
        )
        active_skill_fns = {
            str(row.get("function_name", "") or "")
            for row in list(workspace.get("active_skills", []) or [])
            if isinstance(row, dict)
        }
        leading_hypothesis = dict(ranked_hypotheses[0]) if ranked_hypotheses else {}
        experiment_signatures = {
            self._action_signature(row.get("candidate_action", {})): dict(row)
            for row in list(ranked_experiments or [])
            if isinstance(row, dict)
        }
        hypothesis_disagreement = _clamp01(control_policy.get("hypothesis_disagreement", 0.0))
        commit_risk_pressure = _clamp01(control_policy.get("commit_risk_pressure", world_shift_risk))
        experiment_priority_pressure = _clamp01(control_policy.get("experiment_priority_pressure", 0.0), 0.0)
        experiment_priority_mode = str(control_policy.get("strategy", "") or "") == "experiment_first"
        switch_reasons = [
            str(item or "")
            for item in list(control_policy.get("switch_reasons", []) or [])
            if str(item or "")
        ]
        top_test_functions = [
            str(row.get("function_name", "") or "")
            for row in list(ranked_tests or [])
            if isinstance(row, dict) and str(row.get("function_name", "") or "")
        ]

        ranked: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        rollout_predictions: Dict[str, Dict[str, Any]] = {}
        for action in actions:
            fn_name = extract_action_function_name(action, default="wait")
            kind = extract_action_kind(action, default="call_tool")
            is_probe = kind == "probe" or any(token in fn_name.lower() for token in ("probe", "inspect", "check", "test"))
            meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
            experiment_row = experiment_signatures.get(self._action_signature(action), {})
            discriminating_bonus = float(experiment_row.get("score", 0.0) or 0.0) * 0.25 if experiment_row else 0.0
            experiment_candidate = bool(experiment_row) or fn_name in top_test_functions
            leading_alignment, leading_conflict_penalty = self._leading_hypothesis_action_adjustment(
                leading_hypothesis,
                action,
            )
            counterfactual_delta = float(meta.get("counterfactual_delta", 0.0) or 0.0)
            prediction = meta.get("prediction", {}) if isinstance(meta.get("prediction", {}), dict) else {}
            predicted_success = _clamp01(((prediction.get("success", {}) or {}).get("value", 0.5) if isinstance(prediction.get("success", {}), dict) else 0.5), default=0.5)
            structured_bonus = 0.18 if bool(meta.get("structured_answer_synthesized")) else 0.0
            skill_bonus = 0.10 if fn_name in active_skill_fns else 0.0
            test_bonus = 0.12 if fn_name in top_test_functions else 0.0
            mechanism_projection = self._mechanism_transition_projection(
                action,
                workspace=workspace,
                ranked_hypotheses=ranked_hypotheses,
            )
            info_gain = max(0.0, min(1.0, abs(counterfactual_delta) + (0.25 if is_probe else 0.0) + (0.15 if fn_name in top_test_functions else 0.0)))
            reversibility = 0.85 if fn_name == "wait" or is_probe else 0.45
            long_reward = max(0.0, predicted_success * 0.55 + structured_bonus + skill_bonus)
            short_reward = counterfactual_delta + (0.05 if predicted_success >= 0.6 else -0.02)
            risk = max(0.0, min(1.0, world_shift_risk * (0.4 if not is_probe else 0.2) + (0.08 if fn_name == "wait" else 0.15 if not is_probe else 0.02)))
            info_gain = max(info_gain, float(mechanism_projection.get("info_gain_floor", 0.0) or 0.0))
            reversibility = max(reversibility, float(mechanism_projection.get("reversibility_floor", 0.0) or 0.0))
            long_reward = max(0.0, long_reward + float(mechanism_projection.get("long_reward_bonus", 0.0) or 0.0))
            short_reward += float(mechanism_projection.get("short_reward_bonus", 0.0) or 0.0)
            risk = max(0.0, min(1.0, risk + float(mechanism_projection.get("risk_delta", 0.0) or 0.0)))
            experiment_priority_bonus = 0.0
            reward_first_penalty = 0.0
            if experiment_priority_mode:
                if experiment_candidate:
                    experiment_priority_bonus = (
                        0.32
                        + (experiment_priority_pressure * 0.28)
                        + (_clamp01(experiment_row.get("score", 0.0)) * 0.18 if experiment_row else 0.0)
                    )
                    info_gain = max(
                        info_gain,
                        0.46 + min(0.34, hypothesis_disagreement * 0.30),
                    )
                    reversibility = max(reversibility, 0.74 if is_probe else 0.52)
                elif fn_name != "wait":
                    reward_first_penalty = (
                        0.24
                        + (experiment_priority_pressure * 0.18)
                        + (0.08 if not is_probe else 0.0)
                    )
            score = short_reward + (0.6 * long_reward) + (0.22 * info_gain) + (0.12 * reversibility) + test_bonus
            if probe_before_commit:
                score += 0.22 if is_probe else -0.10
                risk -= 0.10 if is_probe else 0.04
            score += (
                skill_bonus
                + structured_bonus
                + discriminating_bonus
                + experiment_priority_bonus
                + leading_alignment
                + float(mechanism_projection.get("alignment_bonus", 0.0) or 0.0)
            )
            score -= (
                leading_conflict_penalty
                + reward_first_penalty
                + float(mechanism_projection.get("penalty", 0.0) or 0.0)
            )
            score += max(0.0, 0.04 * min(4, depth))
            if score < -0.2 and not is_probe and fn_name != "wait":
                rejected.append({
                    "candidate_type": "action",
                    "function_name": fn_name,
                    "reason": "low_deliberation_score",
                    "score": round(score, 4),
                })
                continue
            enriched = deepcopy(action)
            action_meta = enriched.get("_candidate_meta", {}) if isinstance(enriched.get("_candidate_meta", {}), dict) else {}
            action_meta["deliberation_engine_score"] = round(score, 6)
            action_meta["deliberation_engine_probe_before_commit"] = bool(probe_before_commit)
            action_meta["deliberation_engine_top_test_match"] = fn_name in top_test_functions
            action_meta["deliberation_engine_strategy"] = "experiment_first" if experiment_priority_mode else "reward_first"
            action_meta["experiment_priority_pressure"] = round(experiment_priority_pressure, 6)
            action_meta["experiment_priority_bonus"] = round(experiment_priority_bonus, 6)
            action_meta["reward_first_penalty"] = round(reward_first_penalty, 6)
            action_meta["control_policy_switch_reasons"] = list(switch_reasons)
            action_meta["control_policy_commit_risk_pressure"] = round(commit_risk_pressure, 6)
            action_meta["discriminating_experiment_score"] = round(discriminating_bonus, 6)
            action_meta["leading_hypothesis_alignment"] = round(leading_alignment, 6)
            action_meta["hypothesis_conflict_penalty"] = round(leading_conflict_penalty, 6)
            action_meta["mechanism_transition_alignment"] = round(float(mechanism_projection.get("alignment_bonus", 0.0) or 0.0), 6)
            action_meta["mechanism_transition_penalty"] = round(float(mechanism_projection.get("penalty", 0.0) or 0.0), 6)
            action_meta["mechanism_transition_support_mass"] = round(float(mechanism_projection.get("support_mass", 0.0) or 0.0), 6)
            action_meta["mechanism_transition_phase_shift"] = str(mechanism_projection.get("projected_phase_shift", "") or "")
            action_meta["mechanism_transition_rule_index"] = int(mechanism_projection.get("matched_rule_index", -1))
            action_meta["mechanism_precondition_satisfied"] = bool(mechanism_projection.get("precondition_satisfied", False))
            action_meta["mechanism_unmet_preconditions"] = [
                str(item or "")
                for item in list(mechanism_projection.get("unmet_preconditions", []) or [])
                if str(item or "")
            ][:4]
            action_meta["mechanism_transition_reasons"] = [
                str(item or "")
                for item in list(mechanism_projection.get("reasons", []) or [])
                if str(item or "")
            ][:6]
            enriched["_candidate_meta"] = action_meta
            ranked.append(enriched)
            rollout_predictions[fn_name or f"candidate_{len(rollout_predictions)}"] = {
                "short_reward": round(short_reward, 6),
                "long_reward": round(long_reward, 6),
                "info_gain": round(info_gain, 6),
                "reversibility": round(reversibility, 6),
                "risk": round(max(0.0, min(1.0, risk + uncertainty * 0.15)), 6),
                "mechanism_alignment_bonus": round(float(mechanism_projection.get("alignment_bonus", 0.0) or 0.0), 6),
                "mechanism_penalty": round(float(mechanism_projection.get("penalty", 0.0) or 0.0), 6),
                "mechanism_support_mass": round(float(mechanism_projection.get("support_mass", 0.0) or 0.0), 6),
                "mechanism_matched_rule_index": int(mechanism_projection.get("matched_rule_index", -1)),
                "mechanism_projected_phase_shift": str(mechanism_projection.get("projected_phase_shift", "") or ""),
                "mechanism_precondition_satisfied": bool(mechanism_projection.get("precondition_satisfied", False)),
                "mechanism_unmet_preconditions": [
                    str(item or "")
                    for item in list(mechanism_projection.get("unmet_preconditions", []) or [])
                    if str(item or "")
                ][:4],
                "mechanism_matched_hypothesis_ids": [
                    str(item or "")
                    for item in list(mechanism_projection.get("matched_hypothesis_ids", []) or [])
                    if str(item or "")
                ][:4],
            }

        ranked.sort(
            key=lambda item: float(
                ((item.get("_candidate_meta", {}) if isinstance(item.get("_candidate_meta", {}), dict) else {}).get("deliberation_engine_score", 0.0) or 0.0)
            ),
            reverse=True,
        )
        for index, item in enumerate(ranked):
            meta = item.get("_candidate_meta", {}) if isinstance(item.get("_candidate_meta", {}), dict) else {}
            meta["deliberation_engine_rank"] = index + 1
            item["_candidate_meta"] = meta
        return ranked, rollout_predictions, rejected

    @staticmethod
    def _action_signature(action: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        return action_semantic_signature(action)

    @staticmethod
    def _leading_hypothesis_action_adjustment(
        leading_hypothesis: Dict[str, Any],
        action: Dict[str, Any],
    ) -> Tuple[float, float]:
        if not isinstance(leading_hypothesis, dict):
            return 0.0, 0.0
        fn_name = extract_action_function_name(action, default="")
        if not fn_name:
            return 0.0, 0.0
        prediction = hypothesis_action_prediction(leading_hypothesis, action)
        if not prediction:
            return 0.0, 0.0
        reward_sign = str(prediction.get("reward_sign", prediction.get("predicted_reward_sign", "")) or "").strip().lower()
        valid_state_change = prediction.get("valid_state_change")
        info_gain = _clamp01(prediction.get("predicted_information_gain", 0.0))
        alignment = 0.0
        penalty = 0.0
        if reward_sign in {"positive", "zero"}:
            alignment += 0.08
        elif reward_sign == "negative":
            penalty += 0.12
        if valid_state_change is True:
            alignment += 0.04
        elif valid_state_change is False:
            penalty += 0.05
        alignment += info_gain * 0.06
        return round(alignment, 6), round(penalty, 6)

    @staticmethod
    def _mechanism_transition_projection(
        action: Dict[str, Any],
        *,
        workspace: Dict[str, Any],
        ranked_hypotheses: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        fn_name = extract_action_function_name(action, default="")
        kind = extract_action_kind(action, default="call_tool")
        if not fn_name:
            return {
                "alignment_bonus": 0.0,
                "penalty": 0.0,
                "short_reward_bonus": 0.0,
                "long_reward_bonus": 0.0,
                "info_gain_floor": 0.0,
                "reversibility_floor": 0.0,
                "risk_delta": 0.0,
                "support_mass": 0.0,
                "matched_rule_index": -1,
                "projected_phase_shift": "",
                "matched_hypothesis_ids": [],
                "reasons": [],
            }
        action_family = _action_family(action, fn_name, kind)
        action_tokens = _action_binding_tokens(action, fn_name, action_family)
        mechanisms = _mechanism_rows(workspace, ranked_hypotheses)
        mechanism_control = _mechanism_control_summary(workspace)
        obs = _as_dict(workspace.get("obs_before", workspace.get("obs", {})))
        control_mode = str(mechanism_control.get("control_mode", "") or "").strip().lower()

        alignment_bonus = 0.0
        penalty = 0.0
        short_reward_bonus = 0.0
        long_reward_bonus = 0.0
        info_gain_floor = 0.0
        reversibility_floor = 0.0
        risk_delta = 0.0
        support_mass = 0.0
        matched_rule_index = -1
        projected_phase_shift = ""
        matched_ids: List[str] = []
        reasons: List[str] = []
        matched_preconditions: List[str] = []
        unmet_preconditions: List[str] = []
        precondition_satisfied = False

        for row in mechanisms[:4]:
            prediction = hypothesis_action_prediction(row, action)
            metadata = _as_dict(row.get("metadata", {}))
            transition_rules = [
                dict(item)
                for item in _as_list(metadata.get("transition_rules", []))
                if isinstance(item, dict)
            ]
            preferred_action_families = [
                _normalize_action_family(item)
                for item in _as_list(metadata.get("preferred_action_families", row.get("preferred_action_families", [])))
                if _normalize_action_family(item)
            ]
            rule_index = next(
                (
                    index
                    for index, rule in enumerate(transition_rules)
                    if _normalize_action_family(rule.get("action_family", "")) == action_family
                ),
                -1,
            )
            if not prediction and rule_index < 0 and action_family not in preferred_action_families:
                continue

            target_tokens = _string_tokens(
                metadata.get("target_binding_tokens", []),
                row.get("preferred_target_refs", []),
                row.get("family", ""),
                limit=10,
            )
            runtime_state = build_mechanism_runtime_state(
                obs,
                mechanism_control,
                action_tokens=action_tokens,
                target_tokens=target_tokens,
            )
            precondition_report = {
                "has_preconditions": False,
                "satisfied": True,
                "support": 1.0,
                "matched": [],
                "unmet": [],
            }
            if rule_index >= 0:
                precondition_report = evaluate_mechanism_preconditions(
                    transition_rules[rule_index].get("preconditions", []),
                    runtime_state=runtime_state,
                )

            confidence = _clamp01(row.get("posterior", row.get("confidence", 0.0)), 0.0)
            target_overlap = _token_overlap(target_tokens, action_tokens)
            support_multiplier = 1.0
            if bool(precondition_report.get("has_preconditions", False)) and not bool(precondition_report.get("satisfied", False)):
                support_multiplier = 0.10 + (0.20 * float(precondition_report.get("support", 0.0) or 0.0))
            support = confidence * (
                1.0
                if prediction
                else 0.72
                if rule_index >= 0
                else 0.48
            )
            if target_tokens:
                support *= 0.75 + min(0.25, target_overlap)
            support *= support_multiplier
            support_mass += support

            hypothesis_id = str(row.get("hypothesis_id", row.get("object_id", "")) or "")
            if hypothesis_id and hypothesis_id not in matched_ids:
                matched_ids.append(hypothesis_id)

            if target_overlap >= 0.15:
                alignment_bonus += min(0.10, 0.04 + target_overlap * 0.10) * max(0.6, confidence)
                reasons.append("mechanism_target_match")
            elif target_tokens:
                penalty += 0.02 * max(0.5, confidence)
                reasons.append("mechanism_target_mismatch")

            if prediction:
                reward_sign = str(prediction.get("reward_sign", prediction.get("predicted_reward_sign", "")) or "").strip().lower()
                valid_state_change = prediction.get("valid_state_change")
                predicted_info_gain = _clamp01(prediction.get("predicted_information_gain", 0.0), 0.0)
                phase_shift = str(prediction.get("predicted_phase_shift", prediction.get("phase_shift", "")) or "").strip().lower()
                if reward_sign == "positive":
                    short_reward_bonus += 0.06 * support
                    long_reward_bonus += 0.08 * support
                    reasons.append("mechanism_positive_reward")
                elif reward_sign == "negative":
                    penalty += 0.10 * max(0.5, support)
                    risk_delta += 0.04 * max(0.5, support)
                    reasons.append("mechanism_negative_reward")
                elif reward_sign == "zero":
                    short_reward_bonus += 0.01 * support
                if valid_state_change is True:
                    long_reward_bonus += 0.05 * support
                    reasons.append("mechanism_state_change")
                elif valid_state_change is False and action_family != "pointer_interaction":
                    penalty += 0.04 * max(0.5, support)
                    reasons.append("mechanism_no_state_change")
                info_gain_floor = max(info_gain_floor, predicted_info_gain * (0.55 + 0.25 * support))
                if phase_shift:
                    projected_phase_shift = projected_phase_shift or phase_shift
                    if phase_shift in {"committed", "resolved"}:
                        long_reward_bonus += 0.10 * support
                        reasons.append(f"mechanism_phase:{phase_shift}")
                    elif phase_shift in {"configured", "revealed", "stabilizing", "informed"}:
                        long_reward_bonus += 0.05 * support
                        reasons.append(f"mechanism_phase:{phase_shift}")

            if rule_index >= 0:
                matched_rule_index = rule_index if matched_rule_index < 0 else min(matched_rule_index, rule_index)
                stage_bonus = max(0.04, 0.10 - (rule_index * 0.02)) * max(0.55, confidence)
                alignment_bonus += stage_bonus
                reasons.append(f"mechanism_rule:{rule_index}")
                if bool(precondition_report.get("has_preconditions", False)):
                    matched_preconditions.extend(
                        item for item in list(precondition_report.get("matched", []) or [])
                        if item not in matched_preconditions
                    )
                    unmet_preconditions.extend(
                        item for item in list(precondition_report.get("unmet", []) or [])
                        if item not in unmet_preconditions
                    )
                    if bool(precondition_report.get("satisfied", False)):
                        precondition_satisfied = True
                        alignment_bonus += 0.08 * max(0.55, confidence)
                        long_reward_bonus += 0.03 * max(0.5, support)
                        reasons.append("mechanism_preconditions_satisfied")
                    else:
                        stage_support = float(precondition_report.get("support", 0.0) or 0.0)
                        alignment_bonus -= stage_bonus * (0.65 - min(0.45, stage_support * 0.45))
                        precondition_penalty = (1.0 - float(precondition_report.get("support", 0.0) or 0.0)) * 0.12
                        penalty += precondition_penalty * max(0.55, confidence)
                        short_reward_bonus -= 0.04 * max(0.5, confidence)
                        long_reward_bonus -= 0.05 * max(0.5, confidence)
                        risk_delta += 0.02 * max(0.5, confidence)
                        reasons.append("mechanism_preconditions_unsatisfied")
                if rule_index == 0:
                    info_gain_floor = max(info_gain_floor, 0.22 + confidence * 0.34)
                    reversibility_floor = max(reversibility_floor, 0.74)
                    risk_delta -= 0.03 * max(0.5, confidence)
                else:
                    long_reward_bonus += 0.06 * max(0.5, confidence)
                    if action_family == "confirm_interaction":
                        short_reward_bonus += 0.04 * max(0.5, confidence)
                        reasons.append("mechanism_commit_step")

                if control_mode == "discriminate":
                    if rule_index == 0:
                        alignment_bonus += 0.10 * max(0.6, confidence)
                        info_gain_floor = max(info_gain_floor, 0.30 + confidence * 0.30)
                        reversibility_floor = max(reversibility_floor, 0.8)
                        reasons.append("mechanism_mode:discriminate")
                    else:
                        mode_penalty = 0.06 * max(0.5, confidence)
                        if bool(precondition_report.get("satisfied", False)):
                            mode_penalty *= 0.35
                        penalty += mode_penalty
                elif control_mode == "exploit":
                    if rule_index == 0:
                        mode_penalty = 0.04 * max(0.5, confidence)
                        if bool(precondition_report.get("has_preconditions", False)) and not bool(precondition_report.get("satisfied", False)):
                            mode_penalty *= 0.4
                            alignment_bonus += 0.04 * max(0.5, confidence)
                        penalty += mode_penalty
                    else:
                        if bool(precondition_report.get("has_preconditions", False)) and not bool(precondition_report.get("satisfied", False)):
                            penalty += 0.08 * max(0.55, confidence)
                            reasons.append("mechanism_mode:exploit_blocked")
                        else:
                            alignment_bonus += 0.10 * max(0.6, confidence)
                            long_reward_bonus += 0.08 * max(0.6, confidence)
                            reasons.append("mechanism_mode:exploit")
            elif action_family in preferred_action_families:
                alignment_bonus += 0.04 * max(0.5, confidence)
                reasons.append("mechanism_family_match")

        return {
            "alignment_bonus": round(min(0.36, alignment_bonus), 6),
            "penalty": round(min(0.26, penalty), 6),
            "short_reward_bonus": round(max(-0.2, min(0.2, short_reward_bonus)), 6),
            "long_reward_bonus": round(max(-0.08, min(0.32, long_reward_bonus)), 6),
            "info_gain_floor": round(_clamp01(info_gain_floor, 0.0), 6),
            "reversibility_floor": round(_clamp01(reversibility_floor, 0.0), 6),
            "risk_delta": round(max(-0.12, min(0.12, risk_delta)), 6),
            "support_mass": round(min(1.0, support_mass), 6),
            "matched_rule_index": matched_rule_index,
            "projected_phase_shift": projected_phase_shift,
            "matched_hypothesis_ids": matched_ids[:4],
            "precondition_satisfied": precondition_satisfied,
            "matched_preconditions": matched_preconditions[:6],
            "unmet_preconditions": unmet_preconditions[:6],
            "reasons": list(dict.fromkeys(reasons))[:8],
        }


class DeterministicReasonerBackend(SymbolicReasoningBackend):
    name = "deterministic"


class SearchReasonerBackend(SymbolicReasoningBackend):
    name = "search"

    def deliberate(
        self,
        request: ReasoningRequest,
        budget: DeliberationBudget,
    ) -> ReasoningResult:
        result = super().deliberate(request, budget)
        result.deliberation_trace.append({
            "stage": "search_backend",
            "status": "bounded_search",
            "branch_budget": int(budget.branch_budget),
            "depth": int(budget.depth),
        })
        result.backend = self.name
        return result
