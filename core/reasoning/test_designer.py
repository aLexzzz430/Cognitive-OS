from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

from core.orchestration.action_utils import (
    action_semantic_signature,
    extract_action_function_name,
    extract_action_signature_kwargs,
    serialize_action_semantic_signature,
)
from core.orchestration.commit_candidate_guard import collect_high_confidence_commit_candidates
from core.reasoning.discriminating_experiment import build_discriminating_experiments
from core.reasoning.hypothesis_schema import normalize_hypothesis_rows


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _probe_like(function_name: str) -> bool:
    name = str(function_name or "").strip().lower()
    if not name:
        return False
    return any(token in name for token in ("probe", "inspect", "verify", "check", "test"))


def _action_kwargs(action: Dict[str, Any]) -> Dict[str, Any]:
    return extract_action_signature_kwargs(action)


def _action_signature(action: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return action_semantic_signature(action)


def _test_row_signature(row: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    candidate_action = row.get("candidate_action", {}) if isinstance(row.get("candidate_action", {}), dict) else {}
    if candidate_action:
        return _action_signature(candidate_action)
    function_name = str(row.get("function_name", "") or "").strip()
    kwargs = row.get("kwargs", {}) if isinstance(row.get("kwargs", {}), dict) else {}
    return action_semantic_signature(
        {
            "kind": "probe" if _probe_like(function_name) else "call_tool",
            "payload": {"tool_args": {"function_name": function_name, "kwargs": dict(kwargs)}},
        }
    )


def _test_selector_id(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    explicit = str(row.get("selector_id", "") or "").strip()
    if explicit:
        return explicit
    signature = _test_row_signature(row)
    if signature[0]:
        return f"test_selector::{serialize_action_semantic_signature(signature)}"
    return str(
        row.get("test_id", "")
        or row.get("object_id", "")
        or row.get("experiment_id", "")
        or row.get("function_name", "")
        or ""
    ).strip()


def _hypothesis_disagreement_pressure(
    hypotheses: Sequence[Dict[str, Any]],
    ranked_experiments: Sequence[Dict[str, Any]],
) -> float:
    posterior_values = sorted(
        (
            _clamp01(row.get("posterior", row.get("confidence", 0.0)))
            for row in list(hypotheses or [])
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


def _build_probe_action(function_name: str, reason: str) -> Dict[str, Any]:
    return {
        "kind": "probe" if _probe_like(function_name) else "call_tool",
        "_source": "deliberation_probe",
        "payload": {
            "tool_args": {
                "function_name": function_name,
                "kwargs": {},
            },
        },
        "_candidate_meta": {
            "deliberation_injected": True,
            "deliberation_reason": reason,
            "probe_candidate": True,
        },
    }


def _build_injected_action(row: Dict[str, Any]) -> Dict[str, Any]:
    candidate_action = row.get("candidate_action", {}) if isinstance(row.get("candidate_action", {}), dict) else {}
    reason = str(row.get("reason", "deliberation_test") or "deliberation_test")
    if not candidate_action:
        return _build_probe_action(str(row.get("function_name", "") or ""), reason)

    injected = deepcopy(candidate_action)
    injected["kind"] = str(injected.get("kind", "") or ("probe" if _probe_like(extract_action_function_name(injected, default="")) else "call_tool"))
    injected["_source"] = "deliberation_probe"
    payload = injected.get("payload", {}) if isinstance(injected.get("payload", {}), dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
    tool_args["kwargs"] = _action_kwargs(injected)
    payload["tool_args"] = tool_args
    injected["payload"] = payload
    meta = injected.get("_candidate_meta", {}) if isinstance(injected.get("_candidate_meta", {}), dict) else {}
    meta["deliberation_injected"] = True
    meta["deliberation_reason"] = reason
    meta["probe_candidate"] = True
    injected["_candidate_meta"] = meta
    return injected


def _world_model_summary(workspace: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("active_beliefs_summary", "world_model_summary"):
        raw = workspace.get(key, {})
        if isinstance(raw, dict):
            return raw
    return {}


def design_candidate_tests(
    workspace: Dict[str, Any],
    candidate_actions: Sequence[Dict[str, Any]],
    *,
    available_functions: Sequence[str],
    limit: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool, List[str]]:
    existing = workspace.get("candidate_tests", [])
    if not isinstance(existing, list):
        existing = []
    ranked_tests: List[Dict[str, Any]] = [dict(item) for item in existing if isinstance(item, dict)]
    hypotheses = normalize_hypothesis_rows(
        [dict(item) for item in list(workspace.get("competing_hypotheses", []) or []) if isinstance(item, dict)],
        fallback_id_prefix="test_h",
    )
    ranked_experiments = [
        dict(item)
        for item in list(workspace.get("ranked_discriminating_experiments", []) or [])
        if isinstance(item, dict)
    ]
    if not ranked_experiments:
        ranked_experiments = build_discriminating_experiments(
            hypotheses,
            [dict(action) for action in list(candidate_actions or []) if isinstance(action, dict)],
            limit=max(limit, 5),
        )

    world_model_summary = _world_model_summary(workspace)
    required_probes = list(world_model_summary.get("required_probes", []) or []) if isinstance(world_model_summary, dict) else []
    uncertainty = _clamp01(
        (workspace.get("uncertainty_vector", {}) if isinstance(workspace.get("uncertainty_vector", {}), dict) else {}).get("overall", 0.0)
    )
    rollout_uncertainty = _clamp01(world_model_summary.get("rollout_uncertainty", 0.0)) if isinstance(world_model_summary, dict) else 0.0
    explicit_probe_required = "probe_before_commit" in required_probes
    commit_guarded_candidates = collect_high_confidence_commit_candidates(
        candidate_actions,
        available_functions=available_functions,
    )
    if explicit_probe_required:
        commit_guarded_candidates = [
            row
            for row in commit_guarded_candidates
            if str(row.get("reason", "") or "") not in {"single_visible_surface", "plan_target_surface"}
        ]
    probe_before_commit = (
        not commit_guarded_candidates
        and (
            explicit_probe_required
            or uncertainty >= 0.68
            or rollout_uncertainty >= 0.66
        )
    )
    hypothesis_disagreement = _hypothesis_disagreement_pressure(hypotheses, ranked_experiments)
    commit_risk_pressure = max(
        0.82 if commit_guarded_candidates else 0.0,
        0.74 if explicit_probe_required else 0.0,
        rollout_uncertainty,
    )
    experiment_priority_mode = bool(
        uncertainty >= 0.72
        and hypothesis_disagreement >= 0.55
        and commit_risk_pressure >= 0.6
    )
    experiment_priority_bonus = (
        0.24
        + (hypothesis_disagreement * 0.18)
        + (commit_risk_pressure * 0.10)
    ) if experiment_priority_mode else 0.0

    for row in list(world_model_summary.get("discriminating_tests", []) or []):
        if not isinstance(row, dict):
            continue
        normalized = dict(row)
        info_gain = _clamp01(normalized.get("info_gain_estimate", world_model_summary.get("expected_information_gain", 0.0)))
        normalized["info_gain_estimate"] = round(info_gain, 4)
        normalized["score"] = round(
            max(float(normalized.get("score", 0.0) or 0.0), 0.40 + info_gain * 0.42 + (0.10 if probe_before_commit else 0.0)),
            4,
        )
        normalized.setdefault("reason", "world_model_discriminating_test")
        ranked_tests.append(normalized)

    for row in ranked_experiments:
        if not isinstance(row, dict):
            continue
        ranked_tests.append(
            {
                "test_id": str(row.get("experiment_id", "") or ""),
                "summary": str(row.get("summary", "") or ""),
                "function_name": str(row.get("function_name", "") or ""),
                "info_gain_estimate": round(_clamp01(row.get("expected_information_gain", 0.0)), 4),
                "score": round(
                    float(row.get("score", 0.0) or 0.0)
                    + experiment_priority_bonus,
                    4,
                ),
                "reason": "discriminating_experiment",
                "discriminates_between": list(row.get("discriminates_between", []) or []),
                "expected_outcomes": dict(row.get("expected_outcomes", {}) or {}),
                "candidate_action": deepcopy(row.get("candidate_action", {})) if isinstance(row.get("candidate_action", {}), dict) else {},
                "kwargs": _action_kwargs(row.get("candidate_action", {})) if isinstance(row.get("candidate_action", {}), dict) else {},
                "experiment_priority_mode": experiment_priority_mode,
            }
        )

    candidate_probe_actions: List[Dict[str, Any]] = []
    seen_signatures = set()
    existing_active_signatures = set()
    for action in candidate_actions:
        if not isinstance(action, dict):
            continue
        fn_name = extract_action_function_name(action, default="")
        meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
        if _probe_like(fn_name) or bool(meta.get("probe_candidate")):
            existing_active_signatures.add(_action_signature(action))
        if not _probe_like(fn_name):
            continue
        signature = _action_signature(action)
        seen_signatures.add(signature)
        candidate_probe_actions.append({
            "test_id": f"candidate_test_{fn_name}",
            "summary": f"probe via {fn_name}",
            "function_name": fn_name,
            "score": round(
                0.65
                + (_clamp01(meta.get("counterfactual_delta", 0.0)) * 0.15)
                + (_clamp01(meta.get("expected_information_gain", world_model_summary.get("expected_information_gain", 0.0))) * 0.08)
                + (0.15 if probe_before_commit else 0.0),
                4,
            ),
            "reason": "existing_probe_candidate",
            "candidate_action": deepcopy(action),
            "kwargs": _action_kwargs(action),
        })

    if probe_before_commit:
        for fn_name in available_functions:
            text = str(fn_name or "").strip()
            signature = (text, ())
            if not text or signature in seen_signatures or not _probe_like(text):
                continue
            candidate_probe_actions.append({
                "test_id": f"candidate_test_{text}",
                "summary": f"inject verification probe {text}",
                "function_name": text,
                "info_gain_estimate": round(_clamp01(world_model_summary.get("expected_information_gain", 0.0)), 4),
                "score": round(0.72 + _clamp01(world_model_summary.get("expected_information_gain", 0.0)) * 0.08, 4),
                "reason": "probe_before_commit",
                "kwargs": {},
            })

    top_discriminating_experiment = deepcopy(ranked_experiments[0]) if ranked_experiments else {}
    top_discriminating_function = str(top_discriminating_experiment.get("function_name", "") or "")
    if top_discriminating_function:
        candidate_probe_actions.append(
            {
                "test_id": f"candidate_test_{top_discriminating_function}",
                "summary": f"execute top discriminating experiment via {top_discriminating_function}",
                "function_name": top_discriminating_function,
                "info_gain_estimate": round(_clamp01(top_discriminating_experiment.get("expected_information_gain", 0.0)), 4),
                "score": round(
                    max(0.6, float(top_discriminating_experiment.get("score", 0.0) or 0.0))
                    + experiment_priority_bonus,
                    4,
                ),
                "reason": "discriminating_experiment",
                "top_discriminating_experiment": True,
                "experiment_priority_mode": experiment_priority_mode,
                "candidate_action": deepcopy(top_discriminating_experiment.get("candidate_action", {}))
                if isinstance(top_discriminating_experiment.get("candidate_action", {}), dict)
                else {},
                "kwargs": _action_kwargs(top_discriminating_experiment.get("candidate_action", {}))
                if isinstance(top_discriminating_experiment.get("candidate_action", {}), dict)
                else {},
            }
        )

    ranked_tests.extend(candidate_probe_actions)
    deduped: Dict[Tuple[str, Tuple[Tuple[str, str], ...]] | str, Dict[str, Any]] = {}
    for row in ranked_tests:
        signature = _test_row_signature(row)
        key = signature if signature[0] else str(row.get("test_id", "") or row.get("object_id", "") or "")
        if not key:
            continue
        existing_row = deduped.get(key)
        if existing_row is None or float(row.get("score", 0.0) or 0.0) > float(existing_row.get("score", 0.0) or 0.0):
            deduped[key] = row
    ranked = sorted(
        deduped.values(),
        key=lambda item: float(item.get("score", item.get("confidence", 0.0)) or 0.0),
        reverse=True,
    )[: max(0, int(limit))]
    ranked = [
        {
            **dict(row),
            "selector_id": _test_selector_id(row),
        }
        for row in ranked
    ]

    injected_actions: List[Dict[str, Any]] = []
    active_test_ids: List[str] = []
    active_test_seen = set()
    hard_commit_guard_active = bool(commit_guarded_candidates)
    for row in ranked:
        signature = _test_row_signature(row)
        fn_name = signature[0]
        selector_id = str(row.get("selector_id", "") or "")
        if selector_id and signature in existing_active_signatures and selector_id not in active_test_seen:
            active_test_ids.append(selector_id)
            active_test_seen.add(selector_id)
        if not fn_name or signature in seen_signatures:
            continue
        strong_experiment = bool(
            row.get("top_discriminating_experiment", False)
            or str(row.get("reason", "") or "") in {"world_model_discriminating_test", "discriminating_experiment"}
            or str(row.get("function_name", "") or "") == top_discriminating_function
        )
        should_inject = bool(
            probe_before_commit
            or strong_experiment
            or explicit_probe_required
        )
        if not should_inject:
            continue
        if hard_commit_guard_active and not strong_experiment and not probe_before_commit:
            continue
        injected_actions.append(_build_injected_action(row))
        seen_signatures.add(signature)
        if selector_id and selector_id not in active_test_seen:
            active_test_ids.append(selector_id)
            active_test_seen.add(selector_id)

    return ranked, injected_actions, probe_before_commit, active_test_ids
