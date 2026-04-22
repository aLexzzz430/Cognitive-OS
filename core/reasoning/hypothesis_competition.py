from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from core.reasoning.hypothesis_schema import (
    hypothesis_observation_signature,
    normalize_hypothesis_rows,
)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _ordered_unique(values: Sequence[Any]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in list(values or []):
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _string_tokens(*values: Any, limit: int = 8) -> List[str]:
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
        for raw in re.split(r"[^a-z0-9]+", text.replace("::", "_")):
            token = str(raw or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
            if len(ordered) >= limit:
                return ordered[:limit]
    return ordered[:limit]


def _set_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {str(item or "").strip().lower() for item in list(left or []) if str(item or "").strip()}
    right_set = {str(item or "").strip().lower() for item in list(right or []) if str(item or "").strip()}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / float(max(1, min(len(left_set), len(right_set))))


def _prediction_observation_tokens(prediction: Mapping[str, Any]) -> List[str]:
    return _string_tokens(
        prediction.get("predicted_observation_tokens", []),
        prediction.get("observation_tokens", []),
        limit=8,
    )


def _prediction_target_tokens(prediction: Mapping[str, Any]) -> List[str]:
    return _string_tokens(
        prediction.get("target_kind", ""),
        prediction.get("target_family", ""),
        prediction.get("relation_type", ""),
        prediction.get("anchor_ref", ""),
        prediction.get("predicted_observation_tokens", []),
        limit=8,
    )


def _prediction_action_label(function_name: str, prediction: Mapping[str, Any]) -> str:
    semantic_tokens = _ordered_unique(
        [
            str(prediction.get("anchor_ref", "") or ""),
            str(prediction.get("target_family", prediction.get("target_kind", "")) or ""),
            str(prediction.get("relation_type", "") or ""),
            f"x:{prediction.get('x')}" if prediction.get("x", None) is not None else "",
            f"y:{prediction.get('y')}" if prediction.get("y", None) is not None else "",
        ]
    )
    if not semantic_tokens:
        return str(function_name or "").strip()
    return f"{str(function_name or '').strip()}@{'/'.join(semantic_tokens[:4])}"


def _prediction_entries(
    row: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    predictions = row.get("predictions", {}) if isinstance(row.get("predictions", {}), dict) else {}
    predicted_action_effects = (
        predictions.get("predicted_action_effects", row.get("predicted_action_effects", {}))
        if isinstance(predictions, dict)
        else row.get("predicted_action_effects", {})
    )
    predicted_action_effects_by_signature = (
        predictions.get(
            "predicted_action_effects_by_signature",
            row.get("predicted_action_effects_by_signature", {}),
        )
        if isinstance(predictions, dict)
        else row.get("predicted_action_effects_by_signature", {})
    )
    signature_entries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    function_entries: Dict[str, Dict[str, Any]] = {}

    if isinstance(predicted_action_effects_by_signature, dict):
        for signature_key, payload in predicted_action_effects_by_signature.items():
            if not isinstance(payload, dict):
                continue
            function_name = str(payload.get("function_name", "") or "").strip()
            if not function_name:
                continue
            signature_entries.setdefault(function_name, {})[str(signature_key or "").strip()] = {
                "prediction": dict(payload),
                "label": _prediction_action_label(function_name, payload),
            }

    if isinstance(predicted_action_effects, dict):
        for function_name, payload in predicted_action_effects.items():
            if not isinstance(payload, dict):
                continue
            normalized_name = str(function_name or payload.get("function_name", "") or "").strip()
            if not normalized_name:
                continue
            function_entries[normalized_name] = {
                "prediction": dict(payload),
                "label": _prediction_action_label(normalized_name, payload),
            }

    return signature_entries, function_entries


def _competition_marker(action_name: str, category: str, value: Any) -> str:
    normalized_value = "_".join(_string_tokens(value, limit=4)) or "unknown"
    normalized_action = "_".join(_string_tokens(action_name, limit=4)) or "action"
    return f"competition::{normalized_action}::{category}::{normalized_value}"


def _shared_prediction_conflicts(
    left_prediction: Mapping[str, Any],
    right_prediction: Mapping[str, Any],
    *,
    action_name: str,
    action_label: str = "",
) -> Tuple[List[str], List[str], List[str]]:
    if not left_prediction or not right_prediction:
        return [], [], []
    reason_prefix = str(action_label or action_name or "").strip() or str(action_name or "").strip()
    marker_action_name = str(action_name or "").strip() or reason_prefix
    reasons: List[str] = []
    left_markers: List[str] = []
    right_markers: List[str] = []

    left_phase = str(left_prediction.get("predicted_phase_shift", left_prediction.get("phase_shift", "")) or "").strip().lower()
    right_phase = str(right_prediction.get("predicted_phase_shift", right_prediction.get("phase_shift", "")) or "").strip().lower()
    if left_phase and right_phase and left_phase != right_phase:
        reasons.append(f"{reason_prefix}:phase_shift_conflict")
        left_markers.append(_competition_marker(marker_action_name, "phase", right_phase))
        right_markers.append(_competition_marker(marker_action_name, "phase", left_phase))

    left_reward = str(left_prediction.get("reward_sign", left_prediction.get("predicted_reward_sign", "")) or "").strip().lower()
    right_reward = str(right_prediction.get("reward_sign", right_prediction.get("predicted_reward_sign", "")) or "").strip().lower()
    if left_reward and right_reward and left_reward != right_reward:
        reasons.append(f"{reason_prefix}:reward_conflict")
        left_markers.append(_competition_marker(marker_action_name, "reward", right_reward))
        right_markers.append(_competition_marker(marker_action_name, "reward", left_reward))

    if "valid_state_change" in left_prediction and "valid_state_change" in right_prediction:
        if bool(left_prediction.get("valid_state_change")) != bool(right_prediction.get("valid_state_change")):
            reasons.append(f"{reason_prefix}:state_change_conflict")
            left_markers.append(
                _competition_marker(
                    marker_action_name,
                    "state_change",
                    "changed" if bool(right_prediction.get("valid_state_change")) else "unchanged",
                )
            )
            right_markers.append(
                _competition_marker(
                    marker_action_name,
                    "state_change",
                    "changed" if bool(left_prediction.get("valid_state_change")) else "unchanged",
                )
            )

    left_risk = str(left_prediction.get("risk_type", "") or "").strip().lower()
    right_risk = str(right_prediction.get("risk_type", "") or "").strip().lower()
    if left_risk and right_risk and left_risk != right_risk:
        reasons.append(f"{reason_prefix}:risk_conflict")
        left_markers.append(_competition_marker(marker_action_name, "risk", right_risk))
        right_markers.append(_competition_marker(marker_action_name, "risk", left_risk))

    left_target_tokens = _prediction_target_tokens(left_prediction)
    right_target_tokens = _prediction_target_tokens(right_prediction)
    left_observation_tokens = _prediction_observation_tokens(left_prediction)
    right_observation_tokens = _prediction_observation_tokens(right_prediction)
    if _set_overlap(left_target_tokens, right_target_tokens) == 0.0 and left_target_tokens and right_target_tokens:
        reasons.append(f"{reason_prefix}:target_conflict")
        left_markers.append(_competition_marker(marker_action_name, "target", right_target_tokens[0]))
        right_markers.append(_competition_marker(marker_action_name, "target", left_target_tokens[0]))
    if _set_overlap(left_observation_tokens, right_observation_tokens) == 0.0 and left_observation_tokens and right_observation_tokens:
        reasons.append(f"{reason_prefix}:observation_conflict")
        left_markers.append(_competition_marker(marker_action_name, "observation", right_observation_tokens[0]))
        right_markers.append(_competition_marker(marker_action_name, "observation", left_observation_tokens[0]))

    return reasons, _ordered_unique(left_markers), _ordered_unique(right_markers)


def _build_competition_relations(
    hypotheses: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, List[str]]], Dict[str, List[str]]]:
    conflict_graph: Dict[str, Set[str]] = {}
    conflict_details: Dict[str, Dict[str, List[str]]] = {}
    falsifier_candidates: Dict[str, List[str]] = {}
    rows = [dict(row) for row in list(hypotheses or []) if isinstance(row, dict)]
    for row in rows:
        hypothesis_id = str(row.get("hypothesis_id", "") or "")
        if not hypothesis_id:
            continue
        conflict_graph[hypothesis_id] = {
            str(item).strip()
            for item in list(row.get("conflicts_with", []) or [])
            if str(item).strip() and str(item).strip() != hypothesis_id
        }
        conflict_details[hypothesis_id] = {}
        falsifier_candidates[hypothesis_id] = [
            str(item).strip()
            for item in list(row.get("falsifiers", []) or [])
            if str(item).strip()
        ]

    for index, left_row in enumerate(rows):
        left_id = str(left_row.get("hypothesis_id", "") or "")
        if not left_id:
            continue
        left_signature = hypothesis_observation_signature(left_row)
        left_signature_entries, left_function_entries = _prediction_entries(left_row)
        for right_row in rows[index + 1 :]:
            right_id = str(right_row.get("hypothesis_id", "") or "")
            if not right_id:
                continue
            right_signature = hypothesis_observation_signature(right_row)
            right_signature_entries, right_function_entries = _prediction_entries(right_row)
            pair_reasons: List[str] = []
            left_markers: List[str] = []
            right_markers: List[str] = []
            function_names = sorted(
                set(left_signature_entries.keys())
                | set(right_signature_entries.keys())
                | set(left_function_entries.keys())
                | set(right_function_entries.keys())
            )
            for action_name in function_names:
                left_signature_predictions = left_signature_entries.get(action_name, {})
                right_signature_predictions = right_signature_entries.get(action_name, {})
                if left_signature_predictions and right_signature_predictions:
                    for signature_key in sorted(set(left_signature_predictions.keys()) & set(right_signature_predictions.keys())):
                        left_entry = left_signature_predictions.get(signature_key, {})
                        right_entry = right_signature_predictions.get(signature_key, {})
                        action_reasons, action_left_markers, action_right_markers = _shared_prediction_conflicts(
                            left_entry.get("prediction", {}),
                            right_entry.get("prediction", {}),
                            action_name=action_name,
                            action_label=str(left_entry.get("label", "") or right_entry.get("label", "") or action_name),
                        )
                        pair_reasons.extend(action_reasons)
                        left_markers.extend(action_left_markers)
                        right_markers.extend(action_right_markers)
                    continue
                if left_signature_predictions and not right_signature_predictions and action_name in right_function_entries:
                    right_entry = right_function_entries.get(action_name, {})
                    for left_entry in left_signature_predictions.values():
                        action_reasons, action_left_markers, action_right_markers = _shared_prediction_conflicts(
                            left_entry.get("prediction", {}),
                            right_entry.get("prediction", {}),
                            action_name=action_name,
                            action_label=str(left_entry.get("label", "") or right_entry.get("label", "") or action_name),
                        )
                        pair_reasons.extend(action_reasons)
                        left_markers.extend(action_left_markers)
                        right_markers.extend(action_right_markers)
                    continue
                if right_signature_predictions and not left_signature_predictions and action_name in left_function_entries:
                    left_entry = left_function_entries.get(action_name, {})
                    for right_entry in right_signature_predictions.values():
                        action_reasons, action_left_markers, action_right_markers = _shared_prediction_conflicts(
                            left_entry.get("prediction", {}),
                            right_entry.get("prediction", {}),
                            action_name=action_name,
                            action_label=str(left_entry.get("label", "") or right_entry.get("label", "") or action_name),
                        )
                        pair_reasons.extend(action_reasons)
                        left_markers.extend(action_left_markers)
                        right_markers.extend(action_right_markers)
                    continue
                if action_name in left_function_entries and action_name in right_function_entries:
                    left_entry = left_function_entries.get(action_name, {})
                    right_entry = right_function_entries.get(action_name, {})
                    action_reasons, action_left_markers, action_right_markers = _shared_prediction_conflicts(
                        left_entry.get("prediction", {}),
                        right_entry.get("prediction", {}),
                        action_name=action_name,
                        action_label=str(left_entry.get("label", "") or right_entry.get("label", "") or action_name),
                    )
                    pair_reasons.extend(action_reasons)
                    left_markers.extend(action_left_markers)
                    right_markers.extend(action_right_markers)

            signature_overlap = _set_overlap(left_signature, right_signature)
            left_family = str(left_row.get("family", "") or left_row.get("hypothesis_type", "") or "").strip().lower()
            right_family = str(right_row.get("family", "") or right_row.get("hypothesis_type", "") or "").strip().lower()
            if not pair_reasons and signature_overlap >= 0.55 and left_family and right_family and left_family != right_family:
                pair_reasons.append("observation_signature_competition")

            if not pair_reasons:
                continue

            conflict_graph.setdefault(left_id, set()).add(right_id)
            conflict_graph.setdefault(right_id, set()).add(left_id)
            conflict_details.setdefault(left_id, {})[right_id] = _ordered_unique(pair_reasons)
            conflict_details.setdefault(right_id, {})[left_id] = _ordered_unique(pair_reasons)
            falsifier_candidates[left_id] = _ordered_unique(falsifier_candidates.get(left_id, []) + left_markers)
            falsifier_candidates[right_id] = _ordered_unique(falsifier_candidates.get(right_id, []) + right_markers)

    return conflict_graph, conflict_details, falsifier_candidates

def _world_model_summary(workspace: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("active_beliefs_summary", "world_model_summary"):
        raw = workspace.get(key, {})
        if isinstance(raw, dict):
            return raw
    return {}


def _mechanism_alignment(row: Dict[str, Any], summary: Dict[str, Any]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    best = 0.0
    hypothesis_id = str(row.get("hypothesis_id", "") or "")
    family = str(row.get("family", "") or "")
    summary_text = " ".join(
        value for value in (
            str(row.get("summary", "") or ""),
            str(row.get("expected_transition", "") or ""),
        )
        if value
    ).lower()
    for mechanism in list(summary.get("mechanism_hypotheses", []) or []):
        if not isinstance(mechanism, dict):
            continue
        mechanism_score = 0.0
        mechanism_id = str(mechanism.get("hypothesis_id", mechanism.get("mechanism_id", "")) or "")
        mechanism_family = str(mechanism.get("family", "") or "")
        expected_transition = str(mechanism.get("expected_transition", "") or "").lower()
        if hypothesis_id and mechanism_id and hypothesis_id == mechanism_id:
            mechanism_score += 0.18
            reasons.append(f"mechanism_id:{mechanism_id}")
        if family and mechanism_family and family == mechanism_family:
            mechanism_score += 0.12
            reasons.append(f"mechanism_family:{mechanism_family}")
        if expected_transition and expected_transition in summary_text:
            mechanism_score += 0.10
            reasons.append(f"mechanism_transition:{mechanism_family or mechanism_id}")
        mechanism_score += _clamp01(mechanism.get("expected_information_gain", 0.0)) * 0.08
        mechanism_score += _clamp01(mechanism.get("confidence", 0.0)) * 0.10
        best = max(best, mechanism_score)
    return min(0.3, best), list(dict.fromkeys(reasons))


def _transition_alignment(row: Dict[str, Any], summary: Dict[str, Any]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    best = 0.0
    summary_text = " ".join(
        value for value in (
            str(row.get("summary", "") or ""),
            str(row.get("expected_transition", "") or ""),
        )
        if value
    ).lower()
    for transition in list(summary.get("predicted_transitions", []) or []):
        if not isinstance(transition, dict):
            continue
        to_phase = str(transition.get("to_phase", "") or "").lower()
        function_name = str(transition.get("function_name", "") or "")
        score = 0.0
        if to_phase and to_phase in summary_text:
            score += 0.12
            reasons.append(f"predicted_phase:{to_phase}")
        if function_name and function_name.lower() in summary_text:
            score += 0.06
            reasons.append(f"predicted_fn:{function_name}")
        score += _clamp01(transition.get("expected_information_gain", 0.0)) * 0.05
        score += (1.0 - _clamp01(transition.get("state_shift_risk", 0.0))) * 0.04
        best = max(best, score)
    return min(0.2, best), list(dict.fromkeys(reasons))


def rank_hypotheses(workspace: Dict[str, Any], *, limit: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = workspace.get("competing_hypotheses", [])
    if not isinstance(rows, list) or not rows:
        rows = workspace.get("active_hypotheses_summary", [])
    hypotheses = normalize_hypothesis_rows([dict(row) for row in rows if isinstance(row, dict)], fallback_id_prefix="comp")
    conflict_graph, conflict_details, falsifier_candidates = _build_competition_relations(hypotheses)
    world_model_summary = _world_model_summary(workspace)
    rollout_uncertainty = _clamp01(world_model_summary.get("rollout_uncertainty", 0.0))
    ranked: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for row in hypotheses:
        hypothesis_id = str(row.get("hypothesis_id", "") or "")
        row["conflicts_with"] = _ordered_unique(
            [str(item) for item in list(row.get("conflicts_with", []) or []) if str(item or "").strip()]
            + sorted(conflict_graph.get(hypothesis_id, set()))
        )
        row["falsifiers"] = _ordered_unique(
            [str(item) for item in list(row.get("falsifiers", []) or []) if str(item or "").strip()]
            + list(falsifier_candidates.get(hypothesis_id, []))
        )
        support_bonus = min(0.2, row.get("support_count", 0) * 0.05)
        contradiction_penalty = min(0.25, row.get("contradiction_count", 0) * 0.07)
        posterior = _clamp01(row.get("posterior", row.get("confidence", 0.0)), default=_clamp01(row.get("confidence", 0.0)))
        mechanism_bonus, mechanism_reasons = _mechanism_alignment(row, world_model_summary)
        transition_bonus, transition_reasons = _transition_alignment(row, world_model_summary)
        info_gain_bonus = _clamp01(world_model_summary.get("expected_information_gain", 0.0)) * 0.06
        uncertainty_penalty = rollout_uncertainty * (0.04 if row.get("support_count", 0) > 0 else 0.08)
        conflict_penalty = min(0.08, len(list(row.get("conflicts_with", []) or [])) * 0.02)
        score = (
            posterior * 0.55
            + _clamp01(row.get("confidence", 0.0)) * 0.45
            + support_bonus
            - contradiction_penalty
            + mechanism_bonus
            + transition_bonus
            + info_gain_bonus
            - uncertainty_penalty
            - conflict_penalty
        )
        row["posterior"] = round(posterior, 6)
        row["status"] = str(row.get("status", "") or "active").strip().lower() or "active"
        row["predictions"] = dict(row.get("predictions", {})) if isinstance(row.get("predictions", {}), dict) else {}
        row["metadata"] = dict(row.get("metadata", {})) if isinstance(row.get("metadata", {}), dict) else {}
        row["metadata"]["competition_conflict_targets"] = list(row.get("conflicts_with", []) or [])
        row["metadata"]["competition_conflict_reasons"] = dict(conflict_details.get(hypothesis_id, {}))
        row["metadata"]["competition_falsifier_candidates"] = list(row.get("falsifiers", []) or [])
        row["competition_score"] = round(score, 4)
        row["world_model_support"] = {
            "mechanism_bonus": round(mechanism_bonus, 4),
            "transition_bonus": round(transition_bonus, 4),
            "info_gain_bonus": round(info_gain_bonus, 4),
            "uncertainty_penalty": round(uncertainty_penalty, 4),
            "conflict_penalty": round(conflict_penalty, 4),
            "reasons": list(
                dict.fromkeys(
                    [
                        *mechanism_reasons,
                        *transition_reasons,
                        *[
                            f"competition:{target_id}"
                            for target_id in list(conflict_details.get(hypothesis_id, {}).keys())
                        ],
                    ]
                )
            ),
        }
        if score <= 0.15:
            rejected.append({
                "hypothesis_id": row.get("hypothesis_id", ""),
                "reason": "low_competition_score",
                "score": round(score, 4),
            })
            continue
        ranked.append(row)
    ranked.sort(
        key=lambda item: (
            float(item.get("competition_score", 0.0) or 0.0),
            float(item.get("posterior", 0.0) or 0.0),
            float(item.get("confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    if ranked:
        ranked[0]["status"] = (
            "leading"
            if float(ranked[0].get("posterior", 0.0) or 0.0) >= 0.75
            else str(ranked[0].get("status", "active") or "active")
        )
    return ranked[: max(0, int(limit))], rejected
