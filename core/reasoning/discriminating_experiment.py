from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.orchestration.action_utils import (
    action_semantic_signature,
    extract_action_function_name,
    extract_action_kind,
    serialize_action_semantic_signature,
)
from core.reasoning.hypothesis_schema import (
    hypothesis_action_prediction,
    hypothesis_observation_signature,
    normalize_hypothesis_rows,
)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _action_signature(action: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return action_semantic_signature(action)


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


def _tokenize(*values: Any, limit: int = 12) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for token in _tokenize(*list(value), limit=limit):
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


def _set_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {str(item or "").strip().lower() for item in list(left or []) if str(item or "").strip()}
    right_set = {str(item or "").strip().lower() for item in list(right or []) if str(item or "").strip()}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / float(max(1, min(len(left_set), len(right_set))))


def _prediction_maps(row: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    predictions = _dict(row.get("predictions", {}))
    action_effects = _dict(predictions.get("predicted_action_effects", row.get("predicted_action_effects", {})))
    action_effects_by_signature = _dict(
        predictions.get(
            "predicted_action_effects_by_signature",
            row.get("predicted_action_effects_by_signature", {}),
        )
    )
    return action_effects, action_effects_by_signature


def _prediction_observation_tokens(prediction: Dict[str, Any]) -> List[str]:
    return _ordered_unique(
        _tokenize(
            prediction.get("predicted_observation_tokens", []),
            prediction.get("predicted_observation", ""),
            prediction.get("predicted_observation_signature", []),
            limit=12,
        )
    )


def _prediction_compact_summary(effect_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    prediction = _dict(payload)
    valid_state_change = prediction.get("valid_state_change")
    compact_valid_state_change: Optional[bool] = None
    if valid_state_change is not None:
        compact_valid_state_change = bool(valid_state_change)
    return {
        "effect_key": str(effect_key or "").strip(),
        "reward_sign": _reward_sign_from_prediction(prediction),
        "valid_state_change": compact_valid_state_change,
        "predicted_phase_shift": str(
            prediction.get("predicted_phase_shift", prediction.get("phase_shift", ""))
            or ""
        ).strip().lower(),
        "risk_type": str(prediction.get("risk_type", "") or "").strip().lower(),
        "target_family": str(
            prediction.get("target_family", prediction.get("target", ""))
            or ""
        ).strip().lower(),
        "relation_type": str(prediction.get("relation_type", "") or "").strip().lower(),
        "predicted_observation_tokens": _prediction_observation_tokens(prediction),
    }


def _prediction_collection_summary(effects: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordered_items = sorted(
        [(str(key or "").strip(), value) for key, value in effects.items()],
        key=lambda item: item[0],
    )
    for effect_key, payload in ordered_items:
        rows.append(_prediction_compact_summary(effect_key, payload if isinstance(payload, dict) else {}))
    return rows


def _prediction_token_coverage(row: Dict[str, Any]) -> float:
    action_effects, action_effects_by_signature = _prediction_maps(row)
    action_tokens = _tokenize(list(action_effects.keys()), list(action_effects_by_signature.keys()), limit=12)
    observation_tokens = _tokenize(
        row.get("predicted_observation_tokens", []),
        row.get("summary", ""),
        row.get("expected_transition", ""),
        limit=12,
    )
    combined = _ordered_unique(action_tokens + observation_tokens)
    return min(1.0, len(combined) / 8.0)


def _expected_information_gain(value: Dict[str, Any]) -> float:
    return _clamp01(
        value.get("predicted_information_gain", value.get("expected_information_gain", 0.0))
    )


def _hypothesis_prefilter_score(row: Dict[str, Any]) -> float:
    posterior = _clamp01(row.get("posterior", row.get("confidence", 0.0)))
    confidence = _clamp01(row.get("confidence", posterior))
    support_bonus = min(0.16, max(0, int(row.get("support_count", 0) or 0)) * 0.04)
    contradiction_penalty = min(0.12, max(0, int(row.get("contradiction_count", 0) or 0)) * 0.03)
    information_gain = _clamp01(row.get("predicted_information_gain", row.get("expected_information_gain", 0.0)))
    conflict_bonus = min(0.12, len(list(row.get("conflicts_with", []) or [])) * 0.03)
    coverage_bonus = _prediction_token_coverage(row) * 0.12
    contender_bonus = max(0.0, 1.0 - min(1.0, abs(posterior - 0.55) / 0.55)) * 0.10
    return (
        posterior * 0.38
        + confidence * 0.20
        + information_gain * 0.12
        + support_bonus
        + conflict_bonus
        + coverage_bonus
        + contender_bonus
        - contradiction_penalty
    )


def _action_prefilter_score(action: Dict[str, Any]) -> float:
    fn_name = extract_action_function_name(action, default="")
    if not fn_name:
        return 0.0
    kind = extract_action_kind(action, default="call_tool")
    meta = _dict(action.get("_candidate_meta", {}))
    fn_text = str(fn_name or "").strip().lower()
    role = str(meta.get("role", "") or "").strip().lower()
    expected_information_gain = _clamp01(
        meta.get("expected_information_gain", action.get("expected_information_gain", 0.0))
    )
    counterfactual_delta = _clamp01(meta.get("counterfactual_delta", 0.0))
    risk = _clamp01(action.get("risk", meta.get("risk", 0.25)), default=0.25)
    execution_cost = max(0.0, float(action.get("estimated_cost", 1.0) or 1.0))
    probe_like = bool(
        kind == "probe"
        or meta.get("probe_candidate", False)
        or role in {"probe", "discriminate", "verify"}
        or any(token in fn_text for token in ("probe", "inspect", "verify", "check", "test"))
    )
    commitment_penalty = 0.22 if role == "commit" or fn_text in {"submit", "commit", "apply"} else 0.0
    grounding_bonus = 0.08 if meta.get("grounded_binding_tokens") or _dict(meta.get("intervention_target", {})) else 0.0
    return (
        (0.30 if probe_like else 0.0)
        + expected_information_gain * 0.28
        + counterfactual_delta * 0.18
        + (1.0 - risk) * 0.12
        + (1.0 / (1.0 + execution_cost)) * 0.08
        + grounding_bonus
        - commitment_penalty
    )


def _hypothesis_state_fingerprint(row: Dict[str, Any]) -> str:
    action_effects, action_effects_by_signature = _prediction_maps(row)
    payload = {
        "hypothesis_id": str(row.get("hypothesis_id", row.get("object_id", "")) or ""),
        "posterior": round(_clamp01(row.get("posterior", row.get("confidence", 0.0))), 6),
        "confidence": round(_clamp01(row.get("confidence", 0.0)), 6),
        "expected_transition": str(row.get("expected_transition", "") or ""),
        "observation_signature": list(hypothesis_observation_signature(row)),
        "predicted_action_effects": _prediction_collection_summary(action_effects),
        "predicted_action_effects_by_signature": _prediction_collection_summary(action_effects_by_signature),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _action_state_fingerprint(action: Dict[str, Any]) -> str:
    meta = _dict(action.get("_candidate_meta", {}))
    payload = {
        "semantic_signature": serialize_action_semantic_signature(_action_signature(action)),
        "risk": round(_clamp01(action.get("risk", meta.get("risk", 0.25)), default=0.25), 6),
        "estimated_cost": round(max(0.0, float(action.get("estimated_cost", 1.0) or 1.0)), 6),
        "expected_information_gain": round(
            _clamp01(meta.get("expected_information_gain", action.get("expected_information_gain", 0.0))),
            6,
        ),
        "counterfactual_delta": round(_clamp01(meta.get("counterfactual_delta", 0.0)), 6),
        "role": str(meta.get("role", "") or ""),
        "probe_candidate": bool(meta.get("probe_candidate", False)),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _pair_priority(hypothesis_a: Dict[str, Any], hypothesis_b: Dict[str, Any]) -> float:
    posterior_a = _clamp01(hypothesis_a.get("posterior", hypothesis_a.get("confidence", 0.0)))
    posterior_b = _clamp01(hypothesis_b.get("posterior", hypothesis_b.get("confidence", 0.0)))
    posterior_gap = abs(posterior_a - posterior_b)
    posterior_competition = max(0.0, 1.0 - min(1.0, posterior_gap / 0.35))
    obs_overlap = _set_overlap(
        hypothesis_observation_signature(hypothesis_a),
        hypothesis_observation_signature(hypothesis_b),
    )
    action_effects_a, action_signatures_a = _prediction_maps(hypothesis_a)
    action_effects_b, action_signatures_b = _prediction_maps(hypothesis_b)
    function_overlap = _set_overlap(list(action_effects_a.keys()), list(action_effects_b.keys()))
    signature_overlap = _set_overlap(list(action_signatures_a.keys()), list(action_signatures_b.keys()))
    transition_conflict = 0.0
    if str(hypothesis_a.get("expected_transition", "") or "").strip().lower() != str(
        hypothesis_b.get("expected_transition", "") or ""
    ).strip().lower():
        transition_conflict = 0.12
    return (
        posterior_competition * 0.42
        + obs_overlap * 0.18
        + max(function_overlap, signature_overlap) * 0.24
        + min(_prediction_token_coverage(hypothesis_a), _prediction_token_coverage(hypothesis_b)) * 0.12
        + transition_conflict
    )


def _prefilter_hypotheses(
    hypotheses: List[Dict[str, Any]],
    *,
    limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not hypotheses:
        return [], {
            "pool_count": 0,
            "limit": max(0, int(limit)),
            "kept_count": 0,
            "pruned_count": 0,
            "kept_hypothesis_ids": [],
            "pruned": [],
            "estimated_information_gain_kept": 0.0,
            "estimated_information_gain_pruned": 0.0,
            "loss_estimate": 0.0,
        }
    scored = [
        (
            _hypothesis_prefilter_score(row),
            _clamp01(row.get("posterior", row.get("confidence", 0.0))),
            _expected_information_gain(row),
            str(row.get("hypothesis_id", row.get("object_id", "")) or ""),
            dict(row),
        )
        for row in hypotheses
        if isinstance(row, dict)
    ]
    scored.sort(key=lambda item: (float(item[0]), float(item[1]), float(item[2]), item[3]), reverse=True)
    effective_limit = max(2, int(limit))
    kept = scored[:effective_limit]
    pruned = scored[effective_limit:]
    audit = {
        "pool_count": len(scored),
        "limit": effective_limit,
        "kept_count": len(kept),
        "pruned_count": len(pruned),
        "kept_hypothesis_ids": [str(item[3]) for item in kept],
        "pruned": [
            {
                "hypothesis_id": str(item[3]),
                "rank": index + 1 + len(kept),
                "prefilter_score": round(float(item[0]), 6),
                "posterior": round(float(item[1]), 6),
                "expected_information_gain": round(float(item[2]), 6),
                "prune_reason": "prefilter_rank_exceeds_limit",
            }
            for index, item in enumerate(pruned)
        ],
        "estimated_information_gain_kept": round(sum(float(item[2]) for item in kept), 6),
        "estimated_information_gain_pruned": round(sum(float(item[2]) for item in pruned), 6),
        "loss_estimate": round(sum(float(item[2]) for item in pruned), 6),
    }
    return [dict(item[4]) for item in kept], audit


def _prefilter_actions(
    actions: List[Dict[str, Any]],
    *,
    limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    deduped: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
    for action in list(actions or []):
        if not isinstance(action, dict):
            continue
        signature = _action_signature(action)
        if not signature[0]:
            continue
        current = deduped.get(signature)
        if current is None or _action_prefilter_score(action) > _action_prefilter_score(current):
            deduped[signature] = dict(action)
    ranked = sorted(
        deduped.values(),
        key=lambda item: (
            _action_prefilter_score(item),
            _clamp01(_dict(item.get("_candidate_meta", {})).get("expected_information_gain", item.get("expected_information_gain", 0.0))),
            -_clamp01(item.get("risk", _dict(item.get("_candidate_meta", {})).get("risk", 0.25)), default=0.25),
        ),
        reverse=True,
    )
    effective_limit = max(1, int(limit))
    kept = [dict(item) for item in ranked[:effective_limit]]
    pruned = [dict(item) for item in ranked[effective_limit:]]
    audit = {
        "pool_count": len(ranked),
        "limit": effective_limit,
        "kept_count": len(kept),
        "pruned_count": len(pruned),
        "kept_signatures": [
            serialize_action_semantic_signature(_action_signature(item))
            for item in kept
        ],
        "pruned": [
            {
                "semantic_signature": serialize_action_semantic_signature(_action_signature(item)),
                "function_name": extract_action_function_name(item, default=""),
                "rank": index + 1 + len(kept),
                "prefilter_score": round(_action_prefilter_score(item), 6),
                "expected_information_gain": round(
                    _clamp01(
                        _dict(item.get("_candidate_meta", {})).get(
                            "expected_information_gain",
                            item.get("expected_information_gain", 0.0),
                        )
                    ),
                    6,
                ),
                "risk": round(
                    _clamp01(item.get("risk", _dict(item.get("_candidate_meta", {})).get("risk", 0.25)), default=0.25),
                    6,
                ),
                "prune_reason": "prefilter_rank_exceeds_limit",
            }
            for index, item in enumerate(pruned)
        ],
        "estimated_information_gain_kept": round(
            sum(
                _clamp01(
                    _dict(item.get("_candidate_meta", {})).get(
                        "expected_information_gain",
                        item.get("expected_information_gain", 0.0),
                    )
                )
                for item in kept
            ),
            6,
        ),
        "estimated_information_gain_pruned": round(
            sum(
                _clamp01(
                    _dict(item.get("_candidate_meta", {})).get(
                        "expected_information_gain",
                        item.get("expected_information_gain", 0.0),
                    )
                )
                for item in pruned
            ),
            6,
        ),
    }
    audit["loss_estimate"] = audit["estimated_information_gain_pruned"]
    return kept, audit


def _pair_information_gain_estimate(
    hypothesis_a: Dict[str, Any],
    hypothesis_b: Dict[str, Any],
    pair_priority: float,
) -> float:
    return min(
        1.0,
        max(
            float(pair_priority),
            _expected_information_gain(hypothesis_a) * 0.65 + _expected_information_gain(hypothesis_b) * 0.35,
            _expected_information_gain(hypothesis_b) * 0.65 + _expected_information_gain(hypothesis_a) * 0.35,
        ),
    )


def _prune_hypothesis_pairs(
    hypotheses: List[Dict[str, Any]],
    *,
    pair_budget: int,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any], float, str]], Dict[str, Any]]:
    pairs: List[Tuple[float, float, float, str, Dict[str, Any], Dict[str, Any], str]] = []
    for idx_a in range(len(hypotheses)):
        for idx_b in range(idx_a + 1, len(hypotheses)):
            hyp_a = hypotheses[idx_a]
            hyp_b = hypotheses[idx_b]
            pair_id = "::".join(
                sorted(
                    [
                        str(hyp_a.get("hypothesis_id", hyp_a.get("object_id", "")) or ""),
                        str(hyp_b.get("hypothesis_id", hyp_b.get("object_id", "")) or ""),
                    ]
                )
            )
            pair_priority = _pair_priority(hyp_a, hyp_b)
            pair_fingerprint = json.dumps(
                sorted([_hypothesis_state_fingerprint(hyp_a), _hypothesis_state_fingerprint(hyp_b)]),
                ensure_ascii=False,
                sort_keys=True,
            )
            pair_information_gain = _pair_information_gain_estimate(hyp_a, hyp_b, pair_priority)
            pairs.append(
                (
                    pair_priority,
                    pair_information_gain,
                    max(
                        _clamp01(hyp_a.get("posterior", hyp_a.get("confidence", 0.0))),
                        _clamp01(hyp_b.get("posterior", hyp_b.get("confidence", 0.0))),
                    ),
                    pair_id,
                    dict(hyp_a),
                    dict(hyp_b),
                    pair_fingerprint,
                )
            )
    pairs.sort(key=lambda item: (float(item[0]), float(item[1]), float(item[2]), item[3]), reverse=True)
    effective_budget = max(1, int(pair_budget))
    kept = pairs[:effective_budget]
    pruned = pairs[effective_budget:]
    audit = {
        "pool_count": len(pairs),
        "limit": effective_budget,
        "kept_count": len(kept),
        "pruned_count": len(pruned),
        "kept": [
            {
                "pair_id": str(item[3]),
                "rank": index + 1,
                "pair_priority": round(float(item[0]), 6),
                "expected_information_gain": round(float(item[1]), 6),
            }
            for index, item in enumerate(kept)
        ],
        "pruned": [
            {
                "pair_id": str(item[3]),
                "rank": index + 1 + len(kept),
                "pair_priority": round(float(item[0]), 6),
                "expected_information_gain": round(float(item[1]), 6),
                "prune_reason": "pair_budget_exhausted",
            }
            for index, item in enumerate(pruned)
        ],
        "estimated_information_gain_kept": round(sum(float(item[1]) for item in kept), 6),
        "estimated_information_gain_pruned": round(sum(float(item[1]) for item in pruned), 6),
        "loss_estimate": round(sum(float(item[1]) for item in pruned), 6),
    }
    return [
        (dict(item[4]), dict(item[5]), float(item[0]), str(item[6]))
        for item in kept
    ], audit


def _builder_budget_audit(
    *,
    hypothesis_audit: Dict[str, Any],
    action_audit: Dict[str, Any],
    pair_audit: Dict[str, Any],
) -> Dict[str, Any]:
    total_loss = round(
        float(hypothesis_audit.get("loss_estimate", 0.0) or 0.0)
        + float(action_audit.get("loss_estimate", 0.0) or 0.0)
        + float(pair_audit.get("loss_estimate", 0.0) or 0.0),
        6,
    )
    return {
        "hypothesis_selection": dict(hypothesis_audit),
        "action_selection": dict(action_audit),
        "pair_selection": dict(pair_audit),
        "budget_loss_estimate": {
            "hypothesis_information_gain_pruned": round(
                float(hypothesis_audit.get("estimated_information_gain_pruned", 0.0) or 0.0),
                6,
            ),
            "action_information_gain_pruned": round(
                float(action_audit.get("estimated_information_gain_pruned", 0.0) or 0.0),
                6,
            ),
            "pair_information_gain_pruned": round(
                float(pair_audit.get("estimated_information_gain_pruned", 0.0) or 0.0),
                6,
            ),
            "total": total_loss,
        },
    }


def _reward_sign_from_prediction(prediction: Dict[str, Any]) -> str:
    reward_sign = str(
        prediction.get("reward_sign", prediction.get("predicted_reward_sign", ""))
        or ""
    ).strip().lower()
    if reward_sign:
        return reward_sign
    reward_value = prediction.get("reward", prediction.get("predicted_reward", None))
    try:
        reward = float(reward_value)
    except (TypeError, ValueError):
        return ""
    if reward > 0.0:
        return "positive"
    if reward < 0.0:
        return "negative"
    return "zero"


def _observation_overlap_score(a: List[str], b: List[str]) -> float:
    set_a = {str(item or "").strip().lower() for item in a if str(item or "").strip()}
    set_b = {str(item or "").strip().lower() for item in b if str(item or "").strip()}
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a ^ set_b) / float(len(union))


@dataclass
class ExperimentProposal:
    experiment_id: str
    summary: str
    function_name: str
    candidate_action: Dict[str, Any]
    discriminates_between: List[str] = field(default_factory=list)
    expected_outcomes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    expected_information_gain: float = 0.0
    execution_cost: float = 0.0
    reversibility: float = 0.0
    risk: float = 0.0
    score: float = 0.0
    reason: str = ""


def score_discriminating_experiment(
    action: Dict[str, Any],
    hypothesis_a: Dict[str, Any],
    hypothesis_b: Dict[str, Any],
) -> Dict[str, Any]:
    fn_name = extract_action_function_name(action, default="")
    if not fn_name:
        return {}
    pred_a = hypothesis_action_prediction(hypothesis_a, action)
    pred_b = hypothesis_action_prediction(hypothesis_b, action)
    if not pred_a and not pred_b:
        return {}

    discrimination_power = 0.0
    reasons: List[str] = []
    phase_a = str(pred_a.get("predicted_phase_shift", pred_a.get("phase_shift", hypothesis_a.get("predicted_phase_shift", ""))) or "").strip().lower()
    phase_b = str(pred_b.get("predicted_phase_shift", pred_b.get("phase_shift", hypothesis_b.get("predicted_phase_shift", ""))) or "").strip().lower()
    if phase_a != phase_b:
        discrimination_power += 0.28
        reasons.append("phase_shift")

    valid_a = pred_a.get("valid_state_change")
    valid_b = pred_b.get("valid_state_change")
    if valid_a is not None and valid_b is not None and bool(valid_a) != bool(valid_b):
        discrimination_power += 0.22
        reasons.append("state_change")

    reward_a = _reward_sign_from_prediction(pred_a)
    reward_b = _reward_sign_from_prediction(pred_b)
    if reward_a and reward_b and reward_a != reward_b:
        discrimination_power += 0.18
        reasons.append("reward_sign")

    risk_a = str(pred_a.get("risk_type", "") or "").strip().lower()
    risk_b = str(pred_b.get("risk_type", "") or "").strip().lower()
    if risk_a and risk_b and risk_a != risk_b:
        discrimination_power += 0.12
        reasons.append("risk_type")

    target_a = str(pred_a.get("target_family", pred_a.get("target", "")) or "").strip().lower()
    target_b = str(pred_b.get("target_family", pred_b.get("target", "")) or "").strip().lower()
    relation_a = str(pred_a.get("relation_type", "") or "").strip().lower()
    relation_b = str(pred_b.get("relation_type", "") or "").strip().lower()
    if target_a and target_b and target_a != target_b:
        discrimination_power += 0.08
        reasons.append("target_family")
    if relation_a and relation_b and relation_a != relation_b:
        discrimination_power += 0.08
        reasons.append("relation_type")

    obs_a = pred_a.get("predicted_observation_tokens", hypothesis_observation_signature(hypothesis_a))
    obs_b = pred_b.get("predicted_observation_tokens", hypothesis_observation_signature(hypothesis_b))
    overlap = _observation_overlap_score(obs_a if isinstance(obs_a, list) else [], obs_b if isinstance(obs_b, list) else [])
    if overlap > 0.0:
        discrimination_power += min(0.20, overlap * 0.20)
        reasons.append("observation_tokens")

    discrimination_power = min(1.0, discrimination_power)
    expected_information_gain = min(
        1.0,
        max(
            _clamp01(pred_a.get("predicted_information_gain", hypothesis_a.get("predicted_information_gain", 0.0))),
            _clamp01(pred_b.get("predicted_information_gain", hypothesis_b.get("predicted_information_gain", 0.0))),
            _clamp01(
                (action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}).get(
                    "expected_information_gain",
                    0.0,
                )
            ),
        ),
    )
    fn_text = str(fn_name or "").strip().lower()
    reversibility = 0.85 if fn_text == "wait" or any(token in fn_text for token in ("probe", "inspect", "verify", "check", "test")) else 0.55
    risk = _clamp01(
        action.get("risk", (action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}).get("risk", 0.25)),
        default=0.25,
    )
    execution_cost = max(0.0, float(action.get("estimated_cost", 1.0) or 1.0))
    low_risk_bonus = 1.0 - risk
    low_cost_bonus = 1.0 / (1.0 + execution_cost)
    score = (
        discrimination_power * 0.45
        + expected_information_gain * 0.20
        + reversibility * 0.15
        + low_risk_bonus * 0.10
        + low_cost_bonus * 0.10
    )
    return {
        "discrimination_power": round(discrimination_power, 6),
        "expected_information_gain": round(expected_information_gain, 6),
        "execution_cost": round(execution_cost, 6),
        "reversibility": round(reversibility, 6),
        "risk": round(risk, 6),
        "score": round(score, 6),
        "reason": ",".join(reasons) if reasons else "low_separation",
        "expected_outcomes": {
            str(hypothesis_a.get("hypothesis_id", "") or ""): dict(pred_a),
            str(hypothesis_b.get("hypothesis_id", "") or ""): dict(pred_b),
        },
    }


def build_discriminating_experiments(
    hypotheses: List[Dict[str, Any]],
    candidate_actions: List[Dict[str, Any]],
    *,
    limit: int = 5,
    hypothesis_limit: Optional[int] = None,
    action_limit: Optional[int] = None,
    pair_budget: Optional[int] = None,
    previous_experiments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    normalized_hypotheses = normalize_hypothesis_rows(list(hypotheses or []), fallback_id_prefix="hx")
    actions = [dict(action) for action in list(candidate_actions or []) if isinstance(action, dict)]
    if len(normalized_hypotheses) < 2 or not actions:
        return []

    filtered_hypotheses, hypothesis_audit = _prefilter_hypotheses(
        normalized_hypotheses,
        limit=min(len(normalized_hypotheses), max(int(hypothesis_limit or (limit * 3 or 6)), max(2, int(limit) + 1))),
    )
    filtered_actions, action_audit = _prefilter_actions(
        actions,
        limit=min(len(actions), max(int(action_limit or (limit * 3 or 8)), max(1, int(limit)))),
    )
    if len(filtered_hypotheses) < 2 or not filtered_actions:
        return []
    pruned_pairs, pair_audit = _prune_hypothesis_pairs(
        filtered_hypotheses,
        pair_budget=min(
            max(1, (len(filtered_hypotheses) * max(0, len(filtered_hypotheses) - 1)) // 2),
            max(int(pair_budget or (limit * 6 or 12)), max(1, int(limit))),
        ),
    )
    if not pruned_pairs:
        return []
    budget_audit = _builder_budget_audit(
        hypothesis_audit=hypothesis_audit,
        action_audit=action_audit,
        pair_audit=pair_audit,
    )

    pair_scope_fingerprint = json.dumps(
        [str(item[3]) for item in pruned_pairs],
        ensure_ascii=False,
        sort_keys=True,
    )
    previous_by_signature: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
    for row in list(previous_experiments or []):
        if not isinstance(row, dict):
            continue
        candidate_action = row.get("candidate_action", {}) if isinstance(row.get("candidate_action", {}), dict) else {}
        signature = _action_signature(candidate_action)
        if not signature[0]:
            continue
        previous_by_signature[signature] = dict(row)

    proposals_by_signature: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
    for action in filtered_actions:
        signature = _action_signature(action)
        fn_name = signature[0]
        if not fn_name:
            continue
        action_fingerprint = _action_state_fingerprint(action)
        previous_row = previous_by_signature.get(signature)
        if (
            previous_row is not None
            and str(previous_row.get("_builder_pair_scope_fingerprint", "") or "") == pair_scope_fingerprint
            and str(previous_row.get("_builder_action_fingerprint", "") or "") == action_fingerprint
        ):
            cached = dict(previous_row)
            cached["_builder_cache_hit"] = True
            cached["_builder_hypothesis_count"] = len(filtered_hypotheses)
            cached["_builder_action_count"] = len(filtered_actions)
            cached["_builder_pair_count"] = len(pruned_pairs)
            cached["_builder_budget_audit"] = dict(budget_audit)
            proposals_by_signature[signature] = cached
            continue

        for hyp_a, hyp_b, _pair_priority_score, pair_fingerprint in pruned_pairs:
                score_row = score_discriminating_experiment(action, hyp_a, hyp_b)
                if not score_row or float(score_row.get("score", 0.0) or 0.0) <= 0.0:
                    continue
                proposal = proposals_by_signature.get(signature)
                pair_label = [str(hyp_a.get("hypothesis_id", "") or ""), str(hyp_b.get("hypothesis_id", "") or "")]
                if proposal is None:
                    experiment = ExperimentProposal(
                        experiment_id=f"exp_{len(proposals_by_signature) + 1}_{fn_name}",
                        summary=f"Discriminate {pair_label[0]} vs {pair_label[1]} via {fn_name}",
                        function_name=fn_name,
                        candidate_action=dict(action),
                        discriminates_between=pair_label,
                        expected_outcomes=_dict(score_row.get("expected_outcomes", {})),
                        expected_information_gain=_clamp01(score_row.get("expected_information_gain", 0.0)),
                        execution_cost=float(score_row.get("execution_cost", 0.0) or 0.0),
                        reversibility=_clamp01(score_row.get("reversibility", 0.0)),
                        risk=_clamp01(score_row.get("risk", 0.0)),
                        score=float(score_row.get("score", 0.0) or 0.0),
                        reason=str(score_row.get("reason", "") or ""),
                    )
                    proposal_row = asdict(experiment)
                    proposal_row["_builder_cache_hit"] = False
                    proposal_row["_builder_action_fingerprint"] = action_fingerprint
                    proposal_row["_builder_pair_scope_fingerprint"] = pair_scope_fingerprint
                    proposal_row["_builder_pair_fingerprint"] = pair_fingerprint
                    proposal_row["_builder_hypothesis_count"] = len(filtered_hypotheses)
                    proposal_row["_builder_action_count"] = len(filtered_actions)
                    proposal_row["_builder_pair_count"] = len(pruned_pairs)
                    proposal_row["_builder_budget_audit"] = dict(budget_audit)
                    proposals_by_signature[signature] = proposal_row
                    continue

                proposal["score"] = min(1.0, float(proposal.get("score", 0.0) or 0.0) + float(score_row.get("score", 0.0) or 0.0) * 0.35)
                proposal["expected_information_gain"] = max(
                    _clamp01(proposal.get("expected_information_gain", 0.0)),
                    _clamp01(score_row.get("expected_information_gain", 0.0)),
                )
                proposal["reversibility"] = max(
                    _clamp01(proposal.get("reversibility", 0.0)),
                    _clamp01(score_row.get("reversibility", 0.0)),
                )
                proposal["risk"] = min(
                    _clamp01(proposal.get("risk", 1.0), default=1.0),
                    _clamp01(score_row.get("risk", 1.0), default=1.0),
                )
                proposal["execution_cost"] = min(
                    float(proposal.get("execution_cost", 0.0) or 0.0),
                    float(score_row.get("execution_cost", 0.0) or 0.0),
                )
                discriminates_between = list(proposal.get("discriminates_between", []) or [])
                for hypothesis_id in pair_label:
                    if hypothesis_id and hypothesis_id not in discriminates_between:
                        discriminates_between.append(hypothesis_id)
                proposal["discriminates_between"] = discriminates_between
                expected_outcomes = _dict(proposal.get("expected_outcomes", {}))
                expected_outcomes.update(_dict(score_row.get("expected_outcomes", {})))
                proposal["expected_outcomes"] = expected_outcomes
                reasons = {
                    reason
                    for reason in [
                        *(str(proposal.get("reason", "") or "").split(",")),
                        *(str(score_row.get("reason", "") or "").split(",")),
                    ]
                    if reason
                }
                proposal["reason"] = ",".join(sorted(reasons))
                proposal["summary"] = (
                    f"Discriminate {', '.join(discriminates_between)} via {fn_name}"
                )
                proposal["_builder_cache_hit"] = False
                proposal["_builder_action_fingerprint"] = action_fingerprint
                proposal["_builder_pair_scope_fingerprint"] = pair_scope_fingerprint
                proposal["_builder_pair_fingerprint"] = pair_fingerprint
                proposal["_builder_hypothesis_count"] = len(filtered_hypotheses)
                proposal["_builder_action_count"] = len(filtered_actions)
                proposal["_builder_pair_count"] = len(pruned_pairs)
                proposal["_builder_budget_audit"] = dict(budget_audit)

    proposals = list(proposals_by_signature.values())
    proposals.sort(
        key=lambda item: (
            float(item.get("score", 0.0) or 0.0),
            float(item.get("expected_information_gain", 0.0) or 0.0),
            float(item.get("reversibility", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return [dict(item) for item in proposals[: max(0, int(limit))]]
