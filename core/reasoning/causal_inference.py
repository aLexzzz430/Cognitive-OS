from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from core.orchestration.action_utils import (
    action_semantic_signature_key,
    extract_action_function_name,
    extract_action_kind,
)
from core.reasoning.executable_hypothesis import ExecutableHypothesis, build_executable_hypotheses
from core.reasoning.hypothesis_schema import hypothesis_observation_signature, normalize_hypothesis_rows
from core.world_model.object_graph import build_runtime_object_graph
from modules.world_model.mechanism_runtime import (
    build_mechanism_runtime_state,
    evaluate_mechanism_preconditions,
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _reward_sign(reward: Any) -> str:
    reward_value = float(reward or 0.0)
    if reward_value > 0.0:
        return "positive"
    if reward_value < 0.0:
        return "negative"
    return "zero"


def _verifier_teaching_signal(verifier_teaching: Optional[Mapping[str, Any]]) -> tuple[str, float]:
    payload = _as_dict(verifier_teaching)
    signal = str(payload.get("teaching_signal", "") or "none").strip().lower()
    if signal not in {"positive", "negative", "none"}:
        signal = "none"
    try:
        score = float(payload.get("teaching_signal_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return signal, score


def _state_changed(result: Mapping[str, Any], actual_transition: Mapping[str, Any]) -> bool:
    if "valid_state_change" in actual_transition:
        return bool(actual_transition.get("valid_state_change"))
    return bool(result.get("state_changed", False) or result.get("observation_changed", False))


def _observation_tokens(result: Mapping[str, Any], actual_transition: Mapping[str, Any], runtime_graph: Mapping[str, Any]) -> List[str]:
    tokens = _string_list(actual_transition.get("observation_tokens", []))
    if not tokens:
        tokens = _string_list(actual_transition.get("predicted_observation_tokens", []))
    scene_state = _as_dict(runtime_graph.get("scene_state", {}))
    tokens.extend(_string_list(scene_state.get("signal_tokens", [])))
    tokens.extend(_string_list(scene_state.get("counterevidence_tokens", [])))
    if not tokens:
        tokens.append(str(result.get("belief_phase", result.get("status", "")) or ""))
    ordered: List[str] = []
    seen = set()
    for token in tokens:
        lower = str(token or "").strip().lower()
        if not lower or lower in seen:
            continue
        seen.add(lower)
        ordered.append(lower)
    return ordered[:12]


def _token_overlap(expected: Sequence[str], actual: Sequence[str]) -> float:
    expected_set = {str(item or "").strip().lower() for item in list(expected or []) if str(item or "").strip()}
    actual_set = {str(item or "").strip().lower() for item in list(actual or []) if str(item or "").strip()}
    if not expected_set or not actual_set:
        return 0.0
    return len(expected_set & actual_set) / float(max(1, len(expected_set)))


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


def _marker_slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return "_".join(part for part in text.replace("::", " ").replace("-", " ").split() if part)


def _set_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {str(item or "").strip().lower() for item in list(left or []) if str(item or "").strip()}
    right_set = {str(item or "").strip().lower() for item in list(right or []) if str(item or "").strip()}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / float(max(1, min(len(left_set), len(right_set))))


def _append_evidence_entry(row: Dict[str, Any], field_name: str, payload: Dict[str, Any], *, limit: int = 6) -> None:
    existing = row.get(field_name, [])
    entries = [dict(item) for item in list(existing or []) if isinstance(item, dict)]
    entries.append(dict(payload))
    row[field_name] = entries[-max(1, int(limit)) :]


def _posterior_event(
    *,
    hypothesis_id: str,
    event_type: str,
    delta: float,
    reason_tokens: Sequence[str],
    support: float,
    contradiction: float,
    matched_focus_object_ids: Sequence[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "hypothesis_id": str(hypothesis_id or ""),
        "event_type": str(event_type or "unresolved"),
        "delta": round(float(delta or 0.0), 6),
        "reason": ",".join(_ordered_unique(list(reason_tokens or []))) or "insufficient_evidence",
        "support": round(float(support or 0.0), 6),
        "contradiction": round(float(contradiction or 0.0), 6),
        "matched_focus_object_ids": list(matched_focus_object_ids or []),
    }
    if isinstance(metadata, dict) and metadata:
        event["metadata"] = dict(metadata)
    return event


def _derive_falsifier_tokens(
    row: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    reward: float,
    actual_transition: Mapping[str, Any],
    scene_state: Mapping[str, Any],
) -> List[str]:
    markers: List[str] = []
    reasons = {str(item or "").strip() for item in list(evidence.get("reasons", []) or []) if str(item or "").strip()}
    if "reward_sign_mismatch" in reasons:
        markers.append(f"reward::{_reward_sign(reward)}")
    if "phase_mismatch" in reasons:
        next_phase = str(actual_transition.get("next_phase", "") or scene_state.get("phase", "") or "").strip()
        if next_phase:
            markers.append(f"phase::{_marker_slug(next_phase)}")
    if "state_change_mismatch" in reasons:
        markers.append(
            "state_change::changed"
            if _state_changed(_as_dict({}), actual_transition)
            else "state_change::unchanged"
        )
    if "runtime_precondition_unsatisfied" in reasons:
        for item in list(evidence.get("runtime_unmet_preconditions", []) or [])[:4]:
            slug = _marker_slug(item)
            if slug:
                markers.append(f"precondition::{slug}")
    if "counterevidence_match" in reasons:
        for token in list(scene_state.get("counterevidence_tokens", []) or [])[:4]:
            slug = _marker_slug(token)
            if slug:
                markers.append(f"counterevidence::{slug}")
        if not scene_state.get("counterevidence_tokens"):
            for token in list(evidence.get("actual_tokens", []) or [])[:4]:
                slug = _marker_slug(token)
                if slug:
                    markers.append(f"counterevidence::{slug}")
    if "rival_unique_signal" in reasons:
        for token in list(evidence.get("actual_tokens", []) or [])[:3]:
            slug = _marker_slug(token)
            if slug:
                markers.append(f"rival_signal::{slug}")
    existing = [str(item) for item in list(row.get("falsifiers", []) or []) if str(item or "").strip()]
    return _ordered_unique(existing + markers)


def _infer_conflict_map(
    rows: Sequence[Dict[str, Any]],
    *,
    executable_by_id: Mapping[str, ExecutableHypothesis],
) -> Dict[str, Set[str]]:
    conflict_map: Dict[str, Set[str]] = {}
    for row in list(rows or []):
        hypothesis_id = str(row.get("hypothesis_id", "") or "")
        if not hypothesis_id:
            continue
        conflict_map[hypothesis_id] = {
            str(item).strip()
            for item in list(row.get("conflicts_with", []) or [])
            if str(item).strip() and str(item).strip() != hypothesis_id
        }

    ids = [str(row.get("hypothesis_id", "") or "") for row in list(rows or []) if str(row.get("hypothesis_id", "") or "")]
    for index, left_id in enumerate(ids):
        left_row = next((row for row in rows if str(row.get("hypothesis_id", "") or "") == left_id), {})
        left_exec = executable_by_id.get(left_id)
        if left_exec is None:
            continue
        left_signature = hypothesis_observation_signature(left_row) + list(left_exec.target_tokens or [])
        left_actions = set(left_exec.action_rules.keys())
        left_targets = list(left_exec.target_tokens or [])
        for right_id in ids[index + 1 :]:
            right_row = next((row for row in rows if str(row.get("hypothesis_id", "") or "") == right_id), {})
            right_exec = executable_by_id.get(right_id)
            if right_exec is None:
                continue
            right_signature = hypothesis_observation_signature(right_row) + list(right_exec.target_tokens or [])
            right_actions = set(right_exec.action_rules.keys())
            right_targets = list(right_exec.target_tokens or [])
            semantic_overlap = _set_overlap(left_signature, right_signature)
            target_overlap = _set_overlap(left_targets, right_targets)
            shared_actions = sorted(left_actions & right_actions)
            diverging_predictions = False
            for action_name in shared_actions:
                left_rule = left_exec.action_rules.get(action_name)
                right_rule = right_exec.action_rules.get(action_name)
                if left_rule is None or right_rule is None:
                    continue
                if (
                    left_rule.phase_shift != right_rule.phase_shift
                    or left_rule.reward_sign != right_rule.reward_sign
                    or left_rule.valid_state_change != right_rule.valid_state_change
                ):
                    diverging_predictions = True
                    break
            if diverging_predictions or (
                semantic_overlap >= 0.25
                and target_overlap >= 0.15
                and str(left_row.get("family", "") or "") != str(right_row.get("family", "") or "")
            ):
                conflict_map.setdefault(left_id, set()).add(right_id)
                conflict_map.setdefault(right_id, set()).add(left_id)
    return conflict_map


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


def _action_family(action: Mapping[str, Any]) -> str:
    meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
    for key in ("action_family", "runtime_action_family", "solver_dominant_interaction_mode"):
        family = _normalize_action_family(meta.get(key))
        if family:
            return family
    function_name = extract_action_function_name(action, default="")
    kind = extract_action_kind(action, default="call_tool")
    if kind == "wait" or function_name == "wait":
        return "wait"
    return _normalize_action_family(function_name)


def _runtime_preconditions(
    hypothesis: ExecutableHypothesis,
    *,
    action: Mapping[str, Any],
) -> List[str]:
    metadata = _as_dict(hypothesis.metadata)
    transition_rules = metadata.get("transition_rules", [])
    action_family = _action_family(action)
    if isinstance(transition_rules, list):
        for rule in transition_rules:
            if not isinstance(rule, dict):
                continue
            if _normalize_action_family(rule.get("action_family", "")) != action_family:
                continue
            rows = _string_list(rule.get("preconditions", []))
            if rows:
                return rows
    return _string_list(metadata.get("preconditions", [])) or _string_list(metadata.get("required_state", []))


def _action_binding_tokens(action: Mapping[str, Any]) -> List[str]:
    meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
    intervention_target = meta.get("intervention_target", {}) if isinstance(meta.get("intervention_target", {}), dict) else {}
    tokens = _string_list(meta.get("grounded_binding_tokens", []))
    tokens.extend(
        [
            str(meta.get("anchor_ref", "") or ""),
            str(intervention_target.get("anchor_ref", "") or ""),
            str(intervention_target.get("target_kind", "") or ""),
        ]
    )
    function_name = str(_as_dict(_as_dict(action.get("payload", {})).get("tool_args", {})).get("function_name", "") or action.get("function_name", "") or "")
    tokens.append(function_name)
    ordered: List[str] = []
    seen = set()
    for token in tokens:
        for part in str(token or "").strip().lower().replace("::", "_").replace("-", "_").split("_"):
            normalized = str(part or "").strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
    return ordered


def _evaluate_hypothesis(
    hypothesis: ExecutableHypothesis,
    *,
    action: Mapping[str, Any],
    result: Mapping[str, Any],
    predicted_transition: Mapping[str, Any],
    actual_transition: Mapping[str, Any],
    runtime_graph: Mapping[str, Any],
    reward: float,
    information_gain: float,
    unique_tokens: Optional[Sequence[str]] = None,
    rival_unique_tokens: Optional[Sequence[str]] = None,
    obs_before: Optional[Mapping[str, Any]] = None,
    mechanism_control_summary: Optional[Mapping[str, Any]] = None,
    verifier_teaching: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    function_name = str(
        _as_dict(_as_dict(action.get("payload", {})).get("tool_args", {})).get("function_name", "")
        or action.get("function_name", "")
        or ""
    ).strip()
    signature_key = action_semantic_signature_key(dict(action or {})) if isinstance(action, dict) else ""
    rule = (
        hypothesis.action_rules_by_signature.get(signature_key)
        if signature_key
        else None
    )
    if rule is None:
        rule = hypothesis.action_rules.get(function_name)
    actual_tokens = _observation_tokens(result, actual_transition, runtime_graph)
    action_tokens = _action_binding_tokens(action)
    scene_state = _as_dict(runtime_graph.get("scene_state", {}))
    phase = str(
        actual_transition.get("next_phase", "")
        or result.get("belief_phase", "")
        or scene_state.get("phase", "")
        or ""
    ).strip().lower()
    support = 0.0
    contradiction = 0.0
    unresolved = 0.0
    reasons: List[str] = []
    matched_focus_ids = list(runtime_graph.get("focus_object_ids", []) or [])
    state_changed = _state_changed(result, actual_transition)
    runtime_preconditions = _runtime_preconditions(hypothesis, action=action)
    runtime_state = build_mechanism_runtime_state(
        _as_dict(obs_before),
        _as_dict(mechanism_control_summary),
        action_tokens=action_tokens,
        target_tokens=(rule.target_tokens if rule is not None else hypothesis.target_tokens),
    )
    precondition_report = evaluate_mechanism_preconditions(
        runtime_preconditions,
        runtime_state=runtime_state,
    ) if runtime_preconditions else {
        "has_preconditions": False,
        "satisfied": True,
        "support": 1.0,
        "matched": [],
        "unmet": [],
    }

    if rule is not None:
        if rule.reward_sign:
            if rule.reward_sign == _reward_sign(reward):
                support += 0.18
                reasons.append("reward_sign_match")
            else:
                contradiction += 0.18
                reasons.append("reward_sign_mismatch")
        if rule.valid_state_change is not None:
            if bool(rule.valid_state_change) == bool(state_changed):
                support += 0.16
                reasons.append("state_change_match")
            else:
                contradiction += 0.16
                reasons.append("state_change_mismatch")
        if rule.phase_shift:
            if phase and phase == str(rule.phase_shift or "").strip().lower():
                support += 0.14
                reasons.append("phase_match")
            elif phase:
                contradiction += 0.12
                reasons.append("phase_mismatch")
        overlap = _token_overlap(rule.observation_tokens, actual_tokens)
        if overlap >= 0.15:
            support += min(0.26, 0.12 + overlap * 0.2)
            reasons.append("rule_observation_support")
        elif rule.observation_tokens:
            contradiction += 0.10
            reasons.append("rule_observation_mismatch")
        target_overlap = _token_overlap(rule.target_tokens or hypothesis.target_tokens, action_tokens)
        if target_overlap >= 0.15:
            support += min(0.20, 0.08 + target_overlap * 0.18)
            reasons.append("target_binding_support")
        elif rule.target_tokens:
            contradiction += 0.08
            reasons.append("target_binding_mismatch")
        if information_gain >= max(0.18, float(rule.information_gain or 0.0)):
            support += 0.06
            reasons.append("information_gain_match")
    else:
        generic_overlap = _token_overlap(hypothesis.observation_tokens, actual_tokens)
        if generic_overlap >= 0.15:
            support += min(0.20, 0.08 + generic_overlap * 0.16)
            reasons.append("generic_observation_support")
        binding_overlap = _token_overlap(hypothesis.target_tokens, action_tokens)
        if binding_overlap >= 0.15:
            support += min(0.16, 0.06 + binding_overlap * 0.14)
            reasons.append("generic_binding_support")
        elif state_changed or reward > 0.0:
            contradiction += 0.10
            reasons.append("generic_binding_mismatch")

    if precondition_report.get("has_preconditions", False):
        if bool(precondition_report.get("satisfied", False)):
            support += 0.08
            reasons.append("runtime_precondition_support")
        else:
            partial_support = float(precondition_report.get("support", 0.0) or 0.0)
            if partial_support > 0.0:
                support += 0.02 * partial_support
            if reward < 0.0 or not state_changed:
                contradiction += 0.06 + (0.06 * max(0.0, 1.0 - partial_support))
            else:
                contradiction += 0.03 + (0.03 * max(0.0, 1.0 - partial_support))
                unresolved += 0.02
            reasons.append("runtime_precondition_unsatisfied")
    elif hypothesis.precondition_tokens:
        precondition_overlap = _token_overlap(hypothesis.precondition_tokens, actual_tokens + _string_list(scene_state.get("signal_tokens", [])))
        if precondition_overlap >= 0.15:
            support += 0.05
            reasons.append("precondition_support")
        elif reward < 0.0 and not state_changed:
            contradiction += 0.06
            reasons.append("precondition_unsatisfied")

    if hypothesis.counterevidence_tokens:
        counter_overlap = _token_overlap(hypothesis.counterevidence_tokens, actual_tokens + _string_list(scene_state.get("counterevidence_tokens", [])))
        if counter_overlap >= 0.15:
            contradiction += min(0.28, 0.14 + counter_overlap * 0.18)
            reasons.append("counterevidence_match")

    if hypothesis.recovery_tokens and bool(scene_state.get("recovery_required", False)):
        support += 0.04
        reasons.append("recovery_context_support")

    unique_overlap = _token_overlap(unique_tokens or [], actual_tokens)
    rival_overlap = _token_overlap(rival_unique_tokens or [], actual_tokens)
    if unique_overlap >= 0.15:
        support += min(0.34, 0.14 + unique_overlap * 0.22)
        reasons.append("unique_signal_support")
    if rival_overlap >= 0.15:
        contradiction += min(0.24, 0.10 + rival_overlap * 0.18)
        reasons.append("rival_unique_signal")

    if bool(result.get("solved", False)):
        if reward > 0.0 or _token_overlap(hypothesis.observation_tokens, actual_tokens) >= 0.15:
            support += 0.20
            reasons.append("terminal_support")
    elif reward > 0.0:
        support += 0.10
        reasons.append("progress_support")
    elif reward < 0.0 and not reasons:
        contradiction += 0.10
        reasons.append("negative_outcome")

    teaching_signal, teaching_score = _verifier_teaching_signal(verifier_teaching)
    teaching_strength = min(0.18, abs(float(teaching_score or 0.0)) * 0.18)
    if teaching_signal == "positive" and teaching_strength > 0.0:
        support += teaching_strength
        reasons.append("verifier_teaching_positive")
    elif teaching_signal == "negative" and teaching_strength > 0.0:
        contradiction += teaching_strength
        reasons.append("verifier_teaching_negative")

    if not reasons:
        unresolved += 0.08
        reasons.append("insufficient_evidence")

    return {
        "support": round(support, 6),
        "contradiction": round(contradiction, 6),
        "unresolved": round(unresolved, 6),
        "reasons": reasons,
        "matched_focus_object_ids": matched_focus_ids[:6],
        "action_rule_used": bool(rule is not None),
        "actual_tokens": actual_tokens,
        "action_tokens": action_tokens,
        "runtime_precondition_satisfied": bool(precondition_report.get("satisfied", False)),
        "runtime_unmet_preconditions": list(precondition_report.get("unmet", []) or [])[:6],
    }


def _update_status(posterior: float, support: float, contradiction: float, row: Dict[str, Any]) -> str:
    if posterior <= 0.15 and int(row.get("contradiction_count", 0) or 0) >= max(1, int(row.get("support_count", 0) or 0)):
        return "rejected"
    if contradiction > support:
        return "weakened"
    if support > contradiction and posterior >= 0.75:
        return "leading"
    if support == contradiction == 0.0:
        return "unresolved"
    return "active"


def run_causal_inference(
    hypotheses: List[Dict[str, Any]],
    *,
    action: Dict[str, Any],
    result: Dict[str, Any],
    predicted_transition: Dict[str, Any] | None,
    actual_transition: Dict[str, Any] | None,
    reward: float,
    information_gain: float,
    obs_before: Optional[Dict[str, Any]] = None,
    obs_after: Optional[Dict[str, Any]] = None,
    world_model_summary: Optional[Dict[str, Any]] = None,
    verifier_teaching: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = normalize_hypothesis_rows(list(hypotheses or []), fallback_id_prefix="causal")
    if not normalized:
        return {
            "updated_hypotheses": [],
            "posterior_summary": {
                "leading_hypothesis_id": "",
                "leading_posterior": 0.0,
                "rejected_count": 0,
                "updated_count": 0,
                "support_events": 0,
                "contradiction_events": 0,
                "unresolved_events": 0,
                "leading_target_tokens": [],
                "object_graph_object_count": 0,
                "object_graph_relation_count": 0,
                "executable_hypothesis_count": 0,
            },
            "posterior_events": [],
            "posterior_debug": {},
        }

    runtime_graph = build_runtime_object_graph(
        world_model_summary=world_model_summary,
        obs_before=obs_before,
        obs_after=obs_after,
        action=action,
        actual_transition=actual_transition,
        result=result,
    )
    executable = build_executable_hypotheses(normalized)
    by_id = {item.hypothesis_id: item for item in executable}
    token_presence: Dict[str, int] = {}
    hypothesis_tokens: Dict[str, List[str]] = {}
    for item in executable:
        tokens = list(dict.fromkeys(list(item.observation_tokens or []) + list(item.target_tokens or [])))
        hypothesis_tokens[item.hypothesis_id] = tokens
        for token in tokens:
            token_presence[token] = int(token_presence.get(token, 0) or 0) + 1
    unique_tokens_by_id = {
        key: [token for token in tokens if int(token_presence.get(token, 0) or 0) == 1]
        for key, tokens in hypothesis_tokens.items()
    }
    updated_rows: List[Dict[str, Any]] = []
    posterior_events: List[Dict[str, Any]] = []
    evidence_rows: List[Tuple[str, float, float, float]] = []
    support_events = 0
    contradiction_events = 0
    unresolved_events = 0
    target_scores: Dict[str, float] = {}
    scene_state = _as_dict(runtime_graph.get("scene_state", {}))
    mechanism_control_summary = _as_dict(_as_dict(world_model_summary).get("mechanism_control_summary", {}))

    for row in normalized:
        hypothesis_id = str(row.get("hypothesis_id", "") or "")
        executable_h = by_id[hypothesis_id]
        evidence = _evaluate_hypothesis(
            executable_h,
            action=action,
            result=_as_dict(result),
            predicted_transition=_as_dict(predicted_transition),
            actual_transition=_as_dict(actual_transition),
            runtime_graph=runtime_graph,
            reward=float(reward or 0.0),
            information_gain=float(information_gain or 0.0),
            unique_tokens=unique_tokens_by_id.get(hypothesis_id, []),
            rival_unique_tokens=[
                token
                for other_id, tokens in unique_tokens_by_id.items()
                if other_id != hypothesis_id
                for token in tokens
            ],
            obs_before=_as_dict(obs_before),
            mechanism_control_summary=mechanism_control_summary,
            verifier_teaching=_as_dict(verifier_teaching),
        )
        support = float(evidence.get("support", 0.0) or 0.0)
        contradiction = float(evidence.get("contradiction", 0.0) or 0.0)
        unresolved = float(evidence.get("unresolved", 0.0) or 0.0)
        scene_bonus = 0.04 if scene_state.get("goal_revealed", False) and evidence.get("action_rule_used", False) else 0.0
        delta = support - contradiction + scene_bonus + unresolved * 0.02
        prior = _clamp01(row.get("posterior", row.get("confidence", 0.0)), 0.0)
        posterior = _clamp01(prior + delta, prior)
        row["posterior"] = round(posterior, 6)
        if support > contradiction and support > 0.0:
            row["support_count"] = int(row.get("support_count", 0) or 0) + 1
            support_events += 1
            event_type = "support"
        elif contradiction > support and contradiction > 0.0:
            row["contradiction_count"] = int(row.get("contradiction_count", 0) or 0) + 1
            contradiction_events += 1
            event_type = "contradiction"
        else:
            unresolved_events += 1
            event_type = "unresolved"
        row["status"] = _update_status(posterior, support, contradiction, row)
        evidence_entry = {
            "event_type": event_type,
            "delta": round(delta, 6),
            "support": round(support, 6),
            "contradiction": round(contradiction, 6),
            "reasons": list(evidence.get("reasons", []) or []),
            "matched_focus_object_ids": list(evidence.get("matched_focus_object_ids", []) or []),
        }
        if event_type == "support":
            _append_evidence_entry(row, "supporting_evidence", evidence_entry)
        elif event_type == "contradiction":
            row["falsifiers"] = _derive_falsifier_tokens(
                row,
                evidence=evidence,
                reward=float(reward or 0.0),
                actual_transition=_as_dict(actual_transition),
                scene_state=scene_state,
            )
            contradiction_entry = dict(evidence_entry)
            contradiction_entry["falsifier_tokens"] = list(row.get("falsifiers", []) or [])
            _append_evidence_entry(row, "contradicting_evidence", contradiction_entry)
        row.setdefault("metadata", {})
        if isinstance(row["metadata"], dict):
            row["metadata"]["last_runtime_graph_focus_object_ids"] = list(evidence.get("matched_focus_object_ids", []) or [])
            row["metadata"]["last_causal_reasons"] = list(evidence.get("reasons", []) or [])
            row["metadata"]["last_runtime_unmet_preconditions"] = list(evidence.get("runtime_unmet_preconditions", []) or [])
        target_weight = posterior * max(0.05, support - contradiction + 0.1)
        for token in executable_h.target_tokens[:6]:
            target_scores[token] = float(target_scores.get(token, 0.0) or 0.0) + target_weight
        posterior_events.append(
            _posterior_event(
                hypothesis_id=hypothesis_id,
                event_type=event_type,
                delta=delta,
                reason_tokens=list(evidence.get("reasons", []) or []),
                support=support,
                contradiction=contradiction,
                matched_focus_object_ids=list(evidence.get("matched_focus_object_ids", []) or []),
            )
        )
        evidence_rows.append((hypothesis_id, float(delta), support, contradiction))
        updated_rows.append(row)

    conflict_map = _infer_conflict_map(updated_rows, executable_by_id=by_id)
    evidence_rows.sort(key=lambda item: item[1], reverse=True)
    if len(evidence_rows) >= 2:
        top_id, top_score, top_support, top_contradiction = evidence_rows[0]
        runner_id, runner_score, _runner_support, _runner_contradiction = evidence_rows[1]
        gap = float(top_score) - float(runner_score)
        conflict_map.setdefault(top_id, set()).add(runner_id)
        conflict_map.setdefault(runner_id, set()).add(top_id)
        if top_support > top_contradiction and gap >= 0.08:
            top_boost = min(0.24, max(0.0, gap) * 0.82)
            rival_penalty = min(0.14, max(0.0, gap) * 0.36)
            for row in updated_rows:
                row_id = str(row.get("hypothesis_id", "") or "")
                if row_id == top_id:
                    row["posterior"] = round(_clamp01(float(row.get("posterior", 0.0) or 0.0) + top_boost), 6)
                    support_events += 1
                    posterior_events.append(
                        _posterior_event(
                            hypothesis_id=row_id,
                            event_type="support",
                            delta=top_boost,
                            reason_tokens=["causal_evidence_gap", "contrastive_support"],
                            support=top_support,
                            contradiction=top_contradiction,
                            matched_focus_object_ids=[],
                            metadata={
                                "contrastive_kind": "support",
                                "event_origin": "contrastive",
                                "paired_hypothesis_id": runner_id,
                                "evidence_gap": round(gap, 6),
                            },
                        )
                    )
                elif row_id != runner_id and gap < 0.16:
                    continue
                elif row_id != top_id:
                    row["posterior"] = round(_clamp01(float(row.get("posterior", 0.0) or 0.0) - rival_penalty), 6)
                    contradiction_events += 1
                    posterior_events.append(
                        _posterior_event(
                            hypothesis_id=row_id,
                            event_type="contradiction",
                            delta=-rival_penalty,
                            reason_tokens=["causal_evidence_gap", "contrastive_refute"],
                            support=0.0,
                            contradiction=rival_penalty,
                            matched_focus_object_ids=[],
                            metadata={
                                "contrastive_kind": "refute",
                                "event_origin": "contrastive",
                                "paired_hypothesis_id": top_id,
                                "evidence_gap": round(gap, 6),
                            },
                        )
                    )
                    conflict_map.setdefault(top_id, set()).add(row_id)
                    conflict_map.setdefault(row_id, set()).add(top_id)

    for row in updated_rows:
        hypothesis_id = str(row.get("hypothesis_id", "") or "")
        existing_conflicts = [str(item).strip() for item in list(row.get("conflicts_with", []) or []) if str(item).strip()]
        merged_conflicts = _ordered_unique(existing_conflicts + sorted(conflict_map.get(hypothesis_id, set())))
        row["conflicts_with"] = [item for item in merged_conflicts if item != hypothesis_id]

    updated_rows.sort(key=lambda item: (-float(item.get("posterior", 0.0) or 0.0), str(item.get("hypothesis_id", "") or "")))
    leading = updated_rows[0] if updated_rows else {}
    rejected_count = sum(1 for row in updated_rows if str(row.get("status", "") or "") == "rejected")
    sorted_targets = sorted(target_scores.items(), key=lambda item: (-float(item[1]), item[0]))
    posterior_summary = {
        "leading_hypothesis_id": str(leading.get("hypothesis_id", "") or ""),
        "leading_posterior": round(float(leading.get("posterior", 0.0) or 0.0), 6),
        "rejected_count": rejected_count,
        "updated_count": len(updated_rows),
        "support_events": support_events,
        "contradiction_events": contradiction_events,
        "unresolved_events": unresolved_events,
        "leading_target_tokens": [token for token, _ in sorted_targets[:4]],
        "object_graph_object_count": int(runtime_graph.get("object_count", len(runtime_graph.get("objects", []) or [])) or 0),
        "object_graph_relation_count": int(runtime_graph.get("relation_count", len(runtime_graph.get("relations", []) or [])) or 0),
        "focus_object_ids": list(runtime_graph.get("focus_object_ids", []) or [])[:6],
        "scene_phase": str(scene_state.get("phase", "") or ""),
        "scene_signal_tokens": list(scene_state.get("signal_tokens", []) or [])[:6],
        "scene_counterevidence_tokens": list(scene_state.get("counterevidence_tokens", []) or [])[:6],
        "executable_hypothesis_count": len(executable),
    }
    posterior_debug = {
        "runtime_object_graph": runtime_graph,
    }
    return {
        "updated_hypotheses": updated_rows,
        "posterior_summary": posterior_summary,
        "posterior_events": posterior_events,
        "posterior_debug": posterior_debug,
    }
