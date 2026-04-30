from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence

from core.runtime.failure_learning import failure_objects_to_behavior_rules


COGNITIVE_MODEL_INFLUENCE_VERSION = "conos.cognitive_model_influence/v1"


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple, set)) else []


def _string_list(value: Any) -> list[str]:
    values = _as_list(value) if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0, default: float = 0.0) -> float:
    return max(minimum, min(maximum, _float(value, default)))


def _action_name(action: Mapping[str, Any], extractor: Callable[[Dict[str, Any]], str] | None = None) -> str:
    if extractor is not None:
        try:
            extracted = str(extractor(dict(action)) or "").strip()
            if extracted:
                return extracted
        except Exception:
            pass
    for key in ("function_name", "action", "verb", "name"):
        text = str(action.get(key) or "").strip()
        if text:
            return text
    payload = _as_dict(action.get("payload"))
    tool_args = _as_dict(payload.get("tool_args"))
    for key in ("function_name", "action", "verb", "name"):
        text = str(tool_args.get(key) or "").strip()
        if text:
            return text
    text = str(payload.get("tool_name") or "").strip()
    if text:
        return text
    return str(action.get("kind") or "unknown").strip() or "unknown"


def _score_hint(action: Mapping[str, Any]) -> float:
    for key in ("final_score", "selection_score", "score", "opportunity"):
        if key in action:
            return _clamp(action.get(key), -1.0, 2.0, 0.0)
    meta = _as_dict(action.get("_candidate_meta"))
    for key in ("final_score", "selection_score", "score", "opportunity"):
        if key in meta:
            return _clamp(meta.get(key), -1.0, 2.0, 0.0)
    return 0.0


def _risk_hint(action: Mapping[str, Any]) -> float:
    if "risk" in action:
        return _clamp(action.get("risk"), 0.0, 1.0, 0.0)
    return _clamp(_as_dict(action.get("_candidate_meta")).get("risk"), 0.0, 1.0, 0.0)


def _action_value_map(value: Any) -> Dict[str, float]:
    if isinstance(value, Mapping):
        return {str(key): _clamp(raw, 0.0, 1.0, 0.0) for key, raw in value.items() if str(key).strip()}
    result: Dict[str, float] = {}
    for item in _as_list(value):
        if isinstance(item, Mapping):
            name = str(item.get("action") or item.get("name") or item.get("function_name") or item.get("verb") or "").strip()
            if name:
                result[name] = _clamp(item.get("confidence", item.get("score", item.get("weight", 0.5))), 0.0, 1.0, 0.5)
        else:
            name = str(item or "").strip()
            if name:
                result[name] = 0.5
    return result


def _known_failure_risk(self_model_state: Mapping[str, Any], action_name: str) -> tuple[float, list[str], bool]:
    risk = 0.0
    reasons: list[str] = []
    hard_block = False
    for row in _as_list(self_model_state.get("known_failure_modes")):
        if not isinstance(row, Mapping):
            continue
        names = _string_list(
            row.get("action")
            or row.get("action_name")
            or row.get("function_name")
            or row.get("actions")
            or row.get("affected_actions")
        )
        if names and action_name not in names:
            continue
        severity = _clamp(row.get("risk", row.get("severity", row.get("failure_probability", 0.45))), 0.0, 1.0, 0.45)
        risk = max(risk, severity)
        reason = str(row.get("reason") or row.get("failure_mode") or row.get("summary") or "known_failure_mode")
        reasons.append(reason)
        hard_block = hard_block or bool(row.get("block", False))
    return risk, reasons, hard_block


@dataclass
class ModelInfluenceInput:
    candidate_actions: Sequence[Dict[str, Any]]
    world_model_state: Dict[str, Any] = field(default_factory=dict)
    self_model_state: Dict[str, Any] = field(default_factory=dict)
    tick: int = 0
    episode: int = 0


@dataclass
class ModelInfluenceResult:
    candidate_actions: list[Dict[str, Any]]
    audit_records: list[Dict[str, Any]]
    blocked_count: int = 0
    influenced_count: int = 0


def apply_cognitive_model_influence(
    input_obj: ModelInfluenceInput,
    *,
    extract_action_name: Callable[[Dict[str, Any]], str] | None = None,
) -> ModelInfluenceResult:
    """Let world/self model state alter candidate action ordering and viability.

    This is intentionally domain-neutral: it operates on action names, generic
    priors, confidence, risk, resource pressure, and failure memory. Adapters can
    translate their own observations into this state, but the influence layer
    does not know about files, tests, patches, or any specific environment.
    """

    candidates = [dict(row) for row in list(input_obj.candidate_actions or []) if isinstance(row, Mapping)]
    if not candidates:
        return ModelInfluenceResult(candidate_actions=[], audit_records=[])

    world = _as_dict(input_obj.world_model_state)
    self_state = _as_dict(input_obj.self_model_state)

    preferred = _action_value_map(world.get("preferred_actions") or world.get("action_priors") or world.get("affordances"))
    blocked_world = set(_string_list(world.get("blocked_actions") or world.get("forbidden_actions")))
    risk_by_action = _action_value_map(world.get("risk_by_action"))
    required_observations = set(_string_list(world.get("required_observations") or world.get("required_information_actions")))
    uncertainty = _clamp(world.get("uncertainty", world.get("state_uncertainty", 0.0)), 0.0, 1.0, 0.0)

    capability_ceiling = _as_dict(self_state.get("capability_ceiling") or self_state.get("capability_envelope"))
    blocked_self = set(_string_list(self_state.get("blocked_actions") or capability_ceiling.get("blocked_actions")))
    preferred_self = _action_value_map(self_state.get("preferred_actions") or capability_ceiling.get("preferred_actions"))
    failure_learning_rules = _as_dict(
        self_state.get("failure_learning_behavior_rules")
        or failure_objects_to_behavior_rules(_as_list(self_state.get("failure_learning_objects")))
    )
    preferred_learning = _action_value_map(failure_learning_rules.get("preferred_actions"))
    avoided_learning = _action_value_map(failure_learning_rules.get("avoided_actions"))
    blocked_learning = set(_string_list(failure_learning_rules.get("blocked_actions")))
    budget_tight = bool(self_state.get("budget_tight", False)) or str(self_state.get("resource_tightness", "")).lower() in {"tight", "critical"}
    continuity = _clamp(self_state.get("continuity_confidence", 1.0), 0.0, 1.0, 1.0)

    adjusted: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    audit: list[Dict[str, Any]] = []

    for action in candidates:
        name = _action_name(action, extract_action_name)
        meta = _as_dict(action.get("_candidate_meta"))
        base_score = _score_hint(action)
        base_risk = _risk_hint(action)
        world_bonus = 0.0
        world_penalty = 0.0
        self_bonus = 0.0
        self_penalty = 0.0
        block_reasons: list[str] = []
        reasons: list[str] = []

        if name in preferred:
            world_bonus += 0.35 * preferred[name]
            reasons.append("world_model_preferred_action")
        if name in required_observations:
            world_bonus += 0.25 + (0.2 * uncertainty)
            reasons.append("world_model_required_observation")
        if name in risk_by_action:
            world_penalty += 0.35 * risk_by_action[name]
            reasons.append("world_model_action_risk")
        if name in blocked_world:
            block_reasons.append("world_model_blocked_action")

        if name in preferred_self:
            self_bonus += 0.25 * preferred_self[name]
            reasons.append("self_model_preferred_action")
        if name in preferred_learning:
            self_bonus += 0.24 * preferred_learning[name]
            reasons.append("failure_learning_preferred_action")
        if name in avoided_learning:
            self_penalty += 0.42 * avoided_learning[name]
            reasons.append("failure_learning_avoided_action")
        failure_risk, failure_reasons, failure_block = _known_failure_risk(self_state, name)
        if failure_risk:
            self_penalty += 0.45 * failure_risk
            reasons.extend(failure_reasons)
        if failure_block:
            block_reasons.append("self_model_known_failure_block")
        if name in blocked_self:
            block_reasons.append("self_model_blocked_action")
        if name in blocked_learning:
            block_reasons.append("failure_learning_governance_block")
        if budget_tight and bool(meta.get("high_cost", action.get("high_cost", False))):
            self_penalty += 0.2
            reasons.append("self_model_budget_tight_high_cost")
        if continuity < 0.35 and name not in {"wait", "observe", "inspect", "no_op_complete"}:
            self_penalty += 0.18
            reasons.append("self_model_low_continuity_caution")

        total_delta = world_bonus + self_bonus - world_penalty - self_penalty
        has_effect = bool(block_reasons or reasons or abs(total_delta) > 1e-9)
        if not has_effect:
            adjusted.append(action)
            continue
        adjusted_score = base_score + total_delta - (0.15 * base_risk)
        influence = {
            "schema_version": COGNITIVE_MODEL_INFLUENCE_VERSION,
            "action_name": name,
            "base_score": round(base_score, 6),
            "base_risk": round(base_risk, 6),
            "world_model_bonus": round(world_bonus, 6),
            "world_model_penalty": round(world_penalty, 6),
            "self_model_bonus": round(self_bonus, 6),
            "self_model_penalty": round(self_penalty, 6),
            "score_delta": round(total_delta, 6),
            "adjusted_score": round(adjusted_score, 6),
            "block_reasons": list(block_reasons),
            "reasons": list(dict.fromkeys(reasons)),
        }
        if failure_learning_rules.get("rule_count"):
            influence["failure_learning_rule_count"] = int(failure_learning_rules.get("rule_count", 0) or 0)
        meta["cognitive_model_influence"] = influence
        action["_candidate_meta"] = meta
        action["final_score"] = round(adjusted_score, 6)
        if base_risk or world_penalty or self_penalty:
            action["risk"] = round(min(1.0, base_risk + world_penalty + self_penalty), 6)
        audit.append(
            {
                "event": "cognitive_model_action_influence",
                "schema_version": COGNITIVE_MODEL_INFLUENCE_VERSION,
                "episode": int(input_obj.episode or 0),
                "tick": int(input_obj.tick or 0),
                **influence,
            }
        )
        if block_reasons and name != "wait":
            blocked.append(action)
        else:
            adjusted.append(action)

    usable = adjusted
    if not usable:
        wait_actions = [row for row in blocked if _action_name(row, extract_action_name) == "wait"]
        usable = wait_actions or candidates
        audit.append(
            {
                "event": "cognitive_model_all_actions_blocked_failsafe",
                "schema_version": COGNITIVE_MODEL_INFLUENCE_VERSION,
                "episode": int(input_obj.episode or 0),
                "tick": int(input_obj.tick or 0),
                "blocked_count": len(blocked),
                "returned_count": len(usable),
            }
        )

    usable.sort(
        key=lambda row: _as_dict(row.get("_candidate_meta")).get("cognitive_model_influence", {}).get("adjusted_score", _score_hint(row)),
        reverse=True,
    )
    return ModelInfluenceResult(
        candidate_actions=usable,
        audit_records=audit,
        blocked_count=len(blocked),
        influenced_count=len(audit),
    )
