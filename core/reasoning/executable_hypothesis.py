from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Mapping, Optional

from core.reasoning.hypothesis_schema import normalize_hypothesis_row

_TOKEN_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "controls",
    "control",
    "gate",
    "switch",
    "device",
    "lever",
    "orb",
    "door",
    "cluster",
    "panel",
    "family",
    "target",
    "probe",
    "diagnostic",
    "channel",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _tokenize(*values: Any) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for token in _tokenize(*list(value)):
                if token not in seen:
                    seen.add(token)
                    ordered.append(token)
            continue
        text = str(value or "").strip().lower()
        if not text:
            continue
        canonical = text.replace(" ", "_")
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
        for raw in re.split(r"[^a-z0-9]+", text.replace("::", "_")):
            token = str(raw or "").strip().lower()
            if token and token not in _TOKEN_STOPWORDS and token not in seen:
                seen.add(token)
                ordered.append(token)
    return ordered


@dataclass
class TransitionRule:
    action_name: str
    reward_sign: str = ""
    valid_state_change: Optional[bool] = None
    phase_shift: str = ""
    observation_tokens: List[str] = field(default_factory=list)
    information_gain: float = 0.0
    risk_type: str = ""
    target_tokens: List[str] = field(default_factory=list)


@dataclass
class ExecutableHypothesis:
    hypothesis_id: str
    family: str
    summary: str
    prior: float
    observation_tokens: List[str] = field(default_factory=list)
    action_rules: Dict[str, TransitionRule] = field(default_factory=dict)
    action_rules_by_signature: Dict[str, TransitionRule] = field(default_factory=dict)
    precondition_tokens: List[str] = field(default_factory=list)
    counterevidence_tokens: List[str] = field(default_factory=list)
    recovery_tokens: List[str] = field(default_factory=list)
    target_tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_executable_hypothesis(row: Mapping[str, Any]) -> ExecutableHypothesis:
    normalized = normalize_hypothesis_row(dict(row or {}), fallback_id_prefix="exec")
    summary = str(normalized.get("summary", "") or "")
    metadata = _as_dict(normalized.get("metadata", {}))
    predictions = _as_dict(normalized.get("predictions", {}))
    action_effects = _as_dict(predictions.get("predicted_action_effects", normalized.get("predicted_action_effects", {})))
    action_effects_by_signature = _as_dict(
        predictions.get(
            "predicted_action_effects_by_signature",
            normalized.get("predicted_action_effects_by_signature", {}),
        )
    )
    observation_tokens = _string_list(predictions.get("predicted_observation_tokens", normalized.get("predicted_observation_tokens", [])))
    if not observation_tokens:
        observation_tokens = _tokenize(
            normalized.get("family", ""),
            normalized.get("predicted_phase_shift", ""),
            summary,
            metadata.get("target_binding_tokens", []),
            metadata.get("anchor_ref", ""),
        )[:8]
    action_rules: Dict[str, TransitionRule] = {}
    action_rules_by_signature: Dict[str, TransitionRule] = {}
    target_tokens: List[str] = []

    def _build_rule(function_name: str, payload: Mapping[str, Any]) -> TransitionRule:
        tokens = _tokenize(
            payload.get("target_kind", ""),
            payload.get("target_family", ""),
            payload.get("anchor_ref", ""),
            payload.get("predicted_observation_tokens", []),
            function_name,
        )
        target_tokens.extend(tokens)
        return TransitionRule(
            action_name=str(function_name),
            reward_sign=str(payload.get("reward_sign", payload.get("predicted_reward_sign", "")) or "").strip().lower(),
            valid_state_change=(
                None
                if "valid_state_change" not in payload
                else bool(payload.get("valid_state_change"))
            ),
            phase_shift=str(payload.get("predicted_phase_shift", payload.get("phase_shift", "")) or "").strip().lower(),
            observation_tokens=_string_list(payload.get("predicted_observation_tokens", [])),
            information_gain=float(payload.get("predicted_information_gain", 0.0) or 0.0),
            risk_type=str(payload.get("risk_type", "") or "").strip().lower(),
            target_tokens=tokens,
        )

    for function_name, payload in action_effects.items():
        if not isinstance(payload, dict):
            continue
        action_rules[str(function_name)] = _build_rule(str(function_name), payload)
    for signature_key, payload in action_effects_by_signature.items():
        if not isinstance(payload, dict):
            continue
        function_name = str(payload.get("function_name", "") or "").strip()
        if not function_name:
            continue
        action_rules_by_signature[str(signature_key or "").strip()] = _build_rule(function_name, payload)

    counterevidence_tokens = _tokenize(
        normalized.get("falsifiers", []),
        [item.get("reason", "") for item in _as_list(normalized.get("contradicting_evidence", [])) if isinstance(item, dict)],
        metadata.get("counterevidence_tokens", []),
    )[:8]
    precondition_tokens = _tokenize(
        metadata.get("preconditions", []),
        metadata.get("required_state", []),
        metadata.get("prerequisite_tokens", []),
    )[:8]
    recovery_tokens = _tokenize(
        metadata.get("recovery_actions", []),
        metadata.get("recovery_tokens", []),
        metadata.get("repair_tokens", []),
    )[:8]
    return ExecutableHypothesis(
        hypothesis_id=str(normalized.get("hypothesis_id", "") or ""),
        family=str(normalized.get("family", normalized.get("hypothesis_type", "generic")) or "generic"),
        summary=summary,
        prior=float(normalized.get("posterior", normalized.get("confidence", 0.0)) or 0.0),
        observation_tokens=observation_tokens,
        action_rules=action_rules,
        action_rules_by_signature=action_rules_by_signature,
        precondition_tokens=precondition_tokens,
        counterevidence_tokens=counterevidence_tokens,
        recovery_tokens=recovery_tokens,
        target_tokens=list(
            dict.fromkeys(
                target_tokens
                + _tokenize(metadata.get("target_binding_tokens", []), metadata.get("anchor_ref", ""))
                + _tokenize(summary, normalized.get("family", ""))
            )
        ),
        metadata=metadata,
    )


def build_executable_hypotheses(rows: List[Dict[str, Any]]) -> List[ExecutableHypothesis]:
    return [build_executable_hypothesis(row) for row in list(rows or []) if isinstance(row, dict)]
