from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Dict, List

from core.orchestration.action_utils import action_semantic_signature_key, extract_action_function_name


_OBSERVATION_SIGNATURE_STOPWORDS = {
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
    "mechanism",
}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    rows: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            rows.append(text)
    return rows


def _signature_token_parts(value: Any) -> List[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    parts = re.split(r"[^a-z0-9]+", text.replace("::", "_"))
    ordered: List[str] = []
    seen = set()
    for item in parts:
        token = str(item or "").strip().lower()
        if not token or token in _OBSERVATION_SIGNATURE_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _fallback_hypothesis_id(row: Dict[str, Any], *, fallback_id_prefix: str, index: int) -> str:
    summary = str(row.get("summary", "") or row.get("description", "") or "").strip().lower()
    family = str(row.get("family", "") or row.get("type", "") or "").strip().lower()
    slug = family or summary.replace(" ", "_")[:24]
    slug = slug or "hypothesis"
    return f"{fallback_id_prefix}_{index}_{slug}"


def _prediction_function_name(effect_key: Any, payload: Dict[str, Any]) -> str:
    explicit = str(payload.get("function_name", "") or "").strip()
    if explicit:
        return explicit
    text = str(effect_key or "").strip()
    if not text:
        return ""
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        decoded = None
    if isinstance(decoded, dict):
        decoded_name = str(decoded.get("function_name", "") or "").strip()
        if decoded_name:
            return decoded_name
    if text.startswith("{") or text.startswith("["):
        return ""
    return text


def _prediction_signature_key(function_name: str, payload: Dict[str, Any]) -> str:
    synthetic_action: Dict[str, Any] = {
        "kind": "call_tool",
        "payload": {
            "tool_args": {
                "function_name": str(function_name or "").strip(),
                "kwargs": dict(payload.get("kwargs", {})) if isinstance(payload.get("kwargs", {}), dict) else {},
            }
        },
    }
    for field_name in ("x", "y", "target_family", "relation_type", "anchor_ref"):
        value = payload.get(field_name, None)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        synthetic_action[field_name] = value
    return action_semantic_signature_key(synthetic_action)


@dataclass
class HypothesisPrediction:
    predicted_observation_tokens: List[str] = field(default_factory=list)
    predicted_action_effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    predicted_action_effects_by_signature: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    predicted_phase_shift: str = ""
    predicted_information_gain: float = 0.0


@dataclass
class HypothesisState:
    hypothesis_id: str = ""
    hypothesis_type: str = "generic"
    summary: str = ""
    confidence: float = 0.0
    posterior: float = 0.0
    support_count: int = 0
    contradiction_count: int = 0
    status: str = "active"
    scope: str = "local"
    source: str = "workspace"
    predictions: HypothesisPrediction = field(default_factory=HypothesisPrediction)
    falsifiers: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_hypothesis_row(
    row: Dict[str, Any],
    *,
    fallback_id_prefix: str = "h",
    index: int = 0,
) -> Dict[str, Any]:
    raw = dict(row or {}) if isinstance(row, dict) else {}
    hypothesis_id = str(
        raw.get("hypothesis_id", "")
        or raw.get("object_id", "")
        or raw.get("id", "")
        or ""
    ).strip()
    if not hypothesis_id:
        hypothesis_id = _fallback_hypothesis_id(raw, fallback_id_prefix=fallback_id_prefix, index=index)

    supporting_evidence = _dict_list(raw.get("supporting_evidence", []))
    contradicting_evidence = _dict_list(raw.get("contradicting_evidence", []))
    raw_predictions = _dict(raw.get("predictions", {}))
    predicted_action_effects = _dict(raw_predictions.get("predicted_action_effects", raw.get("predicted_action_effects", {})))
    predicted_action_effects_by_signature = _dict(
        raw_predictions.get(
            "predicted_action_effects_by_signature",
            raw.get("predicted_action_effects_by_signature", {}),
        )
    )
    normalized_effects: Dict[str, Dict[str, Any]] = {}
    normalized_effects_by_signature: Dict[str, Dict[str, Any]] = {}

    def _record_prediction(effect_key: Any, payload: Dict[str, Any], *, prefer_signature_key: bool = False) -> None:
        if not isinstance(payload, dict):
            return
        fn_name = _prediction_function_name(effect_key, payload)
        if not fn_name:
            return
        normalized_payload = dict(payload)
        normalized_payload["function_name"] = fn_name
        reward_sign = str(
            normalized_payload.get("reward_sign", normalized_payload.get("predicted_reward_sign", ""))
            or ""
        ).strip().lower()
        if reward_sign:
            normalized_payload["reward_sign"] = reward_sign
        risk_type = str(normalized_payload.get("risk_type", "") or "").strip().lower()
        if risk_type:
            normalized_payload["risk_type"] = risk_type
        if "valid_state_change" in normalized_payload:
            normalized_payload["valid_state_change"] = bool(normalized_payload.get("valid_state_change"))
        if "predicted_information_gain" in normalized_payload:
            normalized_payload["predicted_information_gain"] = _clamp01(
                normalized_payload.get("predicted_information_gain", 0.0)
            )
        signature_key = str(
            normalized_payload.get("semantic_signature", normalized_payload.get("action_signature", ""))
            or ""
        ).strip()
        if prefer_signature_key and not signature_key:
            signature_key = str(effect_key or "").strip()
        if not signature_key:
            signature_key = _prediction_signature_key(fn_name, normalized_payload)
        normalized_payload["semantic_signature"] = signature_key
        normalized_effects.setdefault(fn_name, dict(normalized_payload))
        normalized_effects_by_signature[signature_key] = dict(normalized_payload)

    for effect_key, payload in predicted_action_effects.items():
        _record_prediction(effect_key, payload)
    for effect_key, payload in predicted_action_effects_by_signature.items():
        _record_prediction(effect_key, payload, prefer_signature_key=True)

    observation_tokens = _string_list(
        raw_predictions.get(
            "predicted_observation_tokens",
            raw.get("predicted_observation_tokens", []),
        )
    )
    if not observation_tokens:
        fallback_tokens = [
            str(raw.get("family", "") or raw.get("type", "") or "").strip(),
            str(raw.get("expected_transition", raw.get("transition", "")) or "").strip(),
            str(raw.get("summary", raw.get("description", "")) or "").strip(),
        ]
        observation_tokens = [
            token.lower().replace(" ", "_")
            for token in fallback_tokens
            if token
        ][:4]

    prediction = HypothesisPrediction(
        predicted_observation_tokens=observation_tokens,
        predicted_action_effects=normalized_effects,
        predicted_action_effects_by_signature=normalized_effects_by_signature,
        predicted_phase_shift=str(
            raw_predictions.get(
                "predicted_phase_shift",
                raw.get("predicted_phase_shift", raw.get("expected_transition", raw.get("transition", ""))),
            )
            or ""
        ).strip(),
        predicted_information_gain=_clamp01(
            raw_predictions.get(
                "predicted_information_gain",
                raw.get("predicted_information_gain", raw.get("expected_information_gain", 0.0)),
            )
        ),
    )

    confidence = _clamp01(raw.get("confidence", raw.get("posterior", 0.0)))
    posterior = _clamp01(raw.get("posterior", confidence), default=confidence)
    status = str(raw.get("status", "") or "").strip().lower() or "active"
    if status not in {"active", "weakened", "rejected", "leading", "unresolved"}:
        status = "active"

    hypothesis = HypothesisState(
        hypothesis_id=hypothesis_id,
        hypothesis_type=str(raw.get("hypothesis_type", raw.get("family", raw.get("type", "generic"))) or "generic").strip() or "generic",
        summary=str(raw.get("summary", raw.get("description", "")) or "").strip(),
        confidence=confidence,
        posterior=posterior,
        support_count=max(
            int(raw.get("support_count", 0) or 0),
            len(supporting_evidence),
        ),
        contradiction_count=max(
            int(raw.get("contradiction_count", 0) or 0),
            len(contradicting_evidence),
        ),
        status=status,
        scope=str(raw.get("scope", "local") or "local").strip() or "local",
        source=str(raw.get("source", "workspace") or "workspace").strip() or "workspace",
        predictions=prediction,
        falsifiers=_string_list(raw.get("falsifiers", [])),
        conflicts_with=_string_list(raw.get("conflicts_with", [])),
        supporting_evidence=supporting_evidence,
        contradicting_evidence=contradicting_evidence,
        tags=_string_list(raw.get("tags", [])),
        metadata=_dict(raw.get("metadata", {})),
    )
    normalized = asdict(hypothesis)
    normalized["object_id"] = hypothesis_id
    normalized["family"] = normalized.get("hypothesis_type", "generic")
    normalized["predictions"] = asdict(prediction)
    normalized["predicted_action_effects"] = dict(normalized["predictions"].get("predicted_action_effects", {}))
    normalized["predicted_action_effects_by_signature"] = dict(
        normalized["predictions"].get("predicted_action_effects_by_signature", {})
    )
    normalized["predicted_observation_tokens"] = list(normalized["predictions"].get("predicted_observation_tokens", []))
    normalized["predicted_phase_shift"] = str(normalized["predictions"].get("predicted_phase_shift", "") or "")
    normalized["predicted_information_gain"] = _clamp01(normalized["predictions"].get("predicted_information_gain", 0.0))
    return normalized


def normalize_hypothesis_rows(rows: List[Dict[str, Any]], *, fallback_id_prefix: str = "h") -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, row in enumerate(rows or []):
        if not isinstance(row, dict):
            continue
        normalized.append(
            normalize_hypothesis_row(
                row,
                fallback_id_prefix=fallback_id_prefix,
                index=index,
            )
        )
    return normalized


def hypothesis_action_prediction(row: Dict[str, Any], action_or_function: Any) -> Dict[str, Any]:
    normalized = normalize_hypothesis_row(row, fallback_id_prefix="action_pred")
    predictions = _dict(normalized.get("predictions", {}))
    action_effects = _dict(predictions.get("predicted_action_effects", normalized.get("predicted_action_effects", {})))
    action_effects_by_signature = _dict(
        predictions.get(
            "predicted_action_effects_by_signature",
            normalized.get("predicted_action_effects_by_signature", {}),
        )
    )
    if isinstance(action_or_function, dict):
        signature_key = action_semantic_signature_key(action_or_function)
        if signature_key and isinstance(action_effects_by_signature.get(signature_key), dict):
            return dict(action_effects_by_signature.get(signature_key, {}))
        fn_name = str(extract_action_function_name(action_or_function, default="") or "").strip()
    else:
        fn_name = str(action_or_function or "").strip()
    if fn_name and isinstance(action_effects.get(fn_name), dict):
        return dict(action_effects.get(fn_name, {}))
    if fn_name:
        for payload in action_effects_by_signature.values():
            if not isinstance(payload, dict):
                continue
            if str(payload.get("function_name", "") or "").strip() == fn_name:
                return dict(payload)
    return {}


def hypothesis_observation_signature(row: Dict[str, Any]) -> List[str]:
    normalized = normalize_hypothesis_row(row, fallback_id_prefix="obs_sig")
    predictions = _dict(normalized.get("predictions", {}))
    tokens = _string_list(predictions.get("predicted_observation_tokens", normalized.get("predicted_observation_tokens", [])))
    expanded_tokens: List[str] = []
    seen = set()
    for token in tokens:
        for part in _signature_token_parts(token):
            if part in seen:
                continue
            seen.add(part)
            expanded_tokens.append(part)
    if expanded_tokens:
        return expanded_tokens[:6]
    summary = str(normalized.get("summary", "") or "").strip().lower()
    family = str(normalized.get("family", "") or "").strip().lower()
    expected_transition = str(normalized.get("predicted_phase_shift", normalized.get("expected_transition", "")) or "").strip().lower()
    for value in (family, expected_transition, summary):
        for part in _signature_token_parts(value):
            if part in seen:
                continue
            seen.add(part)
            expanded_tokens.append(part)
    return expanded_tokens[:6]
