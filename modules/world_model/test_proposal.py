from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def propose_discriminating_tests(
    world_model_summary: Dict[str, Any],
    *,
    available_functions: Sequence[str] | None = None,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    summary = dict(world_model_summary or {})
    rows: List[Dict[str, Any]] = []
    seen = set()

    for raw in _as_list(summary.get("mechanism_hypotheses", [])):
        if not isinstance(raw, dict):
            continue
        mechanism_id = str(raw.get("mechanism_id", raw.get("hypothesis_id", "")) or "")
        info_gain = _clamp01(raw.get("expected_information_gain", raw.get("confidence", 0.0)), 0.0)
        for action_name in _as_list(raw.get("best_discriminating_actions", [])):
            fn_name = str(action_name or "")
            if not fn_name or fn_name in seen:
                continue
            seen.add(fn_name)
            rows.append({
                "test_id": f"mechanism_test::{mechanism_id}::{fn_name}",
                "summary": f"discriminate mechanism via {fn_name}",
                "function_name": fn_name,
                "mechanism_id": mechanism_id,
                "info_gain_estimate": round(info_gain, 4),
                "score": round(0.44 + info_gain * 0.46, 4),
                "reason": "mechanism_discrimination",
            })

    for raw in _as_list(summary.get("candidate_intervention_targets", [])):
        if not isinstance(raw, dict):
            continue
        target_id = str(raw.get("target_id", "") or "")
        features = _as_dict(raw.get("priority_features", {}))
        info_gain = _clamp01(features.get("expected_information_gain", 0.0), 0.0)
        if info_gain <= 0.0:
            continue
        for action_name in _as_list(raw.get("candidate_actions", [])):
            fn_name = str(action_name or "")
            if not fn_name or fn_name in seen:
                continue
            seen.add(fn_name)
            rows.append({
                "test_id": f"target_test::{target_id}::{fn_name}",
                "summary": f"probe target {target_id} via {fn_name}",
                "function_name": fn_name,
                "target_id": target_id,
                "info_gain_estimate": round(info_gain, 4),
                "score": round(0.40 + info_gain * 0.50, 4),
                "reason": "intervention_target_probe",
            })

    available = [str(item or "") for item in list(available_functions or []) if str(item or "")]
    for raw in _as_list(summary.get("required_probes", [])):
        probe_name = str(raw or "")
        if probe_name == "probe_before_commit":
            for fn_name in available:
                if not fn_name or fn_name in seen:
                    continue
                if any(token in fn_name.lower() for token in ("inspect", "probe", "check", "verify", "test")):
                    seen.add(fn_name)
                    rows.append({
                        "test_id": f"required_probe::{fn_name}",
                        "summary": f"required probe via {fn_name}",
                        "function_name": fn_name,
                        "info_gain_estimate": 0.72,
                        "score": 0.72,
                        "reason": "required_probe",
                    })

    rows.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -float(item.get("info_gain_estimate", 0.0) or 0.0),
            str(item.get("function_name", "") or ""),
        )
    )
    return rows[: max(0, int(limit))]
