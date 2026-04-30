from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


OUTCOME_MODEL_UPDATE_VERSION = "outcome_model_update.v0.1"


@dataclass(frozen=True)
class OutcomeModelUpdateResult:
    schema_version: str
    action_name: str
    outcome: str
    verified: bool
    world_patch: Dict[str, Any]
    self_patch: Dict[str, Any]
    learning_patch: Dict[str, Any]
    audit_event: Dict[str, Any]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_name": self.action_name,
            "outcome": self.outcome,
            "verified": self.verified,
            "world_patch_keys": sorted(self.world_patch.keys()),
            "self_patch_keys": sorted(self.self_patch.keys()),
            "learning_patch_keys": sorted(self.learning_patch.keys()),
            "audit_event": dict(self.audit_event),
        }


def build_outcome_model_update(
    *,
    action: Mapping[str, Any],
    result: Mapping[str, Any],
    evidence_entries: Optional[Iterable[Mapping[str, Any]]] = None,
    existing_state: Optional[Mapping[str, Any]] = None,
    episode: int = 0,
    tick: int = 0,
    extract_action_name: Optional[Callable[[Mapping[str, Any]], str]] = None,
) -> OutcomeModelUpdateResult:
    """Translate one executed action outcome into world/self/learning state patches."""
    action_dict = dict(action or {})
    result_dict = dict(result or {})
    state = existing_state if isinstance(existing_state, Mapping) else {}
    action_name = _safe_action_name(action_dict, extract_action_name=extract_action_name)
    outcome = _detect_outcome(result_dict)
    verified = _detect_verified(result_dict, outcome=outcome)
    evidence_refs = _compact_evidence_refs(evidence_entries or [])
    summary = _summarize_outcome(action_name=action_name, outcome=outcome, result=result_dict)

    world_summary = _mapping(state.get("world_summary"))
    self_summary = _mapping(state.get("self_summary"))
    learning_context = _mapping(state.get("learning_context"))

    risk_before = _number(world_summary.get("risk_estimate"), 0.5)
    uncertainty_before = _number(world_summary.get("uncertainty_estimate"), 0.5)
    confidence_before = _number(self_summary.get("confidence"), 0.5)
    stability_before = _number(self_summary.get("stability_estimate"), 0.5)
    adaptation_before = _number(self_summary.get("adaptation_readiness"), 0.5)

    risk_after = _nudged_estimate(risk_before, _risk_target(outcome, verified))
    uncertainty_after = _nudged_estimate(uncertainty_before, _uncertainty_target(outcome, verified))
    confidence_after = _nudged_estimate(confidence_before, _confidence_target(outcome, verified))
    stability_after = _nudged_estimate(stability_before, _stability_target(outcome, verified))
    adaptation_after = _nudged_estimate(adaptation_before, _adaptation_target(outcome, verified))

    fact = {
        "schema_version": OUTCOME_MODEL_UPDATE_VERSION,
        "kind": "action_outcome",
        "action": action_name,
        "outcome": outcome,
        "verified": verified,
        "summary": summary,
        "episode": int(episode or 0),
        "tick": int(tick or 0),
        "evidence_refs": evidence_refs,
    }
    observed_facts = _rolling_list(world_summary.get("observed_facts"), fact, limit=50)

    capability_estimate = _updated_capability_estimate(
        _mapping(self_summary.get("capability_estimate")),
        action_name=action_name,
        outcome=outcome,
        verified=verified,
        episode=episode,
        tick=tick,
    )
    error_flags = _updated_error_flags(self_summary.get("error_flags"), outcome=outcome)
    recent_failures = _updated_recent_failures(
        self_summary.get("recent_failures"),
        action_name=action_name,
        outcome=outcome,
        result=result_dict,
        episode=episode,
        tick=tick,
    )

    prediction_error = {
        "schema_version": OUTCOME_MODEL_UPDATE_VERSION,
        "action": action_name,
        "outcome": outcome,
        "verified": verified,
        "unexpected": _was_unexpected(action_dict, result_dict, outcome=outcome),
        "risk_delta": round(risk_after - risk_before, 4),
        "uncertainty_delta": round(uncertainty_after - uncertainty_before, 4),
        "confidence_delta": round(confidence_after - confidence_before, 4),
        "episode": int(episode or 0),
        "tick": int(tick or 0),
    }
    belief_update = {
        "schema_version": OUTCOME_MODEL_UPDATE_VERSION,
        "kind": "outcome_model_update",
        "action": action_name,
        "outcome": outcome,
        "verified": verified,
        "summary": summary,
        "confidence_after": confidence_after,
        "risk_after": risk_after,
        "uncertainty_after": uncertainty_after,
        "evidence_refs": evidence_refs,
        "episode": int(episode or 0),
        "tick": int(tick or 0),
    }
    belief_updates = _rolling_list(learning_context.get("belief_updates"), belief_update, limit=50)

    world_patch = {
        "world_summary.observed_facts": observed_facts,
        "world_summary.risk_estimate": risk_after,
        "world_summary.uncertainty_estimate": uncertainty_after,
        "world_summary.current_phase": _phase_from_outcome(result_dict, outcome=outcome, verified=verified),
    }
    self_patch = {
        "self_summary.capability_estimate": capability_estimate,
        "self_summary.confidence": confidence_after,
        "self_summary.error_flags": error_flags,
        "self_summary.recent_failures": recent_failures,
        "self_summary.stability_estimate": stability_after,
        "self_summary.adaptation_readiness": adaptation_after,
    }
    learning_patch = {
        "learning_context.prediction_error": prediction_error,
        "learning_context.belief_updates": belief_updates,
    }
    audit_event = {
        "event_type": "outcome_model_update",
        "schema_version": OUTCOME_MODEL_UPDATE_VERSION,
        "episode": int(episode or 0),
        "tick": int(tick or 0),
        "data": {
            "action": action_name,
            "outcome": outcome,
            "verified": verified,
            "risk_after": risk_after,
            "uncertainty_after": uncertainty_after,
            "confidence_after": confidence_after,
            "capability_attempts": capability_estimate.get(action_name, {}).get("attempts", 0),
        },
        "source_module": "core.cognition",
        "source_stage": "evidence_commit",
    }
    return OutcomeModelUpdateResult(
        schema_version=OUTCOME_MODEL_UPDATE_VERSION,
        action_name=action_name,
        outcome=outcome,
        verified=verified,
        world_patch=world_patch,
        self_patch=self_patch,
        learning_patch=learning_patch,
        audit_event=audit_event,
    )


def _safe_action_name(
    action: Mapping[str, Any],
    *,
    extract_action_name: Optional[Callable[[Mapping[str, Any]], str]] = None,
) -> str:
    if extract_action_name is not None:
        try:
            name = extract_action_name(action)
            if isinstance(name, str) and name.strip():
                return name.strip()
        except Exception:
            pass
    for key in ("function_name", "action", "name", "tool_name", "kind"):
        value = action.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    payload = action.get("payload")
    if isinstance(payload, Mapping):
        tool_args = payload.get("tool_args")
        if isinstance(tool_args, Mapping):
            for key in ("function_name", "action", "name", "tool_name"):
                value = tool_args.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return "unknown_action"


def _detect_outcome(result: Mapping[str, Any]) -> str:
    success = result.get("success")
    if isinstance(success, bool):
        return "success" if success else "failure"
    ok = result.get("ok")
    if isinstance(ok, bool):
        return "success" if ok else "failure"
    returncode = result.get("returncode")
    if isinstance(returncode, int):
        return "success" if returncode == 0 else "failure"
    state = str(result.get("state") or result.get("status") or "").strip().lower()
    if state:
        failure_tokens = ("fail", "error", "timeout", "invalid", "reject", "blocked", "crash")
        success_tokens = ("success", "passed", "verified", "complete", "accepted", "ok")
        if any(token in state for token in failure_tokens):
            return "failure"
        if any(token in state for token in success_tokens):
            return "success"
    return "unknown"


def _detect_verified(result: Mapping[str, Any], *, outcome: str) -> bool:
    if outcome != "success":
        return False
    for key in ("verified", "verification_passed", "final_tests_passed", "tests_passed"):
        value = result.get(key)
        if isinstance(value, bool):
            return value
    state = str(result.get("state") or result.get("status") or "").strip().lower()
    return "verified" in state or "completed_verified" in state


def _compact_evidence_refs(entries: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        evidence_id = entry.get("evidence_id") or entry.get("id")
        if not evidence_id:
            continue
        refs.append(
            {
                "evidence_id": str(evidence_id),
                "status": str(entry.get("status") or ""),
                "claim": str(entry.get("claim") or "")[:180],
            }
        )
        if len(refs) >= 8:
            break
    return refs


def _summarize_outcome(*, action_name: str, outcome: str, result: Mapping[str, Any]) -> str:
    message = result.get("summary") or result.get("message") or result.get("error") or result.get("state") or result.get("status")
    if isinstance(message, str) and message.strip():
        compact = " ".join(message.strip().split())
        return f"{action_name} -> {outcome}: {compact[:220]}"
    return f"{action_name} -> {outcome}"


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _nudged_estimate(current: float, target: float, *, alpha: float = 0.35) -> float:
    return round(max(0.0, min(1.0, current * (1.0 - alpha) + target * alpha)), 4)


def _risk_target(outcome: str, verified: bool) -> float:
    if outcome == "failure":
        return 0.78
    if outcome == "success" and verified:
        return 0.22
    if outcome == "success":
        return 0.34
    return 0.58


def _uncertainty_target(outcome: str, verified: bool) -> float:
    if outcome == "failure":
        return 0.76
    if outcome == "success" and verified:
        return 0.24
    if outcome == "success":
        return 0.40
    return 0.62


def _confidence_target(outcome: str, verified: bool) -> float:
    if outcome == "failure":
        return 0.30
    if outcome == "success" and verified:
        return 0.78
    if outcome == "success":
        return 0.62
    return 0.44


def _stability_target(outcome: str, verified: bool) -> float:
    if outcome == "failure":
        return 0.34
    if outcome == "success" and verified:
        return 0.74
    if outcome == "success":
        return 0.62
    return 0.48


def _adaptation_target(outcome: str, verified: bool) -> float:
    if outcome == "failure":
        return 0.80
    if outcome == "success" and verified:
        return 0.32
    if outcome == "success":
        return 0.42
    return 0.58


def _rolling_list(value: Any, item: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []
    rows.append(dict(item))
    return rows[-limit:]


def _updated_capability_estimate(
    current: Mapping[str, Any],
    *,
    action_name: str,
    outcome: str,
    verified: bool,
    episode: int,
    tick: int,
) -> Dict[str, Any]:
    merged = {str(key): dict(value) for key, value in current.items() if isinstance(value, Mapping)}
    row = dict(merged.get(action_name, {}))
    attempts = int(row.get("attempts") or 0) + 1
    successes = int(row.get("successes") or 0)
    failures = int(row.get("failures") or 0)
    unknowns = int(row.get("unknowns") or 0)
    if outcome == "success":
        successes += 1
    elif outcome == "failure":
        failures += 1
    else:
        unknowns += 1
    row.update(
        {
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "unknowns": unknowns,
            "verified_successes": int(row.get("verified_successes") or 0) + (1 if verified else 0),
            "reliability": round(successes / attempts, 4) if attempts else 0.0,
            "last_outcome": outcome,
            "last_episode": int(episode or 0),
            "last_tick": int(tick or 0),
        }
    )
    merged[action_name] = row
    return merged


def _updated_error_flags(value: Any, *, outcome: str) -> List[str]:
    flags = [str(flag) for flag in value if isinstance(flag, str) and flag.strip()] if isinstance(value, list) else []
    if outcome == "failure":
        flags.append("last_action_failed")
    elif outcome == "unknown":
        flags.append("last_action_outcome_unknown")
    else:
        flags = [flag for flag in flags if flag not in {"last_action_failed", "last_action_outcome_unknown"}]
    deduped: List[str] = []
    for flag in flags:
        if flag not in deduped:
            deduped.append(flag)
    return deduped[-20:]


def _updated_recent_failures(
    value: Any,
    *,
    action_name: str,
    outcome: str,
    result: Mapping[str, Any],
    episode: int,
    tick: int,
) -> List[Dict[str, Any]]:
    failures = [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []
    if outcome != "failure":
        return failures[-30:]
    failures.append(
        {
            "schema_version": OUTCOME_MODEL_UPDATE_VERSION,
            "action": action_name,
            "failure_type": str(result.get("error_type") or result.get("state") or result.get("status") or "action_failed"),
            "summary": _summarize_outcome(action_name=action_name, outcome=outcome, result=result),
            "episode": int(episode or 0),
            "tick": int(tick or 0),
        }
    )
    return failures[-30:]


def _was_unexpected(action: Mapping[str, Any], result: Mapping[str, Any], *, outcome: str) -> bool:
    meta = action.get("_candidate_meta")
    if not isinstance(meta, Mapping):
        meta = {}
    expected = meta.get("expected_success")
    if not isinstance(expected, bool):
        expected = result.get("expected_success")
    if isinstance(expected, bool):
        return expected != (outcome == "success")
    return outcome in {"failure", "unknown"}


def _phase_from_outcome(result: Mapping[str, Any], *, outcome: str, verified: bool) -> str:
    phase = result.get("phase") or result.get("state") or result.get("status")
    if isinstance(phase, str) and phase.strip():
        return phase.strip()[:80]
    if verified:
        return "verified_success"
    if outcome == "failure":
        return "outcome_failure"
    if outcome == "success":
        return "outcome_success"
    return "outcome_unknown"
