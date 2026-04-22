from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, (tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    out: List[str] = []
    seen = set()
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


@dataclass
class CapabilityEnvelope:
    reliable_actions: List[str] = field(default_factory=list)
    fragile_actions: List[str] = field(default_factory=list)
    known_blind_spots: List[str] = field(default_factory=list)
    transferable_domains: List[str] = field(default_factory=list)
    teacher_dependence_estimate: float = 0.5
    transfer_readiness: float = 0.0
    budget_multiplier: float = 1.0
    exploration_ratio_target: float = 0.5
    strategy_mode_hint: str = "balanced"
    branch_budget_delta: int = 0
    verification_budget_delta: int = 0
    search_depth_bias: int = 0
    fallback_bias: str = "balanced"
    teacher_off_escalation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reliable_actions": list(self.reliable_actions),
            "fragile_actions": list(self.fragile_actions),
            "known_blind_spots": list(self.known_blind_spots),
            "transferable_domains": list(self.transferable_domains),
            "teacher_dependence_estimate": _clamp01(self.teacher_dependence_estimate, default=0.5),
            "transfer_readiness": _clamp01(self.transfer_readiness, default=0.0),
            "budget_multiplier": _clamp01(self.budget_multiplier, default=1.0),
            "exploration_ratio_target": _clamp01(self.exploration_ratio_target, default=0.5),
            "strategy_mode_hint": str(self.strategy_mode_hint or "balanced"),
            "branch_budget_delta": int(self.branch_budget_delta),
            "verification_budget_delta": int(self.verification_budget_delta),
            "search_depth_bias": int(self.search_depth_bias),
            "fallback_bias": str(self.fallback_bias or "balanced"),
            "teacher_off_escalation": bool(self.teacher_off_escalation),
        }


def build_capability_envelope(
    *,
    capability_profile,
    reliability_tracker,
    continuity_snapshot: Dict[str, Any],
    continuity_confidence: float,
    teacher_present: bool | None = None,
) -> CapabilityEnvelope:
    reliability_by_action = {}
    if hasattr(reliability_tracker, "get_reliability_by_action_type"):
        reliability_by_action = dict(reliability_tracker.get_reliability_by_action_type())
    reliable_actions = sorted(
        [name for name, score in reliability_by_action.items() if _clamp01(score, default=0.0) >= 0.65]
    )[:8]

    failure_profile = []
    if hasattr(reliability_tracker, "get_recent_failure_profile"):
        failure_profile = list(reliability_tracker.get_recent_failure_profile(limit=8))
    fragile_actions = _normalize_list([row.get("action_type", "") for row in failure_profile if isinstance(row, dict)])

    contextual = getattr(capability_profile, "contextual_capabilities", {}) or {}
    transferable_domains: List[str] = []
    blind_spots: List[str] = []
    for fn_name, contexts in contextual.items():
        if not isinstance(contexts, dict):
            continue
        successful_domains = set()
        weak_domains = set()
        for ctx_key, stats in contexts.items():
            if not isinstance(stats, dict):
                continue
            domain = str(stats.get("task_family", ctx_key) or ctx_key).strip()
            total = float(stats.get("total_calls", 0) or 0)
            success_rate = (float(stats.get("success_count", 0) or 0) / total) if total > 0 else 0.0
            if total >= 2 and success_rate >= 0.55:
                successful_domains.add(domain)
            elif total >= 1 and success_rate < 0.4:
                weak_domains.add(domain)
        if len(successful_domains) >= 2:
            transferable_domains.extend(sorted(successful_domains))
        if weak_domains:
            blind_spots.append(f"{fn_name}:{sorted(weak_domains)[0]}")

    if hasattr(reliability_tracker, "estimate_teacher_dependence"):
        teacher_dependence = _clamp01(reliability_tracker.estimate_teacher_dependence(), default=0.5)
    else:
        teacher_dependence = 0.5
    if teacher_present is True:
        teacher_dependence = max(teacher_dependence, 0.55)

    if hasattr(reliability_tracker, "estimate_transfer_readiness"):
        transfer_readiness = _clamp01(reliability_tracker.estimate_transfer_readiness(), default=0.0)
    else:
        transfer_readiness = 0.0
    if not transfer_readiness and transferable_domains:
        transfer_readiness = _clamp01(0.35 + 0.15 * min(len(set(transferable_domains)), 3), default=0.35)

    teacher_off_escalation = bool(teacher_present is False and teacher_dependence >= 0.6)
    if teacher_off_escalation:
        strategy_mode = "recover"
    elif continuity_confidence < 0.45:
        strategy_mode = "verify"
    elif transfer_readiness >= 0.65 and blind_spots:
        strategy_mode = "explore"
    elif reliable_actions and continuity_confidence >= 0.7:
        strategy_mode = "exploit"
    else:
        strategy_mode = "balanced"

    branch_budget_delta = 0
    verification_budget_delta = 0
    search_depth_bias = 0
    fallback_bias = "balanced"
    exploration_ratio_target = 0.5

    if strategy_mode == "recover":
        branch_budget_delta = 1
        verification_budget_delta = 2
        search_depth_bias = -1
        fallback_bias = "self_repair"
        exploration_ratio_target = 0.25
    elif strategy_mode == "verify":
        verification_budget_delta = 1
        fallback_bias = "verify"
        exploration_ratio_target = 0.4
    elif strategy_mode == "explore":
        branch_budget_delta = 1
        search_depth_bias = 1
        fallback_bias = "probe"
        exploration_ratio_target = 0.7
    elif strategy_mode == "exploit":
        fallback_bias = "stabilize"
        exploration_ratio_target = 0.2

    budget_multiplier = _clamp01(
        0.55
        + (_clamp01(continuity_confidence, default=0.5) * 0.25)
        + (_clamp01(1.0 - teacher_dependence, default=0.5) * 0.20)
        + (_clamp01(transfer_readiness, default=0.0) * 0.10)
        - (min(len(blind_spots), 3) * 0.05),
        default=0.75,
    )
    if teacher_off_escalation:
        budget_multiplier = _clamp01(budget_multiplier * 0.82, default=0.6)

    return CapabilityEnvelope(
        reliable_actions=reliable_actions,
        fragile_actions=fragile_actions,
        known_blind_spots=blind_spots[:8],
        transferable_domains=_normalize_list(transferable_domains)[:8],
        teacher_dependence_estimate=teacher_dependence,
        transfer_readiness=transfer_readiness,
        budget_multiplier=budget_multiplier,
        exploration_ratio_target=exploration_ratio_target,
        strategy_mode_hint=strategy_mode,
        branch_budget_delta=branch_budget_delta,
        verification_budget_delta=verification_budget_delta,
        search_depth_bias=search_depth_bias,
        fallback_bias=fallback_bias,
        teacher_off_escalation=teacher_off_escalation,
    )
