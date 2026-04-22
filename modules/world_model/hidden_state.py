from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, List, Optional, Tuple

from modules.world_model.canonical_state import (
    summarize_observation_world,
    summarize_world_transition,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _normalize_phase(raw_phase: Any) -> str:
    phase = str(raw_phase or "").strip().lower()
    if not phase:
        return ""
    if any(token in phase for token in ("commit", "sealed", "stable", "solved", "complete", "done")):
        return "committed"
    if any(token in phase for token in ("stabil", "align", "ready", "warm", "active", "seal_visible")):
        return "stabilizing"
    if any(token in phase for token in ("disrupt", "rupture", "fail", "error", "drift", "shift", "broken")):
        return "disrupted"
    if any(token in phase for token in ("explor", "search", "probe", "scan")):
        return "exploring"
    return phase


@dataclass
class HiddenStateSnapshot:
    episode: int = 0
    tick: int = 0
    phase: str = "exploring"
    phase_confidence: float = 0.35
    hidden_state_depth: int = 0
    drift_score: float = 0.0
    stability_score: float = 0.0
    uncertainty_score: float = 0.65
    explicit_observation_phase: str = ""
    last_function_name: str = ""
    recent_phase_path: List[str] = field(default_factory=list)
    focus_functions: List[str] = field(default_factory=list)
    latent_signature: str = "exploring::none::0"
    dominant_branch_id: str = ""
    latent_branches: List[Dict[str, Any]] = field(default_factory=list)
    expected_next_phase: str = ""
    expected_next_phase_confidence: float = 0.0
    transition_entropy: float = 1.0
    transition_memory: Dict[str, Any] = field(default_factory=dict)
    observed_modality: str = ""
    world_state_signature: str = ""
    world_entity_count: int = 0
    world_relation_count: int = 0
    last_event_signature: str = ""
    novelty_score: float = 0.0
    rollout_uncertainty: float = 0.5
    transition_prior_signature: str = ""
    update_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": int(self.episode),
            "tick": int(self.tick),
            "phase": self.phase,
            "phase_confidence": float(self.phase_confidence),
            "hidden_state_depth": int(self.hidden_state_depth),
            "drift_score": float(self.drift_score),
            "stability_score": float(self.stability_score),
            "uncertainty_score": float(self.uncertainty_score),
            "explicit_observation_phase": self.explicit_observation_phase,
            "last_function_name": self.last_function_name,
            "recent_phase_path": list(self.recent_phase_path),
            "focus_functions": list(self.focus_functions),
            "latent_signature": self.latent_signature,
            "dominant_branch_id": self.dominant_branch_id,
            "latent_branches": [
                dict(item) for item in self.latent_branches
                if isinstance(item, dict)
            ],
            "expected_next_phase": self.expected_next_phase,
            "expected_next_phase_confidence": float(self.expected_next_phase_confidence),
            "transition_entropy": float(self.transition_entropy),
            "observed_modality": self.observed_modality,
            "world_state_signature": self.world_state_signature,
            "world_entity_count": int(self.world_entity_count),
            "world_relation_count": int(self.world_relation_count),
            "last_event_signature": self.last_event_signature,
            "novelty_score": float(self.novelty_score),
            "rollout_uncertainty": float(self.rollout_uncertainty),
            "transition_prior_signature": self.transition_prior_signature,
            "transition_memory": {
                "current_phase": str(self.transition_memory.get("current_phase", "") or ""),
                "expected_next_phase": str(self.transition_memory.get("expected_next_phase", "") or ""),
                "expected_next_phase_confidence": float(self.transition_memory.get("expected_next_phase_confidence", 0.0) or 0.0),
                "phase_transition_entropy": float(self.transition_memory.get("phase_transition_entropy", 1.0) or 1.0),
                "dominant_transitions": [
                    dict(item) for item in (self.transition_memory.get("dominant_transitions", []) or [])
                    if isinstance(item, dict)
                ],
                "stabilizing_functions": [
                    dict(item) for item in (self.transition_memory.get("stabilizing_functions", []) or [])
                    if isinstance(item, dict)
                ],
                "risky_functions": [
                    dict(item) for item in (self.transition_memory.get("risky_functions", []) or [])
                    if isinstance(item, dict)
                ],
                "phase_function_scores": {
                    str(name): dict(payload)
                    for name, payload in (self.transition_memory.get("phase_function_scores", {}) or {}).items()
                    if isinstance(payload, dict)
                },
                "dominant_branch_id": str(
                    self.transition_memory.get("dominant_branch_id", self.dominant_branch_id) or ""
                ),
                "latent_branches": [
                    dict(item) for item in (self.transition_memory.get("latent_branches", self.latent_branches) or [])
                    if isinstance(item, dict)
                ],
                "observed_modality": str(self.transition_memory.get("observed_modality", self.observed_modality) or ""),
                "world_state_signature": str(self.transition_memory.get("world_state_signature", self.world_state_signature) or ""),
                "world_entity_count": int(self.transition_memory.get("world_entity_count", self.world_entity_count) or 0),
                "world_relation_count": int(self.transition_memory.get("world_relation_count", self.world_relation_count) or 0),
                "last_event_signature": str(self.transition_memory.get("last_event_signature", self.last_event_signature) or ""),
                "novelty_score": float(self.transition_memory.get("novelty_score", self.novelty_score) or 0.0),
                "rollout_uncertainty": float(self.transition_memory.get("rollout_uncertainty", self.rollout_uncertainty) or 0.0),
                "transition_prior_signature": str(self.transition_memory.get("transition_prior_signature", self.transition_prior_signature) or ""),
            },
            "update_count": int(self.update_count),
        }


class HiddenStateTracker:
    """Tracks a compact latent state estimate across ticks for prediction/world-model coupling."""

    def __init__(self, *, max_history: int = 16) -> None:
        self._max_history = max_history
        self._phase_history: Deque[str] = deque(maxlen=max_history)
        self._function_history: Deque[str] = deque(maxlen=max_history)
        self._snapshot = HiddenStateSnapshot()
        self._phase_transition_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self._phase_transition_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._phase_function_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._branch_feedback_stats: Dict[str, Dict[str, float]] = {}

    def reset(self, episode: int) -> None:
        self._phase_history.clear()
        self._function_history.clear()
        self._snapshot = HiddenStateSnapshot(episode=int(episode), tick=0)
        self._phase_transition_counts.clear()
        self._phase_transition_stats.clear()
        self._phase_function_stats.clear()
        self._branch_feedback_stats.clear()

    def summary(self) -> Dict[str, Any]:
        return self._feedback_adjusted_summary(self._snapshot.to_dict())

    def update(
        self,
        *,
        episode: int,
        tick: int,
        obs_before: Dict[str, Any],
        result: Dict[str, Any],
        reward: float,
        function_name: str,
        world_model_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        previous = self._snapshot
        wm_summary = dict(world_model_summary or {})
        canonical_before = summarize_observation_world(obs_before)
        canonical_after = summarize_observation_world(result if isinstance(result, dict) else {})
        canonical_transition = summarize_world_transition(obs_before, result if isinstance(result, dict) else {})
        observed_modality = str(canonical_after.get("observed_modality", "") or canonical_before.get("observed_modality", "") or "")
        world_state_signature = str(canonical_after.get("world_state_signature", "") or canonical_before.get("world_state_signature", "") or "")
        after_scene = canonical_after.get("world_scene_summary", {}) if isinstance(canonical_after.get("world_scene_summary", {}), dict) else {}
        world_entity_count = int(after_scene.get("entity_count", 0) or 0)
        world_relation_count = int(after_scene.get("relation_count", 0) or 0)
        last_event_signature = str(canonical_transition.get("event_signature", "") or "")
        novelty_score = _clamp(float(canonical_transition.get("novelty_score", 0.0) or 0.0))
        explicit_phase = self._extract_explicit_phase(result) or self._extract_explicit_phase(obs_before)
        normalized_explicit = _normalize_phase(explicit_phase)
        wm_phase = _normalize_phase(wm_summary.get("predicted_phase", ""))
        wm_confidence = _clamp(float(wm_summary.get("transition_confidence", 0.0) or 0.0))
        wm_shift_risk = _clamp(float(wm_summary.get("shift_risk", 0.0) or 0.0))
        success = bool(result.get("success", True)) and float(reward or 0.0) >= 0.0

        next_phase = self._infer_phase(
            explicit_phase=normalized_explicit,
            wm_phase=wm_phase or previous.phase,
            reward=float(reward or 0.0),
            success=success,
            previous_phase=previous.phase,
            shift_risk=wm_shift_risk,
        )
        phase_changed = next_phase != previous.phase
        hidden_state_depth = self._next_depth(previous.hidden_state_depth, next_phase, previous.phase, phase_changed)
        depth_gain = max(0.0, float(hidden_state_depth - int(previous.hidden_state_depth or 0)))

        drift_score = _clamp(
            (previous.drift_score * 0.55)
            + (0.26 if phase_changed else 0.0)
            + (0.24 if not success else 0.0)
            + (0.18 if float(reward or 0.0) < 0.0 else 0.0)
            + (wm_shift_risk * 0.22)
            + (novelty_score * 0.08)
        )
        stability_score = _clamp(
            (previous.stability_score * 0.58)
            + (0.22 if success else 0.0)
            + (0.18 if float(reward or 0.0) > 0.0 else 0.0)
            + (0.16 if not phase_changed and next_phase in ("stabilizing", "committed") else 0.0)
            - (drift_score * 0.18)
        )
        phase_confidence = _clamp(
            (wm_confidence * 0.42)
            + (0.24 if normalized_explicit else 0.0)
            + (stability_score * 0.26)
            + (0.12 if next_phase == previous.phase else 0.0)
            - (drift_score * 0.22)
            - (novelty_score * 0.08),
            0.12,
            0.98,
        )
        uncertainty_score = _clamp((1.0 - phase_confidence) * 0.64 + drift_score * 0.22 + novelty_score * 0.14)
        wm_rollout_uncertainty = wm_summary.get("rollout_uncertainty", previous.rollout_uncertainty)
        rollout_uncertainty = _clamp(
            (float(wm_rollout_uncertainty or 0.0) * 0.72) + (uncertainty_score * 0.28)
            if wm_rollout_uncertainty not in (None, "")
            else uncertainty_score
        )
        transition_prior_signature = str(
            wm_summary.get("transition_prior_signature", previous.transition_prior_signature) or ""
        )

        fn_name = str(function_name or "wait")
        if fn_name and fn_name != "wait":
            self._function_history.append(fn_name)
        self._phase_history.append(next_phase)
        self._record_transition(
            previous_phase=str(previous.phase or "exploring"),
            next_phase=next_phase,
            success=success,
            reward=float(reward or 0.0),
            function_name=fn_name,
            depth_gain=depth_gain,
        )

        focus_functions = [name for name, _ in Counter(self._function_history).most_common(3)]
        recent_phase_path = list(self._phase_history)[-4:]
        latent_signature = f"{next_phase}::{focus_functions[0] if focus_functions else 'none'}::{hidden_state_depth}"
        transition_memory = self._build_transition_memory(
            current_phase=next_phase,
            phase_confidence=phase_confidence,
            stability_score=stability_score,
            drift_score=drift_score,
            focus_functions=focus_functions,
            rollout_uncertainty=rollout_uncertainty,
            transition_prior_signature=transition_prior_signature,
        )
        transition_memory["observed_modality"] = observed_modality
        transition_memory["world_state_signature"] = world_state_signature
        transition_memory["world_entity_count"] = int(world_entity_count)
        transition_memory["world_relation_count"] = int(world_relation_count)
        transition_memory["last_event_signature"] = last_event_signature
        transition_memory["novelty_score"] = round(float(novelty_score), 4)
        transition_memory["rollout_uncertainty"] = round(float(rollout_uncertainty), 4)
        transition_memory["transition_prior_signature"] = transition_prior_signature

        self._snapshot = HiddenStateSnapshot(
            episode=int(episode),
            tick=int(tick),
            phase=next_phase,
            phase_confidence=phase_confidence,
            hidden_state_depth=int(hidden_state_depth),
            drift_score=drift_score,
            stability_score=stability_score,
            uncertainty_score=uncertainty_score,
            explicit_observation_phase=str(explicit_phase or ""),
            last_function_name=fn_name,
            recent_phase_path=recent_phase_path,
            focus_functions=focus_functions,
            latent_signature=latent_signature,
            dominant_branch_id=str(transition_memory.get("dominant_branch_id", "") or ""),
            latent_branches=[
                dict(item) for item in (transition_memory.get("latent_branches", []) or [])
                if isinstance(item, dict)
            ],
            expected_next_phase=str(transition_memory.get("expected_next_phase", "") or ""),
            expected_next_phase_confidence=float(transition_memory.get("expected_next_phase_confidence", 0.0) or 0.0),
            transition_entropy=float(transition_memory.get("phase_transition_entropy", 1.0) or 1.0),
            transition_memory=transition_memory,
            observed_modality=observed_modality,
            world_state_signature=world_state_signature,
            world_entity_count=int(world_entity_count),
            world_relation_count=int(world_relation_count),
            last_event_signature=last_event_signature,
            novelty_score=float(novelty_score),
            rollout_uncertainty=float(rollout_uncertainty),
            transition_prior_signature=transition_prior_signature,
            update_count=int(previous.update_count) + 1,
        )
        return self.summary()

    def record_prediction_error(self, total_error: float) -> Dict[str, Any]:
        error_value = _clamp(float(total_error or 0.0))
        if error_value <= 0.0:
            return self.summary()
        snapshot = self._snapshot
        drift_score = _clamp(snapshot.drift_score * 0.8 + error_value * 0.25)
        phase_confidence = _clamp(snapshot.phase_confidence - error_value * 0.18, 0.08, 0.98)
        uncertainty_score = _clamp((1.0 - phase_confidence) * 0.7 + drift_score * 0.3)
        self._snapshot = HiddenStateSnapshot(
            episode=snapshot.episode,
            tick=snapshot.tick,
            phase=snapshot.phase,
            phase_confidence=phase_confidence,
            hidden_state_depth=snapshot.hidden_state_depth,
            drift_score=drift_score,
            stability_score=snapshot.stability_score,
            uncertainty_score=uncertainty_score,
            explicit_observation_phase=snapshot.explicit_observation_phase,
            last_function_name=snapshot.last_function_name,
            recent_phase_path=list(snapshot.recent_phase_path),
            focus_functions=list(snapshot.focus_functions),
            latent_signature=snapshot.latent_signature,
            dominant_branch_id=snapshot.dominant_branch_id,
            latent_branches=[dict(item) for item in snapshot.latent_branches if isinstance(item, dict)],
            expected_next_phase=snapshot.expected_next_phase,
            expected_next_phase_confidence=snapshot.expected_next_phase_confidence,
            transition_entropy=snapshot.transition_entropy,
            transition_memory=dict(snapshot.transition_memory),
            observed_modality=snapshot.observed_modality,
            world_state_signature=snapshot.world_state_signature,
            world_entity_count=snapshot.world_entity_count,
            world_relation_count=snapshot.world_relation_count,
            last_event_signature=snapshot.last_event_signature,
            novelty_score=snapshot.novelty_score,
            rollout_uncertainty=snapshot.rollout_uncertainty,
            transition_prior_signature=snapshot.transition_prior_signature,
            update_count=snapshot.update_count,
        )
        return self.summary()

    def record_retention_failure(
        self,
        failure_type: str,
        *,
        severity: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        failure_key = str(failure_type or '').strip().lower()
        severity_value = _clamp(float(severity or 0.0), 0.0, 1.0)
        if not failure_key or severity_value <= 0.0:
            return self.summary()
        payload = dict(context or {})
        branch_id = str(
            payload.get('rollout_branch_id')
            or payload.get('dominant_branch_id')
            or self._snapshot.dominant_branch_id
            or ''
        ).strip()
        target_phase = _normalize_phase(
            payload.get('rollout_branch_target_phase')
            or payload.get('target_phase')
            or ''
        )
        updated = False
        if branch_id:
            self._update_branch_feedback_bucket(f'branch:{branch_id}', failure_key, severity_value)
            updated = True
        if target_phase:
            self._update_branch_feedback_bucket(f'phase:{target_phase}', failure_key, severity_value * 0.82)
            updated = True
        if not updated and self._snapshot.dominant_branch_id:
            self._update_branch_feedback_bucket(f'branch:{self._snapshot.dominant_branch_id}', failure_key, severity_value * 0.88)
        return self.summary()

    def _update_branch_feedback_bucket(self, bucket_key: str, failure_type: str, severity: float) -> None:
        bucket = self._branch_feedback_stats.setdefault(
            str(bucket_key),
            {
                'branch_persistence_collapse': 0.0,
                'planner_target_switch': 0.0,
                'prediction_drift': 0.0,
                'governance_overrule_misfire': 0.0,
                'instability': 0.0,
                'count': 0.0,
            },
        )
        for key in ('branch_persistence_collapse', 'planner_target_switch', 'prediction_drift', 'governance_overrule_misfire'):
            previous = _clamp(float(bucket.get(key, 0.0) or 0.0))
            target = severity if key == failure_type else 0.0
            momentum = 0.74 if target > 0.0 else 0.92
            bucket[key] = _clamp(previous * momentum + target * (1.0 - momentum))
        bucket['instability'] = _clamp(
            bucket.get('branch_persistence_collapse', 0.0) * 0.48
            + bucket.get('planner_target_switch', 0.0) * 0.28
            + bucket.get('prediction_drift', 0.0) * 0.14
            + bucket.get('governance_overrule_misfire', 0.0) * 0.10
        )
        bucket['count'] = float(bucket.get('count', 0.0) or 0.0) + 1.0

    def _branch_feedback_for(self, branch_id: str, target_phase: str) -> Dict[str, float]:
        direct = self._branch_feedback_stats.get(f'branch:{str(branch_id or "").strip()}') if branch_id else None
        phase_bucket = self._branch_feedback_stats.get(f'phase:{str(target_phase or "").strip()}') if target_phase else None
        combined = {
            'branch_persistence_collapse': 0.0,
            'planner_target_switch': 0.0,
            'prediction_drift': 0.0,
            'governance_overrule_misfire': 0.0,
            'instability': 0.0,
            'count': 0.0,
        }
        if isinstance(direct, dict):
            for key in combined:
                combined[key] = max(combined[key], float(direct.get(key, 0.0) or 0.0))
        if isinstance(phase_bucket, dict):
            for key in combined:
                multiplier = 0.82 if key != 'count' else 1.0
                combined[key] = max(combined[key], float(phase_bucket.get(key, 0.0) or 0.0) * multiplier)
        return combined

    def _feedback_adjusted_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(summary, dict):
            return {}
        transition_memory = dict(summary.get('transition_memory', {}) or {})
        raw_branches = summary.get('latent_branches', transition_memory.get('latent_branches', []))
        adjusted_branches: List[Dict[str, Any]] = []
        for item in list(raw_branches or []):
            if not isinstance(item, dict):
                continue
            branch = dict(item)
            feedback = self._branch_feedback_for(
                str(branch.get('branch_id', '') or ''),
                _normalize_phase(branch.get('target_phase', '') or ''),
            )
            feedback_penalty = _clamp(
                feedback.get('branch_persistence_collapse', 0.0) * 0.28
                + feedback.get('planner_target_switch', 0.0) * 0.20
                + feedback.get('prediction_drift', 0.0) * 0.10
                + feedback.get('governance_overrule_misfire', 0.0) * 0.08
            )
            branch['feedback_penalty'] = round(feedback_penalty, 4)
            branch['feedback_instability'] = round(_clamp(feedback.get('instability', 0.0)), 4)
            branch['feedback_count'] = int(feedback.get('count', 0.0) or 0.0)
            branch['confidence'] = round(_clamp(float(branch.get('confidence', 0.0) or 0.0) - feedback_penalty * 0.58), 4)
            branch['support'] = round(_clamp(float(branch.get('support', 0.0) or 0.0) - feedback_penalty * 0.30), 4)
            branch['success_rate'] = round(_clamp(float(branch.get('success_rate', 0.0) or 0.0) - feedback_penalty * 0.46), 4)
            branch['avg_reward'] = round(float(branch.get('avg_reward', 0.0) or 0.0) - feedback_penalty * 0.34, 4)
            branch['uncertainty_pressure'] = round(
                _clamp(float(branch.get('uncertainty_pressure', 0.0) or 0.0) + feedback_penalty * 0.70),
                4,
            )
            adjusted_branches.append(branch)
        adjusted_branches = sorted(
            adjusted_branches,
            key=lambda item: (
                -float(item.get('confidence', 0.0) or 0.0),
                -float(item.get('support', 0.0) or 0.0),
                str(item.get('branch_id', '') or ''),
            ),
        )[:3]
        if adjusted_branches:
            summary['latent_branches'] = adjusted_branches
            summary['dominant_branch_id'] = str(adjusted_branches[0].get('branch_id', '') or '')
            transition_memory['latent_branches'] = [dict(item) for item in adjusted_branches]
            transition_memory['dominant_branch_id'] = summary['dominant_branch_id']
            dominant_penalty = float(adjusted_branches[0].get('feedback_penalty', 0.0) or 0.0)
            summary['phase_confidence'] = round(_clamp(float(summary.get('phase_confidence', 0.0) or 0.0) - dominant_penalty * 0.12, 0.08, 0.98), 4)
            summary['stability_score'] = round(_clamp(float(summary.get('stability_score', 0.0) or 0.0) - dominant_penalty * 0.18), 4)
            summary['uncertainty_score'] = round(_clamp(max(float(summary.get('uncertainty_score', 0.0) or 0.0), float(summary.get('uncertainty_score', 0.0) or 0.0) * 0.82 + dominant_penalty * 0.42)), 4)
        summary['transition_memory'] = transition_memory
        return summary

    def _record_transition(
        self,
        *,
        previous_phase: str,
        next_phase: str,
        success: bool,
        reward: float,
        function_name: str,
        depth_gain: float,
    ) -> None:
        prev = _normalize_phase(previous_phase) or "exploring"
        nxt = _normalize_phase(next_phase) or prev
        self._phase_transition_counts[prev][nxt] += 1

        transition_key = (prev, nxt)
        transition_stat = self._phase_transition_stats.setdefault(
            transition_key,
            {
                "count": 0.0,
                "success_sum": 0.0,
                "reward_sum": 0.0,
                "depth_gain_sum": 0.0,
            },
        )
        transition_stat["count"] += 1.0
        transition_stat["success_sum"] += 1.0 if success else 0.0
        transition_stat["reward_sum"] += float(reward or 0.0)
        transition_stat["depth_gain_sum"] += float(depth_gain or 0.0)

        fn_name = str(function_name or "wait").strip() or "wait"
        if fn_name == "wait":
            return
        phase_function_key = (prev, fn_name)
        fn_stat = self._phase_function_stats.setdefault(
            phase_function_key,
            {
                "count": 0.0,
                "success_sum": 0.0,
                "reward_sum": 0.0,
                "depth_gain_sum": 0.0,
                "stabilizing_sum": 0.0,
                "committed_sum": 0.0,
                "disrupted_sum": 0.0,
            },
        )
        fn_stat["count"] += 1.0
        fn_stat["success_sum"] += 1.0 if success else 0.0
        fn_stat["reward_sum"] += float(reward or 0.0)
        fn_stat["depth_gain_sum"] += float(depth_gain or 0.0)
        if nxt in {"stabilizing", "committed"}:
            fn_stat["stabilizing_sum"] += 1.0
        if nxt == "committed":
            fn_stat["committed_sum"] += 1.0
        if nxt == "disrupted" or float(reward or 0.0) < 0.0:
            fn_stat["disrupted_sum"] += 1.0

    def _build_transition_memory(
        self,
        *,
        current_phase: str,
        phase_confidence: float,
        stability_score: float,
        drift_score: float,
        focus_functions: List[str],
        rollout_uncertainty: float,
        transition_prior_signature: str,
    ) -> Dict[str, Any]:
        phase_key = _normalize_phase(current_phase) or "exploring"
        transitions = self._phase_transition_counts.get(phase_key, Counter())
        total = float(sum(int(value or 0) for value in transitions.values()) or 0.0)
        entropy = 1.0
        dominant_transitions: List[Dict[str, Any]] = []
        expected_next_phase = phase_key if stability_score >= 0.45 else ""
        expected_next_phase_confidence = _clamp(
            (phase_confidence * 0.42) + (stability_score * 0.28) - (drift_score * 0.14),
            0.0,
            1.0,
        ) if expected_next_phase else 0.0

        if total > 0.0:
            ranked_transitions: List[Tuple[float, Dict[str, Any]]] = []
            probabilities: List[float] = []
            for next_phase, count in transitions.items():
                stat = self._phase_transition_stats.get((phase_key, next_phase), {})
                support = float(count or 0.0) / max(total, 1.0)
                success_rate = float(stat.get("success_sum", 0.0) or 0.0) / max(float(stat.get("count", 1.0) or 1.0), 1.0)
                avg_reward = float(stat.get("reward_sum", 0.0) or 0.0) / max(float(stat.get("count", 1.0) or 1.0), 1.0)
                avg_depth_gain = float(stat.get("depth_gain_sum", 0.0) or 0.0) / max(float(stat.get("count", 1.0) or 1.0), 1.0)
                score = (
                    support * 0.45
                    + success_rate * 0.30
                    + _clamp((avg_reward + 1.0) / 2.0) * 0.15
                    + _clamp(avg_depth_gain, 0.0, 1.0) * 0.10
                )
                transition_payload = {
                    "from_phase": phase_key,
                    "to_phase": str(next_phase or phase_key),
                    "support": round(support, 4),
                    "count": int(count or 0),
                    "success_rate": round(success_rate, 4),
                    "avg_reward": round(avg_reward, 4),
                    "avg_depth_gain": round(avg_depth_gain, 4),
                    "score": round(score, 4),
                }
                ranked_transitions.append((score, transition_payload))
                probabilities.append(max(0.0, support))
            dominant_transitions = [
                payload
                for _, payload in sorted(ranked_transitions, key=lambda item: (-float(item[0]), item[1]["to_phase"]))[:3]
            ]
            if probabilities:
                if len(probabilities) == 1:
                    entropy = 0.0
                else:
                    raw_entropy = -sum(
                        p * math.log(max(p, 1e-9))
                        for p in probabilities
                        if p > 0.0
                    )
                    entropy = _clamp(raw_entropy / math.log(len(probabilities)), 0.0, 1.0)
            if dominant_transitions:
                best = dominant_transitions[0]
                expected_next_phase = str(best.get("to_phase", "") or expected_next_phase)
                expected_next_phase_confidence = _clamp(
                    (float(best.get("support", 0.0) or 0.0) * 0.42)
                    + (float(best.get("success_rate", 0.0) or 0.0) * 0.24)
                    + ((1.0 - entropy) * 0.18)
                    + (phase_confidence * 0.10)
                    + (stability_score * 0.10)
                    - (drift_score * 0.12),
                    0.0,
                    1.0,
                )

        phase_function_scores: Dict[str, Dict[str, float]] = {}
        for (phase, function_name), stat in self._phase_function_stats.items():
            if phase != phase_key:
                continue
            count = max(1.0, float(stat.get("count", 0.0) or 0.0))
            success_rate = float(stat.get("success_sum", 0.0) or 0.0) / count
            avg_reward = float(stat.get("reward_sum", 0.0) or 0.0) / count
            avg_depth_gain = float(stat.get("depth_gain_sum", 0.0) or 0.0) / count
            stabilizing_rate = float(stat.get("stabilizing_sum", 0.0) or 0.0) / count
            committed_rate = float(stat.get("committed_sum", 0.0) or 0.0) / count
            disrupted_rate = float(stat.get("disrupted_sum", 0.0) or 0.0) / count
            reward_score = _clamp((avg_reward + 1.0) / 2.0, 0.0, 1.0)
            stabilizing_score = _clamp(
                success_rate * 0.30
                + reward_score * 0.22
                + stabilizing_rate * 0.22
                + committed_rate * 0.18
                + _clamp(avg_depth_gain, 0.0, 1.0) * 0.12
                - disrupted_rate * 0.24,
                0.0,
                1.0,
            )
            risk_score = _clamp(
                (1.0 - success_rate) * 0.24
                + (1.0 - reward_score) * 0.22
                + disrupted_rate * 0.34
                + drift_score * 0.12
                - stabilizing_rate * 0.16,
                0.0,
                1.0,
            )
            phase_function_scores[function_name] = {
                "support": round(count, 4),
                "success_rate": round(success_rate, 4),
                "avg_reward": round(avg_reward, 4),
                "avg_depth_gain": round(avg_depth_gain, 4),
                "stabilizing_rate": round(stabilizing_rate, 4),
                "committed_rate": round(committed_rate, 4),
                "disrupted_rate": round(disrupted_rate, 4),
                "stabilizing_score": round(stabilizing_score, 4),
                "risk_score": round(risk_score, 4),
            }

        stabilizing_functions = [
            {"function_name": name, **payload}
            for name, payload in sorted(
                phase_function_scores.items(),
                key=lambda item: (-float(item[1].get("stabilizing_score", 0.0) or 0.0), float(item[1].get("risk_score", 0.0) or 0.0), item[0]),
            )[:3]
            if float(payload.get("support", 0.0) or 0.0) >= 1.0 and float(payload.get("stabilizing_score", 0.0) or 0.0) >= 0.45
        ]
        risky_functions = [
            {"function_name": name, **payload}
            for name, payload in sorted(
                phase_function_scores.items(),
                key=lambda item: (-float(item[1].get("risk_score", 0.0) or 0.0), float(item[1].get("stabilizing_score", 0.0) or 0.0), item[0]),
            )[:3]
            if float(payload.get("support", 0.0) or 0.0) >= 1.0 and float(payload.get("risk_score", 0.0) or 0.0) >= 0.35
        ]
        latent_branches = self._build_latent_branches(
            current_phase=phase_key,
            dominant_transitions=dominant_transitions,
            phase_function_scores=phase_function_scores,
            focus_functions=focus_functions,
            phase_confidence=phase_confidence,
            stability_score=stability_score,
            drift_score=drift_score,
            entropy=entropy,
        )
        dominant_branch_id = str(latent_branches[0].get("branch_id", "") or "") if latent_branches else ""

        return {
            "current_phase": phase_key,
            "expected_next_phase": expected_next_phase,
            "expected_next_phase_confidence": round(expected_next_phase_confidence, 4),
            "phase_transition_entropy": round(entropy, 4),
            "dominant_transitions": dominant_transitions,
            "stabilizing_functions": stabilizing_functions,
            "risky_functions": risky_functions,
            "phase_function_scores": phase_function_scores,
            "dominant_branch_id": dominant_branch_id,
            "latent_branches": latent_branches,
            "rollout_uncertainty": round(float(rollout_uncertainty), 4),
            "transition_prior_signature": str(transition_prior_signature or ""),
        }

    def _branch_anchor_affinity(self, *, target_phase: str, payload: Dict[str, Any]) -> float:
        support_factor = _clamp(float(payload.get("support", 0.0) or 0.0) / 3.0, 0.0, 1.0)
        success_rate = _clamp(float(payload.get("success_rate", 0.0) or 0.0))
        reward_score = _clamp((float(payload.get("avg_reward", 0.0) or 0.0) + 1.0) / 2.0, 0.0, 1.0)
        depth_gain = _clamp(float(payload.get("avg_depth_gain", 0.0) or 0.0), 0.0, 1.0)
        stabilizing_rate = _clamp(float(payload.get("stabilizing_rate", 0.0) or 0.0))
        committed_rate = _clamp(float(payload.get("committed_rate", 0.0) or 0.0))
        disrupted_rate = _clamp(float(payload.get("disrupted_rate", 0.0) or 0.0))
        stabilizing_score = _clamp(float(payload.get("stabilizing_score", 0.0) or 0.0))
        risk_score = _clamp(float(payload.get("risk_score", 0.0) or 0.0))
        phase_key = _normalize_phase(target_phase) or "exploring"
        if phase_key == "committed":
            return _clamp(
                support_factor * 0.18
                + committed_rate * 0.30
                + stabilizing_score * 0.20
                + success_rate * 0.12
                + reward_score * 0.12
                + depth_gain * 0.10
                - risk_score * 0.22,
            )
        if phase_key == "stabilizing":
            return _clamp(
                support_factor * 0.16
                + stabilizing_rate * 0.26
                + stabilizing_score * 0.24
                + success_rate * 0.12
                + reward_score * 0.10
                + depth_gain * 0.12
                - risk_score * 0.18,
            )
        if phase_key == "disrupted":
            return _clamp(
                support_factor * 0.14
                + disrupted_rate * 0.34
                + risk_score * 0.28
                + (1.0 - success_rate) * 0.12
                + (1.0 - reward_score) * 0.08
                - stabilizing_score * 0.16,
            )
        return _clamp(
            support_factor * 0.16
            + depth_gain * 0.24
            + success_rate * 0.18
            + reward_score * 0.10
            + (1.0 - risk_score) * 0.10
            + (1.0 - committed_rate) * 0.10,
        )

    def _branch_opposition_score(self, *, target_phase: str, payload: Dict[str, Any]) -> float:
        support_factor = _clamp(float(payload.get("support", 0.0) or 0.0) / 3.0, 0.0, 1.0)
        success_rate = _clamp(float(payload.get("success_rate", 0.0) or 0.0))
        reward_score = _clamp((float(payload.get("avg_reward", 0.0) or 0.0) + 1.0) / 2.0, 0.0, 1.0)
        stabilizing_score = _clamp(float(payload.get("stabilizing_score", 0.0) or 0.0))
        committed_rate = _clamp(float(payload.get("committed_rate", 0.0) or 0.0))
        disrupted_rate = _clamp(float(payload.get("disrupted_rate", 0.0) or 0.0))
        risk_score = _clamp(float(payload.get("risk_score", 0.0) or 0.0))
        phase_key = _normalize_phase(target_phase) or "exploring"
        if phase_key in {"stabilizing", "committed"}:
            return _clamp(
                support_factor * 0.16
                + risk_score * 0.34
                + disrupted_rate * 0.24
                + (1.0 - success_rate) * 0.12
                + (1.0 - reward_score) * 0.08
                - stabilizing_score * 0.14,
            )
        if phase_key == "disrupted":
            return _clamp(
                support_factor * 0.16
                + stabilizing_score * 0.28
                + committed_rate * 0.24
                + success_rate * 0.12
                + reward_score * 0.10
                - risk_score * 0.14,
            )
        return _clamp(
            support_factor * 0.14
            + abs(stabilizing_score - risk_score) * 0.28
            + committed_rate * 0.12
            + disrupted_rate * 0.12,
        )

    def _build_latent_branches(
        self,
        *,
        current_phase: str,
        dominant_transitions: List[Dict[str, Any]],
        phase_function_scores: Dict[str, Dict[str, float]],
        focus_functions: List[str],
        phase_confidence: float,
        stability_score: float,
        drift_score: float,
        entropy: float,
    ) -> List[Dict[str, Any]]:
        phase_key = _normalize_phase(current_phase) or "exploring"
        latent_branches: List[Dict[str, Any]] = []
        transition_rows = list(dominant_transitions)
        if not transition_rows:
            fallback_target = phase_key
            transition_rows = [
                {
                    "from_phase": phase_key,
                    "to_phase": fallback_target,
                    "support": round(_clamp(phase_confidence * 0.72 + stability_score * 0.18), 4),
                    "count": 0,
                    "success_rate": round(_clamp(phase_confidence * 0.84 + stability_score * 0.08), 4),
                    "avg_reward": 0.0,
                    "avg_depth_gain": 0.0,
                    "score": round(_clamp(phase_confidence * 0.48 + stability_score * 0.24 - drift_score * 0.12), 4),
                }
            ]

        for transition_row in transition_rows[:3]:
            target_phase = _normalize_phase(transition_row.get("to_phase", phase_key)) or phase_key
            anchor_rows: List[Tuple[float, Dict[str, Any]]] = []
            opposition_rows: List[Tuple[float, Dict[str, Any]]] = []
            for function_name, payload in phase_function_scores.items():
                if not isinstance(payload, dict):
                    continue
                anchor_score = self._branch_anchor_affinity(target_phase=target_phase, payload=payload)
                opposition_score = self._branch_opposition_score(target_phase=target_phase, payload=payload)
                support = float(payload.get("support", 0.0) or 0.0)
                if support >= 1.0 and anchor_score >= 0.38:
                    anchor_rows.append((anchor_score, {"function_name": function_name, "score": round(anchor_score, 4)}))
                if support >= 1.0 and opposition_score >= 0.36:
                    opposition_rows.append((opposition_score, {"function_name": function_name, "score": round(opposition_score, 4)}))

            anchor_rows = sorted(anchor_rows, key=lambda item: (-float(item[0]), item[1]["function_name"]))[:4]
            opposition_rows = sorted(opposition_rows, key=lambda item: (-float(item[0]), item[1]["function_name"]))[:4]
            anchor_functions = [str(item[1]["function_name"]) for item in anchor_rows]
            if not anchor_functions and focus_functions:
                anchor_functions = [str(name) for name in focus_functions[:2] if str(name or "")]
            risky_functions = [str(item[1]["function_name"]) for item in opposition_rows if str(item[1]["function_name"]) not in set(anchor_functions)]

            support = _clamp(float(transition_row.get("support", 0.0) or 0.0), 0.0, 1.0)
            success_rate = _clamp(float(transition_row.get("success_rate", 0.0) or 0.0), 0.0, 1.0)
            transition_score = _clamp(float(transition_row.get("score", 0.0) or 0.0), 0.0, 1.0)
            avg_reward = float(transition_row.get("avg_reward", 0.0) or 0.0)
            avg_depth_gain = _clamp(float(transition_row.get("avg_depth_gain", 0.0) or 0.0), 0.0, 1.0)
            branch_confidence = _clamp(
                transition_score * 0.34
                + support * 0.20
                + success_rate * 0.14
                + phase_confidence * 0.12
                + stability_score * 0.10
                + (1.0 - entropy) * 0.10
                - drift_score * 0.12,
            )
            uncertainty_pressure = _clamp(
                entropy * 0.42
                + drift_score * 0.36
                + (1.0 - support) * 0.22,
            )
            branch_id = f"{phase_key}->{target_phase}"
            latent_branches.append(
                {
                    "branch_id": branch_id,
                    "current_phase": phase_key,
                    "target_phase": target_phase,
                    "confidence": round(branch_confidence, 4),
                    "support": round(support, 4),
                    "transition_score": round(transition_score, 4),
                    "success_rate": round(success_rate, 4),
                    "avg_reward": round(avg_reward, 4),
                    "avg_depth_gain": round(avg_depth_gain, 4),
                    "uncertainty_pressure": round(uncertainty_pressure, 4),
                    "anchor_functions": anchor_functions,
                    "risky_functions": risky_functions[:4],
                    "latent_signature": f"{phase_key}->{target_phase}::{anchor_functions[0] if anchor_functions else 'none'}",
                }
            )

        return sorted(
            latent_branches,
            key=lambda item: (
                -float(item.get("confidence", 0.0) or 0.0),
                -float(item.get("support", 0.0) or 0.0),
                str(item.get("branch_id", "") or ""),
            ),
        )[:3]

    def _extract_explicit_phase(self, payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        world_state = payload.get("world_state", {}) if isinstance(payload.get("world_state", {}), dict) else {}
        for key in ("phase", "state"):
            value = world_state.get(key)
            normalized = _normalize_phase(value)
            if normalized:
                return normalized
        return ""

    def _infer_phase(
        self,
        *,
        explicit_phase: str,
        wm_phase: str,
        reward: float,
        success: bool,
        previous_phase: str,
        shift_risk: float,
    ) -> str:
        if explicit_phase:
            return explicit_phase
        if not success or reward < -0.05:
            return "disrupted" if shift_risk >= 0.45 else "exploring"
        if reward > 0.45:
            return "committed"
        if reward > 0.0:
            return "stabilizing" if wm_phase in ("exploring", "") else wm_phase
        if wm_phase:
            return wm_phase
        if previous_phase in ("stabilizing", "committed") and shift_risk < 0.35:
            return previous_phase
        return "exploring"

    def _next_depth(self, previous_depth: int, next_phase: str, previous_phase: str, phase_changed: bool) -> int:
        depth = int(previous_depth or 0)
        if next_phase in ("stabilizing", "committed"):
            if not phase_changed and previous_phase == next_phase:
                return min(8, depth + 1)
            if previous_phase in ("stabilizing", "committed"):
                return min(8, max(1, depth // 2 + 1))
            return 1
        if next_phase == "disrupted":
            return max(0, depth - 2)
        return max(0, depth - 1)
