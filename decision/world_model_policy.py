"""
decision/world_model_policy.py

Internal world-model policy used by arbiter to project long-horizon effects
and enforce soft/hard behavioral constraints.
"""

from __future__ import annotations

from typing import Any, Dict

from decision.utility_schema import DecisionCandidate
from modules.world_model.rollout import simulate_function_rollout


class WorldModelPolicy:
    """Lightweight internal generator for candidate consequence projection."""

    def project_candidate(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> Dict[str, float]:
        beliefs = self._beliefs(context)
        transition_priors = self._transition_priors(context)
        world_model_summary = self._world_model_summary(context)

        fn = candidate.function_name or ""
        fn_prior = transition_priors.get(fn, {}) if isinstance(transition_priors, dict) else {}
        rollout = simulate_function_rollout(
            fn,
            world_model_summary=world_model_summary,
            transition_priors=transition_priors,
            candidate_intervention_targets=world_model_summary.get('candidate_intervention_targets', []) if isinstance(world_model_summary, dict) else [],
            mechanism_hypotheses=world_model_summary.get('mechanism_hypotheses', []) if isinstance(world_model_summary, dict) else [],
        )

        long_horizon_reward = float(fn_prior.get("long_horizon_reward", 0.0))
        predicted_risk = float(fn_prior.get("predicted_risk", 0.0))
        reversibility = float(fn_prior.get("reversibility", 0.3))
        if rollout:
            long_horizon_reward = max(long_horizon_reward, float(rollout.get("expected_reward", 0.0) or 0.0))
            predicted_risk = max(predicted_risk, float(rollout.get("state_shift_risk", 0.0) or 0.0))
            reversibility = max(reversibility, float(rollout.get("reversibility", 0.0) or 0.0))

        # belief-driven constraint signal
        violation_prob = 0.0
        risky_motion = beliefs.get("observation_camera_motion", {}) if isinstance(beliefs, dict) else {}
        if isinstance(risky_motion, dict):
            if risky_motion.get("posterior") == "high_motion" and fn in {"join_tables", "aggregate_group"}:
                violation_prob = max(violation_prob, 0.7)

        if fn_prior.get("constraint_violation") is not None:
            violation_prob = max(violation_prob, float(fn_prior.get("constraint_violation", 0.0)))

        info_gain = float(fn_prior.get("info_gain", 0.0))
        if rollout:
            info_gain = max(info_gain, float(rollout.get("expected_information_gain", 0.0) or 0.0))
            matched_targets = rollout.get("matched_targets", []) if isinstance(rollout.get("matched_targets", []), list) else []
            matched_mechanisms = rollout.get("matched_mechanisms", []) if isinstance(rollout.get("matched_mechanisms", []), list) else []
            if matched_targets:
                long_horizon_reward += min(0.18, len(matched_targets) * 0.06)
                violation_prob = max(0.0, violation_prob - min(0.10, len(matched_targets) * 0.03))
            if matched_mechanisms:
                long_horizon_reward += min(0.22, max(float(row.get('confidence', 0.0) or 0.0) for row in matched_mechanisms) * 0.18)
                violation_prob = max(0.0, violation_prob - 0.06)

        return {
            "long_horizon_reward": max(-1.0, min(1.0, long_horizon_reward)),
            "predicted_risk": max(0.0, min(1.0, predicted_risk)),
            "reversibility": max(0.0, min(1.0, reversibility)),
            "constraint_violation_prob": max(0.0, min(1.0, violation_prob)),
            "info_gain": max(0.0, min(1.0, info_gain)),
        }

    def _world_model_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wm = context.get("world_model_summary", {})
        return wm if isinstance(wm, dict) else {}

    def _beliefs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wm = context.get("world_model_summary", {})
        if not isinstance(wm, dict):
            return {}
        beliefs = wm.get("beliefs", {})
        return beliefs if isinstance(beliefs, dict) else {}

    def _transition_priors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        priors = context.get("world_model_transition_priors", {})
        return priors if isinstance(priors, dict) else {}
