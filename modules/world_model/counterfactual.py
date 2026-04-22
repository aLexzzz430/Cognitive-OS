"""
modules/world_model/counterfactual.py

Stage D: Minimal Counterfactual Engine

Provides bounded local counterfactual reasoning for high-value mechanisms only.
NOT a full simulator. This is advisory only — does not take control actions.

Methods:
- simulate_action_difference(state_summary, action_a, action_b)
- simulate_object_removed(state_summary, object_id)
- simulate_hypothesis_true_false(hypothesis_id)

Core principle:
- Operates on local slices only (one belief cluster, one object subset)
- Output is structured, low-scope, and auditable
- Remains advisory — governance/recovery/planning consume but do not have to follow
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import random

from modules.world_model.rollout import compare_function_rollouts


class CounterfactualConfidence(Enum):
    """How confident we are in the counterfactual prediction."""
    HIGH = "high"      # Based on established beliefs
    MEDIUM = "medium"  # Based on probable beliefs
    LOW = "low"       # Based on hypothesis or uncertain beliefs


@dataclass
class CounterfactualOutcome:
    """
    Structured output of a counterfactual simulation.
    
    All fields are JSON-safe for logging and auditing.
    """
    outcome_id: str
    question: str                    # What was being counterfactually compared
    preferred_action: str             # Which action seems better (action_a / action_b / equal)
    predicted_difference: str        # Plain-text description of predicted difference
    confidence: CounterfactualConfidence
    reasoning: str                    # Why this prediction was made
    belief_ids_used: List[str]       # Which beliefs informed this prediction
    hypothesis_id: Optional[str] = None  # If a specific hypothesis was being tested
    estimated_delta: Optional[float] = None  # Numeric estimate of difference (if applicable)
    decision_path: str = "priors_fallback"
    mechanism_match: Optional[Dict[str, Any]] = None
    rollout_trace: List[Dict[str, Any]] = field(default_factory=list)
    rollout_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSlice:
    """
    A minimal snapshot of state for counterfactual reasoning.
    
    Only includes what is needed for local counterfactual reasoning.
    NOT a full world state.
    """
    available_functions: List[str] = field(default_factory=list)
    established_beliefs: List[Dict[str, Any]] = field(default_factory=list)
    active_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    recent_actions: List[str] = field(default_factory=list)  # Last 3-5 actions
    current_reward_trend: str = "neutral"  # "positive" / "neutral" / "negative"
    state_features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'available_functions': self.available_functions,
            'established_beliefs': self.established_beliefs,
            'active_hypotheses': self.active_hypotheses,
            'recent_actions': self.recent_actions,
            'current_reward_trend': self.current_reward_trend,
            'state_features': self.state_features,
        }


class CounterfactualEngine:
    """
    Bounded counterfactual reasoning engine.
    
    Advisory only — outputs are suggestions, not directives.
    All predictions are logged for audit.
    """
    
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._history: List[CounterfactualOutcome] = []

    @staticmethod
    def _clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return max(low, min(high, float(default)))

    @staticmethod
    def _function_name(action: Dict[str, Any]) -> str:
        if not isinstance(action, dict):
            return 'wait'
        if action.get('kind') == 'wait':
            return 'wait'
        payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
        tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
        return str(tool_args.get('function_name', 'wait') or 'wait')

    @staticmethod
    def _phase_alias(raw_phase: Any) -> str:
        phase = str(raw_phase or '').strip().lower()
        if not phase:
            return ''
        aliases = {
            'explore': 'exploring',
            'exploration': 'exploring',
            'stabilize': 'stabilizing',
            'stable': 'stabilizing',
            'commit': 'committed',
            'completed': 'committed',
            'complete': 'committed',
            'solved': 'committed',
            'fail': 'disrupted',
            'failed': 'disrupted',
            'error': 'disrupted',
            'drift': 'disrupted',
        }
        return aliases.get(phase, phase)

    def _transition_phase_function_scores(
        self,
        *,
        context: Dict[str, Any],
        phase: str,
        transition_memory: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        target_phase = self._phase_alias(phase) or 'exploring'
        memory_phase = self._phase_alias(transition_memory.get('current_phase', ''))
        if memory_phase == target_phase:
            raw_scores = transition_memory.get('phase_function_scores', {})
            if isinstance(raw_scores, dict):
                return {
                    str(name): dict(payload)
                    for name, payload in raw_scores.items()
                    if isinstance(payload, dict)
                }

        priors = context.get('transition_priors', {})
        priors = priors if isinstance(priors, dict) else {}
        by_signature = priors.get('__by_signature', {}) if isinstance(priors.get('__by_signature', {}), dict) else {}
        phase_scores: Dict[str, Dict[str, float]] = {}
        for entry in by_signature.values():
            if not isinstance(entry, dict):
                continue
            key = entry.get('key', {}) if isinstance(entry.get('key', {}), dict) else {}
            belief_phase = self._phase_alias(key.get('belief_phase', ''))
            if belief_phase != target_phase:
                continue
            fn_name = str(key.get('function_name', '') or '')
            if not fn_name:
                continue
            long_reward = float(entry.get('long_horizon_reward', 0.0) or 0.0)
            predicted_risk = self._clamp(entry.get('predicted_risk', 0.0))
            reversibility = self._clamp(entry.get('reversibility', 0.0))
            info_gain = self._clamp(entry.get('info_gain', 0.0))
            constraint_violation = self._clamp(entry.get('constraint_violation', 0.0))
            affinity = float(entry.get('transition_affinity', 0.0) or 0.0)
            support = max(1.0, float(entry.get('long_horizon_reward_sample_count', 1.0) or 1.0))
            reward_score = self._clamp((long_reward + 1.0) / 2.0)
            stabilizing_score = self._clamp(
                reward_score * 0.34
                + reversibility * 0.24
                + (1.0 - predicted_risk) * 0.20
                + info_gain * 0.10
                + self._clamp((affinity + 1.0) / 2.0) * 0.12
                - constraint_violation * 0.12,
            )
            risk_score = self._clamp(
                predicted_risk * 0.36
                + constraint_violation * 0.24
                + (1.0 - reward_score) * 0.18
                + (1.0 - reversibility) * 0.10
                - max(0.0, affinity) * 0.08,
            )
            phase_scores[fn_name] = {
                'support': support,
                'success_rate': self._clamp(reward_score + (1.0 - predicted_risk) * 0.15, 0.0, 1.0),
                'avg_reward': round(long_reward, 4),
                'avg_depth_gain': round(info_gain * 0.8, 4),
                'stabilizing_rate': round(stabilizing_score, 4),
                'committed_rate': round(max(0.0, affinity), 4),
                'disrupted_rate': round(risk_score, 4),
                'stabilizing_score': round(stabilizing_score, 4),
                'risk_score': round(risk_score, 4),
            }
        return phase_scores

    def _rollout_current_phase(
        self,
        *,
        state_slice: StateSlice,
        world_dynamics: Dict[str, Any],
        hidden_state: Dict[str, Any],
    ) -> str:
        state_features = state_slice.state_features if isinstance(state_slice.state_features, dict) else {}
        return (
            self._phase_alias(hidden_state.get('phase', ''))
            or self._phase_alias(state_features.get('world_phase', ''))
            or self._phase_alias(world_dynamics.get('predicted_phase', ''))
            or 'exploring'
        )

    def _phase_path_next_phase(
        self,
        *,
        current_phase: str,
        expected_next_phase: str,
        expected_next_phase_confidence: float,
        transition_entropy: float,
        fn_stats: Dict[str, Any],
        step_index: int,
    ) -> str:
        current = self._phase_alias(current_phase) or 'exploring'
        expected = self._phase_alias(expected_next_phase)
        stabilizing_score = self._clamp(fn_stats.get('stabilizing_score', 0.0))
        risk_score = self._clamp(fn_stats.get('risk_score', 0.0))
        committed_rate = self._clamp(fn_stats.get('committed_rate', 0.0))

        if step_index == 0 and expected and expected_next_phase_confidence >= 0.48:
            if risk_score < 0.72:
                return expected
        if current == 'committed':
            return 'committed' if risk_score < 0.72 else 'stabilizing'
        if risk_score >= 0.68:
            return 'disrupted'
        if current == 'stabilizing' and (committed_rate >= 0.42 or stabilizing_score >= 0.74):
            return 'committed'
        if stabilizing_score >= 0.56:
            return 'stabilizing'
        if transition_entropy >= 0.72 and current != 'committed':
            return 'exploring'
        return current if current in {'stabilizing', 'committed'} else 'exploring'

    def _normalize_latent_branches(self, raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        branches: List[Dict[str, Any]] = []
        for item in raw[:4]:
            if not isinstance(item, dict):
                continue
            branches.append(
                {
                    'branch_id': str(item.get('branch_id', '') or '').strip(),
                    'current_phase': self._phase_alias(item.get('current_phase', '')) or 'exploring',
                    'target_phase': self._phase_alias(item.get('target_phase', '')) or 'exploring',
                    'confidence': self._clamp(item.get('confidence', 0.0), 0.0, 1.0),
                    'support': self._clamp(item.get('support', 0.0), 0.0, 1.0),
                    'transition_score': self._clamp(item.get('transition_score', 0.0), 0.0, 1.0),
                    'success_rate': self._clamp(item.get('success_rate', 0.0), 0.0, 1.0),
                    'avg_reward': float(item.get('avg_reward', 0.0) or 0.0),
                    'avg_depth_gain': self._clamp(item.get('avg_depth_gain', 0.0), 0.0, 1.0),
                    'uncertainty_pressure': self._clamp(item.get('uncertainty_pressure', 0.0), 0.0, 1.0),
                    'anchor_functions': [
                        str(value or '').strip()
                        for value in list(item.get('anchor_functions', []) or [])[:4]
                        if str(value or '').strip()
                    ],
                    'risky_functions': [
                        str(value or '').strip()
                        for value in list(item.get('risky_functions', []) or [])[:4]
                        if str(value or '').strip()
                    ],
                    'latent_signature': str(item.get('latent_signature', '') or ''),
                }
            )
        return branches

    def _rollout_latent_branch_state(
        self,
        *,
        hidden_state: Dict[str, Any],
        transition_memory: Dict[str, Any],
        world_dynamics: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], str]:
        latent_branches = self._normalize_latent_branches(
            hidden_state.get(
                'latent_branches',
                transition_memory.get('latent_branches', world_dynamics.get('latent_branches', [])),
            )
        )
        dominant_branch_id = str(
            hidden_state.get(
                'dominant_branch_id',
                transition_memory.get(
                    'dominant_branch_id',
                    world_dynamics.get('dominant_branch_id', latent_branches[0].get('branch_id', '') if latent_branches else ''),
                ),
            ) or ''
        ).strip()
        return latent_branches, dominant_branch_id

    def _select_rollout_branch(
        self,
        *,
        latent_branches: List[Dict[str, Any]],
        dominant_branch_id: str,
        fn_name: str,
        current_phase: str,
        expected_next_phase: str,
        expected_next_phase_confidence: float,
    ) -> Dict[str, Any]:
        best_branch: Dict[str, Any] = {}
        best_score = -1e9
        dominant_branch: Dict[str, Any] = {}
        for branch in latent_branches:
            branch_id = str(branch.get('branch_id', '') or '').strip()
            if branch_id and branch_id == dominant_branch_id:
                dominant_branch = dict(branch)
            branch_confidence = self._clamp(branch.get('confidence', 0.0), 0.0, 1.0)
            branch_score = branch_confidence * 0.26
            if branch_id and branch_id == dominant_branch_id:
                branch_score += 0.18
            if current_phase and str(branch.get('current_phase', '') or '') == current_phase:
                branch_score += 0.08
            if expected_next_phase and str(branch.get('target_phase', '') or '') == expected_next_phase:
                branch_score += expected_next_phase_confidence * 0.18
            anchor_functions = set(branch.get('anchor_functions', []) or [])
            risky_functions = set(branch.get('risky_functions', []) or [])
            if fn_name in anchor_functions:
                branch_score += 0.46 + branch_confidence * 0.18
            if fn_name in risky_functions:
                branch_score -= 0.56 + branch_confidence * 0.18
            if branch_score > best_score:
                best_score = branch_score
                best_branch = dict(branch)
        if best_branch and best_score >= 0.12:
            return best_branch
        if dominant_branch and self._clamp(dominant_branch.get('confidence', 0.0), 0.0, 1.0) >= 0.48:
            return dominant_branch
        return {}

    def _branch_phase_score_template(
        self,
        *,
        branch: Dict[str, Any],
        role: str,
    ) -> Dict[str, float]:
        target_phase = self._phase_alias(branch.get('target_phase', '')) or 'exploring'
        confidence = self._clamp(branch.get('confidence', 0.0), 0.0, 1.0)
        success_rate = self._clamp(branch.get('success_rate', 0.0), 0.0, 1.0)
        avg_reward = float(branch.get('avg_reward', 0.0) or 0.0)
        avg_depth_gain = self._clamp(branch.get('avg_depth_gain', 0.0), 0.0, 1.0)
        support = max(1.0, 1.0 + confidence * 2.0)

        if role == 'anchor':
            if target_phase == 'committed':
                stabilizing_score = self._clamp(0.56 + confidence * 0.24)
                committed_rate = self._clamp(0.44 + confidence * 0.30)
                risk_score = self._clamp(0.10 + (1.0 - confidence) * 0.16)
            elif target_phase == 'stabilizing':
                stabilizing_score = self._clamp(0.60 + confidence * 0.22)
                committed_rate = self._clamp(0.10 + confidence * 0.12)
                risk_score = self._clamp(0.14 + (1.0 - confidence) * 0.18)
            elif target_phase == 'exploring':
                stabilizing_score = self._clamp(0.34 + confidence * 0.12)
                committed_rate = self._clamp(0.06 + confidence * 0.08)
                risk_score = self._clamp(0.22 + (1.0 - confidence) * 0.16)
            else:
                stabilizing_score = self._clamp(0.16 + confidence * 0.08)
                committed_rate = self._clamp(0.04 + confidence * 0.06)
                risk_score = self._clamp(0.54 + confidence * 0.24)
            return {
                'support': round(max(support, 1.0 + float(branch.get('support', 0.0) or 0.0) * 2.0), 4),
                'success_rate': round(max(0.48, success_rate * 0.72 + 0.24), 4),
                'avg_reward': round(max(avg_reward, 0.12 + confidence * 0.40), 4),
                'avg_depth_gain': round(max(avg_depth_gain, 0.10 + confidence * 0.26), 4),
                'stabilizing_rate': round(stabilizing_score, 4),
                'committed_rate': round(committed_rate, 4),
                'disrupted_rate': round(self._clamp(risk_score * 0.48), 4),
                'stabilizing_score': round(stabilizing_score, 4),
                'risk_score': round(risk_score, 4),
            }

        risk_score = self._clamp(0.72 + confidence * 0.18)
        stabilizing_score = self._clamp(0.12 + (1.0 - confidence) * 0.10)
        return {
            'support': round(max(1.0, 1.0 + float(branch.get('support', 0.0) or 0.0)), 4),
            'success_rate': round(max(0.08, 0.34 - confidence * 0.12), 4),
            'avg_reward': round(min(avg_reward - 0.24, -0.10), 4),
            'avg_depth_gain': round(min(avg_depth_gain, 0.10 + (1.0 - confidence) * 0.08), 4),
            'stabilizing_rate': round(stabilizing_score, 4),
            'committed_rate': round(self._clamp(0.04 + (1.0 - confidence) * 0.05), 4),
            'disrupted_rate': round(self._clamp(0.66 + confidence * 0.20), 4),
            'stabilizing_score': round(stabilizing_score, 4),
            'risk_score': round(risk_score, 4),
        }

    def _merge_rollout_branch_phase_scores(
        self,
        *,
        phase_scores: Dict[str, Dict[str, float]],
        rollout_branch: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        merged = {
            str(name): dict(payload)
            for name, payload in phase_scores.items()
            if isinstance(payload, dict)
        }
        if not rollout_branch:
            return merged

        confidence = self._clamp(rollout_branch.get('confidence', 0.0), 0.0, 1.0)
        for idx, fn_name in enumerate(list(rollout_branch.get('anchor_functions', []) or [])[:4]):
            if not str(fn_name or '').strip():
                continue
            base = dict(merged.get(fn_name, {}) or {})
            template = self._branch_phase_score_template(branch=rollout_branch, role='anchor')
            anchor_discount = idx * 0.04
            base['support'] = round(max(float(base.get('support', 0.0) or 0.0), float(template.get('support', 1.0) or 1.0) - anchor_discount), 4)
            base['success_rate'] = round(max(float(base.get('success_rate', 0.0) or 0.0), float(template.get('success_rate', 0.5) or 0.5) - anchor_discount * 0.5), 4)
            base['avg_reward'] = round(max(float(base.get('avg_reward', 0.0) or 0.0), float(template.get('avg_reward', 0.0) or 0.0) - anchor_discount * 0.12), 4)
            base['avg_depth_gain'] = round(max(float(base.get('avg_depth_gain', 0.0) or 0.0), float(template.get('avg_depth_gain', 0.0) or 0.0) - anchor_discount * 0.3), 4)
            base['stabilizing_rate'] = round(max(float(base.get('stabilizing_rate', 0.0) or 0.0), float(template.get('stabilizing_rate', 0.0) or 0.0) - anchor_discount), 4)
            base['committed_rate'] = round(max(float(base.get('committed_rate', 0.0) or 0.0), float(template.get('committed_rate', 0.0) or 0.0) - anchor_discount), 4)
            base['disrupted_rate'] = round(min(float(base.get('disrupted_rate', 1.0) or 1.0), float(template.get('disrupted_rate', 0.0) or 0.0) + anchor_discount), 4)
            base['stabilizing_score'] = round(max(float(base.get('stabilizing_score', 0.0) or 0.0), float(template.get('stabilizing_score', 0.0) or 0.0) - anchor_discount), 4)
            base['risk_score'] = round(
                min(
                    max(0.0, float(base.get('risk_score', 1.0) or 1.0) - confidence * 0.08),
                    float(template.get('risk_score', 1.0) or 1.0) + anchor_discount,
                ),
                4,
            )
            merged[str(fn_name)] = base

        for fn_name in list(rollout_branch.get('risky_functions', []) or [])[:4]:
            if not str(fn_name or '').strip():
                continue
            base = dict(merged.get(fn_name, {}) or {})
            template = self._branch_phase_score_template(branch=rollout_branch, role='risky')
            base['support'] = round(max(float(base.get('support', 0.0) or 0.0), float(template.get('support', 1.0) or 1.0)), 4)
            base['success_rate'] = round(min(float(base.get('success_rate', 1.0) or 1.0), float(template.get('success_rate', 0.5) or 0.5)), 4)
            base['avg_reward'] = round(min(float(base.get('avg_reward', 0.0) or 0.0), float(template.get('avg_reward', 0.0) or 0.0)), 4)
            base['avg_depth_gain'] = round(min(float(base.get('avg_depth_gain', 1.0) or 1.0), float(template.get('avg_depth_gain', 0.0) or 0.0)), 4)
            base['stabilizing_rate'] = round(min(float(base.get('stabilizing_rate', 1.0) or 1.0), float(template.get('stabilizing_rate', 0.0) or 0.0)), 4)
            base['committed_rate'] = round(min(float(base.get('committed_rate', 1.0) or 1.0), float(template.get('committed_rate', 0.0) or 0.0)), 4)
            base['disrupted_rate'] = round(max(float(base.get('disrupted_rate', 0.0) or 0.0), float(template.get('disrupted_rate', 0.0) or 0.0)), 4)
            base['stabilizing_score'] = round(min(float(base.get('stabilizing_score', 1.0) or 1.0), float(template.get('stabilizing_score', 0.0) or 0.0)), 4)
            base['risk_score'] = round(max(float(base.get('risk_score', 0.0) or 0.0), float(template.get('risk_score', 0.0) or 0.0)), 4)
            merged[str(fn_name)] = base
        return merged

    def _branch_followup_function(
        self,
        *,
        rollout_branch: Dict[str, Any],
        current_function: str,
        phase_scores: Dict[str, Dict[str, float]],
        used_functions: List[str],
    ) -> str:
        if not rollout_branch:
            return ''
        anchor_functions = [
            str(value or '').strip()
            for value in list(rollout_branch.get('anchor_functions', []) or [])[:4]
            if str(value or '').strip()
        ]
        risky_functions = {
            str(value or '').strip()
            for value in list(rollout_branch.get('risky_functions', []) or [])[:4]
            if str(value or '').strip()
        }
        if not anchor_functions or current_function in risky_functions:
            return ''

        start_idx = anchor_functions.index(current_function) + 1 if current_function in anchor_functions else 0
        for fn_name in anchor_functions[start_idx:]:
            if fn_name in risky_functions or fn_name in used_functions:
                continue
            if fn_name == current_function:
                continue
            if fn_name in phase_scores:
                return fn_name
        for fn_name in anchor_functions:
            if fn_name in risky_functions:
                continue
            if fn_name not in used_functions and fn_name in phase_scores:
                return fn_name
        return ''

    def _branch_guided_next_phase(
        self,
        *,
        base_next_phase: str,
        rollout_branch: Dict[str, Any],
        current_function: str,
        step_index: int,
    ) -> str:
        if not rollout_branch:
            return base_next_phase
        target_phase = self._phase_alias(rollout_branch.get('target_phase', '')) or base_next_phase
        branch_confidence = self._clamp(rollout_branch.get('confidence', 0.0), 0.0, 1.0)
        anchor_functions = [
            str(value or '').strip()
            for value in list(rollout_branch.get('anchor_functions', []) or [])[:4]
            if str(value or '').strip()
        ]
        risky_functions = {
            str(value or '').strip()
            for value in list(rollout_branch.get('risky_functions', []) or [])[:4]
            if str(value or '').strip()
        }
        anchor_index = anchor_functions.index(current_function) if current_function in anchor_functions else -1

        if current_function in risky_functions and target_phase in {'stabilizing', 'committed'}:
            return 'disrupted'
        if current_function in risky_functions and target_phase == 'exploring':
            return 'exploring'
        if anchor_index >= 0:
            if target_phase == 'committed':
                if anchor_index >= max(0, len(anchor_functions) - 2) or step_index >= 1 or branch_confidence >= 0.84:
                    return 'committed'
                return 'stabilizing'
            if target_phase == 'stabilizing':
                return 'stabilizing'
            if target_phase in {'exploring', 'disrupted'}:
                return target_phase
        if target_phase == 'committed' and base_next_phase == 'disrupted' and branch_confidence >= 0.72:
            return 'stabilizing'
        if target_phase == 'stabilizing' and base_next_phase == 'exploring' and branch_confidence >= 0.66:
            return 'stabilizing'
        return base_next_phase

    def _best_followup_function(
        self,
        *,
        phase_scores: Dict[str, Dict[str, float]],
        exclude: List[str],
    ) -> str:
        blocked = {str(value or '') for value in exclude if str(value or '')}
        ranked = [
            (name, payload)
            for name, payload in phase_scores.items()
            if name and name not in blocked and isinstance(payload, dict)
        ]
        if not ranked:
            return 'wait'
        ranked.sort(
            key=lambda item: (
                -float(item[1].get('stabilizing_score', 0.0) or 0.0),
                float(item[1].get('risk_score', 0.0) or 0.0),
                -float(item[1].get('support', 0.0) or 0.0),
                item[0],
            )
        )
        return str(ranked[0][0] or 'wait')

    def _simulate_hidden_state_rollout(
        self,
        *,
        state_slice: StateSlice,
        action: Dict[str, Any],
        context: Dict[str, Any],
        horizon: int = 3,
    ) -> Optional[Dict[str, Any]]:
        context = context if isinstance(context, dict) else {}
        priors = context.get('transition_priors', {})
        priors = priors if isinstance(priors, dict) else {}
        world_dynamics = priors.get('__world_dynamics', {}) if isinstance(priors.get('__world_dynamics', {}), dict) else {}
        hidden_state = world_dynamics.get('hidden_state', {}) if isinstance(world_dynamics.get('hidden_state', {}), dict) else {}
        if not hidden_state:
            state_features = state_slice.state_features if isinstance(state_slice.state_features, dict) else {}
            hidden_state = state_features.get('hidden_state', {}) if isinstance(state_features.get('hidden_state', {}), dict) else {}
        transition_memory = world_dynamics.get('transition_memory', {}) if isinstance(world_dynamics.get('transition_memory', {}), dict) else {}
        if not transition_memory:
            transition_memory = hidden_state.get('transition_memory', {}) if isinstance(hidden_state.get('transition_memory', {}), dict) else {}
        if not transition_memory and not hidden_state:
            return None

        current_phase = self._rollout_current_phase(
            state_slice=state_slice,
            world_dynamics=world_dynamics,
            hidden_state=hidden_state,
        )
        expected_next_phase = self._phase_alias(
            hidden_state.get('expected_next_phase', transition_memory.get('expected_next_phase', world_dynamics.get('expected_next_phase', '')))
        )
        expected_next_phase_confidence = self._clamp(
            hidden_state.get(
                'expected_next_phase_confidence',
                transition_memory.get('expected_next_phase_confidence', world_dynamics.get('expected_next_phase_confidence', 0.0)),
            ),
            0.0,
            1.0,
        )
        transition_entropy = self._clamp(
            hidden_state.get(
                'transition_entropy',
                transition_memory.get('phase_transition_entropy', world_dynamics.get('phase_transition_entropy', 1.0)),
            ),
            0.0,
            1.0,
            1.0,
        )
        transition_confidence = self._clamp(world_dynamics.get('transition_confidence', 0.0), 0.0, 1.0)
        fn_name = self._function_name(action)
        latent_branches, dominant_branch_id = self._rollout_latent_branch_state(
            hidden_state=hidden_state,
            transition_memory=transition_memory,
            world_dynamics=world_dynamics,
        )
        rollout_branch = self._select_rollout_branch(
            latent_branches=latent_branches,
            dominant_branch_id=dominant_branch_id,
            fn_name=fn_name,
            current_phase=current_phase,
            expected_next_phase=expected_next_phase,
            expected_next_phase_confidence=expected_next_phase_confidence,
        )
        phase_scores = self._merge_rollout_branch_phase_scores(
            phase_scores=self._transition_phase_function_scores(
                context=context,
                phase=current_phase,
                transition_memory=transition_memory,
            ),
            rollout_branch=rollout_branch,
        )
        if fn_name not in phase_scores and fn_name != 'wait':
            legacy = priors.get(fn_name, {}) if isinstance(priors.get(fn_name, {}), dict) else {}
            if legacy:
                phase_scores[fn_name] = {
                    'support': 1.0,
                    'success_rate': self._clamp((float(legacy.get('long_horizon_reward', 0.0) or 0.0) + 1.0) / 2.0),
                    'avg_reward': float(legacy.get('long_horizon_reward', 0.0) or 0.0),
                    'avg_depth_gain': self._clamp(float(legacy.get('info_gain', 0.0) or 0.0)),
                    'stabilizing_rate': self._clamp(float(legacy.get('reversibility', 0.0) or 0.0)),
                    'committed_rate': self._clamp(max(0.0, float(legacy.get('transition_affinity', 0.0) or 0.0))),
                    'disrupted_rate': self._clamp(float(legacy.get('predicted_risk', 0.0) or 0.0)),
                    'stabilizing_score': self._clamp(
                        self._clamp((float(legacy.get('long_horizon_reward', 0.0) or 0.0) + 1.0) / 2.0) * 0.45
                        + self._clamp(float(legacy.get('reversibility', 0.0) or 0.0)) * 0.25
                        - self._clamp(float(legacy.get('predicted_risk', 0.0) or 0.0)) * 0.15
                    ),
                    'risk_score': self._clamp(
                        self._clamp(float(legacy.get('predicted_risk', 0.0) or 0.0)) * 0.6
                        + self._clamp(float(legacy.get('constraint_violation', 0.0) or 0.0)) * 0.3
                    ),
                }
        phase_scores = self._merge_rollout_branch_phase_scores(
            phase_scores=phase_scores,
            rollout_branch=rollout_branch,
        )

        trace: List[Dict[str, Any]] = []
        total_value = 0.0
        total_risk = 0.0
        used_functions = [fn_name]
        phase = current_phase
        current_function = fn_name
        branch_guided_steps = 0.0
        branch_id = str(rollout_branch.get('branch_id', '') or '')
        branch_target_phase = str(rollout_branch.get('target_phase', '') or '')
        branch_confidence = self._clamp(rollout_branch.get('confidence', 0.0), 0.0, 1.0)
        branch_anchor_functions = [
            str(value or '').strip()
            for value in list(rollout_branch.get('anchor_functions', []) or [])[:4]
            if str(value or '').strip()
        ]
        branch_risky_functions = {
            str(value or '').strip()
            for value in list(rollout_branch.get('risky_functions', []) or [])[:4]
            if str(value or '').strip()
        }

        for step_idx in range(max(1, int(horizon or 1))):
            phase_scores = self._merge_rollout_branch_phase_scores(
                phase_scores=self._transition_phase_function_scores(
                    context=context,
                    phase=phase,
                    transition_memory=transition_memory,
                ),
                rollout_branch=rollout_branch,
            )
            fn_stats = phase_scores.get(current_function, {}) if isinstance(phase_scores.get(current_function, {}), dict) else {}
            support = self._clamp(float(fn_stats.get('support', 0.0) or 0.0) / 3.0, 0.0, 1.0)
            stabilizing_score = self._clamp(fn_stats.get('stabilizing_score', 0.0))
            risk_score = self._clamp(fn_stats.get('risk_score', 0.0))
            avg_reward = float(fn_stats.get('avg_reward', 0.0) or 0.0)
            avg_depth_gain = self._clamp(fn_stats.get('avg_depth_gain', 0.0))
            anchor_match = bool(current_function and current_function in branch_anchor_functions)
            risky_match = bool(current_function and current_function in branch_risky_functions)
            anchor_index = branch_anchor_functions.index(current_function) if anchor_match else -1
            branch_alignment = 0.0
            if rollout_branch:
                branch_alignment += branch_confidence * 0.28
                if anchor_match:
                    branch_alignment += 0.42 - max(0.0, anchor_index) * 0.06
                if risky_match:
                    branch_alignment -= 0.56

            phase_bonus = 0.18 if phase == 'committed' else 0.08 if phase == 'stabilizing' else -0.18 if phase == 'disrupted' else 0.0
            step_value = (
                avg_reward * 0.44
                + stabilizing_score * 0.18
                + avg_depth_gain * 0.14
                + phase_bonus
                + support * 0.10
                - risk_score * 0.32
            )
            step_risk = self._clamp(risk_score * 0.62 + transition_entropy * 0.12 + (0.10 if phase == 'disrupted' else 0.0), 0.0, 1.0)
            if rollout_branch:
                if anchor_match:
                    step_value += 0.10 + branch_confidence * 0.16 - max(0.0, anchor_index) * 0.02
                    step_risk = self._clamp(step_risk - (0.08 + branch_confidence * 0.10), 0.0, 1.0)
                if risky_match:
                    step_value -= 0.18 + branch_confidence * 0.14
                    step_risk = self._clamp(step_risk + 0.16 + branch_confidence * 0.12, 0.0, 1.0)
                if branch_target_phase == 'committed' and anchor_match:
                    step_value += 0.04 + branch_confidence * 0.08
                elif branch_target_phase == 'disrupted' and risky_match:
                    step_value += 0.02 + branch_confidence * 0.04

            base_next_phase = self._phase_path_next_phase(
                current_phase=phase,
                expected_next_phase=expected_next_phase,
                expected_next_phase_confidence=expected_next_phase_confidence,
                transition_entropy=transition_entropy,
                fn_stats=fn_stats,
                step_index=step_idx,
            )
            next_phase = self._branch_guided_next_phase(
                base_next_phase=base_next_phase,
                rollout_branch=rollout_branch,
                current_function=current_function,
                step_index=step_idx,
            )
            if rollout_branch and next_phase == branch_target_phase and not risky_match:
                branch_alignment += 0.16
            branch_memory_retained = bool(rollout_branch) and branch_alignment > 0.0 and not risky_match
            if branch_memory_retained:
                branch_guided_steps += 1.0

            discount = 0.74 ** step_idx
            total_value += step_value * discount
            total_risk += step_risk * discount
            trace.append(
                {
                    'step': step_idx,
                    'phase': phase,
                    'action': current_function,
                    'next_phase': next_phase,
                    'support': round(support, 4),
                    'stabilizing_score': round(stabilizing_score, 4),
                    'risk_score': round(risk_score, 4),
                    'step_value': round(step_value, 4),
                    'step_risk': round(step_risk, 4),
                    'branch_id': branch_id,
                    'branch_target_phase': branch_target_phase,
                    'branch_confidence': round(branch_confidence, 4),
                    'branch_anchor_index': anchor_index,
                    'anchor_match': anchor_match,
                    'risky_match': risky_match,
                    'branch_alignment': round(branch_alignment, 4),
                    'branch_memory_retained': branch_memory_retained,
                }
            )
            phase = next_phase
            if step_idx >= horizon - 1:
                continue
            next_phase_scores = self._merge_rollout_branch_phase_scores(
                phase_scores=self._transition_phase_function_scores(
                    context=context,
                    phase=phase,
                    transition_memory=transition_memory,
                ),
                rollout_branch=rollout_branch,
            )
            branch_followup = self._branch_followup_function(
                rollout_branch=rollout_branch,
                current_function=current_function,
                phase_scores=next_phase_scores,
                used_functions=used_functions if phase != 'committed' else [],
            )
            current_function = branch_followup or self._best_followup_function(
                phase_scores=next_phase_scores,
                exclude=used_functions if phase != 'committed' else [],
            )
            used_functions.append(current_function)

        rollout_confidence = self._clamp(
            transition_confidence * 0.34
            + expected_next_phase_confidence * 0.28
            + (1.0 - transition_entropy) * 0.16
            + (trace[0]['support'] if trace else 0.0) * 0.12
            + (0.10 if len(trace) >= 2 else 0.0)
            + branch_confidence * 0.10,
            0.0,
            1.0,
        )
        return {
            'action': fn_name,
            'value': round(total_value, 4),
            'risk': round(total_risk, 4),
            'confidence': round(rollout_confidence, 4),
            'phase_path': [trace[0]['phase']] + [row['next_phase'] for row in trace] if trace else [current_phase],
            'trace': trace,
            'horizon': int(horizon),
            'rollout_branch_id': branch_id,
            'rollout_branch_target_phase': branch_target_phase,
            'rollout_branch_confidence': round(branch_confidence, 4),
            'branch_persistence_ratio': round(branch_guided_steps / max(len(trace), 1), 4),
            'anchor_path': list(branch_anchor_functions),
        }

    def _expected_phase_from_text(self, text: str) -> str:
        lowered = str(text or '').strip().lower()
        if not lowered:
            return ''
        if any(token in lowered for token in ('disrupt', 'fail', 'error', 'collapse', 'break', 'drift', 'rupture')):
            return 'disrupted'
        if any(token in lowered for token in ('commit', 'seal', 'solve', 'complete', 'resolved', 'ready')):
            return 'committed'
        if any(token in lowered for token in ('stabil', 'align', 'warm', 'settle', 'prepare')):
            return 'stabilizing'
        if any(token in lowered for token in ('explor', 'probe', 'scan', 'search', 'inspect')):
            return 'exploring'
        return ''

    def _relevant_hypothesis_branches(
        self,
        *,
        state_slice: StateSlice,
        action_name: str,
        context: Dict[str, Any],
        current_phase: str,
    ) -> List[Dict[str, Any]]:
        hypotheses = list(state_slice.active_hypotheses or [])
        if not hypotheses:
            return []
        action_token = str(action_name or '').strip().lower()
        phase_token = self._phase_alias(current_phase) or 'exploring'
        state_features = state_slice.state_features if isinstance(state_slice.state_features, dict) else {}
        feature_blob = str(state_features).lower()
        available_functions = [str(fn or '').strip() for fn in list(state_slice.available_functions or []) if str(fn or '').strip()]
        branches: List[Dict[str, Any]] = []

        for hyp in hypotheses:
            if not isinstance(hyp, dict):
                continue
            claim = str(hyp.get('claim', '') or '')
            trigger = str(hyp.get('trigger_condition', '') or '')
            transition = str(hyp.get('expected_transition', '') or '')
            text = f"{claim} {trigger} {transition}".strip().lower()
            if not text:
                continue
            confidence = self._clamp(hyp.get('confidence', 0.0), 0.0, 1.0)
            fn_match = bool(action_token and action_token in text)
            phase_match = bool(phase_token and phase_token in text)
            feature_match = any(token and token in feature_blob for token in text.split()[:6])
            expected_phase = self._expected_phase_from_text(transition or claim or trigger)
            branch_functions = [fn for fn in available_functions if fn.lower() in text]
            action_linked = bool(fn_match or any(str(fn).lower() == action_token for fn in branch_functions))
            function_specific = bool(fn_match or branch_functions)
            if action_linked and action_name not in branch_functions:
                branch_functions.append(action_name)
            relevance = confidence * 0.46
            if fn_match:
                relevance += 0.28
            if phase_match:
                relevance += 0.10
            if feature_match:
                relevance += 0.08
            if expected_phase:
                relevance += 0.12
            if function_specific and not action_linked:
                relevance -= 0.10
            relevance = self._clamp(relevance, 0.0, 1.0)
            if relevance < 0.42:
                continue
            weight = min(0.55, confidence * max(0.35, relevance))
            branches.append(
                {
                    'hypothesis_id': str(hyp.get('id', '') or ''),
                    'claim': claim,
                    'trigger_condition': trigger,
                    'expected_transition': transition,
                    'expected_phase': expected_phase or phase_token,
                    'confidence': confidence,
                    'relevance': relevance,
                    'weight': weight,
                    'branch_functions': branch_functions,
                    'function_specific': function_specific,
                    'action_linked': action_linked,
                }
            )
        branches.sort(key=lambda item: (-float(item['weight']), -float(item['relevance']), item['hypothesis_id']))
        return branches[:3]

    def _hypothesis_branch_context(
        self,
        *,
        state_slice: StateSlice,
        action_name: str,
        context: Dict[str, Any],
        branch: Dict[str, Any],
        current_phase: str,
    ) -> Dict[str, Any]:
        ctx = dict(context or {})
        priors = dict(ctx.get('transition_priors', {}) or {})
        world_dynamics = dict(priors.get('__world_dynamics', {}) or {})
        hidden_state = dict(world_dynamics.get('hidden_state', {}) or {})
        transition_memory = dict(world_dynamics.get('transition_memory', {}) or {})
        if not hidden_state:
            state_features = state_slice.state_features if isinstance(state_slice.state_features, dict) else {}
            hidden_state = dict(state_features.get('hidden_state', {}) or {})
        if not transition_memory:
            transition_memory = dict(hidden_state.get('transition_memory', {}) or {})

        expected_phase = self._phase_alias(branch.get('expected_phase', '')) or (self._phase_alias(current_phase) or 'exploring')
        action_linked = bool(branch.get('action_linked', False))
        function_specific = bool(branch.get('function_specific', False))
        apply_phase_override = action_linked or not function_specific
        branch_conf = self._clamp(branch.get('confidence', 0.0), 0.0, 1.0)
        branch_weight = self._clamp(branch.get('weight', 0.0), 0.0, 1.0)
        branch_strength = self._clamp(branch_conf * max(0.5, branch_weight), 0.0, 1.0)
        transition_entropy = self._clamp(
            hidden_state.get('transition_entropy', transition_memory.get('phase_transition_entropy', world_dynamics.get('phase_transition_entropy', 1.0))),
            0.0,
            1.0,
            1.0,
        )

        if apply_phase_override:
            hidden_state['expected_next_phase'] = expected_phase
            hidden_state['expected_next_phase_confidence'] = max(
                self._clamp(hidden_state.get('expected_next_phase_confidence', 0.0), 0.0, 1.0),
                round(branch_strength, 4),
            )
            hidden_state['transition_entropy'] = self._clamp(
                transition_entropy - (0.18 * branch_strength) if expected_phase in {'committed', 'stabilizing'} else transition_entropy + (0.14 * branch_strength),
                0.0,
                1.0,
            )

        transition_memory['current_phase'] = self._phase_alias(current_phase) or 'exploring'
        transition_memory['expected_next_phase'] = str(hidden_state.get('expected_next_phase', expected_phase) or expected_phase)
        transition_memory['expected_next_phase_confidence'] = float(
            hidden_state.get('expected_next_phase_confidence', transition_memory.get('expected_next_phase_confidence', 0.0)) or 0.0
        )
        transition_memory['phase_transition_entropy'] = float(
            hidden_state.get('transition_entropy', transition_memory.get('phase_transition_entropy', transition_entropy)) or transition_entropy
        )
        phase_scores = transition_memory.get('phase_function_scores', {})
        phase_scores = dict(phase_scores) if isinstance(phase_scores, dict) else {}
        branch_functions = [str(fn or '').strip() for fn in list(branch.get('branch_functions', []) or []) if str(fn or '').strip()]
        if action_linked and action_name not in branch_functions:
            branch_functions.append(action_name)
        for fn_name in branch_functions:
            payload = dict(phase_scores.get(fn_name, {}) or {})
            payload.setdefault('support', 1.0)
            payload.setdefault('success_rate', 0.5)
            payload.setdefault('avg_reward', 0.0)
            payload.setdefault('avg_depth_gain', 0.0)
            payload.setdefault('stabilizing_rate', 0.5)
            payload.setdefault('committed_rate', 0.0)
            payload.setdefault('disrupted_rate', 0.0)
            payload.setdefault('stabilizing_score', 0.5)
            payload.setdefault('risk_score', 0.3)

            if expected_phase in {'committed', 'stabilizing'}:
                payload['stabilizing_score'] = round(self._clamp(float(payload.get('stabilizing_score', 0.0) or 0.0) + 0.22 * branch_strength), 4)
                payload['risk_score'] = round(self._clamp(float(payload.get('risk_score', 0.0) or 0.0) - 0.18 * branch_strength), 4)
                payload['committed_rate'] = round(self._clamp(float(payload.get('committed_rate', 0.0) or 0.0) + (0.20 * branch_strength if expected_phase == 'committed' else 0.08 * branch_strength)), 4)
                payload['avg_reward'] = round(float(payload.get('avg_reward', 0.0) or 0.0) + 0.35 * branch_strength, 4)
                payload['avg_depth_gain'] = round(self._clamp(float(payload.get('avg_depth_gain', 0.0) or 0.0) + 0.12 * branch_strength), 4)
            elif expected_phase == 'disrupted':
                payload['stabilizing_score'] = round(self._clamp(float(payload.get('stabilizing_score', 0.0) or 0.0) - 0.24 * branch_strength), 4)
                payload['risk_score'] = round(self._clamp(float(payload.get('risk_score', 0.0) or 0.0) + 0.26 * branch_strength), 4)
                payload['disrupted_rate'] = round(self._clamp(float(payload.get('disrupted_rate', 0.0) or 0.0) + 0.22 * branch_strength), 4)
                payload['avg_reward'] = round(float(payload.get('avg_reward', 0.0) or 0.0) - 0.38 * branch_strength, 4)
            else:
                payload['avg_depth_gain'] = round(self._clamp(float(payload.get('avg_depth_gain', 0.0) or 0.0) + 0.18 * branch_strength), 4)
                payload['stabilizing_score'] = round(self._clamp(float(payload.get('stabilizing_score', 0.0) or 0.0) + 0.05 * branch_strength), 4)
            payload['support'] = round(float(payload.get('support', 1.0) or 1.0) + branch_strength, 4)
            phase_scores[fn_name] = payload

        transition_memory['phase_function_scores'] = phase_scores
        hidden_state['transition_memory'] = transition_memory
        world_dynamics['hidden_state'] = hidden_state
        world_dynamics['transition_memory'] = transition_memory
        world_dynamics['expected_next_phase'] = str(hidden_state.get('expected_next_phase', expected_phase) or expected_phase)
        world_dynamics['expected_next_phase_confidence'] = float(
            hidden_state.get('expected_next_phase_confidence', world_dynamics.get('expected_next_phase_confidence', 0.0)) or 0.0
        )
        world_dynamics['phase_transition_entropy'] = float(
            hidden_state.get('transition_entropy', world_dynamics.get('phase_transition_entropy', transition_entropy)) or transition_entropy
        )
        priors['__world_dynamics'] = world_dynamics
        ctx['transition_priors'] = priors
        return ctx

    def _simulate_hidden_hypothesis_branch_rollout(
        self,
        *,
        state_slice: StateSlice,
        action: Dict[str, Any],
        context: Dict[str, Any],
        horizon: int = 3,
    ) -> Optional[Dict[str, Any]]:
        base_rollout = self._simulate_hidden_state_rollout(
            state_slice=state_slice,
            action=action,
            context=context,
            horizon=horizon,
        )
        if not base_rollout:
            return None
        action_name = self._function_name(action)
        phase_path = list(base_rollout.get('phase_path', []) or [])
        current_phase = str(phase_path[0] if phase_path else 'exploring')
        branches = self._relevant_hypothesis_branches(
            state_slice=state_slice,
            action_name=action_name,
            context=context,
            current_phase=current_phase,
        )
        if not branches:
            return base_rollout

        branch_rollouts: List[Dict[str, Any]] = []
        branch_weight_total = 0.0
        weighted_value = float(base_rollout.get('value', 0.0) or 0.0)
        weighted_risk = float(base_rollout.get('risk', 0.0) or 0.0)
        weighted_confidence = float(base_rollout.get('confidence', 0.0) or 0.0)
        residual_weight = 1.0

        for branch in branches:
            branch_weight = self._clamp(branch.get('weight', 0.0), 0.0, 0.55)
            if branch_weight <= 0.0:
                continue
            residual_weight = max(0.15, residual_weight - branch_weight)
            branch_context = self._hypothesis_branch_context(
                state_slice=state_slice,
                action_name=action_name,
                context=context,
                branch=branch,
                current_phase=current_phase,
            )
            branch_rollout = self._simulate_hidden_state_rollout(
                state_slice=state_slice,
                action=action,
                context=branch_context,
                horizon=horizon,
            )
            if not branch_rollout:
                continue
            deferred_trigger_penalty = 0.0
            if bool(branch.get('function_specific', False)) and not bool(branch.get('action_linked', False)):
                deferred_trigger_penalty = round(
                    0.18 * self._clamp(max(float(branch.get('weight', 0.0) or 0.0), float(branch.get('confidence', 0.0) or 0.0) * 0.5)),
                    4,
                )
                branch_rollout = dict(branch_rollout)
                branch_rollout['value'] = round(float(branch_rollout.get('value', 0.0) or 0.0) - deferred_trigger_penalty, 4)
                branch_rollout['risk'] = round(
                    self._clamp(float(branch_rollout.get('risk', 0.0) or 0.0) + deferred_trigger_penalty * 0.6, 0.0, 1.0),
                    4,
                )
                branch_rollout['deferred_trigger_penalty'] = deferred_trigger_penalty
            branch_rollouts.append(
                {
                    'hypothesis_id': branch.get('hypothesis_id', ''),
                    'claim': branch.get('claim', ''),
                    'expected_phase': branch.get('expected_phase', ''),
                    'action_linked': bool(branch.get('action_linked', False)),
                    'deferred_trigger_penalty': deferred_trigger_penalty,
                    'weight': round(branch_weight, 4),
                    'relevance': round(float(branch.get('relevance', 0.0) or 0.0), 4),
                    'rollout': branch_rollout,
                }
            )
            branch_weight_total += branch_weight

        if not branch_rollouts:
            return base_rollout

        weighted_value = float(base_rollout.get('value', 0.0) or 0.0) * residual_weight
        weighted_risk = float(base_rollout.get('risk', 0.0) or 0.0) * residual_weight
        weighted_confidence = float(base_rollout.get('confidence', 0.0) or 0.0) * residual_weight
        total_weight = residual_weight
        for branch in branch_rollouts:
            weight = float(branch.get('weight', 0.0) or 0.0)
            rollout = branch.get('rollout', {}) if isinstance(branch.get('rollout', {}), dict) else {}
            weighted_value += float(rollout.get('value', 0.0) or 0.0) * weight
            weighted_risk += float(rollout.get('risk', 0.0) or 0.0) * weight
            weighted_confidence += float(rollout.get('confidence', 0.0) or 0.0) * weight
            total_weight += weight
        aggregate = dict(base_rollout)
        aggregate['value'] = round(weighted_value / max(total_weight, 1e-6), 4)
        aggregate['risk'] = round(weighted_risk / max(total_weight, 1e-6), 4)
        aggregate['confidence'] = round(weighted_confidence / max(total_weight, 1e-6), 4)
        aggregate['branch_rollups'] = branch_rollouts
        aggregate['branch_weight_total'] = round(branch_weight_total, 4)
        aggregate['branch_residual_weight'] = round(residual_weight, 4)
        strongest_branch = max(branch_rollouts, key=lambda item: float(item.get('weight', 0.0) or 0.0))
        aggregate['phase_path'] = list((strongest_branch.get('rollout', {}) if isinstance(strongest_branch.get('rollout', {}), dict) else {}).get('phase_path', base_rollout.get('phase_path', [])) or base_rollout.get('phase_path', []))
        aggregate['trace'] = list(base_rollout.get('trace', []))
        return aggregate
    
    def simulate_action_difference(
        self,
        state_slice: StateSlice,
        action_a: Dict[str, Any],
        action_b: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> CounterfactualOutcome:
        """
        Compare predicted outcomes of two actions.
        
        Args:
            state_slice: Current state snapshot
            action_a: First action candidate
            action_b: Second action candidate
            context: Optional context (e.g., goal, urgency)
        
        Returns:
            CounterfactualOutcome with comparison prediction
        """
        fn_a = self._function_name(action_a)
        fn_b = self._function_name(action_b)
        
        # Determine confidence based on established beliefs
        confidence = self._assess_confidence(state_slice, [fn_a, fn_b])
        rollout_summary: Dict[str, Any] = {}
        rollout_trace: List[Dict[str, Any]] = []
        mechanism_preferred, mechanism_reasoning, mechanism_match = self._compare_actions_via_mechanism(
            state_slice=state_slice,
            fn_a=fn_a,
            fn_b=fn_b,
            context=context,
        )
        if mechanism_preferred:
            preferred, reasoning = mechanism_preferred, mechanism_reasoning
            confidence = self._confidence_from_mechanism(mechanism_match)
            decision_path = "mechanism_match"
        else:
            mixed_world_model = context.get('world_model_summary', {}) if isinstance(context, dict) and isinstance(context.get('world_model_summary', {}), dict) else {}
            if mixed_world_model and (
                mixed_world_model.get('predicted_transitions')
                or mixed_world_model.get('candidate_intervention_targets')
                or mixed_world_model.get('mechanism_hypotheses')
            ):
                mixed_rollout = compare_function_rollouts(
                    fn_a,
                    fn_b,
                    world_model_summary=mixed_world_model,
                    transition_priors=context.get('transition_priors', {}) if isinstance(context, dict) else {},
                    candidate_intervention_targets=mixed_world_model.get('candidate_intervention_targets', []),
                    mechanism_hypotheses=mixed_world_model.get('mechanism_hypotheses', []),
                )
                rollout_delta = float(mixed_rollout.get('estimated_delta', 0.0) or 0.0)
                if abs(rollout_delta) >= 0.05:
                    preferred = str(mixed_rollout.get('preferred_action', fn_a) or fn_a)
                    winning_rollout = mixed_rollout.get('action_a', {}) if preferred == fn_a else mixed_rollout.get('action_b', {})
                    losing_rollout = mixed_rollout.get('action_b', {}) if preferred == fn_a else mixed_rollout.get('action_a', {})
                    reasoning = (
                        f"Mixed world-model rollout prefers {preferred}: phase path "
                        f"{' -> '.join(list(winning_rollout.get('phase_path', []) or []))} with "
                        f"reward {float(winning_rollout.get('expected_reward', 0.0) or 0.0):.2f}, "
                        f"info gain {float(winning_rollout.get('expected_information_gain', 0.0) or 0.0):.2f}, "
                        f"risk {float(winning_rollout.get('state_shift_risk', 0.0) or 0.0):.2f}."
                    )
                    decision_path = "mixed_world_model_rollout"
                    confidence = (
                        CounterfactualConfidence.HIGH
                        if max(float(winning_rollout.get('confidence', 0.0) or 0.0), float(losing_rollout.get('confidence', 0.0) or 0.0)) >= 0.72
                        else CounterfactualConfidence.MEDIUM
                    )
                    rollout_summary = {
                        'preferred_action': preferred,
                        'rollout_delta': round(rollout_delta, 4),
                        'action_a': mixed_rollout.get('action_a', {}),
                        'action_b': mixed_rollout.get('action_b', {}),
                        'mixed_world_model_rollout': True,
                    }
                    rollout_trace = [
                        {
                            'candidate': 'action_a',
                            'step': 'mixed_rollout',
                            'next_phase': str((_as := mixed_rollout.get('action_a', {})).get('to_phase', '') or ''),
                            'risk': round(float(_as.get('state_shift_risk', 0.0) or 0.0), 4),
                            'info_gain': round(float(_as.get('expected_information_gain', 0.0) or 0.0), 4),
                        },
                        {
                            'candidate': 'action_b',
                            'step': 'mixed_rollout',
                            'next_phase': str((_bs := mixed_rollout.get('action_b', {})).get('to_phase', '') or ''),
                            'risk': round(float(_bs.get('state_shift_risk', 0.0) or 0.0), 4),
                            'info_gain': round(float(_bs.get('expected_information_gain', 0.0) or 0.0), 4),
                        },
                    ]
                else:
                    preferred = ""
                    reasoning = ""
                    decision_path = ""
            else:
                preferred = ""
                reasoning = ""
                decision_path = ""

            if not preferred:
                decision_path = "priors_fallback"
            rollout_a = self._simulate_hidden_hypothesis_branch_rollout(
                state_slice=state_slice,
                action=action_a,
                context=context or {},
            )
            rollout_b = self._simulate_hidden_hypothesis_branch_rollout(
                state_slice=state_slice,
                action=action_b,
                context=context or {},
            )
            if not preferred and rollout_a and rollout_b:
                rollout_delta = float(rollout_a.get('value', 0.0) or 0.0) - float(rollout_b.get('value', 0.0) or 0.0)
                used_branch_rollout = bool(rollout_a.get('branch_rollups') or rollout_b.get('branch_rollups'))
                if abs(rollout_delta) >= 0.08 or max(float(rollout_a.get('confidence', 0.0) or 0.0), float(rollout_b.get('confidence', 0.0) or 0.0)) >= 0.58:
                    preferred = fn_a if rollout_delta >= 0.0 else fn_b
                    winning_rollout = rollout_a if preferred == fn_a else rollout_b
                    losing_rollout = rollout_b if preferred == fn_a else rollout_a
                    strongest_branch = {}
                    if used_branch_rollout:
                        strongest_branch = max(
                            list(winning_rollout.get('branch_rollups', []) or []),
                            key=lambda item: float(item.get('weight', 0.0) or 0.0),
                            default={},
                        )
                    if used_branch_rollout and strongest_branch:
                        reasoning = (
                            f"Hypothesis-branch rollout predicts {preferred} best satisfies "
                            f"{strongest_branch.get('hypothesis_id', 'latent hypothesis')} and drives phase path "
                            f"{' -> '.join(winning_rollout.get('phase_path', []))} with value "
                            f"{winning_rollout.get('value', 0.0):.2f} vs {losing_rollout.get('value', 0.0):.2f}."
                        )
                        decision_path = "hidden_hypothesis_branch_rollout"
                    else:
                        reasoning = (
                            f"Hidden-state rollout predicts {preferred} drives phase path "
                            f"{' -> '.join(winning_rollout.get('phase_path', []))} with value "
                            f"{winning_rollout.get('value', 0.0):.2f} vs {losing_rollout.get('value', 0.0):.2f}."
                        )
                        decision_path = "hidden_state_rollout"
                    confidence = (
                        CounterfactualConfidence.HIGH
                        if max(float(rollout_a.get('confidence', 0.0) or 0.0), float(rollout_b.get('confidence', 0.0) or 0.0)) >= 0.72
                        else CounterfactualConfidence.MEDIUM
                    )
                    rollout_summary = {
                        'preferred_action': preferred,
                        'rollout_delta': round(rollout_delta, 4),
                        'used_hypothesis_branch_rollout': used_branch_rollout,
                        'action_a': rollout_a,
                        'action_b': rollout_b,
                    }
                    rollout_trace = [
                        {'candidate': 'action_a', **row} for row in list(rollout_a.get('trace', []))
                    ] + [
                        {'candidate': 'action_b', **row} for row in list(rollout_b.get('trace', []))
                    ]
                    for candidate_name, rollout in (('action_a', rollout_a), ('action_b', rollout_b)):
                        for branch in list(rollout.get('branch_rollups', []) or [])[:2]:
                            branch_rollout = branch.get('rollout', {}) if isinstance(branch.get('rollout', {}), dict) else {}
                            rollout_trace.append(
                                {
                                    'candidate': candidate_name,
                                    'step': 'branch',
                                    'hypothesis_id': str(branch.get('hypothesis_id', '') or ''),
                                    'expected_phase': str(branch.get('expected_phase', '') or ''),
                                    'action_linked': bool(branch.get('action_linked', False)),
                                    'weight': round(float(branch.get('weight', 0.0) or 0.0), 4),
                                    'branch_value': round(float(branch_rollout.get('value', 0.0) or 0.0), 4),
                                }
                            )
                else:
                    preferred, reasoning = self._compare_actions(state_slice, fn_a, fn_b, context)
                    decision_path = "priors_fallback"
            elif not preferred:
                # Simple comparison logic (rule-based fallback when no beliefs)
                preferred, reasoning = self._compare_actions(state_slice, fn_a, fn_b, context)
                decision_path = "priors_fallback"
        
        # Calculate estimated delta if possible
        delta = float(rollout_summary.get('rollout_delta', 0.0) or 0.0) if rollout_summary else self._estimate_delta(state_slice, fn_a, fn_b, preferred)
        
        outcome = CounterfactualOutcome(
            outcome_id=f"cf_{len(self._history)}_{fn_a}_vs_{fn_b}",
            question=f"Which action is better: {fn_a} or {fn_b}?",
            preferred_action=preferred,
            predicted_difference=f"Based on current beliefs, {preferred} is predicted to yield better outcomes.",
            confidence=confidence,
            reasoning=reasoning,
            belief_ids_used=[b['belief_id'] for b in state_slice.established_beliefs],
            estimated_delta=delta,
            decision_path=decision_path,
            mechanism_match=mechanism_match,
            rollout_trace=rollout_trace,
            rollout_summary=rollout_summary,
        )
        
        self._history.append(outcome)
        return outcome

    def _compare_actions_via_mechanism(
        self,
        state_slice: StateSlice,
        fn_a: str,
        fn_b: str,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], str, Optional[Dict[str, Any]]]:
        context = context if isinstance(context, dict) else {}
        candidates = context.get('mechanism_candidates', [])
        if not isinstance(candidates, list):
            return None, "", None
        features = state_slice.state_features if isinstance(state_slice.state_features, dict) else {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            status = str(candidate.get('status', 'candidate')).lower()
            if status not in {'candidate', 'active', 'confirmed'}:
                continue
            trigger_conditions = candidate.get('trigger_conditions', [])
            if not isinstance(trigger_conditions, list):
                trigger_conditions = [str(trigger_conditions)]
            trigger_blob = " ".join(str(v).lower() for v in trigger_conditions)
            if trigger_blob and not self._trigger_matches(feature_blob=str(features).lower(), fn_a=fn_a, fn_b=fn_b, trigger_blob=trigger_blob):
                continue
            expected_transition = str(candidate.get('expected_transition', ''))
            pref = str(candidate.get('preferred_action', '')).strip()
            preferred = pref if pref in {fn_a, fn_b} else (fn_a if fn_a in expected_transition else (fn_b if fn_b in expected_transition else ""))
            if not preferred:
                continue
            mechanism_id = str(candidate.get('mechanism_id', 'unknown'))
            reasoning = (
                f"Mechanism {mechanism_id} predicts transition '{expected_transition}' when "
                f"trigger conditions hold; choosing {preferred}."
            )
            return preferred, reasoning, candidate
        return None, "", None

    def _trigger_matches(self, feature_blob: str, fn_a: str, fn_b: str, trigger_blob: str) -> bool:
        if fn_a.lower() in trigger_blob or fn_b.lower() in trigger_blob:
            return True
        return any(token in feature_blob for token in trigger_blob.split() if token)

    def _confidence_from_mechanism(self, mechanism_match: Optional[Dict[str, Any]]) -> CounterfactualConfidence:
        if not isinstance(mechanism_match, dict):
            return CounterfactualConfidence.MEDIUM
        confidence = float(mechanism_match.get('confidence', 0.0) or 0.0)
        if confidence >= 0.75:
            return CounterfactualConfidence.HIGH
        if confidence >= 0.4:
            return CounterfactualConfidence.MEDIUM
        return CounterfactualConfidence.LOW
    
    def simulate_object_removed(
        self,
        state_slice: StateSlice,
        object_id: str,
    ) -> CounterfactualOutcome:
        """
        Predict what would change if an object were removed.
        
        Used to assess asset utility — if removing an object significantly
        degrades predicted outcomes, the object has high value.
        """
        # Check if object is in established beliefs or active hypotheses
        in_beliefs = any(b.get('belief_id') == object_id for b in state_slice.established_beliefs)
        in_hypotheses = any(h.get('id') == object_id for h in state_slice.active_hypotheses)
        
        if in_beliefs:
            confidence = CounterfactualConfidence.HIGH
            reasoning = f"Object {object_id} is in established beliefs — removal would likely affect predictions."
            preferred = "remove_impact_high"
        elif in_hypotheses:
            confidence = CounterfactualConfidence.MEDIUM
            reasoning = f"Object {object_id} is in active hypotheses — removal impact uncertain."
            preferred = "remove_impact_medium"
        else:
            confidence = CounterfactualConfidence.LOW
            reasoning = f"Object {object_id} not found in beliefs or hypotheses — minimal predicted impact."
            preferred = "remove_impact_low"
        
        outcome = CounterfactualOutcome(
            outcome_id=f"cf_obj_remove_{len(self._history)}_{object_id[:8]}",
            question=f"What changes if object {object_id} is removed?",
            preferred_action=preferred,
            predicted_difference=reasoning,
            confidence=confidence,
            reasoning=reasoning,
            belief_ids_used=[b['belief_id'] for b in state_slice.established_beliefs if b.get('belief_id') == object_id],
            hypothesis_id=object_id if in_hypotheses else None,
        )
        
        self._history.append(outcome)
        return outcome
    
    def simulate_hypothesis_true_false(
        self,
        state_slice: StateSlice,
        hypothesis_id: str,
        hypothesis_claim: str,
        hypothesis_trigger: Optional[str] = None,
        hypothesis_transition: Optional[str] = None,
    ) -> CounterfactualOutcome:
        """
        Compare predicted outcomes if hypothesis is TRUE vs FALSE.
        
        Used to decide whether to invest in testing a hypothesis.
        If TRUE and FALSE lead to very different outcomes, testing is urgent.
        """
        # Check competing hypotheses
        competing = [h for h in state_slice.active_hypotheses if h.get('id') != hypothesis_id]
        
        if not competing:
            confidence = CounterfactualConfidence.LOW
            reasoning = "No competing hypotheses to compare against."
            preferred = "equal"
        else:
            # Check if hypothesis has established beliefs supporting it
            supported = any(hypothesis_id in b.get('supporting_beliefs', []) for b in state_slice.established_beliefs)
            if supported:
                confidence = CounterfactualConfidence.HIGH
                reasoning = f"Hypothesis {hypothesis_id[:8]}... is supported by established beliefs."
                preferred = "true_better"
            else:
                confidence = CounterfactualConfidence.MEDIUM
                reasoning = f"Hypothesis {hypothesis_id[:8]}... lacks established belief support."
                preferred = "untested"
        
        outcome = CounterfactualOutcome(
            outcome_id=f"cf_hyp_{len(self._history)}_{hypothesis_id[:8]}",
            question=f"Would hypothesis '{hypothesis_claim[:40]}...' be TRUE or FALSE?",
            preferred_action=preferred,
            predicted_difference=f"If TRUE: {hypothesis_transition or 'unknown transition'}. If FALSE: competing hypothesis applies.",
            confidence=confidence,
            reasoning=reasoning,
            belief_ids_used=[b['belief_id'] for b in state_slice.established_beliefs if b.get('belief_id', '').startswith(hypothesis_id[:8])],
            hypothesis_id=hypothesis_id,
        )
        
        self._history.append(outcome)
        return outcome
    
    def _assess_confidence(
        self,
        state_slice: StateSlice,
        functions: List[str],
    ) -> CounterfactualConfidence:
        """Assess how confident our counterfactual prediction is."""
        if len(state_slice.established_beliefs) >= 3:
            return CounterfactualConfidence.HIGH
        elif len(state_slice.established_beliefs) >= 1:
            return CounterfactualConfidence.MEDIUM
        return CounterfactualConfidence.LOW
    
    def _compare_actions(
        self,
        state_slice: StateSlice,
        fn_a: str,
        fn_b: str,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Compare two actions based on state slice."""
        context = context if isinstance(context, dict) else {}
        priors = context.get('transition_priors', {})
        priors = priors if isinstance(priors, dict) else {}
        prior_a = priors.get(fn_a, {}) if isinstance(priors.get(fn_a, {}), dict) else {}
        prior_b = priors.get(fn_b, {}) if isinstance(priors.get(fn_b, {}), dict) else {}
        if prior_a or prior_b:
            score_a = (
                float(prior_a.get('long_horizon_reward', 0.0) or 0.0) * 0.5
                + float(prior_a.get('info_gain', 0.0) or 0.0) * 0.25
                + float(prior_a.get('reversibility', 0.0) or 0.0) * 0.15
                - float(prior_a.get('predicted_risk', 0.0) or 0.0) * 0.45
                - float(prior_a.get('constraint_violation', 0.0) or 0.0) * 0.35
            )
            score_b = (
                float(prior_b.get('long_horizon_reward', 0.0) or 0.0) * 0.5
                + float(prior_b.get('info_gain', 0.0) or 0.0) * 0.25
                + float(prior_b.get('reversibility', 0.0) or 0.0) * 0.15
                - float(prior_b.get('predicted_risk', 0.0) or 0.0) * 0.45
                - float(prior_b.get('constraint_violation', 0.0) or 0.0) * 0.35
            )
            if score_a > score_b:
                return (fn_a, "Transition priors favor action_a on reward/risk tradeoff.")
            if score_b > score_a:
                return (fn_b, "Transition priors favor action_b on reward/risk tradeoff.")

        raw_features = getattr(state_slice, 'state_features', {})
        features = raw_features if isinstance(raw_features, dict) else {}
        camera_relative = str(features.get('observation_mode', '')).lower() == 'camera_relative'
        high_motion = bool(features.get('high_motion', False))
        if camera_relative or high_motion:
            risky_fns = {'join_tables', 'aggregate_group'}
            if fn_a in risky_fns and fn_b not in risky_fns:
                return (fn_b, "State features indicate motion/coordinate risk for action_a.")
            if fn_b in risky_fns and fn_a not in risky_fns:
                return (fn_a, "State features indicate motion/coordinate risk for action_b.")

        # If we have established beliefs, use them
        for belief in state_slice.established_beliefs:
            posterior = belief.get('posterior', '')
            if posterior == fn_a:
                return (fn_a, f"Established belief supports {fn_a}.")
            elif posterior == fn_b:
                return (fn_b, f"Established belief supports {fn_b}.")
        
        # Fallback: lightweight recency heuristic
        recent = state_slice.recent_actions
        if recent:
            last = recent[-1]
            if last == fn_a:
                return (fn_a, "Recently used successfully, continuing.")
            elif last == fn_b:
                return (fn_b, "Switching to alternative action.")
        
        # Default to action with fewer recent uses
        count_a = recent.count(fn_a) if recent else 0
        count_b = recent.count(fn_b) if recent else 0
        if count_a <= count_b:
            return (fn_a, "Exploring less-used action.")
        return (fn_b, "Exploring less-used action.")
    
    def _estimate_delta(
        self,
        state_slice: StateSlice,
        fn_a: str,
        fn_b: str,
        preferred: str,
    ) -> Optional[float]:
        """Estimate numeric delta between actions."""
        # Only estimate if we have enough data
        if len(state_slice.established_beliefs) < 2:
            return None
        
        # Simple heuristic: higher confidence = more confident prediction
        base = 0.1
        belief_bonus = len(state_slice.established_beliefs) * 0.05
        return base + belief_bonus if preferred != "equal" else 0.0
    
    def get_recent_outcomes(self, n: int = 10) -> List[CounterfactualOutcome]:
        """Get the n most recent counterfactual outcomes."""
        return self._history[-n:]
    
    def outcome_count(self) -> int:
        """Total number of counterfactual outcomes generated."""
        return len(self._history)
