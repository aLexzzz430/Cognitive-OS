"""
planner/plan_reviser.py

Sprint 3: 正式规划器官

基于反馈修订计划.

Rules:
- 第一版只做简单局部修订
- 不做复杂计划搜索或重规划
"""

from __future__ import annotations
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from planner.plan_schema import Plan, PlanStatus, PlanStep, StepStatus
from planner.objective_decomposer import ObjectiveDecomposer
from modules.world_model.protocol import WorldModelControlProtocol


class RevisionTrigger(Enum):
    """修订触发原因"""
    RECOVERY = "recovery"
    TEST_RESULT = "test_result"
    HYPOTHESIS_RESOLVED = "hypothesis_resolved"
    NOVEL_DISCOVERY = "novel_discovery"
    EXIT_CRITERIA_MET = "exit_criteria_met"
    PLAN_BLOCKED = "plan_blocked"
    ASSUMPTION_INVALIDATED = "assumption_invalidated"
    CONFIDENCE_DROP = "confidence_drop"
    BUDGET_EXCEEDED = "budget_exceeded"
    ENVIRONMENT_SHIFT = "environment_shift"
    PHASE_TRANSITION = "phase_transition"


class PlanReviser:
    """
    计划修订器.
    
    第一版职责:
    1. 检测修订触发条件
    2. 局部修订计划（跳过步骤、添加步骤）
    3. 调用 decomposer 生成新计划
    
    不做:
    - 完整重规划
    - 多分支搜索
    - 复杂依赖分析
    """
    
    def __init__(self):
        self._decomposer = ObjectiveDecomposer()
        self._revision_count = 0
    
    @property
    def revision_count(self) -> int:
        return self._revision_count

    @staticmethod
    def _planner_control_profile(context: Dict[str, Any]) -> Dict[str, Any]:
        raw = context.get('planner_control_profile', {})
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _failure_strategy_profile(context: Dict[str, Any]) -> Dict[str, Any]:
        raw = context.get('failure_strategy_profile', {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _strategy_mode(self, context: Dict[str, Any]) -> str:
        failure_profile = self._failure_strategy_profile(context)
        hinted_mode = str(failure_profile.get('strategy_mode_hint', '') or '').strip()
        if hinted_mode:
            return hinted_mode
        profile = self._planner_control_profile(context)
        return str(profile.get('strategy_mode', 'balanced') or 'balanced')

    def _resolve_branch_budget(self, context: Dict[str, Any]) -> int:
        failure_profile = self._failure_strategy_profile(context)
        hinted_branch_budget = failure_profile.get('branch_budget_hint')
        if hinted_branch_budget not in (None, ''):
            try:
                return max(1, min(4, int(hinted_branch_budget)))
            except (TypeError, ValueError):
                pass
        profile = self._planner_control_profile(context)
        branch_budget = profile.get('branch_budget', 2)
        try:
            return max(1, min(4, int(branch_budget)))
        except (TypeError, ValueError):
            return 2

    def _resolve_verification_budget(self, context: Dict[str, Any]) -> int:
        failure_profile = self._failure_strategy_profile(context)
        hinted_verification_budget = failure_profile.get('verification_budget_hint')
        if hinted_verification_budget not in (None, ''):
            try:
                return max(0, min(3, int(hinted_verification_budget)))
            except (TypeError, ValueError):
                pass
        profile = self._planner_control_profile(context)
        verification_budget = profile.get('verification_budget', 0)
        try:
            return max(0, min(3, int(verification_budget)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _world_model_hidden_state(context: Dict[str, Any]) -> Dict[str, Any]:
        hidden = context.get('world_model_hidden_state', {})
        if isinstance(hidden, dict) and hidden:
            return dict(hidden)
        wm = context.get('world_model_summary', {})
        if not isinstance(wm, dict):
            return {}
        nested = wm.get('hidden_state', {})
        return dict(nested) if isinstance(nested, dict) else {}

    @staticmethod
    def _world_model_control(context: Dict[str, Any]) -> Dict[str, Any]:
        raw = context.get('world_model_control', {})
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _clamp(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
        try:
            return max(minimum, min(maximum, float(value)))
        except (TypeError, ValueError):
            return max(minimum, min(maximum, float(default)))

    @classmethod
    def _dominant_branch_profile(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        control = cls._world_model_control(context)
        hidden = cls._world_model_hidden_state(context)
        transition_memory = hidden.get('transition_memory', {}) if isinstance(hidden.get('transition_memory', {}), dict) else {}
        dominant_branch_id = str(
            control.get(
                'dominant_branch_id',
                hidden.get(
                    'dominant_branch_id',
                    transition_memory.get('dominant_branch_id', ''),
                ),
            ) or ''
        ).strip()
        latent_branches = control.get('latent_branches')
        if not isinstance(latent_branches, list) or not latent_branches:
            latent_branches = hidden.get('latent_branches', transition_memory.get('latent_branches', []))
        if not isinstance(latent_branches, list):
            return {}

        selected = {}
        for branch in latent_branches:
            if not isinstance(branch, dict):
                continue
            if dominant_branch_id and str(branch.get('branch_id', '') or '').strip() == dominant_branch_id:
                selected = branch
                break
        if not selected:
            for branch in latent_branches:
                if isinstance(branch, dict):
                    selected = branch
                    break
        if not selected:
            return {}
        return {
            'branch_id': str(selected.get('branch_id', dominant_branch_id) or dominant_branch_id or ''),
            'confidence': cls._clamp(
                selected.get('confidence', selected.get('support', selected.get('transition_score', 0.0))),
                0.0,
                1.0,
                0.0,
            ),
            'anchor_functions': [
                str(value or '').strip()
                for value in (
                    selected.get('anchor_functions', selected.get('anchored_functions', []))
                    if isinstance(selected.get('anchor_functions', selected.get('anchored_functions', [])), list)
                    else []
                )
                if str(value or '').strip()
            ],
            'risky_functions': [
                str(value or '').strip()
                for value in (selected.get('risky_functions', []) if isinstance(selected.get('risky_functions', []), list) else [])
                if str(value or '').strip()
            ],
        }

    @classmethod
    def _world_model_competition_profile(
        cls,
        context: Dict[str, Any],
        *,
        planning_universe: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        control = cls._world_model_control(context)
        hidden = cls._world_model_hidden_state(context)
        wm = context.get('world_model_summary', {})
        if not isinstance(wm, dict):
            wm = {}
        dominant_branch = cls._dominant_branch_profile(context)
        required_probes = [
            str(value or '').strip()
            for value in (
                control.get('required_probes', wm.get('required_probes', []))
                if isinstance(control.get('required_probes', wm.get('required_probes', [])), list)
                else []
            )
            if str(value or '').strip()
        ]
        planning_universe = planning_universe if isinstance(planning_universe, list) else []
        anchor_functions = [
            fn_name
            for fn_name in list(dominant_branch.get('anchor_functions', []) or [])
            if fn_name in planning_universe or not planning_universe
        ]
        risky_functions = [
            fn_name
            for fn_name in list(dominant_branch.get('risky_functions', []) or [])
            if fn_name in planning_universe or not planning_universe
        ]
        control_trust = cls._clamp(control.get('control_trust', wm.get('control_trust', 0.5)), 0.0, 1.0, 0.5)
        transition_confidence = cls._clamp(
            control.get('transition_confidence', wm.get('transition_confidence', 0.5)),
            0.0,
            1.0,
            0.5,
        )
        state_shift_risk = cls._clamp(
            control.get('state_shift_risk', wm.get('shift_risk', 0.0)),
            0.0,
            1.0,
            0.0,
        )
        hidden_drift = cls._clamp(
            control.get('hidden_drift_score', context.get('world_model_hidden_drift_score', hidden.get('drift_score', 0.0))),
            0.0,
            1.0,
            0.0,
        )
        hidden_uncertainty = cls._clamp(
            control.get('hidden_uncertainty_score', context.get('world_model_hidden_uncertainty_score', hidden.get('uncertainty_score', 0.0))),
            0.0,
            1.0,
            0.0,
        )
        dominant_branch_confidence = cls._clamp(dominant_branch.get('confidence', 0.0), 0.0, 1.0, 0.0)
        probe_pressure = min(1.0, len(required_probes) / 3.0)
        latent_instability = cls._clamp(
            (1.0 - dominant_branch_confidence) * 0.38
            + hidden_drift * 0.28
            + hidden_uncertainty * 0.22
            + state_shift_risk * 0.12,
            0.0,
            1.0,
            0.0,
        )
        probe_pressure_active = (
            probe_pressure >= 0.34
            and (
                control_trust <= 0.52
                or transition_confidence <= 0.48
                or hidden_drift >= 0.55
                or hidden_uncertainty >= 0.62
                or latent_instability >= 0.58
                or state_shift_risk >= 0.58
            )
        )
        return {
            'required_probes': required_probes,
            'probe_pressure': float(probe_pressure),
            'probe_pressure_active': bool(probe_pressure_active),
            'control_trust': float(control_trust),
            'transition_confidence': float(transition_confidence),
            'state_shift_risk': float(state_shift_risk),
            'hidden_drift_score': float(hidden_drift),
            'hidden_uncertainty_score': float(hidden_uncertainty),
            'latent_instability': float(latent_instability),
            'dominant_branch_id': str(dominant_branch.get('branch_id', '') or ''),
            'dominant_branch_confidence': float(dominant_branch_confidence),
            'dominant_anchor_functions': anchor_functions,
            'dominant_risky_functions': risky_functions,
        }

    @classmethod
    def _dominant_branch_functions(cls, context: Dict[str, Any]) -> List[str]:
        return list(cls._dominant_branch_profile(context).get('anchor_functions', []) or [])

    @staticmethod
    def _is_verification_function(fn_name: str) -> bool:
        name = str(fn_name or '').strip().lower()
        if not name:
            return False
        return any(token in name for token in ('probe', 'inspect', 'verify', 'check', 'test'))

    def _pick_verification_function(
        self,
        context: Dict[str, Any],
        *,
        exclude: Optional[str] = None,
    ) -> Optional[str]:
        failure_profile = self._failure_strategy_profile(context)
        preferred = failure_profile.get('preferred_verification_functions', [])
        if isinstance(preferred, list):
            excluded = str(exclude or '').strip()
            for fn in preferred:
                fn_name = str(fn or '').strip()
                if not fn_name or fn_name == excluded:
                    continue
                if self._is_verification_function(fn_name):
                    return fn_name
        hidden = self._world_model_hidden_state(context)
        focus_functions = hidden.get('focus_functions', []) if isinstance(hidden.get('focus_functions', []), list) else []
        dominant_branch_functions = self._dominant_branch_functions(context)
        pools = (
            dominant_branch_functions,
            focus_functions,
            context.get('visible_functions', []) or [],
            context.get('discovered_functions', []) or [],
            context.get('available_functions', []) or [],
        )
        excluded = str(exclude or '').strip()
        for pool in pools:
            for fn in pool:
                fn_name = str(fn or '').strip()
                if not fn_name or fn_name == excluded:
                    continue
                if self._is_verification_function(fn_name):
                    return fn_name
        return None

    def _fallback_candidates(
        self,
        context: Dict[str, Any],
        *,
        exclude: Optional[List[str]] = None,
    ) -> List[str]:
        failure_profile = self._failure_strategy_profile(context)
        hidden = self._world_model_hidden_state(context)
        focus_functions = hidden.get('focus_functions', []) if isinstance(hidden.get('focus_functions', []), list) else []
        dominant_branch_functions = self._dominant_branch_functions(context)
        pools = (
            failure_profile.get('preferred_fallback_functions', []) if isinstance(failure_profile.get('preferred_fallback_functions', []), list) else [],
            dominant_branch_functions,
            focus_functions,
            context.get('visible_functions', []) or [],
            context.get('discovered_functions', []) or [],
            context.get('available_functions', []) or [],
        )
        blocked = failure_profile.get('blocked_action_classes', []) if isinstance(failure_profile.get('blocked_action_classes', []), list) else []
        excluded = {str(value).strip() for value in (exclude or []) if str(value or '').strip()}
        excluded.update({str(value).strip() for value in blocked if str(value or '').strip()})
        candidates: List[str] = []
        seen = set()
        for pool in pools:
            if not isinstance(pool, list):
                continue
            for fn in pool:
                fn_name = str(fn or '').strip()
                if not fn_name or fn_name in excluded or fn_name in seen:
                    continue
                seen.add(fn_name)
                candidates.append(fn_name)
        return candidates

    @staticmethod
    def _surface_functions(context: Dict[str, Any]) -> List[str]:
        ordered: List[str] = []
        for key in ('visible_functions', 'available_functions', 'discovered_functions'):
            values = context.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                fn_name = str(value or '').strip()
                if fn_name and fn_name not in ordered:
                    ordered.append(fn_name)
        return ordered

    def _should_rebuild_from_surface(
        self,
        *,
        reason: str,
        surface_functions: List[str],
    ) -> bool:
        if not surface_functions:
            return False
        return reason in {
            'self_model_high_planner_bias',
            'world_model_plan_value_guard',
            'planner_namespace_mismatch',
            'policy_blockage_or_world_shift',
        }

    def _rebuild_blocked_plan_from_surface(
        self,
        plan: Plan,
        *,
        context: Dict[str, Any],
        reason: str,
        strategy_mode: str,
    ) -> Plan:
        surface_functions = self._surface_functions(context)
        rebuilt_context = dict(context)
        rebuilt_context['visible_functions'] = list(surface_functions)
        rebuilt_context['available_functions'] = list(surface_functions)
        rebuilt_context['discovered_functions'] = list(surface_functions)
        rebuilt_context.setdefault('reward_trend', 'negative')

        rebuilt = self._decomposer.decompose(
            SimpleNamespace(goal_id='explore_replan'),
            rebuilt_context,
        )
        rebuilt.plan_id = f"{plan.plan_id}_rev_{self._revision_count}"
        rebuilt.goal = plan.goal
        rebuilt.parent_plan_id = plan.plan_id
        rebuilt.revision_count = plan.revision_count + 1
        rebuilt.created_episode = context.get('episode', plan.created_episode)
        rebuilt.created_tick = context.get('tick', plan.created_tick)
        rebuilt.revision_reasons = [
            f"blocked_rebuild: {reason}",
            f"strategy_mode={strategy_mode}",
            *list(rebuilt.revision_reasons),
        ]
        return rebuilt

    def _make_verification_step(
        self,
        fn_name: str,
        *,
        suffix: str,
        verification_budget: int,
        reason: str,
    ) -> PlanStep:
        return PlanStep(
            step_id=f"verify_{suffix}_{fn_name}",
            description=f"验证阻塞上下文并收集证据 ({fn_name})",
            intent="test",
            target_function=fn_name,
            constraints={
                'require_probe': True,
                'verification_budget': verification_budget,
                'meta_control_injected': True,
                'blocked_reason': reason,
            },
        )

    def _make_fallback_step(
        self,
        fn_name: str,
        *,
        suffix: str,
        branch_rank: int,
        reason: str,
        strategy_mode: str,
    ) -> PlanStep:
        if strategy_mode == 'verify':
            intent = 'test'
        elif strategy_mode == 'exploit':
            intent = 'compute'
        else:
            intent = 'explore'
        return PlanStep(
            step_id=f"fallback_{suffix}_{branch_rank}_{fn_name}",
            description=f"切换到候选路径 {fn_name} 以绕过阻塞",
            intent=intent,
            target_function=fn_name,
            constraints={
                'meta_control_injected': True,
                'blocked_reason': reason,
                'branch_rank': branch_rank,
                'strategy_mode': strategy_mode,
            },
        )

    @staticmethod
    def _step_constraints(step: Optional[PlanStep]) -> Dict[str, Any]:
        if step is None:
            return {}
        raw = getattr(step, 'constraints', {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _extract_branch_frontier(self, step: Optional[PlanStep]) -> List[Dict[str, Any]]:
        constraints = self._step_constraints(step)
        raw_frontier = constraints.get('branch_frontier', [])
        if not isinstance(raw_frontier, list):
            return []

        frontier: List[Dict[str, Any]] = []
        for row in raw_frontier:
            if not isinstance(row, dict):
                continue
            residual_chain = [
                str(fn or '').strip()
                for fn in (row.get('residual_chain', []) if isinstance(row.get('residual_chain', []), list) else [])
                if str(fn or '').strip()
            ]
            target_function = str(row.get('target_function', '') or '').strip()
            if not residual_chain and target_function:
                residual_chain = [target_function]
            if not residual_chain:
                continue
            frontier.append({
                'rank': int(row.get('rank', 0) or 0),
                'target_function': target_function or residual_chain[0],
                'residual_chain': residual_chain,
                'score': float(row.get('score', 0.0) or 0.0),
                'score_gap': float(row.get('score_gap', 0.0) or 0.0),
                'risk_score': float(row.get('risk_score', 0.0) or 0.0),
                'info_gain_score': float(row.get('info_gain_score', 0.0) or 0.0),
                'structure_score': float(row.get('structure_score', 0.0) or 0.0),
                'belief_score': float(row.get('belief_score', 0.0) or 0.0),
                'belief_branch_id': str(row.get('belief_branch_id', '') or ''),
                'belief_target_phase': str(row.get('belief_target_phase', '') or ''),
                'belief_branch_confidence': float(row.get('belief_branch_confidence', 0.0) or 0.0),
                'belief_hypothesis_ids': [
                    str(value or '').strip()
                    for value in (row.get('belief_hypothesis_ids', []) if isinstance(row.get('belief_hypothesis_ids', []), list) else [])
                    if str(value or '').strip()
                ],
                'belief_anchor_functions': [
                    str(value or '').strip()
                    for value in (row.get('belief_anchor_functions', []) if isinstance(row.get('belief_anchor_functions', []), list) else [])
                    if str(value or '').strip()
                ],
                'uncertainty_reduction_score': float(row.get('uncertainty_reduction_score', 0.0) or 0.0),
            })
        return frontier

    def _score_branch_frontier_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        competition: Dict[str, Any],
        reason: str,
    ) -> float:
        residual_chain = [
            str(fn_name or '').strip()
            for fn_name in (candidate.get('residual_chain', []) if isinstance(candidate.get('residual_chain', []), list) else [])
            if str(fn_name or '').strip()
        ]
        if not residual_chain:
            return float(candidate.get('score', 0.0) or 0.0)

        required_probes = set(competition.get('required_probes', []) or [])
        anchor_functions = set(competition.get('dominant_anchor_functions', []) or [])
        risky_functions = set(competition.get('dominant_risky_functions', []) or [])
        probe_pressure = float(competition.get('probe_pressure', 0.0) or 0.0)
        latent_instability = float(competition.get('latent_instability', 0.0) or 0.0)
        probe_pressure_active = bool(competition.get('probe_pressure_active', False))
        score = float(candidate.get('score', 0.0) or 0.0)
        score += float(candidate.get('belief_score', 0.0) or 0.0) * 0.14
        score += float(candidate.get('uncertainty_reduction_score', 0.0) or 0.0) * (
            0.32 + probe_pressure * 0.12 if probe_pressure_active else 0.12
        )
        score -= float(candidate.get('risk_score', 0.0) or 0.0) * (0.10 + latent_instability * 0.12)
        score += max(0.0, 0.12 - float(candidate.get('score_gap', 0.0) or 0.0))

        if reason in {'latent_branch_conflict', 'hidden_state_drift', 'policy_blockage_or_world_shift'}:
            score += float(candidate.get('uncertainty_reduction_score', 0.0) or 0.0) * 0.14

        for idx, fn_name in enumerate(residual_chain[:3]):
            early_weight = 0.18 if idx == 0 else (0.10 if idx == 1 else 0.05)
            if fn_name in anchor_functions:
                score += early_weight
            if fn_name in risky_functions:
                score -= early_weight * (1.10 + latent_instability * 0.40)
            if fn_name in required_probes:
                score += early_weight * (1.20 if probe_pressure_active else 0.55)
            if self._is_verification_function(fn_name):
                score += early_weight * (0.82 if probe_pressure_active else 0.24)

        return float(score)

    def _rank_fallback_candidates(
        self,
        candidates: List[str],
        *,
        context: Dict[str, Any],
    ) -> List[str]:
        if len(candidates) <= 1:
            return list(candidates)
        competition = self._world_model_competition_profile(context, planning_universe=list(candidates))
        if not (
            bool(competition.get('probe_pressure_active', False))
            or list(competition.get('required_probes', []) or [])
            or list(competition.get('dominant_anchor_functions', []) or [])
            or list(competition.get('dominant_risky_functions', []) or [])
        ):
            return list(candidates)
        required_probes = set(competition.get('required_probes', []) or [])
        anchor_functions = set(competition.get('dominant_anchor_functions', []) or [])
        risky_functions = set(competition.get('dominant_risky_functions', []) or [])
        probe_pressure_active = bool(competition.get('probe_pressure_active', False))
        latent_instability = float(competition.get('latent_instability', 0.0) or 0.0)

        def _candidate_score(item: tuple[int, str]) -> tuple[float, int]:
            index, fn_name = item
            score = 0.0
            if fn_name in anchor_functions:
                score += 0.26
            if fn_name in required_probes:
                score += 0.34 if probe_pressure_active else 0.12
            if self._is_verification_function(fn_name):
                score += 0.22 if probe_pressure_active else 0.06
            if fn_name in risky_functions:
                score -= 0.34 + latent_instability * 0.12
            return (score, -index)

        ranked = sorted(enumerate(candidates), key=_candidate_score, reverse=True)
        return [fn_name for _, fn_name in ranked]

    def _build_branch_salvage_steps(
        self,
        step: Optional[PlanStep],
        *,
        context: Dict[str, Any],
        reason: str,
        strategy_mode: str,
    ) -> List[PlanStep]:
        frontier = self._extract_branch_frontier(step)
        if not frontier:
            return []

        planning_universe: List[str] = []
        for candidate in frontier:
            for fn_name in candidate.get('residual_chain', []) or []:
                text = str(fn_name or '').strip()
                if text and text not in planning_universe:
                    planning_universe.append(text)
        competition = self._world_model_competition_profile(context, planning_universe=planning_universe)
        ranked_frontier = sorted(
            frontier,
            key=lambda row: (
                self._score_branch_frontier_candidate(row, competition=competition, reason=reason),
                -int(row.get('rank', 0) or 0),
            ),
            reverse=True,
        )

        preferred = ranked_frontier[0]
        residual_chain = [
            str(fn or '').strip()
            for fn in preferred.get('residual_chain', [])
            if str(fn or '').strip()
        ]
        if not residual_chain:
            return []

        alternative_heads: List[str] = []
        for candidate in ranked_frontier[1:]:
            fn_name = str(candidate.get('target_function', '') or '').strip()
            if fn_name and fn_name not in alternative_heads and fn_name != residual_chain[0]:
                alternative_heads.append(fn_name)

        salvage_steps: List[PlanStep] = []
        inherited_constraints = self._step_constraints(step)
        original_intent = str(getattr(step, 'intent', '') or '')
        blocked_target = str(getattr(step, 'target_function', '') or '').strip()
        fallback_limit = 2
        for offset, fn_name in enumerate(residual_chain):
            if original_intent:
                intent = original_intent if offset == 0 else ('test' if self._is_verification_function(fn_name) else 'explore')
            else:
                intent = 'test' if self._is_verification_function(fn_name) else 'explore'
            constraints: Dict[str, Any] = {
                'meta_control_injected': True,
                'blocked_reason': reason,
                'strategy_mode': strategy_mode,
                'salvage_source': 'branch_frontier',
                'salvage_rank': int(preferred.get('rank', 0) or 0),
                'salvage_from': blocked_target,
            }
            if offset == 0:
                constraints['fallback_functions'] = list(alternative_heads[:fallback_limit])
                constraints['salvage_score_gap'] = float(preferred.get('score_gap', 0.0) or 0.0)
                constraints['world_model_probe_pressure'] = float(competition.get('probe_pressure', 0.0) or 0.0)
                constraints['world_model_latent_instability'] = float(competition.get('latent_instability', 0.0) or 0.0)
                constraints['dominant_branch_id'] = str(competition.get('dominant_branch_id', '') or '')
            else:
                constraints['fallback_functions'] = []
            if 'search_rollout_score' in inherited_constraints:
                constraints['search_rollout_score'] = float(inherited_constraints.get('search_rollout_score', 0.0) or 0.0)
            if preferred.get('belief_branch_id'):
                constraints['belief_branch_id'] = str(preferred.get('belief_branch_id', '') or '')
                constraints['belief_target_phase'] = str(preferred.get('belief_target_phase', '') or '')
                constraints['belief_branch_confidence'] = float(preferred.get('belief_branch_confidence', 0.0) or 0.0)
                constraints['belief_hypothesis_ids'] = list(preferred.get('belief_hypothesis_ids', []) or [])
                constraints['belief_anchor_functions'] = list(preferred.get('belief_anchor_functions', []) or [])
                constraints['belief_score'] = float(preferred.get('belief_score', 0.0) or 0.0)
                constraints['belief_uncertainty_reduction'] = float(preferred.get('uncertainty_reduction_score', 0.0) or 0.0)
            salvage_steps.append(
                PlanStep(
                    step_id=f"salvage_{preferred.get('rank', 0)}_{offset}_{fn_name}",
                    description=f"沿搜索前沿切换到 {fn_name}",
                    intent=intent,
                    target_function=fn_name,
                    constraints=constraints,
                )
            )
        return salvage_steps

    def detect_replan_trigger(self, plan: Plan, context: Dict[str, Any]) -> Optional[str]:
        """从上下文检测条件化重规划触发器。"""
        if not plan or plan.status != PlanStatus.ACTIVE:
            return None

        if bool(context.get('critical_assumption_invalidated', False)):
            return RevisionTrigger.ASSUMPTION_INVALIDATED.value

        confidence = float(context.get('plan_confidence', 1.0))
        min_conf = float(context.get('min_plan_confidence', 0.45))
        if confidence < min_conf:
            return RevisionTrigger.CONFIDENCE_DROP.value

        spent = float(context.get('budget_spent', 0.0))
        limit = float(context.get('budget_limit', 1.0))
        if limit > 0 and spent > limit:
            return RevisionTrigger.BUDGET_EXCEEDED.value

        if bool(context.get('environment_shift_detected', False)):
            return RevisionTrigger.ENVIRONMENT_SHIFT.value

        wm_signal = self._world_model_replan_signal(context)
        if wm_signal:
            return wm_signal

        return None

    def _world_model_replan_signal(self, context: Dict[str, Any]) -> Optional[str]:
        """Read world-model phase/shift predictions and map to replan trigger."""
        profile = self._planner_control_profile(context)
        protocol = WorldModelControlProtocol.from_context(context)
        wm = context.get('world_model_summary', {})
        if not isinstance(wm, dict):
            wm = {}

        predicted_phase = context.get('world_model_predicted_phase') or protocol.predicted_phase or wm.get('predicted_phase')
        current_phase = context.get('current_phase') or wm.get('current_phase')
        shift_risk = context.get('world_model_shift_risk', protocol.state_shift_risk)
        hidden_phase = context.get('world_model_hidden_phase') or protocol.hidden_state_phase or ((wm.get('hidden_state', {}) if isinstance(wm.get('hidden_state', {}), dict) else {}).get('phase'))
        hidden_drift = context.get('world_model_hidden_drift_score', protocol.hidden_drift_score)

        if predicted_phase and current_phase and str(predicted_phase) != str(current_phase):
            return RevisionTrigger.PHASE_TRANSITION.value

        try:
            shift_threshold = float(context.get('world_model_shift_replan_threshold', profile.get('world_shift_replan_threshold', 0.65)))
            hidden_drift_threshold = float(context.get('world_model_hidden_drift_replan_threshold', profile.get('hidden_drift_replan_threshold', 0.68)))
            if float(shift_risk) >= shift_threshold:
                return RevisionTrigger.ENVIRONMENT_SHIFT.value
            if str(hidden_phase or '').lower() == 'disrupted' and float(hidden_drift) >= hidden_drift_threshold:
                return RevisionTrigger.ENVIRONMENT_SHIFT.value
        except (TypeError, ValueError):
            return None

        return None

    def should_revise(
        self,
        plan: Plan,
        trigger: str,
        context: Dict[str, Any],
    ) -> bool:
        """
        判断是否应该修订.
        
        Args:
            plan: 当前计划
            trigger: 触发原因 (RevisionTrigger value)
            context: 上下文
        
        Returns:
            True if should revise
        """
        if not plan or plan.status != PlanStatus.ACTIVE:
            return False
        
        # 自动检测条件化重规划触发器
        if trigger in ("replan_check", "auto"):
            detected = self.detect_replan_trigger(plan, context)
            if detected:
                trigger = detected
            else:
                return False

        # Recovery 总是触发修订
        if trigger == "recovery":
            return True
        
        # 测试结果触发修订
        if trigger == "test_result":
            test_passed = context.get('test_passed', False)
            if not test_passed:
                return True  # 测试失败需要修订
        
        # 新发现触发修订
        if trigger == "novel_discovery":
            new_fn = context.get('new_function')
            if new_fn:
                return True
        
        # 计划阻塞触发修订
        if trigger == "plan_blocked":
            return True
        
        # Hypothesis 解决触发修订
        if trigger == "hypothesis_resolved":
            return True

        if trigger in {
            RevisionTrigger.ASSUMPTION_INVALIDATED.value,
            RevisionTrigger.CONFIDENCE_DROP.value,
            RevisionTrigger.BUDGET_EXCEEDED.value,
            RevisionTrigger.ENVIRONMENT_SHIFT.value,
            RevisionTrigger.PHASE_TRANSITION.value,
        }:
            return True

        return False
    
    def revise(
        self,
        plan: Plan,
        trigger: str,
        context: Dict[str, Any],
    ) -> Optional[Plan]:
        """
        修订计划.
        
        Args:
            plan: 当前计划
            trigger: 触发原因
            context: 上下文
        
        Returns:
            修订后的新计划，或 None 如果无法修订
        """
        if not plan:
            return None
        
        self._revision_count += 1
        
        # 根据触发原因选择修订策略
        if trigger == "recovery":
            return self._revise_for_recovery(plan, context)
        elif trigger == "test_result":
            return self._revise_for_test(plan, context)
        elif trigger == "novel_discovery":
            return self._revise_for_discovery(plan, context)
        elif trigger == "plan_blocked":
            return self._revise_for_blocked(plan, context)
        elif trigger in {
            RevisionTrigger.ASSUMPTION_INVALIDATED.value,
            RevisionTrigger.CONFIDENCE_DROP.value,
            RevisionTrigger.BUDGET_EXCEEDED.value,
            RevisionTrigger.ENVIRONMENT_SHIFT.value,
            RevisionTrigger.PHASE_TRANSITION.value,
        }:
            reason = context.get('block_reason', trigger)
            return self._revise_for_blocked(plan, {**context, 'block_reason': reason})
        else:
            return self._revise_generic(plan, context)
    
    def _revise_for_recovery(
        self,
        plan: Plan,
        context: Dict[str, Any],
    ) -> Plan:
        """为 recovery 修订计划"""
        recovery_type = context.get('recovery_type', 'generic')
        strategy_mode = self._strategy_mode(context)
        verification_budget = self._resolve_verification_budget(context)
        branch_budget = self._resolve_branch_budget(context)
        verification_fn = self._pick_verification_function(context)
        inserted_steps: List[PlanStep] = []

        if verification_budget > 0 and verification_fn:
            inserted_steps.append(
                self._make_verification_step(
                    verification_fn,
                    suffix='recovery',
                    verification_budget=verification_budget,
                    reason=str(recovery_type),
                )
            )

        # 创建 recovery 步骤
        recovery_step = PlanStep(
            step_id=f"recovery_{recovery_type}",
            description=f"执行 {recovery_type} 恢复",
            intent="recovery",
            constraints={
                'meta_control_injected': True,
                'strategy_mode': strategy_mode,
                'verification_budget': verification_budget,
                'branch_budget': branch_budget,
            },
        )
        inserted_steps.append(recovery_step)

        if branch_budget >= 3:
            excluded = [verification_fn]
            fallback_candidates = self._fallback_candidates(context, exclude=excluded)
            if fallback_candidates:
                inserted_steps.append(
                    self._make_fallback_step(
                        fallback_candidates[0],
                        suffix='recovery',
                        branch_rank=1,
                        reason=str(recovery_type),
                        strategy_mode=strategy_mode,
                    )
                )

        # 在当前步骤前插入 recovery / verification / fallback
        new_steps = inserted_steps + plan.steps[plan.current_step_index:]
        
        # 标记旧计划
        new_plan = Plan(
            plan_id=f"{plan.plan_id}_rev_{self._revision_count}",
            goal=f"[RECOVERY] {plan.goal}",
            steps=new_steps,
            exit_criteria=plan.exit_criteria,
            revision_count=plan.revision_count + 1,
            parent_plan_id=plan.plan_id,
            revision_reasons=[
                f"recovery: {recovery_type}",
                f"strategy_mode={strategy_mode}",
                f"verification_budget={verification_budget}",
                f"branch_budget={branch_budget}",
            ],
            created_episode=context.get('episode', plan.created_episode),
            created_tick=context.get('tick', 0),
        )
        
        return new_plan
    
    def _revise_for_test(
        self,
        plan: Plan,
        context: Dict[str, Any],
    ) -> Plan:
        """为测试结果修订计划"""
        test_passed = context.get('test_passed', False)
        failed_hyp_id = context.get('failed_hypothesis', '')
        
        if test_passed:
            # 测试通过，标记 hypothesis 已解决，继续
            reason = f"test passed: {failed_hyp_id}"
        else:
            # 测试失败，跳过当前 hypothesis 相关步骤
            reason = f"test failed: {failed_hyp_id}, skipping"
        
        new_steps = []
        for step in plan.steps:
            if step.step_id.startswith('test_') and failed_hyp_id in step.step_id:
                step.mark_skipped(reason)
            new_steps.append(step)
        
        new_plan = Plan(
            plan_id=f"{plan.plan_id}_rev_{self._revision_count}",
            goal=plan.goal,
            steps=new_steps,
            exit_criteria=plan.exit_criteria,
            current_step_index=plan.current_step_index,
            revision_count=plan.revision_count + 1,
            parent_plan_id=plan.plan_id,
            revision_reasons=[reason],
            created_episode=plan.created_episode,
            created_tick=context.get('tick', plan.created_tick),
        )
        
        return new_plan
    
    def _revise_for_discovery(
        self,
        plan: Plan,
        context: Dict[str, Any],
    ) -> Plan:
        """为新发现修订计划"""
        new_fn = context.get('new_function', 'unknown')
        strategy_mode = self._strategy_mode(context)
        verification_budget = self._resolve_verification_budget(context)
        branch_budget = self._resolve_branch_budget(context)
        verification_fn = self._pick_verification_function(context, exclude=str(new_fn))
        inserted_steps: List[PlanStep] = []

        if verification_budget > 0 and verification_fn:
            inserted_steps.append(
                self._make_verification_step(
                    verification_fn,
                    suffix='discovery',
                    verification_budget=verification_budget,
                    reason=f"novel:{new_fn}",
                )
            )
        
        # 添加新函数利用步骤
        exploit_step = PlanStep(
            step_id=f"exploit_new_{new_fn}",
            description=f"利用新发现 {new_fn}",
            intent="exploit",
            target_function=new_fn,
            constraints={
                'meta_control_injected': True,
                'strategy_mode': strategy_mode,
                'verification_budget': verification_budget,
            },
        )
        inserted_steps.append(exploit_step)

        if branch_budget >= 4:
            fallback_candidates = self._fallback_candidates(context, exclude=[str(new_fn), verification_fn])
            if fallback_candidates:
                inserted_steps.append(
                    self._make_fallback_step(
                        fallback_candidates[0],
                        suffix='discovery',
                        branch_rank=1,
                        reason=f"novel:{new_fn}",
                        strategy_mode=strategy_mode,
                    )
                )
        
        # 在当前步骤后插入
        new_steps = plan.steps[:plan.current_step_index + 1] + inserted_steps + plan.steps[plan.current_step_index + 1:]
        
        new_plan = Plan(
            plan_id=f"{plan.plan_id}_rev_{self._revision_count}",
            goal=f"{plan.goal} + 利用 {new_fn}",
            steps=new_steps,
            exit_criteria=plan.exit_criteria,
            revision_count=plan.revision_count + 1,
            parent_plan_id=plan.plan_id,
            revision_reasons=[
                f"novel discovery: {new_fn}",
                f"strategy_mode={strategy_mode}",
                f"verification_budget={verification_budget}",
                f"branch_budget={branch_budget}",
            ],
            created_episode=plan.created_episode,
            created_tick=context.get('tick', plan.created_tick),
        )
        
        return new_plan
    
    def _revise_for_blocked(
        self,
        plan: Plan,
        context: Dict[str, Any],
    ) -> Plan:
        """为阻塞修订计划"""
        reason = context.get('block_reason', 'unknown')
        strategy_mode = self._strategy_mode(context)
        verification_budget = self._resolve_verification_budget(context)
        branch_budget = self._resolve_branch_budget(context)
        surface_functions = self._surface_functions(context)
        if self._should_rebuild_from_surface(
            reason=str(reason),
            surface_functions=surface_functions,
        ):
            return self._rebuild_blocked_plan_from_surface(
                plan,
                context=context,
                reason=str(reason),
                strategy_mode=strategy_mode,
            )
        
        # 跳过当前步骤，继续
        new_steps = list(plan.steps)
        blocked_target = None
        current_step = None
        if plan.current_step_index < len(new_steps):
            current_step = new_steps[plan.current_step_index]
            blocked_target = str(current_step.target_function or '').strip() or None
            current_step.mark_skipped(reason)

        inserted_steps: List[PlanStep] = []
        current_constraints = self._step_constraints(current_step)
        belief_target_phase = str(current_constraints.get('belief_target_phase', '') or '').strip().lower()
        competition = self._world_model_competition_profile(
            context,
            planning_universe=self._surface_functions(context),
        )
        effective_verification_budget = verification_budget
        verification_fn = self._pick_verification_function(context, exclude=blocked_target)
        if verification_fn and (
            belief_target_phase in {'exploring', 'disrupted'}
            or bool(competition.get('probe_pressure_active', False))
            or float(competition.get('latent_instability', 0.0) or 0.0) >= 0.55
        ):
            effective_verification_budget = max(effective_verification_budget, 1)
        if effective_verification_budget > 0 and verification_fn:
            inserted_steps.append(
                self._make_verification_step(
                    verification_fn,
                    suffix='blocked',
                    verification_budget=effective_verification_budget,
                    reason=str(reason),
                )
            )

        branch_salvage_steps = self._build_branch_salvage_steps(
            current_step,
            context=context,
            reason=str(reason),
            strategy_mode=strategy_mode,
        )
        if (
            inserted_steps
            and branch_salvage_steps
            and inserted_steps[0].target_function
            and inserted_steps[0].target_function == branch_salvage_steps[0].target_function
        ):
            branch_salvage_steps = branch_salvage_steps[1:]
        inserted_steps.extend(branch_salvage_steps)

        fallback_limit = max(0, branch_budget - len(inserted_steps))
        salvage_targets = [step.target_function for step in branch_salvage_steps if step.target_function]
        fallback_candidates = self._fallback_candidates(context, exclude=[blocked_target, verification_fn, *salvage_targets])
        belief_anchor_hints = [
            str(value or '').strip()
            for value in (current_constraints.get('belief_anchor_functions', []) if isinstance(current_constraints.get('belief_anchor_functions', []), list) else [])
            if str(value or '').strip()
        ]
        merged_fallback_candidates: List[str] = []
        seen = set()
        for fn_name in [*belief_anchor_hints, *fallback_candidates]:
            if not fn_name or fn_name in seen or fn_name in {blocked_target, verification_fn, *salvage_targets}:
                continue
            seen.add(fn_name)
            merged_fallback_candidates.append(fn_name)
        merged_fallback_candidates = self._rank_fallback_candidates(
            merged_fallback_candidates,
            context=context,
        )
        for rank, candidate in enumerate(merged_fallback_candidates[:fallback_limit], start=1):
            inserted_steps.append(
                self._make_fallback_step(
                    candidate,
                    suffix='blocked',
                    branch_rank=rank,
                    reason=str(reason),
                    strategy_mode=strategy_mode,
                )
            )

        splice_index = min(plan.current_step_index + 1, len(new_steps))
        if inserted_steps:
            new_steps = new_steps[:splice_index] + inserted_steps + new_steps[splice_index:]
        
        new_plan = Plan(
            plan_id=f"{plan.plan_id}_rev_{self._revision_count}",
            goal=plan.goal,
            steps=new_steps,
            exit_criteria=plan.exit_criteria,
            current_step_index=splice_index,
            revision_count=plan.revision_count + 1,
            parent_plan_id=plan.plan_id,
            revision_reasons=[
                f"blocked: {reason}",
                f"strategy_mode={strategy_mode}",
                f"verification_budget={effective_verification_budget}",
                f"branch_budget={branch_budget}",
                f"branch_salvage={1 if branch_salvage_steps else 0}",
            ],
            created_episode=plan.created_episode,
            created_tick=context.get('tick', plan.created_tick),
        )
        
        return new_plan
    
    def _revise_generic(
        self,
        plan: Plan,
        context: Dict[str, Any],
    ) -> Plan:
        """通用修订（调用 decomposer 重分解）"""
        goal = context.get('goal')
        if not goal:
            # 无法修订
            return plan
        
        new_plan = self._decomposer.decompose(goal, context)
        
        # 标记为修订版
        new_plan.parent_plan_id = plan.plan_id
        new_plan.revision_reasons = ["generic revision"]
        new_plan.revision_count = plan.revision_count + 1
        
        return new_plan
