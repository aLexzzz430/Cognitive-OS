"""
planner/objective_decomposer.py

Sprint 3: 正式规划器官

将高层目标分解为可执行计划.

Rules:
- 第一版只做简单线性分解
- 不做复杂搜索或多分支
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from planner.plan_schema import Plan, PlanStep, ExitCriteria, StepStatus, PlanStatus
from planner.planning_policy import PlanningPolicy, resolve_planning_policy


@dataclass(frozen=True)
class _SearchChainCandidate:
    ordered_functions: List[str]
    total_score: float
    coverage_score: float
    risk_score: float
    info_gain_score: float
    verification_functions: List[str]
    structure_score: float = 0.0
    belief_score: float = 0.0
    belief_branch_id: str = ""
    belief_target_phase: str = ""
    belief_branch_confidence: float = 0.0
    hypothesis_ids: List[str] = field(default_factory=list)
    belief_anchor_functions: List[str] = field(default_factory=list)
    uncertainty_reduction_score: float = 0.0


@dataclass(frozen=True)
class _BeliefBranch:
    branch_id: str
    target_phase: str
    confidence: float
    anchored_functions: List[str] = field(default_factory=list)
    risky_functions: List[str] = field(default_factory=list)
    supporting_hypothesis_ids: List[str] = field(default_factory=list)
    uncertainty_pressure: float = 0.0
    summary: str = ""
    source: str = "latent"


@dataclass(frozen=True)
class _SearchResult:
    best_candidate: _SearchChainCandidate
    ranked_candidates: List[_SearchChainCandidate]


class ObjectiveDecomposer:
    """
    目标分解器.
    
    第一版职责:
    1. 将 continuity top goal 转换为 Plan
    2. 基于 hypothesis 生成探索步骤
    3. 基于已发现函数生成利用步骤
    
    不做:
    - 多分支计划搜索
    - 复杂依赖图
    - 条件分支
    """
    
    def __init__(self, policy: Optional[PlanningPolicy] = None):
        self._plan_counter = 0
        self._beam_width = 2  # lightweight branching for sprint-3.1
        self._policy = policy

    def _extract_policy_tags(self, context: Dict[str, Any]) -> List[str]:
        tags: List[str] = []
        env_tags = context.get('environment_tags', [])
        if isinstance(env_tags, list):
            for value in env_tags:
                if value:
                    tags.append(str(value))

        for key in ('task_family', 'domain'):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                tags.append(value.strip())
        return tags

    def _resolve_policy(self, context: Dict[str, Any]) -> PlanningPolicy:
        return resolve_planning_policy(
            tags=self._extract_policy_tags(context),
            injected_policy=self._policy,
        )

    @staticmethod
    def _ordered_context_functions(context: Dict[str, Any]) -> List[str]:
        ordered: List[str] = []
        for key in ('available_functions', 'visible_functions', 'discovered_functions'):
            values = context.get(key, [])
            if isinstance(values, dict):
                values = list(values.keys())
            if not isinstance(values, list):
                continue
            for value in values:
                name = str(value or '').strip()
                if name and name not in ordered:
                    ordered.append(name)
        return ordered

    @staticmethod
    def _is_local_machine_context(context: Dict[str, Any]) -> bool:
        tags = {
            str(context.get('task_family', '') or '').strip().lower(),
            str(context.get('domain', '') or '').strip().lower(),
        }
        env_tags = context.get('environment_tags', [])
        if isinstance(env_tags, list):
            tags.update(str(item or '').strip().lower() for item in env_tags if str(item or '').strip())
        return 'local_machine' in tags

    def _decompose_local_machine(self, goal: Any, context: Dict[str, Any]) -> Plan:
        functions = self._ordered_context_functions(context)
        function_set = set(functions)
        local_mirror = context.get('local_mirror', {}) if isinstance(context.get('local_mirror', {}), dict) else {}
        workspace_count = int(context.get('workspace_file_count', local_mirror.get('workspace_file_count', 0)) or 0)
        default_command_present = bool(
            context.get('default_command_present', False)
            or local_mirror.get('default_command_present', False)
        )
        allow_empty_exec = bool(context.get('allow_empty_exec', False) or local_mirror.get('allow_empty_exec', False))
        terminal_after_plan = bool(context.get('terminal_after_plan', local_mirror.get('terminal_after_plan', True)))

        steps: List[PlanStep] = []

        def add_step(step_id: str, description: str, intent: str, target_function: Optional[str] = None, **constraints: Any) -> None:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=description,
                    intent=intent,
                    target_function=target_function,
                    constraints=dict(constraints),
                )
            )

        needs_acquire_first = (
            workspace_count <= 0
            and 'mirror_acquire' in function_set
            and not (allow_empty_exec and default_command_present)
        )
        if needs_acquire_first:
            add_step(
                'local_machine_acquire',
                'Materialize instruction-relevant source files into the local mirror',
                'test',
                'mirror_acquire',
                optional=not default_command_present,
                local_machine_stage='acquire',
            )

        if default_command_present or 'mirror_exec' in function_set:
            add_step(
                'local_machine_execute',
                'Run the configured allowlisted command inside the local mirror',
                'compute',
                'mirror_exec',
                required=True,
                local_machine_stage='execute',
                allow_empty_exec=allow_empty_exec,
                min_reward_for_success=0.0,
                max_attempts=1,
            )

        if 'mirror_plan' in function_set or default_command_present or steps:
            add_step(
                'local_machine_sync_plan',
                'Build an auditable sync plan from mirror changes',
                'test',
                'mirror_plan',
                required=True,
                local_machine_stage='plan',
                min_reward_for_success=0.0,
                max_attempts=1,
            )

        if not terminal_after_plan:
            add_step(
                'local_machine_wait_approval',
                'Pause in WAITING_APPROVAL until the sync plan is reviewed',
                'wait',
                None,
                required=True,
                local_machine_stage='approval',
                expected_status='WAITING_APPROVAL',
            )

        if not steps:
            add_step(
                'local_machine_inspect_surface',
                'Inspect the local-machine mirror surface before choosing a write path',
                'test',
                'mirror_acquire' if 'mirror_acquire' in function_set else None,
                local_machine_stage='inspect',
            )

        success_criteria = [
            'source writes require mirror_plan/mirror_apply',
            'no source write before approval',
        ]
        if default_command_present:
            success_criteria.extend([
                'command_executed == true',
                'workspace_file_count > 0',
                'sync_plan.plan_id non_empty',
                'sync_plan.actionable_change_count > 0',
            ])
        if not terminal_after_plan:
            success_criteria.append('run_status == WAITING_APPROVAL')

        return Plan(
            plan_id=f"local_machine_plan_{self._plan_counter}",
            goal=f"执行本机镜像任务 (goal={getattr(goal, 'goal_id', 'unknown')})",
            steps=steps,
            exit_criteria=ExitCriteria(
                max_steps=len(steps) + 1,
                max_ticks=context.get('max_ticks', 50),
                success_indicator=None,
            ),
            status=PlanStatus.ACTIVE,
            created_episode=context.get('episode', 0),
            created_tick=context.get('tick', 0),
            revision_reasons=[
                'domain=local_machine',
                f"default_command_present={int(default_command_present)}",
                f"allow_empty_exec={int(allow_empty_exec)}",
                f"workspace_file_count={workspace_count}",
            ],
            planning_contract={
                'domain': 'local_machine',
                'compiler': 'local_machine_plan_compiler/v1',
                'allowed_stages': ['acquire', 'execute', 'plan', 'approval'],
            },
            approval_contract={
                'source_writes_require_sync_plan': True,
                'approval_status_for_daemon': 'WAITING_APPROVAL',
            },
            verification_contract={
                'success_criteria': success_criteria,
                'require_artifacts_when_default_command_present': bool(default_command_present),
            },
            completion_contract={
                'requires_artifact_contract': bool(default_command_present),
                'terminal_after_plan': bool(terminal_after_plan),
            },
        )

    @staticmethod
    def _planner_control_profile(context: Dict[str, Any]) -> Dict[str, Any]:
        raw = context.get('planner_control_profile', {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _resolve_branch_budget(self, context: Dict[str, Any]) -> int:
        profile = self._planner_control_profile(context)
        branch_budget = profile.get('branch_budget', self._beam_width)
        try:
            return max(1, min(4, int(branch_budget)))
        except (TypeError, ValueError):
            return self._beam_width

    def _resolve_verification_budget(self, context: Dict[str, Any]) -> int:
        profile = self._planner_control_profile(context)
        verification_budget = profile.get('verification_budget', 0)
        try:
            return max(0, min(3, int(verification_budget)))
        except (TypeError, ValueError):
            return 0

    def _resolve_search_depth(self, context: Dict[str, Any], planning_universe: List[str]) -> int:
        if not planning_universe:
            return 0
        if len(planning_universe) == 1:
            return 1

        branch_budget = self._resolve_branch_budget(context)
        verification_budget = self._resolve_verification_budget(context)
        profile = self._planner_control_profile(context)
        strategy_mode = str(profile.get('strategy_mode', 'balanced') or 'balanced').strip().lower()
        hidden = self._world_model_hidden_state(context)
        hidden_depth = max(0, int(hidden.get('depth', hidden.get('hidden_state_depth', 0)) or 0))
        hidden_confidence = max(0.0, min(1.0, float(hidden.get('phase_confidence', 0.0) or 0.0)))

        depth = 2 + max(0, branch_budget - 1)
        depth += min(1, verification_budget)
        if strategy_mode in {'explore', 'recover'}:
            depth += 1
        if hidden_depth >= 3 and hidden_confidence >= 0.45:
            depth += 1
        return max(2, min(len(planning_universe), depth, 6))

    def _resolve_beam_width(self, context: Dict[str, Any]) -> int:
        branch_budget = self._resolve_branch_budget(context)
        profile = self._planner_control_profile(context)
        strategy_mode = str(profile.get('strategy_mode', 'balanced') or 'balanced').strip().lower()
        width = 2 + branch_budget
        if strategy_mode in {'explore', 'recover'}:
            width += 1
        return max(2, min(8, width))

    @staticmethod
    def _is_verification_function(fn_name: str) -> bool:
        name = str(fn_name or '').strip().lower()
        if not name:
            return False
        return any(token in name for token in ('probe', 'inspect', 'verify', 'check', 'test'))

    def _pick_verification_function(self, context: Dict[str, Any]) -> Optional[str]:
        deliberation_tests = context.get('deliberation_candidate_tests', [])
        if isinstance(deliberation_tests, list):
            for row in deliberation_tests:
                if not isinstance(row, dict):
                    continue
                fn_name = str(row.get('function_name', '') or '').strip()
                if self._is_verification_function(fn_name):
                    return fn_name
        discovered = [fn for fn in (context.get('discovered_functions', []) or []) if fn]
        visible = [fn for fn in (context.get('visible_functions', []) or []) if fn]
        for pool in (visible, discovered):
            for fn in pool:
                if self._is_verification_function(fn):
                    return fn
        return None

    def _make_verification_step(self, fn_name: str, *, suffix: str, verification_budget: int) -> PlanStep:
        return PlanStep(
            step_id=f"verify_{suffix}_{fn_name}",
            description=f"验证当前状态并收集高价值证据 ({fn_name})",
            intent="test",
            target_function=fn_name,
            constraints={
                'require_probe': True,
                'verification_budget': verification_budget,
                'meta_control_injected': True,
            },
        )

    @staticmethod
    def _world_model_transition_priors(context: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        raw = context.get('world_model_transition_priors', {})
        return dict(raw) if isinstance(raw, dict) else {}

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

    def _world_model_latent_branches(
        self,
        context: Dict[str, Any],
        *,
        planning_universe: List[str],
    ) -> List[Dict[str, Any]]:
        hidden = self._world_model_hidden_state(context)
        transition_memory = hidden.get('transition_memory', {}) if isinstance(hidden.get('transition_memory', {}), dict) else {}
        raw_branches = hidden.get('latent_branches', transition_memory.get('latent_branches', []))
        if not isinstance(raw_branches, list):
            return []
        dominant_branch_id = str(
            hidden.get(
                'dominant_branch_id',
                transition_memory.get('dominant_branch_id', ''),
            ) or ''
        ).strip()
        hidden_focus = [
            str(fn or '').strip()
            for fn in (hidden.get('focus_functions', []) if isinstance(hidden.get('focus_functions', []), list) else [])
            if str(fn or '').strip() in planning_universe
        ]

        branches: List[Dict[str, Any]] = []
        for index, item in enumerate(raw_branches[:4]):
            if not isinstance(item, dict):
                continue
            branch_id = str(item.get('branch_id', '') or f"wm_branch_{index}").strip()
            target_phase = self._phase_alias(item.get('target_phase', item.get('current_phase', ''))) or 'exploring'
            anchor_functions = [
                str(fn or '').strip()
                for fn in (item.get('anchor_functions', []) if isinstance(item.get('anchor_functions', []), list) else [])
                if str(fn or '').strip() in planning_universe
            ]
            risky_functions = [
                str(fn or '').strip()
                for fn in (item.get('risky_functions', []) if isinstance(item.get('risky_functions', []), list) else [])
                if str(fn or '').strip() in planning_universe
            ]
            if branch_id == dominant_branch_id and not anchor_functions and hidden_focus:
                anchor_functions = list(hidden_focus[:3])

            confidence = self._clamp(item.get('confidence', item.get('transition_score', 0.0)), 0.0, 1.0)
            support = self._clamp(item.get('support', 0.0), 0.0, 1.0)
            transition_score = self._clamp(item.get('transition_score', 0.0), 0.0, 1.0)
            if branch_id == dominant_branch_id:
                confidence = self._clamp(
                    confidence * 0.82
                    + max(confidence, support, transition_score) * 0.18
                    + 0.04,
                    0.0,
                    1.0,
                )
            if not (target_phase or anchor_functions or risky_functions):
                continue
            branches.append(
                {
                    'branch_id': branch_id,
                    'target_phase': target_phase,
                    'confidence': confidence,
                    'support': support,
                    'transition_score': transition_score,
                    'anchored_functions': anchor_functions[:4],
                    'risky_functions': risky_functions[:4],
                    'uncertainty_pressure': self._clamp(
                        item.get(
                            'uncertainty_pressure',
                            hidden.get('transition_entropy', hidden.get('uncertainty_score', 0.5)),
                        ),
                        0.0,
                        1.0,
                    ),
                    'summary': str(item.get('latent_signature', '') or branch_id),
                    'dominant': branch_id == dominant_branch_id,
                }
            )

        branches.sort(
            key=lambda row: (
                not bool(row.get('dominant', False)),
                -float(row.get('confidence', 0.0) or 0.0),
                -len(row.get('anchored_functions', []) or []),
                str(row.get('branch_id', '') or ''),
            )
        )
        return branches

    def _world_model_dynamics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        priors = self._world_model_transition_priors(context)
        raw = priors.get('__world_dynamics', {})
        if isinstance(raw, dict) and raw:
            return dict(raw)
        summary = context.get('world_model_summary', {})
        return dict(summary) if isinstance(summary, dict) else {}

    @staticmethod
    def _world_model_control(context: Dict[str, Any]) -> Dict[str, Any]:
        raw = context.get('world_model_control', {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _world_model_competition_profile(
        self,
        context: Dict[str, Any],
        *,
        planning_universe: List[str],
    ) -> Dict[str, Any]:
        control = self._world_model_control(context)
        hidden = self._world_model_hidden_state(context)
        dynamics = self._world_model_dynamics(context)
        latent_branches = self._world_model_latent_branches(
            context,
            planning_universe=planning_universe,
        )
        dominant_branch = dict(latent_branches[0]) if latent_branches else {}
        control_profile = self._planner_control_profile(context)

        required_probes = [
            str(item or '').strip()
            for item in (control.get('required_probes', dynamics.get('required_probes', [])) if isinstance(control.get('required_probes', dynamics.get('required_probes', [])), list) else [])
            if str(item or '').strip()
        ]
        control_trust = self._clamp(
            control.get('control_trust', dynamics.get('control_trust', 0.5)),
            0.0,
            1.0,
            0.5,
        )
        transition_confidence = self._clamp(
            control.get('transition_confidence', dynamics.get('transition_confidence', 0.5)),
            0.0,
            1.0,
            0.5,
        )
        state_shift_risk = self._clamp(
            control.get('state_shift_risk', dynamics.get('state_shift_risk', dynamics.get('shift_risk', 0.0))),
            0.0,
            1.0,
            0.0,
        )
        hidden_drift = self._clamp(
            control.get('hidden_drift_score', hidden.get('drift_score', 0.0)),
            0.0,
            1.0,
            0.0,
        )
        hidden_uncertainty = self._clamp(
            control.get('hidden_uncertainty_score', hidden.get('uncertainty_score', 0.0)),
            0.0,
            1.0,
            0.0,
        )
        dominant_branch_confidence = self._clamp(dominant_branch.get('confidence', 0.0), 0.0, 1.0)
        probe_pressure = min(1.0, len(required_probes) / 3.0)
        belief_branch_margin_threshold = self._clamp(
            control_profile.get('belief_branch_margin_threshold', 0.10),
            0.0,
            0.5,
            0.10,
        )
        branch_persistence_margin_threshold = self._clamp(
            control_profile.get('branch_persistence_margin_threshold', 0.18),
            0.0,
            0.5,
            0.18,
        )
        low_branch_persistence_threshold = self._clamp(
            control_profile.get('low_branch_persistence_threshold', 0.38),
            0.0,
            1.0,
            0.38,
        )
        latent_instability = self._clamp(
            (1.0 - dominant_branch_confidence) * 0.38
            + hidden_drift * 0.28
            + hidden_uncertainty * 0.22
            + state_shift_risk * 0.12,
            0.0,
            1.0,
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
                or dominant_branch_confidence <= min(
                    1.0,
                    low_branch_persistence_threshold + branch_persistence_margin_threshold,
                )
            )
        )
        return {
            'required_probes': required_probes,
            'probe_pressure': float(probe_pressure),
            'control_trust': float(control_trust),
            'transition_confidence': float(transition_confidence),
            'state_shift_risk': float(state_shift_risk),
            'hidden_drift': float(hidden_drift),
            'hidden_uncertainty': float(hidden_uncertainty),
            'latent_instability': float(latent_instability),
            'dominant_branch_confidence': float(dominant_branch_confidence),
            'belief_branch_margin_threshold': float(belief_branch_margin_threshold),
            'branch_persistence_margin_threshold': float(branch_persistence_margin_threshold),
            'low_branch_persistence_threshold': float(low_branch_persistence_threshold),
            'probe_pressure_active': bool(probe_pressure_active),
        }

    @staticmethod
    def _transition_quad(prior: Any) -> Dict[str, float]:
        data = prior if isinstance(prior, dict) else {}
        return {
            'long_reward': float(data.get('long_horizon_reward', 0.0) or 0.0),
            'risk': float(data.get('predicted_risk', 0.0) or 0.0),
            'reversibility': float(data.get('reversibility', 0.0) or 0.0),
            'info_gain': float(data.get('info_gain', 0.0) or 0.0),
        }

    def _transition_prior_for_function(self, fn_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        priors = self._world_model_transition_priors(context)
        direct = priors.get(fn_name, {})
        if isinstance(direct, dict) and direct:
            return dict(direct)

        legacy = priors.get('__legacy_by_function', {})
        legacy_entry = legacy.get(fn_name, {}) if isinstance(legacy, dict) else {}
        if isinstance(legacy_entry, dict) and legacy_entry:
            return dict(legacy_entry)

        by_signature = priors.get('__by_signature', {})
        if isinstance(by_signature, dict):
            matches: List[Dict[str, Any]] = []
            for payload in by_signature.values():
                if not isinstance(payload, dict):
                    continue
                key = payload.get('key', {})
                if isinstance(key, dict) and str(key.get('function_name', '') or '') == fn_name:
                    matches.append(payload)
            if matches:
                averaged: Dict[str, float] = {
                    'long_horizon_reward': 0.0,
                    'predicted_risk': 0.0,
                    'reversibility': 0.0,
                    'info_gain': 0.0,
                    'constraint_violation': 0.0,
                    'long_horizon_reward_confidence': 0.0,
                    'predicted_risk_confidence': 0.0,
                    'reversibility_confidence': 0.0,
                    'info_gain_confidence': 0.0,
                }
                for payload in matches:
                    for field in tuple(averaged.keys()):
                        averaged[field] += float(payload.get(field, 0.0) or 0.0)
                denom = float(len(matches))
                for field in tuple(averaged.keys()):
                    averaged[field] /= denom
                return averaged

        cold_start = priors.get('__cold_start_prior', {})
        return dict(cold_start) if isinstance(cold_start, dict) else {}

    @staticmethod
    def _transition_confidence(prior: Dict[str, Any]) -> float:
        confidence_fields = (
            'long_horizon_reward_confidence',
            'predicted_risk_confidence',
            'reversibility_confidence',
            'info_gain_confidence',
        )
        values = [float(prior.get(field, 0.0) or 0.0) for field in confidence_fields if field in prior]
        if not values:
            metrics = prior.get('metrics', {})
            if isinstance(metrics, dict):
                for metric_name in ('long_horizon_reward', 'predicted_risk', 'reversibility', 'info_gain'):
                    metric_payload = metrics.get(metric_name, {})
                    if isinstance(metric_payload, dict):
                        values.append(float(metric_payload.get('confidence', 0.0) or 0.0))
        if not values:
            return 0.35
        bounded = [max(0.0, min(1.0, value)) for value in values]
        return sum(bounded) / max(1, len(bounded))

    @staticmethod
    def _action_family(fn_name: str) -> str:
        name = str(fn_name or '').strip().lower()
        if not name:
            return 'generic'
        if any(token in name for token in ('probe', 'inspect', 'verify', 'check', 'test')):
            return 'probe'
        if 'scan' in name:
            return 'scan'
        if any(token in name for token in ('calibrate', 'align', 'tune')):
            return 'calibrate'
        if any(token in name for token in ('route', 'select', 'choose', 'rank')):
            return 'route'
        if any(token in name for token in ('commit', 'apply', 'submit', 'advance', 'finalize')):
            return 'commit'
        if any(token in name for token in ('compute', 'aggregate', 'transform', 'join', 'filter', 'group')):
            return 'compute'
        return 'generic'

    @staticmethod
    def _clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return max(low, min(high, float(default)))

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
            'complete': 'committed',
            'completed': 'committed',
            'solve': 'committed',
            'solved': 'committed',
            'fail': 'disrupted',
            'failed': 'disrupted',
            'error': 'disrupted',
            'drift': 'disrupted',
        }
        return aliases.get(phase, phase)

    def _phase_alias_from_text(self, text: Any) -> str:
        lowered = str(text or '').strip().lower()
        if not lowered:
            return ''
        if any(token in lowered for token in ('disrupt', 'fail', 'error', 'collapse', 'break', 'drift', 'rupture')):
            return 'disrupted'
        if any(token in lowered for token in ('commit', 'seal', 'solve', 'complete', 'resolved', 'ready', 'finalize')):
            return 'committed'
        if any(token in lowered for token in ('stabil', 'align', 'warm', 'settle', 'prepare', 'calibrate', 'tune')):
            return 'stabilizing'
        if any(token in lowered for token in ('explor', 'probe', 'scan', 'search', 'inspect', 'verify', 'check')):
            return 'exploring'
        return ''

    @staticmethod
    def _hypothesis_value(hyp: Any, key: str, default: Any = None) -> Any:
        if isinstance(hyp, dict):
            return hyp.get(key, default)
        return getattr(hyp, key, default)

    def _active_hypothesis_views(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_hypotheses = context.get('active_hypotheses', []) or []
        views: List[Dict[str, Any]] = []
        for index, hyp in enumerate(raw_hypotheses):
            claim = str(self._hypothesis_value(hyp, 'claim', '') or '').strip()
            trigger = str(self._hypothesis_value(hyp, 'trigger_condition', '') or '').strip()
            transition = str(self._hypothesis_value(hyp, 'expected_transition', '') or '').strip()
            hyp_id = str(self._hypothesis_value(hyp, 'id', '') or f'hyp_{index}').strip()
            confidence = self._clamp(self._hypothesis_value(hyp, 'confidence', 0.0), 0.0, 1.0)
            hyp_type = str(self._hypothesis_value(hyp, 'type', '') or '').strip()
            if not (claim or trigger or transition):
                continue
            views.append(
                {
                    'id': hyp_id,
                    'claim': claim,
                    'trigger_condition': trigger,
                    'expected_transition': transition,
                    'confidence': confidence,
                    'type': hyp_type,
                }
            )
        return views

    def _hypothesis_functions(
        self,
        hypothesis_view: Dict[str, Any],
        *,
        planning_universe: List[str],
    ) -> List[str]:
        text = " ".join(
            str(hypothesis_view.get(key, '') or '').strip().lower()
            for key in ('claim', 'trigger_condition', 'expected_transition')
        ).strip()
        if not text:
            return []
        anchored: List[str] = []
        for fn_name in planning_universe:
            normalized = str(fn_name or '').strip()
            if normalized and normalized.lower() in text and normalized not in anchored:
                anchored.append(normalized)
        return anchored

    @staticmethod
    def _branch_alignment_score(
        *,
        target_phase: str,
        anchored_functions: List[str],
        world_branch: Dict[str, Any],
    ) -> float:
        if not isinstance(world_branch, dict):
            return 0.0
        alignment = 0.0
        world_target = str(world_branch.get('target_phase', '') or '').strip()
        if target_phase and world_target and target_phase == world_target:
            alignment += 0.42
        world_anchor_set = {
            str(fn or '').strip()
            for fn in (world_branch.get('anchored_functions', []) if isinstance(world_branch.get('anchored_functions', []), list) else [])
            if str(fn or '').strip()
        }
        if anchored_functions and world_anchor_set:
            overlap = len(world_anchor_set & set(anchored_functions))
            alignment += overlap / max(1, len(set(anchored_functions)))
        return alignment

    def _build_belief_branches(
        self,
        *,
        planning_universe: List[str],
        context: Dict[str, Any],
    ) -> List[_BeliefBranch]:
        if not planning_universe:
            return []

        hidden = self._world_model_hidden_state(context)
        hidden_phase = self._phase_alias(
            hidden.get('expected_next_phase', hidden.get('phase', ''))
        ) or self._phase_alias(hidden.get('phase', ''))
        hidden_confidence = max(
            self._clamp(hidden.get('expected_next_phase_confidence', 0.0), 0.0, 1.0),
            self._clamp(hidden.get('phase_confidence', 0.0), 0.0, 1.0) * 0.72,
        )
        hidden_uncertainty = self._clamp(
            hidden.get('transition_entropy', hidden.get('uncertainty_score', 0.0)),
            0.0,
            1.0,
        )
        hidden_focus = [
            str(fn or '').strip()
            for fn in (hidden.get('focus_functions', []) if isinstance(hidden.get('focus_functions', []), list) else [])
            if str(fn or '').strip() in planning_universe
        ]
        branches: List[_BeliefBranch] = []
        world_model_latent_branches = self._world_model_latent_branches(
            context,
            planning_universe=planning_universe,
        )
        for branch_view in world_model_latent_branches:
            branches.append(
                _BeliefBranch(
                    branch_id=str(branch_view.get('branch_id', '') or 'wm::latent'),
                    target_phase=str(branch_view.get('target_phase', '') or 'exploring'),
                    confidence=max(0.28, float(branch_view.get('confidence', 0.0) or 0.0)),
                    anchored_functions=list(branch_view.get('anchored_functions', []) or [])[:4],
                    risky_functions=list(branch_view.get('risky_functions', []) or [])[:4],
                    supporting_hypothesis_ids=[],
                    uncertainty_pressure=float(branch_view.get('uncertainty_pressure', hidden_uncertainty) or hidden_uncertainty),
                    summary=str(branch_view.get('summary', '') or branch_view.get('branch_id', '') or 'world model latent branch'),
                    source='world_model',
                )
            )

        if not world_model_latent_branches and (hidden_phase or hidden_focus):
            branches.append(
                _BeliefBranch(
                    branch_id=f"hidden::{hidden_phase or 'latent'}",
                    target_phase=hidden_phase or 'exploring',
                    confidence=max(0.34, hidden_confidence),
                    anchored_functions=list(hidden_focus[:3]),
                    risky_functions=[],
                    supporting_hypothesis_ids=[],
                    uncertainty_pressure=hidden_uncertainty,
                    summary=f"hidden phase {hidden_phase or 'latent'}",
                    source='hidden',
                )
            )

        for hypothesis_view in self._active_hypothesis_views(context):
            target_phase = (
                self._phase_alias_from_text(hypothesis_view.get('expected_transition', ''))
                or self._phase_alias_from_text(hypothesis_view.get('claim', ''))
                or self._phase_alias_from_text(hypothesis_view.get('trigger_condition', ''))
                or hidden_phase
                or 'exploring'
            )
            anchored_functions = self._hypothesis_functions(hypothesis_view, planning_universe=planning_universe)
            if not anchored_functions and not target_phase:
                continue
            confidence = self._clamp(hypothesis_view.get('confidence', 0.0), 0.0, 1.0)
            supporting_world_branch = None
            supporting_alignment = 0.0
            for world_branch in world_model_latent_branches:
                alignment = self._branch_alignment_score(
                    target_phase=target_phase or 'exploring',
                    anchored_functions=anchored_functions,
                    world_branch=world_branch,
                )
                if alignment > supporting_alignment:
                    supporting_alignment = alignment
                    supporting_world_branch = world_branch
            if supporting_world_branch is not None:
                world_anchor_functions = list(supporting_world_branch.get('anchored_functions', []) or [])
                if not anchored_functions and world_anchor_functions:
                    anchored_functions = list(world_anchor_functions[:4])
                elif world_anchor_functions:
                    anchored_functions = list(dict.fromkeys(list(anchored_functions) + world_anchor_functions))[:4]
                confidence = self._clamp(
                    confidence * 0.88 + float(supporting_world_branch.get('confidence', 0.0) or 0.0) * 0.12,
                    0.0,
                    1.0,
                )
            uncertainty_pressure = self._clamp(
                hidden_uncertainty * 0.55
                + (0.32 if target_phase in {'disrupted', 'exploring'} else 0.12)
                + max(0.0, 0.58 - confidence) * 0.35
                - supporting_alignment * 0.10,
                0.0,
                1.0,
            )
            branches.append(
                _BeliefBranch(
                    branch_id=str(hypothesis_view.get('id', '') or f"latent::{target_phase}"),
                    target_phase=target_phase or 'exploring',
                    confidence=max(0.25, confidence),
                    anchored_functions=anchored_functions[:4],
                    risky_functions=list(supporting_world_branch.get('risky_functions', []) or [])[:4] if supporting_world_branch is not None else [],
                    supporting_hypothesis_ids=[str(hypothesis_view.get('id', '') or '')] if str(hypothesis_view.get('id', '') or '') else [],
                    uncertainty_pressure=uncertainty_pressure,
                    summary=str(hypothesis_view.get('claim', '') or hypothesis_view.get('expected_transition', '') or target_phase),
                    source='hypothesis',
                )
            )

        if not branches:
            branches.append(
                _BeliefBranch(
                    branch_id='latent::default',
                    target_phase='exploring',
                    confidence=0.3,
                    anchored_functions=[],
                    risky_functions=[],
                    supporting_hypothesis_ids=[],
                    uncertainty_pressure=0.5,
                    summary='default latent exploration branch',
                    source='default',
                )
            )

        deduped: List[_BeliefBranch] = []
        seen = set()
        for branch in branches:
            key = (branch.branch_id, branch.target_phase, tuple(branch.anchored_functions))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(branch)
        deduped.sort(
            key=lambda row: (
                -float(row.confidence),
                -len(row.anchored_functions),
                row.branch_id,
            )
        )
        return deduped[: max(4, self._resolve_branch_budget(context) + 1)]

    def _evaluate_prefix_under_belief_branch(
        self,
        ordered_functions: List[str],
        *,
        branch: _BeliefBranch,
        context: Dict[str, Any],
        policy: PlanningPolicy,
    ) -> Dict[str, Any]:
        if not ordered_functions:
            return {
                'score': branch.confidence * 0.04,
                'retention_score': 0.0,
                'uncertainty_reduction': 0.0,
                'matched_functions': [],
            }

        score = branch.confidence * 0.08
        uncertainty_reduction = 0.0
        matched_functions: List[str] = []
        anchor_index = 0
        probe_seen = False

        for position, fn_name in enumerate(ordered_functions):
            family = self._action_family(fn_name)
            if self._is_verification_function(fn_name):
                probe_seen = True
                if position <= 1:
                    uncertainty_reduction += 0.06 + branch.uncertainty_pressure * 0.06

            if fn_name in set(branch.risky_functions):
                score -= (0.20 if position <= 1 else 0.14) * max(0.4, branch.confidence)

            if (
                anchor_index < len(branch.anchored_functions)
                and fn_name not in set(branch.anchored_functions)
                and not self._is_verification_function(fn_name)
                and position < len(branch.anchored_functions)
            ):
                score -= 0.08 * max(0.4, branch.confidence)

            if anchor_index < len(branch.anchored_functions):
                expected = branch.anchored_functions[anchor_index]
                if fn_name == expected:
                    early_weight = 0.18 if position == 0 else (0.12 if position == 1 else 0.08)
                    score += early_weight * max(0.45, branch.confidence)
                    matched_functions.append(fn_name)
                    anchor_index += 1
                elif fn_name in branch.anchored_functions[anchor_index + 1:]:
                    score -= 0.10 * max(0.4, branch.confidence)

            if branch.target_phase in {'exploring', 'disrupted'}:
                if family == 'probe' and position <= 1:
                    score += 0.08 * max(0.45, branch.confidence)
                if family == 'commit' and position <= 1 and not probe_seen:
                    score -= 0.18 * max(0.4, branch.confidence)
            elif branch.target_phase == 'stabilizing':
                if family in {'calibrate', 'probe'} and position <= 1:
                    score += 0.07 * max(0.45, branch.confidence)
                if family == 'commit' and position <= 1:
                    score -= 0.10 * max(0.45, branch.confidence)
            elif branch.target_phase == 'committed':
                if family == 'commit':
                    has_setup = any(
                        self._action_family(previous) in {'probe', 'route', 'calibrate', 'compute'}
                        for previous in ordered_functions[:position]
                    )
                    if has_setup:
                        score += 0.12 * max(0.45, branch.confidence)
                    elif position <= 1:
                        score -= 0.16 * max(0.45, branch.confidence)

            if fn_name in policy.canonical_chain:
                score += 0.01

        anchor_coverage = (
            len(matched_functions) / max(1, len(branch.anchored_functions))
            if branch.anchored_functions
            else 0.35
        )
        if branch.anchored_functions and len(matched_functions) == len(branch.anchored_functions):
            score += 0.10 * max(0.45, branch.confidence)
        if branch.target_phase == 'committed' and any(self._action_family(fn) == 'commit' for fn in ordered_functions):
            score += 0.08
        retention_score = self._clamp(
            anchor_coverage * 0.62
            + uncertainty_reduction * 0.48
            + (0.14 if branch.target_phase in {'stabilizing', 'committed'} else 0.0)
            + branch.confidence * 0.10,
            0.0,
            1.2,
        )
        score += retention_score * 0.22
        return {
            'score': score,
            'retention_score': retention_score,
            'uncertainty_reduction': uncertainty_reduction,
            'matched_functions': matched_functions,
        }

    def _best_belief_branch_for_prefix(
        self,
        ordered_functions: List[str],
        *,
        belief_branches: List[_BeliefBranch],
        context: Dict[str, Any],
        policy: PlanningPolicy,
    ) -> Dict[str, Any]:
        if not belief_branches:
            return {
                'branch': None,
                'score': 0.0,
                'retention_score': 0.0,
                'uncertainty_reduction': 0.0,
                'matched_functions': [],
            }

        best_branch: Optional[_BeliefBranch] = None
        best_eval: Dict[str, Any] = {
            'score': float('-inf'),
            'retention_score': 0.0,
            'uncertainty_reduction': 0.0,
            'matched_functions': [],
        }
        for branch in belief_branches:
            branch_eval = self._evaluate_prefix_under_belief_branch(
                ordered_functions,
                branch=branch,
                context=context,
                policy=policy,
            )
            total = float(branch_eval.get('score', 0.0) or 0.0)
            if best_branch is None or total > float(best_eval.get('score', float('-inf'))):
                best_branch = branch
                best_eval = branch_eval
        return {
            'branch': best_branch,
            'score': float(best_eval.get('score', 0.0) or 0.0),
            'retention_score': float(best_eval.get('retention_score', 0.0) or 0.0),
            'uncertainty_reduction': float(best_eval.get('uncertainty_reduction', 0.0) or 0.0),
            'matched_functions': list(best_eval.get('matched_functions', []) or []),
        }

    def _candidate_step_score(
        self,
        fn_name: str,
        *,
        position: int,
        context: Dict[str, Any],
        policy: PlanningPolicy,
        planning_universe: List[str],
    ) -> Dict[str, float]:
        prior = self._transition_prior_for_function(fn_name, context)
        quad = self._transition_quad(prior)
        prior_confidence = self._transition_confidence(prior)
        cold_start = bool(prior.get('cold_start', False))
        constraint_violation = max(0.0, min(1.0, float(prior.get('constraint_violation', 0.0) or 0.0)))
        hidden = self._world_model_hidden_state(context)
        dynamics = self._world_model_dynamics(context)
        hidden_focus = set(hidden.get('focus_functions', []) or [])
        hidden_phase = str(hidden.get('phase', '') or '')
        hidden_depth = max(0, int(hidden.get('depth', hidden.get('hidden_state_depth', 0)) or 0))
        hidden_confidence = max(0.0, min(1.0, float(hidden.get('phase_confidence', 0.0) or 0.0)))
        dominant_world_branch = self._world_model_latent_branches(
            context,
            planning_universe=planning_universe,
        )[:1]
        dominant_world_branch = dominant_world_branch[0] if dominant_world_branch else {}
        competition = self._world_model_competition_profile(
            context,
            planning_universe=planning_universe,
        )
        active_hyps = self._active_hypothesis_views(context)
        hinted_targets = {
            fn_name
            for hyp in active_hyps
            for fn_name in self._hypothesis_functions(hyp, planning_universe=planning_universe)
        }
        verification_budget = self._resolve_verification_budget(context)
        control_profile = self._planner_control_profile(context)
        planner_bias = max(0.0, min(1.0, float(control_profile.get('planner_bias', 0.5) or 0.5)))
        verification_bias = max(0.0, min(1.0, float(control_profile.get('verification_bias', 0.5) or 0.5)))
        risk_tolerance = max(0.0, min(1.0, float(control_profile.get('risk_tolerance', 0.5) or 0.5)))
        recovery_bias = max(0.0, min(1.0, float(control_profile.get('recovery_bias', 0.5) or 0.5)))
        preferred_action_classes = {
            str(value or '').strip().lower()
            for value in (dynamics.get('preferred_action_classes', []) if isinstance(dynamics.get('preferred_action_classes', []), list) else [])
            if str(value or '').strip()
        }
        blocked_functions = {
            str(value or '').strip()
            for value in (dynamics.get('blocked_functions', []) if isinstance(dynamics.get('blocked_functions', []), list) else [])
            if str(value or '').strip()
        }
        family = self._action_family(fn_name)

        reward_term = quad['long_reward'] * (0.48 + planner_bias * 0.10)
        reversibility_term = quad['reversibility'] * (0.14 + recovery_bias * 0.12)
        info_gain_term = quad['info_gain'] * (0.18 + verification_bias * 0.16)
        risk_term = quad['risk'] * (0.30 + (1.0 - risk_tolerance) * 0.24 + constraint_violation * 0.16)
        position_discount = max(0.0, 1.0 - (position * 0.06))
        confidence_scale = 0.72 + prior_confidence * 0.28
        score = (reward_term + reversibility_term + info_gain_term - risk_term) * position_discount * confidence_scale

        if fn_name in hinted_targets:
            score += 0.14
        if fn_name in hidden_focus:
            score += min(0.18, hidden_depth * 0.04) * max(0.4, hidden_confidence)
        if hidden_phase in {'disrupted', 'exploring'} and self._is_verification_function(fn_name):
            score += 0.10 + verification_budget * 0.03
        if hidden_phase == 'disrupted' and not self._is_verification_function(fn_name):
            score -= 0.08 * max(0.4, hidden_confidence)
        if hidden_phase in {'stabilizing', 'committed'} and fn_name in hidden_focus:
            score += 0.08
        dominant_anchor_functions = {
            str(value or '').strip()
            for value in (dominant_world_branch.get('anchored_functions', []) if isinstance(dominant_world_branch.get('anchored_functions', []), list) else [])
            if str(value or '').strip()
        }
        dominant_risky_functions = {
            str(value or '').strip()
            for value in (dominant_world_branch.get('risky_functions', []) if isinstance(dominant_world_branch.get('risky_functions', []), list) else [])
            if str(value or '').strip()
        }
        dominant_branch_confidence = self._clamp(dominant_world_branch.get('confidence', 0.0), 0.0, 1.0)
        dominant_branch_phase = str(dominant_world_branch.get('target_phase', '') or '').strip()
        world_model_probe_pressure = bool(competition.get('probe_pressure_active', False))
        probe_pressure = float(competition.get('probe_pressure', 0.0) or 0.0)
        latent_instability = float(competition.get('latent_instability', 0.0) or 0.0)
        if fn_name in dominant_anchor_functions:
            branch_weight = 0.22 if position == 0 else (0.16 if position == 1 else 0.10)
            score += branch_weight * max(0.4, dominant_branch_confidence)
            if dominant_branch_phase in {'exploring', 'disrupted'} and self._is_verification_function(fn_name):
                score += 0.05 * max(0.4, dominant_branch_confidence)
            if dominant_branch_phase == 'committed' and family == 'commit':
                score += 0.32 * max(0.4, dominant_branch_confidence)
                score += quad['risk'] * 0.22 * max(0.4, dominant_branch_confidence)
        if fn_name in dominant_risky_functions:
            branch_penalty = 0.36 if position == 0 else (0.26 if position == 1 else 0.18)
            score -= branch_penalty * max(0.4, dominant_branch_confidence)
        if world_model_probe_pressure:
            if self._is_verification_function(fn_name):
                score += 0.12 + probe_pressure * 0.08 + verification_budget * 0.03
                if fn_name in dominant_anchor_functions:
                    score += 0.04 + latent_instability * 0.05
            elif family in {'commit', 'route'} and fn_name not in hidden_focus:
                score -= 0.10 + latent_instability * 0.12
            elif family == 'compute':
                score += 0.03 + min(0.04, probe_pressure * 0.06)
            if fn_name in dominant_risky_functions:
                score -= 0.05 + max(latent_instability, probe_pressure) * 0.10
        if fn_name in policy.canonical_chain:
            chain_index = list(policy.canonical_chain).index(fn_name)
            score += max(0.0, 0.10 - chain_index * 0.015)
        if fn_name in planning_universe[:2]:
            score += 0.03
        if family in preferred_action_classes:
            score += 0.08
        if any(pref and pref in str(fn_name or '').strip().lower() for pref in preferred_action_classes):
            score += 0.05
        if fn_name in blocked_functions:
            score -= 0.28
        if cold_start:
            score += min(0.06, 0.02 + quad['info_gain'] * 0.08)

        return {
            'score': float(score),
            'risk': float(quad['risk']),
            'info_gain': float(quad['info_gain']),
            'reward': float(quad['long_reward']),
        }

    def _pairwise_transition_adjustment(
        self,
        prefix: List[str],
        next_fn: str,
        *,
        context: Dict[str, Any],
        policy: PlanningPolicy,
    ) -> float:
        if not prefix:
            if next_fn in policy.canonical_chain[:1]:
                return 0.06
            return 0.0

        previous = prefix[-1]
        hidden = self._world_model_hidden_state(context)
        hidden_phase = str(hidden.get('phase', '') or '')
        hidden_focus = set(hidden.get('focus_functions', []) or [])
        verification_budget = self._resolve_verification_budget(context)

        adjustment = 0.0
        previous_family = self._action_family(previous)
        next_family = self._action_family(next_fn)
        chain_positions = {fn: idx for idx, fn in enumerate(policy.canonical_chain)}
        if previous in chain_positions and next_fn in chain_positions:
            delta = chain_positions[next_fn] - chain_positions[previous]
            if delta == 1:
                adjustment += 0.16
            elif delta <= 0:
                adjustment -= 0.20
            else:
                adjustment -= 0.06 * max(1, delta - 1)

        if previous_family == 'probe' and hidden_phase in {'disrupted', 'exploring'} and next_fn in hidden_focus:
            adjustment += 0.09
        if previous_family == next_family and next_family in {'probe', 'commit'}:
            adjustment -= 0.05
        if next_family == 'commit':
            has_route_setup = any(self._action_family(fn) == 'route' for fn in prefix)
            has_calibration_setup = any(self._action_family(fn) == 'calibrate' for fn in prefix)
            if 'route' in chain_positions and not has_route_setup:
                adjustment -= 0.14
            if 'calibrate' in chain_positions and not has_calibration_setup:
                adjustment -= 0.10
        if next_family == 'probe' and any(self._is_verification_function(fn) for fn in prefix[-1:]) and verification_budget <= 1:
            adjustment -= 0.07
        return adjustment

    def _prefix_structure_adjustment(
        self,
        ordered_functions: List[str],
        *,
        seed_orders: List[List[str]],
        context: Dict[str, Any],
        policy: PlanningPolicy,
    ) -> float:
        if not ordered_functions:
            return 0.0

        adjustment = 0.0
        hidden = self._world_model_hidden_state(context)
        hidden_phase = str(hidden.get('phase', '') or '')
        hidden_depth = max(0, int(hidden.get('depth', hidden.get('hidden_state_depth', 0)) or 0))
        hidden_confidence = max(0.0, min(1.0, float(hidden.get('phase_confidence', 0.0) or 0.0)))
        hidden_focus = set(hidden.get('focus_functions', []) or [])
        dominant_world_branch = self._world_model_latent_branches(
            context,
            planning_universe=list(dict.fromkeys(list(ordered_functions) + list(hidden_focus))),
        )[:1]
        dominant_world_branch = dominant_world_branch[0] if dominant_world_branch else {}
        active_hyps = self._active_hypothesis_views(context)
        hinted_targets = {
            fn_name
            for hyp in active_hyps
            for fn_name in self._hypothesis_functions(hyp, planning_universe=ordered_functions)
        }
        dynamics = self._world_model_dynamics(context)
        competition = self._world_model_competition_profile(
            context,
            planning_universe=list(dict.fromkeys(list(ordered_functions) + list(hidden_focus))),
        )
        blocked_functions = {
            str(value or '').strip()
            for value in (dynamics.get('blocked_functions', []) if isinstance(dynamics.get('blocked_functions', []), list) else [])
            if str(value or '').strip()
        }
        required_probes = list(dynamics.get('required_probes', []) or []) if isinstance(dynamics.get('required_probes', []), list) else []
        external_verification_available = self._resolve_verification_budget(context) > 0 and self._pick_verification_function(context) is not None

        best_seed_alignment = 0.0
        for seed in seed_orders:
            seed_alignment = 0.0
            for idx, fn_name in enumerate(ordered_functions):
                if idx < len(seed) and fn_name == seed[idx]:
                    seed_alignment += 0.012
            best_seed_alignment = max(best_seed_alignment, seed_alignment)
        adjustment += best_seed_alignment

        chain_positions = {fn: idx for idx, fn in enumerate(policy.canonical_chain)}
        canonical_sequence = [fn for fn in ordered_functions if fn in chain_positions]
        if canonical_sequence:
            canonical_indexes = [chain_positions[fn] for fn in canonical_sequence]
            if canonical_indexes == sorted(canonical_indexes):
                adjustment += 0.04 * len(canonical_sequence)
            else:
                inversions = 0
                for idx, value in enumerate(canonical_indexes):
                    inversions += sum(1 for later in canonical_indexes[idx + 1:] if later < value)
                adjustment -= 0.10 * max(1, inversions)

            prefix_len = 0
            for expected in policy.canonical_chain:
                if prefix_len < len(ordered_functions) and ordered_functions[prefix_len] == expected:
                    prefix_len += 1
                else:
                    break
            adjustment += 0.07 * prefix_len

        verification_positions = [
            idx for idx, fn_name in enumerate(ordered_functions)
            if self._is_verification_function(fn_name)
        ]
        probe_required = bool(required_probes) or (
            hidden_phase in {'disrupted', 'exploring'} and hidden_confidence >= 0.55
        )
        if bool(competition.get('probe_pressure_active', False)):
            probe_required = True
        if probe_required and not external_verification_available:
            if verification_positions:
                earliest = verification_positions[0]
                if earliest == 0:
                    adjustment += 0.14
                elif earliest == 1:
                    adjustment += 0.06
                else:
                    adjustment -= 0.14 * max(0.4, hidden_confidence)
            elif len(ordered_functions) >= 2:
                adjustment -= 0.14 * max(0.4, hidden_confidence)

        if bool(competition.get('probe_pressure_active', False)):
            probe_pressure = float(competition.get('probe_pressure', 0.0) or 0.0)
            latent_instability = float(competition.get('latent_instability', 0.0) or 0.0)
            if verification_positions:
                earliest = verification_positions[0]
                if earliest == 0:
                    adjustment += 0.12 + probe_pressure * 0.08
                elif earliest == 1:
                    adjustment += 0.05 + latent_instability * 0.06
                else:
                    adjustment -= 0.10 + latent_instability * 0.10
            else:
                adjustment -= 0.12 + latent_instability * 0.10

        for idx, fn_name in enumerate(ordered_functions[:3]):
            early_weight = 0.05 if idx == 0 else (0.03 if idx == 1 else 0.015)
            if fn_name in hidden_focus:
                adjustment += early_weight * max(1.0, hidden_depth * 0.20)
            if fn_name in hinted_targets:
                adjustment += early_weight
            if fn_name in blocked_functions:
                adjustment -= early_weight * 2.4

        dominant_anchor_functions = list(dominant_world_branch.get('anchored_functions', []) or [])
        dominant_risky_functions = {
            str(value or '').strip()
            for value in (dominant_world_branch.get('risky_functions', []) if isinstance(dominant_world_branch.get('risky_functions', []), list) else [])
            if str(value or '').strip()
        }
        dominant_branch_confidence = self._clamp(dominant_world_branch.get('confidence', 0.0), 0.0, 1.0)
        if dominant_anchor_functions:
            for idx, expected in enumerate(dominant_anchor_functions[:len(ordered_functions)]):
                current = ordered_functions[idx]
                weight = (0.12 if idx == 0 else (0.10 if idx == 1 else 0.06)) * max(0.4, dominant_branch_confidence)
                if current == expected:
                    adjustment += weight
                elif current in set(dominant_anchor_functions[idx + 1:]):
                    adjustment -= weight * 0.80
                elif not self._is_verification_function(current):
                    adjustment -= weight
        for idx, fn_name in enumerate(ordered_functions[:3]):
            if fn_name in dominant_risky_functions:
                adjustment -= (0.16 if idx <= 1 else 0.10) * max(0.4, dominant_branch_confidence)
                if bool(competition.get('probe_pressure_active', False)):
                    adjustment -= 0.05 + float(competition.get('latent_instability', 0.0) or 0.0) * 0.06

        if any(self._action_family(fn_name) == 'commit' for fn_name in ordered_functions[:2]):
            route_missing = 'route' in chain_positions and 'route' not in ordered_functions[:2]
            calibrate_missing = 'calibrate' in chain_positions and 'calibrate' not in ordered_functions[:2]
            if route_missing:
                adjustment -= 0.16
            if calibrate_missing:
                adjustment -= 0.10
            if bool(competition.get('probe_pressure_active', False)) and not verification_positions:
                adjustment -= 0.12 + float(competition.get('latent_instability', 0.0) or 0.0) * 0.08

        return adjustment

    @staticmethod
    def _search_candidate_rank_key(candidate: _SearchChainCandidate) -> tuple:
        return (
            candidate.total_score,
            candidate.belief_score,
            candidate.structure_score,
            candidate.coverage_score,
            candidate.uncertainty_reduction_score,
            candidate.info_gain_score,
            -candidate.risk_score,
        )

    def _competition_rank_bonus(
        self,
        candidate: _SearchChainCandidate,
        *,
        context: Dict[str, Any],
        verification_budget: int,
    ) -> float:
        competition = self._world_model_competition_profile(
            context,
            planning_universe=list(candidate.ordered_functions),
        )
        probe_pressure = float(competition.get('probe_pressure', 0.0) or 0.0)
        latent_instability = float(competition.get('latent_instability', 0.0) or 0.0)
        bonus = 0.0
        if bool(competition.get('probe_pressure_active', False)):
            bonus += float(candidate.uncertainty_reduction_score or 0.0) * (0.32 + probe_pressure * 0.10)
            bonus += float(candidate.belief_score or 0.0) * 0.16
            if verification_budget > 0:
                bonus += min(0.14, len(candidate.verification_functions) * 0.05)
            else:
                bonus += min(0.08, len(candidate.verification_functions) * 0.03)
            bonus -= float(candidate.risk_score or 0.0) * (0.08 + latent_instability * 0.08)
        elif latent_instability >= 0.55:
            bonus += float(candidate.uncertainty_reduction_score or 0.0) * 0.16
            bonus -= float(candidate.risk_score or 0.0) * 0.06
        return float(bonus)

    def _rank_search_candidates(
        self,
        candidates: List[_SearchChainCandidate],
        *,
        verification_budget: int,
        context: Dict[str, Any],
    ) -> List[_SearchChainCandidate]:
        ranked = [candidate for candidate in candidates if candidate.ordered_functions]
        return sorted(
            ranked,
            key=lambda row: (
                row.total_score + self._competition_rank_bonus(row, context=context, verification_budget=verification_budget),
                row.belief_score,
                row.structure_score,
                row.coverage_score,
                row.uncertainty_reduction_score,
                row.info_gain_score + min(0.12, len(row.verification_functions) * 0.03)
                if verification_budget > 0
                else row.info_gain_score,
                -row.risk_score,
            ),
            reverse=True,
        )

    def _seed_orders_for_search(
        self,
        branch_orders: List[List[str]],
        *,
        planning_universe: List[str],
        policy: PlanningPolicy,
        context: Dict[str, Any],
    ) -> List[List[str]]:
        seeds: List[List[str]] = []
        for order in branch_orders:
            if order:
                seeds.append(list(order))
        canonical = [fn for fn in policy.canonical_chain if fn in planning_universe]
        if canonical:
            seeds.append(canonical + [fn for fn in planning_universe if fn not in canonical])
        hidden_focus = [
            fn for fn in (self._world_model_hidden_state(context).get('focus_functions', []) or [])
            if fn in planning_universe
        ]
        if hidden_focus:
            seeds.append(hidden_focus + [fn for fn in planning_universe if fn not in hidden_focus])
        for branch in self._build_belief_branches(planning_universe=planning_universe, context=context):
            if not branch.anchored_functions:
                continue
            seeds.append(
                list(branch.anchored_functions)
                + [fn for fn in planning_universe if fn not in branch.anchored_functions]
            )

        unique: List[List[str]] = []
        seen = set()
        for order in seeds:
            normalized = [fn for fn in order if fn]
            signature = tuple(normalized)
            if not signature or signature in seen:
                continue
            seen.add(signature)
            unique.append(normalized)
        return unique or [list(planning_universe)]

    def _beam_search_exploration_chain(
        self,
        *,
        planning_universe: List[str],
        branch_orders: List[List[str]],
        context: Dict[str, Any],
        policy: PlanningPolicy,
    ) -> _SearchResult:
        if not planning_universe:
            empty = _SearchChainCandidate(
                ordered_functions=[],
                total_score=0.0,
                coverage_score=0.0,
                risk_score=0.0,
                info_gain_score=0.0,
                verification_functions=[],
                structure_score=0.0,
            )
            return _SearchResult(best_candidate=empty, ranked_candidates=[empty])

        verification_budget = self._resolve_verification_budget(context)
        search_depth = self._resolve_search_depth(context, planning_universe)
        beam_width = self._resolve_beam_width(context)
        belief_branches = self._build_belief_branches(
            planning_universe=planning_universe,
            context=context,
        )
        seed_orders = self._seed_orders_for_search(
            branch_orders,
            planning_universe=planning_universe,
            policy=policy,
            context=context,
        )

        beam: List[_SearchChainCandidate] = [
            _SearchChainCandidate(
                ordered_functions=[],
                total_score=0.0,
                coverage_score=0.0,
                risk_score=0.0,
                info_gain_score=0.0,
                verification_functions=[],
                structure_score=0.0,
            )
        ]

        for _depth in range(search_depth):
            expanded: List[_SearchChainCandidate] = []
            for candidate in beam:
                used = set(candidate.ordered_functions)
                remaining = [fn for fn in planning_universe if fn not in used]
                if not remaining:
                    expanded.append(candidate)
                    continue
                for fn_name in remaining:
                    next_prefix = list(candidate.ordered_functions) + [fn_name]
                    step_eval = self._candidate_step_score(
                        fn_name,
                        position=len(next_prefix) - 1,
                        context=context,
                        policy=policy,
                        planning_universe=planning_universe,
                    )
                    previous_structure = self._prefix_structure_adjustment(
                        candidate.ordered_functions,
                        seed_orders=seed_orders,
                        context=context,
                        policy=policy,
                    )
                    next_structure = self._prefix_structure_adjustment(
                        next_prefix,
                        seed_orders=seed_orders,
                        context=context,
                        policy=policy,
                    )
                    belief_eval = self._best_belief_branch_for_prefix(
                        next_prefix,
                        belief_branches=belief_branches,
                        context=context,
                        policy=policy,
                    )
                    previous_belief = float(candidate.belief_score or 0.0)
                    belief_delta = float(belief_eval.get('score', 0.0) or 0.0) - previous_belief
                    selected_branch = belief_eval.get('branch')
                    pairwise_adjustment = self._pairwise_transition_adjustment(
                        candidate.ordered_functions,
                        fn_name,
                        context=context,
                        policy=policy,
                    )
                    structure_delta = (next_structure - previous_structure) + pairwise_adjustment
                    expanded.append(
                        _SearchChainCandidate(
                            ordered_functions=next_prefix,
                            total_score=float(candidate.total_score + step_eval['score'] + structure_delta + belief_delta),
                            coverage_score=float(candidate.coverage_score + max(0.0, step_eval['reward']) + max(0.0, step_eval['info_gain'] * 0.5)),
                            risk_score=float(candidate.risk_score + step_eval['risk']),
                            info_gain_score=float(candidate.info_gain_score + step_eval['info_gain']),
                            verification_functions=list(candidate.verification_functions)
                            + ([fn_name] if self._is_verification_function(fn_name) else []),
                            structure_score=float(candidate.structure_score + structure_delta),
                            belief_score=float(belief_eval.get('score', 0.0) or 0.0),
                            belief_branch_id=str(getattr(selected_branch, 'branch_id', '') or ''),
                            belief_target_phase=str(getattr(selected_branch, 'target_phase', '') or ''),
                            belief_branch_confidence=float(getattr(selected_branch, 'confidence', 0.0) or 0.0),
                            hypothesis_ids=list(getattr(selected_branch, 'supporting_hypothesis_ids', []) or []),
                            belief_anchor_functions=list(getattr(selected_branch, 'anchored_functions', []) or []),
                            uncertainty_reduction_score=float(belief_eval.get('uncertainty_reduction', 0.0) or 0.0),
                        )
                    )
            ranked = self._rank_search_candidates(
                expanded,
                verification_budget=verification_budget,
                context=context,
            )
            if not ranked:
                break
            beam = ranked[:beam_width]

        if not beam:
            fallback = _SearchChainCandidate(
                ordered_functions=list(planning_universe[:search_depth]),
                total_score=0.0,
                coverage_score=0.0,
                risk_score=0.0,
                info_gain_score=0.0,
                verification_functions=[],
                structure_score=0.0,
            )
            return _SearchResult(best_candidate=fallback, ranked_candidates=[fallback])
        ranked_candidates = self._rank_search_candidates(
            beam,
            verification_budget=verification_budget,
            context=context,
        )
        return _SearchResult(
            best_candidate=ranked_candidates[0],
            ranked_candidates=ranked_candidates[:max(beam_width, 4)],
        )

    @staticmethod
    def _branch_frontier_entry(
        candidate: _SearchChainCandidate,
        *,
        rank: int,
        step_index: int,
        best_score: float,
    ) -> Dict[str, Any]:
        residual_chain = list(candidate.ordered_functions[step_index:])
        return {
            'rank': rank,
            'target_function': residual_chain[0] if residual_chain else '',
            'residual_chain': residual_chain,
            'score': float(candidate.total_score),
            'score_gap': float(best_score - candidate.total_score),
            'risk_score': float(candidate.risk_score),
            'info_gain_score': float(candidate.info_gain_score),
            'structure_score': float(candidate.structure_score),
            'belief_score': float(candidate.belief_score),
            'belief_branch_id': str(candidate.belief_branch_id or ''),
            'belief_target_phase': str(candidate.belief_target_phase or ''),
            'belief_branch_confidence': float(candidate.belief_branch_confidence or 0.0),
            'belief_hypothesis_ids': list(candidate.hypothesis_ids),
            'belief_anchor_functions': list(candidate.belief_anchor_functions),
            'uncertainty_reduction_score': float(candidate.uncertainty_reduction_score),
        }

    def _build_branch_frontiers(
        self,
        search_result: _SearchResult,
    ) -> List[List[Dict[str, Any]]]:
        best_order = list(search_result.best_candidate.ordered_functions)
        if not best_order:
            return []

        frontiers: List[List[Dict[str, Any]]] = [[] for _ in best_order]
        best_score = float(search_result.best_candidate.total_score)
        ranked_candidates = list(search_result.ranked_candidates)

        for step_index, best_fn in enumerate(best_order):
            prefix = best_order[:step_index]
            seen = set()
            for rank, candidate in enumerate(ranked_candidates[1:], start=2):
                if len(candidate.ordered_functions) <= step_index:
                    continue
                if candidate.ordered_functions[:step_index] != prefix:
                    continue
                candidate_fn = str(candidate.ordered_functions[step_index] or '').strip()
                if not candidate_fn or candidate_fn == best_fn or candidate_fn in seen:
                    continue
                seen.add(candidate_fn)
                frontiers[step_index].append(
                    self._branch_frontier_entry(
                        candidate,
                        rank=rank,
                        step_index=step_index,
                        best_score=best_score,
                    )
                )
                if len(frontiers[step_index]) >= 3:
                    break
        return frontiers
    
    def decompose(
        self,
        goal: Any,  # continuity Goal object
        context: Dict[str, Any],
    ) -> Plan:
        """
        将高层目标分解为可执行计划.
        
        Args:
            goal: continuity 层的目标对象
            context: 上下文，包含:
                - episode: 当前 episode
                - tick: 当前 tick
                - discovered_functions: 已发现函数列表
                - active_hypotheses: 活跃 hypothesis 列表
                - reward_trend: reward 趋势
        
        Returns:
            Plan: 可执行计划
        """
        self._plan_counter += 1
        if self._is_local_machine_context(context):
            return self._decompose_local_machine(goal, context)
        
        goal_id = getattr(goal, 'goal_id', '') or ''
        goal_type = self._classify_goal(goal_id, context)
        
        if goal_type == 'exploration':
            return self._decompose_exploration(goal, context)
        elif goal_type == 'exploitation':
            return self._decompose_exploitation(goal, context)
        elif goal_type == 'testing':
            return self._decompose_testing(goal, context)
        elif goal_type == 'confirmation':
            return self._decompose_confirmation(goal, context)
        else:
            return self._decompose_generic(goal, context)
    
    def _classify_goal(self, goal_id: str, context: Dict[str, Any]) -> str:
        """分类目标类型"""
        goal_lower = goal_id.lower()
        
        if 'explore' in goal_lower or 'discover' in goal_lower:
            return 'exploration'
        if 'exploit' in goal_lower or 'use' in goal_lower:
            return 'exploitation'
        if 'test' in goal_lower or 'probe' in goal_lower:
            return 'testing'
        if 'confirm' in goal_lower or 'verify' in goal_lower:
            return 'confirmation'
        
        return 'generic'
    
    def _build_exploration_orders(self, context: Dict[str, Any]) -> List[List[str]]:
        """Build lightweight branch orders for exploration planning."""
        policy = self._resolve_policy(context)
        branch_budget = self._resolve_branch_budget(context)
        verification_budget = self._resolve_verification_budget(context)
        base_order = list(policy.exploration_base_order)
        safe_order = list(policy.exploration_safe_order)
        visible_functions = context.get('visible_functions', []) or []
        discovered_functions = context.get('discovered_functions', []) or []

        preferred_pool = visible_functions if visible_functions else discovered_functions
        planning_universe: List[str] = []
        for fn in preferred_pool:
            if fn and fn not in planning_universe:
                planning_universe.append(fn)
        if not planning_universe:
            planning_universe = list(base_order)

        active_hyps = self._active_hypothesis_views(context)
        hinted = []
        for hyp in active_hyps:
            for fn_name in self._hypothesis_functions(hyp, planning_universe=planning_universe):
                if fn_name not in hinted:
                    hinted.append(fn_name)
        dominant_world_branch = self._world_model_latent_branches(
            context,
            planning_universe=planning_universe,
        )[:1]
        dominant_world_branch = dominant_world_branch[0] if dominant_world_branch else {}
        dominant_anchor_functions = [
            fn for fn in (dominant_world_branch.get('anchored_functions', []) if isinstance(dominant_world_branch.get('anchored_functions', []), list) else [])
            if fn in planning_universe
        ]
        dominant_risky_functions = {
            str(fn or '').strip()
            for fn in (dominant_world_branch.get('risky_functions', []) if isinstance(dominant_world_branch.get('risky_functions', []), list) else [])
            if str(fn or '').strip() in planning_universe
        }
        for fn_name in dominant_anchor_functions:
            if fn_name not in hinted:
                hinted.append(fn_name)

        reward_trend = str(context.get('reward_trend', 'neutral') or 'neutral').lower()
        policy_profile = context.get('policy_profile', {}) if isinstance(context.get('policy_profile', {}), dict) else {}
        representation_profile = context.get('representation_profile', {}) if isinstance(context.get('representation_profile', {}), dict) else {}
        planner_bias = float(policy_profile.get('planner_bias', context.get('planner_bias', 0.0)) or 0.0)
        probe_bias = float(policy_profile.get('probe_bias', 0.5) or 0.5)
        retrieval_aggressiveness = float(policy_profile.get('retrieval_aggressiveness', 0.5) or 0.5)
        retrieval_pressure = float(representation_profile.get('retrieval_pressure', retrieval_aggressiveness) or retrieval_aggressiveness)

        primary = (
            [fn for fn in hinted if fn in planning_universe]
            + [fn for fn in planning_universe if fn not in hinted and fn not in dominant_risky_functions]
            + [fn for fn in planning_universe if fn in dominant_risky_functions and fn not in hinted]
        )
        secondary = [fn for fn in safe_order if fn in planning_universe] + [
            fn for fn in planning_universe if fn not in safe_order
        ]

        conservative_pressure = 0.0
        if reward_trend == 'negative':
            conservative_pressure += 0.6
        conservative_pressure += max(0.0, planner_bias - 0.5)
        conservative_pressure += max(0.0, 0.5 - retrieval_aggressiveness)
        conservative_pressure += max(0.0, 0.45 - retrieval_pressure)

        exploration_pressure = max(0.0, probe_bias - 0.5) + max(0.0, retrieval_pressure - 0.6)

        if conservative_pressure >= 0.4 and conservative_pressure >= exploration_pressure:
            primary, secondary = secondary, primary
        elif exploration_pressure >= 0.35:
            # Favor hinted branch for higher probe bias
            hinted_first = [fn for fn in hinted if fn in planning_universe]
            primary = (
                hinted_first
                + [fn for fn in planning_universe if fn not in hinted_first and fn not in dominant_risky_functions]
                + [fn for fn in planning_universe if fn in dominant_risky_functions and fn not in hinted_first]
            )
        verification_fn = self._pick_verification_function(context)
        orders: List[List[str]] = [primary, secondary]
        if verification_budget > 0 and verification_fn and verification_fn in planning_universe:
            verification_order = [verification_fn] + [fn for fn in primary if fn != verification_fn]
            orders.append(verification_order)
        if branch_budget >= 4:
            novelty_order = [fn for fn in planning_universe if fn not in hinted] + [fn for fn in hinted if fn in planning_universe]
            orders.append(novelty_order)

        unique_orders: List[List[str]] = []
        seen = set()
        for order in orders:
            signature = tuple(order)
            if not signature or signature in seen:
                continue
            seen.add(signature)
            unique_orders.append(order)
        return unique_orders[:branch_budget]

    def _build_steps_from_order(
        self,
        order: List[str],
        discovered: List[str],
        fallback_order: Optional[List[str]] = None,
        branch_frontiers: Optional[List[List[Dict[str, Any]]]] = None,
        rollout_score: Optional[float] = None,
        search_candidate: Optional[_SearchChainCandidate] = None,
    ) -> List[PlanStep]:
        steps: List[PlanStep] = []
        fallback_order = fallback_order or []
        branch_frontiers = branch_frontiers or []
        for step_index, fn in enumerate(order):
            if fn in discovered:
                continue
            fallback_fns = [f for f in fallback_order if f != fn][:2]
            constraints: Dict[str, Any] = {'fallback_functions': fallback_fns}
            if step_index < len(branch_frontiers) and branch_frontiers[step_index]:
                constraints['branch_frontier'] = list(branch_frontiers[step_index])
            constraints['search_step_index'] = step_index
            if rollout_score is not None:
                constraints['search_rollout_score'] = float(rollout_score)
            if isinstance(search_candidate, _SearchChainCandidate) and search_candidate.belief_branch_id:
                constraints['belief_branch_id'] = str(search_candidate.belief_branch_id or '')
                constraints['belief_target_phase'] = str(search_candidate.belief_target_phase or '')
                constraints['belief_branch_confidence'] = float(search_candidate.belief_branch_confidence or 0.0)
                constraints['belief_hypothesis_ids'] = list(search_candidate.hypothesis_ids)
                constraints['belief_anchor_functions'] = list(search_candidate.belief_anchor_functions)
                constraints['belief_retention_score'] = float(search_candidate.belief_score or 0.0)
                constraints['belief_uncertainty_reduction'] = float(search_candidate.uncertainty_reduction_score or 0.0)
            steps.append(PlanStep(
                step_id=f"explore_{fn}",
                description=f"探索函数 {fn}",
                intent="explore",
                target_function=fn,
                target_state=f"capable_of_{fn}",
                constraints=constraints,
            ))
        return steps

    def _build_exploration_success_indicator(
        self,
        planning_universe: List[str],
        discovered: List[str],
    ) -> Optional[str]:
        """Build a dynamic success indicator from the current planning universe."""
        discovered_set = {fn for fn in discovered if fn}
        undiscovered_targets = [fn for fn in planning_universe if fn and fn not in discovered_set]

        if undiscovered_targets:
            primary_target = undiscovered_targets[0]
            return (
                f"discovered_functions contains any_of({','.join(undiscovered_targets)}) "
                f"or discovered_ratio_in_universe >= 0.5 (first_target={primary_target})"
            )

        if planning_universe:
            return (
                f"discovered_ratio_in_universe >= 1.0 "
                f"(universe={','.join(planning_universe)})"
            )

        return "discovered_functions non_empty"

    def _pick_confirmation_function(
        self,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Pick a stable, low-risk function for confirmation if available.

        Priority:
        1) discovered_functions ∩ stable candidates
        2) visible_functions ∩ stable candidates
        3) any discovered function
        4) any visible function
        """
        policy = self._resolve_policy(context)
        stable_low_risk = list(policy.confirmation_low_risk_functions)
        discovered = [fn for fn in (context.get('discovered_functions', []) or []) if fn]
        visible = [fn for fn in (context.get('visible_functions', []) or []) if fn]

        for fn in stable_low_risk:
            if fn in discovered:
                return fn
        for fn in stable_low_risk:
            if fn in visible:
                return fn
        if discovered:
            return discovered[0]
        if visible:
            return visible[0]
        return None

    def _decompose_exploration(
        self,
        goal: Any,
        context: Dict[str, Any],
    ) -> Plan:
        """分解探索目标"""
        self._plan_counter += 1
        discovered = [fn for fn in (context.get('discovered_functions', []) or []) if fn]
        discovered_set = set(discovered)
        visible_functions = [fn for fn in (context.get('visible_functions', []) or []) if fn]
        visible_set = set(visible_functions)
        exploration_discovered = [fn for fn in discovered if fn not in visible_set]
        exploration_skip_set = set(exploration_discovered)
        verification_budget = self._resolve_verification_budget(context)
        verification_fn = self._pick_verification_function(context)
        policy = self._resolve_policy(context)

        if not visible_functions and not discovered:
            probe_target = verification_fn or 'inspect'
            probe_step = PlanStep(
                step_id='explore_hidden_surface_probe',
                description='探测隐藏 surface 并等待可执行函数显露',
                intent='test',
                target_function=probe_target,
                constraints={
                    'require_probe': True,
                    'surface_probe': True,
                    'fallback_functions': [],
                },
            )
            return Plan(
                plan_id=f"explore_plan_{self._plan_counter}",
                goal=f"探索新函数能力 (goal={getattr(goal, 'goal_id', 'unknown')})",
                steps=[probe_step],
                exit_criteria=ExitCriteria(
                    max_steps=2,
                    max_ticks=context.get('max_ticks', 50),
                    success_indicator='discovered_functions non_empty',
                ),
                status=PlanStatus.ACTIVE,
                created_episode=context.get('episode', 0),
                created_tick=context.get('tick', 0),
                revision_reasons=[
                    'surface_probe_mode=hidden_surface',
                    f"verification_budget={verification_budget}",
                ],
            )
        
        branch_orders = self._build_exploration_orders(context)
        primary_order = branch_orders[0] if branch_orders else []
        fallback_order = branch_orders[1] if len(branch_orders) > 1 else []
        planning_universe = list(primary_order)

        # 只探索未发现的（主分支），并为每步保留 fallback 函数提示
        steps = self._build_steps_from_order(primary_order, exploration_discovered, fallback_order)

        # Hard-partial-observable style exploration should keep the canonical chain
        # to avoid 1-step collapse after "scan" completes.
        canonical_chain = list(policy.canonical_chain)
        has_chain_signal = any(fn in discovered_set or fn in visible_functions for fn in canonical_chain)
        if has_chain_signal:
            chain_intents = dict(policy.canonical_chain_intents)
            chain_fallback_map = {
                fn: list(fallbacks) for fn, fallbacks in policy.canonical_chain_fallback_map.items()
            }
            chain_steps: List[PlanStep] = []
            for fn in canonical_chain:
                if fn in exploration_skip_set:
                    continue
                fallback_fns = [c for c in chain_fallback_map.get(fn, []) if c != fn]
                fallback_fns.extend([f for f in fallback_order if f != fn and f not in fallback_fns][:2])
                chain_steps.append(PlanStep(
                    step_id=f"explore_chain_{fn}",
                    description=f"探索链路步骤 {fn}",
                    intent=chain_intents.get(fn, 'explore'),
                    target_function=fn,
                    target_state=f"capable_of_{fn}",
                    constraints={'fallback_functions': fallback_fns},
                ))

            # 把 chain 放到前面；其余探索步骤补充在后，避免重复
            existing_targets = {step.target_function for step in chain_steps if step.target_function}
            extra_steps = [step for step in steps if step.target_function not in existing_targets]
            steps = chain_steps + extra_steps
            planning_universe = canonical_chain + [fn for fn in planning_universe if fn not in canonical_chain]

        search_result = self._beam_search_exploration_chain(
            planning_universe=[
                fn for fn in planning_universe
                if fn and fn not in exploration_skip_set and (verification_fn is None or verification_budget <= 0 or fn != verification_fn)
            ],
            branch_orders=branch_orders,
            context=context,
            policy=policy,
        )
        search_candidate = search_result.best_candidate
        searched_order = [fn for fn in search_candidate.ordered_functions if fn]
        branch_frontiers = self._build_branch_frontiers(search_result)
        if searched_order:
            fallback_order = [fn for fn in planning_universe if fn not in searched_order]
            steps = self._build_steps_from_order(
                searched_order,
                exploration_discovered,
                fallback_order,
                branch_frontiers=branch_frontiers,
                rollout_score=search_candidate.total_score,
                search_candidate=search_candidate,
            )
            planning_universe = searched_order + [fn for fn in planning_universe if fn not in searched_order]

        if verification_budget > 0 and verification_fn:
            first_target = steps[0].target_function if steps else ''
            if verification_fn != first_target:
                steps = [self._make_verification_step(verification_fn, suffix='explore', verification_budget=verification_budget)] + steps

        # 如果都发现了，添加组合探索
        if len(steps) == 0:
            fallback_target = planning_universe[0] if planning_universe else None
            steps.append(PlanStep(
                step_id="explore_combinations",
                description="探索函数组合",
                intent="explore",
                target_function=fallback_target,
            ))
        
        # 探索退出条件
        exit_criteria = ExitCriteria(
            max_steps=len(steps) + 2,
            max_ticks=context.get('max_ticks', 50),
            success_indicator=self._build_exploration_success_indicator(planning_universe, exploration_discovered),
        )
        
        return Plan(
            plan_id=f"explore_plan_{self._plan_counter}",
            goal=f"探索新函数能力 (goal={getattr(goal, 'goal_id', 'unknown')})",
            steps=steps,
            exit_criteria=exit_criteria,
            status=PlanStatus.ACTIVE,
            created_episode=context.get('episode', 0),
            created_tick=context.get('tick', 0),
            revision_reasons=[
                f"branch_orders={len(branch_orders)}",
                f"branch_budget={self._resolve_branch_budget(context)}",
                f"verification_budget={verification_budget}",
                f"search_strategy=beam",
                f"search_depth={self._resolve_search_depth(context, [fn for fn in planning_universe if fn and fn not in exploration_skip_set])}",
                f"search_beam_width={self._resolve_beam_width(context)}",
                f"search_score={search_candidate.total_score:.3f}",
                f"search_coverage={search_candidate.coverage_score:.3f}",
                f"search_structure={search_candidate.structure_score:.3f}",
                f"belief_space_search={1 if search_candidate.belief_branch_id else 0}",
                f"belief_branch={search_candidate.belief_branch_id or 'none'}",
                f"belief_phase={search_candidate.belief_target_phase or 'unknown'}",
                f"belief_score={search_candidate.belief_score:.3f}",
                f"belief_uncertainty_reduction={search_candidate.uncertainty_reduction_score:.3f}",
                f"search_frontier_depths={sum(1 for frontier in branch_frontiers if frontier)}",
            ],
        )
    
    def _decompose_exploitation(
        self,
        goal: Any,
        context: Dict[str, Any],
    ) -> Plan:
        """分解利用目标"""
        self._plan_counter += 1
        discovered = context.get('discovered_functions', [])
        
        # 利用已发现函数
        steps = []
        for fn in discovered:
            steps.append(PlanStep(
                step_id=f"use_{fn}",
                description=f"使用函数 {fn}",
                intent="exploit",
                target_function=fn,
            ))
        
        # 如果没有已发现函数，等待
        if len(steps) == 0:
            steps.append(PlanStep(
                step_id="wait_for_discovery",
                description="等待发现",
                intent="wait",
            ))
        
        exit_criteria = ExitCriteria(
            max_steps=len(steps),
            max_ticks=context.get('max_ticks', 30),
        )
        
        return Plan(
            plan_id=f"exploit_plan_{self._plan_counter}",
            goal=f"利用已有能力 (goal={getattr(goal, 'goal_id', 'unknown')})",
            steps=steps,
            exit_criteria=exit_criteria,
            created_episode=context.get('episode', 0),
            created_tick=context.get('tick', 0),
        )
    
    def _decompose_testing(
        self,
        goal: Any,
        context: Dict[str, Any],
    ) -> Plan:
        """分解测试目标"""
        self._plan_counter += 1
        active_hyps = self._active_hypothesis_views(context)
        verification_budget = self._resolve_verification_budget(context)
        verification_fn = self._pick_verification_function(context)

        steps = []
        if verification_budget > 0 and verification_fn:
            steps.append(self._make_verification_step(verification_fn, suffix='test', verification_budget=verification_budget))

        # 测试 hypotheses
        max_hyp_steps = max(1, min(3, 1 + verification_budget))
        for hyp in active_hyps[:max_hyp_steps]:  # 最多 3 个
            hyp_id = str(hyp.get('id', str(hyp)))[:20]
            hyp_target = next(
                iter(self._hypothesis_functions(hyp, planning_universe=[str(fn or '').strip() for fn in (context.get('visible_functions', []) or []) if str(fn or '').strip()])),
                None,
            ) or None
            steps.append(PlanStep(
                step_id=f"test_{hyp_id}",
                description=f"测试 hypothesis {hyp_id}",
                intent="test",
                target_function=hyp_target,
            ))
        
        # 如果没有 hypothesis，通用探索
        if len(steps) == 0:
            steps.append(PlanStep(
                step_id="explore_for_tests",
                description="探索以生成 hypothesis",
                intent="explore",
            ))
        
        exit_criteria = ExitCriteria(
            max_steps=len(steps) + 1,
            max_ticks=context.get('max_ticks', 40),
        )
        
        return Plan(
            plan_id=f"test_plan_{self._plan_counter}",
            goal=f"测试 hypothesis (goal={getattr(goal, 'goal_id', 'unknown')})",
            steps=steps,
            exit_criteria=exit_criteria,
            created_episode=context.get('episode', 0),
            created_tick=context.get('tick', 0),
        )
    
    def _decompose_confirmation(
        self,
        goal: Any,
        context: Dict[str, Any],
    ) -> Plan:
        """分解确认目标"""
        self._plan_counter += 1

        confirm_fn = self._pick_confirmation_function(context)
        verification_budget = self._resolve_verification_budget(context)
        verification_fn = self._pick_verification_function(context)
        if confirm_fn:
            steps = []
            if verification_budget > 0 and verification_fn and verification_fn != confirm_fn:
                steps.append(self._make_verification_step(verification_fn, suffix='confirm', verification_budget=verification_budget))
            steps.append(
                PlanStep(
                    step_id="confirm_basic",
                    description=f"确认基础能力 ({confirm_fn})",
                    intent="compute",
                    target_function=confirm_fn,
                ),
            )
        else:
            steps = [
                PlanStep(
                    step_id="confirm_collect_observation",
                    description="收集观测并等待可确认函数出现",
                    intent="wait",
                    constraints={'reason': 'no_visible_or_discovered_function'},
                ),
            ]
        
        exit_criteria = ExitCriteria(
            max_steps=3,
            max_ticks=20,
            target_reward=1.0,
        )
        
        return Plan(
            plan_id=f"confirm_plan_{self._plan_counter}",
            goal=f"确认系统能力 (goal={getattr(goal, 'goal_id', 'unknown')})",
            steps=steps,
            exit_criteria=exit_criteria,
            created_episode=context.get('episode', 0),
            created_tick=context.get('tick', 0),
        )
    
    def _decompose_generic(
        self,
        goal: Any,
        context: Dict[str, Any],
    ) -> Plan:
        """分解通用目标"""
        self._plan_counter += 1
        
        steps = [
            PlanStep(
                step_id="generic_step_1",
                description="执行通用步骤",
                intent="explore",
            ),
            PlanStep(
                step_id="generic_step_2",
                description="验证结果",
                intent="compute",
            ),
        ]
        verification_budget = self._resolve_verification_budget(context)
        verification_fn = self._pick_verification_function(context)
        if verification_budget > 0 and verification_fn:
            steps.insert(0, self._make_verification_step(verification_fn, suffix='generic', verification_budget=verification_budget))
        
        return Plan(
            plan_id=f"generic_plan_{self._plan_counter}",
            goal=f"通用目标 (goal={getattr(goal, 'goal_id', 'unknown')})",
            steps=steps,
            exit_criteria=ExitCriteria(max_steps=5, max_ticks=30),
            created_episode=context.get('episode', 0),
            created_tick=context.get('tick', 0),
        )
