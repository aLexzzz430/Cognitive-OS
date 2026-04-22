"""
decision/risk_model.py

Sprint 2: 正式决策器官

风险评分模型.

Rules:
- 第一版只做简单风险评估
- 输入: action kind, uncertainty, recovery pending state, self-model placeholder
- 输出: RiskScore
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None

from decision.utility_schema import DecisionCandidate, RiskScore


@dataclass
class TypedUncertaintyProfile:
    """类型化不确定性分解。"""

    epistemic: float = 0.0
    aleatoric: float = 0.0
    model: float = 0.0

    @property
    def total(self) -> float:
        return min(1.0, self.epistemic * 0.4 + self.aleatoric * 0.35 + self.model * 0.25)


@dataclass
class FunctionRiskFeatures:
    """风险特征层：函数族、历史失败率、可逆性、资源压强、跨任务约束等。"""

    family: str = "generic"
    base_uncertainty: float = 0.5
    historical_failure_rate: float = 0.5
    reversibility: float = 0.5
    resource_pressure_sensitivity: float = 0.3
    cross_task_penalty: float = 0.08
    recovery_base: float = 0.3
    allowed_task_families: tuple[str, ...] = ()
    self_model_fragility_boost: float = 0.08
    self_model_recovery_discount: float = 0.05


@dataclass
class CandidateRiskFeatures:
    """候选级特征（包含函数特征 + 当前上下文增量）。"""

    function_name: str
    profile: FunctionRiskFeatures
    is_unknown_function: bool
    failure_bias: float = 0.0
    uncertainty_bias: float = 0.0
    recovery_bias: float = 0.0


class RiskModel:
    """风险评分模型."""

    def __init__(self, risk_profile_path: Optional[str] = None):
        self._function_risks = {
            'compute_stats': 0.1,
            'filter_by_predicate': 0.2,
            'join_tables': 0.3,
            'aggregate_group': 0.3,
            'array_transform': 0.25,
        }
        self._default_risk = 0.5
        self._last_uncertainty_profile = TypedUncertaintyProfile()
        self._risk_profile_path = Path(risk_profile_path) if risk_profile_path else self._default_profile_path()
        self._risk_profile_mtime: Optional[float] = None
        self._risk_features: Dict[str, FunctionRiskFeatures] = {}
        self._default_feature = FunctionRiskFeatures()
        self._load_risk_profile(self._risk_profile_path)

    def _default_profile_path(self) -> Path:
        return Path(__file__).resolve().parent / "risk_profiles" / "default.yaml"

    def _parse_yaml_without_dependency(self, text: str) -> Dict[str, Any]:
        """极简 YAML 解析，仅覆盖本项目 risk profile 结构。"""
        root: Dict[str, Any] = {}
        current_section: Optional[str] = None
        current_fn: Optional[str] = None
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if line.strip() == "functions:":
                root.setdefault("functions", {})
                current_section = "functions"
                continue
            if current_section != "functions":
                continue
            if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
                current_fn = line.strip()[:-1]
                root["functions"].setdefault(current_fn, {})
                continue
            if line.startswith("    ") and ":" in line and current_fn:
                key, value = line.strip().split(":", 1)
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    parsed = [x.strip().strip("'\"") for x in value[1:-1].split(",") if x.strip()]
                else:
                    v = value.strip("'\"")
                    try:
                        parsed = float(v)
                    except ValueError:
                        parsed = v
                root["functions"][current_fn][key] = parsed
        return root

    def _load_risk_profile(self, path: Path) -> None:
        if not path.exists():
            self._risk_features = {}
            self._risk_profile_mtime = None
            return
        text = path.read_text(encoding="utf-8")
        if yaml is None:
            raw = self._parse_yaml_without_dependency(text)
        else:
            raw = yaml.safe_load(text) or {}
        features = raw.get("functions", {}) if isinstance(raw, dict) else {}
        parsed: Dict[str, FunctionRiskFeatures] = {}
        if isinstance(features, dict):
            for fn, payload in features.items():
                if not isinstance(payload, dict):
                    continue
                allowed = payload.get("allowed_task_families", ())
                if not isinstance(allowed, (list, tuple)):
                    allowed = ()
                parsed[str(fn)] = FunctionRiskFeatures(
                    family=str(payload.get("family", "generic")),
                    base_uncertainty=self._read_float(payload.get("base_uncertainty", 0.5), 0.5),
                    historical_failure_rate=self._read_float(payload.get("historical_failure_rate", 0.5), 0.5),
                    reversibility=self._read_float(payload.get("reversibility", 0.5), 0.5),
                    resource_pressure_sensitivity=self._read_float(payload.get("resource_pressure_sensitivity", 0.3), 0.3),
                    cross_task_penalty=self._read_float(payload.get("cross_task_penalty", 0.08), 0.08),
                    recovery_base=self._read_float(payload.get("recovery_base", 0.3), 0.3),
                    allowed_task_families=tuple(str(x) for x in allowed),
                    self_model_fragility_boost=self._read_float(payload.get("self_model_fragility_boost", 0.08), 0.08),
                    self_model_recovery_discount=self._read_float(payload.get("self_model_recovery_discount", 0.05), 0.05),
                )
        self._risk_features = parsed
        self._risk_profile_mtime = path.stat().st_mtime

    def _hot_swap_risk_profile_if_needed(self, context: Dict[str, Any]) -> None:
        path_hint = context.get("risk_profile_path")
        if path_hint:
            next_path = Path(str(path_hint))
            if next_path != self._risk_profile_path:
                self._risk_profile_path = next_path
                self._load_risk_profile(self._risk_profile_path)
                return
        if not self._risk_profile_path.exists():
            return
        current_mtime = self._risk_profile_path.stat().st_mtime
        if self._risk_profile_mtime is None or current_mtime > self._risk_profile_mtime:
            self._load_risk_profile(self._risk_profile_path)

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _read_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _world_model_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        summary = context.get('world_model_summary', {})
        return summary if isinstance(summary, dict) else {}

    def _wait_context_profile(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mirror ValueModel wait-gating logic so early unjustified wait is no longer
        treated as nearly risk-free.
        """
        meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action, dict) else {}
        if not isinstance(meta, dict):
            meta = {}

        existing = meta.get('wait_gate_profile', {})
        if isinstance(existing, dict) and ('suppress_wait' in existing or 'soft_penalty' in existing):
            return existing

        world_model_summary = self._world_model_summary(context)
        plan_summary = context.get('plan_summary', {}) if isinstance(context.get('plan_summary', {}), dict) else {}
        state_dynamics = world_model_summary.get('state_dynamics', {}) if isinstance(world_model_summary.get('state_dynamics', {}), dict) else {}

        tick = int(context.get('tick', 0) or 0)
        recent_failures = int(context.get('recent_failures', 0) or 0)
        reward_trend = str(context.get('reward_trend', 'neutral') or 'neutral').lower()
        predicted_phase = str(world_model_summary.get('predicted_phase', '') or '').lower()

        required_probes = [
            str(item) for item in list(world_model_summary.get('required_probes', []) or [])
            if str(item or '')
        ]
        preferred_action_classes = [
            str(item) for item in list(world_model_summary.get('preferred_action_classes', []) or [])
            if str(item or '')
        ]
        hard_constraints = [
            str(item) for item in list(world_model_summary.get('hard_constraints', []) or [])
            if str(item or '')
        ]

        explicit_wait_justification = any(
            bool(meta.get(key))
            for key in (
                'wait_justified',
                'pending_delayed_effect',
                'cooldown_wait',
                'time_dependent_wait',
            )
        )
        planner_requested_wait = bool(meta.get('planner_matches_step')) and str(meta.get('planner_step_intent', '') or '').lower() == 'wait'
        no_function_surface = bool(meta.get('no_function_surface'))
        filtered_invalid = bool(meta.get('filtered_invalid_call_candidate'))
        injected_no_viable = str(meta.get('wait_injection_reason', '') or '') == 'no_viable_non_wait'
        trace_count = int(state_dynamics.get('trace_count', 0) or 0)

        early_exploration = tick <= 2 and recent_failures <= 0
        exploration_pressure = early_exploration and (
            bool(required_probes)
            or bool(preferred_action_classes)
            or bool(plan_summary.get('has_plan'))
            or predicted_phase in {'exploring', 'stabilizing'}
            or trace_count == 0
        )

        unjustified_wait = not (
            explicit_wait_justification
            or planner_requested_wait
            or no_function_surface
        )

        suppress_wait = exploration_pressure and unjustified_wait and reward_trend != 'positive'
        soft_penalty = (
            not suppress_wait
            and unjustified_wait
            and (early_exploration or filtered_invalid or injected_no_viable)
        )

        profile = {
            'tick': tick,
            'recent_failures': recent_failures,
            'reward_trend': reward_trend,
            'predicted_phase': predicted_phase,
            'required_probes': required_probes[:4],
            'preferred_action_classes': preferred_action_classes[:4],
            'hard_constraints': hard_constraints[:4],
            'planner_requested_wait': planner_requested_wait,
            'explicit_wait_justification': explicit_wait_justification,
            'no_function_surface': no_function_surface,
            'filtered_invalid_call_candidate': filtered_invalid,
            'wait_injection_reason': str(meta.get('wait_injection_reason', '') or ''),
            'suppress_wait': suppress_wait,
            'soft_penalty': soft_penalty,
        }

        if isinstance(candidate.action, dict):
            merged_meta = dict(meta)
            merged_meta['wait_gate_profile'] = profile
            candidate.action['_candidate_meta'] = merged_meta

        return profile

    @property
    def last_uncertainty_profile(self) -> TypedUncertaintyProfile:
        return self._last_uncertainty_profile

    def score(
        self,
        candidate: DecisionCandidate,
        context: Dict[str, Any],
    ) -> RiskScore:
        self._hot_swap_risk_profile_if_needed(context)
        risk_features = self._build_risk_features(candidate, context)
        profile = self._build_uncertainty_profile(candidate, context, risk_features)
        self._last_uncertainty_profile = profile
        uncertainty = profile.total
        failure_likelihood = self._score_failure_likelihood(candidate, context, risk_features)
        recovery_difficulty = self._score_recovery_difficulty(candidate, context, risk_features)

        return RiskScore(
            uncertainty=uncertainty,
            failure_likelihood=failure_likelihood,
            recovery_difficulty=recovery_difficulty,
        )

    def _build_uncertainty_profile(
        self,
        candidate: DecisionCandidate,
        context: Dict[str, Any],
        risk_features: Optional[CandidateRiskFeatures] = None,
    ) -> TypedUncertaintyProfile:
        risk_features = risk_features or self._build_risk_features(candidate, context)
        base = self._score_base_uncertainty(candidate, context, risk_features)
        uncertainty_signals = context.get('uncertainty_signals', {})
        if not isinstance(uncertainty_signals, dict):
            uncertainty_signals = {}

        epistemic = self._read_float(uncertainty_signals.get('epistemic', base), base)
        aleatoric = self._read_float(uncertainty_signals.get('aleatoric', base * 0.8), base * 0.8)
        model = self._read_float(uncertainty_signals.get('model', base * 0.9), base * 0.9)

        epistemic, model = self._apply_counterfactual_uncertainty_adjustment(candidate, epistemic, model)

        if context.get('recovery_pending'):
            epistemic += 0.1
            model += 0.08

        if candidate.is_probe:
            epistemic += 0.12

        if candidate.is_wait:
            wait_profile = self._wait_context_profile(candidate, context)
            if wait_profile['suppress_wait']:
                epistemic = max(epistemic, 0.42)
                aleatoric = max(aleatoric, 0.18)
                model = max(model, 0.40)
            elif wait_profile['soft_penalty']:
                epistemic = max(epistemic, 0.28)
                aleatoric = max(aleatoric, 0.14)
                model = max(model, 0.24)
            else:
                aleatoric = min(aleatoric, 0.12)
                model = min(model, 0.15)

        epistemic += risk_features.uncertainty_bias
        model += max(0.0, risk_features.failure_bias * 0.5)

        return TypedUncertaintyProfile(
            epistemic=self._clamp(epistemic),
            aleatoric=self._clamp(aleatoric),
            model=self._clamp(model),
        )

    def _apply_counterfactual_uncertainty_adjustment(
        self,
        candidate: DecisionCandidate,
        epistemic: float,
        model: float,
    ) -> tuple[float, float]:
        """Apply counterfactual confidence hints to uncertainty components."""
        meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action, dict) else {}
        cf_conf = str(meta.get('counterfactual_confidence', '')).lower()
        if meta.get('counterfactual_advantage'):
            if cf_conf == 'high':
                epistemic -= 0.08
                model -= 0.08
            elif cf_conf == 'medium':
                epistemic -= 0.04
                model -= 0.04
        elif cf_conf == 'low':
            epistemic += 0.05
            model += 0.05
        return epistemic, model

    def _build_risk_features(
        self,
        candidate: DecisionCandidate,
        context: Dict[str, Any],
    ) -> CandidateRiskFeatures:
        fn = str(candidate.function_name or "")
        profile = self._risk_features.get(fn)
        unknown = profile is None
        if profile is None:
            fallback_base = self._function_risks.get(fn, self._default_risk)
            profile = FunctionRiskFeatures(
                base_uncertainty=fallback_base,
                historical_failure_rate=fallback_base,
                recovery_base=max(0.1, fallback_base * 0.9),
            )

        features = CandidateRiskFeatures(function_name=fn, profile=profile, is_unknown_function=unknown)
        task_family = str(context.get("task_family", "") or context.get("continuity_snapshot", {}).get("task_family", "")).strip()
        if task_family and profile.allowed_task_families and task_family not in profile.allowed_task_families:
            features.failure_bias += profile.cross_task_penalty
            features.uncertainty_bias += profile.cross_task_penalty * 0.5

        summary = context.get('self_model_summary', {})
        if bool(context.get('enable_high_level_self_model', True)) and isinstance(summary, dict):
            state = summary.get('self_model_state', {})
            if isinstance(state, dict):
                continuity_confidence = self._read_float(state.get('continuity_confidence', 0.5), 0.5)
                features.failure_bias += (0.6 - continuity_confidence) * 0.15
                features.uncertainty_bias += max(0.0, (0.55 - continuity_confidence) * 0.12)

                known_failure_modes = state.get('known_failure_modes', [])
                if isinstance(known_failure_modes, list) and known_failure_modes:
                    features.failure_bias += 0.03

                fragile_regions = state.get('fragile_regions', [])
                if isinstance(fragile_regions, list):
                    for region in fragile_regions[:5]:
                        if isinstance(region, dict) and str(region.get('function_name', '')) == fn:
                            features.failure_bias += profile.self_model_fragility_boost
                            features.uncertainty_bias += profile.self_model_fragility_boost * 0.4
                            break

                recovered_regions = state.get('recovered_regions', [])
                if isinstance(recovered_regions, list):
                    for region in recovered_regions[:5]:
                        if isinstance(region, dict) and str(region.get('function_name', '')) == fn:
                            features.failure_bias -= profile.self_model_recovery_discount
                            features.recovery_bias -= profile.self_model_recovery_discount
                            break

        self_model_reliability = self._read_float(context.get('self_model_reliability', 0.5), 0.5)
        features.failure_bias += (0.5 - self_model_reliability) * 0.25
        features.uncertainty_bias += max(0.0, (0.5 - self_model_reliability) * 0.15)
        return features

    def _score_base_uncertainty(
        self,
        candidate: DecisionCandidate,
        context: Dict[str, Any],
        risk_features: CandidateRiskFeatures,
    ) -> float:
        from decision.utility_schema import CandidateSource

        if candidate.is_wait:
            wait_profile = self._wait_context_profile(candidate, context)
            if wait_profile['suppress_wait']:
                return 0.34
            if wait_profile['soft_penalty']:
                return 0.18
            return 0.1

        base = risk_features.profile.base_uncertainty

        if candidate.source == CandidateSource.BASE_GENERATION:
            uncertainty = base
        elif candidate.source == CandidateSource.SKILL_REWRITE:
            uncertainty = base * 1.2
        elif candidate.source == CandidateSource.LLM_REWRITE:
            uncertainty = base * 1.3
        elif candidate.source == CandidateSource.ARM_EVALUATION:
            uncertainty = base * 0.9
        elif candidate.source == CandidateSource.RECOVERY:
            uncertainty = base * 1.5
        elif candidate.source == CandidateSource.PROBE:
            uncertainty = base * 1.4
        else:
            uncertainty = base

        return min(1.0, uncertainty)

    def _score_failure_likelihood(
        self,
        candidate: DecisionCandidate,
        context: Dict[str, Any],
        risk_features: CandidateRiskFeatures,
    ) -> float:
        failure = risk_features.profile.historical_failure_rate + risk_features.failure_bias

        if context.get('recovery_pending'):
            failure = min(1.0, failure * 1.5)

        recent_failures = context.get('recent_failures', 0)
        if recent_failures > 0:
            failure = min(1.0, failure + recent_failures * 0.1)

        resource_pressure = str(context.get('resource_pressure', 'normal') or 'normal').lower()
        if resource_pressure in {'tight', 'critical'}:
            pressure_gain = 0.12 + risk_features.profile.resource_pressure_sensitivity * 0.1
            failure = min(1.0, failure + pressure_gain)

        failure_profile = context.get('recent_failure_profile', [])
        if isinstance(failure_profile, list) and failure_profile:
            mode_recent = 0.0
            for item in failure_profile[:3]:
                if isinstance(item, dict):
                    mode_recent += self._read_float(item.get('recent', 0.0), 0.0)
            failure = min(1.0, failure + min(0.2, mode_recent * 0.03))

        if risk_features.is_unknown_function:
            failure = min(1.0, failure + 0.2)

        if candidate.is_wait:
            wait_profile = self._wait_context_profile(candidate, context)
            if wait_profile['suppress_wait']:
                failure = max(failure, 0.35)
            elif wait_profile['soft_penalty']:
                failure = max(failure, 0.18)
            else:
                failure = 0.05

        return min(1.0, failure)

    def _score_recovery_difficulty(
        self,
        candidate: DecisionCandidate,
        context: Dict[str, Any],
        risk_features: CandidateRiskFeatures,
    ) -> float:
        difficulty = risk_features.profile.recovery_base
        difficulty += (1.0 - risk_features.profile.reversibility) * 0.25
        difficulty += risk_features.recovery_bias

        if context.get('recovery_pending'):
            difficulty = difficulty * 0.8

        if candidate.is_wait:
            wait_profile = self._wait_context_profile(candidate, context)
            if wait_profile['suppress_wait']:
                difficulty = max(difficulty, 0.15)
            elif wait_profile['soft_penalty']:
                difficulty = max(difficulty, 0.10)
            else:
                difficulty = 0.05

        return min(1.0, difficulty)

    def is_high_risk(
        self,
        candidate: DecisionCandidate,
        context: Dict[str, Any],
    ) -> bool:
        score = self.score(candidate, context)
        return score.failure_likelihood > 0.6 or score.level == "high"
