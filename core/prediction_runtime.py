from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional

from modules.world_model.counterfactual import CounterfactualEngine, StateSlice


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _extract_function_name(action: Dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return "wait"
    if action.get("kind") == "wait":
        return "wait"
    payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload, dict) else {}
    return str(tool_args.get("function_name", "wait") or "wait")


def _normalize_novel_api(obs: Dict[str, Any]) -> Dict[str, Any]:
    api_raw = obs.get("novel_api", {}) if isinstance(obs, dict) else {}
    if hasattr(api_raw, "raw"):
        api_raw = api_raw.raw
    return api_raw if isinstance(api_raw, dict) else {}


def _normalize_latent_branches(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    branches: List[Dict[str, Any]] = []
    for item in raw[:4]:
        if not isinstance(item, dict):
            continue
        branches.append({
            "branch_id": str(item.get("branch_id", "") or ""),
            "current_phase": str(item.get("current_phase", "") or ""),
            "target_phase": str(item.get("target_phase", "") or ""),
            "confidence": _clamp(float(item.get("confidence", 0.0) or 0.0)),
            "support": _clamp(float(item.get("support", 0.0) or 0.0)),
            "transition_score": _clamp(float(item.get("transition_score", 0.0) or 0.0)),
            "success_rate": _clamp(float(item.get("success_rate", 0.0) or 0.0)),
            "avg_reward": float(item.get("avg_reward", 0.0) or 0.0),
            "avg_depth_gain": _clamp(float(item.get("avg_depth_gain", 0.0) or 0.0)),
            "uncertainty_pressure": _clamp(float(item.get("uncertainty_pressure", 0.0) or 0.0)),
            "anchor_functions": [
                str(value) for value in list(item.get("anchor_functions", []) or [])[:4]
                if str(value or "")
            ],
            "risky_functions": [
                str(value) for value in list(item.get("risky_functions", []) or [])[:4]
                if str(value or "")
            ],
            "latent_signature": str(item.get("latent_signature", "") or ""),
        })
    return branches


@dataclass
class PredictionValue:
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value}


@dataclass
class PredictionBundle:
    action_id: str
    function_name: str
    success: PredictionValue
    information_gain: PredictionValue
    reward_sign: PredictionValue
    risk_type: PredictionValue
    overall_confidence: float
    source: str = "ensemble_transition_model"
    state_context: Dict[str, Any] = field(default_factory=dict)
    predictor_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "function_name": self.function_name,
            "success": self.success.to_dict(),
            "information_gain": self.information_gain.to_dict(),
            "reward_sign": self.reward_sign.to_dict(),
            "risk_type": self.risk_type.to_dict(),
            "overall_confidence": float(self.overall_confidence),
            "source": self.source,
            "state_context": dict(self.state_context),
            "predictor_details": {
                str(name): dict(detail) for name, detail in self.predictor_details.items()
            },
        }


@dataclass
class PredictionOutcome:
    episode: int
    tick: int
    action_id: str
    function_name: str
    actual_success: bool
    actual_information_gain: float
    actual_reward_sign: str
    actual_risk_type: str
    actual_reward: float
    hypothesis_delta: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": int(self.episode),
            "tick": int(self.tick),
            "action_id": self.action_id,
            "function_name": self.function_name,
            "actual_success": bool(self.actual_success),
            "actual_information_gain": float(self.actual_information_gain),
            "actual_reward_sign": self.actual_reward_sign,
            "actual_risk_type": self.actual_risk_type,
            "actual_reward": float(self.actual_reward),
            "hypothesis_delta": int(self.hypothesis_delta),
        }


@dataclass
class PredictionError:
    action_id: str
    function_name: str
    success_error: float
    information_gain_error: float
    reward_sign_error: float
    risk_type_error: float
    total_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "function_name": self.function_name,
            "success_error": float(self.success_error),
            "information_gain_error": float(self.information_gain_error),
            "reward_sign_error": float(self.reward_sign_error),
            "risk_type_error": float(self.risk_type_error),
            "total_error": float(self.total_error),
        }


class _FunctionHistoryPredictor:
    name = "function_history"

    def __init__(self) -> None:
        self._stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "success_sum": 0.0, "info_gain_sum": 0.0, "reward_sum": 0.0}
        )

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        fn_name = str(context.get("function_name", "wait") or "wait")
        stats = self._stats.get(fn_name, {})
        count = float(stats.get("count", 0.0) or 0.0)
        candidate_meta = context.get("candidate_meta", {}) if isinstance(context.get("candidate_meta", {}), dict) else {}
        cf_delta = float(candidate_meta.get("counterfactual_delta", 0.0) or 0.0)
        cf_advantage = bool(candidate_meta.get("counterfactual_advantage", False))
        recent_functions = set(context.get("recent_functions", []))
        reliability = float(context.get("global_reliability", 0.5) or 0.5)
        shift_risk = float(context.get("shift_risk", 0.0) or 0.0)
        hidden_phase = str(context.get("hidden_phase", "") or "")
        hidden_phase_confidence = float(context.get("hidden_phase_confidence", 0.0) or 0.0)
        hidden_state_depth = int(context.get("hidden_state_depth", 0) or 0)
        hidden_drift_score = float(context.get("hidden_drift_score", 0.0) or 0.0)
        focus_functions = set(context.get("focus_functions", []) or [])

        if count > 0:
            success = (float(stats.get("success_sum", 0.0)) + 1.0) / (count + 2.0)
            info_gain = float(stats.get("info_gain_sum", 0.0)) / max(count, 1.0)
            reward_avg = float(stats.get("reward_sum", 0.0)) / max(count, 1.0)
        else:
            success = 0.5
            info_gain = 0.08
            reward_avg = 0.0

        success += (cf_delta * 0.18) + (0.06 if cf_advantage else 0.0) + ((reliability - 0.5) * 0.12)
        if fn_name in recent_functions:
            success -= 0.04
        else:
            info_gain += 0.12

        if hidden_phase == "committed":
            success += 0.08 * max(0.35, hidden_phase_confidence)
            info_gain -= 0.03
        elif hidden_phase == "stabilizing":
            success += 0.05 * max(0.35, hidden_phase_confidence)
            info_gain += 0.02
        elif hidden_phase == "exploring":
            info_gain += 0.07
        elif hidden_phase == "disrupted":
            success -= 0.14 * max(0.35, hidden_phase_confidence)
            info_gain += 0.04
        if fn_name in focus_functions and hidden_state_depth > 0 and hidden_phase != "disrupted":
            success += min(0.1, hidden_state_depth * 0.025)
        success -= hidden_drift_score * 0.16

        reward_avg += (cf_delta * 0.25) + (0.08 if cf_advantage else 0.0)
        success = _clamp(success)
        info_gain = _clamp(info_gain)

        reward_sign = "positive" if reward_avg > 0.12 else "negative" if reward_avg < -0.12 else "zero"
        risk_type = "execution_failure" if success < 0.45 else "state_shift" if shift_risk > 0.45 else "opportunity_cost"
        confidence = _clamp(0.42 + min(0.28, count * 0.05) + abs(cf_delta) * 0.12, 0.2, 0.95)
        return {
            "success": success,
            "information_gain": info_gain,
            "reward_sign": reward_sign,
            "risk_type": risk_type,
            "confidence": confidence,
        }

    def update(self, bundle: PredictionBundle, outcome: PredictionOutcome, error: PredictionError) -> None:
        fn_name = str(outcome.function_name or bundle.function_name or "wait")
        stats = self._stats[fn_name]
        stats["count"] += 1.0
        stats["success_sum"] += 1.0 if outcome.actual_success else 0.0
        stats["info_gain_sum"] += float(outcome.actual_information_gain)
        stats["reward_sum"] += float(outcome.actual_reward)


class _ContextualTransitionPredictor:
    name = "contextual_transition"

    def __init__(self) -> None:
        self._function_bias: Dict[str, float] = defaultdict(float)

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        fn_name = str(context.get("function_name", "wait") or "wait")
        recent_functions = set(context.get("recent_functions", []))
        uncertain_count = int(context.get("uncertain_count", 0) or 0)
        discovered = set(context.get("discovered_functions", []))
        step_intent = str(context.get("step_intent", "") or "")
        shift_risk = float(context.get("shift_risk", 0.0) or 0.0)
        recovery_pending = bool(context.get("recovery_pending", False))
        function_bias = float(self._function_bias.get(fn_name, 0.0) or 0.0)
        hidden_phase = str(context.get("hidden_phase", "") or "")
        hidden_phase_confidence = float(context.get("hidden_phase_confidence", 0.0) or 0.0)
        hidden_state_depth = int(context.get("hidden_state_depth", 0) or 0)
        hidden_drift_score = float(context.get("hidden_drift_score", 0.0) or 0.0)
        focus_functions = set(context.get("focus_functions", []) or [])

        success = 0.52 + function_bias
        info_gain = 0.06

        if fn_name == "wait":
            success = 0.72 if recovery_pending or shift_risk > 0.55 else 0.34
            info_gain = 0.02
        else:
            if fn_name not in recent_functions:
                info_gain += 0.16
            if fn_name not in discovered:
                info_gain += 0.12
            info_gain += min(0.18, uncertain_count * 0.06)

            lowered = fn_name.lower()
            if any(token in lowered for token in ("inspect", "probe", "sample", "test")):
                info_gain += 0.12
                success -= 0.04
            if step_intent and step_intent in lowered:
                success += 0.1
            if recovery_pending:
                success -= 0.08

        if hidden_phase == "stabilizing":
            success += 0.04 * max(0.35, hidden_phase_confidence)
        elif hidden_phase == "committed":
            success += 0.07 * max(0.35, hidden_phase_confidence)
            info_gain -= 0.03
        elif hidden_phase == "disrupted":
            success -= 0.12 * max(0.35, hidden_phase_confidence)
            info_gain += 0.05
        elif hidden_phase == "exploring":
            info_gain += 0.08
        if fn_name in focus_functions and hidden_state_depth > 0:
            success += min(0.09, hidden_state_depth * 0.03)
        success -= hidden_drift_score * 0.18

        success = _clamp(success)
        info_gain = _clamp(info_gain)
        reward_sign = "positive" if success > 0.68 else "negative" if success < 0.34 else "zero"
        risk_type = "state_shift" if shift_risk > 0.45 or info_gain > 0.48 else "execution_failure" if success < 0.45 else "opportunity_cost"
        confidence = _clamp(0.46 + min(0.18, uncertain_count * 0.03) + (0.08 if step_intent else 0.0), 0.25, 0.9)
        return {
            "success": success,
            "information_gain": info_gain,
            "reward_sign": reward_sign,
            "risk_type": risk_type,
            "confidence": confidence,
        }

    def update(self, bundle: PredictionBundle, outcome: PredictionOutcome, error: PredictionError) -> None:
        fn_name = str(outcome.function_name or bundle.function_name or "wait")
        current = float(self._function_bias.get(fn_name, 0.0) or 0.0)
        if outcome.actual_success and outcome.actual_reward_sign != "negative":
            current = min(0.18, current * 0.7 + 0.05)
        else:
            current = max(-0.22, current * 0.7 - 0.08)
        self._function_bias[fn_name] = current


class _HiddenStateRolloutPredictor:
    name = "hidden_state_rollout"

    def __init__(self) -> None:
        self._rollout_engine = CounterfactualEngine(seed=0)

    def _rollout_preview(
        self,
        *,
        context: Dict[str, Any],
        hidden_phase: str,
        hidden_phase_confidence: float,
        hidden_transition_memory: Dict[str, Any],
        latent_branches: List[Dict[str, Any]],
        dominant_branch_id: str,
        expected_next_phase: str,
        expected_next_phase_confidence: float,
        transition_entropy: float,
        fn_name: str,
    ) -> Dict[str, Any]:
        if not hidden_transition_memory and not latent_branches and not hidden_phase:
            return {}

        hidden_state_payload = {
            "phase": hidden_phase,
            "phase_confidence": hidden_phase_confidence,
            "expected_next_phase": expected_next_phase,
            "expected_next_phase_confidence": expected_next_phase_confidence,
            "transition_entropy": transition_entropy,
            "dominant_branch_id": dominant_branch_id,
            "latent_branches": [dict(item) for item in latent_branches if isinstance(item, dict)],
            "transition_memory": dict(hidden_transition_memory) if isinstance(hidden_transition_memory, dict) else {},
        }
        state_slice = StateSlice(
            available_functions=[
                str(value) for value in list(context.get("discovered_functions", []) or [])[:8]
                if str(value or "")
            ],
            recent_actions=[
                str(value) for value in list(context.get("recent_functions", []) or [])[-5:]
                if str(value or "")
            ],
            state_features={
                "world_phase": hidden_phase,
                "hidden_state": hidden_state_payload,
            },
        )
        rollout_context = {
            "transition_priors": {
                "__world_dynamics": {
                    "predicted_phase": hidden_phase,
                    "transition_confidence": hidden_phase_confidence,
                    "expected_next_phase": expected_next_phase,
                    "expected_next_phase_confidence": expected_next_phase_confidence,
                    "phase_transition_entropy": transition_entropy,
                    "dominant_branch_id": dominant_branch_id,
                    "latent_branches": [dict(item) for item in latent_branches if isinstance(item, dict)],
                    "hidden_state": hidden_state_payload,
                    "transition_memory": dict(hidden_transition_memory) if isinstance(hidden_transition_memory, dict) else {},
                }
            }
        }
        return dict(
            self._rollout_engine._simulate_hidden_state_rollout(
                state_slice=state_slice,
                action={
                    "kind": "call_tool",
                    "payload": {"tool_args": {"function_name": fn_name, "kwargs": {}}},
                },
                context=rollout_context,
                horizon=3,
            )
            or {}
        )

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        fn_name = str(context.get("function_name", "wait") or "wait")
        hidden_phase = str(context.get("hidden_phase", "") or "")
        expected_next_phase = str(context.get("expected_next_phase", "") or "")
        expected_next_phase_confidence = float(context.get("expected_next_phase_confidence", 0.0) or 0.0)
        transition_entropy = float(context.get("phase_transition_entropy", 1.0) or 1.0)
        hidden_drift_score = float(context.get("hidden_drift_score", 0.0) or 0.0)
        latent_branches = _normalize_latent_branches(context.get("latent_branches", []))
        dominant_branch_id = str(context.get("dominant_branch_id", "") or "")
        hidden_transition_memory = context.get("hidden_transition_memory", {})
        phase_function_scores = (
            hidden_transition_memory.get("phase_function_scores", {})
            if isinstance(hidden_transition_memory, dict)
            and isinstance(hidden_transition_memory.get("phase_function_scores", {}), dict)
            else {}
        )
        fn_stats = phase_function_scores.get(fn_name, {}) if isinstance(phase_function_scores.get(fn_name, {}), dict) else {}
        support = float(fn_stats.get("support", 0.0) or 0.0)
        support_factor = _clamp(support / 3.0, 0.0, 1.0)
        stabilizing_score = _clamp(float(fn_stats.get("stabilizing_score", 0.0) or 0.0))
        risk_score = _clamp(float(fn_stats.get("risk_score", 0.0) or 0.0))
        commit_rate = _clamp(float(fn_stats.get("committed_rate", 0.0) or 0.0))
        depth_gain = _clamp(float(fn_stats.get("avg_depth_gain", 0.0) or 0.0))
        success_rate = _clamp(float(fn_stats.get("success_rate", 0.5) or 0.5))
        best_branch: Dict[str, Any] = {}
        best_branch_affinity = 0.0
        dominant_branch: Dict[str, Any] = {}
        for branch in latent_branches:
            branch_id = str(branch.get("branch_id", "") or "")
            if branch_id and branch_id == dominant_branch_id:
                dominant_branch = dict(branch)
            anchor_functions = set(branch.get("anchor_functions", []) or [])
            risky_functions = set(branch.get("risky_functions", []) or [])
            branch_confidence = _clamp(float(branch.get("confidence", 0.0) or 0.0))
            branch_affinity = 0.0
            if fn_name in anchor_functions:
                branch_affinity += 0.44 + branch_confidence * 0.18
            if fn_name in risky_functions:
                branch_affinity -= 0.38 + branch_confidence * 0.12
            if expected_next_phase and str(branch.get("target_phase", "") or "") == expected_next_phase:
                branch_affinity += expected_next_phase_confidence * 0.16
            if hidden_phase and str(branch.get("current_phase", "") or "") == hidden_phase:
                branch_affinity += 0.05
            if branch_id and branch_id == dominant_branch_id:
                branch_affinity += 0.08
            if branch_affinity > best_branch_affinity:
                best_branch_affinity = branch_affinity
                best_branch = dict(branch)
        if not dominant_branch and latent_branches:
            dominant_branch = dict(latent_branches[0])
        rollout_preview = self._rollout_preview(
            context=context,
            hidden_phase=hidden_phase,
            hidden_phase_confidence=float(context.get("hidden_phase_confidence", 0.0) or 0.0),
            hidden_transition_memory=hidden_transition_memory if isinstance(hidden_transition_memory, dict) else {},
            latent_branches=latent_branches,
            dominant_branch_id=dominant_branch_id,
            expected_next_phase=expected_next_phase,
            expected_next_phase_confidence=expected_next_phase_confidence,
            transition_entropy=transition_entropy,
            fn_name=fn_name,
        )

        success = 0.48
        info_gain = 0.05
        if fn_stats:
            success += (stabilizing_score - risk_score) * 0.32
            success += commit_rate * 0.14
            success += success_rate * 0.08
            info_gain += depth_gain * 0.18
            info_gain += (1.0 - stabilizing_score) * 0.04
        if best_branch:
            branch_confidence = _clamp(float(best_branch.get("confidence", 0.0) or 0.0))
            branch_target_phase = str(best_branch.get("target_phase", "") or "")
            if best_branch_affinity > 0.0:
                success += best_branch_affinity * 0.16
                if branch_target_phase in {"exploring", "disrupted"}:
                    info_gain += 0.04 + branch_confidence * 0.10
                else:
                    success += branch_confidence * 0.08
            elif best_branch_affinity < 0.0:
                success += best_branch_affinity * 0.10
        if dominant_branch:
            dominant_branch_target = str(dominant_branch.get("target_phase", "") or "")
            dominant_branch_confidence = _clamp(float(dominant_branch.get("confidence", 0.0) or 0.0))
            dominant_risky_functions = set(dominant_branch.get("risky_functions", []) or [])
            dominant_anchor_functions = set(dominant_branch.get("anchor_functions", []) or [])
            if fn_name in dominant_risky_functions and dominant_branch_target in {"stabilizing", "committed"}:
                success -= 0.10 + dominant_branch_confidence * 0.12
            if fn_name in dominant_anchor_functions and dominant_branch_target in {"exploring", "disrupted"}:
                info_gain += 0.04 + dominant_branch_confidence * 0.08

        if expected_next_phase in {"stabilizing", "committed"} and fn_stats:
            success += expected_next_phase_confidence * max(0.0, stabilizing_score - risk_score) * 0.24
        elif expected_next_phase == "disrupted" and fn_name != "wait":
            success -= max(0.0, 0.10 + hidden_drift_score * 0.12)
        if hidden_phase == "exploring":
            info_gain += 0.05
        if transition_entropy >= 0.65 and fn_name != "wait":
            info_gain += 0.05
        rollout_branch_id = str(rollout_preview.get("rollout_branch_id", "") or "")
        rollout_branch_target_phase = str(rollout_preview.get("rollout_branch_target_phase", "") or "")
        rollout_branch_confidence = _clamp(float(rollout_preview.get("rollout_branch_confidence", 0.0) or 0.0))
        branch_persistence_ratio = _clamp(float(rollout_preview.get("branch_persistence_ratio", 0.0) or 0.0))
        phase_path = [
            str(value or "") for value in list(rollout_preview.get("phase_path", []) or [])[:4]
            if str(value or "")
        ]
        anchor_path = [
            str(value or "") for value in list(rollout_preview.get("anchor_path", []) or [])[:4]
            if str(value or "")
        ]
        rollout_trace = list(rollout_preview.get("trace", []) or [])
        if rollout_trace:
            avg_step_value = sum(float(row.get("step_value", 0.0) or 0.0) for row in rollout_trace) / float(len(rollout_trace))
            avg_step_risk = sum(float(row.get("step_risk", 0.0) or 0.0) for row in rollout_trace) / float(len(rollout_trace))
            final_phase = str(phase_path[-1] if phase_path else hidden_phase or "")
            rollout_confidence = _clamp(float(rollout_preview.get("confidence", 0.0) or 0.0))
            success += avg_step_value * 0.18
            success -= avg_step_risk * 0.14
            success += branch_persistence_ratio * 0.10
            if final_phase == "committed":
                success += 0.08 + rollout_branch_confidence * 0.06
            elif final_phase == "stabilizing":
                success += 0.03 + branch_persistence_ratio * 0.04
            elif final_phase == "disrupted":
                success -= 0.12 + avg_step_risk * 0.10
            if rollout_branch_target_phase in {"exploring", "disrupted"}:
                info_gain += 0.04 + rollout_branch_confidence * 0.08
            elif branch_persistence_ratio >= 0.5 and rollout_branch_target_phase == "committed":
                info_gain += 0.02 + rollout_branch_confidence * 0.03
            if final_phase == "exploring":
                info_gain += 0.04
            elif final_phase == "disrupted":
                info_gain += 0.02
        else:
            rollout_confidence = 0.0
            final_phase = hidden_phase
        success -= hidden_drift_score * 0.16

        success = _clamp(success)
        info_gain = _clamp(info_gain)
        reward_sign = "positive" if success > 0.68 else "negative" if success < 0.34 else "zero"
        dominant_branch_target = str(dominant_branch.get("target_phase", "") or "") if dominant_branch else ""
        dominant_branch_confidence = _clamp(float(dominant_branch.get("confidence", 0.0) or 0.0)) if dominant_branch else 0.0
        risk_type = (
            "state_shift"
            if risk_score >= 0.55
            or expected_next_phase == "disrupted"
            or dominant_branch_target == "disrupted"
            or final_phase == "disrupted"
            else "execution_failure" if success < 0.45 else "opportunity_cost"
        )
        confidence = _clamp(
            0.26
            + support_factor * 0.34
            + expected_next_phase_confidence * 0.22
            + (1.0 - transition_entropy) * 0.10
            + dominant_branch_confidence * 0.10
            + max(0.0, best_branch_affinity) * 0.06,
            0.18,
            0.92,
        )
        confidence = _clamp(confidence + rollout_confidence * 0.14 + branch_persistence_ratio * 0.08, 0.18, 0.95)
        return {
            "success": success,
            "information_gain": info_gain,
            "reward_sign": reward_sign,
            "risk_type": risk_type,
            "confidence": confidence,
            "rollout_branch_id": rollout_branch_id,
            "rollout_branch_target_phase": rollout_branch_target_phase,
            "rollout_branch_confidence": rollout_branch_confidence,
            "branch_persistence_ratio": branch_persistence_ratio,
            "phase_path": phase_path,
            "anchor_path": anchor_path,
        }

    def update(self, bundle: PredictionBundle, outcome: PredictionOutcome, error: PredictionError) -> None:
        return None


class PredictionEngine:
    """Lightweight online prediction ensemble used by the main loop."""

    def __init__(self) -> None:
        self.predictors = [
            _FunctionHistoryPredictor(),
            _ContextualTransitionPredictor(),
            _HiddenStateRolloutPredictor(),
        ]

    def predict_candidates(self, *, candidate_actions: List[Dict[str, Any]], **context: Any) -> Dict[str, PredictionBundle]:
        bundles: Dict[str, PredictionBundle] = {}
        for action in candidate_actions:
            bundle = self.predict_action(action=action, **context)
            bundles[bundle.action_id] = bundle
        return bundles

    def predict_action(self, *, action: Dict[str, Any], **context: Any) -> PredictionBundle:
        action_id = str(action.get("_action_id", "") or "")
        function_name = _extract_function_name(action)
        candidate_meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
        obs = context.get("obs", {}) if isinstance(context.get("obs", {}), dict) else {}
        novel_api = _normalize_novel_api(obs)
        discovered_functions = list(novel_api.get("discovered_functions", []) if isinstance(novel_api.get("discovered_functions", []), list) else [])
        recent_trace = list(context.get("recent_trace", []) or [])
        recent_functions = []
        for entry in recent_trace[-5:]:
            if not isinstance(entry, dict):
                continue
            action_payload = entry.get("action", {})
            recent_functions.append(_extract_function_name(action_payload if isinstance(action_payload, dict) else {}))
        belief_summary = context.get("belief_summary", {}) if isinstance(context.get("belief_summary", {}), dict) else {}
        hidden_state = belief_summary.get("hidden_state", {}) if isinstance(belief_summary.get("hidden_state", {}), dict) else {}
        self_model_summary = context.get("self_model_summary", {}) if isinstance(context.get("self_model_summary", {}), dict) else {}
        reliability_subscores = self_model_summary.get("reliability_subscores", {}) if isinstance(self_model_summary.get("reliability_subscores", {}), dict) else {}
        global_reliability = float(
            reliability_subscores.get("global_reliability", self_model_summary.get("global_reliability", 0.5)) or 0.5
        )
        hidden_phase = str(hidden_state.get("phase", "") or "")
        hidden_phase_confidence = _clamp(float(hidden_state.get("phase_confidence", 0.0) or 0.0))
        hidden_state_depth = max(0, int(hidden_state.get("hidden_state_depth", 0) or 0))
        hidden_drift_score = _clamp(float(hidden_state.get("drift_score", 0.0) or 0.0))
        hidden_uncertainty_score = _clamp(float(hidden_state.get("uncertainty_score", 0.0) or 0.0))
        focus_functions = [
            str(value) for value in list(hidden_state.get("focus_functions", []) or [])[:3]
            if str(value or "")
        ]
        hidden_transition_memory = hidden_state.get("transition_memory", {}) if isinstance(hidden_state.get("transition_memory", {}), dict) else {}
        latent_branches = _normalize_latent_branches(hidden_state.get("latent_branches", hidden_transition_memory.get("latent_branches", [])))
        dominant_branch_id = str(
            hidden_state.get(
                "dominant_branch_id",
                hidden_transition_memory.get("dominant_branch_id", latent_branches[0].get("branch_id", "") if latent_branches else ""),
            ) or ""
        )
        expected_next_phase = str(hidden_state.get("expected_next_phase", hidden_transition_memory.get("expected_next_phase", "")) or "")
        expected_next_phase_confidence = _clamp(float(hidden_state.get("expected_next_phase_confidence", hidden_transition_memory.get("expected_next_phase_confidence", 0.0)) or 0.0))
        phase_transition_entropy = _clamp(float(hidden_state.get("transition_entropy", hidden_transition_memory.get("phase_transition_entropy", 1.0)) or 1.0))
        shared_context = {
            "function_name": function_name,
            "candidate_meta": candidate_meta,
            "recent_functions": recent_functions,
            "global_reliability": global_reliability,
            "shift_risk": float(belief_summary.get("shift_risk", 0.0) or 0.0),
            "uncertain_count": len(list(belief_summary.get("uncertain_high_impact_beliefs", []) or [])),
            "discovered_functions": discovered_functions,
            "step_intent": context.get("step_intent", ""),
            "recovery_pending": bool((context.get("recovery_context", {}) or {}).get("pending_replan") or (context.get("recovery_context", {}) or {}).get("pending_recovery_probe")),
            "hidden_phase": hidden_phase,
            "hidden_phase_confidence": hidden_phase_confidence,
            "hidden_state_depth": hidden_state_depth,
            "hidden_drift_score": hidden_drift_score,
            "hidden_uncertainty_score": hidden_uncertainty_score,
            "focus_functions": focus_functions,
            "hidden_transition_memory": hidden_transition_memory,
            "latent_branches": latent_branches,
            "dominant_branch_id": dominant_branch_id,
            "expected_next_phase": expected_next_phase,
            "expected_next_phase_confidence": expected_next_phase_confidence,
            "phase_transition_entropy": phase_transition_entropy,
        }
        state_context = {
            "predicted_phase": str(belief_summary.get("predicted_phase", "") or ""),
            "hidden_phase": hidden_phase,
            "hidden_phase_confidence": hidden_phase_confidence,
            "hidden_state_depth": hidden_state_depth,
            "hidden_drift_score": hidden_drift_score,
            "hidden_uncertainty_score": hidden_uncertainty_score,
            "focus_functions": focus_functions,
            "latent_signature": str(hidden_state.get("latent_signature", "") or ""),
            "dominant_branch_id": dominant_branch_id,
            "latent_branches": latent_branches,
            "expected_next_phase": expected_next_phase,
            "expected_next_phase_confidence": expected_next_phase_confidence,
            "phase_transition_entropy": phase_transition_entropy,
        }

        predictor_details: Dict[str, Dict[str, Any]] = {}
        weighted_success = 0.0
        weighted_info_gain = 0.0
        total_weight = 0.0
        reward_votes: Dict[str, float] = defaultdict(float)
        risk_votes: Dict[str, float] = defaultdict(float)

        for predictor in self.predictors:
            raw = predictor.predict(shared_context)
            confidence = _clamp(float(raw.get("confidence", 0.5) or 0.5), 0.1, 1.0)
            success = _clamp(float(raw.get("success", 0.5) or 0.5))
            info_gain = _clamp(float(raw.get("information_gain", 0.0) or 0.0))
            reward_sign = str(raw.get("reward_sign", "zero") or "zero")
            risk_type = str(raw.get("risk_type", "execution_failure") or "execution_failure")
            detail = {
                "success": success,
                "information_gain": info_gain,
                "reward_sign": reward_sign,
                "risk_type": risk_type,
                "confidence": confidence,
            }
            for extra_key in (
                "rollout_branch_id",
                "rollout_branch_target_phase",
                "rollout_branch_confidence",
                "branch_persistence_ratio",
                "phase_path",
                "anchor_path",
            ):
                if extra_key not in raw:
                    continue
                value = raw.get(extra_key)
                if isinstance(value, list):
                    detail[extra_key] = list(value)
                elif isinstance(value, (str, float, int, bool)) or value is None:
                    detail[extra_key] = value
            predictor_details[predictor.name] = detail
            weighted_success += success * confidence
            weighted_info_gain += info_gain * confidence
            total_weight += confidence
            reward_votes[reward_sign] += confidence
            risk_votes[risk_type] += confidence

        if total_weight <= 0.0:
            total_weight = 1.0
        overall_confidence = _clamp(total_weight / max(len(self.predictors), 1), 0.2, 0.95)
        reward_sign = max(reward_votes.items(), key=lambda item: item[1])[0] if reward_votes else "zero"
        risk_type = max(risk_votes.items(), key=lambda item: item[1])[0] if risk_votes else "execution_failure"
        return PredictionBundle(
            action_id=action_id,
            function_name=function_name,
            success=PredictionValue(_clamp(weighted_success / total_weight)),
            information_gain=PredictionValue(_clamp(weighted_info_gain / total_weight)),
            reward_sign=PredictionValue(reward_sign),
            risk_type=PredictionValue(risk_type),
            overall_confidence=overall_confidence,
            state_context=state_context,
            predictor_details=predictor_details,
        )


class PredictionAdjudicator:
    """Builds post-action outcomes and compares them against prediction bundles."""

    def build_outcome_record(
        self,
        *,
        episode: int,
        tick: int,
        action_id: str,
        function_name: str,
        result: Dict[str, Any],
        reward: float,
        obs_before: Dict[str, Any],
        hypotheses_before: Iterable[Any],
        hypotheses_after: Iterable[Any],
    ) -> PredictionOutcome:
        before_api = _normalize_novel_api(obs_before)
        after_api = _normalize_novel_api(result if isinstance(result, dict) else {})
        before_discovered = set(before_api.get("discovered_functions", []) or [])
        after_discovered = set(after_api.get("discovered_functions", []) or [])
        discovered_delta = max(0, len(after_discovered - before_discovered))
        hyp_before_count = len(list(hypotheses_before or []))
        hyp_after_count = len(list(hypotheses_after or []))
        hypothesis_delta = abs(hyp_after_count - hyp_before_count)

        actual_success = bool(result.get("success", True)) and float(reward or 0.0) >= 0.0
        actual_information_gain = _clamp((discovered_delta * 0.55) + (0.12 if hypothesis_delta > 0 else 0.0))
        actual_reward_sign = "positive" if reward > 0.05 else "negative" if reward < -0.05 else "zero"

        err = result.get("error", {}) if isinstance(result.get("error", {}), dict) else {}
        if str(err.get("type", "") or "").lower():
            actual_risk_type = "execution_failure"
        elif actual_information_gain > 0.45:
            actual_risk_type = "state_shift"
        elif actual_reward_sign == "negative":
            actual_risk_type = "execution_failure"
        else:
            actual_risk_type = "opportunity_cost"

        return PredictionOutcome(
            episode=int(episode),
            tick=int(tick),
            action_id=str(action_id or ""),
            function_name=str(function_name or ""),
            actual_success=actual_success,
            actual_information_gain=actual_information_gain,
            actual_reward_sign=actual_reward_sign,
            actual_risk_type=actual_risk_type,
            actual_reward=float(reward or 0.0),
            hypothesis_delta=hypothesis_delta,
        )

    def compare(self, bundle: PredictionBundle, outcome: PredictionOutcome) -> PredictionError:
        success_error = abs(float(bundle.success.value) - (1.0 if outcome.actual_success else 0.0))
        information_gain_error = abs(float(bundle.information_gain.value) - float(outcome.actual_information_gain))
        reward_sign_error = 0.0 if str(bundle.reward_sign.value) == str(outcome.actual_reward_sign) else 1.0
        risk_type_error = 0.0 if str(bundle.risk_type.value) == str(outcome.actual_risk_type) else 1.0
        total_error = _clamp(
            (success_error * 0.45)
            + (information_gain_error * 0.25)
            + (reward_sign_error * 0.2)
            + (risk_type_error * 0.1)
        )
        return PredictionError(
            action_id=bundle.action_id,
            function_name=bundle.function_name,
            success_error=success_error,
            information_gain_error=information_gain_error,
            reward_sign_error=reward_sign_error,
            risk_type_error=risk_type_error,
            total_error=total_error,
        )


class PredictionRegistry:
    """Stores prediction history and maintains simple calibration/trust summaries."""

    def __init__(self, *, max_history: int = 200) -> None:
        self._predictions: Deque[PredictionBundle] = deque(maxlen=max_history)
        self._outcomes: Deque[PredictionOutcome] = deque(maxlen=max_history)
        self._errors: Deque[PredictionError] = deque(maxlen=max_history)
        self._predictor_error_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=max_history))

    def record_prediction(self, bundle: PredictionBundle) -> None:
        self._predictions.append(bundle)

    def record_outcome(self, outcome: PredictionOutcome) -> None:
        self._outcomes.append(outcome)

    def record_error(self, error: PredictionError) -> None:
        self._errors.append(error)

    def update_calibration(self, bundle: PredictionBundle, outcome: PredictionOutcome, error: PredictionError) -> None:
        actual_success = 1.0 if outcome.actual_success else 0.0
        for name, detail in bundle.predictor_details.items():
            success_error = abs(float(detail.get("success", bundle.success.value) or 0.0) - actual_success)
            info_error = abs(float(detail.get("information_gain", bundle.information_gain.value) or 0.0) - float(outcome.actual_information_gain))
            reward_error = 0.0 if str(detail.get("reward_sign", bundle.reward_sign.value) or "zero") == outcome.actual_reward_sign else 1.0
            predictor_error = _clamp((success_error * 0.5) + (info_error * 0.3) + (reward_error * 0.2))
            self._predictor_error_history[str(name)].append(predictor_error)
        if not bundle.predictor_details:
            self._predictor_error_history["ensemble"].append(float(error.total_error))

    def get_recent_errors(self, n: Optional[int] = None) -> List[PredictionError]:
        if n is None:
            return list(self._errors)
        return list(self._errors)[-max(0, int(n or 0)) :]

    def get_predictor_trust(self) -> Dict[str, str]:
        trust: Dict[str, str] = {}
        for name, history in self._predictor_error_history.items():
            if not history:
                trust[name] = "medium"
                continue
            avg_error = sum(float(item) for item in history) / max(len(history), 1)
            if avg_error <= 0.25:
                trust[name] = "high"
            elif avg_error <= 0.5:
                trust[name] = "medium"
            else:
                trust[name] = "low"
        return trust

    def summarize(self) -> Dict[str, Any]:
        return {
            "type": "prediction_registry",
            "prediction_count": len(self._predictions),
            "outcome_count": len(self._outcomes),
            "error_count": len(self._errors),
            "predictor_trust": self.get_predictor_trust(),
        }
