"""
governance/gate.py — 最小治理门实现

根据 specs/module_contracts.md 和 specs/main_loop.md 实现。

只做三件事：
1. hard veto — 硬约束拦截
2. fallback — 所有候选被否决时的安全动作
3. mode switch — normal / cautious 模式切换
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from modules.world_model.protocol import WorldModelControlProtocol


# =============================================================================
# Serialization helpers — keep formal state free of rich objects
# =============================================================================

# Fields that are safe to write to formal state (all JSON-serializable)
_SAFE_ACTION_FIELDS = frozenset([
    'action', 'action_type', 'score', 'final_score',
    'intent', 'risk', 'opportunity_estimate',
    'wm_long_reward', 'wm_risk', 'wm_reversibility', 'wm_info_gain', 'wm_score_delta',
    'lineage', 'candidate_source', 'chain', 'step',
    '_arm_block_reason', '_object_consumption_allowed',
    '_contradiction_boosted', '_contract_boosted', '_contract_reason',
])
_PROTECTED_ACTIVE_STEP_CONFIDENCE = 0.78
_LOW_TRUST_PROTECTION_THRESHOLD = 0.72
_DOMINANT_BRANCH_HARD_VETO_CONFIDENCE = 0.74
_DOMINANT_BRANCH_MIN_TRUST = 0.45


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _active_procedure_strength(candidate: Dict[str, Any]) -> float:
    raw_action = candidate.get("raw_action", {}) if isinstance(candidate.get("raw_action", {}), dict) else {}
    meta = raw_action.get("_candidate_meta", {}) if isinstance(raw_action.get("_candidate_meta", {}), dict) else {}
    procedure = meta.get("procedure", {}) if isinstance(meta.get("procedure", {}), dict) else {}
    procedure_guidance = meta.get("procedure_guidance", {}) if isinstance(meta.get("procedure_guidance", {}), dict) else {}
    if not bool(procedure.get("is_next_step", False) or procedure_guidance.get("active_next_step", False)):
        return 0.0

    mapping_confidence = _clamp01(procedure.get("mapping_confidence", 0.0), 0.0)
    family_binding_confidence = _clamp01(procedure.get("family_binding_confidence", 0.0), 0.0)
    alignment_strength = _clamp01(procedure_guidance.get("alignment_strength", 0.0), 0.0)
    procedure_bonus = _clamp01(float(procedure.get("procedure_bonus", 0.0) or 0.0) * 4.0, 0.0)
    hit_source = str(procedure.get("hit_source", "") or "").strip()

    if hit_source == "latent_mechanism_abstraction":
        latent_strength = (mapping_confidence * 0.65) + (family_binding_confidence * 0.35)
        return _clamp01(max(latent_strength, alignment_strength, procedure_bonus), 0.0)
    return _clamp01(max(mapping_confidence, alignment_strength, procedure_bonus), 0.0)


def _protected_active_procedure_under_low_trust(
    candidate: Dict[str, Any],
    world_model_control: WorldModelControlProtocol,
) -> bool:
    return (
        _active_procedure_strength(candidate) >= _PROTECTED_ACTIVE_STEP_CONFIDENCE
        and float(world_model_control.control_trust or 0.5) < _LOW_TRUST_PROTECTION_THRESHOLD
    )


def _as_function_names(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    names: List[str] = []
    for item in raw[:4]:
        if isinstance(item, dict):
            name = str(item.get("function_name", "") or "").strip()
        else:
            name = str(item or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _world_model_latent_branches(world_model_control: WorldModelControlProtocol) -> List[Dict[str, Any]]:
    branches: List[Dict[str, Any]] = []
    for item in list(getattr(world_model_control, "latent_branches", []) or [])[:4]:
        if not isinstance(item, dict):
            continue
        branches.append({
            "branch_id": str(item.get("branch_id", "") or "").strip(),
            "target_phase": str(item.get("target_phase", "") or "").strip().lower(),
            "confidence": _clamp01(item.get("confidence", 0.0), 0.0),
            "anchor_functions": _as_function_names(item.get("anchor_functions", [])),
            "risky_functions": _as_function_names(item.get("risky_functions", [])),
        })
    return branches


def _dominant_latent_branch(world_model_control: WorldModelControlProtocol) -> Dict[str, Any]:
    branches = _world_model_latent_branches(world_model_control)
    dominant_branch_id = str(getattr(world_model_control, "dominant_branch_id", "") or "").strip()
    for item in branches:
        if dominant_branch_id and str(item.get("branch_id", "") or "") == dominant_branch_id:
            return dict(item)
    if branches:
        return dict(branches[0])
    return {}


def _is_commit_like_action(action: str) -> bool:
    name = str(action or "").strip().lower()
    if not name:
        return False
    return any(token in name for token in ("commit", "apply", "submit", "advance", "finalize", "seal"))


def _latent_branch_hard_veto_reason(
    action: str,
    *,
    candidate: Dict[str, Any],
    world_model_control: WorldModelControlProtocol,
    protected_low_trust_step: bool,
) -> Optional[str]:
    if protected_low_trust_step:
        return None

    dominant_branch = _dominant_latent_branch(world_model_control)
    if not dominant_branch:
        return None

    branch_confidence = _clamp01(dominant_branch.get("confidence", 0.0), 0.0)
    control_trust = _clamp01(getattr(world_model_control, "control_trust", 0.5), 0.5)
    prediction_trust = _clamp01(getattr(world_model_control, "prediction_trust_score", 0.5), 0.5)
    effective_trust = max(control_trust, prediction_trust)
    if branch_confidence < _DOMINANT_BRANCH_HARD_VETO_CONFIDENCE or effective_trust < _DOMINANT_BRANCH_MIN_TRUST:
        return None

    branch_id = str(dominant_branch.get("branch_id", "") or "").strip()
    target_phase = str(dominant_branch.get("target_phase", "") or "").strip().lower()
    risky_functions = set(_as_function_names(dominant_branch.get("risky_functions", [])))
    anchor_functions = _as_function_names(dominant_branch.get("anchor_functions", []))
    anchor_set = set(anchor_functions)
    fn_name = str(candidate.get("function_name") or action or "").strip()
    action_name = fn_name or str(action or "").strip()

    if action_name and action_name in risky_functions:
        return f"latent_branch_risky:{action_name}:branch={branch_id}"

    if (
        target_phase == "committed"
        and len(anchor_functions) >= 2
        and action_name
        and action_name in anchor_set
        and action_name != anchor_functions[0]
        and _is_commit_like_action(action_name)
    ):
        return f"latent_branch_setup_required:{anchor_functions[0]}->{action_name}:branch={branch_id}"

    return None


def _safe_action_dict(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a rich action dict to a plain serializable dict for formal state.
    
    Only writes fields that are safe (JSON-serializable primitives).
    Rich runtime objects (metadata with experiment_plan, _score_result, etc.)
    are excluded from formal state and kept only in memory.
    """
    safe = {}
    for k, v in action.items():
        if k in _SAFE_ACTION_FIELDS:
            # These fields should all be JSON-serializable primitives
            if isinstance(v, (dict, list, str, float, int, bool, type(None))):
                safe[k] = v
        # Skip all other keys (rich objects: metadata, _score_result, experiment_plan, etc.)
    return safe


def _candidate_has_grounded_execution_target(candidate: Dict[str, Any]) -> bool:
    if not isinstance(candidate, dict):
        return False
    raw_action = candidate.get("raw_action", {}) if isinstance(candidate.get("raw_action", {}), dict) else {}
    payload = raw_action.get("payload", {}) if isinstance(raw_action.get("payload", {}), dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
    kwargs = tool_args.get("kwargs", {}) if isinstance(tool_args.get("kwargs", {}), dict) else {}
    if kwargs:
        return True
    meta = raw_action.get("_candidate_meta", {}) if isinstance(raw_action.get("_candidate_meta", {}), dict) else {}
    return bool(meta.get("surface_click_candidate", False) or meta.get("explicit_perception_target", False))


def _sole_grounded_non_wait_action(candidates: List[Dict[str, Any]]) -> str:
    non_wait_functions = {
        str(candidate.get("function_name") or candidate.get("action") or "").strip()
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("function_name") or candidate.get("action") or "").strip() not in {"", "wait"}
    }
    if len(non_wait_functions) != 1:
        return ""
    function_name = next(iter(non_wait_functions))
    matching_candidates = [
        candidate for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("function_name") or candidate.get("action") or "").strip() == function_name
    ]
    if any(_candidate_has_grounded_execution_target(candidate) for candidate in matching_candidates):
        return function_name
    return ""


def _forced_exploration_non_wait_action(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    external = [
        cand for cand in candidates
        if isinstance(cand, dict)
        and str(cand.get("action", "") or "").strip().upper().startswith("ACTION")
        and str(cand.get("action", "") or "").strip().lower() != "wait"
    ]
    if not external:
        return None
    return max(external, key=lambda c: float(c.get("final_score", c.get("score", 0.0)) or 0.0))


@dataclass
class Veto:
    """否决记录"""
    action: str
    reason: str
    source: str  # "hard_constraint" | "budget_exhausted" | "risk_threshold"


@dataclass
class GovernanceResult:
    """治理结果"""
    selected_action: Dict[str, Any]
    selection_reason: str
    veto_flags: List[Veto]
    mode_switched: bool
    new_mode: str  # "normal" | "cautious"


class GovernanceGate:
    """
    最小治理门。

    根据 specs/module_contracts.md：
    - 输入：predicted_consequences, hard_constraints, soft_constraints,
            budget_state, confidence, uncertainty_estimate
    - 输出：selected_action, selection_reason, veto_flags
    - 写状态：decision_context.selected_action/selection_reason/veto_flags,
              governance_context.mode/escalation_flags
    """

    def __init__(self):
        self.mode = "normal"  # normal | cautious
        self.veto_history: List[Veto] = []
        self._override_count = 0
        self._consecutive_fail_count = 0  # 连续 veto 计数
        self._wm_hard_risk_threshold = 0.75
        self._wm_hard_reversibility_threshold = 0.30

    def reset(self) -> None:
        """重置治理状态"""
        self.mode = "normal"
        self.veto_history = []
        self._override_count = 0
        self._consecutive_fail_count = 0

    def evaluate(
        self,
        candidates: List[Dict[str, Any]],
        hard_constraints: List[str],
        soft_constraints: List[str],
        budget_state: Dict[str, float],
        confidence: float,
        uncertainty: float,
        state_mgr,
        world_model_control: Optional[WorldModelControlProtocol] = None,
        policy_profile: Optional[Dict[str, float]] = None,
    ) -> GovernanceResult:
        """
        对候选动作进行治理过滤和最终选择。

        参数:
            candidates: 候选动作列表（含 predicted_consequences）
            hard_constraints: 硬约束列表
            soft_constraints: 软约束列表
            budget_state: 预算状态 {energy, time, ...}
            confidence: self_model 给出的决策把握度
            uncertainty: 世界模型给出的不确定性
            state_mgr: 状态管理器（用于写入 decision_context）

        返回:
            GovernanceResult
        """
        self.veto_history = []
        world_model_control = world_model_control or WorldModelControlProtocol()
        policy_profile = policy_profile or {}
        vetoes: List[Veto] = []
        passed: List[Dict[str, Any]] = []
        protected_sole_grounded_action = _sole_grounded_non_wait_action(candidates)

        # Step 1: 硬约束检查
        for cand in candidates:
            action_name = cand.get("action", "")
            candidate_for_check = cand
            if (
                protected_sole_grounded_action
                and action_name == protected_sole_grounded_action
                and _candidate_has_grounded_execution_target(cand)
            ):
                candidate_for_check = dict(cand)
                candidate_for_check["_protect_sole_grounded_action"] = True
            veto_reason = self._check_hard_constraints(action_name, candidate_for_check, hard_constraints, budget_state, world_model_control)

            if veto_reason:
                vetoes.append(Veto(
                    action=action_name,
                    reason=veto_reason,
                    source="hard_constraint"
                ))
            else:
                passed.append(cand)

        # Step 2: Risk threshold check (CAUTIOUS mode only)
        # In normal mode: trust the planner's scoring, only use ALL-BAD override at 0.90
        # In cautious mode: enforce stricter risk threshold (0.40)
        final_passed: List[Dict[str, Any]] = []
        planner_bias = float(policy_profile.get('planner_bias', 0.0) or 0.0)
        retrieval_aggressiveness = float(policy_profile.get('retrieval_aggressiveness', 0.5) or 0.5)
        base_threshold = 0.7 if self.mode == "normal" else 0.4
        risk_threshold = max(0.25, min(0.85, base_threshold - planner_bias * 0.15 + (retrieval_aggressiveness - 0.5) * 0.05))

        for cand in passed:
            action_name = cand.get("action", "")
            risk = cand.get("risk", 0.5)

            if self.mode == "cautious" and risk > risk_threshold:
                vetoes.append(Veto(
                    action=action_name,
                    reason=f"cautious_mode_risk_exceeded:risk={risk:.2f}>={risk_threshold}",
                    source="risk_threshold"
                ))
            else:
                final_passed.append(cand)

        # Step 3: ALL-BAD 覆盖 — 当所有移动候选风险 > 1.05 时，governance 直接接管
        # 阈值 1.05：move_risk 已 clamped 到 1.0，所以这个 threshold 不会触发
        # 真正的高危险由 posterior > 0.95 时的 confidence-based check 处理
        all_bad_override = False
        if candidates:
            move_risks = [c.get("risk", 0.0) for c in candidates if "move" in c.get("action", "")]
            if move_risks and all(r > 1.05 for r in move_risks):
                all_bad_override = True

        # Step 4: 选择或 fallback
        if all_bad_override:
            # 全局高危险 — 接管选择，强制 wait
            selected_action = {"action": "wait", "intent": "governance_all_bad_override", "risk": 0.05}
            selection_reason = "all_bad_override:all_candidates_risk_gt_0.7"
        elif not final_passed:
            forced_exploration = _forced_exploration_non_wait_action(candidates)
            if forced_exploration is not None:
                selected_action = dict(forced_exploration)
                selection_reason = "forced_exploration_due_to_governance_filter"
            else:
                # 全否决 → fallback
                selected_action = {"action": "wait", "intent": "fallback_due_to_governance_filter", "risk": 0.05}
                selection_reason = "fallback_due_to_governance_filter"
        else:
            # 选择最优（final_score 最高的 — Step 6 已由 planner 计算）
            best = max(final_passed, key=lambda c: c.get("final_score", c.get("score", 0)))
            selected_action = best
            best_risk = best.get("risk", 0)
            best_opp = best.get("opportunity_estimate", best.get("opportunity", 0))
            selection_reason = f"selected:{best['action']}:opp={best_opp:.2f}:risk={best_risk:.2f}:score={best.get('final_score',0):.3f}"

        # Step 4: mode switch 检查
        mode_switched = False
        # 连续 veto 跟踪 (持久化, 不在每次重置)
        if len(vetoes) >= 1:
            self._consecutive_fail_count += 1
        else:
            self._consecutive_fail_count = 0

        probe_bias = float(policy_profile.get('probe_bias', 0.5) or 0.5)
        escalation_flags: List[str] = []
        if probe_bias >= 0.75 and uncertainty >= 0.5:
            escalation_flags.append('probe_bias_high_under_uncertainty')
        if planner_bias >= 0.8 and confidence < 0.4:
            escalation_flags.append('planner_bias_high_low_confidence')

        # Mode switch: 连续 5 次 veto 才切换到 cautious
        if self._consecutive_fail_count >= 5 and self.mode == "normal":
            self.mode = "cautious"
            mode_switched = True
        # Mode switch: 连续 5 次无 veto 才切回 normal
        elif self._consecutive_fail_count == 0 and self.mode == "cautious":
            self.mode = "normal"
            mode_switched = True

        # Step 5: 写入 state（通过 state_mgr）
        # CRITICAL: only write serializable data — rich objects stay in memory
        state_mgr.update_state({
            "decision_context.selected_action": _safe_action_dict(selected_action),
            "decision_context.selection_reason": selection_reason,
            "decision_context.veto_flags": [f"{v.source}:{v.action}:{v.reason}" for v in vetoes],
            "governance_context.escalation_flags": escalation_flags,
            "governance_context.world_model_control": world_model_control.to_dict(),
            "governance_context.policy_profile": {
                "planner_bias": planner_bias,
                "retrieval_aggressiveness": retrieval_aggressiveness,
            },
        }, reason="governance_evaluate", module="governance")

        if mode_switched:
            state_mgr.update_state({
                "governance_context.mode": self.mode,
            }, reason=f"governance_mode_switch:{self.mode}", module="governance")

        return GovernanceResult(
            selected_action=selected_action,
            selection_reason=selection_reason,
            veto_flags=vetoes,
            mode_switched=mode_switched,
            new_mode=self.mode,
        )

    def _check_hard_constraints(
        self,
        action: str,
        cand: Dict[str, Any],
        hard_constraints: List[str],
        budget_state: Dict[str, float],
        world_model_control: WorldModelControlProtocol,
    ) -> Optional[str]:
        """
        检查硬约束。
        返回 None 表示通过，返回字符串表示否决原因。
        """
        # 能量预算检查
        energy = budget_state.get("energy", 100)
        cost = cand.get("estimated_cost", 1.0)
        if cost > energy:
            return f"insufficient_energy:cost={cost:.1f}:available={energy:.1f}"

        protected_low_trust_step = _protected_active_procedure_under_low_trust(cand, world_model_control)

        # World-model blocked functions
        if not protected_low_trust_step and action in set(world_model_control.blocked_functions or []):
            return f"world_model_blocked:{action}"

        # 显式硬约束列表
        for constraint in list(hard_constraints or []):
            if "no_" in constraint and constraint.replace("no_", "") in action:
                return f"hard_constraint:{constraint}"

        # 低可信 world-model 对高置信 active procedure 尾步只保留弱建议，避免错误硬拦截。
        if not protected_low_trust_step:
            for constraint in list(world_model_control.hard_constraints or []):
                if "no_" in constraint and constraint.replace("no_", "") in action:
                    return f"hard_constraint:{constraint}"

        latent_branch_reason = _latent_branch_hard_veto_reason(
            action,
            candidate=cand,
            world_model_control=world_model_control,
            protected_low_trust_step=protected_low_trust_step,
        )
        if latent_branch_reason:
            return latent_branch_reason

        # World-model risk hard veto — but skip when wm_risk is at boundary (1.0)
        # which indicates uninformative default rather than genuine high-risk signal
        wm_risk = float(cand.get("wm_risk", cand.get("risk", 0.0)) or 0.0)
        wm_reversibility = float(cand.get("wm_reversibility", 1.0) or 1.0)
        if (
            not protected_low_trust_step
            and not bool(cand.get("_protect_sole_grounded_action", False))
            and wm_risk >= self._wm_hard_risk_threshold
            and wm_reversibility <= self._wm_hard_reversibility_threshold
            and wm_risk < 1.0
        ):
            return f"wm_hard_veto:risk={wm_risk:.2f}:reversibility={wm_reversibility:.2f}"

        # 危险动作检查
        if action == "shutdown" and self.mode == "normal":
            return "hard_constraint:shutdown_only_in_emergency"

        return None

    def get_mode(self) -> str:
        return self.mode


# 全局单例
_gate: Optional[GovernanceGate] = None


def get_governance() -> GovernanceGate:
    global _gate
    if _gate is None:
        _gate = GovernanceGate()
    return _gate
