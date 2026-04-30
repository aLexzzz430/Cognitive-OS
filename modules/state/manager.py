"""
state/manager.py — 全局状态管理器

提供受控的状态读写接口，禁止模块直接修改内部结构。
所有状态变更必须通过本模块。
"""

import json
import logging
import os
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.runtime_paths import default_state_path

from .schema import StateSchema


# 全局状态管理器实例
_state_manager: Optional["StateManager"] = None


# 便捷写接口一次性决策表（keep/delete + module 归属）。
# 说明：仅保留模块边界清晰且仍具可读性价值的便捷接口，全部经 update_state 落审计。
CONVENIENCE_WRITE_INTERFACE_DECISIONS: Dict[str, Dict[str, str]] = {
    "add_governance_veto": {"decision": "keep", "module": "governance"},
    "add_shutdown_flag": {"decision": "keep", "module": "governance"},
    "clear_shutdown_flags": {"decision": "keep", "module": "governance"},
    "add_anomaly": {"decision": "keep", "module": "governance"},
    "clear_anomalies": {"decision": "keep", "module": "governance"},
    "add_error_flag": {"decision": "keep", "module": "learning"},
    "push_observation": {"decision": "keep", "module": "core"},
    "push_action": {"decision": "keep", "module": "core"},
    "push_outcome": {"decision": "keep", "module": "core"},
    "set_candidates": {"decision": "keep", "module": "planner_policy"},
    "set_selected_action": {"decision": "keep", "module": "governance"},
    "set_prediction": {"decision": "keep", "module": "world_model"},
    "set_world_summary_field": {"decision": "keep", "module": "world_model"},
    "set_self_summary_field": {"decision": "keep", "module": "learning"},
    "set_learning_update": {"decision": "keep", "module": "learning"},
    "set_active_context": {"decision": "keep", "module": "core"},
    "append_major_event": {"decision": "keep", "module": "core"},
    "set_tick_latency": {"decision": "keep", "module": "core"},
    "set_module_cost": {"decision": "keep", "module": "core"},
}


class StateManager:
    """
    唯一合法的全局状态读写入口。

    所有模块必须通过 get_state() / update_state() / commit_tick() 操作状态。
    禁止直接修改 self._state 的内部字段。
    """

    def __init__(
        self,
        state_path: str = None,
        core_compat_writes: Optional[List[str]] = None,
        validate_module_prefix: bool = False,
    ):
        if state_path is None:
            state_path = str(default_state_path())
        self._state_path = str(state_path)
        self._state: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []  # 用于审计
        self._initialized = False
        self._logger = logging.getLogger(__name__)
        self._core_compat_writes = self._resolve_core_compat_writes(core_compat_writes)
        self._validate_module_prefix = bool(validate_module_prefix)
        self._write_auth_telemetry = {
            "denied_writes": 0,
            "compat_writes": 0,
            "offending_paths": Counter(),
            "compat_paths": Counter(),
        }

    def _resolve_core_compat_writes(self, configured: Optional[List[str]]) -> List[str]:
        """解析 core 兼容写入路径（参数优先，其次环境变量 STATE_CORE_COMPAT_WRITES）。"""
        if configured is not None:
            return [p.strip() for p in configured if isinstance(p, str) and p.strip()]
        raw = os.getenv("STATE_CORE_COMPAT_WRITES", "")
        if not raw.strip():
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]

    @staticmethod
    def _prefix_matches(path: str, allowed_prefixes: List[str]) -> bool:
        for prefix in allowed_prefixes:
            if prefix == "*" or path == prefix or path.startswith(prefix + "."):
                return True
        return False

    @staticmethod
    def _is_write_strict_mode() -> bool:
        return os.getenv("STATE_WRITE_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}

    def initialize(
        self,
        agent_id: str = None,
        run_id: str = None,
        episode_id: str = None,
        top_goal: str = "survive_and_progress",
    ) -> None:
        """初始化新状态（只在首次或新episode开始时调用）"""
        self._state = StateSchema.get_default_state()
        self._state["identity"]["agent_id"] = agent_id or str(uuid.uuid4())
        self._state["identity"]["run_id"] = run_id or str(uuid.uuid4())
        self._state["identity"]["episode_id"] = episode_id or str(uuid.uuid4())
        self._state["identity"]["tick_id"] = 0
        self._state["identity"]["wall_clock_time"] = datetime.now().isoformat()
        self._state["goal_stack"]["top_goal"] = top_goal
        self._state["goal_stack"]["current_focus"] = top_goal
        self._initialized = True
        self._save()

    def load(self) -> bool:
        """从磁盘加载已有状态"""
        path = Path(self._state_path)
        if path.exists():
            try:
                with open(path, "r") as f:
                    self._state = json.load(f)
                self._initialized = True
                return True
            except Exception:
                return False
        return False

    def _save(self) -> None:
        """保存状态到磁盘"""
        Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # 只读访问接口
    # -------------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """返回完整状态的深拷贝（只读，禁止直接修改返回值）"""
        import copy
        return copy.deepcopy(self._state)

    def get(self, path: str, default: Any = None) -> Any:
        """
        按路径读取状态字段，路径格式为 "top_level.field" 或 "top_level.sub.field"。
        例如: get("identity.tick_id"), get("governance_context.mode")
        """
        parts = path.split(".")
        val = self._state
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                return default
            if val is None:
                return default
        return val

    # -------------------------------------------------------------------------
    # 状态更新接口（仅允许通过 patch）
    # -------------------------------------------------------------------------

    def update_state(self, patch: Dict[str, Any], reason: str = "", module: str = "core") -> None:
        """
        通过 patch 字典部分更新状态。

        patch 格式示例:
            {"identity.tick_id": 42, "governance_context.mode": "cautious"}

        参数:
            module: 调用模块名（如 "memory_subsystem", "governance", "core"）
                    默认为 "core"。将根据 specs/module_contracts.md 检查写权限。
                    已知模块（memory_subsystem/world_model/planner_policy/
                    governance/learning/core）必须有对应写权限。

        禁止:
            - 直接修改 self._state 的内部字段
            - 传入未经验证的嵌套 dict
        """
        if self._validate_module_prefix:
            self._warn_module_prefix_mismatch(patch=patch, module=module, reason=reason)
        errors = self._check_write_authorization(patch, module, reason=reason)
        if errors:
            raise PermissionError(
                f"Module '{module}' unauthorized write: {errors}. "
                f"Use state_manager.get_state() to inspect, use state_manager.update_state(module=...) to write."
            )

        for path, value in patch.items():
            self._set(path, value)

        # 记录变更历史（用于审计）
        self._history.append({
            "tick": self._state.get("identity", {}).get("tick_id", -1),
            "timestamp": datetime.now().isoformat(),
            "patch": patch,
            "reason": reason,
            "module": module or "unknown",
        })
        self._save()

    def _check_write_authorization(self, patch: Dict[str, Any], module: str, reason: str = "") -> List[str]:
        """
        检查 patch 是否在模块的允许写范围内。
        返回空列表表示授权通过，非空列表包含违规路径。
        """
        errors = []

        # 显式模块写权限表（core=显式白名单；兼容路径另行通过 core_compat_writes 放行）
        module_allowed_writes = {
            "memory_subsystem": [
                "episodic_memory_context",
            ],
            "world_model": [
                "world_model.belief_state",
                "world_model.belief_state.prior",
                "world_model.belief_state.evidence",
                "world_model.belief_state.posterior",
                "world_model.belief_state.confidence",
                "world_model.belief_state.evidence_log",
                "world_model.belief_state.step_count",
                "world_model.hidden_state",
                "world_summary",
                "decision_context.predicted_consequences",
            ],
            "planner_policy": [
                "decision_context.candidate_actions",
                "decision_context.alternative_rankings",
            ],
            "goal_runtime": [
                "goal_stack.subgoals",
                "goal_stack.current_focus",
                "goal_stack.goal_status",
                "goal_stack.goal_priority",
                "goal_stack.goal_history",
            ],
            "governance": [
                "decision_context.selected_action",
                "decision_context.selection_reason",
                "decision_context.veto_flags",
                "governance_context.mode",
                "governance_context.shutdown_flags",
                "governance_context.escalation_flags",
                "governance_context.world_model_control",
                "governance_context.policy_profile",
                "telemetry_summary.anomaly_flags",
            ],
            "learning": [
                "learning_context",
                "learning_context.prediction_error",
                "self_summary",
                "self_summary.error_flags",
                "self_summary.recent_failures",
                "goal_stack.goal_status",
                "telemetry_summary.performance_snapshot",
            ],
            "decision": [
                "decision_context.retrieval_aux_decisions",
                "decision_context.retrieval_surfacing_protocol",
            ],
            "meta_control": [
                "decision_context.policy_profile",
                "decision_context.policy_read_fallback_events",
            ],
            "continuity": [
                "continuity",
            ],
            "core.reasoning": [
                "object_workspace.ranked_discriminating_experiments",
                "object_workspace.posterior_summary",
                "object_workspace.competing_hypotheses",
                "object_workspace.competing_hypothesis_objects",
                "object_workspace.active_hypotheses_summary",
                "object_workspace.active_tests",
                "object_workspace.candidate_tests",
                "object_workspace.candidate_programs",
                "object_workspace.candidate_outputs",
            ],
            "core.runtime.evidence_ledger": [
                "object_workspace.formal_evidence_ledger",
                "object_workspace.formal_evidence_recent",
            ],
            # core 仅允许编排级写入（显式白名单）
            "core": [
                "identity.agent_id",
                "identity.run_id",
                "identity.episode_id",
                "identity.tick_id",
                "identity.wall_clock_time",
                "working_memory.recent_observations",
                "working_memory.recent_actions",
                "working_memory.recent_outcomes",
                "working_memory.active_context",
                "telemetry_summary.major_events",
                "telemetry_summary.tick_latency_ms",
                "telemetry_summary.module_costs",
                "decision_context.retrieval_aux_decisions",
                "decision_context.retrieval_surfacing_protocol",
                "decision_context.policy_profile",
                "decision_context.policy_read_fallback_events",
                "decision_context.representation_profile",
                "decision_context.planner_meta_control_snapshot_id",
                "decision_context.planner_meta_control_inputs_hash",
                "decision_context.governance_meta_control_snapshot_id",
                "decision_context.governance_meta_control_inputs_hash",
                "world_model.belief_state",
                "world_model.belief_state.confidence",
                "world_model.active_mechanisms",
                "world_model.boundary_flags",
                "continuity",
                "object_workspace",
                "core",
            ],
        }

        allowed = module_allowed_writes.get(module, [])
        strict_mode = self._is_write_strict_mode()
        current_tick = self._state.get("identity", {}).get("tick_id", -1)

        for path in patch.keys():
            # 检查路径是否在允许范围内（前缀匹配）
            authorized = self._prefix_matches(path, allowed)
            compat_authorized = (module == "core") and self._prefix_matches(path, self._core_compat_writes)

            if authorized:
                continue

            if compat_authorized:
                if strict_mode:
                    self._write_auth_telemetry["denied_writes"] += 1
                    self._write_auth_telemetry["offending_paths"][path] += 1
                    errors.append(
                        f"path='{path}' matched core_compat_writes but strict mode enabled "
                        f"(module={module}, reason={reason}, tick={current_tick})"
                    )
                    continue

                self._write_auth_telemetry["compat_writes"] += 1
                self._write_auth_telemetry["compat_paths"][path] += 1
                self._logger.warning(
                    "state_write_compat path=%s reason=%s module=%s tick=%s note=core_compat_temporarily_allowed",
                    path,
                    reason or "",
                    module,
                    current_tick,
                )
                continue

            if not authorized:
                self._write_auth_telemetry["denied_writes"] += 1
                self._write_auth_telemetry["offending_paths"][path] += 1
                errors.append(f"path='{path}' not in {module}'s allowed_writes")

        return errors

    def _warn_module_prefix_mismatch(self, patch: Dict[str, Any], module: str, reason: str = "") -> None:
        """Optional warning when module name and state key prefix look mismatched."""
        if not isinstance(patch, dict) or not patch:
            return
        current_tick = self._state.get("identity", {}).get("tick_id", -1)
        prefix_map = {
            'core': ('core', 'identity', 'working_memory', 'telemetry_summary'),
            'world_model': ('world_model', 'world_summary'),
            'goal_runtime': ('goal_stack',),
            'decision': ('decision_context',),
            'meta_control': ('decision_context',),
            'continuity': ('continuity',),
            'governance': ('governance_context', 'decision_context', 'telemetry_summary'),
            'planner_policy': ('decision_context',),
            'learning': ('learning_context', 'self_summary', 'goal_stack', 'telemetry_summary'),
            'memory_subsystem': ('episodic_memory_context',),
        }
        allowed_prefixes = prefix_map.get(module, ())
        for path in patch.keys():
            top = str(path).split('.', 1)[0]
            if allowed_prefixes and all(not path.startswith(prefix) for prefix in allowed_prefixes) and top not in allowed_prefixes:
                self._logger.warning(
                    "state_module_prefix_mismatch module=%s path=%s reason=%s tick=%s",
                    module,
                    path,
                    reason or "",
                    current_tick,
                )

    def get_write_auth_telemetry(self, top_n: int = 5) -> Dict[str, Any]:
        """返回结构化授权检查遥测信息。"""
        offending_paths = self._write_auth_telemetry["offending_paths"]
        compat_paths = self._write_auth_telemetry["compat_paths"]
        return {
            "denied_writes": int(self._write_auth_telemetry["denied_writes"]),
            "compat_writes": int(self._write_auth_telemetry["compat_writes"]),
            "top_offending_paths": offending_paths.most_common(top_n),
            "top_compat_paths": compat_paths.most_common(top_n),
        }

    def _set(self, path: str, value: Any) -> None:
        """内部：按路径设置状态字段"""
        parts = path.split(".")
        target = self._state
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    def _append_to_list_path(
        self,
        path: str,
        item: Any,
        *,
        reason: str,
        module: str = "core",
        max_items: Optional[int] = None,
        unique: bool = False,
    ) -> None:
        """
        统一的 list append helper：
        读取当前值 -> 构造新列表 -> 通过 update_state patch 回写。
        """
        current = self.get(path, [])
        if not isinstance(current, list):
            current = []
        new_list = list(current)
        if unique and item in new_list:
            return
        new_list.append(item)
        if max_items is not None:
            new_list = new_list[-max_items:]
        self.update_state({path: new_list}, reason=reason, module=module)

    # -------------------------------------------------------------------------
    # tick 生命周期接口
    # -------------------------------------------------------------------------

    def commit_tick(self, normalized_observation: Any = None) -> None:
        """
        每个 tick 结束时调用，提交本 tick 的状态变更。

        根据 specs/state_schema.md 的生命周期规则：
        - tick_id += 1（已在 step_state_refresh 中完成，这里只做收尾）
        - 清理过期的滑动窗口字段
        - 更新 wall_clock_time
        """
        # 注意：tick_id += 1 在 step_state_refresh 中已完成
        # 这里只做收尾工作，并以一次 patch 提交以保留审计 history
        patch = {
            "identity.wall_clock_time": datetime.now().isoformat(),
        }
        patch.update(self._trim_sliding_windows())
        self.update_state(patch, reason="workflow:commit_tick", module="core")

    def _trim_sliding_windows(self) -> Dict[str, Any]:
        """裁剪滑动窗口字段，保持在合理大小；返回需提交的 patch。"""
        MAX_WINDOW_SIZE = 20

        windows = {
            "working_memory.recent_observations": self._state.get("working_memory", {}).get("recent_observations", []),
            "working_memory.recent_actions": self._state.get("working_memory", {}).get("recent_actions", []),
            "working_memory.recent_outcomes": self._state.get("working_memory", {}).get("recent_outcomes", []),
        }

        patch: Dict[str, Any] = {}
        for path, window in windows.items():
            if len(window) > MAX_WINDOW_SIZE:
                patch[path] = list(window[-MAX_WINDOW_SIZE:])
        return patch

    def reset_for_new_episode(self, episode_id: str = None) -> None:
        """
        新 episode 开始时重置短期状态，保留长期状态。
        """
        # 保存长期状态
        long_term_keys = ["identity", "goal_stack", "world_summary", "self_summary"]

        # 重新初始化
        self.initialize(
            agent_id=self._state.get("identity", {}).get("agent_id"),
            run_id=self._state.get("identity", {}).get("run_id"),
            episode_id=episode_id or str(uuid.uuid4()),
            top_goal=self._state.get("goal_stack", {}).get("top_goal", "survive_and_progress"),
        )

    # -------------------------------------------------------------------------
    # 治理接口
    # -------------------------------------------------------------------------

    def add_governance_veto(self, reason: str) -> None:
        """记录一个治理否决"""
        self._append_to_list_path(
            "decision_context.veto_flags",
            reason,
            reason="workflow:add_governance_veto",
            module="governance",
        )

    def add_shutdown_flag(self, flag: str) -> None:
        """添加关闭/熔断标志"""
        self._append_to_list_path(
            "governance_context.shutdown_flags",
            flag,
            reason="workflow:add_shutdown_flag",
            module="governance",
            unique=True,
        )

    def clear_shutdown_flags(self) -> None:
        """清除所有关闭标志"""
        self.update_state(
            {"governance_context.shutdown_flags": []},
            reason="workflow:clear_shutdown_flags",
            module="governance",
        )

    def add_anomaly(self, anomaly: str) -> None:
        """添加异常标记"""
        self._append_to_list_path(
            "telemetry_summary.anomaly_flags",
            anomaly,
            reason="workflow:add_anomaly",
            module="governance",
            unique=True,
        )

    def clear_anomalies(self) -> None:
        """清除所有异常标记"""
        self.update_state(
            {"telemetry_summary.anomaly_flags": []},
            reason="workflow:clear_anomalies",
            module="governance",
        )

    def add_error_flag(self, error: str) -> None:
        """添加自我模型错误标记"""
        self._append_to_list_path(
            "self_summary.error_flags",
            error,
            reason="workflow:add_error_flag",
            module="learning",
            unique=True,
        )

    # -------------------------------------------------------------------------
    # 工作流便捷接口
    # -------------------------------------------------------------------------

    def push_observation(self, observation: Dict[str, Any]) -> None:
        """Step 1/2: 记录新观测到 working memory"""
        self._append_to_list_path(
            "working_memory.recent_observations",
            observation,
            reason="workflow:push_observation",
            module="core",
        )

    def push_action(self, action: Dict[str, Any]) -> None:
        """Step 8: 记录动作到 working memory"""
        self._append_to_list_path(
            "working_memory.recent_actions",
            action,
            reason="workflow:push_action",
            module="core",
        )

    def push_outcome(self, outcome: Dict[str, Any]) -> None:
        """Step 9: 记录结果到 working memory"""
        self._append_to_list_path(
            "working_memory.recent_outcomes",
            outcome,
            reason="workflow:push_outcome",
            module="core",
        )

    def set_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        """Step 5: 设置候选动作"""
        self.update_state(
            {"decision_context.candidate_actions": candidates},
            reason="workflow:set_candidates",
            module="planner_policy",
        )

    def set_selected_action(self, action: Dict[str, Any], reason: str) -> None:
        """Step 7: 设置最终选择的动作"""
        self.update_state(
            {
                "decision_context.selected_action": action,
                "decision_context.selection_reason": reason,
            },
            reason="workflow:set_selected_action",
            module="governance",
        )

    def set_prediction(self, predictions: List[Dict[str, Any]]) -> None:
        """Step 6: 设置预测后果"""
        self.update_state(
            {"decision_context.predicted_consequences": predictions},
            reason="workflow:set_prediction",
            module="world_model",
        )

    def set_world_summary_field(self, field: str, value: Any) -> None:
        """Step 4: 更新 world_summary 字段"""
        self.update_state(
            {f"world_summary.{field}": value},
            reason="workflow:set_world_summary_field",
            module="world_model",
        )

    def set_self_summary_field(self, field: str, value: Any) -> None:
        """Step 4: 更新 self_summary 字段"""
        self.update_state(
            {f"self_summary.{field}": value},
            reason="workflow:set_self_summary_field",
            module="learning",
        )

    def set_learning_update(self, prediction_error: Dict[str, Any]) -> None:
        """Step 9/10: 记录预测误差"""
        self.update_state(
            {"learning_context.prediction_error": prediction_error},
            reason="workflow:set_learning_update",
            module="learning",
        )

    def set_active_context(self, context: Dict[str, Any]) -> None:
        """Step 2: 设置 active_context"""
        self.update_state(
            {"working_memory.active_context": context},
            reason="workflow:set_active_context",
            module="core",
        )

    def append_major_event(self, event: Dict[str, Any]) -> None:
        """记录重大事件"""
        self._append_to_list_path(
            "telemetry_summary.major_events",
            event,
            reason="workflow:append_major_event",
            module="core",
            max_items=50,
        )

    def set_tick_latency(self, latency_ms: float) -> None:
        """记录 tick 延迟"""
        self.update_state(
            {"telemetry_summary.tick_latency_ms": latency_ms},
            reason="workflow:set_tick_latency",
            module="core",
        )

    def set_module_cost(self, module: str, cost_ms: float) -> None:
        """记录模块耗时"""
        costs = dict(self.get("telemetry_summary.module_costs", {}))
        costs[module] = cost_ms
        self.update_state(
            {"telemetry_summary.module_costs": costs},
            reason="workflow:set_module_cost",
            module="core",
        )

    # -------------------------------------------------------------------------
    # 查询接口（用于模块）
    # -------------------------------------------------------------------------

    def get_tick_id(self) -> int:
        return self._state.get("identity", {}).get("tick_id", 0)

    def get_episode_id(self) -> str:
        return self._state.get("identity", {}).get("episode_id", "")

    def get_top_goal(self) -> str:
        return self._state.get("goal_stack", {}).get("top_goal", "")

    def get_current_focus(self) -> str:
        return self._state.get("goal_stack", {}).get("current_focus", "")

    def get_governance_mode(self) -> str:
        return self._state.get("governance_context", {}).get("mode", "normal")

    def has_shutdown_flags(self) -> bool:
        return len(self._state.get("governance_context", {}).get("shutdown_flags", [])) > 0

    def get_confidence(self) -> float:
        return self._state.get("self_summary", {}).get("confidence", 0.5)

    def get_risk_estimate(self) -> float:
        return self._state.get("world_summary", {}).get("risk_estimate", 0.5)

    def get_opportunity_estimate(self) -> float:
        return self._state.get("world_summary", {}).get("opportunity_estimate", 0.5)

    def get_energy(self) -> float:
        return self._state.get("governance_context", {}).get("budget_state", {}).get("energy", 100)


def get_state_manager() -> StateManager:
    """获取全局状态管理器实例（单例）"""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def reset_state_manager(
    *,
    state_path: Optional[str] = None,
    core_compat_writes: Optional[List[str]] = None,
    validate_module_prefix: bool = False,
) -> StateManager:
    """Install a fresh singleton manager for a new runtime session."""
    global _state_manager
    _state_manager = StateManager(
        state_path=state_path,
        core_compat_writes=core_compat_writes,
        validate_module_prefix=validate_module_prefix,
    )
    return _state_manager


def init_state(**kwargs) -> StateManager:
    """初始化并返回状态管理器"""
    mgr = get_state_manager()
    mgr.initialize(**kwargs)
    return mgr
