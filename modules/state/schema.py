"""
state/schema.py — AGI_V2 State Schema v0.1 定义

本文件是 state_schema.md 的机器可读版本。
所有顶层结构和字段定义必须与 specs/state_schema.md 保持同步。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable


@dataclass
class FieldDef:
    """单个状态的字段定义"""
    name: str
    field_type: str  # type annotation as string
    description: str
    writer: str      # 模块名
    reader: str      # 模块名
    update_time: str  # 何时更新
    expiry: str      # 何时失效
    default: Any = None


@dataclass
class SubSchema:
    """子结构定义"""
    name: str
    purpose: str
    fields: List[FieldDef]
    invariants: List[str] = field(default_factory=list)


# =============================================================================
# 顶层状态结构定义
# =============================================================================

IDENTITY_FIELDS = [
    FieldDef("agent_id", "str", "系统实例唯一标识", "runtime", "全模块", "启动时", "永不失效"),
    FieldDef("run_id", "str", "本次运行唯一标识", "runtime", "eval/telemetry/logging", "运行开始时", "本run结束"),
    FieldDef("episode_id", "str", "当前episode标识", "env_adapter/runtime", "memory/eval/telemetry", "每次新episode开始", "当前episode结束"),
    FieldDef("tick_id", "int", "当前episode内tick计数", "core_main_loop", "全模块", "每tick开始", "当前episode结束"),
    FieldDef("wall_clock_time", "str", "当前时间戳", "runtime", "telemetry", "每tick可选更新", "无硬性保留要求"),
]

GOAL_STACK_FIELDS = [
    FieldDef("top_goal", "str", "当前最高层目标", "planner/runtime_setup", "policy/governance/learning", "初始化或明确重设时", "被新top_goal替代"),
    FieldDef("subgoals", "list[dict]", "子目标列表", "planner", "policy/world_model/learning", "每tick可更新", "目标完成/废弃时清理"),
    FieldDef("current_focus", "str", "当前聚焦目标", "planner/policy", "全模块", "每tick决策前后可更新", "下一focus覆盖"),
    FieldDef("goal_status", "dict", "各目标状态", "planner/learning", "eval/policy/governance", "结果评估后更新", "目标生命周期结束后归档"),
    FieldDef("goal_priority", "dict", "各目标优先级", "planner/governance", "policy", "目标切换时更新", "目标生命周期结束"),
    FieldDef("goal_history", "list[dict]", "近期目标切换与理由", "planner/telemetry", "eval/audit", "目标切换时更新", "保留最近N条"),
]

WORLD_SUMMARY_FIELDS = [
    FieldDef("observed_facts", "list[dict]", "已观测事实摘要", "observe/world_model", "planner/policy/learning", "每tick更新", "过期事实可移除或降权"),
    FieldDef("latent_hypotheses", "list[dict]", "未证实但影响决策的假设", "world_model", "policy/governance/learning", "每tick可更新", "被证伪/过期时清理"),
    FieldDef("opportunity_estimate", "float", "当前机会强度估计[0,1]", "world_model", "policy", "每tick更新", "下一tick覆盖"),
    FieldDef("risk_estimate", "float", "当前风险强度估计[0,1]", "world_model/governance", "policy/governance", "每tick更新", "下一tick覆盖"),
    FieldDef("uncertainty_estimate", "float", "当前局势不确定性估计[0,1]", "world_model/self_model", "policy/governance", "每tick更新", "下一tick覆盖"),
    FieldDef("current_phase", "str", "当前环境阶段判断", "world_model", "planner/policy/learning", "每tick可更新", "下一阶段覆盖"),
    FieldDef("predicted_dynamics", "list[dict]", "对局势短期演化的摘要预测", "world_model", "policy/learning", "每tick可更新", "下一tick覆盖"),
    FieldDef("notable_entities", "list[dict]", "当前关键实体摘要", "observe/world_model", "policy/planner", "每tick可更新", "实体不再重要时降权"),
]

SELF_SUMMARY_FIELDS = [
    FieldDef("capability_estimate", "dict", "对自身在若干任务维度上的能力估计", "self_model/learning", "planner/policy/governance", "每tick或每episode更新", "可持续保留"),
    FieldDef("confidence", "float", "对当前决策把握的总体估计[0,1]", "self_model", "policy/governance", "每tick更新", "下一tick覆盖"),
    FieldDef("error_flags", "list[str]", "当前检测到的异常或偏差标记", "runtime/self_model/governance", "全模块", "触发时更新", "被处理后清除"),
    FieldDef("resource_budget", "dict", "当前资源预算摘要", "runtime/governance", "policy/planner", "每tick更新", "下一tick覆盖"),
    FieldDef("recent_failures", "list[dict]", "近期失败摘要", "learning/telemetry", "self_model/planner", "结果评估后更新", "保留最近N条"),
    FieldDef("stability_estimate", "float", "当前策略稳定性估计[0,1]", "self_model/telemetry", "governance/planner", "每tick或周期更新", "下一次更新覆盖"),
    FieldDef("adaptation_readiness", "float", "当前是否适合探索或调整策略[0,1]", "self_model/learning", "planner/policy", "每tick或阶段切换时更新", "下一次更新覆盖"),
]

WORKING_MEMORY_FIELDS = [
    FieldDef("recent_observations", "list[dict]", "最近若干步标准化观测", "observe", "world_model/policy/learning", "每tick更新", "滑动窗口淘汰"),
    FieldDef("recent_actions", "list[dict]", "最近若干步动作记录", "act", "learning/policy/telemetry", "每tick更新", "滑动窗口淘汰"),
    FieldDef("recent_outcomes", "list[dict]", "最近若干步结果记录", "evaluate_outcome", "learning/self_model", "每tick更新", "滑动窗口淘汰"),
    FieldDef("active_context", "dict", "当前这一拍最重要的局部上下文", "state_refresh/planner", "全模块", "每tick更新", "下一tick覆盖"),
    FieldDef("attention_targets", "list[str]", "当前高度关注对象", "planner/policy", "observe/world_model", "每tick可更新", "下一tick覆盖"),
]

EPISODIC_MEMORY_CONTEXT_FIELDS = [
    FieldDef("retrieved_episodes", "list[dict]", "当前检索出的相关episode摘要", "memory_subsystem", "world_model/policy/learning", "每tick决策前更新", "下一tick覆盖"),
    FieldDef("retrieval_scores", "list[float]", "检索匹配分数", "memory_subsystem", "telemetry/learning", "每tick更新", "下一tick覆盖"),
    FieldDef("memory_notes", "list[str]", "从记忆中抽出的结构化提示", "memory_subsystem/learning", "policy/planner", "每tick更新", "下一tick覆盖"),
    FieldDef("retrieval_query_summary", "dict", "本次检索依据了哪些线索", "memory_subsystem", "telemetry/audit", "每tick更新", "下一tick覆盖"),
]

DECISION_CONTEXT_FIELDS = [
    FieldDef("candidate_actions", "list[dict]", "候选动作集合", "policy", "governance/telemetry/learning", "每tick决策时更新", "下一tick覆盖"),
    FieldDef("predicted_consequences", "list[dict]", "候选动作的预测后果", "world_model/evaluator", "governance/policy/learning", "每tick更新", "下一tick覆盖"),
    FieldDef("selected_action", "dict|null", "被选中的动作", "policy/governance", "act/telemetry", "每tick更新", "行动后保留至下次覆盖"),
    FieldDef("selection_reason", "str", "选择理由", "policy", "telemetry/audit/learning", "每tick更新", "下一tick覆盖"),
    FieldDef("veto_flags", "list[str]", "被治理层拦截的原因", "governance", "telemetry/audit", "每tick可更新", "下一tick覆盖"),
    FieldDef("alternative_rankings", "list[dict]", "候选动作排序与分数", "policy/evaluator", "telemetry/learning", "每tick更新", "下一tick覆盖"),
    # C2: Error Taxonomy
    FieldDef("active_error_types", "list[str]", "Step5-7: 当前活跃的错误类型 (error_taxonomy_v0_1)", "governance/learning", "policy/evaluator/act", "error触发时写入", "下一tick覆盖"),
    FieldDef("recovery_candidates", "list[dict]", "Step7: 基于active_error_types生成的候选恢复结构", "policy/evaluator", "governance/act", "每tick更新", "下一tick覆盖"),
]

LEARNING_CONTEXT_FIELDS = [
    FieldDef("prediction_error", "dict", "预测与实际偏差摘要", "learning/evaluator", "self_model/telemetry", "每tick结果后更新", "下一tick可覆盖或滚动保留"),
    FieldDef("belief_updates", "list[dict]", "假设、估计或策略信念更新", "learning", "world_model/self_model/audit", "每tick或阶段更新", "保留最近N条"),
    FieldDef("strategy_adjustments", "list[dict]", "策略偏好微调摘要", "learning/planner", "policy/telemetry", "每tick或阶段更新", "保留最近N条"),
    FieldDef("lesson_candidates", "list[str]", "本次经验中抽出的潜在教训", "learning", "memory_subsystem/planner", "每tick更新", "经审核固化或丢弃"),
    FieldDef("repeated_failure_patterns", "list[dict]", "重复失败模式摘要", "learning/telemetry", "planner/self_model/governance", "周期性更新", "保留最近N条"),
    # C2: Error Taxonomy
    FieldDef("error_taxonomy_hits", "list[dict]", "Step9: 错误类型命中记录 (error_type, severity, evidence, recovery_used, outcome)", "learning/telemetry", "governance/self_model/audit", "每error触发后更新", "保留最近N条"),
    FieldDef("error_to_recovery_map", "dict", "Step10: 错误类型->恢复结构的当前映射 (由长期学习更新)", "learning", "governance/policy", "条件变化时更新", "长期保持"),
]

GOVERNANCE_CONTEXT_FIELDS = [
    FieldDef("hard_constraints", "list[str]", "不可违反的硬约束", "governance/runtime_setup", "policy/act", "初始化与条件变化时更新", "明确解除前持续有效"),
    FieldDef("soft_constraints", "list[str]", "可权衡的软约束", "governance", "policy/planner", "每tick或阶段更新", "条件变化后更新"),
    FieldDef("budget_state", "dict", "当前预算状态", "governance/runtime", "policy/planner", "每tick更新", "下一tick覆盖"),
    FieldDef("escalation_flags", "list[str]", "需要更保守或更高审查等级的信号", "governance/self_model", "policy/planner", "触发时更新", "风险解除后清理"),
    FieldDef("shutdown_flags", "list[str]", "需要暂停、退出或熔断的信号", "governance/runtime", "main_loop", "触发时更新", "被处理后清理"),
    FieldDef("mode", "str", "当前治理模式", "governance", "policy/planner/act", "每tick或阶段更新", "下一次更新覆盖"),
    # C2: Error Taxonomy
    FieldDef("recovery_requests", "list[dict]", "Step8: 当前活跃的恢复请求 (error_type, recovery_struct, urgency)", "governance/learning", "policy/act", "error触发时写入", "处理后清理"),
]

TELEMETRY_SUMMARY_FIELDS = [
    FieldDef("tick_latency_ms", "float", "当前tick总耗时", "runtime", "telemetry/eval/self_model", "每tick更新", "下一tick覆盖"),
    FieldDef("module_costs", "dict", "各模块耗时或开销摘要", "runtime", "eval/optimization/self_model", "每tick更新", "可滚动聚合"),
    FieldDef("major_events", "list[dict]", "本tick关键事件", "runtime/all_modules", "telemetry/audit/learning", "每tick更新", "保留最近N条"),
    FieldDef("anomaly_flags", "list[str]", "运行异常标记", "runtime/self_model/governance", "eval/audit", "触发时更新", "问题解除后清理"),
    FieldDef("performance_snapshot", "dict", "当前表现摘要", "eval_bridge/telemetry", "self_model/learning", "每tick或每episode更新", "可滚动聚合"),
    # C2: Error Taxonomy
    FieldDef("error_recovery_trace", "list[dict]", "Step9: 错误恢复追踪 (tick, error_type, recovery, outcome, delta)", "telemetry/learning", "audit/self_model/governance", "每recovery执行后更新", "保留最近N条"),
]

REPRESENTATION_CONTEXT_FIELDS = [
    FieldDef("retrieved_card_ids", "list[str]", "Step3: 从store检索到的相关card_ids", "representations", "core/planner/learning", "每tick更新", "下一tick覆盖"),
    FieldDef("match_scores", "dict[str,float]", "Step3: 每张卡的结构匹配分数", "representations", "core/learning", "每tick更新", "下一tick覆盖"),
    FieldDef("active_card_ids", "list[str]", "Step4: activation超过阈值的card_ids", "representations", "core/planner/learning", "每tick更新", "下一tick覆盖"),
    FieldDef("activation_scores", "dict[str,float]", "Step4: 每张卡的激活分数", "representations", "core/planner/learning", "每tick更新", "下一tick覆盖"),
    FieldDef("activation_reasons", "dict[str,str]", "Step4: 每张卡激活原因的人类可读描述", "representations", "core/planner/learning", "每tick更新", "下一tick覆盖"),
    FieldDef("candidate_effects", "dict[str,dict]", "Step5: 每张卡对候选的具体影响(仅当consumer实现后)", "representations", "core", "每tick更新", "下一tick覆盖"),
    FieldDef("advisory_log", "list[dict]", "Step5: 本拍advisory效果记录(仅当consumer实现后)", "representations", "core/telemetry", "每tick更新", "下一tick覆盖"),
    FieldDef("recent_rep_updates", "list[dict]", "Step10: 本拍representation更新的摘要", "representations/lifecycle", "core/learning/audit", "每tick更新", "保留最近N条"),
]

OBJECT_WORKSPACE_FIELDS = [
    FieldDef("surfaced_object_ids", "list[str]", "Phase1: 当前被 surface 到主路径的认知对象 ID", "core/post_commit", "planner/world_model/self_model", "每tick或commit后更新", "下一tick可覆盖"),
    FieldDef("mechanism_object_ids", "list[str]", "Phase6+: 当前被 surface 到主路径的 mechanism memory object ID", "memory/world_model", "world_model/planner/governance", "每tick或commit后更新", "下一tick可覆盖"),
    FieldDef("durable_object_records", "list[dict]", "Phase6+: 跨运行恢复的 formal non-episodic object 快照", "core/memory", "bootstrap/context_builder", "episode end 或 durable snapshot 更新时", "显式覆盖前持续有效"),
    FieldDef("analyst_hypothesis_candidates", "list[dict]", "LLM analyst / initial-goal analyst 生成的瞬时 hypothesis candidates；仅供 reasoning 参考，不写入 formal truth", "llm_analyst_runtime", "context_builder/reasoning", "tick0 初始目标分析或 post-action 分析后可更新", "episode 内软保留或最近一拍覆盖"),
    FieldDef("llm_world_model_snapshot", "dict", "LLM 读到的 world-model 结构化快照；由 raw obs 经 world model 提炼后生成，仅供候选提案与分析参考", "llm_analyst_runtime", "audit/context_builder", "tick0 初始目标分析或 post-action 分析前后可更新", "下一次分析快照覆盖"),
    FieldDef("llm_proposal_candidates", "list[dict]", "LLM 基于 world-model snapshot 产出的 transient goal/relation/action proposals；等待 world model 验证", "llm_analyst_runtime", "audit/context_builder/reasoning", "tick0 初始目标分析或 post-action 分析后更新", "episode 内软保留或最近几拍覆盖"),
    FieldDef("llm_proposal_validation_feedback", "list[dict]", "World model 对 LLM proposal 的一致性/预测反馈；作为后续 analyst 的参考，不直接写 formal truth", "world_model/llm_analyst_runtime", "audit/context_builder/llm_analyst_runtime", "proposal 生成后更新", "episode 内软保留或最近几拍覆盖"),
    FieldDef("object_competitions", "list[dict]", "Phase1: 活跃对象竞争摘要（主要是 hypothesis competition）", "core/post_commit", "world_model/reasoning", "每tick或commit后更新", "下一tick可覆盖"),
    FieldDef("competing_hypotheses", "list[dict]", "PhaseX: 当前 tick 的运行时 hypothesis 状态快照", "core/reasoning", "planner/world_model/audit", "需要时更新", "下一tick可覆盖"),
    FieldDef("competing_hypothesis_objects", "list[dict]", "PhaseX: 当前 tick 的 typed hypothesis object 视图，承载 posterior/predictions 等 richer 语义", "core/reasoning", "context_builder/planner/world_model/audit", "需要时更新", "下一tick可覆盖"),
    FieldDef("active_hypotheses_summary", "list[dict]", "PhaseX: 当前 tick 供统一上下文消费的 hypothesis summary 视图", "core/reasoning", "context_builder/planner/world_model/audit", "需要时更新", "下一tick可覆盖"),
    FieldDef("current_identity_snapshot", "dict", "Phase1: 当前 durable identity object 的摘要快照", "continuity/self_model", "core/planner/governance", "identity commit 后更新", "显式覆盖前持续有效"),
    FieldDef("autobiographical_summary", "dict", "Phase1: 当前 autobiographical object 的摘要", "continuity/memory", "self_model/planner", "autobiographical commit 后更新", "显式覆盖前持续有效"),
    FieldDef("active_tests", "list[str]", "Phase1: 当前 tick 已选中/已激活的 discriminating test selector IDs；不是 ranked candidate 列表，也不承载 full test rows", "core/post_commit", "world_model/reasoning/context_builder", "每tick或commit后更新", "下一tick可覆盖"),
    FieldDef("candidate_tests", "list[dict]", "PhaseX: 当前 tick 的 full candidate test rows；与 active_tests 的 selector IDs 配套", "core/reasoning", "context_builder/planner/world_model", "需要时更新", "下一tick可覆盖"),
    FieldDef("candidate_programs", "list[dict]", "Phase1: 预留给后续 program search 的候选程序槽位", "core/reasoning", "structured_answer/planner", "需要时更新", "下一tick可覆盖"),
    FieldDef("candidate_outputs", "list[dict]", "Phase1: 预留给后续 output search 的候选输出槽位", "core/reasoning", "structured_answer/governance", "需要时更新", "下一tick可覆盖"),
    FieldDef("ranked_discriminating_experiments", "list[dict]", "PhaseX: 当前 tick 的区分性实验候选", "core/reasoning", "planner/governance/audit", "需要时更新", "下一tick可覆盖"),
    FieldDef("posterior_summary", "dict", "PhaseX: 当前 hypothesis posterior 更新摘要", "core/reasoning", "planner/governance/audit", "需要时更新", "下一tick可覆盖"),
    FieldDef("formal_evidence_ledger", "dict", "Formal Evidence Ledger 当前运行摘要；记录 object-layer evidence 的权威位置和最近 evidence id", "core/runtime/evidence_ledger", "context_builder/posterior/audit", "每次 evidence commit 后更新", "最近状态摘要"),
    FieldDef("formal_evidence_recent", "list[dict]", "Formal Evidence Ledger 最近证据的 compact context 视图；不是聊天记忆", "core/runtime/evidence_ledger", "context_builder/posterior/audit", "每次 evidence commit 后更新", "保留最近N条"),
]


class StateSchema:
    """状态 schema 的机器可读定义"""

    TOP_LEVEL_SCHEMAS = {
        "identity": SubSchema("identity", "标识当前系统实例、运行轮次、episode与tick", IDENTITY_FIELDS),
        "goal_stack": SubSchema("goal_stack", "定义系统当前追求什么，以及长短期目标层级关系", GOAL_STACK_FIELDS),
        "world_summary": SubSchema("world_summary", "保存系统对当前外部环境的压缩理解", WORLD_SUMMARY_FIELDS),
        "self_summary": SubSchema("self_summary", "保存系统对自身能力、限制、当前可靠性与资源状态的估计", SELF_SUMMARY_FIELDS),
        "working_memory": SubSchema("working_memory", "保存当前窗口内最高相关的短期上下文", WORKING_MEMORY_FIELDS),
        "episodic_memory_context": SubSchema("episodic_memory_context", "保存当前决策正在利用的历史经验入口", EPISODIC_MEMORY_CONTEXT_FIELDS),
        "decision_context": SubSchema("decision_context", "记录本拍决策的候选、评估、选择和否决痕迹", DECISION_CONTEXT_FIELDS),
        "learning_context": SubSchema("learning_context", "记录系统如何根据本拍结果更新belief、偏好与可复用经验", LEARNING_CONTEXT_FIELDS),
        "governance_context": SubSchema("governance_context", "记录当前有效的执行边界、预算、刹车条件与升级信号", GOVERNANCE_CONTEXT_FIELDS),
        "telemetry_summary": SubSchema("telemetry_summary", "保存供系统自我理解与评估所需的最关键运行摘要", TELEMETRY_SUMMARY_FIELDS),
        "representation_context": SubSchema("representation_context", "中层表征运行时上下文——激活的表征及其对planner的 advisory 影响", REPRESENTATION_CONTEXT_FIELDS),
        "object_workspace": SubSchema("object_workspace", "Phase1 认知对象工作区——已 surface 对象、对象竞争与 identity/autobio 摘要", OBJECT_WORKSPACE_FIELDS),
    }

    # 滑动窗口字段（每 tick 覆盖）
    SLIDING_WINDOW_FIELDS = {
        "working_memory.recent_observations",
        "working_memory.recent_actions",
        "working_memory.recent_outcomes",
        "goal_stack.goal_history",
        "self_summary.recent_failures",
        "telemetry_summary.major_events",
    }

    # 每 tick 覆盖的字段
    PER_TICK_OVERWRITE_FIELDS = {
        "goal_stack.current_focus",
        "world_summary.opportunity_estimate",
        "world_summary.risk_estimate",
        "world_summary.uncertainty_estimate",
        "self_summary.confidence",
        "decision_context.selected_action",
        "decision_context.selection_reason",
        "decision_context.veto_flags",
        "representation_context.retrieved_card_ids",
        "representation_context.match_scores",
        "representation_context.active_card_ids",
        "representation_context.activation_scores",
        "representation_context.activation_reasons",
        "representation_context.candidate_effects",
        "representation_context.advisory_log",
        "representation_context.recent_rep_updates",
        "object_workspace.surfaced_object_ids",
        "object_workspace.object_competitions",
        "object_workspace.competing_hypotheses",
        "object_workspace.active_tests",
        "object_workspace.candidate_tests",
        "object_workspace.candidate_programs",
        "object_workspace.candidate_outputs",
        "object_workspace.ranked_discriminating_experiments",
        "object_workspace.posterior_summary",
        "object_workspace.analyst_hypothesis_candidates",
    }

    @classmethod
    def get_default_state(cls) -> Dict[str, Any]:
        """返回符合 schema 的默认初始状态"""
        return {
            "identity": {
                "agent_id": "",
                "run_id": "",
                "episode_id": "",
                "tick_id": 0,
                "wall_clock_time": "",
            },
            "goal_stack": {
                "top_goal": "survive_and_progress",
                "subgoals": [],
                "current_focus": "",
                "goal_status": {},
                "goal_priority": {},
                "goal_history": [],
            },
            "world_summary": {
                "observed_facts": [],
                "latent_hypotheses": [],
                "opportunity_estimate": 0.5,
                "risk_estimate": 0.5,
                "uncertainty_estimate": 0.5,
                "current_phase": "unknown",
                "predicted_dynamics": [],
                "notable_entities": [],
            },
            "self_summary": {
                "capability_estimate": {},
                "confidence": 0.5,
                "error_flags": [],
                "resource_budget": {
                    "time_budget": 1.0,
                    "energy_budget": 1.0,
                    "compute_budget": 1.0,
                    "risk_budget": 1.0,
                },
                "recent_failures": [],
                "stability_estimate": 0.5,
                "adaptation_readiness": 0.5,
            },
            "working_memory": {
                "recent_observations": [],
                "recent_actions": [],
                "recent_outcomes": [],
                "active_context": {
                    "focus_reason": "",
                    "active_constraints": [],
                    "active_time_horizon": "medium",
                },
                "attention_targets": [],
            },
            "episodic_memory_context": {
                "retrieved_episodes": [],
                "retrieval_scores": [],
                "memory_notes": [],
                "retrieval_query_summary": {},
            },
            "decision_context": {
                "candidate_actions": [],
                "predicted_consequences": [],
                "selected_action": None,
                "selection_reason": "",
                "veto_flags": [],
                "alternative_rankings": [],
            },
            "learning_context": {
                "prediction_error": {},
                "belief_updates": [],
                "strategy_adjustments": [],
                "lesson_candidates": [],
                "repeated_failure_patterns": [],
            },
            "governance_context": {
                "hard_constraints": [],
                "soft_constraints": [],
                "budget_state": {
                    "energy": 100,
                    "energy_explore": 50,
                    "energy_exploit": 50,
                },
                "escalation_flags": [],
                "shutdown_flags": [],
                "mode": "normal",
            },
            "telemetry_summary": {
                "tick_latency_ms": 0.0,
                "module_costs": {},
                "major_events": [],
                "anomaly_flags": [],
                "performance_snapshot": {},
            },
            "representation_context": {
                "retrieved_card_ids": [],
                "match_scores": {},
                "active_card_ids": [],
                "activation_scores": {},
                "activation_reasons": {},
                "candidate_effects": {},
                "advisory_log": [],
                "recent_rep_updates": [],
            },
            "object_workspace": {
                "surfaced_object_ids": [],
                "mechanism_object_ids": [],
                "durable_object_records": [],
                "analyst_hypothesis_candidates": [],
                "llm_world_model_snapshot": {},
                "llm_proposal_candidates": [],
                "llm_proposal_validation_feedback": [],
                "object_competitions": [],
                "competing_hypotheses": [],
                "current_identity_snapshot": {},
                "autobiographical_summary": {},
                "active_tests": [],
                "candidate_tests": [],
                "candidate_programs": [],
                "candidate_outputs": [],
            },
        }

    @classmethod
    def validate_state(cls, state: Dict[str, Any]) -> List[str]:
        """验证状态是否符合 schema，返回错误列表"""
        errors = []
        for top_key, sub_schema in cls.TOP_LEVEL_SCHEMAS.items():
            if top_key not in state:
                errors.append(f"Missing top-level key: {top_key}")
        return errors
