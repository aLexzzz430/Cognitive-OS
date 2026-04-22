# Planner — 长短期计划与子目标管理

**唯一职责**：管理系统的长程目标与规划，维护目标层级的连贯性。

## 契约

**输入**：当前状态 + 活跃目标 + memory 检索结果
**输出**：子目标列表 / 计划更新建议
**可读状态**：state/, memory/, self_model
**禁止副作用**：不得直接修改 state/

## 核心能力

1. **目标分解**：将长程目标分解为可执行的子目标
2. **计划维护**：跟踪计划进度，检测失败
3. **目标优先级**：在多个目标间做出优先级决策
4. **目标修订**：当环境变化时修订计划

## 关键原则

- planner 是 advisory 层，不直接驱动动作
- planner 输出影响 policy 的候选评分
- 目标变更必须走 evolution/ 流程（如涉及系统能力变更）

## 约束

- planner 不准强制 policy 执行特定动作（只能影响评分）
- planner 不准在无充分理由时放弃长程目标
- 规划失败必须有反思记录

## 状态

- 版本：0.1.0（已实现子集）
- 已实现：
  - 计划 schema（`Plan` / `PlanStep` / `ExitCriteria`）
  - 目标分解与策略选择（`ObjectiveDecomposer` + `planning_policy`）
  - 有界 beam search + 搜索前沿保留
  - 单活动计划状态管理（`PlanState`）
  - 基于触发器的计划修订（`PlanReviser`）
  - 阻塞后的 branch salvage 与 surface rebuild
  - 计划约束过滤与可行性记录（`constraint_policy`）
  - 基于计划后缀 rollout 的 world-model lookahead 重规划
- 未完成项（仍在演进）：
  - 多计划并行/栈式管理
  - 更深的分支搜索与依赖图规划
  - 完整跨 episode 计划历史与回滚机制
