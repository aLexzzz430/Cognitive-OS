# Core — 唯一主循环与运行时

**唯一职责**：承载系统主 tick 循环，是唯一允许直接驱动状态更新的入口。

## 契约（必须先定义）

**输入**：
- 当前 tick 编号
- 全局状态（只读）

**输出**：
- 更新后的全局状态
- 各模块 telemetry 数据

**可读共享状态**：
- `state/` 下的全局状态
- 各模块注册的标准接口

**禁止副作用**：
- 禁止直接调用 governance/policy/world_model 以外的决策逻辑
- 禁止在 core/ 内硬编码任何模块的特殊路径

## 架构原则

```
Tick_N:
  1. perception感知()        → encoded_obs
  2. state更新()             → state（N层状态融合）
  3. memory检索()            → retrieved
  4. world_model预测()       → prediction
  5. policy候选生成()       → candidates
  6. governance过滤()       → filtered_candidates
  7. policy行动选择()       → chosen_action
  8. action执行()           → reward/observation
  9. feedback更新()         → state_update
  10. 长程目标检查()         → goal_check
```

## 分层策略

| 层 | 内容 | 每 tick 必走？ |
|---|------|--------------|
| 在线必需 | 1,2,7,8,9 | 是 |
| 在线 advisory | 3,4,5,6 | 否（按需 gate） |
| 离线分析 | telemetry, eval | 异步 |
| 离线进化 | evolution | 后台 |

## 约束

- core/ 是唯一的正式主循环，不允许第二条并行主循环
- 主循环必须可配置切换各层模块（禁用/替换/熔断）
- 所有昂贵操作必须通过配置 gate 控制

## 状态

- 版本：0.1.0（已实现）
- 10 步全部实现（见 core/main_loop.py）
- tick_id 递增正确（5 ticks → tick_id=5）
- 决策链路：observe → refresh → retrieve → model → candidates → predict → govern → act → evaluate → learn

## 实现文件

```
core/
├── __init__.py
├── main_loop.py                            # 主循环入口与编排
├── main_loop_components.py                 # 主循环组件装配
├── orchestration/runtime_stage_modules.py  # 分阶段运行时模块
└── orchestration/planner_runtime.py        # planner 运行时桥接
```
