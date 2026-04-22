# The-AGI

当前仓库正在从 “AGI project bundle” 收口成更清晰的 `Con OS + adapters + evals + private research` 形态。它不再把 ARC / WebArena / latent transfer / 结构化 ARC solver 叙述成公开 Con OS kernel 本体，而是把这些能力逐步降级成 adapter、评测层或私有研究层。

## 当前仓库重点

- 公开 Con OS 内核：`core/`、`planner/`、`decision/`、`modules/`、`evolution/`
- adapter 层：`integrations/arc_agi3/`、`integrations/webarena/`、`integrations/survival_world/`
- eval 层：`eval/`、`conos_evals/`、`scripts/arc_agi2_eval.py`、`scripts/arc_agi2_compare_modes.py`、`scripts/arc_agi2_curriculum_eval.py`、`scripts/capability_smoke_eval.py`、`scripts/cognitive_curriculum_eval.py`、`scripts/object_relation_blind_eval.py`、`scripts/regression_runtime_report.py`、`scripts/run_arc_agi3_all_games_llm_batch.py`
- private research 层：`private_cognitive_core/`、`scripts/` 下的 latent / multi-domain research shims、`modules/hypothesis/mechanism_posterior_updater.py`、`core/orchestration/structured_answer.py`

## Repo layering

仓库已经开始按逻辑层分为 4 类：

- `conos-core/`
  当前是 staging 目录；公开 kernel 的真实代码还主要位于 `core/`、`planner/`、`decision/`、`modules/`、`evolution/`
- `conos-evals/`
  当前是 staging 目录；真实评测代码主要位于 `eval/`、`conos_evals/` 和若干兼容 shim `scripts/`
- `private-cognitive-core/`
  当前是 staging 目录；承接 latent transfer、mechanism posterior、structured ARC solver 这类研究型能力

- `private_cognitive_core/`
  已经开始承接真正迁出的 research pack 实现；旧 `scripts/` 下对应的 multi-domain 入口逐步退化为兼容 shim
- `integrations/`
  明确是 adapter 层，不再被视为内核本体

可以运行下面的检查来验证这个边界：

```bash
python scripts/check_conos_repo_layout.py
```

`conos-core` 现在也开始通过统一 `AdapterRegistry` 访问 adapter，而不是在主循环里散落具体 `integrations.*` 模块路径。

## 安装

### Core runtime

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### Dev / test extras

```bash
pip install -e ".[dev]"
```

如果你更习惯 requirements 文件，也可以运行：

```bash
pip install -r requirements-dev.txt
```

### ARC-AGI-3 optional extras

```bash
pip install -e ".[arc3]"
```

如果本地拿不到 `arc-agi` / `arcengine`，主仓库仍应可运行；相关能力保持为可选集成路径。

## 常用检查

```bash
python scripts/check_docs_references.py
pytest -q tests/test_primary_execution_authority.py
pytest -q tests/test_unified_context_contract.py tests/test_structured_answer_synthesizer.py
```

## 运行时布局

仓库现在明确区分 3 类目录：

- source tree：受版本控制的代码、测试、文档；这里不应落运行时状态、评测结果或模型产物
- runtime tree：默认位于 `runtime/`，承接主循环和本地工具产生的可变状态
- eval outputs：默认位于 `runtime/evals/`，承接评测报告、批量运行输出、审计和数据集缓存

默认运行时路径：

- `runtime/logs/event_log.jsonl`
- `runtime/state/state.json`
- `runtime/representations/runtime_updates.jsonl`
- `runtime/reports/`
- `runtime/models/`
- `runtime/evals/reports/`
- `runtime/evals/runs/`
- `runtime/evals/logs/`
- `runtime/evals/audits/`
- `runtime/evals/datasets/`

可以通过环境变量覆盖：

- `THE_AGI_RUNTIME_ROOT`
- `THE_AGI_EVAL_ROOT`
- `THE_AGI_MODEL_ARTIFACTS_ROOT`
- `THE_AGI_EVENT_LOG_PATH`
- `THE_AGI_STATE_PATH`
- `THE_AGI_REPRESENTATION_UPDATES_PATH`

回归耗时观测默认也写入评测树：

```bash
python3 scripts/regression_runtime_report.py
python3 scripts/regression_runtime_report.py --include-full
```

默认会生成：

- `runtime/evals/reports/regression_runtime_report.json`
- `runtime/evals/reports/regression_runtime_report.md`

默认情况下，`CoreMainLoop` 会以 fresh state 启动，不再自动从已有 `state.json` 恢复运行态。
如果确实需要显式恢复持久化状态，请设置：

```bash
export THE_AGI_RESUME_STATE=1
```

## 配置说明（MiniMax）

`modules/llm/minimax_client.py` 不再内置任何本地绝对路径。请通过以下方式注入凭据（按优先级）：

1. 构造参数 `api_token`
2. 构造参数 `token_file`
3. 环境变量 `MINIMAX_API_TOKEN`
4. 环境变量 `MINIMAX_TOKEN_FILE`

最小示例：

```bash
export MINIMAX_API_TOKEN="your_token_here"
python -c "from modules.llm.minimax_client import MinimaxClient; print(MinimaxClient())"
```

或显式注入：

```python
from modules.llm.minimax_client import MinimaxClient

client = MinimaxClient(api_token="your_token_here")
# client = MinimaxClient(token_file="/path/to/api_token.txt")
```
