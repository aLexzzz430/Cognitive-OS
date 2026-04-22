# core/main_loop.py Migration Note

这份文档保留的是一次旧的迁移意图说明，不再描述当前仓库中的真实文件布局。

## Current Status

- `core/main_loop.py` 已经是当前仓库中的正式主循环实现。
- `modules/state/manager.py` 仍然是正式状态入口。
- `PRIMARY_EXECUTION_AUTHORITY.md` 和 `tests/test_primary_execution_authority.py` 才是当前执行权威约束的最新文档/测试来源。

## Historical Context

早期计划曾打算把一个外部评审 runner 的逻辑迁入主循环，但相关 `eval/...` 目录和文件已经不在当前仓库里，因此这里不再引用那些旧路径。

## What To Use Instead

- 想看正式主路径：`core/main_loop.py`
- 想看 post-commit / orchestration 分层：`core/orchestration/`
- 想看主循环约束：`PRIMARY_EXECUTION_AUTHORITY.md`
- 想看回归约束：`tests/test_primary_execution_authority.py`
