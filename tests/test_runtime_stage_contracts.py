from __future__ import annotations

from dataclasses import is_dataclass
import inspect
from pathlib import Path

from core.orchestration import governance_stage, planner_stage, retrieval_stage, state_sync_stage
from core.orchestration.stage_types import (
    GovernanceStageInput,
    GovernanceStageOutput,
    PlannerStageInput,
    PlannerStageOutput,
    RetrievalStageInput,
    RetrievalStageOutput,
    StateSyncStageInput,
    StateSyncStageOutput,
)


def test_runtime_stage_contracts_are_dataclass_boundaries() -> None:
    for contract in (
        RetrievalStageInput,
        RetrievalStageOutput,
        PlannerStageInput,
        PlannerStageOutput,
        GovernanceStageInput,
        GovernanceStageOutput,
        StateSyncStageInput,
        StateSyncStageOutput,
    ):
        assert is_dataclass(contract)


def test_stage_modules_expose_loop_independent_run_contracts() -> None:
    for stage_cls in (
        retrieval_stage.RetrievalStage,
        planner_stage.PlannerStage,
        governance_stage.GovernanceStage,
        state_sync_stage.StateSyncStage,
    ):
        signature = inspect.signature(stage_cls.run)
        assert list(signature.parameters) == ["self", "loop", "stage_input"]


def test_main_loop_split_target_has_existing_stage_modules() -> None:
    stage_files = [
        Path("core/orchestration/retrieval_stage.py"),
        Path("core/orchestration/planner_stage.py"),
        Path("core/orchestration/governance_stage.py"),
        Path("core/orchestration/staged_execution_runtime.py"),
        Path("core/orchestration/state_sync_stage.py"),
    ]

    assert all(path.exists() for path in stage_files)
    assert sum(len(path.read_text(encoding="utf-8").splitlines()) for path in stage_files) > 0
