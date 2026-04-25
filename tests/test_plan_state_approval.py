from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.plan_schema import ExitCriteria, Plan, PlanStatus, PlanStep
from planner.plan_state import PlanState


def test_plan_state_preserves_explicit_high_risk_approval_override() -> None:
    plan = Plan(
        plan_id="plan-local",
        goal="inspect local mirror",
        steps=[
            PlanStep(
                step_id="record-finding",
                description="Persist the key investigation finding with evidence references",
                intent="test",
                target_function="note_write",
                approval_requirement={
                    "required": False,
                    "risk_level": "low",
                    "reason": "local_machine_mirror_control_write",
                    "allow_high_risk_without_approval": True,
                },
            )
        ],
        exit_criteria=ExitCriteria(),
        status=PlanStatus.ACTIVE,
    )
    state = PlanState()
    state.set_plan(plan)

    summary = state.get_plan_summary()
    active_requirement = summary["active_task_node"]["approval_requirement"]
    authority_requirement = summary["execution_authority"]["active_task"]["approval_requirement"]

    assert active_requirement["required"] is False
    assert active_requirement["allow_high_risk_without_approval"] is True
    assert authority_requirement["required"] is False
    assert authority_requirement["allow_high_risk_without_approval"] is True
