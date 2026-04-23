from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.adapter_registry import find_adapter_registry_violations
from core.conos_repository_layout import (
    LAYER_ADAPTER,
    LAYER_CONOS_CORE,
    LAYER_PRIVATE_COGNITIVE_CORE,
    LAYER_RUNTIME,
    classify_repo_path,
    describe_repo_layers,
    find_forbidden_public_core_imports,
)
from core.orchestration.goal_task_control import (
    GOAL_TASK_AUTHORITY_BUILDER_VERSION,
    GoalContract,
    TaskGraph,
    TaskNode,
    TaskVerificationGate,
    build_goal_task_authority_context,
)
from core.orchestration.execution_control import (
    ApprovalPolicy,
    ToolCapabilityRegistry,
    build_policy_block_result,
    issue_execution_ticket,
)
from core.orchestration.verifier_runtime import VERIFIER_RUNTIME_VERSION, build_verifier_runtime
from core.reasoning.posterior_update import update_hypothesis_posteriors
from scripts.check_runtime_preflight import _run_checks


def test_core_path_classification() -> None:
    assert classify_repo_path("core/adapter_registry.py") == LAYER_CONOS_CORE


def test_adapter_path_classification() -> None:
    candidate = Path("integrations/arc_agi3/perception_bridge.py")
    if not candidate.exists():
        adapter_paths = sorted(Path("integrations").glob("**/*.py"))
        assert adapter_paths, "No adapter paths found in integrations/."
        candidate = adapter_paths[0]
    assert classify_repo_path(candidate.as_posix()) == LAYER_ADAPTER


def test_adapter_registry_has_no_boundary_violations() -> None:
    assert find_adapter_registry_violations() == []


def test_repo_layer_summary_is_non_empty_and_contains_key_layers() -> None:
    summaries = describe_repo_layers()
    assert summaries
    layer_names = {summary.layer_name for summary in summaries}
    assert LAYER_CONOS_CORE in layer_names
    assert LAYER_ADAPTER in layer_names
    assert LAYER_PRIVATE_COGNITIVE_CORE in layer_names
    assert LAYER_RUNTIME in layer_names


def test_private_classified_modules_are_boundary_import_checked(tmp_path: Path) -> None:
    private_module = tmp_path / "core" / "orchestration" / "structured_answer.py"
    private_module.parent.mkdir(parents=True)
    private_module.write_text("from scripts.local_mirror import main\n", encoding="utf-8")

    findings = find_forbidden_public_core_imports(tmp_path)

    assert findings
    assert findings[0]["path"] == "core/orchestration/structured_answer.py"
    assert findings[0]["layer"] == LAYER_PRIVATE_COGNITIVE_CORE
    assert findings[0]["import"] == "scripts.local_mirror"


def test_runtime_preflight_exposes_public_checks() -> None:
    results = _run_checks(strict_dev=False)
    result_names = {result.name for result in results}
    assert "python_version" in result_names
    assert "core_import" in result_names
    assert "repo_layout" in result_names
    assert "entrypoint:run_arc_agi3.py" in result_names
    assert "entrypoint:run_local_machine.py" in result_names
    assert "entrypoint:run_webarena.py" in result_names
    assert "entrypoint:conos.py" in result_names
    assert "entrypoint:local_mirror.py" in result_names
    assert "dev_dependency:pytest" in result_names


def test_goal_task_authority_builder_is_single_context_entrypoint() -> None:
    goal = GoalContract(
        goal_id="goal-smoke",
        title="finish smoke",
        success_criteria=["smoke passes"],
    )
    task = TaskNode(
        node_id="task-smoke",
        title="run smoke",
        status="active",
        goal_id=goal.goal_id,
        success_criteria=["smoke passes"],
        metadata={"intent": "verify", "target_function": "check_smoke"},
    )
    graph = TaskGraph(
        graph_id="graph-smoke",
        goal_id=goal.goal_id,
        active_node_id=task.node_id,
        nodes=[task],
    )
    context = build_goal_task_authority_context(
        goal_contract=goal,
        task_graph=graph,
        authority_source="public_smoke",
        authority_integrity="complete",
        run_id="smoke",
        episode=1,
        tick=2,
    )
    assert context["goal_ref"] == goal.goal_id
    assert context["task_ref"] == task.node_id
    assert context["graph_ref"] == graph.graph_id
    assert context["authority_snapshot"]["source"] == "public_smoke"
    assert context["authority_snapshot"]["builder_version"] == GOAL_TASK_AUTHORITY_BUILDER_VERSION
    assert context["task_contract"]["freshness_binding"]["authority_source"] == "public_smoke"
    assert context["completion_gate"]["verifier_authority"] == context["verifier_authority"]
    assert context["verifier_runtime"]["runtime_version"] == VERIFIER_RUNTIME_VERSION


def test_verifier_runtime_unifies_completion_rollback_and_posterior_teaching() -> None:
    goal = GoalContract(
        goal_id="goal-verify",
        title="verify before finish",
        success_criteria=["verification passes"],
    )
    task = TaskNode(
        node_id="task-verify",
        title="verify output",
        status="active",
        goal_id=goal.goal_id,
        verification_gate=TaskVerificationGate(
            required=True,
            last_verified=False,
            last_verdict="failed",
            verifier_function="check_output",
        ),
        rollback_edge={
            "target_step_id": "prepare",
            "target_node_id": "task-prepare",
            "reason": "verification_failed",
        },
    )
    runtime = build_verifier_runtime(
        goal_contract=goal,
        task_graph=TaskGraph(
            graph_id="graph-verify",
            goal_id=goal.goal_id,
            active_node_id=task.node_id,
            nodes=[task],
        ),
        active_task=task,
        completion_gate={
            "requires_verification": True,
            "failed_verification_node_ids": [task.node_id],
            "blocked_reasons": ["verification_failed"],
        },
    )
    assert runtime.completion_gate["verifier_authority"] == runtime.verifier_authority
    assert runtime.verifier_authority["decision"] == "block_completion"
    assert runtime.rollback["eligible"] is True
    assert runtime.rollback["target_step_id"] == "prepare"
    assert runtime.posterior_teaching["teaching_signal"] == "negative"


def test_posterior_update_consumes_verifier_teaching_signal() -> None:
    update = update_hypothesis_posteriors(
        [
            {
                "hypothesis_id": "h1",
                "summary": "checking output should solve the task",
                "posterior": 0.5,
                "target_tokens": ["check", "output"],
            }
        ],
        action={"function_name": "check_output"},
        result={},
        predicted_transition={},
        actual_transition={},
        reward=0.0,
        information_gain=0.0,
        verifier_teaching={
            "teaching_signal": "negative",
            "teaching_signal_score": -1.0,
        },
    )
    events = list(update.get("posterior_events", []) or [])
    assert events
    assert "verifier_teaching_negative" in events[0]["reason"]


def test_execution_ticket_marks_sandbox_best_effort_and_audits_boundaries() -> None:
    action = {
        "payload": {
            "tool_args": {
                "function_name": "write_file",
                "kwargs": {
                    "output_path": "runtime/reports/out.json",
                    "url": "https://example.com/upload",
                    "api_key": "redacted-by-audit",
                },
            }
        }
    }
    registry = ToolCapabilityRegistry(
        tools=[
            {
                "name": "write_file",
                "side_effect_class": "filesystem_write",
                "capability_class": "filesystem_mutation",
                "approval_required": True,
                "risk_level": "high",
            }
        ]
    )
    context = {
        "goal_ref": "goal-audit",
        "task_ref": "task-audit",
        "graph_ref": "graph-audit",
        "run_id": "run-audit",
        "episode": 1,
        "tick": 2,
    }
    decision = ApprovalPolicy().evaluate(action, registry, context=context)
    ticket = issue_execution_ticket(action=action, decision=decision, context=context)
    audit = ticket.metadata["sandbox_audit"]

    assert audit["sandbox_label"] == "best_effort"
    assert audit["not_os_security_sandbox"] is True
    assert audit["file_audit"]["detected"] is True
    assert audit["write_path_audit"]["paths"][0]["path"] == "runtime/reports/out.json"
    assert audit["network_audit"]["targets"][0]["host"] == "example.com"
    assert "kwargs.api_key" in audit["credential_audit"]["sensitive_arg_keys"]
    assert audit["credential_audit"]["values_redacted"] is True

    block = build_policy_block_result(ticket=ticket, decision=decision)
    assert block["audit_event"]["payload"]["sandbox_audit"]["sandbox_label"] == "best_effort"
