from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cognition.unified_context import UnifiedCognitiveContext
from core.runtime.end_to_end_learning import (
    EndToEndLearningRuntime,
    apply_learning_context_to_unified_context,
)
from core.runtime.failure_learning import FAILURE_LEARNING_VERSION, normalize_failure_learning_object
from core.runtime.state_store import RuntimeStateStore
from integrations.local_machine.runner import run_local_machine_task


def test_learning_store_dedupes_and_marks_lessons_used(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "state.sqlite3")
    first = store.record_learning_lesson(
        task_family="local_machine",
        trigger="local_machine:placeholder_test_generation",
        lesson={"title": "Avoid placeholder tests", "hint": "write concrete assertions"},
        source_run_id="run-a",
        confidence=0.5,
    )
    second = store.record_learning_lesson(
        task_family="local_machine",
        trigger="local_machine:placeholder_test_generation",
        lesson={"title": "Avoid placeholder tests", "hint": "write concrete assertions"},
        source_run_id="run-b",
        confidence=0.8,
    )

    lessons = store.list_learning_lessons(task_family="local_machine")
    store.mark_learning_lesson_used(first)
    used = store.list_learning_lessons(task_family="local_machine")[0]

    assert first == second
    assert len(lessons) == 1
    assert lessons[0]["confidence"] == 0.8
    assert used["use_count"] == 1


def test_failure_learning_store_dedupes_and_marks_objects_used(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "state.sqlite3")
    failure_object = normalize_failure_learning_object(
        failure_mode="api_surface_mismatch",
        title="API mismatch",
        summary="AttributeError during validation",
        violated_assumption="Generated test API existed without source evidence",
        evidence_refs=["validation:attribute_error"],
        new_regression_test={"description": "outline source before repair"},
        new_governance_rule={"description": "require API evidence before accepting tests"},
        retrieval_tags=["python_generation", "requires_tests"],
        source_run_id="run-a",
        source_task_family="local_machine",
        confidence=0.8,
    )
    first = store.record_failure_learning_object(
        task_family="local_machine",
        failure_mode="api_surface_mismatch",
        failure_object=failure_object,
        source_run_id="run-a",
        confidence=0.6,
    )
    second = store.record_failure_learning_object(
        task_family="local_machine",
        failure_mode="api_surface_mismatch",
        failure_object=failure_object,
        source_run_id="run-b",
        confidence=0.9,
    )

    failures = store.list_failure_learning_objects(task_family="local_machine")
    store.mark_failure_learning_object_used(first)
    used = store.list_failure_learning_objects(task_family="local_machine")[0]

    assert first == second
    assert len(failures) == 1
    assert failures[0]["confidence"] == 0.9
    assert failures[0]["failure_object"]["schema_version"] == FAILURE_LEARNING_VERSION
    assert used["use_count"] == 1


def test_local_machine_audit_records_lessons_and_updates_unified_context(tmp_path: Path) -> None:
    runtime = EndToEndLearningRuntime(db_path=tmp_path / "state.sqlite3")
    report = runtime.learn_from_local_machine_audit(
        run_id="failed-run",
        instruction="build a GitHub-ready AI program",
        audit={
            "local_machine_artifact_check": {
                "ok": False,
                "failures": [
                    "latest_command_succeeded",
                    "required_workspace_path:generated/*/tests/*.py",
                ],
            },
            "final_surface_raw": {
                "local_mirror": {
                    "sync_plan": {"actionable_changes": [{"relative_path": "generated/app.py"}]},
                    "audit_events": [
                        {
                            "event_type": "mirror_command_executed",
                            "payload": {
                                "returncode": 1,
                                "stderr_tail": "RuntimeError: placeholder content detected",
                            },
                        }
                    ],
                }
            },
        },
    )
    lessons = runtime.state_store.list_learning_lessons(task_family="local_machine")
    context = runtime.learning_context_for_task(task_family="local_machine", objective="build another AI project")
    unified = UnifiedCognitiveContext.from_parts(current_goal="build")
    applied = apply_learning_context_to_unified_context(unified, context)

    triggers = {item["trigger"] for item in lessons}
    failure_modes = {
        item["failure_mode"]
        for item in runtime.state_store.list_failure_learning_objects(task_family="local_machine")
    }
    assert report["recorded_count"] >= 3
    assert report["recorded_failure_object_count"] >= 3
    assert "local_machine:placeholder_test_generation" in triggers
    assert "local_machine:missing_required_tests" in triggers
    assert "placeholder_generation" in failure_modes
    assert "missing_required_artifact" in failure_modes
    assert "Avoid placeholder tests" in context["hint_text"]
    assert "Structured failure objects" in context["failure_hint_text"]
    assert context["failure_behavior_rules"]["rule_count"] > 0
    assert context["failure_behavior_rules"]["retrieval_objects"]
    assert applied["lessons"]["applied"] == len(context["lessons"])
    assert applied["failure_learning"]["applied"] == len(context["failure_objects"])
    assert unified.recent_failure_profile
    assert unified.evidence_queue
    assert unified.posterior_summary["end_to_end_learning"]["lesson_count"] == len(context["lessons"])
    assert unified.posterior_summary["failure_learning"]["failure_object_count"] == len(context["failure_objects"])


def test_local_machine_audit_records_api_surface_mismatch_lesson(tmp_path: Path) -> None:
    runtime = EndToEndLearningRuntime(db_path=tmp_path / "state.sqlite3")
    report = runtime.learn_from_local_machine_audit(
        run_id="api-mismatch-run",
        instruction="build a GitHub-ready AI program with tests",
        audit={
            "local_machine_artifact_check": {"ok": False, "failures": ["latest_command_succeeded"]},
            "final_surface_raw": {
                "local_mirror": {
                    "audit_events": [
                        {
                            "event_type": "mirror_command_executed",
                            "payload": {
                                "returncode": 1,
                                "stderr_tail": (
                                    "tests/test_analyzer.py:4: AttributeError: "
                                    "'SentimentAnalyzer' object has no attribute 'analyze'"
                                ),
                            },
                        }
                    ],
                }
            },
        },
    )
    lessons = runtime.state_store.list_learning_lessons(task_family="local_machine")
    failures = runtime.state_store.list_failure_learning_objects(task_family="local_machine")
    api_lesson = next(item for item in lessons if item["trigger"] == "local_machine:api_surface_mismatch")
    api_failure = next(item for item in failures if item["failure_mode"] == "api_surface_mismatch")

    assert report["recorded_count"] >= 2
    assert report["recorded_failure_object_count"] >= 2
    assert "api_surface" in api_lesson["lesson"]["tags"]
    assert api_failure["failure_object"]["violated_assumption"]
    assert api_failure["failure_object"]["new_governance_rule"]["required_signal"] == "api_surface_evidence_ref"
    assert api_lesson["lesson"]["evidence"]["attribute_errors"] == [
        {"kind": "object", "class_name": "SentimentAnalyzer", "attribute": "analyze"}
    ]


def test_local_machine_audit_records_governance_block_failure_object(tmp_path: Path) -> None:
    runtime = EndToEndLearningRuntime(db_path=tmp_path / "state.sqlite3")
    report = runtime.learn_from_local_machine_audit(
        run_id="governance-block-run",
        instruction="调查项目并修复一个小问题",
        audit={
            "local_machine_artifact_check": {"ok": True, "failures": []},
            "final_surface_raw": {
                "local_mirror": {
                    "action_governance": {
                        "last_decision": {
                            "status": "BLOCKED",
                            "blocked_reason": "evidence_refs_required_before_mirror_write",
                            "required_evidence": ["file_read"],
                            "request": {
                                "action_name": "apply_patch",
                                "permissions_required": ["propose_patch", "edit_mirror"],
                            },
                        }
                    }
                }
            },
        },
    )

    failures = runtime.state_store.list_failure_learning_objects(task_family="local_machine")
    governance_failure = next(item for item in failures if item["failure_mode"] == "governance_block")

    assert report["recorded_failure_object_count"] == 1
    assert governance_failure["failure_object"]["bad_policy"] == "evidence_refs_required_before_mirror_write"
    assert governance_failure["failure_object"]["new_regression_test"]["blocked_reason"] == "evidence_refs_required_before_mirror_write"


def test_learning_context_filters_generation_lessons_out_of_project_maintenance(tmp_path: Path) -> None:
    runtime = EndToEndLearningRuntime(db_path=tmp_path / "state.sqlite3")
    runtime.record_lesson(
        task_family="local_machine",
        trigger="local_machine:placeholder_test_generation",
        lesson={
            "title": "Avoid placeholder tests",
            "hint": "write concrete assertions for generated AI projects",
            "tags": ["ai_project_generation", "python_generation", "requires_tests"],
        },
        source_run_id="ai-run",
        confidence=0.9,
    )
    runtime.record_lesson(
        task_family="local_machine",
        trigger="local_machine:successful_artifact_contract",
        lesson={
            "title": "Preserve artifact-first build order",
            "hint": "execute, verify, then plan sync",
            "tags": ["local_machine", "general"],
        },
        source_run_id="success-run",
        confidence=0.5,
    )

    context = runtime.learning_context_for_task(
        task_family="local_machine",
        objective="在电脑上任意选一个小型项目，然后改进 README 和维护文件",
        mark_used=False,
    )
    triggers = {lesson["trigger"] for lesson in context["lessons"]}

    assert "project_maintenance" in context["objective_tags"]
    assert "local_machine:placeholder_test_generation" not in triggers
    assert "Avoid placeholder tests" not in context["hint_text"]


def test_learning_context_keeps_generation_lessons_for_ai_project_task(tmp_path: Path) -> None:
    runtime = EndToEndLearningRuntime(db_path=tmp_path / "state.sqlite3")
    runtime.record_lesson(
        task_family="local_machine",
        trigger="local_machine:api_surface_mismatch",
        lesson={
            "title": "Repair source and tests against the real API surface",
            "hint": "include source and tests when AttributeError appears",
            "tags": ["ai_project_generation", "api_surface", "python_generation", "requires_tests"],
        },
        source_run_id="failed-ai-run",
        confidence=0.85,
    )

    context = runtime.learning_context_for_task(
        task_family="local_machine",
        objective="制作一个可以上线GitHub的AI程序，并包含 pytest 测试",
        mark_used=False,
    )

    assert "ai_project_generation" in context["objective_tags"]
    assert context["lesson_count"] == 1
    assert "real API surface" in context["hint_text"]


def test_local_machine_learning_persists_into_next_run_environment(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    first_source = tmp_path / "source-a"
    first_source.mkdir()
    first_mirror = tmp_path / "mirror-a"

    first_audit = run_local_machine_task(
        instruction="build a GitHub-ready AI program",
        source_root=str(first_source),
        mirror_root=str(first_mirror),
        default_command=[
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "Path('generated').mkdir(); "
                "Path('generated/app.py').write_text('print(\"partial\")\\n', encoding='utf-8'); "
                "raise SystemExit('placeholder content detected')"
            ),
        ],
        allowed_commands=[sys.executable],
        run_id="learning-first",
        max_ticks_per_episode=3,
        reset_mirror=True,
        daemon=True,
        supervisor_db=str(db_path),
        allow_empty_exec=True,
        require_artifacts=True,
        required_artifact_paths=["generated/*/tests/*.py"],
        execution_backend="local",
    )

    store = RuntimeStateStore(db_path)
    assert first_audit["long_run_supervisor"]["run"]["status"] == "FAILED"
    assert any(
        lesson["trigger"] == "local_machine:placeholder_test_generation"
        for lesson in store.list_learning_lessons(task_family="local_machine")
    )

    second_source = tmp_path / "source-b"
    second_source.mkdir()
    second_mirror = tmp_path / "mirror-b"
    second_audit = run_local_machine_task(
        instruction="build a second GitHub-ready AI program",
        source_root=str(second_source),
        mirror_root=str(second_mirror),
        default_command=[
            sys.executable,
            "-c",
            (
                "import os; "
                "from pathlib import Path; "
                "Path('generated').mkdir(); "
                "Path('generated/hints.txt').write_text(os.environ.get('CONOS_LEARNING_HINTS', ''), encoding='utf-8')"
            ),
        ],
        allowed_commands=[sys.executable],
        run_id="learning-second",
        max_ticks_per_episode=3,
        reset_mirror=True,
        supervisor_db=str(db_path),
        allow_empty_exec=True,
        execution_backend="local",
    )

    hints = (second_mirror / "workspace" / "generated" / "hints.txt").read_text(encoding="utf-8")
    injected = second_audit["end_to_end_learning"]["injected"]

    assert injected["lesson_count"] >= 1
    assert "Avoid placeholder tests" in hints
    assert "concrete assertions" in second_audit["local_machine_instruction"]
