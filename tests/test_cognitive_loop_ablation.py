from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evaluation.cognitive_loop_ablation import (
    ARM_BASELINE_LLM,
    ARM_FULL,
    ARM_NO_EXPERIMENT,
    ARM_NO_POSTERIOR,
    ARM_NO_SEMANTIC,
    DEFAULT_ARMS,
    BASELINE_LLM_DECISIONS_VERSION,
    COGNITIVE_LOOP_ABLATION_VERSION,
    build_cognitive_loop_benchmark_tasks,
    collect_baseline_llm_decisions,
    render_cognitive_loop_ablation_report,
    run_cognitive_loop_ablation,
)
import core.evaluation.cognitive_loop_ablation as ablation_module
from scripts.cognitive_loop_ablation import main as ablation_main


def test_build_cognitive_loop_benchmark_tasks_covers_required_task_types() -> None:
    tasks = build_cognitive_loop_benchmark_tasks(task_count=25, seed=3)
    categories = {task.category for task in tasks}

    assert len(tasks) == 25
    assert categories == {
        "hidden_rule",
        "hypothesis_conflict",
        "wrong_hypothesis_recovery",
        "noisy_observation",
        "semantic_action",
    }
    assert any(task.noisy_first_probe for task in tasks)
    assert any(task.semantic_task for task in tasks)


def test_cognitive_loop_ablation_reports_all_arms_and_hard_metrics() -> None:
    report = run_cognitive_loop_ablation(task_count=25, seed=5)
    metrics = report["metrics"]
    full = metrics[ARM_FULL]

    assert report["schema_version"] == COGNITIVE_LOOP_ABLATION_VERSION
    assert set(metrics) == set(DEFAULT_ARMS)
    for metric_name in (
        "success_rate",
        "mean_steps_to_success",
        "wrong_commit_rate",
        "false_rejection_rate",
        "hypothesis_flip_accuracy",
        "experiment_usefulness_rate",
        "posterior_calibration_error",
        "recovery_after_wrong_hypothesis",
        "posterior_leading_hypothesis_accuracy",
        "semantic_mismatch_false_experiment_rate",
    ):
        assert metric_name in full
    assert report["passing_lines"]["passed"] is True
    assert report["status"] == "PASSED"


def test_full_arm_beats_required_ablation_controls() -> None:
    report = run_cognitive_loop_ablation(task_count=25, seed=7)
    metrics = report["metrics"]

    assert metrics[ARM_FULL]["success_rate"] >= metrics[ARM_BASELINE_LLM]["success_rate"] + 0.20
    assert metrics[ARM_FULL]["success_rate"] >= metrics[ARM_NO_POSTERIOR]["success_rate"] + 0.10
    assert metrics[ARM_FULL]["success_rate"] >= metrics[ARM_NO_EXPERIMENT]["success_rate"] + 0.10
    assert metrics[ARM_FULL]["wrong_commit_rate"] <= 0.10
    assert metrics[ARM_FULL]["false_rejection_rate"] <= 0.15
    assert metrics[ARM_FULL]["posterior_leading_hypothesis_accuracy"] >= 0.75


def test_semantic_strictness_ablation_exposes_false_experiments() -> None:
    report = run_cognitive_loop_ablation(task_count=25, seed=11)
    metrics = report["metrics"]

    assert metrics[ARM_FULL]["semantic_mismatch_false_experiment_rate"] == 0.0
    assert metrics[ARM_NO_SEMANTIC]["semantic_mismatch_false_experiment_rate"] > 0.0


def test_cognitive_loop_ablation_script_writes_report(tmp_path: Path, capsys) -> None:
    output = tmp_path / "ablation.json"

    assert ablation_main(["--task-count", "25", "--seed", "13", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    rendered = capsys.readouterr().out
    assert payload["status"] == "PASSED"
    assert "Cognitive loop ablation benchmark" in rendered
    assert "Full" in render_cognitive_loop_ablation_report(payload)


def test_collect_baseline_llm_decisions_uses_ollama_client_hook(monkeypatch) -> None:
    tasks = build_cognitive_loop_benchmark_tasks(task_count=20, seed=19)[:2]

    class FakeBaselineClient:
        _base_url = "http://fake-ollama"
        _model = "fake:baseline"

        def complete_json(self, prompt: str, **_kwargs):
            first_task = tasks[0]
            if first_task.task_id in prompt:
                return {"hypothesis_id": first_task.hypotheses[0].hypothesis_id, "confidence": 0.61, "reason": "prior"}
            second_task = tasks[1]
            return {"hypothesis_id": second_task.hypotheses[1].hypothesis_id, "confidence": 0.62, "reason": "prior"}

    monkeypatch.setattr(
        ablation_module,
        "_build_baseline_llm_client",
        lambda **_kwargs: FakeBaselineClient(),
    )

    report = collect_baseline_llm_decisions(tasks, provider="ollama", base_url="http://fake-ollama", model="fake:baseline")

    assert report["schema_version"] == BASELINE_LLM_DECISIONS_VERSION
    assert report["decisions"][tasks[0].task_id] == tasks[0].hypotheses[0].hypothesis_id
    assert report["decisions"][tasks[1].task_id] == tasks[1].hypotheses[1].hypothesis_id
    assert all(row["valid_model_choice"] for row in report["responses"])


def test_wrapped_baseline_decision_file_marks_external_mode(tmp_path: Path) -> None:
    tasks = build_cognitive_loop_benchmark_tasks(task_count=20, seed=23)
    decisions = {task.task_id: task.correct_hypothesis_id for task in tasks}
    decision_path = tmp_path / "baseline_decisions.json"
    decision_path.write_text(
        json.dumps({"schema_version": BASELINE_LLM_DECISIONS_VERSION, "decisions": decisions}),
        encoding="utf-8",
    )

    report = run_cognitive_loop_ablation(
        task_count=20,
        seed=23,
        arms=(ARM_BASELINE_LLM,),
        baseline_decisions_path=decision_path,
    )

    assert report["baseline_llm_mode"] == "external_decisions"
    assert report["baseline_decision_count"] == 20
    assert report["metrics"][ARM_BASELINE_LLM]["success_rate"] == 1.0
