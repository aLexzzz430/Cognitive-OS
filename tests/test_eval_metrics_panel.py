from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evaluation.metrics_panel import (
    EVAL_METRICS_PANEL_VERSION,
    build_eval_metrics_panel,
    build_eval_metrics_panel_from_paths,
    render_eval_metrics_panel,
)


def _verified_success_audit() -> dict:
    return {
        "run_id": "run-verified",
        "total_reward": 1.0,
        "recovery_log": [{"recovery_type": "request_replan", "resolved": True}],
        "planner_runtime_log": [
            {
                "verifier_runtime": {
                    "verifier_authority": {
                        "required": True,
                        "verdict": "passed",
                        "verifier_function": "check_goal",
                    }
                }
            }
        ],
    }


def _manual_failed_audit() -> dict:
    return {
        "run_id": "run-manual-failed",
        "total_reward": 0.0,
        "teacher_log": [{"human_intervention": True}],
        "recovery_log": [{"recovery_type": "request_probe", "resolved": False}],
        "governance_log": [
            {
                "completion_gate": {
                    "requires_verification": True,
                    "verifier_authority": {
                        "required": True,
                        "verdict": "failed",
                        "verifier_function": "check_goal",
                    },
                }
            }
        ],
    }


def test_eval_metrics_panel_aligns_document_success_metrics() -> None:
    panel = build_eval_metrics_panel([
        _verified_success_audit(),
        _manual_failed_audit(),
    ])

    assert panel["schema_version"] == EVAL_METRICS_PANEL_VERSION
    assert panel["run_count"] == 2
    assert panel["metrics"]["verified_success_rate"]["numerator"] == 1
    assert panel["metrics"]["verified_success_rate"]["denominator"] == 2
    assert panel["metrics"]["verified_success_rate"]["value"] == 0.5
    assert panel["metrics"]["human_intervention_rate"]["value"] == 0.5
    assert panel["metrics"]["recovery_rate"]["value"] == 0.5
    assert panel["metrics"]["verifier_coverage"]["value"] == 1.0

    rendered = render_eval_metrics_panel(panel)
    assert "verified_success_rate" in rendered
    assert "human_intervention_rate" in rendered
    assert "recovery_rate" in rendered
    assert "verifier_coverage" in rendered


def test_eval_metrics_panel_loads_wrapped_audit_files(tmp_path: Path) -> None:
    audit_path = tmp_path / "run.json"
    audit_path.write_text(
        json.dumps(
            {
                "game_id": "vc33",
                "total_reward": 1.0,
                "raw_audit": _verified_success_audit(),
            }
        ),
        encoding="utf-8",
    )

    panel = build_eval_metrics_panel_from_paths([tmp_path])

    assert panel["run_count"] == 1
    assert panel["source_files"] == [str(audit_path)]
    assert panel["runs"][0]["run_id"] == "run-verified"
    assert panel["metrics"]["verified_success_rate"]["value"] == 1.0


def test_human_intervention_rate_ignores_runtime_intervention_targets() -> None:
    panel = build_eval_metrics_panel(
        [
            {
                "run_id": "autonomous-run",
                "total_reward": 0.0,
                "episode_trace": [
                    {
                        "action": {
                            "_candidate_meta": {
                                "intervention_target": {
                                    "target_kind": "panel",
                                    "anchor_ref": "cell_1",
                                }
                            }
                        }
                    }
                ],
            }
        ]
    )

    assert panel["metrics"]["human_intervention_rate"]["numerator"] == 0
    assert panel["runs"][0]["human_intervention_events"] == 0
