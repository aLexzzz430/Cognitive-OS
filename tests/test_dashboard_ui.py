from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import conos_cli
from core.evaluation.dashboard_app import (
    DASHBOARD_UI_VERSION,
    render_dashboard_html,
    write_dashboard_html,
)
from core.evaluation.metrics_panel import build_eval_metrics_panel


def _panel() -> dict:
    panel = build_eval_metrics_panel(
        [
            {
                "run_id": "ui-run",
                "total_reward": 1.0,
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
        ],
        source_paths=["audit/ui-run.json"],
    )
    panel["dashboard_ui_version"] = DASHBOARD_UI_VERSION
    panel["generated_at"] = "2026-04-23T00:00:00+00:00"
    return panel


def test_dashboard_html_renders_core_product_surfaces() -> None:
    html = render_dashboard_html(_panel())

    assert "Cognitive OS" in html
    assert "Verified Success" in html
    assert "conos run" in html
    assert "conos app" in html
    assert "conos ui" in html
    assert "conos preflight" in html
    assert "ui-run" in html
    assert "audit/ui-run.json" in html


def test_conos_ui_writes_static_dashboard(tmp_path: Path, capsys) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "run_id": "static-ui-run",
                "total_reward": 1.0,
                "verified": True,
                "verifier_function": "check_goal",
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "dashboard.html"

    assert conos_cli.main(["ui", str(tmp_path), "--output", str(output_path)]) == 0

    output = capsys.readouterr().out
    assert "dashboard_html=" in output
    html = output_path.read_text(encoding="utf-8")
    assert "Cognitive OS" in html
    assert "static-ui-run" in html


def test_write_dashboard_html_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "dashboard.html"

    written = write_dashboard_html([], output_path)

    assert written == output_path
    assert output_path.exists()
