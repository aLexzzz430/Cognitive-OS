from __future__ import annotations

import json
import plistlib
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import conos_cli
from core.app.desktop_client import APP_VERSION, build_app_state, write_dashboard_snapshot
from scripts.build_macos_app import create_macos_app_bundle


def test_desktop_app_state_summarizes_metrics(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "run_id": "app-run",
                "total_reward": 1.0,
                "verified": True,
                "verifier_function": "check_goal",
            }
        ),
        encoding="utf-8",
    )

    state = build_app_state([tmp_path])

    assert state["schema_version"] == APP_VERSION
    assert state["run_count"] == 1
    assert state["metrics"]["verified_success_rate"]["display"] == "100.0%"


def test_conos_app_summary_json_runs_headless(tmp_path: Path, capsys) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"run_id": "headless-app-run", "total_reward": 0}), encoding="utf-8")

    assert conos_cli.main(["app", "--summary-json", str(tmp_path)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == APP_VERSION
    assert payload["run_count"] == 1


def test_write_dashboard_snapshot_creates_html(tmp_path: Path) -> None:
    output = tmp_path / "ui" / "dashboard.html"

    written = write_dashboard_snapshot([], output)

    assert written == output
    assert "Cognitive OS" in output.read_text(encoding="utf-8")


def test_create_macos_app_bundle(tmp_path: Path) -> None:
    app_path = create_macos_app_bundle(REPO_ROOT, tmp_path)

    executable = app_path / "Contents" / "MacOS" / "Cognitive OS"
    plist_path = app_path / "Contents" / "Info.plist"
    assert executable.exists()
    assert executable.stat().st_mode & stat.S_IXUSR
    assert plist_path.exists()
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["CFBundleExecutable"] == "Cognitive OS"
    assert plist["CFBundleIdentifier"] == "ai.cognitive-os.client"
