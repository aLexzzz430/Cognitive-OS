from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping


PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))

from run_probe import run_probe  # noqa: E402


VARIANTS = ("full", "no_posterior", "no_discriminating_experiment")
TICK_LIMITS = (8, 10, 12, 15)
REPORT_ROOT = PROBE_DIR / "reports"
GRID_ROOT = REPORT_ROOT / "grid"
SUMMARY_PATH = REPORT_ROOT / "grid_summary.json"


def _action_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in list(report.get("action_sequence", []) or [])
        if isinstance(row, Mapping)
    ]


def _first_tick(report: Mapping[str, Any], function_name: str, *, target: str | None = None) -> int | None:
    for row in _action_rows(report):
        if row.get("function_name") != function_name or not bool(row.get("success")):
            continue
        kwargs = row.get("kwargs", {}) if isinstance(row.get("kwargs", {}), Mapping) else {}
        if target is not None and kwargs.get("target") != target:
            continue
        return int(row.get("tick", 0) or 0)
    return None


def _success_rate(reports: list[Mapping[str, Any]]) -> float:
    if not reports:
        return 0.0
    return round(sum(1 for row in reports if bool(row.get("success"))) / len(reports), 6)


def _median_tick(reports: list[Mapping[str, Any]], function_name: str, *, target: str | None = None) -> float | None:
    ticks = [
        tick
        for tick in (_first_tick(row, function_name, target=target) for row in reports)
        if tick is not None
    ]
    return float(median(ticks)) if ticks else None


def _patch_target_paths(reports: list[Mapping[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for report in reports:
        if not bool(report.get("success")):
            continue
        summary = report.get("final_diff_summary", {}) if isinstance(report.get("final_diff_summary", {}), Mapping) else {}
        for path in list(summary.get("changed_paths", []) or []):
            if str(path or ""):
                paths.add(str(path))
    return paths


def _wrong_file_reads(report: Mapping[str, Any], useful_paths: set[str]) -> list[str]:
    rows: list[str] = []
    for action in _action_rows(report):
        if action.get("function_name") != "file_read":
            continue
        kwargs = action.get("kwargs", {}) if isinstance(action.get("kwargs", {}), Mapping) else {}
        path = str(kwargs.get("path") or "")
        if path and not path.startswith("tests/") and useful_paths and path not in useful_paths:
            rows.append(path)
    return rows


def run_grid() -> dict[str, Any]:
    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    reports_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    runs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for ticks in TICK_LIMITS:
            output_path = GRID_ROOT / f"{variant}_ticks{ticks}.json"
            report = run_probe(variant=variant, max_ticks=ticks, report_path=output_path)
            report["_grid_max_ticks"] = ticks
            reports_by_variant[variant].append(report)
            runs.append(
                {
                    "variant": variant,
                    "max_ticks": ticks,
                    "report_path": str(output_path),
                    "success": bool(report.get("success")),
                    "final_tests_passed": bool(report.get("final_tests_passed")),
                    "patch_tick": _first_tick(report, "apply_patch"),
                    "final_verification_tick": _first_tick(report, "run_test", target="."),
                    "posterior_action_switch": bool(report.get("next_action_changed_after_posterior_update")),
                }
            )
    useful_paths = _patch_target_paths([row for rows in reports_by_variant.values() for row in rows])
    wrong_file_reads: dict[str, dict[str, list[str]]] = {}
    for variant, reports in reports_by_variant.items():
        wrong_file_reads[variant] = {}
        for report in reports:
            key = f"ticks{int(report.get('_grid_max_ticks', report.get('ticks', 0)) or 0)}"
            wrong_file_reads[variant][key] = _wrong_file_reads(report, useful_paths)
    summary = {
        "schema_version": "conos.closed_loop_probe.grid/v1",
        "variants": list(VARIANTS),
        "tick_limits": list(TICK_LIMITS),
        "runs": runs,
        "success_rate_by_variant": {
            variant: _success_rate(reports)
            for variant, reports in reports_by_variant.items()
        },
        "median_ticks_to_patch": {
            variant: _median_tick(reports, "apply_patch")
            for variant, reports in reports_by_variant.items()
        },
        "median_ticks_to_final_verification": {
            variant: _median_tick(reports, "run_test", target=".")
            for variant, reports in reports_by_variant.items()
        },
        "wrong_file_reads": wrong_file_reads,
        "posterior_action_switch_count": {
            variant: sum(1 for report in reports if bool(report.get("next_action_changed_after_posterior_update")))
            for variant, reports in reports_by_variant.items()
        },
        "useful_patch_paths": sorted(useful_paths),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def main() -> int:
    summary = run_grid()
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
