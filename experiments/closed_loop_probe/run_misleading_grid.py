from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.closed_loop_probe.run_misleading_localization import (  # noqa: E402
    TRUE_BUG_FILE,
    TRACEBACK_FILE,
    run_misleading_localization,
)


REPORT_ROOT = PROBE_DIR / "reports"
GRID_ROOT = REPORT_ROOT / "misleading_grid"
SUMMARY_PATH = REPORT_ROOT / "misleading_grid_summary.json"

FULL_TICKS = (10, 12, 15, 20)
ABLATION_TICKS = (20,)
REPEATS = 5


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _rate(values: Sequence[bool]) -> float:
    return round(sum(1 for value in values if value) / max(1, len(values)), 6)


def _median_int(values: Sequence[int | None]) -> float | None:
    cleaned = [int(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _run_key(variant: str, ticks: int) -> str:
    return f"{variant}_ticks{ticks}"


def _run_matrix() -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for ticks in FULL_TICKS:
        for repeat in range(1, REPEATS + 1):
            rows.append(("full", int(ticks), repeat))
    for variant in ("no_posterior", "no_discriminating_experiment"):
        for ticks in ABLATION_TICKS:
            for repeat in range(1, REPEATS + 1):
                rows.append((variant, int(ticks), repeat))
    return rows


def _patch_tick(report: Mapping[str, Any]) -> int | None:
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        if row.get("executed_function_name") == "apply_patch" and bool(row.get("success")):
            return int(row.get("tick", 0) or 0)
    return None


def _final_verification_tick(report: Mapping[str, Any]) -> int | None:
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        if row.get("executed_function_name") == "run_test" and kwargs.get("target") == "." and bool(row.get("success")):
            return int(row.get("tick", 0) or 0)
    return None


def _group_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for report in reports:
        key = _run_key(str(report.get("variant") or ""), int(report.get("max_ticks", report.get("requested_max_ticks", 0)) or 0))
        grouped.setdefault(key, []).append(report)
    return grouped


def _summarize(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = _group_reports(reports)

    def by_group(mapper) -> dict[str, Any]:
        return {key: mapper(rows) for key, rows in sorted(grouped.items())}

    return {
        "schema_version": "conos.closed_loop_probe.misleading_grid/v1",
        "grid_root": str(GRID_ROOT),
        "traceback_file": TRACEBACK_FILE,
        "true_bug_file": TRUE_BUG_FILE,
        "run_count": len(reports),
        "success_rate_by_variant_and_tick": by_group(lambda rows: _rate([bool(row.get("success")) for row in rows])),
        "patched_true_bug_file_rate": by_group(lambda rows: _rate([str(row.get("patched_file") or "") == TRUE_BUG_FILE for row in rows])),
        "patched_traceback_file_rate": by_group(lambda rows: _rate([bool(row.get("patched_traceback_file")) for row in rows])),
        "wrong_file_patch_attempt_count": by_group(lambda rows: sum(int(row.get("wrong_file_patch_attempt_count", 0) or 0) for row in rows)),
        "posterior_shift_success_rate": by_group(lambda rows: _rate([bool(row.get("posterior_shift_from_traceback_file_to_true_bug_file")) for row in rows])),
        "median_ticks_to_patch": by_group(lambda rows: _median_int([_patch_tick(row) for row in rows])),
        "median_ticks_to_final_verification": by_group(lambda rows: _median_int([_final_verification_tick(row) for row in rows])),
        "completed_before_verification_count": by_group(lambda rows: sum(1 for row in rows if bool(row.get("completed_before_verification")))),
        "side_effect_after_verified_completion_count": by_group(lambda rows: sum(int(row.get("side_effect_after_verified_completion_count", 0) or 0) for row in rows)),
        "repair_dependency_ratio": by_group(
            lambda rows: round(
                sum(float(row.get("repair_dependency_ratio", 0.0) or 0.0) for row in rows) / max(1, len(rows)),
                6,
            )
        ),
        "runs": [
            {
                "variant": str(row.get("variant") or ""),
                "max_ticks": int(row.get("max_ticks", row.get("requested_max_ticks", 0)) or 0),
                "repeat": int(row.get("repeat", 0) or 0),
                "success": bool(row.get("success")),
                "ticks": int(row.get("ticks", 0) or 0),
                "patched_file": str(row.get("patched_file") or ""),
                "first_file_read_after_failure": str(row.get("first_file_read_after_failure") or ""),
                "posterior_shift": bool(row.get("posterior_shift_from_traceback_file_to_true_bug_file")),
                "patch_tick": _patch_tick(row),
                "final_verification_tick": _final_verification_tick(row),
                "wrong_file_patch_attempt_count": int(row.get("wrong_file_patch_attempt_count", 0) or 0),
                "repair_dependency_ratio": float(row.get("repair_dependency_ratio", 0.0) or 0.0),
                "report_path": str(row.get("report_path") or ""),
            }
            for row in reports
        ],
    }


def run_misleading_grid() -> dict[str, Any]:
    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for variant, ticks, repeat in _run_matrix():
        report_path = GRID_ROOT / f"{variant}_ticks{ticks}_run{repeat}.json"
        report = run_misleading_localization(
            variant=variant,
            max_ticks=ticks,
            report_path=report_path,
        )
        report["requested_max_ticks"] = ticks
        report["max_ticks"] = ticks
        report["repeat"] = repeat
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        reports.append(report)
    summary = _summarize(reports)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def main() -> int:
    summary = run_misleading_grid()
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
