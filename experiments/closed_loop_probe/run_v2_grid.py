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

from experiments.closed_loop_probe.run_probe_v2_mainloop import run_v2_mainloop  # noqa: E402


REPORT_ROOT = PROBE_DIR / "reports"
GRID_ROOT = REPORT_ROOT / "v2_grid"
SUMMARY_PATH = REPORT_ROOT / "v2_grid_summary.json"

FULL_TICKS = (8, 10, 12, 15, 20)
ABLATION_TICKS = (15,)
REPEATS = 5


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _median_int(values: Sequence[int | None]) -> float | None:
    cleaned = [int(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _rate(values: Sequence[bool]) -> float:
    return round(sum(1 for value in values if value) / max(1, len(values)), 6)


def _run_key(variant: str, ticks: int) -> str:
    return f"{variant}_ticks{ticks}"


def _run_matrix() -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for ticks in FULL_TICKS:
        for run_index in range(1, REPEATS + 1):
            rows.append(("full", int(ticks), run_index))
    for variant in ("no_posterior", "no_discriminating_experiment"):
        for ticks in ABLATION_TICKS:
            for run_index in range(1, REPEATS + 1):
                rows.append((variant, int(ticks), run_index))
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


def _posterior_bonus_count(report: Mapping[str, Any]) -> int:
    return sum(
        1
        for row in _as_list(report.get("posterior_action_bonus_by_tick"))
        if float(_as_dict(row).get("posterior_action_bonus", 0.0) or 0.0) > 0.0
    )


def _count_selected_apply_patch(report: Mapping[str, Any]) -> int:
    return sum(
        1
        for row in _as_list(report.get("selected_actions"))
        if _as_dict(row).get("function_name") == "apply_patch"
    )


def _count_bridge_apply_patch(report: Mapping[str, Any]) -> int:
    return sum(
        1
        for row in _as_list(report.get("selected_actions"))
        if _as_dict(row).get("executed_function_name") == "apply_patch"
        and (
            _as_dict(row).get("source") == "local_machine_action_grounding_bridge"
            or float(_as_dict(_as_dict(row).get("candidate_meta")).get("posterior_action_bonus", 0.0) or 0.0) > 0.0
        )
    )


def _terminal_tick(report: Mapping[str, Any]) -> int | None:
    value = report.get("terminal_tick")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _post_completion_non_noop_count(report: Mapping[str, Any]) -> int:
    noop_names = {"no_op_complete", "emit_final_report", "task_done", "wait", ""}
    count = 0
    for row in _as_list(report.get("post_completion_action_sequence")):
        item = _as_dict(row)
        selected = str(item.get("function_name") or "")
        executed = str(item.get("executed_function_name") or selected)
        if selected not in noop_names or executed not in noop_names:
            count += 1
    return count


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
        "schema_version": "conos.closed_loop_probe.v2_grid/v1",
        "grid_root": str(GRID_ROOT),
        "run_count": len(reports),
        "success_rate_by_variant_and_tick": by_group(lambda rows: _rate([bool(row.get("success")) for row in rows])),
        "median_ticks_to_patch": by_group(lambda rows: _median_int([_patch_tick(row) for row in rows])),
        "median_ticks_to_final_verification": by_group(lambda rows: _median_int([_final_verification_tick(row) for row in rows])),
        "patch_selected_by_mainloop_count": by_group(lambda rows: sum(_count_selected_apply_patch(row) for row in rows)),
        "patch_selected_after_posterior_bridge_count": by_group(lambda rows: sum(_count_bridge_apply_patch(row) for row in rows)),
        "posterior_action_bonus_count": by_group(lambda rows: sum(_posterior_bonus_count(row) for row in rows)),
        "empty_kwargs_attempt_count": by_group(lambda rows: sum(int(row.get("empty_kwargs_attempt_count", 0) or 0) for row in rows)),
        "repaired_action_count": by_group(lambda rows: sum(int(row.get("repaired_action_count", 0) or 0) for row in rows)),
        "invalid_action_kwargs_event_count": by_group(lambda rows: sum(len(_as_list(row.get("invalid_action_kwargs_events"))) for row in rows)),
        "stale_apply_patch_attempt_count": by_group(lambda rows: sum(int(row.get("stale_apply_patch_attempt_count", 0) or 0) for row in rows)),
        "stale_apply_patch_repaired_count": by_group(lambda rows: sum(int(row.get("stale_apply_patch_repaired_count", 0) or 0) for row in rows)),
        "terminal_tick_median": by_group(lambda rows: _median_int([_terminal_tick(row) for row in rows])),
        "completed_before_verification_count": by_group(lambda rows: sum(1 for row in rows if bool(row.get("completed_before_verification")))),
        "side_effect_after_verified_completion_count": by_group(lambda rows: sum(int(row.get("side_effect_after_verified_completion_count", 0) or 0) for row in rows)),
        "post_completion_noop_count": by_group(lambda rows: sum(int(row.get("post_completion_noop_count", 0) or 0) for row in rows)),
        "post_completion_non_noop_count": by_group(lambda rows: sum(_post_completion_non_noop_count(row) for row in rows)),
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
                "patch_tick": _patch_tick(row),
                "final_verification_tick": _final_verification_tick(row),
                "terminal_tick": _terminal_tick(row),
                "completed_before_verification": bool(row.get("completed_before_verification")),
                "repaired_action_count": int(row.get("repaired_action_count", 0) or 0),
                "stale_apply_patch_attempt_count": int(row.get("stale_apply_patch_attempt_count", 0) or 0),
                "side_effect_after_verified_completion_count": int(row.get("side_effect_after_verified_completion_count", 0) or 0),
                "post_completion_noop_count": int(row.get("post_completion_noop_count", 0) or 0),
                "post_completion_action_sequence": list(row.get("post_completion_action_sequence", []) or []),
                "repair_dependency_ratio": float(row.get("repair_dependency_ratio", 0.0) or 0.0),
                "report_path": str(row.get("report_path") or ""),
            }
            for row in reports
        ],
    }


def run_v2_grid() -> dict[str, Any]:
    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for variant, ticks, run_index in _run_matrix():
        report_path = GRID_ROOT / f"{variant}_ticks{ticks}_run{run_index}.json"
        report = run_v2_mainloop(
            variant=variant,
            max_ticks=ticks,
            report_path=report_path,
        )
        report["requested_max_ticks"] = ticks
        report["max_ticks"] = ticks
        report["repeat"] = run_index
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        reports.append(report)
    summary = _summarize(reports)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def main() -> int:
    summary = run_v2_grid()
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
