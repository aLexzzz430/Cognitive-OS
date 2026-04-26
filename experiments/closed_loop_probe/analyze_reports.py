from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping


REPORT_ROOT = Path(__file__).resolve().parent / "reports"
BASE_VARIANTS = ("full", "no_posterior", "no_discriminating_experiment")


def _load_reports() -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(REPORT_ROOT.glob("*.json")):
        if path.stem not in BASE_VARIANTS:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"failed to read {path}: {exc}") from exc
        if not isinstance(payload, dict):
            continue
        reports[path.stem] = payload
    return reports


def _action_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in list(report.get("action_sequence", []) or [])
        if isinstance(row, Mapping)
    ]


def _patch_tick(report: Mapping[str, Any]) -> int | None:
    for row in _action_rows(report):
        if row.get("function_name") == "apply_patch" and bool(row.get("success")):
            return int(row.get("tick", 0) or 0)
    return None


def _final_verification_tick(report: Mapping[str, Any]) -> int | None:
    for row in _action_rows(report):
        kwargs = row.get("kwargs", {}) if isinstance(row.get("kwargs", {}), Mapping) else {}
        if row.get("function_name") == "run_test" and kwargs.get("target") == "." and bool(row.get("success")):
            return int(row.get("tick", 0) or 0)
    return None


def _bool(report: Mapping[str, Any], key: str) -> bool:
    return bool(report.get(key, False))


def _matrix(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for variant in BASE_VARIANTS:
        report = dict(reports.get(variant, {}) or {})
        rows[variant] = {
            "success": _bool(report, "success"),
            "final_tests_passed": _bool(report, "final_tests_passed"),
            "completed_before_verification": _bool(report, "completed_before_verification"),
            "posterior_changed_after_test_failure": _bool(report, "posterior_changed_after_test_failure"),
            "next_action_changed_after_posterior_update": _bool(report, "next_action_changed_after_posterior_update"),
            "patch_tick": _patch_tick(report),
            "final_verification_tick": _final_verification_tick(report),
            "ticks": int(report.get("ticks", 0) or 0),
        }
    return rows


def _full_vs_ablation(matrix: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    full = dict(matrix.get("full", {}) or {})
    full_success = 1 if full.get("success") else 0
    full_patch_tick = full.get("patch_tick")
    full_verify_tick = full.get("final_verification_tick")
    deltas: dict[str, dict[str, Any]] = {}
    for variant in ("no_posterior", "no_discriminating_experiment"):
        row = dict(matrix.get(variant, {}) or {})
        patch_tick = row.get("patch_tick")
        verify_tick = row.get("final_verification_tick")
        deltas[variant] = {
            "success_delta": full_success - (1 if row.get("success") else 0),
            "patch_tick_delta": (
                None
                if full_patch_tick is None or patch_tick is None
                else int(patch_tick) - int(full_patch_tick)
            ),
            "final_verification_tick_delta": (
                None
                if full_verify_tick is None or verify_tick is None
                else int(verify_tick) - int(full_verify_tick)
            ),
        }
    return deltas


def analyze() -> dict[str, Any]:
    reports = _load_reports()
    missing = [variant for variant in BASE_VARIANTS if variant not in reports]
    if missing:
        raise SystemExit(f"missing v1 report(s): {', '.join(missing)}")
    matrix = _matrix(reports)
    patch_ticks = [row["patch_tick"] for row in matrix.values() if row["patch_tick"] is not None]
    final_ticks = [
        row["final_verification_tick"]
        for row in matrix.values()
        if row["final_verification_tick"] is not None
    ]
    return {
        "schema_version": "conos.closed_loop_probe.analysis/v1",
        "report_root": str(REPORT_ROOT),
        "success_failure_matrix": matrix,
        "full_vs_ablation_delta": _full_vs_ablation(matrix),
        "posterior_changed_after_test_failure": {
            variant: bool(row["posterior_changed_after_test_failure"])
            for variant, row in matrix.items()
        },
        "next_action_changed_after_posterior_update": {
            variant: bool(row["next_action_changed_after_posterior_update"])
            for variant, row in matrix.items()
        },
        "completed_before_verification": {
            variant: bool(row["completed_before_verification"])
            for variant, row in matrix.items()
        },
        "patch_tick": {variant: row["patch_tick"] for variant, row in matrix.items()},
        "final_verification_tick": {
            variant: row["final_verification_tick"]
            for variant, row in matrix.items()
        },
        "median_patch_tick_observed": median(patch_ticks) if patch_ticks else None,
        "median_final_verification_tick_observed": median(final_ticks) if final_ticks else None,
    }


def _full_gates_pass(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    matrix = dict(summary.get("success_failure_matrix", {}) or {})
    full = dict(matrix.get("full", {}) or {})
    gates = {
        "success": bool(full.get("success")),
        "final_tests_passed": bool(full.get("final_tests_passed")),
        "completed_before_verification_false": not bool(full.get("completed_before_verification")),
        "posterior_changed_after_test_failure": bool(full.get("posterior_changed_after_test_failure")),
        "next_action_changed_after_posterior_update": bool(full.get("next_action_changed_after_posterior_update")),
    }
    failures = [name for name, passed in gates.items() if not passed]
    return not failures, failures


def main() -> int:
    summary = analyze()
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    ok, failures = _full_gates_pass(summary)
    if not ok:
        print("full report gate failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
