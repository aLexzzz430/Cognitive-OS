from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.closed_loop_probe.run_probe_v2_mainloop import (  # noqa: E402
    VALID_VARIANTS,
    _as_dict,
    _as_list,
    run_v2_mainloop,
)


FIXTURE_ROOT = REPO_ROOT / "fixtures" / "misleading_localization_repo"
REPORT_ROOT = PROBE_DIR / "reports" / "misleading"
TRACEBACK_FILE = "ledger_core/invoice.py"
TRUE_BUG_FILE = "ledger_core/currency.py"


def _patch_target_from_text(patch: str) -> str:
    for line in str(patch or "").splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].strip().split("\t", 1)[0]
        for prefix in ("a/", "b/"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
        return "" if raw in {"/dev/null", "dev/null"} else raw
    return ""


def _action_sequence(selected_actions: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(_as_dict(action).get("function_name") or _as_dict(action).get("executed_function_name") or "")
        for action in selected_actions
    ]


def _first_failed_test_tick(selected_actions: Sequence[Mapping[str, Any]]) -> int | None:
    for action in selected_actions:
        row = _as_dict(action)
        if row.get("executed_function_name") == "run_test" and not bool(row.get("success", True)):
            return int(row.get("tick", 0) or 0)
    return None


def _first_file_read_after_failure(selected_actions: Sequence[Mapping[str, Any]]) -> str:
    failed_tick = _first_failed_test_tick(selected_actions)
    if failed_tick is None:
        return ""
    for action in selected_actions:
        row = _as_dict(action)
        if int(row.get("tick", 0) or 0) <= failed_tick:
            continue
        if row.get("executed_function_name") == "file_read":
            kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
            return str(kwargs.get("path") or "")
    return ""


def _patched_file(report: Mapping[str, Any]) -> str:
    changed = [
        str(path)
        for path in _as_list(_as_dict(report.get("final_diff_summary")).get("changed_paths"))
        if str(path)
    ]
    return changed[0] if changed else ""


def _patch_attempt_paths(selected_actions: Sequence[Mapping[str, Any]]) -> list[str]:
    paths: list[str] = []
    for action in selected_actions:
        row = _as_dict(action)
        if str(row.get("function_name") or row.get("executed_function_name") or "") != "apply_patch":
            continue
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        path = str(kwargs.get("path") or "") or _patch_target_from_text(str(kwargs.get("patch") or ""))
        if path:
            paths.append(path)
    return paths


def _wrong_file_patch_blocked_count(report: Mapping[str, Any], traceback_file: str) -> int:
    count = 0
    for event in _as_list(report.get("invalid_action_kwargs_events")):
        item = _as_dict(event)
        requested = _as_dict(item.get("requested_action"))
        if requested.get("function_name") != "apply_patch":
            continue
        kwargs = _as_dict(requested.get("kwargs"))
        path = str(kwargs.get("path") or "") or _patch_target_from_text(str(kwargs.get("patch") or ""))
        if path == traceback_file:
            count += 1
    return count


def _first_suspected_file(report: Mapping[str, Any], traceback_file: str) -> str:
    for row in _as_list(report.get("posterior_action_bonus_by_tick")):
        item = _as_dict(row)
        target = str(item.get("target_file") or "")
        if target:
            return target
    return _first_file_read_after_failure(_as_list(report.get("selected_actions"))) or traceback_file


def _posterior_shifted(report: Mapping[str, Any], traceback_file: str, true_bug_file: str) -> bool:
    selected_actions = [_as_dict(action) for action in _as_list(report.get("selected_actions"))]
    first_read = _first_file_read_after_failure(selected_actions)
    patched = _patched_file(report)
    if first_read == traceback_file and patched == true_bug_file:
        return True
    seen_traceback = False
    for action in selected_actions:
        row = _as_dict(action)
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        if row.get("executed_function_name") == "file_read" and kwargs.get("path") == traceback_file:
            seen_traceback = True
        if seen_traceback and row.get("executed_function_name") == "file_read" and kwargs.get("path") == true_bug_file:
            return True
    return False


def _first_patch_tick(selected_actions: Sequence[Mapping[str, Any]]) -> int | None:
    for action in selected_actions:
        row = _as_dict(action)
        if row.get("executed_function_name") == "apply_patch" and bool(row.get("success")):
            return int(row.get("tick", 0) or 0)
    return None


def _target_was_run_before_patch(selected_actions: Sequence[Mapping[str, Any]], target: str) -> bool:
    patch_tick = _first_patch_tick(selected_actions)
    if patch_tick is None:
        patch_tick = 10**9
    wanted = str(target or "").split("::", 1)[0]
    if not wanted:
        return False
    for action in selected_actions:
        row = _as_dict(action)
        if int(row.get("tick", 0) or 0) >= patch_tick:
            continue
        if row.get("executed_function_name") != "run_test":
            continue
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        if str(kwargs.get("target") or "").split("::", 1)[0] == wanted:
            return True
    return False


def _file_was_read_before_patch(selected_actions: Sequence[Mapping[str, Any]], path: str) -> bool:
    patch_tick = _first_patch_tick(selected_actions)
    if patch_tick is None:
        patch_tick = 10**9
    wanted = str(path or "")
    if not wanted:
        return False
    for action in selected_actions:
        row = _as_dict(action)
        if int(row.get("tick", 0) or 0) >= patch_tick:
            continue
        if row.get("executed_function_name") != "file_read":
            continue
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        if str(kwargs.get("path") or "") == wanted:
            return True
    return False


def _posterior_shift_reason(
    report: Mapping[str, Any],
    *,
    traceback_file: str,
    true_bug_file: str,
) -> str:
    if not _posterior_shifted(report, traceback_file, true_bug_file):
        return ""
    first_read = _first_file_read_after_failure(_as_list(report.get("selected_actions")))
    patched = _patched_file(report)
    if first_read == traceback_file and patched == true_bug_file:
        return "first post-failure read followed traceback surface, but final patch landed on deeper true bug file"
    return "trace shows posterior bridge moved from traceback surface file to deeper implementation file before patch"


def enrich_localization_report(
    report: Mapping[str, Any],
    *,
    schema_version: str,
    traceback_file: str,
    true_bug_file: str,
    bug_type: str = "",
    direct_unit_test_file: str = "",
    require_posterior_shift_for_success: bool = True,
) -> dict[str, Any]:
    enriched = dict(report)
    selected_actions = [_as_dict(action) for action in _as_list(enriched.get("selected_actions"))]
    patched = _patched_file(enriched)
    patch_attempts = _patch_attempt_paths(selected_actions)
    terminal_state = ""
    for row in reversed(_as_list(enriched.get("terminal_state_by_tick"))):
        state = str(_as_dict(row).get("terminal_state") or "")
        if state:
            terminal_state = state
            break
    shifted = _posterior_shifted(enriched, traceback_file, true_bug_file)
    task_success = bool(
        enriched.get("success")
        and enriched.get("final_tests_passed")
        and patched == true_bug_file
    )
    enriched.update(
        {
            "schema_version": schema_version,
            "action_sequence": _action_sequence(selected_actions),
            "bug_type": bug_type,
            "traceback_file": traceback_file,
            "true_bug_file": true_bug_file,
            "first_suspected_file": _first_suspected_file(enriched, traceback_file),
            "first_file_read_after_failure": _first_file_read_after_failure(selected_actions),
            "direct_unit_test_file": direct_unit_test_file,
            "direct_unit_test_was_run_before_patch": _target_was_run_before_patch(selected_actions, direct_unit_test_file),
            "direct_unit_test_was_read_before_patch": _file_was_read_before_patch(selected_actions, direct_unit_test_file),
            "patched_file": patched,
            "patched_traceback_file": patched == traceback_file,
            "wrong_file_patch_attempt_count": sum(1 for path in patch_attempts if path == traceback_file),
            "wrong_file_patch_blocked_count": _wrong_file_patch_blocked_count(enriched, traceback_file),
            "posterior_shift_from_traceback_file_to_true_bug_file": shifted,
            "posterior_shift_reason": _posterior_shift_reason(
                enriched,
                traceback_file=traceback_file,
                true_bug_file=true_bug_file,
            ),
            "patch_selected_by_mainloop": bool(enriched.get("whether_patch_was_selected_by_mainloop")),
            "patch_selected_after_posterior_bridge": bool(enriched.get("whether_patch_was_selected_after_posterior_bridge")),
            "terminal_state": terminal_state,
            "task_success": task_success,
            "cognitive_success": bool(task_success and (shifted or not require_posterior_shift_for_success)),
            "success": task_success,
        }
    )
    return enriched


def enrich_misleading_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return enrich_localization_report(
        report,
        schema_version="conos.closed_loop_probe.misleading_localization/v1",
        traceback_file=TRACEBACK_FILE,
        true_bug_file=TRUE_BUG_FILE,
    )


def run_misleading_localization(
    *,
    variant: str,
    max_ticks: int,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown misleading localization variant: {variant}")
    path = Path(report_path) if report_path is not None else REPORT_ROOT / f"{variant}_ticks{max_ticks}.json"
    instruction = (
        f"[closed_loop_probe_variant={variant}] "
        "Investigate this Python repository. Invoice total behavior is failing around grouped currency inputs. "
        "Do not assume the first traceback source frame is the root cause. Maintain competing hypotheses, "
        "choose discriminating experiments, make the smallest mirror patch, verify the full test suite, "
        "and build a sync plan."
    )
    report = run_v2_mainloop(
        variant=variant,
        max_ticks=max_ticks,
        fixture_root=FIXTURE_ROOT,
        instruction=instruction,
        run_id="closed-loop-probe-misleading-localization",
        report_path=path,
    )
    enriched = enrich_misleading_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return enriched


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the misleading localization closed-loop probe.")
    parser.add_argument("--variant", default="full", choices=sorted(VALID_VARIANTS))
    parser.add_argument("--max-ticks", type=int, default=20)
    args = parser.parse_args(argv)
    report = run_misleading_localization(variant=args.variant, max_ticks=int(args.max_ticks))
    print(
        json.dumps(
            {
                "variant": report["variant"],
                "success": report["success"],
                "final_tests_passed": report["final_tests_passed"],
                "ticks": report["ticks"],
                "traceback_file": report["traceback_file"],
                "true_bug_file": report["true_bug_file"],
                "first_file_read_after_failure": report["first_file_read_after_failure"],
                "patched_file": report["patched_file"],
                "patched_traceback_file": report["patched_traceback_file"],
                "posterior_shift_from_traceback_file_to_true_bug_file": report["posterior_shift_from_traceback_file_to_true_bug_file"],
                "wrong_file_patch_attempt_count": report["wrong_file_patch_attempt_count"],
                "terminal_state": report["terminal_state"],
                "terminal_tick": report["terminal_tick"],
                "report_path": str(REPORT_ROOT / f"{args.variant}_ticks{int(args.max_ticks)}.json"),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
