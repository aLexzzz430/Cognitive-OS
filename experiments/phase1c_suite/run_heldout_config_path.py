from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.closed_loop_probe.run_probe_v2_mainloop import VALID_VARIANTS  # noqa: E402
from experiments.phase1c_suite.run_suite import REPORT_DIR, run_fixture_probe  # noqa: E402


HELDOUT_FIXTURE: dict[str, Any] = {
    "fixture_id": "heldout_config_path_resolution_bug",
    "bug_type": "config_path_resolution",
    "fixture_root": "fixtures/config_path_resolution_bug_repo",
    "traceback_file": "app_core/loader.py",
    "true_bug_file": "app_core/paths.py",
    "direct_unit_test_file": "tests/test_paths.py",
    "require_posterior_shift_for_success": False,
    "run_id": "phase1c-heldout-config-path",
    "instruction": (
        "[closed_loop_probe_variant={variant}] Investigate this Python repository. "
        "Configuration loading fails for a nested relative config path. Do not assume the first "
        "traceback source frame is the root cause. Maintain competing hypotheses, choose "
        "discriminating experiments, make the smallest mirror patch, verify the full test suite, "
        "and build a sync plan."
    ),
}


def run_heldout_config_path(
    *,
    variant: str,
    max_ticks: int,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown held-out variant: {variant}")
    output = Path(report_path) if report_path is not None else REPORT_DIR / f"heldout_config_path_{variant}_ticks{int(max_ticks)}.json"
    report = run_fixture_probe(
        HELDOUT_FIXTURE,
        variant=variant,
        max_ticks=int(max_ticks),
        repeat=1,
        report_path=output,
        suite_kind="phase1c_heldout",
    )
    report["heldout"] = True
    report["freeze_rule_compliance"] = {
        "action_grounding_patch_inference_modified": False,
        "posterior_bridge_modified": False,
        "stage2_ranking_modified": False,
        "local_machine_repair_policy_modified": False,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase1C held-out config path probe.")
    parser.add_argument("--variant", default="full", choices=sorted(VALID_VARIANTS))
    parser.add_argument("--max-ticks", type=int, default=25)
    args = parser.parse_args(argv)
    report = run_heldout_config_path(variant=args.variant, max_ticks=int(args.max_ticks))
    print(
        json.dumps(
            {
                "variant": report["variant"],
                "success": report["success"],
                "final_tests_passed": report["final_tests_passed"],
                "ticks": report["ticks"],
                "bug_type": report["bug_type"],
                "traceback_file": report["traceback_file"],
                "true_bug_file": report["true_bug_file"],
                "first_file_read_after_failure": report["first_file_read_after_failure"],
                "patched_file": report["patched_file"],
                "patched_traceback_file": report["patched_traceback_file"],
                "posterior_shift_from_traceback_file_to_true_bug_file": report["posterior_shift_from_traceback_file_to_true_bug_file"],
                "direct_evidence_before_patch": report["direct_evidence_before_patch"],
                "final_diff_summary": report["final_diff_summary"],
                "residual_failure": report.get("residual_failure", ""),
                "report_path": report["report_path"],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
