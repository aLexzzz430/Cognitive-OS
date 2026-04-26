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
    "fixture_id": "heldout_aggregation_key_bug",
    "bug_type": "aggregation_wrong_key_grouping_logic",
    "fixture_root": "fixtures/aggregation_key_bug_repo",
    "traceback_file": "sales_core/report.py",
    "true_bug_file": "sales_core/grouping.py",
    "direct_unit_test_file": "tests/test_z_grouping.py",
    "require_posterior_shift_for_success": False,
    "run_id": "phase1e-heldout-aggregation-key",
    "instruction": (
        "[closed_loop_probe_variant={variant}] Investigate this Python repository. "
        "Department aggregation fails when rows should be grouped by the requested key. "
        "Do not assume the first surface file is the root cause. Maintain competing "
        "hypotheses, choose discriminating experiments, make the smallest mirror patch, "
        "verify the full test suite, and build a sync plan."
    ),
}


def run_heldout_aggregation_key(
    *,
    variant: str,
    max_ticks: int,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown held-out variant: {variant}")
    output = Path(report_path) if report_path is not None else REPORT_DIR / f"heldout_aggregation_key_{variant}_ticks{int(max_ticks)}.json"
    return run_fixture_probe(
        HELDOUT_FIXTURE,
        variant=variant,
        max_ticks=int(max_ticks),
        repeat=1,
        report_path=output,
        suite_kind="phase1e_heldout",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase1E held-out aggregation key probe.")
    parser.add_argument("--variant", default="full", choices=sorted(VALID_VARIANTS))
    parser.add_argument("--max-ticks", type=int, default=25)
    args = parser.parse_args(argv)
    report = run_heldout_aggregation_key(variant=args.variant, max_ticks=int(args.max_ticks))
    print(json.dumps({
        "variant": report["variant"],
        "success": report["success"],
        "task_success": report.get("task_success"),
        "cognitive_success": report.get("cognitive_success"),
        "final_tests_passed": report["final_tests_passed"],
        "patched_file": report["patched_file"],
        "true_bug_file": report["true_bug_file"],
        "patch_proposal_verified": report.get("patch_proposal_verified"),
        "target_binding_confidence": report.get("target_binding_confidence"),
        "report_path": report["report_path"],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
