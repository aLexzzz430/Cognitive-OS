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
    "fixture_id": "heldout_ambiguous_two_patches_bug",
    "bug_type": "ambiguous_two_plausible_patches_policy_disambiguation",
    "fixture_root": "fixtures/ambiguous_two_valid_patches_repo",
    "traceback_file": "promo_core/checkout.py",
    "true_bug_file": "promo_core/policy.py",
    "direct_unit_test_file": "tests/test_z_policy.py",
    "discriminating_test_files": ["tests/test_z_policy.py"],
    "decoy_files": ["promo_core/checkout.py"],
    "require_posterior_shift_for_success": False,
    "run_id": "phase1f-heldout-ambiguous-two-patches",
    "instruction": (
        "[closed_loop_probe_variant={variant}] Investigate this Python repository. "
        "The visible checkout failure supports more than one plausible small patch. "
        "Use direct discriminating evidence before choosing a bounded patch; if evidence "
        "is insufficient, refuse and request human review instead of guessing."
    ),
}


def run_heldout_ambiguous_two_patches(
    *,
    variant: str,
    max_ticks: int,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown held-out variant: {variant}")
    output = Path(report_path) if report_path is not None else REPORT_DIR / f"heldout_ambiguous_two_patches_{variant}_ticks{int(max_ticks)}.json"
    return run_fixture_probe(
        HELDOUT_FIXTURE,
        variant=variant,
        max_ticks=int(max_ticks),
        repeat=1,
        report_path=output,
        suite_kind="phase1f_heldout",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase1F held-out ambiguous two-patches probe.")
    parser.add_argument("--variant", default="full", choices=sorted(VALID_VARIANTS))
    parser.add_argument("--max-ticks", type=int, default=30)
    args = parser.parse_args(argv)
    report = run_heldout_ambiguous_two_patches(variant=args.variant, max_ticks=int(args.max_ticks))
    print(json.dumps({
        "variant": report["variant"],
        "success": report["success"],
        "task_success": report.get("task_success"),
        "cognitive_success": report.get("cognitive_success"),
        "needs_human_review": report.get("needs_human_review"),
        "refusal_reason": report.get("refusal_reason"),
        "final_tests_passed": report["final_tests_passed"],
        "patched_file": report["patched_file"],
        "true_bug_file": report["true_bug_file"],
        "decoy_patch_selected": report.get("decoy_patch_selected"),
        "wrong_patch_attempt_count": report.get("wrong_patch_attempt_count"),
        "rollback_count": report.get("rollback_count"),
        "verification_waste_ticks": report.get("verification_waste_ticks"),
        "discriminating_test_selected_before_patch": report.get("discriminating_test_selected_before_patch"),
        "report_path": report["report_path"],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
