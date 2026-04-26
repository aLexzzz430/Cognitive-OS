from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.closed_loop_probe.run_misleading_localization import (  # noqa: E402
    enrich_localization_report,
)
from experiments.closed_loop_probe.run_probe_v2_mainloop import (  # noqa: E402
    VALID_VARIANTS,
    run_v2_mainloop,
)


FIXTURE_ROOT = REPO_ROOT / "fixtures" / "semantic_discount_bug_repo"
REPORT_ROOT = PROBE_DIR / "reports" / "semantic_misleading"
TRACEBACK_FILE = "ledger_core/invoice.py"
TRUE_BUG_FILE = "ledger_core/discounts.py"
DIRECT_UNIT_TEST_FILE = "tests/test_discounts.py"
BUG_TYPE = "semantic_boundary_condition_threshold_inclusive"


def run_semantic_misleading(
    *,
    variant: str,
    max_ticks: int,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown semantic misleading variant: {variant}")
    path = Path(report_path) if report_path is not None else REPORT_ROOT / f"{variant}_ticks{max_ticks}.json"
    instruction = (
        f"[closed_loop_probe_variant={variant}] "
        "Investigate this Python repository. Invoice checkout behavior is failing at an exact bulk-discount "
        "threshold boundary. Do not assume the first traceback source frame is the root cause. Maintain "
        "competing hypotheses, choose discriminating experiments, make the smallest mirror patch, verify "
        "the full test suite, and build a sync plan."
    )
    report = run_v2_mainloop(
        variant=variant,
        max_ticks=max_ticks,
        fixture_root=FIXTURE_ROOT,
        instruction=instruction,
        run_id="closed-loop-probe-semantic-misleading",
        report_path=path,
    )
    enriched = enrich_localization_report(
        report,
        schema_version="conos.closed_loop_probe.semantic_misleading/v1",
        traceback_file=TRACEBACK_FILE,
        true_bug_file=TRUE_BUG_FILE,
        bug_type=BUG_TYPE,
        direct_unit_test_file=DIRECT_UNIT_TEST_FILE,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return enriched


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the semantic misleading closed-loop probe.")
    parser.add_argument("--variant", default="full", choices=sorted(VALID_VARIANTS))
    parser.add_argument("--max-ticks", type=int, default=20)
    args = parser.parse_args(argv)
    report = run_semantic_misleading(variant=args.variant, max_ticks=int(args.max_ticks))
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
                "direct_unit_test_file": report["direct_unit_test_file"],
                "direct_unit_test_was_run_before_patch": report["direct_unit_test_was_run_before_patch"],
                "direct_unit_test_was_read_before_patch": report["direct_unit_test_was_read_before_patch"],
                "first_file_read_after_failure": report["first_file_read_after_failure"],
                "patched_file": report["patched_file"],
                "patched_traceback_file": report["patched_traceback_file"],
                "posterior_shift_from_traceback_file_to_true_bug_file": report["posterior_shift_from_traceback_file_to_true_bug_file"],
                "posterior_shift_reason": report["posterior_shift_reason"],
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
