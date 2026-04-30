from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.open_task_benchmark.core import (
    OPEN_TASK_BENCHMARK_SUMMARY_VERSION,
    analyze_report_payloads,
    load_report_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze open-task benchmark result JSON files.")
    parser.add_argument("--input", action="append", default=[], help="Result JSON file or directory. May be repeated.")
    parser.add_argument("--output", default="", help="Optional output summary JSON path.")
    parser.add_argument("--os-agent", default="conos")
    parser.add_argument("--baseline-agent", default="baseline_llm")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        summary = {
            "schema_version": OPEN_TASK_BENCHMARK_SUMMARY_VERSION,
            "dry_run": True,
            "expected_result_schema": {
                "agent_name": "conos|codex|claude|baseline_llm|...",
                "task_id": "task id from task package",
                "commands_run": [],
                "files_read": [],
                "changed_paths": [],
                "tests_modified": False,
                "final_pytest_passed": False,
                "cost": {"total_usd": 0.0},
                "budget_summary": {},
                "llm_route_usage": [],
                "llm_failure_policy_decisions": [],
                "failure_recovery_events": [],
                "unknown_cost_reason": "",
                "no_llm_reason": "",
                "final_diff_summary": "",
                "raw_transcript_path": "",
            },
        }
    else:
        reports = load_report_files([Path(item) for item in args.input])
        summary = analyze_report_payloads(reports, os_agent=args.os_agent, baseline_agent=args.baseline_agent)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
