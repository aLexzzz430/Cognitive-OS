#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.cognitive_loop_ablation import (  # noqa: E402
    DEFAULT_ARMS,
    build_cognitive_loop_benchmark_tasks,
    collect_baseline_llm_decisions,
    render_cognitive_loop_ablation_report,
    run_cognitive_loop_ablation,
    write_ablation_report,
    write_baseline_llm_decisions,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the controlled Cognitive OS closed-loop ablation benchmark.",
    )
    parser.add_argument("--task-count", type=int, default=25, help="Controlled task count, clamped to 20-50.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--arm", action="append", choices=DEFAULT_ARMS, default=[], help="Run a subset of arms.")
    parser.add_argument(
        "--baseline-decisions",
        default="",
        help="Optional JSON mapping task_id -> hypothesis_id from a naked external LLM baseline.",
    )
    parser.add_argument(
        "--baseline-provider",
        choices=("deterministic", "ollama"),
        default="deterministic",
        help="Collect a naked-model BaselineLLM decision set before running the ablation.",
    )
    parser.add_argument("--baseline-base-url", default="", help="Ollama base URL for --baseline-provider ollama.")
    parser.add_argument("--baseline-model", default="", help="Ollama model for --baseline-provider ollama.")
    parser.add_argument("--baseline-timeout", type=float, default=30.0, help="Per-request baseline LLM timeout.")
    parser.add_argument("--baseline-max-tokens", type=int, default=128, help="Max tokens for each baseline decision.")
    parser.add_argument("--baseline-output", default="", help="Optional JSON output path for collected baseline decisions.")
    parser.add_argument("--output", default="", help="Optional JSON report output path.")
    parser.add_argument("--format", choices=("text", "json", "both"), default="text")
    args = parser.parse_args(list(argv) if argv is not None else None)

    collected_baseline = None
    baseline_decisions = None
    if args.baseline_provider == "ollama" and not args.baseline_decisions:
        tasks = build_cognitive_loop_benchmark_tasks(task_count=args.task_count, seed=args.seed)
        collected_baseline = collect_baseline_llm_decisions(
            tasks,
            provider="ollama",
            base_url=args.baseline_base_url or None,
            model=args.baseline_model or None,
            timeout_sec=args.baseline_timeout,
            max_tokens=args.baseline_max_tokens,
            temperature=0.0,
            fail_on_error=True,
        )
        baseline_decisions = collected_baseline.get("decisions", {})
        if args.baseline_output:
            write_baseline_llm_decisions(collected_baseline, args.baseline_output)

    report = run_cognitive_loop_ablation(
        task_count=args.task_count,
        seed=args.seed,
        arms=tuple(args.arm) if args.arm else DEFAULT_ARMS,
        baseline_decisions_path=args.baseline_decisions or None,
        baseline_decisions=baseline_decisions,
    )
    if collected_baseline is not None:
        report["baseline_llm_collection"] = collected_baseline
    if args.output:
        write_ablation_report(report, args.output)
    if args.format in {"text", "both"}:
        print(render_cognitive_loop_ablation_report(report))
    if args.format in {"json", "both"}:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
