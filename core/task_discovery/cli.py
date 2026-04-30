from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from core.task_discovery.creative import CreativeTaskGenerator
from core.task_discovery.engine import TaskDiscoveryEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conos discover-tasks",
        description="Discover, score, and queue autonomous task candidates without executing them.",
    )
    parser.add_argument("--active-goals", type=Path, default=Path("runtime/task_discovery/active_goals.json"))
    parser.add_argument("--evidence-ledger", type=Path, default=Path("runtime/task_discovery/evidence_ledger.jsonl"))
    parser.add_argument("--user-feedback-log", type=Path, default=Path("runtime/task_discovery/user_feedback_log.jsonl"))
    parser.add_argument("--run-traces", type=Path, default=Path("runtime/runs"))
    parser.add_argument("--repo-scan-summary", type=Path, default=Path("runtime/task_discovery/repo_scan_summary.json"))
    parser.add_argument("--hypothesis-registry", type=Path, default=Path("runtime/task_discovery/hypothesis_registry.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/task_discovery"))
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--creative", action="store_true", help="Use an LLM Creating pass to propose extra evidence-grounded candidates.")
    parser.add_argument("--creative-provider", default="ollama", choices=("ollama", "openai", "codex", "codex-cli"))
    parser.add_argument("--creative-base-url", default=None)
    parser.add_argument("--creative-model", default=None)
    parser.add_argument("--creative-timeout", type=float, default=60.0)
    parser.add_argument("--creative-max-candidates", type=int, default=3)
    parser.add_argument("--print-report", action="store_true", help="Print a compact human-readable report after JSON.")
    return parser


def _human_report(report: dict) -> str:
    top = report.get("top_candidate") if isinstance(report.get("top_candidate"), dict) else {}
    lines = [
        "Autonomous Task Discovery Report",
        f"North Star: {report.get('north_star', '')}",
        f"Candidates: {report.get('candidate_count', 0)}",
        f"Queued: {report.get('queued_count', 0)}",
        f"Needs approval: {report.get('needs_approval_count', 0)}",
        f"Deferred: {report.get('deferred_count', 0)}",
    ]
    if top:
        lines.extend(
            [
                f"Top source: {top.get('source', '')}",
                f"Top priority: {top.get('priority', 0)}",
                f"Top gap: {top.get('gap', '')}",
                f"Recommended action: {top.get('proposed_task', '')}",
            ]
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv or []))
    creative_generator = _build_creative_generator(args) if bool(args.creative) else None
    engine = TaskDiscoveryEngine(creative_generator=creative_generator)
    result = engine.discover_from_paths(
        active_goals_path=args.active_goals,
        evidence_ledger_path=args.evidence_ledger,
        user_feedback_log_path=args.user_feedback_log,
        run_traces_dir=args.run_traces,
        repo_scan_summary_path=args.repo_scan_summary,
        hypothesis_registry_path=args.hypothesis_registry,
        repo_root=args.repo_root,
        max_candidates=args.max_candidates,
        enable_creative=bool(args.creative),
    )
    outputs = engine.write_outputs(result, args.output_dir)
    payload = dict(result.report)
    payload["outputs"] = outputs
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.print_report:
        print()
        print(_human_report(result.report))
    return 0


def _build_creative_generator(args: argparse.Namespace) -> CreativeTaskGenerator | None:
    try:
        client = _build_llm_client(args)
    except Exception:
        client = None
    if client is None:
        return None
    return CreativeTaskGenerator(
        client,
        max_candidates=int(args.creative_max_candidates),
        timeout_sec=float(args.creative_timeout),
    )


def _build_llm_client(args: argparse.Namespace) -> Any:
    provider = str(args.creative_provider or "ollama")
    if provider == "ollama":
        from modules.llm.ollama_client import OllamaClient

        return OllamaClient(
            base_url=args.creative_base_url,
            model=args.creative_model,
            timeout_sec=float(args.creative_timeout),
        )
    if provider == "openai":
        from modules.llm.openai_client import OpenAIClient

        return OpenAIClient(
            base_url=args.creative_base_url,
            model=args.creative_model or "",
            timeout_sec=float(args.creative_timeout),
        )
    if provider in {"codex", "codex-cli"}:
        from modules.llm.codex_cli_client import CodexCliClient

        return CodexCliClient(
            model=args.creative_model or None,
            timeout_sec=max(60.0, float(args.creative_timeout)),
        )
    return None


if __name__ == "__main__":
    raise SystemExit(main())
