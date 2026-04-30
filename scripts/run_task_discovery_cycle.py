from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.task_discovery.creative import CreativeTaskGenerator
from core.task_discovery.engine import TaskDiscoveryEngine


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_load(path: Path, *, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _jsonl_tail(path: Path, *, limit: int = 500) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]:
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"message": text}
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def build_real_inputs(
    *,
    repo_root: Path,
    reports_dir: Path,
    event_log: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    inputs_dir = output_dir / "inputs"
    report_paths = sorted(reports_dir.glob("*.json")) if reports_dir.exists() else []
    report_rows: list[Dict[str, Any]] = []
    report_summaries: list[Dict[str, Any]] = []
    for path in report_paths:
        data = _json_load(path, default={})
        if not isinstance(data, Mapping):
            continue
        recovery = list(data.get("recovery_log") or [])
        commit = list(data.get("commit_log") or [])
        final_known = any(key in data for key in ("success", "final_tests_passed", "terminal_state", "completion_reason"))
        unknown_recovery = sum(
            1
            for item in recovery
            if isinstance(item, Mapping)
            and str((dict(item.get("diagnosis") or {}) if isinstance(item.get("diagnosis"), Mapping) else {}).get("error_type") or "").lower()
            == "unknown"
        )
        failed_recovery = sum(1 for item in recovery if isinstance(item, Mapping) and "failed" in str(item.get("summary") or "").lower())
        summary = {
            "path": str(path),
            "cycles": len(data.get("transfer_trace") or []),
            "commit_log_count": len(commit),
            "recovery_log_count": len(recovery),
            "unknown_recovery_count": unknown_recovery,
            "failed_recovery_count": failed_recovery,
            "has_final_outcome_fields": final_known,
            "total_reward": data.get("total_reward"),
        }
        report_summaries.append(summary)
        if len(commit) == 0 or not final_known or unknown_recovery:
            report_rows.append(
                {
                    "evidence_id": f"open_report_{path.stem}",
                    "source_path": str(path),
                    "event_type": "open_task_report_observed",
                    "status": "needs_human_review" if len(commit) == 0 or not final_known else "observed",
                    "message": (
                        f"real open task report {path.name}: commit_log={len(commit)}, "
                        f"recovery_log={len(recovery)}, unknown_recovery={unknown_recovery}, "
                        f"failed_recovery={failed_recovery}, final_outcome_fields_present={final_known}. "
                        "unclosed_loop evidence: no accepted patch/commit and/or recovery remained unknown."
                    ),
                    "raw_summary": summary,
                }
            )
    event_rows = _jsonl_tail(event_log, limit=400)
    action_counts: Counter[str] = Counter()
    for row in event_rows:
        if row.get("event_type") != "action_executed":
            continue
        data = row.get("data") if isinstance(row.get("data"), Mapping) else {}
        action_counts[str(data.get("function_name") or "unknown")] += 1
    repeated = [name for name, count in action_counts.items() if count >= 3 and name in {"repo_grep", "mirror_plan", "run_test"}]
    if action_counts:
        report_rows.append(
            {
                "evidence_id": "event_log_recent_action_loop",
                "source_path": str(event_log),
                "event_type": "recent_action_loop",
                "status": "needs_human_review" if repeated else "observed",
                "message": (
                    f"recent runtime event log action counts: {dict(action_counts)}; "
                    f"repeated low-progress actions={repeated}; possible no-progress loop signal."
                ),
                "raw_summary": {"action_counts": dict(action_counts), "repeated_actions": repeated, "event_count": len(event_rows)},
            }
        )
    repo_scan = _scan_repo_health(repo_root)
    total_reports = len(report_summaries)
    final_outcome_rate = (
        sum(1 for row in report_summaries if row["has_final_outcome_fields"]) / total_reports if total_reports else 0.0
    )
    accepted_patch_rate = (
        sum(1 for row in report_summaries if int(row["commit_log_count"] or 0) > 0) / total_reports if total_reports else 0.0
    )
    unknown_recovery_rate = (
        sum(1 for row in report_summaries if int(row["unknown_recovery_count"] or 0) > 0) / total_reports if total_reports else 0.0
    )
    active_goals = {
        "north_star": "Con OS 要成为本地优先、证据治理、可持续运行的通用智能系统",
        "active_goals": [
            {
                "goal_id": "open_task_verified_outcome_observability",
                "description": "开放任务报告必须有可验证最终结果字段",
                "metric": "final_outcome_rate",
                "current": round(final_outcome_rate, 4),
                "target": 1.0,
            },
            {
                "goal_id": "open_task_patch_acceptance",
                "description": "真实开放任务应能产生 verifier-gated 最小补丁",
                "metric": "accepted_patch_rate",
                "current": round(accepted_patch_rate, 4),
                "target": 0.7,
            },
            {
                "goal_id": "reduce_unknown_recovery",
                "description": "降低 unknown fallback recovery 占比",
                "metric": "unknown_recovery_rate",
                "current": round(unknown_recovery_rate, 4),
                "target": 0.2,
                "direction": "lower_is_better",
            },
        ],
        "constraints": [
            "不准未经审核改主 repo",
            "不准无限 API 调用",
            "自主任务发现只允许 L0/L1/受限 L2",
        ],
        "evidence": [],
        "open_gaps": [
            {
                "gap_id": "open-task-reports-no-final-outcome",
                "description": "开放任务报告缺少 success/final_tests_passed/terminal_state，无法稳定评估真实开放任务效果",
            }
        ],
    }
    hypotheses = {
        "hypotheses": [
            {
                "hypothesis_id": "h-open-loop-recovery-unknown",
                "summary": "开放任务失败主要卡在 unknown recovery / fallback_review 不能给出可执行下一步",
                "status": "unverified",
            },
            {
                "hypothesis_id": "h-report-schema-gap",
                "summary": "开放任务效果判断困难主要来自 report schema 缺少 terminal/verifier 字段",
                "status": "unverified",
            },
        ]
    }
    user_feedback = _jsonl_tail(output_dir / "user_feedback_log.jsonl", limit=200)
    paths = {
        "active_goals": inputs_dir / "active_goals.json",
        "evidence_ledger": inputs_dir / "evidence_ledger.jsonl",
        "user_feedback_log": inputs_dir / "user_feedback_log.jsonl",
        "repo_scan_summary": inputs_dir / "repo_scan_summary.json",
        "hypothesis_registry": inputs_dir / "hypothesis_registry.json",
        "real_signal_summary": inputs_dir / "real_signal_summary.json",
    }
    _write_json(paths["active_goals"], active_goals)
    _write_jsonl(paths["evidence_ledger"], report_rows)
    _write_jsonl(paths["user_feedback_log"], user_feedback)
    _write_json(paths["repo_scan_summary"], repo_scan)
    _write_json(paths["hypothesis_registry"], hypotheses)
    _write_json(
        paths["real_signal_summary"],
        {
            "generated_at": _utc_now(),
            "open_task_report_count": total_reports,
            "final_outcome_rate": final_outcome_rate,
            "accepted_patch_rate": accepted_patch_rate,
            "unknown_recovery_rate": unknown_recovery_rate,
            "report_summaries": report_summaries,
            "repo_scan": repo_scan,
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _scan_repo_health(repo_root: Path) -> Dict[str, Any]:
    todo_hits: list[str] = []
    large_files: list[str] = []
    for rel in ("core", "integrations", "modules", "tests"):
        base = repo_root / rel
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > 200_000:
                large_files.append(str(path.relative_to(repo_root)))
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if any(token in line for token in ("TODO", "FIXME", "HACK", "XXX")):
                    todo_hits.append(f"{path.relative_to(repo_root)}:{line_no}:{line.strip()[:120]}")
    return {
        "todo_count": sum(1 for hit in todo_hits if "TODO" in hit),
        "fixme_count": sum(1 for hit in todo_hits if "FIXME" in hit),
        "large_files": large_files,
        "slow_tests": [],
        "failing_tests": [],
        "opportunities": ["Open task reports exist but lack one-page outcome/diagnostic dashboard"],
        "sample_hits": todo_hits[:12],
    }


def _build_creative_generator(args: argparse.Namespace) -> CreativeTaskGenerator | None:
    if not args.creative:
        return None
    try:
        if args.creative_provider == "ollama":
            from modules.llm.ollama_client import OllamaClient

            client = OllamaClient(base_url=args.creative_base_url, model=args.creative_model, timeout_sec=args.creative_timeout)
        elif args.creative_provider in {"codex", "codex-cli"}:
            from modules.llm.codex_cli_client import CodexCliClient

            client = CodexCliClient(model=args.creative_model or None, timeout_sec=max(60.0, args.creative_timeout))
        else:
            from modules.llm.openai_client import OpenAIClient

            client = OpenAIClient(base_url=args.creative_base_url, model=args.creative_model or "", timeout_sec=args.creative_timeout)
    except Exception:
        return None
    return CreativeTaskGenerator(client, max_candidates=args.creative_max_candidates, timeout_sec=args.creative_timeout)


def run_cycle(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    inputs = build_real_inputs(
        repo_root=repo_root,
        reports_dir=args.reports_dir,
        event_log=args.event_log,
        output_dir=output_dir,
    )
    engine = TaskDiscoveryEngine(creative_generator=_build_creative_generator(args))
    result = engine.discover_from_paths(
        active_goals_path=Path(inputs["active_goals"]),
        evidence_ledger_path=Path(inputs["evidence_ledger"]),
        user_feedback_log_path=Path(inputs["user_feedback_log"]),
        run_traces_dir=args.run_traces,
        repo_scan_summary_path=Path(inputs["repo_scan_summary"]),
        hypothesis_registry_path=Path(inputs["hypothesis_registry"]),
        repo_root=repo_root,
        max_candidates=args.max_candidates,
        enable_creative=bool(args.creative),
    )
    outputs = engine.write_outputs(result, output_dir / "latest")
    summary = {
        "generated_at": _utc_now(),
        "inputs": inputs,
        "outputs": outputs,
        "report": result.report,
    }
    _write_json(output_dir / "latest_cycle.json", summary)
    _append_jsonl(output_dir / "discovery_cycles.jsonl", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one autonomous task-discovery cycle from real local runtime signals.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=Path("runtime/reports/open_tasks"))
    parser.add_argument("--event-log", type=Path, default=Path("runtime/logs/event_log.jsonl"))
    parser.add_argument("--run-traces", type=Path, default=Path("runtime/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/task_discovery/autonomous"))
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--creative", action="store_true")
    parser.add_argument("--creative-provider", default="ollama", choices=("ollama", "openai", "codex", "codex-cli"))
    parser.add_argument("--creative-base-url", default=None)
    parser.add_argument("--creative-model", default=None)
    parser.add_argument("--creative-timeout", type=float, default=120.0)
    parser.add_argument("--creative-max-candidates", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = run_cycle(args)
    print(json.dumps(summary["report"], ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
