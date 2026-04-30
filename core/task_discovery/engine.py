from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from core.task_discovery.creative import CreativeTaskGenerator
from core.task_discovery.detectors import DiscoveryContext, TaskDetector, default_detectors
from core.task_discovery.models import GoalLedger, TASK_DISCOVERY_VERSION, TaskCandidate


DEFAULT_CANDIDATE_QUEUE = "task_candidates.jsonl"
DEFAULT_COMMIT_QUEUE = "task_queue.jsonl"
DEFAULT_REPORT = "discovery_report.json"


def _json_default(value: Any) -> str:
    return str(value)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path | None, *, default: Any) -> Any:
    if path is None or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _read_jsonl(path: Path | None, *, max_records: int = 2000) -> list[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: list[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-max_records:]:
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = {"message": text}
        if isinstance(value, Mapping):
            records.append(dict(value))
        else:
            records.append({"value": value})
    return records


def _read_trace_dir(path: Path | None, *, max_files: int = 80) -> list[Dict[str, Any]]:
    if path is None or not path.exists() or not path.is_dir():
        return []
    records: list[Dict[str, Any]] = []
    candidates = sorted(
        [item for item in path.rglob("*.json") if item.is_file()] + [item for item in path.rglob("*.jsonl") if item.is_file()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:max_files]
    for item in candidates:
        if item.suffix == ".jsonl":
            for row in _read_jsonl(item, max_records=80):
                row.setdefault("trace_path", str(item))
                records.append(row)
            continue
        value = _read_json(item, default={})
        if isinstance(value, Mapping):
            row = dict(value)
            row.setdefault("trace_path", str(item))
            records.append(row)
        elif isinstance(value, list):
            for entry in value[:80]:
                if isinstance(entry, Mapping):
                    row = dict(entry)
                    row.setdefault("trace_path", str(item))
                    records.append(row)
    return records


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: list[TaskCandidate]
    task_queue: list[TaskCandidate]
    report: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": TASK_DISCOVERY_VERSION,
            "candidates": [item.to_dict() for item in self.candidates],
            "task_queue": [item.to_dict() for item in self.task_queue],
            "report": dict(self.report),
        }


class TaskDiscoveryEngine:
    """Deterministic discovery layer for autonomous task candidates.

    The engine is intentionally conservative: it discovers, scores, deduplicates,
    and queues tasks, but it never executes them.
    """

    def __init__(
        self,
        detectors: Sequence[TaskDetector] | None = None,
        *,
        queue_threshold: float = 0.65,
        approval_risk_threshold: float = 0.60,
        low_evidence_threshold: float = 0.30,
        distraction_defer_threshold: float = 0.70,
        creative_generator: CreativeTaskGenerator | None = None,
    ) -> None:
        self.detectors = list(detectors or default_detectors())
        self.queue_threshold = float(queue_threshold)
        self.approval_risk_threshold = float(approval_risk_threshold)
        self.low_evidence_threshold = float(low_evidence_threshold)
        self.distraction_defer_threshold = float(distraction_defer_threshold)
        self.creative_generator = creative_generator

    def build_context(
        self,
        *,
        active_goals_path: Path | None = None,
        evidence_ledger_path: Path | None = None,
        user_feedback_log_path: Path | None = None,
        run_traces_dir: Path | None = None,
        repo_scan_summary_path: Path | None = None,
        hypothesis_registry_path: Path | None = None,
        repo_root: Path | str = ".",
    ) -> DiscoveryContext:
        ledger = GoalLedger.from_mapping(_read_json(active_goals_path, default={}))
        evidence_records = [*ledger.evidence, *_read_jsonl(evidence_ledger_path)]
        user_feedback_records = _read_jsonl(user_feedback_log_path)
        run_trace_records = _read_trace_dir(run_traces_dir)
        repo_scan_summary = _read_json(repo_scan_summary_path, default={})
        if not isinstance(repo_scan_summary, Mapping):
            repo_scan_summary = {}
        hypothesis_payload = _read_json(hypothesis_registry_path, default=[])
        hypothesis_records = _coerce_records(hypothesis_payload, preferred_keys=("hypotheses", "items", "records"))
        return DiscoveryContext(
            goal_ledger=ledger,
            evidence_records=evidence_records,
            user_feedback_records=user_feedback_records,
            run_trace_records=run_trace_records,
            repo_scan_summary=dict(repo_scan_summary),
            hypothesis_records=hypothesis_records,
            repo_root=str(repo_root),
        )

    def discover(
        self,
        context: DiscoveryContext,
        *,
        max_candidates: int | None = None,
        enable_creative: bool = False,
    ) -> DiscoveryResult:
        raw: list[TaskCandidate] = []
        for detector in self.detectors:
            raw.extend(detector.detect(context))
        deterministic = [self._prepare_candidate(candidate) for candidate in raw]
        creative_trace: Dict[str, Any] = {"enabled": bool(enable_creative), "accepted_count": 0}
        creative: list[TaskCandidate] = []
        if enable_creative:
            if self.creative_generator is None:
                creative_trace["error"] = "creative_generator_unavailable"
            else:
                seed_candidates = sorted(deterministic, key=lambda item: item.priority, reverse=True)[:8]
                creative = self.creative_generator.generate(context, seed_candidates)
                creative_trace = dict(self.creative_generator.last_trace or creative_trace)
        scored = deterministic + [self._prepare_candidate(candidate) for candidate in creative]
        deduped = self._dedup(scored)
        deduped.sort(key=lambda item: item.priority, reverse=True)
        if max_candidates is not None and max_candidates > 0:
            deduped = deduped[:max_candidates]
        task_queue = [item for item in deduped if item.status in {"queued", "needs_approval"}]
        report = self._build_report(context, deduped, task_queue, creative_trace=creative_trace)
        return DiscoveryResult(candidates=deduped, task_queue=task_queue, report=report)

    def discover_from_paths(
        self,
        *,
        active_goals_path: Path | None = None,
        evidence_ledger_path: Path | None = None,
        user_feedback_log_path: Path | None = None,
        run_traces_dir: Path | None = None,
        repo_scan_summary_path: Path | None = None,
        hypothesis_registry_path: Path | None = None,
        repo_root: Path | str = ".",
        max_candidates: int | None = None,
        enable_creative: bool = False,
    ) -> DiscoveryResult:
        context = self.build_context(
            active_goals_path=active_goals_path,
            evidence_ledger_path=evidence_ledger_path,
            user_feedback_log_path=user_feedback_log_path,
            run_traces_dir=run_traces_dir,
            repo_scan_summary_path=repo_scan_summary_path,
            hypothesis_registry_path=hypothesis_registry_path,
            repo_root=repo_root,
        )
        return self.discover(context, max_candidates=max_candidates, enable_creative=enable_creative)

    def write_outputs(
        self,
        result: DiscoveryResult,
        output_dir: Path,
        *,
        candidates_name: str = DEFAULT_CANDIDATE_QUEUE,
        queue_name: str = DEFAULT_COMMIT_QUEUE,
        report_name: str = DEFAULT_REPORT,
    ) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = output_dir / candidates_name
        queue_path = output_dir / queue_name
        report_path = output_dir / report_name
        _write_jsonl(candidates_path, [candidate.to_dict() for candidate in result.candidates])
        _write_jsonl(queue_path, [candidate.to_dict() for candidate in result.task_queue])
        report_path.write_text(json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "task_candidates": str(candidates_path),
            "task_queue": str(queue_path),
            "report": str(report_path),
        }

    def _prepare_candidate(self, candidate: TaskCandidate) -> TaskCandidate:
        candidate.priority = self.score_candidate(candidate)
        candidate.dedup_key = candidate.dedup_key or _dedup_key(candidate)
        candidate.task_id = candidate.task_id or f"task_{_stable_hash(candidate.dedup_key)[:12]}"
        candidate.created_at = candidate.created_at or _now_iso()
        if candidate.evidence_strength < self.low_evidence_threshold:
            candidate.allowed_actions = _read_only_actions(candidate.allowed_actions)
            candidate.permission_level = "L1"
            if not candidate.proposed_task.lower().startswith("investigate"):
                candidate.proposed_task = f"Investigate evidence for: {candidate.proposed_task}"
            candidate.metadata["low_evidence_limited_to_investigation"] = True
        if candidate.risk >= self.approval_risk_threshold:
            candidate.requires_human_approval = True
            candidate.status = "needs_approval" if candidate.priority >= self.queue_threshold else "candidate"
        if candidate.distraction_penalty > self.distraction_defer_threshold:
            candidate.status = "deferred"
            candidate.metadata["defer_reason"] = "distraction_penalty_above_threshold"
        elif candidate.priority >= self.queue_threshold:
            candidate.status = "needs_approval" if candidate.requires_human_approval else "queued"
        else:
            candidate.status = "candidate"
        return candidate

    @staticmethod
    def score_candidate(candidate: TaskCandidate) -> float:
        priority = (
            0.30 * candidate.expected_value
            + 0.20 * candidate.goal_alignment
            + 0.15 * candidate.evidence_strength
            + 0.15 * candidate.feasibility
            + 0.10 * candidate.reversibility
            - 0.20 * candidate.risk
            - 0.15 * candidate.cost
            - 0.25 * candidate.distraction_penalty
        )
        return max(0.0, min(1.0, round(priority, 4)))

    @staticmethod
    def _dedup(candidates: Iterable[TaskCandidate]) -> list[TaskCandidate]:
        selected: dict[str, TaskCandidate] = {}
        for candidate in candidates:
            key = candidate.dedup_key or _dedup_key(candidate)
            existing = selected.get(key)
            if existing is None or candidate.priority > existing.priority:
                selected[key] = candidate
            elif existing is not None:
                for ref in candidate.source_refs:
                    if ref not in existing.source_refs:
                        existing.source_refs.append(ref)
                existing.metadata["deduped_count"] = int(existing.metadata.get("deduped_count") or 1) + 1
        return list(selected.values())

    def _build_report(
        self,
        context: DiscoveryContext,
        candidates: Sequence[TaskCandidate],
        task_queue: Sequence[TaskCandidate],
        *,
        creative_trace: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        counts_by_source: dict[str, int] = {}
        for item in candidates:
            counts_by_source[item.source] = counts_by_source.get(item.source, 0) + 1
        top_candidate = candidates[0].to_dict() if candidates else None
        top_gap = top_candidate.get("gap") if isinstance(top_candidate, Mapping) else ""
        return {
            "schema_version": TASK_DISCOVERY_VERSION,
            "generated_at": _now_iso(),
            "north_star": context.goal_ledger.north_star,
            "candidate_count": len(candidates),
            "queued_count": len([item for item in task_queue if item.status == "queued"]),
            "needs_approval_count": len([item for item in task_queue if item.status == "needs_approval"]),
            "deferred_count": len([item for item in candidates if item.status == "deferred"]),
            "candidate_count_by_source": counts_by_source,
            "top_gap": top_gap,
            "top_candidate": top_candidate,
            "thresholds": {
                "queue_priority": self.queue_threshold,
                "approval_risk": self.approval_risk_threshold,
                "low_evidence_investigation_only": self.low_evidence_threshold,
                "distraction_defer": self.distraction_defer_threshold,
            },
            "creative_generation": dict(creative_trace or {"enabled": False, "accepted_count": 0}),
            "execution_policy": {
                "autonomous_execution": False,
                "default_permission_levels": ["L0", "L1", "limited_L2"],
                "l3_requires_human_approval": True,
                "note": "Discovery is high-recall; execution is high-precision and gated.",
            },
        }


def _dedup_key(candidate: TaskCandidate) -> str:
    return "|".join(
        [
            candidate.source.strip().lower(),
            candidate.gap.strip().lower(),
            candidate.proposed_task.strip().lower(),
        ]
    )


def _read_only_actions(actions: Sequence[str]) -> list[str]:
    allowed = {"read_logs", "read_reports", "read_files", "run_readonly_analysis", "write_report", "run_eval", "run_tests"}
    result = [action for action in actions if action in allowed]
    return result or ["read_logs", "read_reports", "run_readonly_analysis", "write_report"]


def _coerce_records(payload: Any, *, preferred_keys: Sequence[str] = ("items", "records")) -> list[Dict[str, Any]]:
    if isinstance(payload, Mapping):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=_json_default) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
