from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


EVAL_METRICS_PANEL_VERSION = "conos.eval_metrics_panel/v1"


@dataclass(frozen=True)
class EvalMetric:
    name: str
    value: Optional[float]
    numerator: int
    denominator: int
    definition: str
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalRunSummary:
    run_id: str
    task_id: str
    source_path: str
    total_reward: float
    success_detected: bool
    verified_success: bool
    verifier_relevant: bool
    verifier_covered: bool
    verification_required: bool
    verification_passed: bool
    verification_failed: bool
    human_intervention_events: int
    recovery_events: int
    recovery_success_detected: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _boolish(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "passed", "pass", "success", "succeeded", "verified"}:
            return True
        if text in {"false", "no", "n", "0", "failed", "fail", "failure", "rejected"}:
            return False
    return None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _iter_dicts(value: Any) -> Iterator[Dict[str, Any]]:
    stack: List[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            yield item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_input_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            yield path
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*.json")):
                if child.is_file():
                    yield child
            for child in sorted(path.rglob("*.jsonl")):
                if child.is_file():
                    yield child


def _payload_from_json_value(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _load_payloads_from_file(path: Path) -> List[Tuple[Dict[str, Any], str]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Tuple[Dict[str, Any], str]] = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for row in _payload_from_json_value(payload):
                rows.append((row, f"{path}:{index}"))
        return rows
    return [(payload, str(path)) for payload in _payload_from_json_value(_load_json(path))]


def _combined_audit_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    root = _as_dict(payload)
    raw_audit = _as_dict(root.get("raw_audit", {}))
    if not raw_audit:
        return dict(root)
    combined = dict(raw_audit)
    for key, value in root.items():
        if key == "raw_audit":
            continue
        combined[key] = value
    return combined


def _first_text(payload: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _total_reward(payload: Mapping[str, Any]) -> float:
    if "total_reward" in payload:
        return _float(payload.get("total_reward", 0.0))
    scorecard = _as_dict(payload.get("arc_scorecard", {}))
    for key in ("total_reward", "reward", "score"):
        if key in scorecard:
            return _float(scorecard.get(key, 0.0))
    return 0.0


def _success_detected(payload: Mapping[str, Any]) -> bool:
    root = _as_dict(payload)
    scorecard = _as_dict(root.get("arc_scorecard", {}))
    final_surface = _as_dict(root.get("final_surface_raw", {}))
    containers = [root, scorecard, final_surface]
    success_keys = (
        "verified_success",
        "task_success",
        "goal_completed",
        "solved",
        "success",
        "passed",
    )
    for container in containers:
        for key in success_keys:
            if key not in container:
                continue
            value = _boolish(container.get(key))
            if value is True:
                return True
    return _total_reward(root) > 0.0


def _verifier_signal_dict(row: Mapping[str, Any]) -> bool:
    keys = {str(key).strip().lower() for key in row.keys()}
    return any("verifier" in key or "verification" in key or key == "verified" for key in keys)


def _verification_required(payload: Mapping[str, Any]) -> bool:
    for row in _iter_dicts(payload):
        if bool(row.get("requires_verification", False)):
            return True
        gate = _as_dict(row.get("verification_gate", {}))
        if bool(gate.get("required", False)):
            return True
        authority = _as_dict(row.get("verifier_authority", {}))
        if bool(authority.get("required", False)):
            return True
        if _verifier_signal_dict(row) and bool(row.get("required", False)):
            return True
    return False


def _verifier_covered(payload: Mapping[str, Any]) -> bool:
    for row in _iter_dicts(payload):
        if _as_dict(row.get("verifier_authority", {})):
            return True
        if _as_dict(row.get("verifier_runtime", {})):
            return True
        if _as_dict(row.get("verification_gate", {})):
            return True
        if str(row.get("verifier_function", "") or "").strip():
            return True
        if _verifier_signal_dict(row) and any(
            key in row
            for key in (
                "verdict",
                "last_verdict",
                "verified",
                "last_verified",
                "verification_passed",
                "verification_ok",
            )
        ):
            return True
    return False


def _verification_result(payload: Mapping[str, Any]) -> Tuple[bool, bool]:
    passed = False
    failed = False
    for row in _iter_dicts(payload):
        if not _verifier_signal_dict(row):
            continue
        for key in ("verdict", "last_verdict", "verification_verdict"):
            if key not in row:
                continue
            text = str(row.get(key, "") or "").strip().lower()
            if text in {"passed", "pass", "success", "succeeded", "verified", "ok"}:
                passed = True
            elif text in {"failed", "fail", "failure", "rejected", "blocked"}:
                failed = True
        for key in ("verified", "last_verified", "verification_passed", "verification_ok"):
            if key not in row:
                continue
            value = _boolish(row.get(key))
            if value is True:
                passed = True
            elif value is False:
                failed = True
    return passed, failed


def _human_intervention_events(payload: Mapping[str, Any]) -> int:
    events = 0
    teacher_log = _as_list(payload.get("teacher_log", []))
    events += len([item for item in teacher_log if isinstance(item, dict)])
    human_tokens = ("human", "manual", "teacher", "user_override", "operator")
    for row in _iter_dicts(payload):
        for key, value in row.items():
            key_text = str(key or "").strip().lower()
            if not any(token in key_text for token in human_tokens):
                continue
            if key_text in {"instruction", "webarena_instruction"}:
                continue
            flag = _boolish(value)
            if flag is True:
                events += 1
            elif isinstance(value, (dict, list)) and value:
                events += 1
        approved_by = str(row.get("approved_by", "") or "").strip().lower()
        if approved_by in {"human", "user", "operator"}:
            events += 1
        approval_sources = [
            str(item or "").strip().lower()
            for item in _as_list(row.get("approval_sources", []))
            if str(item or "").strip()
        ]
        if any(source in {"human", "user", "operator"} for source in approval_sources):
            events += 1
    return events


def _recovery_events(payload: Mapping[str, Any]) -> int:
    recovery_log = [item for item in _as_list(payload.get("recovery_log", [])) if isinstance(item, dict)]
    if recovery_log:
        return len(recovery_log)
    events = 0
    for row in _iter_dicts(payload):
        if any("recovery" in str(key).strip().lower() for key in row.keys()):
            events += 1
    return events


def _recovery_success_detected(payload: Mapping[str, Any], *, run_success: bool) -> bool:
    recovery_log = [item for item in _as_list(payload.get("recovery_log", [])) if isinstance(item, dict)]
    rows = recovery_log if recovery_log else list(_iter_dicts(payload))
    for row in rows:
        if not any("recovery" in str(key).strip().lower() for key in row.keys()) and row not in recovery_log:
            continue
        for key in ("resolved", "success", "succeeded", "recovered", "recovery_success"):
            if key in row and _boolish(row.get(key)) is True:
                return True
    return bool(run_success and _recovery_events(payload) > 0)


def summarize_eval_run(payload: Mapping[str, Any], *, source_path: str = "") -> EvalRunSummary:
    combined = _combined_audit_payload(payload)
    task_id = _first_text(
        combined,
        ("task_id", "webarena_task_id", "game_id", "arc_game_id"),
    )
    run_id = _first_text(combined, ("run_id", "audit_id", "session_id")) or task_id or Path(source_path).stem
    success = _success_detected(combined)
    verification_passed, verification_failed = _verification_result(combined)
    verification_required = _verification_required(combined)
    verifier_covered = _verifier_covered(combined)
    verifier_relevant = bool(verification_required or success or verifier_covered)
    recovery_events = _recovery_events(combined)
    return EvalRunSummary(
        run_id=run_id or "unknown_run",
        task_id=task_id,
        source_path=source_path,
        total_reward=_total_reward(combined),
        success_detected=success,
        verified_success=bool(success and verification_passed),
        verifier_relevant=verifier_relevant,
        verifier_covered=verifier_covered,
        verification_required=verification_required,
        verification_passed=verification_passed,
        verification_failed=verification_failed,
        human_intervention_events=_human_intervention_events(combined),
        recovery_events=recovery_events,
        recovery_success_detected=_recovery_success_detected(combined, run_success=success),
    )


def _ratio_metric(name: str, numerator: int, denominator: int, definition: str) -> EvalMetric:
    if denominator <= 0:
        return EvalMetric(
            name=name,
            value=None,
            numerator=int(numerator),
            denominator=int(denominator),
            definition=definition,
            status="not_applicable",
        )
    return EvalMetric(
        name=name,
        value=float(numerator) / float(denominator),
        numerator=int(numerator),
        denominator=int(denominator),
        definition=definition,
    )


def build_eval_metrics_panel(
    audits: Sequence[Mapping[str, Any]],
    *,
    source_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    paths = list(source_paths or [])
    runs = [
        summarize_eval_run(payload, source_path=paths[index] if index < len(paths) else "")
        for index, payload in enumerate(list(audits or []))
        if isinstance(payload, Mapping)
    ]
    total_runs = len(runs)
    recovery_denominator = sum(1 for run in runs if run.recovery_events > 0)
    verifier_denominator = sum(1 for run in runs if run.verifier_relevant)
    metrics = {
        "verified_success_rate": _ratio_metric(
            "verified_success_rate",
            sum(1 for run in runs if run.verified_success),
            total_runs,
            "Runs with detected success and a passing verifier signal divided by all evaluated runs.",
        ),
        "human_intervention_rate": _ratio_metric(
            "human_intervention_rate",
            sum(1 for run in runs if run.human_intervention_events > 0),
            total_runs,
            "Runs with teacher, manual, user, or human-intervention evidence divided by all evaluated runs.",
        ),
        "recovery_rate": _ratio_metric(
            "recovery_rate",
            sum(1 for run in runs if run.recovery_events > 0 and run.recovery_success_detected),
            recovery_denominator,
            "Recovery-attempted runs that resolved or later succeeded divided by recovery-attempted runs.",
        ),
        "verifier_coverage": _ratio_metric(
            "verifier_coverage",
            sum(1 for run in runs if run.verifier_relevant and run.verifier_covered),
            verifier_denominator,
            "Success, completion, or verification-required runs with verifier authority or verifier-result evidence.",
        ),
    }
    return {
        "schema_version": EVAL_METRICS_PANEL_VERSION,
        "run_count": total_runs,
        "source_files": list(paths),
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
        "runs": [run.to_dict() for run in runs],
    }


def build_eval_metrics_panel_from_paths(paths: Iterable[str | Path]) -> Dict[str, Any]:
    payloads: List[Dict[str, Any]] = []
    source_paths: List[str] = []
    for path in _iter_input_files(paths):
        for payload, source_path in _load_payloads_from_file(path):
            payloads.append(payload)
            source_paths.append(source_path)
    return build_eval_metrics_panel(payloads, source_paths=source_paths)


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.1f}%"


def render_eval_metrics_panel(panel: Mapping[str, Any]) -> str:
    metrics = _as_dict(panel.get("metrics", {}))
    lines = [
        "Cognitive OS evaluation metrics",
        f"schema: {panel.get('schema_version', EVAL_METRICS_PANEL_VERSION)}",
        f"runs: {int(panel.get('run_count', 0) or 0)}",
        "",
        f"{'metric':<28} {'value':>8} {'count':>9}  definition",
        "-" * 92,
    ]
    for name in (
        "verified_success_rate",
        "human_intervention_rate",
        "recovery_rate",
        "verifier_coverage",
    ):
        metric = _as_dict(metrics.get(name, {}))
        value = _format_value(metric.get("value"))
        count = f"{int(metric.get('numerator', 0) or 0)}/{int(metric.get('denominator', 0) or 0)}"
        definition = str(metric.get("definition", "") or "")
        lines.append(f"{name:<28} {value:>8} {count:>9}  {definition}")
    return "\n".join(lines)
