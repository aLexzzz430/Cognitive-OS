from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Protocol

from core.task_discovery.models import (
    ANALYSIS_ACTIONS,
    READ_ONLY_ACTIONS,
    GoalLedger,
    TaskCandidate,
    clamp01,
    string_list,
)


def _compact_text(value: Any, *, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _record_text(record: Mapping[str, Any]) -> str:
    parts = []
    for key in (
        "message",
        "summary",
        "detail",
        "error",
        "reason",
        "failure_mode",
        "event_type",
        "status",
        "observation",
        "content",
        "text",
        "raw",
    ):
        value = record.get(key)
        if value:
            parts.append(str(value))
    if not parts:
        parts.append(str(dict(record)))
    return _compact_text(" ".join(parts), limit=900)


def _source_ref(prefix: str, record: Mapping[str, Any], index: int) -> str:
    for key in ("evidence_id", "event_id", "run_id", "trace_id", "feedback_id", "hypothesis_id", "id"):
        value = str(record.get(key) or "").strip()
        if value:
            return f"{prefix}:{value}"
    return f"{prefix}:record_{index}"


def _candidate(
    *,
    source: str,
    observation: str,
    gap: str,
    proposed_task: str,
    expected_value: float,
    goal_alignment: float,
    evidence_strength: float,
    feasibility: float,
    risk: float,
    cost: float,
    reversibility: float,
    distraction_penalty: float,
    success_condition: str,
    evidence_needed: Iterable[str],
    allowed_actions: Iterable[str],
    forbidden_actions: Iterable[str],
    source_refs: Iterable[str] = (),
    permission_level: str = "L1",
    metadata: Mapping[str, Any] | None = None,
) -> TaskCandidate:
    return TaskCandidate(
        task_id="",
        source=source,
        observation=_compact_text(observation, limit=420),
        gap=_compact_text(gap, limit=420),
        proposed_task=_compact_text(proposed_task, limit=420),
        expected_value=expected_value,
        goal_alignment=goal_alignment,
        evidence_strength=evidence_strength,
        feasibility=feasibility,
        risk=risk,
        cost=cost,
        reversibility=reversibility,
        distraction_penalty=distraction_penalty,
        success_condition=_compact_text(success_condition, limit=320),
        evidence_needed=list(evidence_needed),
        allowed_actions=list(allowed_actions),
        forbidden_actions=list(forbidden_actions),
        source_refs=list(source_refs),
        permission_level=permission_level,
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class DiscoveryContext:
    goal_ledger: GoalLedger
    evidence_records: list[Dict[str, Any]]
    user_feedback_records: list[Dict[str, Any]]
    run_trace_records: list[Dict[str, Any]]
    repo_scan_summary: Dict[str, Any]
    hypothesis_records: list[Dict[str, Any]]
    repo_root: str = "."


class TaskDetector(Protocol):
    name: str

    def detect(self, context: DiscoveryContext) -> list[TaskCandidate]:
        ...


class FailureResidueDetector:
    name = "failure_residue"

    _PATTERNS = {
        "timeout": ("timeout", "timed out", "超时", "model timeout", "wall clock"),
        "malformed_llm_output": ("malformed", "invalid json", "format error", "no valid response", "无法输出有效"),
        "verifier_failed": ("verifier failed", "verification failed", "final_tests_passed false", "tests failed"),
        "crash": ("exception", "traceback", "crash", "segmentation", "panic"),
        "unclosed_loop": ("needs_human_review", "patch_proposal_not_generated", "llm_patch_proposal_unavailable", "diff_too_large"),
    }

    def detect(self, context: DiscoveryContext) -> list[TaskCandidate]:
        grouped: dict[str, list[tuple[str, str]]] = {}
        records = [*context.evidence_records, *context.run_trace_records]
        for index, record in enumerate(records):
            text = _record_text(record)
            lowered = text.lower()
            for mode, patterns in self._PATTERNS.items():
                if any(pattern in lowered for pattern in patterns):
                    grouped.setdefault(mode, []).append((_source_ref("failure", record, index), text))
        candidates: list[TaskCandidate] = []
        for mode, matches in sorted(grouped.items()):
            refs = [ref for ref, _ in matches[:8]]
            examples = "; ".join(text for _, text in matches[:2])
            candidates.append(
                _candidate(
                    source="failure_residue",
                    observation=f"{len(matches)} recent evidence record(s) indicate {mode}: {examples}",
                    gap="A failed or incomplete run left unresolved evidence that has not been converted into a bounded investigation or regression task.",
                    proposed_task=f"Investigate and isolate the recurring {mode} residue; reproduce the smallest case, attribute cause, and propose a verifier-gated follow-up.",
                    expected_value=0.94 if mode in {"timeout", "unclosed_loop", "verifier_failed"} else 0.84,
                    goal_alignment=0.94,
                    evidence_strength=clamp01(0.42 + 0.12 * len(matches)),
                    feasibility=0.86,
                    risk=0.12,
                    cost=0.22,
                    reversibility=0.96,
                    distraction_penalty=0.08,
                    success_condition="A report explains the failure mode with cited evidence and identifies either a bounded fix, a regression eval, or a justified refusal.",
                    evidence_needed=[
                        "recent failed run records",
                        "per-step latency or failure reason where available",
                        "final verifier outcome",
                    ],
                    allowed_actions=ANALYSIS_ACTIONS,
                    forbidden_actions=["modify_core_runtime_without_approval", "sync_back_without_verified_patch"],
                    source_refs=refs,
                    permission_level="L1",
                    metadata={"failure_mode": mode, "match_count": len(matches)},
                )
            )
        return candidates


class GoalGapDetector:
    name = "goal_gap"

    def detect(self, context: DiscoveryContext) -> list[TaskCandidate]:
        candidates: list[TaskCandidate] = []
        for index, goal in enumerate(context.goal_ledger.active_goals):
            target = _float_or_none(_first_present(goal, "target", "target_value"))
            current = _float_or_none(_first_present(goal, "current", "current_value"))
            if target is None or current is None:
                continue
            description = str(goal.get("description") or goal.get("goal") or goal.get("name") or f"goal {index + 1}").strip()
            metric = str(goal.get("metric") or "target_metric").strip()
            lower_is_better = _goal_lower_is_better(goal, description=description, metric=metric)
            if lower_is_better:
                if current <= target:
                    continue
                gap_size = max(0.0, min(1.0, (current - target) / max(abs(current), abs(target), 1.0)))
                direction_text = "above maximum target"
            else:
                if current >= target:
                    continue
                gap_size = max(0.0, min(1.0, target - current if target <= 1.0 else (target - current) / max(target, 1.0)))
                direction_text = "below target"
            candidates.append(
                _candidate(
                    source="goal_gap",
                    observation=f"Active goal '{description}' has {metric}={current:g}, {direction_text} {target:g}.",
                    gap=f"The current measured gap is {gap_size:.2f}, so the system lacks evidence that this stage objective is being met.",
                    proposed_task=f"Identify the highest-evidence bottleneck preventing '{description}' from reaching its target, then propose the smallest measurable intervention.",
                    expected_value=clamp01(0.78 + 0.18 * gap_size),
                    goal_alignment=0.98,
                    evidence_strength=0.78,
                    feasibility=0.78,
                    risk=0.16,
                    cost=0.28,
                    reversibility=0.90,
                    distraction_penalty=0.05,
                    success_condition=f"A follow-up task names a bottleneck, evidence source, and metric movement plan for {metric}.",
                    evidence_needed=["active goal metric", "current benchmark or run summary", "bottleneck evidence"],
                    allowed_actions=ANALYSIS_ACTIONS,
                    forbidden_actions=["change_goal_threshold_without_user_approval"],
                    source_refs=[f"active_goal:{goal.get('goal_id') or goal.get('id') or index}"],
                    permission_level="L1",
                    metadata={"goal": goal, "gap_size": gap_size},
                )
            )
        for index, gap in enumerate(context.goal_ledger.open_gaps):
            text = str(gap.get("description") or gap.get("gap") or gap).strip()
            if not text:
                continue
            candidates.append(
                _candidate(
                    source="goal_gap",
                    observation=f"Goal ledger lists an open gap: {text}",
                    gap=text,
                    proposed_task="Convert the open gap into a measurable investigation task with success criteria and forbidden actions.",
                    expected_value=0.70,
                    goal_alignment=0.90,
                    evidence_strength=0.58,
                    feasibility=0.72,
                    risk=0.20,
                    cost=0.28,
                    reversibility=0.90,
                    distraction_penalty=0.10,
                    success_condition="The gap is converted into a scored, evidence-backed task or explicitly deferred.",
                    evidence_needed=["goal ledger open gap", "current metric or run evidence"],
                    allowed_actions=READ_ONLY_ACTIONS,
                    forbidden_actions=["execute_open_gap_task_without_scoring"],
                    source_refs=[f"open_gap:{gap.get('gap_id') or gap.get('id') or index}"],
                    permission_level="L0",
                    metadata={"open_gap": gap},
                )
            )
        return candidates


class UserSignalDetector:
    name = "user_feedback"

    _SIGNALS = {
        "action_granularity": ("mirror_exec", "原子动作", "空 kwargs", "kwargs", "自由写 shell", "超长命令"),
        "latency_or_timeout": ("太慢", "超时", "timeout", "slow", "卡住", "耗时"),
        "cost": ("太贵", "cost", "成本", "api 调用", "api调用"),
        "loop_failure": ("无法闭环", "不能闭环", "没有闭环", "不稳定", "失败", "无效"),
        "wrong_goal": ("不是我的目标", "不对", "错误", "这不是", "不要"),
    }

    def detect(self, context: DiscoveryContext) -> list[TaskCandidate]:
        candidates: list[TaskCandidate] = []
        for index, record in enumerate(context.user_feedback_records[-40:]):
            text = _record_text(record)
            lowered = text.lower()
            matched = [name for name, patterns in self._SIGNALS.items() if any(pattern.lower() in lowered for pattern in patterns)]
            if not matched:
                continue
            signal = matched[0]
            proposed = _task_for_user_signal(signal)
            candidates.append(
                _candidate(
                    source="user_feedback",
                    observation=f"User feedback indicates {signal}: {text}",
                    gap="User feedback is a high-value sensor and should be converted into a bounded task instead of remaining chat memory.",
                    proposed_task=proposed,
                    expected_value=0.88 if signal != "wrong_goal" else 0.92,
                    goal_alignment=0.90,
                    evidence_strength=0.84,
                    feasibility=0.82,
                    risk=0.14,
                    cost=0.24,
                    reversibility=0.90,
                    distraction_penalty=0.08,
                    success_condition="The feedback-driven task produces a measurable report, regression, or bounded patch proposal that addresses the stated user signal.",
                    evidence_needed=["user feedback excerpt", "related run trace or failing behavior", "current module boundary"],
                    allowed_actions=ANALYSIS_ACTIONS,
                    forbidden_actions=["reinterpret_user_feedback_without_evidence", "modify_core_runtime_without_approval"],
                    source_refs=[_source_ref("feedback", record, index)],
                    permission_level="L1",
                    metadata={"user_signal": signal},
                )
            )
        return candidates


class RepoHealthDetector:
    name = "code_health"

    def detect(self, context: DiscoveryContext) -> list[TaskCandidate]:
        summary = dict(context.repo_scan_summary or {})
        candidates: list[TaskCandidate] = []
        todo_count = int(summary.get("todo_count") or summary.get("todos") or 0)
        fixme_count = int(summary.get("fixme_count") or summary.get("fixmes") or 0)
        large_files = string_list(summary.get("large_files"))
        failing_tests = string_list(summary.get("failing_tests"))
        slow_tests = string_list(summary.get("slow_tests"))
        if failing_tests:
            candidates.append(
                _candidate(
                    source="code_health",
                    observation=f"Repo scan reports failing tests: {', '.join(failing_tests[:5])}",
                    gap="The repository has failing validation signals that can block trustworthy task execution.",
                    proposed_task="Reproduce the failing tests and classify whether they are product blockers, stale tests, or fixture-only failures.",
                    expected_value=0.72,
                    goal_alignment=0.76,
                    evidence_strength=0.70,
                    feasibility=0.70,
                    risk=0.20,
                    cost=0.38,
                    reversibility=0.88,
                    distraction_penalty=0.22,
                    success_condition="Each failing test has a classification, reproduction command, and next action recommendation.",
                    evidence_needed=["repo scan failing test list", "test output"],
                    allowed_actions=ANALYSIS_ACTIONS,
                    forbidden_actions=["delete_or_weaken_tests_without_approval"],
                    source_refs=["repo_scan:failing_tests"],
                    permission_level="L1",
                    metadata={"failing_tests": failing_tests},
                )
            )
        if todo_count or fixme_count or large_files or slow_tests:
            candidates.append(
                _candidate(
                    source="code_health",
                    observation=f"Repo scan reports TODO={todo_count}, FIXME={fixme_count}, large_files={len(large_files)}, slow_tests={len(slow_tests)}.",
                    gap="Code health signals exist, but they may distract from the current North Star unless tied to a measured failure.",
                    proposed_task="Triage code-health signals and promote only items linked to active goals, failures, or verifier reliability.",
                    expected_value=0.46,
                    goal_alignment=0.44,
                    evidence_strength=0.52,
                    feasibility=0.86,
                    risk=0.12,
                    cost=0.24,
                    reversibility=0.92,
                    distraction_penalty=0.76,
                    success_condition="The scan produces a short defer/promote list without editing source code.",
                    evidence_needed=["repo scan summary", "active goal linkage"],
                    allowed_actions=READ_ONLY_ACTIONS,
                    forbidden_actions=["bulk_refactor", "rename_cleanup_without_goal_link"],
                    source_refs=["repo_scan:health"],
                    permission_level="L0",
                    metadata={
                        "todo_count": todo_count,
                        "fixme_count": fixme_count,
                        "large_files": large_files[:8],
                        "slow_tests": slow_tests[:8],
                    },
                )
            )
        return candidates


class HypothesisDebtDetector:
    name = "hypothesis"

    def detect(self, context: DiscoveryContext) -> list[TaskCandidate]:
        candidates: list[TaskCandidate] = []
        for index, hypothesis in enumerate(context.hypothesis_records):
            status = str(hypothesis.get("status") or "unverified").lower()
            evidence_refs = string_list(hypothesis.get("evidence_refs"))
            confidence = clamp01(hypothesis.get("confidence") or hypothesis.get("posterior"), fallback=0.5)
            if status not in {"unverified", "active", "open", "unknown"} and evidence_refs:
                continue
            summary = str(hypothesis.get("summary") or hypothesis.get("claim") or hypothesis.get("hypothesis") or hypothesis).strip()
            candidates.append(
                _candidate(
                    source="hypothesis",
                    observation=f"Unverified hypothesis remains in the registry: {summary}",
                    gap="A system claim exists without enough discriminating evidence, so it should become a testable research task before guiding future work.",
                    proposed_task="Design and run a minimal A/B or discriminating test for the unverified hypothesis, then update the evidence ledger.",
                    expected_value=0.72,
                    goal_alignment=0.84,
                    evidence_strength=0.46 if evidence_refs else 0.28,
                    feasibility=0.66,
                    risk=0.18,
                    cost=0.42,
                    reversibility=0.90,
                    distraction_penalty=0.12,
                    success_condition="The hypothesis is supported, weakened, or retired with cited evidence and a recorded outcome.",
                    evidence_needed=["hypothesis statement", "discriminating test design", "result evidence"],
                    allowed_actions=ANALYSIS_ACTIONS,
                    forbidden_actions=["treat_unverified_hypothesis_as_policy"],
                    source_refs=[_source_ref("hypothesis", hypothesis, index)],
                    permission_level="L1",
                    metadata={"hypothesis": hypothesis, "confidence": confidence},
                )
            )
        return candidates


class OpportunityDetector:
    name = "opportunity"

    def detect(self, context: DiscoveryContext) -> list[TaskCandidate]:
        summary = dict(context.repo_scan_summary or {})
        opportunities = summary.get("opportunities") or summary.get("low_cost_opportunities") or []
        if not isinstance(opportunities, list):
            return []
        candidates: list[TaskCandidate] = []
        for index, item in enumerate(opportunities[:12]):
            text = _compact_text(item if isinstance(item, str) else dict(item), limit=280)
            candidates.append(
                _candidate(
                    source="opportunity",
                    observation=f"Repo scan found a low-cost opportunity: {text}",
                    gap="The opportunity may improve operator visibility or reliability, but it must not pull the system away from active goals.",
                    proposed_task="Validate whether this opportunity is tied to an active goal; if not, defer it.",
                    expected_value=0.38,
                    goal_alignment=0.34,
                    evidence_strength=0.42,
                    feasibility=0.82,
                    risk=0.10,
                    cost=0.22,
                    reversibility=0.90,
                    distraction_penalty=0.72,
                    success_condition="The opportunity is either linked to a measurable goal or deferred.",
                    evidence_needed=["opportunity source", "active goal mapping"],
                    allowed_actions=READ_ONLY_ACTIONS,
                    forbidden_actions=["build_dashboard_without_goal_link", "cosmetic_cleanup_without_goal_link"],
                    source_refs=[f"repo_scan:opportunity:{index}"],
                    permission_level="L0",
                    metadata={"opportunity": item},
                )
            )
        return candidates


class LearningPressureDetector:
    name = "learning_pressure"

    def detect(self, context: DiscoveryContext) -> list[TaskCandidate]:
        records = [*context.evidence_records, *context.run_trace_records]
        stats = _collect_action_learning_stats(records)
        candidates: list[TaskCandidate] = []
        for action_name, row in sorted(stats.items()):
            attempts = int(row.get("attempts", 0) or 0)
            failures = int(row.get("failures", 0) or 0)
            successes = int(row.get("successes", 0) or 0)
            verified_successes = int(row.get("verified_successes", 0) or 0)
            reliability = _safe_ratio(successes, attempts, fallback=float(row.get("reliability", 0.0) or 0.0))
            refs = string_list(row.get("source_refs"))[:8]
            if attempts and failures and (reliability < 0.55 or failures >= successes):
                candidates.append(
                    _candidate(
                        source="self_model",
                        observation=(
                            f"Self-model learning records show action '{action_name}' has "
                            f"attempts={attempts}, failures={failures}, reliability={reliability:.2f}."
                        ),
                        gap="The system has direct evidence that one action family is unreliable, but that pressure has not yet become a bounded capability-improvement task.",
                        proposed_task=(
                            f"Investigate why action '{action_name}' is unreliable; isolate the smallest repeated failure pattern and define a verifier-gated improvement or refusal rule."
                        ),
                        expected_value=0.88,
                        goal_alignment=0.90,
                        evidence_strength=clamp01(0.38 + 0.10 * min(attempts, 5)),
                        feasibility=0.82,
                        risk=0.14,
                        cost=0.26,
                        reversibility=0.94,
                        distraction_penalty=0.05,
                        success_condition="The unreliable action has a cited failure pattern, a bounded regression or guard proposal, and a clear non-action condition.",
                        evidence_needed=["self_summary.capability_estimate", "recent failure records", "outcome_model_update events"],
                        allowed_actions=ANALYSIS_ACTIONS,
                        forbidden_actions=[
                            "modify_core_runtime_without_approval",
                            "promote_failed_skill_without_verifier",
                        ],
                        source_refs=refs,
                        permission_level="L1",
                        metadata={
                            "action_name": action_name,
                            "learning_pressure": "capability_reliability_low",
                            "attempts": attempts,
                            "failures": failures,
                            "successes": successes,
                            "reliability": reliability,
                        },
                    )
                )
            if verified_successes >= 2:
                candidates.append(
                    _candidate(
                        source="skill_learning",
                        observation=(
                            f"Learning records show action '{action_name}' produced "
                            f"{verified_successes} verified success(es)."
                        ),
                        gap="A repeated verified behavior exists but has not been evaluated as a reusable skill candidate.",
                        proposed_task=(
                            f"Compile the verified '{action_name}' behavior into a candidate skill card with applicability, negative examples, and verifier requirements."
                        ),
                        expected_value=0.82,
                        goal_alignment=0.86,
                        evidence_strength=clamp01(0.44 + 0.12 * min(verified_successes, 4)),
                        feasibility=0.84,
                        risk=0.16,
                        cost=0.24,
                        reversibility=0.94,
                        distraction_penalty=0.08,
                        success_condition="A skill candidate is produced only if evidence includes repeated verified success plus explicit failure boundaries.",
                        evidence_needed=["verified outcome evidence", "applicability conditions", "negative examples or refusal boundary"],
                        allowed_actions=[
                            "read_logs",
                            "read_reports",
                            "run_readonly_analysis",
                            "write_report",
                            "propose_skill_candidate",
                        ],
                        forbidden_actions=[
                            "install_skill_without_review",
                            "treat_single_success_as_skill",
                        ],
                        source_refs=refs,
                        permission_level="L1",
                        metadata={
                            "action_name": action_name,
                            "learning_pressure": "skill_candidate_from_verified_success",
                            "verified_successes": verified_successes,
                            "attempts": attempts,
                            "reliability": reliability,
                        },
                    )
                )
        return candidates


def default_detectors() -> list[TaskDetector]:
    return [
        FailureResidueDetector(),
        GoalGapDetector(),
        UserSignalDetector(),
        RepoHealthDetector(),
        HypothesisDebtDetector(),
        OpportunityDetector(),
        LearningPressureDetector(),
    ]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return None


def _goal_lower_is_better(goal: Mapping[str, Any], *, description: str, metric: str) -> bool:
    direction = str(goal.get("direction") or goal.get("objective") or "").strip().lower()
    if direction in {"lower_is_better", "minimize", "decrease", "reduce"}:
        return True
    if direction in {"higher_is_better", "maximize", "increase"}:
        return False
    text = f"{description} {metric}".lower()
    return any(
        token in text
        for token in (
            "降低",
            "减少",
            "lower",
            "reduce",
            "minimize",
            "cost",
            "latency",
            "timeout",
            "error_rate",
            "failure_rate",
            "call_rate",
            "strong_model_call_rate",
        )
    )


def _task_for_user_signal(signal: str) -> str:
    if signal == "action_granularity":
        return "Audit open-task action grounding and convert unsafe broad execution patterns into atomic read/test/propose actions with schema validation."
    if signal == "latency_or_timeout":
        return "Measure per-step latency, token budget, and timeout attribution for recent runs, then recommend a budget or escalation policy change."
    if signal == "cost":
        return "Build a cost attribution report that separates deterministic, small-model, and strong-model work for recent tasks."
    if signal == "loop_failure":
        return "Find the shortest failed loop trace, identify the missing transition, and propose a regression that proves closure before patching."
    if signal == "wrong_goal":
        return "Reconcile the user correction with the active goal ledger and update task scoring constraints before executing more work."
    return "Convert the user feedback into a bounded investigation with evidence requirements and forbidden actions."


def _safe_ratio(numerator: int, denominator: int, *, fallback: float = 0.0) -> float:
    if denominator <= 0:
        return clamp01(fallback)
    return clamp01(numerator / denominator)


def _collect_action_learning_stats(records: Iterable[Mapping[str, Any]]) -> dict[str, Dict[str, Any]]:
    stats: dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        ref = _source_ref("learning", record, index)
        for event in _extract_outcome_model_events(record):
            action = str(event.get("action") or "").strip()
            if not action:
                continue
            _merge_learning_stat(
                stats,
                action_name=action,
                attempts=1,
                successes=1 if str(event.get("outcome") or "").lower() == "success" else 0,
                failures=1 if str(event.get("outcome") or "").lower() == "failure" else 0,
                verified_successes=1 if event.get("verified") is True and str(event.get("outcome") or "").lower() == "success" else 0,
                source_ref=ref,
            )
        for action, capability in _extract_capability_estimates(record).items():
            _merge_learning_stat(
                stats,
                action_name=action,
                attempts=int(capability.get("attempts", 0) or 0),
                successes=int(capability.get("successes", 0) or 0),
                failures=int(capability.get("failures", 0) or 0),
                verified_successes=int(capability.get("verified_successes", 0) or 0),
                reliability=clamp01(capability.get("reliability"), fallback=0.0),
                source_ref=ref,
            )
    return stats


def _extract_outcome_model_events(record: Mapping[str, Any]) -> list[Dict[str, Any]]:
    events: list[Dict[str, Any]] = []
    event_type = str(record.get("event_type") or record.get("kind") or "").strip()
    data = record.get("data") if isinstance(record.get("data"), Mapping) else {}
    if event_type == "outcome_model_update":
        events.append(dict(data or record))

    audit_event = record.get("audit_event")
    if isinstance(audit_event, Mapping) and str(audit_event.get("event_type") or "") == "outcome_model_update":
        audit_data = audit_event.get("data") if isinstance(audit_event.get("data"), Mapping) else {}
        events.append(dict(audit_data or audit_event))

    learning_context = record.get("learning_context")
    if isinstance(learning_context, Mapping):
        for item in learning_context.get("belief_updates", []) or []:
            if isinstance(item, Mapping) and str(item.get("kind") or "") == "outcome_model_update":
                events.append(dict(item))
    for item in record.get("belief_updates", []) or [] if isinstance(record.get("belief_updates"), list) else []:
        if isinstance(item, Mapping) and str(item.get("kind") or "") == "outcome_model_update":
            events.append(dict(item))
    return events


def _extract_capability_estimates(record: Mapping[str, Any]) -> dict[str, Dict[str, Any]]:
    payload: Any = None
    self_summary = record.get("self_summary")
    if isinstance(self_summary, Mapping):
        payload = self_summary.get("capability_estimate")
    if payload is None:
        payload = record.get("capability_estimate")
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(action).strip(): dict(value)
        for action, value in payload.items()
        if str(action).strip() and isinstance(value, Mapping)
    }


def _merge_learning_stat(
    stats: dict[str, Dict[str, Any]],
    *,
    action_name: str,
    attempts: int,
    successes: int,
    failures: int,
    verified_successes: int,
    source_ref: str,
    reliability: float | None = None,
) -> None:
    row = stats.setdefault(
        action_name,
        {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "verified_successes": 0,
            "source_refs": [],
        },
    )
    row["attempts"] = int(row.get("attempts", 0) or 0) + max(0, int(attempts or 0))
    row["successes"] = int(row.get("successes", 0) or 0) + max(0, int(successes or 0))
    row["failures"] = int(row.get("failures", 0) or 0) + max(0, int(failures or 0))
    row["verified_successes"] = int(row.get("verified_successes", 0) or 0) + max(0, int(verified_successes or 0))
    if reliability is not None:
        row["reliability"] = clamp01(reliability)
    refs = string_list(row.get("source_refs"))
    if source_ref and source_ref not in refs:
        refs.append(source_ref)
    row["source_refs"] = refs[-12:]
