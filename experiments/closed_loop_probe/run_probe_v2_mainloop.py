from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.local_machine.runner import run_local_machine_task  # noqa: E402


FIXTURE_ROOT = REPO_ROOT / "fixtures" / "closed_loop_bug_repo"
REPORT_ROOT = PROBE_DIR / "reports"
REPORT_PATH = REPORT_ROOT / "v2_mainloop_full.json"
VALID_VARIANTS = {"full", "no_posterior", "no_discriminating_experiment"}


def _copy_fixture_to_temp(fixture_root: str | Path | None = None) -> tuple[Path, Path, tempfile.TemporaryDirectory[str]]:
    temp = tempfile.TemporaryDirectory(prefix="conos_closed_loop_probe_v2_")
    root = Path(temp.name)
    source_root = root / "source"
    mirror_root = root / "mirror"
    source_fixture = Path(fixture_root) if fixture_root is not None else FIXTURE_ROOT
    shutil.copytree(source_fixture, source_root, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return source_root, mirror_root, temp


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _tool_kwargs(action: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(action.get("payload"))
    tool_args = _as_dict(payload.get("tool_args"))
    kwargs = tool_args.get("kwargs", action.get("kwargs", {}))
    return dict(kwargs) if isinstance(kwargs, Mapping) else {}


def _tool_name(action: Mapping[str, Any]) -> str:
    snapshot = _as_dict(action.get("action_snapshot"))
    if snapshot.get("function_name"):
        return str(snapshot.get("function_name") or "")
    payload = _as_dict(action.get("payload"))
    tool_args = _as_dict(payload.get("tool_args"))
    for value in (
        tool_args.get("function_name"),
        payload.get("function_name"),
        action.get("function_name"),
        action.get("action"),
    ):
        if str(value or "").strip():
            return str(value).strip()
    return "wait" if action.get("kind") == "wait" else ""


def _trace_action(row: Mapping[str, Any]) -> dict[str, Any]:
    action = _as_dict(row.get("action"))
    snapshot = _as_dict(row.get("action_snapshot"))
    function_name = str(snapshot.get("function_name") or _tool_name(action) or "")
    kwargs = _as_dict(snapshot.get("kwargs")) or _tool_kwargs(action)
    outcome = _as_dict(row.get("outcome"))
    grounding = _as_dict(outcome.get("local_machine_action_grounding"))
    repaired_action = _as_dict(grounding.get("repaired_action"))
    executed_function_name = str(outcome.get("function_name") or repaired_action.get("function_name") or function_name)
    executed_kwargs = _as_dict(repaired_action.get("kwargs")) or kwargs
    meta = _as_dict(action.get("_candidate_meta"))
    return {
        "tick": int(row.get("tick", 0) or 0),
        "function_name": function_name,
        "kwargs": kwargs,
        "executed_function_name": executed_function_name,
        "executed_kwargs": executed_kwargs,
        "success": bool(outcome.get("success", True)),
        "state": str(outcome.get("state", "") or ""),
        "terminal_state": str(outcome.get("terminal_state") or _as_dict(outcome.get("local_machine_investigation_phase")).get("terminal_state") or ""),
        "completion_reason": str(outcome.get("completion_reason") or _as_dict(outcome.get("local_machine_investigation_phase")).get("completion_reason") or ""),
        "terminal_tick": _as_dict(outcome.get("local_machine_investigation_phase")).get("terminal_tick", outcome.get("terminal_tick")),
        "verified_completion": bool(outcome.get("verified_completion", _as_dict(outcome.get("local_machine_investigation_phase")).get("verified_completion", False))),
        "source": str(action.get("_source") or ""),
        "candidate_meta": meta,
        "action_grounding_status": str(outcome.get("action_grounding_status") or ""),
        "source_hypothesis_id": str(outcome.get("source_hypothesis_id") or ""),
        "leading_hypothesis_before_patch": _as_dict(outcome.get("leading_hypothesis_before_patch")),
        "hypothesis_target_file": str(outcome.get("hypothesis_target_file") or ""),
    }


def _local_mirror_from_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    observation = _as_dict(row.get("observation"))
    mirror = _as_dict(observation.get("local_mirror"))
    if not mirror:
        raw = _as_dict(observation.get("raw"))
        mirror = _as_dict(raw.get("local_mirror"))
    return mirror


def _hypotheses_from_tick(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    mirror = _local_mirror_from_trace(row)
    investigation = _as_dict(mirror.get("investigation"))
    hypotheses = [
        dict(item)
        for item in _as_list(investigation.get("hypotheses"))
        if isinstance(item, Mapping)
    ]
    if hypotheses:
        return hypotheses
    return [
        dict(item)
        for item in _as_list(row.get("competing_hypotheses"))
        if isinstance(item, Mapping)
    ]


def _ranked_experiments_from_tick(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    ranked = [
        dict(item)
        for item in _as_list(row.get("ranked_discriminating_experiments"))
        if isinstance(item, Mapping)
    ]
    if ranked:
        return ranked
    mirror = _local_mirror_from_trace(row)
    investigation = _as_dict(mirror.get("investigation"))
    return [
        dict(item)
        for item in _as_list(investigation.get("discriminating_tests"))
        if isinstance(item, Mapping)
    ]


def _posterior_summary_from_tick(row: Mapping[str, Any]) -> dict[str, Any]:
    mirror = _local_mirror_from_trace(row)
    investigation = _as_dict(mirror.get("investigation"))
    lifecycle = _as_dict(investigation.get("hypothesis_lifecycle"))
    if int(lifecycle.get("hypothesis_count", 0) or 0) > 0:
        return lifecycle
    return {}


def _posterior_events_from_tick(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    mirror = _local_mirror_from_trace(row)
    investigation = _as_dict(mirror.get("investigation"))
    known_ids = {
        str(item.get("hypothesis_id") or "")
        for item in _as_list(investigation.get("hypotheses"))
        if isinstance(item, Mapping) and str(item.get("hypothesis_id") or "")
    }
    events: list[dict[str, Any]] = []
    for item in _as_list(investigation.get("hypothesis_events")):
        if isinstance(item, Mapping):
            payload = dict(item)
            if str(payload.get("hypothesis_id") or "") in known_ids and str(payload.get("event_type") or "").startswith("evidence_"):
                events.append(payload)
    if not events and known_ids:
        for item in _as_list(row.get("hypothesis_posterior_events")):
            if (
                isinstance(item, Mapping)
                and str(item.get("hypothesis_id") or "") in known_ids
                and str(item.get("event_type") or "").startswith("evidence_")
            ):
                events.append(dict(item))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        key = json.dumps(event, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _run_test_target(action: Mapping[str, Any]) -> str:
    return str((_as_dict(action.get("executed_kwargs")) or _as_dict(action.get("kwargs"))).get("target") or "")


def _final_tests_passed(selected_actions: Sequence[Mapping[str, Any]]) -> bool:
    for action in selected_actions:
        if (
            (action.get("executed_function_name") or action.get("function_name")) == "run_test"
            and _run_test_target(action) == "."
            and bool(action.get("success"))
        ):
            return True
    return False


def _completed_before_verification(selected_actions: Sequence[Mapping[str, Any]]) -> bool:
    final_verify_tick = None
    plan_tick = None
    for action in selected_actions:
        tick = int(action.get("tick", 0) or 0)
        executed_name = action.get("executed_function_name") or action.get("function_name")
        if executed_name == "run_test" and _run_test_target(action) == "." and bool(action.get("success")):
            final_verify_tick = tick if final_verify_tick is None else min(final_verify_tick, tick)
        if executed_name == "mirror_plan" and bool(action.get("success")):
            plan_tick = tick if plan_tick is None else min(plan_tick, tick)
    return bool(plan_tick is not None and (final_verify_tick is None or plan_tick < final_verify_tick))


def _whether_posterior_changed_action(
    selected_actions: Sequence[Mapping[str, Any]],
    events_by_tick: Sequence[Mapping[str, Any]],
) -> bool:
    actions_by_tick = {int(row.get("tick", 0) or 0): dict(row) for row in selected_actions}
    for row in events_by_tick:
        tick = int(row.get("tick", 0) or 0)
        if not _as_list(row.get("events")):
            continue
        current_action = actions_by_tick.get(tick, {})
        next_action = actions_by_tick.get(tick + 1, {})
        if not current_action or not next_action:
            continue
        current_name = str(current_action.get("function_name") or "")
        next_name = str(next_action.get("function_name") or "")
        if current_name and next_name and current_name != next_name:
            return True
    return False


def _grounding_event_from_tick(row: Mapping[str, Any]) -> dict[str, Any]:
    outcome = _as_dict(row.get("outcome"))
    return _as_dict(outcome.get("local_machine_action_grounding"))


def _phase_event_from_tick(row: Mapping[str, Any]) -> dict[str, Any]:
    outcome = _as_dict(row.get("outcome"))
    phase = _as_dict(outcome.get("local_machine_investigation_phase"))
    if phase:
        return phase
    mirror = _local_mirror_from_trace(row)
    return {"phase_after": str(mirror.get("investigation_phase") or "")}


def _terminal_state_from_tick(row: Mapping[str, Any]) -> dict[str, Any]:
    tick = int(row.get("tick", 0) or 0)
    phase = _phase_event_from_tick(row)
    mirror = _local_mirror_from_trace(row)
    investigation = _as_dict(mirror.get("investigation"))
    return {
        "tick": tick,
        "terminal_state": str(phase.get("terminal_state") or mirror.get("terminal_state") or investigation.get("terminal_state") or ""),
        "completion_reason": str(phase.get("completion_reason") or mirror.get("completion_reason") or investigation.get("completion_reason") or ""),
        "terminal_tick": phase.get("terminal_tick", mirror.get("terminal_tick", investigation.get("terminal_tick"))),
        "verified_completion": bool(
            phase.get("verified_completion", mirror.get("verified_completion", investigation.get("verified_completion", False)))
        ),
    }


def _target_binding_from_tick(row: Mapping[str, Any]) -> dict[str, Any]:
    outcome = _as_dict(row.get("outcome"))
    phase = _as_dict(outcome.get("local_machine_investigation_phase"))
    if _as_dict(phase.get("target_binding")):
        return {"tick": int(row.get("tick", 0) or 0), **_as_dict(phase.get("target_binding"))}
    mirror = _local_mirror_from_trace(row)
    investigation = _as_dict(mirror.get("investigation"))
    binding = _as_dict(investigation.get("target_binding"))
    return {"tick": int(row.get("tick", 0) or 0), **binding}


def _patch_proposal_from_tick(row: Mapping[str, Any]) -> dict[str, Any]:
    outcome = _as_dict(row.get("outcome"))
    if not (
        any(str(key).startswith("patch_proposal") for key in outcome.keys())
        or bool(outcome.get("needs_human_review", False))
    ):
        return {}
    return {
        "tick": int(row.get("tick", 0) or 0),
        "state": str(outcome.get("state") or ""),
        "success": bool(outcome.get("success", False)),
        "patch_proposals_generated": int(outcome.get("patch_proposals_generated", 0) or 0),
        "patch_proposal_selected": _as_dict(outcome.get("patch_proposal_selected")),
        "patch_proposal_source": str(outcome.get("patch_proposal_source") or ""),
        "patch_proposal_rationale": str(outcome.get("patch_proposal_rationale") or ""),
        "patch_proposal_llm_trace": _as_list(outcome.get("patch_proposal_llm_trace")),
        "patch_proposal_applied": bool(outcome.get("patch_proposal_applied", False)),
        "patch_proposal_verified": bool(outcome.get("patch_proposal_verified", False)),
        "patch_proposal_rollback_count": int(outcome.get("patch_proposal_rollback_count", 0) or 0),
        "rejected_patch_proposals": _as_list(outcome.get("rejected_patch_proposals")),
        "proposal_test_results": _as_list(outcome.get("proposal_test_results")),
        "needs_human_review": bool(outcome.get("needs_human_review", False)),
        "refusal_reason": str(outcome.get("refusal_reason") or ""),
        "target_binding": _as_dict(outcome.get("target_binding")),
        "source_hypothesis_id": str(outcome.get("source_hypothesis_id") or ""),
        "leading_hypothesis_before_patch": _as_dict(outcome.get("leading_hypothesis_before_patch")),
        "hypothesis_target_file": str(outcome.get("hypothesis_target_file") or ""),
    }


def _posterior_action_bonus_from_tick(row: Mapping[str, Any]) -> dict[str, Any]:
    action = _as_dict(row.get("action"))
    meta = _as_dict(action.get("_candidate_meta"))
    outcome = _as_dict(row.get("outcome"))
    grounding = _as_dict(outcome.get("local_machine_action_grounding"))
    repaired = _as_dict(grounding.get("repaired_action"))
    repaired_kwargs = _as_dict(repaired.get("kwargs"))
    target_file = str(meta.get("target_file") or repaired_kwargs.get("path") or "")
    return {
        "tick": int(row.get("tick", 0) or 0),
        "posterior_action_bonus": float(meta.get("posterior_action_bonus", 0.0) or 0.0),
        "posterior_action_reason": str(meta.get("posterior_action_reason") or ""),
        "leading_hypothesis_id": str(meta.get("leading_hypothesis_id") or ""),
        "target_file": target_file,
        "source": str(action.get("_source") or ""),
    }


def _verify_after_patch_bonus_from_tick(row: Mapping[str, Any]) -> dict[str, Any]:
    action = _as_dict(row.get("action"))
    meta = _as_dict(action.get("_candidate_meta"))
    return {
        "tick": int(row.get("tick", 0) or 0),
        "verify_after_patch_bonus": float(meta.get("verify_after_patch_bonus", 0.0) or 0.0),
        "patch_fingerprint": str(meta.get("patch_fingerprint") or ""),
        "verification_pending": bool(meta.get("verification_pending", False)),
        "source": str(action.get("_source") or ""),
    }


def _stale_patch_penalty_from_tick(row: Mapping[str, Any]) -> dict[str, Any]:
    action = _as_dict(row.get("action"))
    meta = _as_dict(action.get("_candidate_meta"))
    grounding = _grounding_event_from_tick(row)
    return {
        "tick": int(row.get("tick", 0) or 0),
        "stale_patch_penalty": float(meta.get("stale_patch_penalty", grounding.get("stale_patch_penalty", 0.0)) or 0.0),
        "patch_fingerprint": str(meta.get("patch_fingerprint") or grounding.get("patch_fingerprint") or ""),
        "verification_pending": bool(meta.get("verification_pending", grounding.get("verification_pending", False))),
        "source": str(action.get("_source") or ""),
    }


def _repair_dependency_ratio(selected_actions: Sequence[Mapping[str, Any]], repaired_count: int) -> float:
    executable_count = sum(
        1
        for action in selected_actions
        if str(action.get("function_name") or "") not in {"", "wait"}
    )
    return round(float(repaired_count) / max(1, executable_count), 6)


def _terminal_tick_from_states(
    terminal_state_by_tick: Sequence[Mapping[str, Any]],
    *,
    terminal_states: set[str] | None = None,
) -> int | None:
    accepted = terminal_states or {"completed_verified"}
    for row in terminal_state_by_tick:
        if str(row.get("terminal_state") or "") in accepted:
            return int(row.get("tick", 0) or 0)
    return None


def _completion_reason_from_states(
    terminal_state_by_tick: Sequence[Mapping[str, Any]],
    *,
    terminal_states: set[str] | None = None,
) -> str:
    accepted = terminal_states or {"completed_verified"}
    for row in terminal_state_by_tick:
        if str(row.get("terminal_state") or "") in accepted:
            return str(row.get("completion_reason") or "")
    return ""


def _post_completion_action_sequence(
    selected_actions: Sequence[Mapping[str, Any]],
    terminal_tick: int | None,
) -> list[dict[str, Any]]:
    if terminal_tick is None:
        return []
    return [
        {
            "tick": int(action.get("tick", 0) or 0),
            "function_name": str(action.get("function_name") or ""),
            "executed_function_name": str(action.get("executed_function_name") or ""),
            "state": str(action.get("state") or ""),
            "success": bool(action.get("success", False)),
        }
        for action in selected_actions
        if int(action.get("tick", 0) or 0) > terminal_tick
    ]


def _post_completion_noop_count(post_completion_actions: Sequence[Mapping[str, Any]]) -> int:
    noop_names = {"no_op_complete", "emit_final_report", "task_done", "wait", ""}
    return sum(
        1
        for action in post_completion_actions
        if str(action.get("function_name") or "") in noop_names
        and str(action.get("executed_function_name") or action.get("function_name") or "") in noop_names
    )


def _final_diff_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    final_raw = _as_dict(audit.get("final_surface_raw"))
    mirror = _as_dict(final_raw.get("local_mirror"))
    diff_summary = _as_dict(mirror.get("diff_summary"))
    diff_entries = [
        dict(item)
        for item in _as_list(mirror.get("diff_entries"))
        if isinstance(item, Mapping)
    ]
    changed_paths = [
        str(item.get("relative_path") or "")
        for item in diff_entries
        if str(item.get("status") or "") in {"added", "modified", "removed_in_mirror"}
    ]
    return {
        "diff_summary": diff_summary,
        "changed_paths": [path for path in changed_paths if path],
        "diff_entries": diff_entries,
    }


def _compact_llm_calls(calls: Sequence[Any], *, limit: int = 5, max_chars: int = 900) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for raw in list(calls or [])[: max(0, int(limit))]:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        compacted.append(
            {
                "tick": int(row.get("tick", 0) or 0),
                "function_name": str(row.get("function_name") or row.get("selected_function") or ""),
                "capability": str(row.get("capability") or ""),
                "route_name": str(row.get("route_name") or ""),
                "prompt_excerpt": str(row.get("prompt") or "")[:max_chars],
                "response_excerpt": str(row.get("response") or "")[:max_chars],
                "parsed_kwargs": _as_dict(row.get("parsed_kwargs")),
                "thinking_policy": _as_dict(row.get("thinking_policy")),
                "request_kwargs": _as_dict(row.get("request_kwargs")),
                "error": str(row.get("error") or ""),
                "cache_hit": bool(row.get("cache_hit", False)),
            }
        )
    return compacted


def run_v2_mainloop(
    *,
    variant: str,
    max_ticks: int,
    llm_provider: str = "none",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_timeout: float = 60.0,
    deterministic_fallback_enabled: bool = True,
    prefer_llm_kwargs: bool = False,
    prefer_llm_patch_proposals: bool = False,
    llm_thinking_mode: str = "auto",
    require_llm_call: bool = False,
    report_path: str | Path | None = None,
    fixture_root: str | Path | None = None,
    instruction: str | None = None,
    run_id: str = "closed-loop-probe-v2-mainloop-full",
) -> dict[str, Any]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown v2 mainloop probe variant: {variant}")
    source_fixture = Path(fixture_root) if fixture_root is not None else FIXTURE_ROOT
    source_root, mirror_root, temp = _copy_fixture_to_temp(source_fixture)
    try:
        task_instruction = instruction or (
            f"[closed_loop_probe_variant={variant}] "
            "Investigate this Python repository. Locate the failing behavior, "
            "maintain competing hypotheses, choose discriminating experiments, "
            "make the smallest mirror patch, verify the tests, and build a sync plan."
        )
        audit = run_local_machine_task(
            instruction=task_instruction,
            source_root=str(source_root),
            mirror_root=str(mirror_root),
            run_id=run_id,
            max_episodes=1,
            max_ticks_per_episode=max_ticks,
            reset_mirror=True,
            terminal_after_plan=False,
            allow_empty_exec=True,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_timeout=float(llm_timeout),
            llm_mode="integrated",
            deterministic_fallback_enabled=bool(deterministic_fallback_enabled),
            prefer_llm_kwargs=bool(prefer_llm_kwargs),
            prefer_llm_patch_proposals=bool(prefer_llm_patch_proposals),
            llm_thinking_mode=llm_thinking_mode,
        )
        trace = [
            dict(row)
            for row in _as_list(audit.get("episode_trace"))
            if isinstance(row, Mapping)
        ]
        selected_actions = [_trace_action(row) for row in trace]
        competing_by_tick = [
            {"tick": int(row.get("tick", 0) or 0), "hypotheses": _hypotheses_from_tick(row)}
            for row in trace
        ]
        ranked_by_tick = [
            {"tick": int(row.get("tick", 0) or 0), "experiments": _ranked_experiments_from_tick(row)}
            for row in trace
        ]
        posterior_by_tick = [
            {"tick": int(row.get("tick", 0) or 0), "posterior_summary": _posterior_summary_from_tick(row)}
            for row in trace
        ]
        events_by_tick = [
            {"tick": int(row.get("tick", 0) or 0), "events": _posterior_events_from_tick(row)}
            for row in trace
        ]
        grounding_events = [_grounding_event_from_tick(row) for row in trace]
        invalid_action_kwargs_events = [
            event for event in grounding_events if event.get("event_type") == "invalid_action_kwargs"
        ]
        repaired_actions = [
            event for event in grounding_events if event.get("event_type") == "local_machine_action_kwargs_repaired"
        ]
        investigation_phase_by_tick = [
            {"tick": int(row.get("tick", 0) or 0), **_phase_event_from_tick(row)}
            for row in trace
        ]
        terminal_state_by_tick = [
            _terminal_state_from_tick(row)
            for row in trace
        ]
        target_binding_by_tick = [
            _target_binding_from_tick(row)
            for row in trace
        ]
        patch_proposal_events = [
            event
            for event in (_patch_proposal_from_tick(row) for row in trace)
            if event
        ]
        patch_proposal_llm_trace_count = sum(len(_as_list(event.get("patch_proposal_llm_trace"))) for event in patch_proposal_events)
        posterior_action_bonus_by_tick = [
            _posterior_action_bonus_from_tick(row)
            for row in trace
        ]
        verify_after_patch_bonus_by_tick = [
            _verify_after_patch_bonus_from_tick(row)
            for row in trace
        ]
        stale_patch_penalty_by_tick = [
            _stale_patch_penalty_from_tick(row)
            for row in trace
        ]
        verification_pending_by_tick = [
            {
                "tick": int(row.get("tick", 0) or 0),
                "verification_pending": bool(
                    _verify_after_patch_bonus_from_tick(row).get("verification_pending")
                    or _stale_patch_penalty_from_tick(row).get("verification_pending")
                    or _phase_event_from_tick(row).get("phase_before") == "verify"
                    or _phase_event_from_tick(row).get("phase_after") == "verify"
                ),
                "phase_before": str(_phase_event_from_tick(row).get("phase_before") or ""),
                "phase_after": str(_phase_event_from_tick(row).get("phase_after") or ""),
            }
            for row in trace
        ]
        completed_before_verification = _completed_before_verification(selected_actions)
        diff_summary = _final_diff_summary(audit)
        whether_changed_action = _whether_posterior_changed_action(selected_actions, events_by_tick)
        whether_patch_was_selected_by_mainloop = any(
            action.get("function_name") == "apply_patch"
            for action in selected_actions
        )
        whether_patch_was_selected_after_posterior_bridge = any(
            (action.get("executed_function_name") or action.get("function_name")) == "apply_patch"
            and (
                str(action.get("source") or "") == "local_machine_action_grounding_bridge"
                or float(_as_dict(action.get("candidate_meta")).get("posterior_action_bonus", 0.0) or 0.0) > 0.0
            )
            for action in selected_actions
        )
        whether_patch_proposal_selected = any(
            action.get("function_name") == "propose_patch"
            for action in selected_actions
        )
        patch_proposals_generated = sum(int(event.get("patch_proposals_generated", 0) or 0) for event in patch_proposal_events)
        verified_patch_events = [event for event in patch_proposal_events if bool(event.get("patch_proposal_verified"))]
        final_tests_passed = bool(_final_tests_passed(selected_actions) or verified_patch_events)
        selected_patch_event = patch_proposal_events[-1] if patch_proposal_events else {}
        needs_human_review_events = [event for event in patch_proposal_events if bool(event.get("needs_human_review"))]
        refusal_reason = next(
            (str(event.get("refusal_reason") or "") for event in needs_human_review_events if str(event.get("refusal_reason") or "")),
            "",
        )
        patch_proposal_selected_via_target_binding = bool(
            whether_patch_proposal_selected
            and any(
                str(_as_dict(event.get("patch_proposal_selected")).get("target_file") or "")
                == str(_as_dict(event.get("target_binding")).get("top_target_file") or "")
                and float(_as_dict(event.get("target_binding")).get("target_confidence", 0.0) or 0.0) >= 0.55
                for event in patch_proposal_events
            )
        )
        rejected_patch_proposals = [
            item
            for event in patch_proposal_events
            for item in _as_list(event.get("rejected_patch_proposals"))
        ]
        empty_kwargs_attempt_count = sum(
            1
            for action in selected_actions
            if action.get("function_name") not in {"", "wait"} and not _as_dict(action.get("kwargs"))
        )
        stale_apply_patch_attempt_count = 0
        patch_seen = False
        for action in selected_actions:
            selected_name = str(action.get("function_name") or "")
            executed_name = str(action.get("executed_function_name") or "")
            if executed_name == "apply_patch" and bool(action.get("success")):
                patch_seen = True
                continue
            if patch_seen and selected_name == "apply_patch":
                stale_apply_patch_attempt_count += 1
        stale_apply_patch_repaired_count = sum(
            1
            for event in repaired_actions
            if bool(event.get("stale_apply_patch_repaired", False))
            or (
                _as_dict(event.get("original_action")).get("function_name") == "apply_patch"
                and _as_dict(event.get("repaired_action")).get("function_name") == "run_test"
            )
        )
        side_effect_after_verified_completion_events = [
            event for event in grounding_events if event.get("event_type") == "side_effect_after_verified_completion"
        ]
        terminal_tick = _terminal_tick_from_states(terminal_state_by_tick, terminal_states={"completed_verified", "needs_human_review"})
        verified_terminal_tick = _terminal_tick_from_states(terminal_state_by_tick)
        completion_reason = _completion_reason_from_states(terminal_state_by_tick, terminal_states={"completed_verified", "needs_human_review"})
        post_completion_action_sequence = _post_completion_action_sequence(selected_actions, terminal_tick)
        post_completion_noop_count = _post_completion_noop_count(post_completion_action_sequence)
        repair_dependency_ratio = _repair_dependency_ratio(selected_actions, len(repaired_actions))
        success = bool(
            final_tests_passed
            and diff_summary.get("changed_paths")
            and not completed_before_verification
            and whether_changed_action
        )
        llm_tool_trace = _as_dict(audit.get("local_machine_llm_tool_trace"))
        llm_budget = _as_dict(audit.get("llm_budget"))
        local_llm_budget = _as_dict(_as_dict(_as_dict(audit.get("final_surface_raw")).get("local_mirror")).get("llm_budget"))
        llm_calls = _as_list(llm_tool_trace.get("llm_calls"))
        llm_call_trace_count = int(llm_tool_trace.get("llm_call_count", len(llm_calls)) or 0)
        llm_enabled = str(llm_provider or "none").strip().lower() not in {"", "none", "off", "disabled"}
        report = {
            "schema_version": "conos.closed_loop_probe.v2_mainloop/v1",
            "variant": variant,
            "runner": "CoreMainLoop via integrations.local_machine.runner.run_local_machine_task",
            "manual_action_selection": False,
            "source_fixture": str(source_fixture),
            "ephemeral_source_root": str(source_root),
            "ephemeral_mirror_root": str(mirror_root),
            "success": success,
            "ticks": len(selected_actions),
            "selected_actions": selected_actions,
            "competing_hypotheses_by_tick": competing_by_tick,
            "ranked_discriminating_experiments_by_tick": ranked_by_tick,
            "posterior_summary_by_tick": posterior_by_tick,
            "hypothesis_posterior_events_by_tick": events_by_tick,
            "invalid_action_kwargs_events": invalid_action_kwargs_events,
            "repaired_action_count": len(repaired_actions),
            "repaired_actions": repaired_actions,
            "empty_kwargs_attempt_count": empty_kwargs_attempt_count,
            "investigation_phase_by_tick": investigation_phase_by_tick,
            "terminal_state_by_tick": terminal_state_by_tick,
            "terminal_tick": terminal_tick,
            "completion_reason": completion_reason,
            "verified_completion": bool(verified_terminal_tick is not None),
            "target_binding_by_tick": target_binding_by_tick,
            "posterior_action_bonus_by_tick": posterior_action_bonus_by_tick,
            "patch_proposal_events": patch_proposal_events,
            "patch_proposals_generated": patch_proposals_generated,
            "patch_proposal_llm_trace_count": patch_proposal_llm_trace_count,
            "patch_proposal_selected": whether_patch_proposal_selected,
            "patch_proposal_source": str(selected_patch_event.get("patch_proposal_source") or ""),
            "patch_proposal_rationale": str(selected_patch_event.get("patch_proposal_rationale") or ""),
            "patch_proposal_applied": any(bool(event.get("patch_proposal_applied")) for event in patch_proposal_events),
            "patch_proposal_verified": bool(verified_patch_events),
            "patch_proposal_selected_via_target_binding": patch_proposal_selected_via_target_binding,
            "patch_proposal_rollback_count": sum(int(event.get("patch_proposal_rollback_count", 0) or 0) for event in patch_proposal_events),
            "rejected_patch_proposals": rejected_patch_proposals,
            "needs_human_review": bool(needs_human_review_events),
            "refusal_reason": refusal_reason,
            "stale_apply_patch_attempt_count": stale_apply_patch_attempt_count,
            "stale_apply_patch_repaired_count": stale_apply_patch_repaired_count,
            "side_effect_after_verified_completion_count": len(side_effect_after_verified_completion_events),
            "side_effect_after_verified_completion_events": side_effect_after_verified_completion_events,
            "post_completion_action_sequence": post_completion_action_sequence,
            "post_completion_noop_count": post_completion_noop_count,
            "verify_after_patch_bonus_by_tick": verify_after_patch_bonus_by_tick,
            "stale_patch_penalty_by_tick": stale_patch_penalty_by_tick,
            "repair_dependency_ratio": repair_dependency_ratio,
            "verification_pending_by_tick": verification_pending_by_tick,
            "whether_patch_was_selected_by_mainloop": whether_patch_was_selected_by_mainloop,
            "whether_patch_was_selected_after_posterior_bridge": whether_patch_was_selected_after_posterior_bridge,
            "final_diff_summary": diff_summary,
            "final_tests_passed": final_tests_passed,
            "completed_before_verification": completed_before_verification,
            "whether_posterior_changed_action": whether_changed_action,
            "llm_provider": str(audit.get("llm_provider") or llm_provider or "none"),
            "llm_base_url": str(audit.get("llm_base_url") or llm_base_url or ""),
            "llm_model": str(audit.get("llm_model") or llm_model or ""),
            "llm_mode": str(audit.get("llm_mode") or "integrated"),
            "llm_enabled": llm_enabled,
            "llm_call_trace_count": llm_call_trace_count,
            "llm_budget": llm_budget,
            "llm_budget_policy": local_llm_budget,
            "tool_call_trace_count": int(llm_tool_trace.get("tool_call_count", 0) or 0),
            "llm_call_required": bool(require_llm_call),
            "llm_call_required_passed": (not bool(require_llm_call)) or llm_call_trace_count > 0,
            "llm_call_trace_excerpt": _compact_llm_calls(llm_calls),
            "deterministic_fallback_enabled": bool(deterministic_fallback_enabled),
            "prefer_llm_kwargs": bool(prefer_llm_kwargs),
            "prefer_llm_patch_proposals": bool(prefer_llm_patch_proposals),
            "llm_thinking_mode": str(llm_thinking_mode or "auto"),
            "audit_excerpt": {
                "total_reward": audit.get("total_reward"),
                "final_surface_terminal": audit.get("final_surface_terminal"),
                "llm_provider": audit.get("llm_provider"),
                "llm_base_url": audit.get("llm_base_url"),
                "llm_model": audit.get("llm_model"),
                "llm_mode": audit.get("llm_mode"),
                "llm_call_trace_count": llm_call_trace_count,
                "llm_budget_total_calls": int(llm_budget.get("total_calls", 0) or 0),
                "llm_budget_output_tokens": int(llm_budget.get("output_tokens", 0) or 0),
                "tool_call_trace_count": int(llm_tool_trace.get("tool_call_count", 0) or 0),
                "deterministic_fallback_enabled": bool(deterministic_fallback_enabled),
                "prefer_llm_kwargs": bool(prefer_llm_kwargs),
                "prefer_llm_patch_proposals": bool(prefer_llm_patch_proposals),
                "llm_thinking_mode": str(llm_thinking_mode or "auto"),
            },
        }
    finally:
        temp.cleanup()
    output_path = Path(report_path) if report_path is not None else REPORT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Closed Loop Probe v2 through the existing CoreMainLoop.")
    parser.add_argument("--variant", default="full", choices=sorted(VALID_VARIANTS))
    parser.add_argument("--max-ticks", type=int, default=15)
    parser.add_argument("--llm-provider", default="none", choices=["none", "ollama", "openai", "minimax", "codex", "codex-cli"])
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--disable-deterministic-fallback", action="store_true")
    parser.add_argument("--prefer-llm-kwargs", action="store_true")
    parser.add_argument("--prefer-llm-patch-proposals", action="store_true")
    parser.add_argument("--llm-thinking-mode", default="auto", choices=["auto", "off", "on"])
    parser.add_argument("--require-llm-call", action="store_true")
    args = parser.parse_args(argv)
    report = run_v2_mainloop(
        variant=args.variant,
        max_ticks=int(args.max_ticks),
        llm_provider=args.llm_provider,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_timeout=float(args.llm_timeout),
        deterministic_fallback_enabled=not bool(args.disable_deterministic_fallback),
        prefer_llm_kwargs=bool(args.prefer_llm_kwargs),
        prefer_llm_patch_proposals=bool(args.prefer_llm_patch_proposals),
        llm_thinking_mode=args.llm_thinking_mode,
        require_llm_call=bool(args.require_llm_call),
    )
    print(
        json.dumps(
            {
                "variant": report["variant"],
                "success": report["success"],
                "ticks": report["ticks"],
                "final_tests_passed": report["final_tests_passed"],
                "completed_before_verification": report["completed_before_verification"],
                "whether_posterior_changed_action": report["whether_posterior_changed_action"],
                "repaired_action_count": report["repaired_action_count"],
                "invalid_action_kwargs_events": len(report["invalid_action_kwargs_events"]),
                "stale_apply_patch_attempt_count": report["stale_apply_patch_attempt_count"],
                "stale_apply_patch_repaired_count": report["stale_apply_patch_repaired_count"],
                "terminal_tick": report["terminal_tick"],
                "completion_reason": report["completion_reason"],
                "side_effect_after_verified_completion_count": report["side_effect_after_verified_completion_count"],
                "post_completion_noop_count": report["post_completion_noop_count"],
                "repair_dependency_ratio": report["repair_dependency_ratio"],
                "whether_patch_was_selected_by_mainloop": report["whether_patch_was_selected_by_mainloop"],
                "whether_patch_was_selected_after_posterior_bridge": report["whether_patch_was_selected_after_posterior_bridge"],
                "llm_provider": report["llm_provider"],
                "llm_model": report["llm_model"],
                "llm_call_trace_count": report["llm_call_trace_count"],
                "patch_proposal_llm_trace_count": report["patch_proposal_llm_trace_count"],
                "llm_call_required_passed": report["llm_call_required_passed"],
                "deterministic_fallback_enabled": report["deterministic_fallback_enabled"],
                "prefer_llm_kwargs": report["prefer_llm_kwargs"],
                "prefer_llm_patch_proposals": report["prefer_llm_patch_proposals"],
                "llm_thinking_mode": report["llm_thinking_mode"],
                "changed_paths": report["final_diff_summary"].get("changed_paths", []),
                "report_path": str(REPORT_PATH),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0 if bool(report.get("llm_call_required_passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
