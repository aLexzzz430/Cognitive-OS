from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.closed_loop_probe.run_misleading_localization import enrich_localization_report  # noqa: E402
from experiments.closed_loop_probe.run_probe_v2_mainloop import VALID_VARIANTS, _as_dict, _as_list, run_v2_mainloop  # noqa: E402
from experiments.phase1c_suite.analyze_suite import summarize_reports  # noqa: E402
from experiments.phase1c_suite.external_baseline import run_external_baseline  # noqa: E402


CONFIG_PATH = SUITE_DIR / "suite_config.json"
REPORT_DIR = SUITE_DIR / "reports"
SUMMARY_PATH = SUITE_DIR / "suite_summary.json"
TOOL_ONLY_GREEDY_VARIANT = "tool_only_greedy"
EXTERNAL_BASELINE_VARIANT = "external_coding_agent_baseline"
SUITE_VALID_VARIANTS = set(VALID_VARIANTS) | {TOOL_ONLY_GREEDY_VARIANT, EXTERNAL_BASELINE_VARIANT}


def _runtime_probe_variant(variant: str) -> str:
    # Keep Phase1G frozen: reuse the existing no_posterior runtime semantics
    # instead of adding a new core grounding/ranking branch.
    return "no_posterior" if variant == TOOL_ONLY_GREEDY_VARIANT else str(variant)


def load_suite_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _action_sequence(selected_actions: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(_as_dict(action).get("function_name") or _as_dict(action).get("executed_function_name") or "")
        for action in selected_actions
    ]


def _first_patch_tick(selected_actions: Sequence[Mapping[str, Any]]) -> int | None:
    for action in selected_actions:
        row = _as_dict(action)
        if row.get("executed_function_name") == "propose_patch" and bool(row.get("success")):
            return int(row.get("tick", 0) or 0)
        if row.get("executed_function_name") == "apply_patch" and bool(row.get("success")):
            return int(row.get("tick", 0) or 0)
    return None


def _target_was_run_before_patch(selected_actions: Sequence[Mapping[str, Any]], target: str) -> bool:
    patch_tick = _first_patch_tick(selected_actions)
    if patch_tick is None:
        patch_tick = 10**9
    wanted = str(target or "").split("::", 1)[0]
    if not wanted:
        return False
    for action in selected_actions:
        row = _as_dict(action)
        if int(row.get("tick", 0) or 0) >= patch_tick:
            continue
        if row.get("executed_function_name") != "run_test":
            continue
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        if str(kwargs.get("target") or "").split("::", 1)[0] == wanted:
            return True
    return False


def _file_was_read_before_patch(selected_actions: Sequence[Mapping[str, Any]], path: str) -> bool:
    patch_tick = _first_patch_tick(selected_actions)
    if patch_tick is None:
        patch_tick = 10**9
    wanted = str(path or "")
    if not wanted:
        return False
    for action in selected_actions:
        row = _as_dict(action)
        if int(row.get("tick", 0) or 0) >= patch_tick:
            continue
        if row.get("executed_function_name") != "file_read":
            continue
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        if str(kwargs.get("path") or "") == wanted:
            return True
    return False


def _max_target_binding_confidence(report: Mapping[str, Any]) -> float:
    best = 0.0
    for row in _as_list(report.get("target_binding_by_tick")):
        try:
            best = max(best, float(_as_dict(row).get("target_confidence", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return round(best, 6)


def _has_successful_mirror_plan(selected_actions: Sequence[Mapping[str, Any]]) -> bool:
    for action in selected_actions:
        row = _as_dict(action)
        if row.get("executed_function_name") == "mirror_plan" and bool(row.get("success")):
            return True
    return False


def _changed_paths(report: Mapping[str, Any]) -> list[str]:
    return [
        str(path)
        for path in _as_list(_as_dict(report.get("final_diff_summary")).get("changed_paths"))
        if str(path)
    ]


def _patch_proposal_events(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in _as_list(report.get("patch_proposal_events"))
        if isinstance(row, Mapping)
    ]


def _first_patch_attempt_tick(report: Mapping[str, Any]) -> int | None:
    for event in _patch_proposal_events(report):
        if bool(event.get("patch_proposal_applied")):
            return int(event.get("tick", 0) or 0)
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        if row.get("executed_function_name") == "apply_patch":
            return int(row.get("tick", 0) or 0)
    return None


def _wrong_patch_attempt_count(report: Mapping[str, Any], true_bug_file: str, decoy_files: set[str]) -> int:
    if not true_bug_file and not decoy_files:
        return 0
    count = 0
    for event in _patch_proposal_events(report):
        selected = _as_dict(event.get("patch_proposal_selected"))
        target = str(selected.get("target_file") or "")
        if not target or not bool(event.get("patch_proposal_applied")):
            continue
        if target in decoy_files or (true_bug_file and target != true_bug_file):
            count += 1
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        if row.get("executed_function_name") != "apply_patch":
            continue
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        target = str(kwargs.get("path") or _patch_header_target(str(kwargs.get("patch") or "")) or "")
        if target in decoy_files or (true_bug_file and target and target != true_bug_file):
            count += 1
    return count


def _patch_header_target(patch: str) -> str:
    for line in str(patch or "").splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].strip().split("\t", 1)[0]
        for prefix in ("a/", "b/"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
        return "" if raw in {"/dev/null", "dev/null"} else raw
    return ""


def _rollback_count(report: Mapping[str, Any]) -> int:
    events = _patch_proposal_events(report)
    if events:
        return sum(int(event.get("patch_proposal_rollback_count", 0) or 0) for event in events)
    return int(report.get("patch_proposal_rollback_count", 0) or 0)


def _discriminating_test_tick(
    selected_actions: Sequence[Mapping[str, Any]],
    test_files: Sequence[str],
) -> int | None:
    wanted = {str(item).split("::", 1)[0] for item in test_files if str(item)}
    if not wanted:
        return None
    patch_tick = _first_patch_tick(selected_actions)
    if patch_tick is None:
        patch_tick = 10**9
    best: int | None = None
    for action in selected_actions:
        row = _as_dict(action)
        tick = int(row.get("tick", 0) or 0)
        if tick >= patch_tick:
            continue
        executed = str(row.get("executed_function_name") or row.get("function_name") or "")
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        path = str(kwargs.get("target") or kwargs.get("path") or "").split("::", 1)[0]
        if executed in {"run_test", "file_read"} and path in wanted:
            best = tick if best is None else min(best, tick)
    return best


def _verification_waste_ticks(report: Mapping[str, Any]) -> int:
    first_attempt = _first_patch_attempt_tick(report)
    final_tick = None
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        if row.get("executed_function_name") == "run_test" and bool(row.get("success")):
            kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
            if str(kwargs.get("target") or "") == ".":
                final_tick = int(row.get("tick", 0) or 0)
    if bool(report.get("patch_proposal_verified")):
        for event in _patch_proposal_events(report):
            if bool(event.get("patch_proposal_verified")):
                final_tick = int(event.get("tick", 0) or 0)
                break
    failed_verifier_actions = 0
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        if row.get("executed_function_name") != "run_test" or bool(row.get("success")):
            continue
        tick = int(row.get("tick", 0) or 0)
        if first_attempt is not None and tick >= first_attempt and (final_tick is None or tick <= final_tick):
            failed_verifier_actions += 1
    return failed_verifier_actions + _rollback_count(report)


def _capability_matrix(variant: str) -> dict[str, Any]:
    posterior_enabled = str(variant) not in {"no_posterior", TOOL_ONLY_GREEDY_VARIANT, EXTERNAL_BASELINE_VARIANT}
    discriminating_enabled = str(variant) not in {"no_discriminating_experiment", TOOL_ONLY_GREEDY_VARIANT, EXTERNAL_BASELINE_VARIANT}
    return {
        "can_run_tests": True,
        "can_read_files": True,
        "can_target_bind": True,
        "can_generate_patch_proposal": True,
        "can_apply_patch": True,
        "can_verify_patch": True,
        "posterior_writeback_enabled": posterior_enabled,
        "discriminating_experiment_enabled": discriminating_enabled,
        "discriminating_bonus_enabled": discriminating_enabled,
        "target_binding_enabled": True,
        "patch_proposal_enabled": True,
        "explicit_hypothesis_lifecycle_enabled": str(variant) != EXTERNAL_BASELINE_VARIANT,
        "posterior_to_action_bridge_enabled": str(variant) not in {"no_posterior", TOOL_ONLY_GREEDY_VARIANT, EXTERNAL_BASELINE_VARIANT},
        "leading_hypothesis_requirement_enabled": str(variant) != EXTERNAL_BASELINE_VARIANT,
    }


def _test_failure_observed(selected_actions: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        str(_as_dict(action).get("executed_function_name") or "") == "run_test"
        and not bool(_as_dict(action).get("success"))
        for action in selected_actions
    )


def _competing_hypotheses_created(report: Mapping[str, Any]) -> bool:
    for row in _as_list(report.get("competing_hypotheses_by_tick")):
        if len(_as_list(_as_dict(row).get("hypotheses"))) >= 2:
            return True
    for row in _as_list(report.get("posterior_summary_by_tick")):
        summary = _as_dict(_as_dict(row).get("posterior_summary"))
        snapshot = _as_dict(summary.get("deliberation_snapshot"))
        try:
            if int(snapshot.get("ranked_hypothesis_count", 0) or 0) >= 2:
                return True
        except (TypeError, ValueError):
            continue
        if len(_as_list(summary.get("ranked_hypothesis_object_ids"))) >= 2:
            return True
    return False


def _hypothesis_ids_seen(report: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for row in _as_list(report.get("competing_hypotheses_by_tick")):
        for hypothesis in _as_list(_as_dict(row).get("hypotheses")):
            hypothesis_id = str(_as_dict(hypothesis).get("hypothesis_id") or "")
            if hypothesis_id and hypothesis_id not in ids:
                ids.append(hypothesis_id)
    return sorted(ids)


def _competing_hypotheses_created_count(report: Mapping[str, Any]) -> int:
    return len(_hypothesis_ids_seen(report))


def _min_hypotheses_before_patch(report: Mapping[str, Any]) -> int:
    patch_tick_value = _first_patch_tick(_as_list(report.get("selected_actions")))
    if patch_tick_value is None:
        patch_tick_value = 10**9
    counts = [
        len(_as_list(_as_dict(row).get("hypotheses")))
        for row in _as_list(report.get("competing_hypotheses_by_tick"))
        if int(_as_dict(row).get("tick", 0) or 0) <= patch_tick_value
        and len(_as_list(_as_dict(row).get("hypotheses"))) > 0
    ]
    return min(counts) if counts else 0


def _hypothesis_refs_from_experiment(experiment: Mapping[str, Any]) -> list[str]:
    refs = []
    for raw in (
        experiment.get("discriminates_between"),
        experiment.get("hypotheses"),
        [experiment.get("hypothesis_a"), experiment.get("hypothesis_b")],
    ):
        for item in _as_list(raw):
            text = str(item or "").strip()
            if text and text not in refs:
                refs.append(text)
    return refs


def _discriminating_experiments_bound_count(report: Mapping[str, Any]) -> int:
    seen: set[str] = set()
    count = 0
    for row in _as_list(report.get("ranked_discriminating_experiments_by_tick")):
        for experiment in _as_list(_as_dict(row).get("experiments")):
            exp = _as_dict(experiment)
            refs = _hypothesis_refs_from_experiment(exp)
            if len(refs) < 2:
                continue
            key = str(exp.get("test_id") or json.dumps({"refs": refs, "action": exp.get("action")}, sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            count += 1
    return count


def _posterior_events_bound_count(report: Mapping[str, Any], ids_seen: Sequence[str]) -> int:
    known = set(str(item) for item in ids_seen if str(item))
    seen: set[str] = set()
    count = 0
    for row in _as_list(report.get("hypothesis_posterior_events_by_tick")):
        for event in _as_list(_as_dict(row).get("events")):
            payload = _as_dict(event)
            hypothesis_id = str(payload.get("hypothesis_id") or "")
            if hypothesis_id not in known:
                continue
            event_type = str(payload.get("event_type") or "")
            key = str(payload.get("_event_key") or json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))
            if event_type.startswith("evidence_") and float(payload.get("delta", 0.0) or 0.0) != 0.0 and key not in seen:
                seen.add(key)
                count += 1
    return count


def _leading_hypothesis_before_patch(report: Mapping[str, Any]) -> str:
    for event in _as_list(report.get("patch_proposal_events")):
        source = str(_as_dict(event).get("source_hypothesis_id") or "")
        if source:
            return source
        leading = str(_as_dict(_as_dict(event).get("leading_hypothesis_before_patch")).get("hypothesis_id") or "")
        if leading:
            return leading
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        if str(row.get("executed_function_name") or "") not in {"apply_patch", "propose_patch"}:
            continue
        source = str(row.get("source_hypothesis_id") or "")
        if source:
            return source
        leading = str(_as_dict(row.get("leading_hypothesis_before_patch")).get("hypothesis_id") or "")
        if leading:
            return leading
    patch_tick_value = _first_patch_tick(_as_list(report.get("selected_actions")))
    if patch_tick_value is None:
        patch_tick_value = 10**9
    leading = ""
    for row in _as_list(report.get("posterior_summary_by_tick")):
        tick = int(_as_dict(row).get("tick", 0) or 0)
        if tick > patch_tick_value:
            continue
        summary = _as_dict(_as_dict(row).get("posterior_summary"))
        candidate = str(summary.get("leading_hypothesis_id") or "")
        if candidate:
            leading = candidate
    return leading


def _patch_referenced_hypothesis(report: Mapping[str, Any], leading_id: str, ids_seen: Sequence[str]) -> bool:
    known = set(str(item) for item in ids_seen if str(item))
    refs: list[str] = []
    for event in _as_list(report.get("patch_proposal_events")):
        payload = _as_dict(event)
        refs.append(str(payload.get("source_hypothesis_id") or ""))
        refs.append(str(_as_dict(payload.get("leading_hypothesis_before_patch")).get("hypothesis_id") or ""))
        refs.append(str(_as_dict(payload.get("patch_proposal_selected")).get("source_hypothesis_id") or ""))
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        refs.append(str(row.get("source_hypothesis_id") or ""))
        refs.append(str(_as_dict(row.get("leading_hypothesis_before_patch")).get("hypothesis_id") or ""))
    refs = [ref for ref in refs if ref]
    if leading_id and leading_id in refs:
        return True
    return any(ref in known for ref in refs)


def _hypothesis_lifecycle_complete(report: Mapping[str, Any]) -> bool:
    ids_seen = _hypothesis_ids_seen(report)
    leading = _leading_hypothesis_before_patch(report)
    final_ok = bool(report.get("final_tests_passed") or (report.get("needs_human_review") and report.get("refusal_reason")))
    return bool(
        len(ids_seen) >= 2
        and _discriminating_experiments_bound_count(report) >= 1
        and _posterior_events_bound_count(report, ids_seen) >= 1
        and leading
        and _patch_referenced_hypothesis(report, leading, ids_seen)
        and final_ok
    )


def _target_binding_present(report: Mapping[str, Any], confidence: float) -> bool:
    if confidence >= 0.55:
        return True
    for row in _as_list(report.get("target_binding_by_tick")):
        binding = _as_dict(row)
        if str(binding.get("top_target_file") or ""):
            return True
    return False


def _mechanism_failure_reason(
    *,
    variant: str,
    task_success: bool,
    capability_matrix: Mapping[str, Any],
    test_failure_observed: bool,
    discriminating_evidence_selected: bool,
    target_binding_present: bool,
    patch_proposal_generated: bool,
    patch_selected: bool,
    patch_proposal_rejected: bool,
    verifier_acceptance: bool,
    terminal_completion: bool,
) -> str:
    if task_success:
        return ""
    if not bool(capability_matrix.get("can_run_tests")) or not bool(capability_matrix.get("can_read_files")):
        return "tool_unavailable"
    if not bool(capability_matrix.get("target_binding_enabled")):
        return "target_binding_disabled"
    if not bool(capability_matrix.get("patch_proposal_enabled")):
        return "patch_proposal_disabled"
    if not test_failure_observed:
        return "test_failure_not_observed"
    if not target_binding_present:
        return "target_confidence_below_threshold"
    if str(variant) in {"no_discriminating_experiment", TOOL_ONLY_GREEDY_VARIANT} and not discriminating_evidence_selected:
        return "no_disambiguating_evidence"
    if not patch_selected and not patch_proposal_generated:
        return "patch_proposal_not_selected"
    if patch_proposal_rejected:
        return "verifier_rejected_patch"
    if not verifier_acceptance:
        return "verifier_acceptance_missing"
    if not terminal_completion:
        return "terminal_completion_missing"
    return "unknown_failure"


def _mechanism_path(
    *,
    report: Mapping[str, Any],
    variant: str,
    task_success: bool,
    capability_matrix: Mapping[str, Any],
    selected_actions: Sequence[Mapping[str, Any]],
    discriminating_before_patch: bool,
    target_binding_confidence: float,
) -> dict[str, Any]:
    patch_proposal_generated = int(report.get("patch_proposals_generated", 0) or 0) > 0
    patch_selected = bool(
        report.get("patch_proposal_selected")
        or report.get("whether_patch_was_selected_by_mainloop")
        or _first_patch_tick(selected_actions) is not None
    )
    patch_proposal_rejected = bool(_as_list(report.get("rejected_patch_proposals")))
    verifier_acceptance = bool(report.get("final_tests_passed") or report.get("patch_proposal_verified"))
    terminal_state = str(report.get("terminal_state") or "")
    terminal_completion = bool(terminal_state in {"completed_verified", "needs_human_review"} or report.get("verified_completion"))
    failure_observed = _test_failure_observed(selected_actions)
    competing = _competing_hypotheses_created(report)
    posterior_shift = bool(report.get("posterior_shift_from_traceback_file_to_true_bug_file"))
    target_binding = _target_binding_present(report, target_binding_confidence)
    failure_reason = _mechanism_failure_reason(
        variant=variant,
        task_success=task_success,
        capability_matrix=capability_matrix,
        test_failure_observed=failure_observed,
        discriminating_evidence_selected=discriminating_before_patch,
        target_binding_present=target_binding,
        patch_proposal_generated=patch_proposal_generated,
        patch_selected=patch_selected,
        patch_proposal_rejected=patch_proposal_rejected,
        verifier_acceptance=verifier_acceptance,
        terminal_completion=terminal_completion,
    )
    return {
        "test_failure_observed": failure_observed,
        "competing_hypotheses_created": competing,
        "hypothesis_lifecycle_complete": _hypothesis_lifecycle_complete(report),
        "discriminating_evidence_selected": bool(discriminating_before_patch),
        "hypothesis_pair_distinguished_before_patch": bool(discriminating_before_patch),
        "posterior_shift": posterior_shift,
        "target_binding": target_binding,
        "target_binding_confidence": float(target_binding_confidence),
        "patch_proposal_generated": patch_proposal_generated,
        "patch_selected": patch_selected,
        "verifier_acceptance": verifier_acceptance,
        "terminal_completion": terminal_completion,
        "failure_reason": failure_reason,
        "ablation_contaminated": failure_reason in {"target_binding_disabled", "patch_proposal_disabled", "tool_unavailable"},
    }


def _report_path_for(*, fixture_id: str, variant: str, max_ticks: int, repeat: int, report_dir: Path) -> Path:
    return report_dir / f"{fixture_id}_{variant}_ticks{int(max_ticks)}_run{int(repeat)}.json"


def run_fixture_probe(
    fixture: Mapping[str, Any],
    *,
    variant: str,
    max_ticks: int,
    repeat: int = 1,
    report_path: str | Path | None = None,
    suite_kind: str = "phase1c_suite",
    llm_provider: str = "none",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_timeout: float = 60.0,
    deterministic_fallback_enabled: bool = True,
    prefer_llm_kwargs: bool = False,
    prefer_llm_patch_proposals: bool = False,
    llm_thinking_mode: str = "auto",
    require_llm_call: bool = False,
) -> dict[str, Any]:
    if variant not in SUITE_VALID_VARIANTS:
        raise ValueError(f"unknown Phase1C variant: {variant}")
    runtime_variant = _runtime_probe_variant(str(variant))
    fixture_id = str(fixture.get("fixture_id") or Path(str(fixture.get("fixture_root") or "fixture")).name)
    output_path = Path(report_path) if report_path is not None else _report_path_for(
        fixture_id=fixture_id,
        variant=variant,
        max_ticks=max_ticks,
        repeat=repeat,
        report_dir=REPORT_DIR,
    )
    instruction = str(fixture.get("instruction") or "").format(variant=runtime_variant)
    if str(variant) == EXTERNAL_BASELINE_VARIANT:
        raw_report = run_external_baseline(
            fixture_root=REPO_ROOT / str(fixture.get("fixture_root") or ""),
            instruction=instruction,
            run_id=str(fixture.get("run_id") or f"phase1c-{fixture_id}"),
            max_ticks=int(max_ticks),
            report_path=output_path,
        )
    else:
        raw_report = run_v2_mainloop(
            variant=runtime_variant,
            max_ticks=int(max_ticks),
            fixture_root=REPO_ROOT / str(fixture.get("fixture_root") or ""),
            instruction=instruction,
            run_id=str(fixture.get("run_id") or f"phase1c-{fixture_id}"),
            report_path=output_path,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_timeout=float(llm_timeout),
            deterministic_fallback_enabled=bool(deterministic_fallback_enabled),
            prefer_llm_kwargs=bool(prefer_llm_kwargs),
            prefer_llm_patch_proposals=bool(prefer_llm_patch_proposals),
            llm_thinking_mode=llm_thinking_mode,
            require_llm_call=bool(require_llm_call),
        )
    enriched = enrich_localization_report(
        raw_report,
        schema_version=f"conos.phase1c_suite.{fixture_id}/v1",
        traceback_file=str(fixture.get("traceback_file") or ""),
        true_bug_file=str(fixture.get("true_bug_file") or ""),
        bug_type=str(fixture.get("bug_type") or ""),
        direct_unit_test_file=str(fixture.get("direct_unit_test_file") or ""),
        require_posterior_shift_for_success=bool(fixture.get("require_posterior_shift_for_success", True)),
    )
    selected_actions = [_as_dict(action) for action in _as_list(enriched.get("selected_actions"))]
    direct_unit_test_file = str(fixture.get("direct_unit_test_file") or "")
    discriminating_test_files = [
        str(item)
        for item in _as_list(fixture.get("discriminating_test_files"))
        if str(item)
    ]
    if direct_unit_test_file and direct_unit_test_file not in discriminating_test_files:
        discriminating_test_files.append(direct_unit_test_file)
    direct_evidence = bool(
        _target_was_run_before_patch(selected_actions, direct_unit_test_file)
        or _file_was_read_before_patch(selected_actions, direct_unit_test_file)
    )
    discriminating_tick = _discriminating_test_tick(selected_actions, discriminating_test_files)
    patch_tick_value = _first_patch_tick(selected_actions)
    true_bug_file = str(fixture.get("true_bug_file") or "")
    decoy_files = {str(item) for item in _as_list(fixture.get("decoy_files")) if str(item)}
    wrong_patch_attempt_count = _wrong_patch_attempt_count(enriched, true_bug_file, decoy_files)
    rollback_count = _rollback_count(enriched)
    decoy_patch_selected = wrong_patch_attempt_count > 0
    discriminating_before_patch = bool(discriminating_tick is not None and (patch_tick_value is None or discriminating_tick < patch_tick_value))
    verification_waste_ticks = _verification_waste_ticks(enriched)
    target_binding_confidence = _max_target_binding_confidence(enriched)
    patch_proposal_selected_via_binding = bool(enriched.get("patch_proposal_selected_via_target_binding"))
    changed_paths = _changed_paths(enriched)
    expect_refusal = bool(fixture.get("expect_refusal", False))
    needs_human_review = bool(enriched.get("needs_human_review", False))
    refusal_reason = str(enriched.get("refusal_reason") or "")
    unsafe_patch_avoided = bool(not changed_paths and not _has_successful_mirror_plan(selected_actions))
    if expect_refusal:
        expected_reasons = {str(item) for item in _as_list(fixture.get("expected_refusal_reasons")) if str(item)}
        if not expected_reasons:
            expected_reasons = {"evidence_insufficient", "ambiguous_spec"}
        task_success = bool(needs_human_review and unsafe_patch_avoided and refusal_reason in expected_reasons)
        cognitive_success = False
        enriched["patched_file"] = ""
        enriched["patched_traceback_file"] = False
    else:
        task_success = bool(enriched.get("success"))
        cognitive_success = bool(
            (
                bool(enriched.get("posterior_shift_from_traceback_file_to_true_bug_file"))
                or target_binding_confidence >= float(fixture.get("target_binding_threshold", 0.55) or 0.55)
            )
            and (
                bool(enriched.get("patch_selected_after_posterior_bridge"))
                or patch_proposal_selected_via_binding
            )
            and bool(enriched.get("final_tests_passed"))
            and not bool(enriched.get("completed_before_verification"))
        )
        if str(variant) == EXTERNAL_BASELINE_VARIANT:
            cognitive_success = False
    capability = _capability_matrix(str(variant))
    mechanism_path = _mechanism_path(
        report=enriched,
        variant=str(variant),
        task_success=task_success,
        capability_matrix=capability,
        selected_actions=selected_actions,
        discriminating_before_patch=discriminating_before_patch,
        target_binding_confidence=target_binding_confidence,
    )
    enriched.update(
        {
            "suite_kind": suite_kind,
            "variant": str(variant),
            "runtime_probe_variant": runtime_variant,
            "fixture_id": fixture_id,
            "fixture_root": str(fixture.get("fixture_root") or ""),
            "expect_refusal": expect_refusal,
            "repeat": int(repeat),
            "max_ticks": int(max_ticks),
            "requested_max_ticks": int(max_ticks),
            "direct_evidence_before_patch": direct_evidence,
            "discriminating_test_files": discriminating_test_files,
            "wrong_patch_attempt_count": wrong_patch_attempt_count,
            "rollback_count": rollback_count,
            "decoy_patch_selected": decoy_patch_selected,
            "discriminating_test_selected_before_patch": discriminating_before_patch,
            "hypothesis_pair_distinguished_before_patch": discriminating_before_patch,
            "patch_after_disambiguation": bool(patch_tick_value is not None and discriminating_before_patch),
            "verification_waste_ticks": verification_waste_ticks,
            "target_binding_confidence": target_binding_confidence,
            "patch_proposal_selected_via_target_binding": patch_proposal_selected_via_binding,
            "unsafe_patch_avoided": unsafe_patch_avoided,
            "task_success": task_success,
            "cognitive_success": cognitive_success,
            "capability_matrix": capability,
            "mechanism_path": mechanism_path,
            "competing_hypotheses_created_count": _competing_hypotheses_created_count(enriched),
            "min_hypotheses_before_patch": _min_hypotheses_before_patch(enriched),
            "hypothesis_ids_seen": _hypothesis_ids_seen(enriched),
            "discriminating_experiments_bound_to_hypotheses_count": _discriminating_experiments_bound_count(enriched),
            "posterior_events_bound_to_hypotheses_count": _posterior_events_bound_count(enriched, _hypothesis_ids_seen(enriched)),
            "leading_hypothesis_before_patch": _leading_hypothesis_before_patch(enriched),
            "patch_referenced_hypothesis": _patch_referenced_hypothesis(
                enriched,
                _leading_hypothesis_before_patch(enriched),
                _hypothesis_ids_seen(enriched),
            ),
            "hypothesis_lifecycle_complete": _hypothesis_lifecycle_complete(enriched),
            "failure_reason": str(mechanism_path.get("failure_reason") or ""),
            "ablation_contaminated": bool(mechanism_path.get("ablation_contaminated", False)),
            "success": task_success,
            "action_sequence": _action_sequence(selected_actions),
            "report_path": str(output_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return enriched


def run_suite(*, repeats: int, max_ticks: int, variants: Sequence[str] | None = None) -> dict[str, Any]:
    return run_suite_with_options(repeats=repeats, max_ticks=max_ticks, variants=variants)


def run_suite_with_options(
    *,
    repeats: int,
    max_ticks: int,
    variants: Sequence[str] | None = None,
    llm_provider: str = "none",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_timeout: float = 60.0,
    deterministic_fallback_enabled: bool = True,
    prefer_llm_kwargs: bool = False,
    prefer_llm_patch_proposals: bool = False,
    llm_thinking_mode: str = "auto",
    require_llm_call: bool = False,
) -> dict[str, Any]:
    config = load_suite_config()
    selected_variants = list(variants or config.get("variants") or sorted(SUITE_VALID_VARIANTS))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for fixture in _as_list(config.get("fixtures")):
        if not isinstance(fixture, Mapping):
            continue
        for variant in selected_variants:
            for repeat in range(1, int(repeats) + 1):
                reports.append(
                    run_fixture_probe(
                        fixture,
                        variant=str(variant),
                        max_ticks=int(max_ticks),
                        repeat=repeat,
                        llm_provider=llm_provider,
                        llm_base_url=llm_base_url,
                        llm_model=llm_model,
                        llm_timeout=float(llm_timeout),
                        deterministic_fallback_enabled=bool(deterministic_fallback_enabled),
                        prefer_llm_kwargs=bool(prefer_llm_kwargs),
                        prefer_llm_patch_proposals=bool(prefer_llm_patch_proposals),
                        llm_thinking_mode=llm_thinking_mode,
                        require_llm_call=bool(require_llm_call),
                    )
                )
    summary = summarize_reports(reports)
    summary["suite_config"] = str(CONFIG_PATH)
    summary["active_report_paths"] = [
        str(report.get("report_path") or "")
        for report in reports
        if str(report.get("report_path") or "")
    ]
    summary["phase1f_scope_note"] = {
        "core_behavior_files_modified": [
            "integrations/local_machine/action_grounding.py",
            "integrations/local_machine/target_binding.py",
        ],
        "note": "Phase1F intentionally tightened generic discriminating-evidence grounding; no fixture-specific patch heuristic was added.",
    }
    summary["llm_runtime_config"] = {
        "llm_provider": str(llm_provider or "none"),
        "llm_base_url": str(llm_base_url or ""),
        "llm_model": str(llm_model or ""),
        "llm_timeout": float(llm_timeout),
        "deterministic_fallback_enabled": bool(deterministic_fallback_enabled),
        "prefer_llm_kwargs": bool(prefer_llm_kwargs),
        "prefer_llm_patch_proposals": bool(prefer_llm_patch_proposals),
        "llm_thinking_mode": str(llm_thinking_mode or "auto"),
        "require_llm_call": bool(require_llm_call),
    }
    llm_missing = [
        str(report.get("report_path") or "")
        for report in reports
        if bool(report.get("llm_call_required")) and not bool(report.get("llm_call_required_passed"))
    ]
    summary["llm_call_requirement"] = {
        "required": bool(require_llm_call),
        "missing_call_report_count": len(llm_missing),
        "missing_call_report_paths": llm_missing,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase1C closed-loop fixture suite.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-ticks", type=int, default=20)
    parser.add_argument("--variant", action="append", choices=sorted(SUITE_VALID_VARIANTS))
    parser.add_argument("--include-external-baseline", action="store_true")
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
    variants = args.variant
    if args.include_external_baseline and not variants:
        config_variants = [
            str(item)
            for item in _as_list(load_suite_config().get("variants"))
            if str(item)
        ]
        variants = config_variants + [EXTERNAL_BASELINE_VARIANT]
    elif args.include_external_baseline and variants and EXTERNAL_BASELINE_VARIANT not in variants:
        variants = list(variants) + [EXTERNAL_BASELINE_VARIANT]
    summary = run_suite_with_options(
        repeats=int(args.repeats),
        max_ticks=int(args.max_ticks),
        variants=variants,
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
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if bool(args.require_llm_call) and int(_as_dict(summary.get("llm_call_requirement")).get("missing_call_report_count", 0) or 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
