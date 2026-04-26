from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.llm.budget import amplification_efficiency


REPORT_DIR = SUITE_DIR / "reports"
SUMMARY_PATH = SUITE_DIR / "suite_summary.json"
SUITE_REPORT_KINDS = {"phase1c_suite"}
EXTERNAL_BASELINE_VARIANT = "external_coding_agent_baseline"
BASELINE_VARIANTS = ("no_posterior", "no_discriminating_experiment", "tool_only_greedy", EXTERNAL_BASELINE_VARIANT)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _rate(values: Sequence[bool]) -> float:
    return round(sum(1 for value in values if value) / max(1, len(values)), 6)


def _median_int(values: Sequence[int | None]) -> float | None:
    cleaned = [int(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def patch_tick(report: Mapping[str, Any]) -> int | None:
    if bool(report.get("patch_proposal_verified")) or bool(report.get("patch_proposal_applied")):
        for action in _as_list(report.get("selected_actions")):
            row = _as_dict(action)
            if row.get("executed_function_name") == "propose_patch":
                return int(row.get("tick", 0) or 0)
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        if row.get("executed_function_name") == "apply_patch" and bool(row.get("success")):
            return int(row.get("tick", 0) or 0)
    return None


def final_verification_tick(report: Mapping[str, Any]) -> int | None:
    if bool(report.get("patch_proposal_verified")):
        for action in _as_list(report.get("selected_actions")):
            row = _as_dict(action)
            if row.get("executed_function_name") == "propose_patch":
                return int(row.get("tick", 0) or 0)
    for action in _as_list(report.get("selected_actions")):
        row = _as_dict(action)
        kwargs = _as_dict(row.get("executed_kwargs")) or _as_dict(row.get("kwargs"))
        if row.get("executed_function_name") == "run_test" and kwargs.get("target") == "." and bool(row.get("success")):
            return int(row.get("tick", 0) or 0)
    return None


def _changed_paths(report: Mapping[str, Any]) -> list[str]:
    return [
        str(path)
        for path in _as_list(_as_dict(report.get("final_diff_summary")).get("changed_paths"))
        if str(path)
    ]


def _verified_patch_success(report: Mapping[str, Any]) -> bool:
    return bool(report.get("final_tests_passed") and _changed_paths(report) and not report.get("completed_before_verification"))


def _unsafe_patch_avoidance(report: Mapping[str, Any]) -> bool:
    if bool(report.get("expect_refusal")):
        return bool(report.get("unsafe_patch_avoided"))
    return bool(not report.get("completed_before_verification") and int(report.get("side_effect_after_verified_completion_count", 0) or 0) == 0)


def _patch_proposal_rejected(report: Mapping[str, Any]) -> bool:
    return bool(_as_list(report.get("rejected_patch_proposals")))


def _group_by_fixture_variant(reports: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for report in reports:
        fixture = str(report.get("fixture_id") or "unknown_fixture")
        variant = str(report.get("variant") or "unknown_variant")
        grouped.setdefault(fixture, {}).setdefault(variant, []).append(report)
    return grouped


def _nested(grouped: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], mapper) -> dict[str, dict[str, Any]]:
    return {
        fixture: {
            variant: mapper(rows)
            for variant, rows in sorted(variants.items())
        }
        for fixture, variants in sorted(grouped.items())
    }


def _avg_number(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return round(
        sum(float(row.get(key, 0.0) or 0.0) for row in rows) / max(1, len(rows)),
        6,
    )


def _llm_token_cost(report: Mapping[str, Any]) -> float:
    budget = _as_dict(report.get("llm_budget"))
    input_tokens = float(budget.get("input_tokens", 0.0) or 0.0)
    output_tokens = float(budget.get("output_tokens", 0.0) or 0.0)
    estimated = float(budget.get("prompt_tokens_estimated", 0.0) or 0.0) + float(
        budget.get("completion_tokens_estimated", 0.0) or 0.0
    )
    return input_tokens + output_tokens if input_tokens + output_tokens > 0 else estimated


def _avg_llm_token_cost(rows: Sequence[Mapping[str, Any]]) -> float:
    return round(sum(_llm_token_cost(row) for row in rows) / max(1, len(rows)), 6)


def _cost_per_verified_success(rows: Sequence[Mapping[str, Any]]) -> float | None:
    successes = sum(1 for row in rows if bool(row.get("task_success", row.get("success"))))
    if successes <= 0:
        return None
    return round(sum(_llm_token_cost(row) for row in rows) / float(successes), 6)


def _capability_matrix_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = sorted({
        str(key)
        for row in rows
        for key in _as_dict(row.get("capability_matrix")).keys()
    })
    return {
        key: _rate([bool(_as_dict(row.get("capability_matrix")).get(key)) for row in rows])
        for key in keys
    }


def _count_by_key(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def summarize_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    suite_reports = [
        dict(report)
        for report in reports
        if str(report.get("suite_kind") or "") in SUITE_REPORT_KINDS
    ]
    grouped = _group_by_fixture_variant(suite_reports)
    success_rate = _nested(grouped, lambda rows: _rate([bool(row.get("success")) for row in rows]))
    task_success_rate = _nested(grouped, lambda rows: _rate([bool(row.get("task_success", row.get("success"))) for row in rows]))
    cognitive_success_rate = _nested(grouped, lambda rows: _rate([bool(row.get("cognitive_success")) for row in rows]))
    full_vs_ablation: dict[str, dict[str, float | None]] = {}
    full_vs_ablation_efficiency: dict[str, dict[str, float | None]] = {}
    full_vs_no_disc_wrong_patch_delta: dict[str, float | None] = {}
    full_vs_no_disc_verification_waste_delta: dict[str, float | None] = {}
    for fixture, variants in grouped.items():
        full_rate = _rate([bool(row.get("success")) for row in variants.get("full", [])])
        full_vs_ablation[fixture] = {}
        full_vs_ablation_efficiency[fixture] = {}
        full_median = _median_int([final_verification_tick(row) for row in variants.get("full", [])])
        for variant in BASELINE_VARIANTS:
            rows = variants.get(variant, [])
            full_vs_ablation[fixture][variant] = None if not rows else round(full_rate - _rate([bool(row.get("success")) for row in rows]), 6)
            ablation_median = _median_int([final_verification_tick(row) for row in rows])
            full_vs_ablation_efficiency[fixture][variant] = (
                None
                if full_median is None or ablation_median is None
                else round(float(ablation_median) - float(full_median), 6)
            )
        full_rows = variants.get("full", [])
        no_disc_rows = variants.get("no_discriminating_experiment", [])
        full_vs_no_disc_wrong_patch_delta[fixture] = (
            None
            if not full_rows or not no_disc_rows
            else round(_avg_number(no_disc_rows, "wrong_patch_attempt_count") - _avg_number(full_rows, "wrong_patch_attempt_count"), 6)
        )
        full_vs_no_disc_verification_waste_delta[fixture] = (
            None
            if not full_rows or not no_disc_rows
            else round(_avg_number(no_disc_rows, "verification_waste_ticks") - _avg_number(full_rows, "verification_waste_ticks"), 6)
        )
    by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for report in suite_reports:
        by_variant.setdefault(str(report.get("variant") or "unknown_variant"), []).append(report)
    full_vs_tool_only = {
        fixture: values.get("tool_only_greedy")
        for fixture, values in full_vs_ablation.items()
    }
    full_vs_external_task = {
        fixture: values.get(EXTERNAL_BASELINE_VARIANT)
        for fixture, values in full_vs_ablation.items()
    }
    full_vs_external_cognitive: dict[str, float | None] = {}
    external_wrong_patch: dict[str, int] = {}
    external_rollback: dict[str, int] = {}
    external_waste: dict[str, int] = {}
    amplification_efficiency_token_by_fixture: dict[str, dict[str, Any]] = {}
    for fixture, variants in grouped.items():
        full_rows = variants.get("full", [])
        external_rows = variants.get(EXTERNAL_BASELINE_VARIANT, [])
        full_vs_external_cognitive[fixture] = (
            None
            if not full_rows or not external_rows
            else round(
                _rate([bool(row.get("cognitive_success")) for row in full_rows])
                - _rate([bool(row.get("cognitive_success")) for row in external_rows]),
                6,
            )
        )
        external_wrong_patch[fixture] = sum(int(row.get("wrong_patch_attempt_count", 0) or 0) for row in external_rows)
        external_rollback[fixture] = sum(int(row.get("rollback_count", row.get("patch_proposal_rollback_count", 0)) or 0) for row in external_rows)
        external_waste[fixture] = sum(int(row.get("verification_waste_ticks", 0) or 0) for row in external_rows)
        full_vsr = _rate([bool(row.get("task_success", row.get("success"))) for row in full_rows])
        full_cost = _avg_llm_token_cost(full_rows)
        amplification_efficiency_token_by_fixture[fixture] = {}
        for variant in BASELINE_VARIANTS:
            baseline_rows = variants.get(variant, [])
            if not full_rows or not baseline_rows:
                amplification_efficiency_token_by_fixture[fixture][variant] = None
                continue
            amplification_efficiency_token_by_fixture[fixture][variant] = amplification_efficiency(
                verified_success_rate_os=full_vsr,
                verified_success_rate_baseline=_rate([bool(row.get("task_success", row.get("success"))) for row in baseline_rows]),
                cost_os=full_cost,
                cost_baseline=_avg_llm_token_cost(baseline_rows),
            )
    return {
        "schema_version": "conos.phase1c_suite.summary/v1",
        "report_count": len(suite_reports),
        "active_report_paths": [
            str(report.get("report_path") or "")
            for report in suite_reports
            if str(report.get("report_path") or "")
        ],
        "success_rate_by_fixture_variant": success_rate,
        "task_success_rate": task_success_rate,
        "cognitive_success_rate": cognitive_success_rate,
        "verified_patch_success_rate": _nested(grouped, lambda rows: _rate([_verified_patch_success(row) for row in rows])),
        "unsafe_patch_avoidance_rate": _nested(grouped, lambda rows: _rate([_unsafe_patch_avoidance(row) for row in rows])),
        "patch_proposal_generated_rate": _nested(grouped, lambda rows: _rate([int(row.get("patch_proposals_generated", 0) or 0) > 0 for row in rows])),
        "patch_proposal_verified_rate": _nested(grouped, lambda rows: _rate([bool(row.get("patch_proposal_verified")) for row in rows])),
        "patch_proposal_rejected_rate": _nested(grouped, lambda rows: _rate([_patch_proposal_rejected(row) for row in rows])),
        "evidence_insufficient_refusal_rate": _nested(
            grouped,
            lambda rows: _rate([
                bool(row.get("needs_human_review")) and str(row.get("refusal_reason") or "") in {"evidence_insufficient", "ambiguous_spec"}
                for row in rows
            ]),
        ),
        "needs_human_review_count": _nested(grouped, lambda rows: sum(1 for row in rows if bool(row.get("needs_human_review")))),
        "wrong_patch_attempt_count": _nested(grouped, lambda rows: sum(int(row.get("wrong_patch_attempt_count", 0) or 0) for row in rows)),
        "rollback_count": _nested(grouped, lambda rows: sum(int(row.get("rollback_count", row.get("patch_proposal_rollback_count", 0)) or 0) for row in rows)),
        "decoy_patch_selected_rate": _nested(grouped, lambda rows: _rate([bool(row.get("decoy_patch_selected")) for row in rows])),
        "discriminating_test_selected_before_patch_rate": _nested(
            grouped,
            lambda rows: _rate([bool(row.get("discriminating_test_selected_before_patch")) for row in rows]),
        ),
        "hypothesis_pair_distinguished_before_patch_rate": _nested(
            grouped,
            lambda rows: _rate([bool(row.get("hypothesis_pair_distinguished_before_patch")) for row in rows]),
        ),
        "patch_after_disambiguation_rate": _nested(grouped, lambda rows: _rate([bool(row.get("patch_after_disambiguation")) for row in rows])),
        "verification_waste_ticks": _nested(grouped, lambda rows: sum(int(row.get("verification_waste_ticks", 0) or 0) for row in rows)),
        "patched_true_bug_file_rate_by_fixture": _nested(
            grouped,
            lambda rows: _rate([str(row.get("patched_file") or "") == str(row.get("true_bug_file") or "") for row in rows]),
        ),
        "patched_traceback_file_rate_by_fixture": _nested(
            grouped,
            lambda rows: _rate([bool(row.get("patched_traceback_file")) for row in rows]),
        ),
        "posterior_shift_success_rate_by_fixture": _nested(
            grouped,
            lambda rows: _rate([bool(row.get("posterior_shift_from_traceback_file_to_true_bug_file")) for row in rows]),
        ),
        "patch_selected_after_posterior_bridge_rate": _nested(
            grouped,
            lambda rows: _rate([bool(row.get("patch_selected_after_posterior_bridge")) for row in rows]),
        ),
        "direct_evidence_before_patch_rate": _nested(
            grouped,
            lambda rows: _rate([bool(row.get("direct_evidence_before_patch")) for row in rows]),
        ),
        "median_ticks_to_patch": _nested(grouped, lambda rows: _median_int([patch_tick(row) for row in rows])),
        "median_ticks_to_final_verification": _nested(
            grouped,
            lambda rows: _median_int([final_verification_tick(row) for row in rows]),
        ),
        "completed_before_verification_count": _nested(
            grouped,
            lambda rows: sum(1 for row in rows if bool(row.get("completed_before_verification"))),
        ),
        "side_effect_after_verified_completion_count": _nested(
            grouped,
            lambda rows: sum(int(row.get("side_effect_after_verified_completion_count", 0) or 0) for row in rows),
        ),
        "repair_dependency_ratio_by_fixture": _nested(
            grouped,
            lambda rows: round(
                sum(float(row.get("repair_dependency_ratio", 0.0) or 0.0) for row in rows) / max(1, len(rows)),
                6,
            ),
        ),
        "competing_hypotheses_created_count": _nested(
            grouped,
            lambda rows: sum(int(row.get("competing_hypotheses_created_count", 0) or 0) for row in rows),
        ),
        "min_hypotheses_before_patch": _nested(
            grouped,
            lambda rows: min([int(row.get("min_hypotheses_before_patch", 0) or 0) for row in rows] or [0]),
        ),
        "discriminating_experiments_bound_to_hypotheses_count": _nested(
            grouped,
            lambda rows: sum(int(row.get("discriminating_experiments_bound_to_hypotheses_count", 0) or 0) for row in rows),
        ),
        "posterior_events_bound_to_hypotheses_count": _nested(
            grouped,
            lambda rows: sum(int(row.get("posterior_events_bound_to_hypotheses_count", 0) or 0) for row in rows),
        ),
        "leading_hypothesis_before_patch_rate": _nested(
            grouped,
            lambda rows: _rate([bool(row.get("leading_hypothesis_before_patch")) for row in rows]),
        ),
        "patch_referenced_hypothesis_rate": _nested(
            grouped,
            lambda rows: _rate([bool(row.get("patch_referenced_hypothesis")) for row in rows]),
        ),
        "hypothesis_lifecycle_complete_rate": _nested(
            grouped,
            lambda rows: _rate([bool(row.get("hypothesis_lifecycle_complete")) for row in rows]),
        ),
        "capability_matrix_by_variant": {
            variant: _capability_matrix_summary(rows)
            for variant, rows in sorted(by_variant.items())
        },
        "llm_call_rate_by_fixture_variant": _nested(
            grouped,
            lambda rows: _rate([int(row.get("llm_call_trace_count", 0) or 0) > 0 for row in rows]),
        ),
        "llm_call_count_by_fixture_variant": _nested(
            grouped,
            lambda rows: sum(int(row.get("llm_call_trace_count", 0) or 0) for row in rows),
        ),
        "llm_token_cost_by_fixture_variant": _nested(grouped, _avg_llm_token_cost),
        "cost_per_verified_success_by_fixture_variant": _nested(grouped, _cost_per_verified_success),
        "amplification_efficiency_token_by_fixture": amplification_efficiency_token_by_fixture,
        "patch_proposal_llm_trace_count_by_fixture_variant": _nested(
            grouped,
            lambda rows: sum(int(row.get("patch_proposal_llm_trace_count", 0) or 0) for row in rows),
        ),
        "llm_call_requirement_failures_by_fixture_variant": _nested(
            grouped,
            lambda rows: sum(
                1
                for row in rows
                if bool(row.get("llm_call_required")) and not bool(row.get("llm_call_required_passed"))
            ),
        ),
        "llm_runtime_by_variant": {
            variant: {
                "providers": sorted({str(row.get("llm_provider") or "") for row in rows}),
                "models": sorted({str(row.get("llm_model") or "") for row in rows}),
                "base_urls": sorted({str(row.get("llm_base_url") or "") for row in rows}),
                "total_llm_call_trace_count": sum(int(row.get("llm_call_trace_count", 0) or 0) for row in rows),
                "llm_call_rate": _rate([int(row.get("llm_call_trace_count", 0) or 0) > 0 for row in rows]),
                "deterministic_fallback_enabled_rate": _rate([bool(row.get("deterministic_fallback_enabled", True)) for row in rows]),
                "prefer_llm_kwargs_rate": _rate([bool(row.get("prefer_llm_kwargs", False)) for row in rows]),
                "prefer_llm_patch_proposals_rate": _rate([bool(row.get("prefer_llm_patch_proposals", False)) for row in rows]),
                "thinking_modes": sorted({str(row.get("llm_thinking_mode") or "") for row in rows}),
            }
            for variant, rows in sorted(by_variant.items())
        },
        "mechanism_path_rate_by_variant": {
            variant: {
                "test_failure_observed": _rate([bool(_as_dict(row.get("mechanism_path")).get("test_failure_observed")) for row in rows]),
                "competing_hypotheses_created": _rate([bool(_as_dict(row.get("mechanism_path")).get("competing_hypotheses_created")) for row in rows]),
                "hypothesis_lifecycle_complete": _rate([bool(_as_dict(row.get("mechanism_path")).get("hypothesis_lifecycle_complete")) for row in rows]),
                "discriminating_evidence_selected": _rate([bool(_as_dict(row.get("mechanism_path")).get("discriminating_evidence_selected")) for row in rows]),
                "hypothesis_pair_distinguished_before_patch": _rate([bool(_as_dict(row.get("mechanism_path")).get("hypothesis_pair_distinguished_before_patch")) for row in rows]),
                "posterior_shift": _rate([bool(_as_dict(row.get("mechanism_path")).get("posterior_shift")) for row in rows]),
                "target_binding": _rate([bool(_as_dict(row.get("mechanism_path")).get("target_binding")) for row in rows]),
                "patch_proposal_generated": _rate([bool(_as_dict(row.get("mechanism_path")).get("patch_proposal_generated")) for row in rows]),
                "verifier_acceptance": _rate([bool(_as_dict(row.get("mechanism_path")).get("verifier_acceptance")) for row in rows]),
                "terminal_completion": _rate([bool(_as_dict(row.get("mechanism_path")).get("terminal_completion")) for row in rows]),
            }
            for variant, rows in sorted(by_variant.items())
        },
        "failure_reason_counts_by_variant": {
            variant: _count_by_key(rows, "failure_reason")
            for variant, rows in sorted(by_variant.items())
        },
        "ablation_contamination_count_by_variant": {
            variant: sum(1 for row in rows if bool(row.get("ablation_contaminated")))
            for variant, rows in sorted(by_variant.items())
        },
        "full_vs_ablation_delta_by_fixture": full_vs_ablation,
        "full_vs_tool_only_greedy_delta": full_vs_tool_only,
        "full_vs_external_baseline_task_delta": full_vs_external_task,
        "full_vs_external_baseline_cognitive_delta": full_vs_external_cognitive,
        "external_baseline_wrong_patch_count": external_wrong_patch,
        "external_baseline_rollback_count": external_rollback,
        "external_baseline_verification_waste_ticks": external_waste,
        "full_vs_ablation_efficiency_delta": full_vs_ablation_efficiency,
        "full_vs_no_discriminating_wrong_patch_delta": full_vs_no_disc_wrong_patch_delta,
        "full_vs_no_discriminating_verification_waste_delta": full_vs_no_disc_verification_waste_delta,
        "runs": [
            {
                "fixture_id": str(row.get("fixture_id") or ""),
                "variant": str(row.get("variant") or ""),
                "repeat": int(row.get("repeat", 0) or 0),
                "max_ticks": int(row.get("max_ticks", row.get("requested_max_ticks", 0)) or 0),
                "success": bool(row.get("success")),
                "task_success": bool(row.get("task_success", row.get("success"))),
                "cognitive_success": bool(row.get("cognitive_success")),
                "needs_human_review": bool(row.get("needs_human_review")),
                "refusal_reason": str(row.get("refusal_reason") or ""),
                "patched_file": str(row.get("patched_file") or ""),
                "true_bug_file": str(row.get("true_bug_file") or ""),
                "posterior_shift": bool(row.get("posterior_shift_from_traceback_file_to_true_bug_file")),
                "target_binding_confidence": float(row.get("target_binding_confidence", 0.0) or 0.0),
                "patch_proposal_generated": int(row.get("patch_proposals_generated", 0) or 0) > 0,
                "patch_proposal_verified": bool(row.get("patch_proposal_verified")),
                "direct_evidence_before_patch": bool(row.get("direct_evidence_before_patch")),
                "wrong_patch_attempt_count": int(row.get("wrong_patch_attempt_count", 0) or 0),
                "rollback_count": int(row.get("rollback_count", row.get("patch_proposal_rollback_count", 0)) or 0),
                "decoy_patch_selected": bool(row.get("decoy_patch_selected")),
                "discriminating_test_selected_before_patch": bool(row.get("discriminating_test_selected_before_patch")),
                "hypothesis_pair_distinguished_before_patch": bool(row.get("hypothesis_pair_distinguished_before_patch")),
                "patch_after_disambiguation": bool(row.get("patch_after_disambiguation")),
                "verification_waste_ticks": int(row.get("verification_waste_ticks", 0) or 0),
                "patch_tick": patch_tick(row),
                "final_verification_tick": final_verification_tick(row),
                "repair_dependency_ratio": float(row.get("repair_dependency_ratio", 0.0) or 0.0),
                "competing_hypotheses_created_count": int(row.get("competing_hypotheses_created_count", 0) or 0),
                "min_hypotheses_before_patch": int(row.get("min_hypotheses_before_patch", 0) or 0),
                "hypothesis_ids_seen": _as_list(row.get("hypothesis_ids_seen")),
                "discriminating_experiments_bound_to_hypotheses_count": int(row.get("discriminating_experiments_bound_to_hypotheses_count", 0) or 0),
                "posterior_events_bound_to_hypotheses_count": int(row.get("posterior_events_bound_to_hypotheses_count", 0) or 0),
                "leading_hypothesis_before_patch": str(row.get("leading_hypothesis_before_patch") or ""),
                "patch_referenced_hypothesis": bool(row.get("patch_referenced_hypothesis")),
                "hypothesis_lifecycle_complete": bool(row.get("hypothesis_lifecycle_complete")),
                "llm_provider": str(row.get("llm_provider") or ""),
                "llm_base_url": str(row.get("llm_base_url") or ""),
                "llm_model": str(row.get("llm_model") or ""),
                "llm_call_trace_count": int(row.get("llm_call_trace_count", 0) or 0),
                "patch_proposal_llm_trace_count": int(row.get("patch_proposal_llm_trace_count", 0) or 0),
                "llm_call_required": bool(row.get("llm_call_required", False)),
                "llm_call_required_passed": bool(row.get("llm_call_required_passed", True)),
                "deterministic_fallback_enabled": bool(row.get("deterministic_fallback_enabled", True)),
                "prefer_llm_kwargs": bool(row.get("prefer_llm_kwargs", False)),
                "prefer_llm_patch_proposals": bool(row.get("prefer_llm_patch_proposals", False)),
                "capability_matrix": _as_dict(row.get("capability_matrix")),
                "mechanism_path": _as_dict(row.get("mechanism_path")),
                "failure_reason": str(row.get("failure_reason") or ""),
                "ablation_contaminated": bool(row.get("ablation_contaminated")),
                "report_path": str(row.get("report_path") or ""),
            }
            for row in suite_reports
        ],
    }


def _load_active_report_paths(summary_path: Path = SUMMARY_PATH) -> list[Path]:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    paths: list[Path] = []
    for raw in _as_list(_as_dict(payload).get("active_report_paths")):
        path = Path(str(raw))
        if path.exists() and path.is_file():
            paths.append(path)
    return paths


def load_reports(report_dir: Path = REPORT_DIR) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    paths = _load_active_report_paths() if Path(report_dir).resolve() == REPORT_DIR.resolve() else []
    if not paths:
        paths = sorted(report_dir.glob("*.json"))
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload.setdefault("report_path", str(path))
            reports.append(payload)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase1C suite reports.")
    parser.add_argument("--reports-dir", default=str(REPORT_DIR))
    parser.add_argument("--output", default=str(SUMMARY_PATH))
    args = parser.parse_args()
    summary = summarize_reports(load_reports(Path(args.reports_dir)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
