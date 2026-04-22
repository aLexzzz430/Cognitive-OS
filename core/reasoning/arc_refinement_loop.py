from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Set

from core.reasoning.arc_output_critic import rank_arc_candidate_outputs
from core.reasoning.arc_program_dsl import normalize_arc_program_rows


PredictOutputsFn = Callable[[Sequence[Dict[str, Any]]], Sequence[Dict[str, Any]]]
RefineProgramsFn = Callable[[Sequence[Dict[str, Any]], Set[str], int], Sequence[Dict[str, Any]]]


@dataclass(frozen=True)
class ArcRefinementConfig:
    program_limit: int = 12
    output_limit: int = 8
    top_k: int = 3
    rounds: int = 1


def _program_name(row: Dict[str, Any]) -> str:
    return str(row.get("name", row.get("program_name", "")) or "")


def _merge_program_rows(
    current: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    merged = normalize_arc_program_rows([*current, *new_rows], limit=limit if limit > 0 else None)
    merged.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    if limit > 0:
        return merged[:limit]
    return merged


def _merge_output_rows(
    task_payload: Dict[str, Any],
    current: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in [*current, *new_rows]:
        if not isinstance(row, dict):
            continue
        key = str(row.get("program_name", row.get("program_id", row.get("output_id", ""))) or "")
        if not key:
            continue
        previous = merged.get(key)
        if previous is None or float(row.get("critic_score", row.get("score", 0.0)) or 0.0) > float(previous.get("critic_score", previous.get("score", 0.0)) or 0.0):
            merged[key] = dict(row)
    ranked = rank_arc_candidate_outputs(task_payload, list(merged.values()), limit=limit if limit > 0 else None)
    if limit > 0:
        return ranked[:limit]
    return ranked


def run_arc_refinement_loop(
    *,
    task_payload: Dict[str, Any],
    base_programs: Sequence[Dict[str, Any]],
    predict_outputs: PredictOutputsFn,
    refine_programs: RefineProgramsFn | None = None,
    config: ArcRefinementConfig | None = None,
) -> Dict[str, Any]:
    active_config = config or ArcRefinementConfig()
    candidate_programs = normalize_arc_program_rows(
        base_programs,
        limit=active_config.program_limit if active_config.program_limit > 0 else None,
    )
    raw_outputs = [
        dict(item)
        for item in predict_outputs(candidate_programs)
        if isinstance(item, dict)
    ]
    candidate_outputs = rank_arc_candidate_outputs(
        task_payload,
        raw_outputs,
        limit=active_config.output_limit if active_config.output_limit > 0 else None,
    )

    rounds_used = 0
    refinement_used = False
    if refine_programs is not None and candidate_programs and active_config.rounds > 0:
        for _ in range(active_config.rounds):
            top_program_names = {
                str(row.get("program_name", "") or "")
                for row in candidate_outputs[: max(1, active_config.top_k)]
                if isinstance(row, dict)
            }
            top_programs = [
                dict(row)
                for row in candidate_programs
                if _program_name(row) in top_program_names
            ][: max(1, active_config.top_k)]
            if not top_programs:
                break
            existing_names = {_program_name(row) for row in candidate_programs}
            refined_programs = normalize_arc_program_rows(
                refine_programs(top_programs, existing_names, active_config.program_limit),
                limit=active_config.program_limit if active_config.program_limit > 0 else None,
            )
            refined_programs = [
                row
                for row in refined_programs
                if _program_name(row) not in existing_names
            ]
            if not refined_programs:
                break
            candidate_programs = _merge_program_rows(
                candidate_programs,
                refined_programs,
                limit=active_config.program_limit,
            )
            refined_outputs = [
                dict(item)
                for item in predict_outputs(refined_programs)
                if isinstance(item, dict)
            ]
            candidate_outputs = _merge_output_rows(
                task_payload,
                candidate_outputs,
                refined_outputs,
                limit=active_config.output_limit,
            )
            rounds_used += 1
            refinement_used = True

    selected_output = candidate_outputs[0] if candidate_outputs else {}
    selected_program_name = str(selected_output.get("program_name", "") or "")
    selected_program = next(
        (dict(row) for row in candidate_programs if _program_name(row) == selected_program_name),
        dict(candidate_programs[0]) if candidate_programs else {},
    )
    return {
        "candidate_programs": candidate_programs,
        "candidate_outputs": candidate_outputs,
        "selected_program": selected_program,
        "selected_output": selected_output,
        "refinement_rounds_used": rounds_used,
        "used_refinement": refinement_used,
        "solver_path": "candidate_output_search",
    }
