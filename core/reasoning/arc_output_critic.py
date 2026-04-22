from __future__ import annotations

from typing import Any, Dict, List, Sequence

from core.orchestration.state_abstraction import (
    score_arc_transition_alignment,
    summarize_arc_transition_profile,
    summarize_grid_state,
)
from core.reasoning.arc_primitives import (
    copy_grid,
    grid_similarity,
    is_grid,
    object_correspondence_score,
    output_stability_score,
)


def _score_key_value(score_key: Sequence[Any], index: int, default: float = 0.0) -> float:
    try:
        return float(score_key[index])
    except (IndexError, TypeError, ValueError):
        return default


def score_arc_candidate_output(
    task_payload: Dict[str, Any],
    candidate_output: Dict[str, Any],
) -> Dict[str, Any]:
    enriched = dict(candidate_output)
    predicted_outputs = [
        copy_grid(grid)
        for grid in list(candidate_output.get("predicted_outputs", []) or [])
        if is_grid(grid)
    ]
    enriched["predicted_outputs"] = predicted_outputs
    if not predicted_outputs:
        enriched["critic_score"] = float(candidate_output.get("score", 0.0) or 0.0)
        enriched["critic_breakdown"] = {}
        return enriched

    train = list(task_payload.get("train", []) or []) if isinstance(task_payload, dict) else []
    test_inputs = list(task_payload.get("test_inputs", []) or []) if isinstance(task_payload, dict) else []
    train_count = max(len(train), 1)
    score_key = list(candidate_output.get("score_key", []) or [])
    baseline_score = float(candidate_output.get("score", 0.0) or 0.0)
    complexity = int(candidate_output.get("program_complexity", candidate_output.get("complexity", 0)) or 0)

    train_exact_ratio = _score_key_value(score_key, 0) / train_count
    holdout_exact_ratio = _score_key_value(score_key, 1) / train_count
    holdout_similarity = _score_key_value(score_key, 3, default=0.5)
    avg_similarity = _score_key_value(score_key, 4, default=0.5)
    avg_transition_alignment = _score_key_value(score_key, 6, default=0.5)

    train_pair_consistency = max(0.0, min(1.0, (train_exact_ratio * 0.55) + (avg_similarity * 0.45)))
    simplicity = 1.0 / (1.0 + max(0, complexity))
    cross_example_stability = max(
        0.0,
        min(1.0, (holdout_exact_ratio * 0.45) + (holdout_similarity * 0.30) + (output_stability_score(predicted_outputs) * 0.25)),
    )

    transition_profile = summarize_arc_transition_profile(train)
    object_scores: List[float] = []
    contextual_scores: List[float] = []
    for input_grid, output_grid in zip(test_inputs, predicted_outputs):
        if not is_grid(input_grid) or not is_grid(output_grid):
            continue
        object_scores.append(object_correspondence_score(input_grid, output_grid))
        contextual_scores.append(score_arc_transition_alignment(input_grid, output_grid, transition_profile))
    object_correspondence = (
        sum(object_scores) / len(object_scores)
        if object_scores
        else max(0.0, min(1.0, avg_similarity))
    )
    contextual_rule_consistency = (
        sum(contextual_scores) / len(contextual_scores)
        if contextual_scores
        else max(0.0, min(1.0, avg_transition_alignment))
    )

    first_output = predicted_outputs[0]
    if is_grid(first_output):
        enriched["first_output_state"] = summarize_grid_state(first_output)

    critic_score = round(
        (baseline_score * 0.40)
        + (train_pair_consistency * 0.20)
        + (simplicity * 0.10)
        + (cross_example_stability * 0.13)
        + (object_correspondence * 0.08)
        + (contextual_rule_consistency * 0.09),
        6,
    )
    enriched["critic_breakdown"] = {
        "train_pair_consistency": round(train_pair_consistency, 6),
        "simplicity": round(simplicity, 6),
        "cross_example_stability": round(cross_example_stability, 6),
        "object_correspondence_preservation": round(object_correspondence, 6),
        "contextual_rule_consistency": round(contextual_rule_consistency, 6),
        "baseline_score": round(baseline_score, 6),
    }
    enriched["critic_score"] = critic_score
    enriched["solver_path"] = "candidate_output_search"
    return enriched


def rank_arc_candidate_outputs(
    task_payload: Dict[str, Any],
    candidate_outputs: Sequence[Dict[str, Any]],
    *,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for row in candidate_outputs:
        if not isinstance(row, dict):
            continue
        ranked.append(score_arc_candidate_output(task_payload, row))
    ranked.sort(
        key=lambda item: (
            float(item.get("critic_score", 0.0) or 0.0),
            float(((item.get("critic_breakdown", {}) if isinstance(item.get("critic_breakdown", {}), dict) else {}).get("train_pair_consistency", 0.0) or 0.0)),
            float(item.get("score", 0.0) or 0.0),
            max(
                (
                    grid_similarity(predicted, target.get("output"))
                    for predicted, target in zip(
                        item.get("predicted_outputs", []),
                        list(task_payload.get("train", []) or []),
                    )
                    if is_grid(predicted) and isinstance(target, dict) and is_grid(target.get("output"))
                ),
                default=0.0,
            ),
        ),
        reverse=True,
    )
    if limit is not None and limit > 0:
        return ranked[:limit]
    return ranked
