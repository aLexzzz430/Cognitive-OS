from __future__ import annotations

import re
from typing import Any, Dict, List


def extract_uncertain_high_impact_beliefs(world_model_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(world_model_summary, dict):
        return []
    beliefs = world_model_summary.get('beliefs', {})
    high_value_beliefs = world_model_summary.get('high_value_beliefs', [])
    impact_by_variable = {}
    for item in high_value_beliefs if isinstance(high_value_beliefs, list) else []:
        if not isinstance(item, dict):
            continue
        variable = str(item.get('variable', '') or '').strip()
        if not variable:
            continue
        impact_scope = str(item.get('impact_scope', '') or '')
        impact_by_variable[variable] = 1.0 if impact_scope == 'planner+decision' else 0.7

    candidates: List[Dict[str, Any]] = []
    for variable, payload in beliefs.items() if isinstance(beliefs, dict) else []:
        if not isinstance(payload, dict):
            continue
        confidence = max(0.0, min(1.0, float(payload.get('confidence', 0.0) or 0.0)))
        uncertainty = 1.0 - confidence
        impact = impact_by_variable.get(str(variable), 0.65 if confidence < 0.6 else 0.45)
        if confidence > 0.72 or impact < 0.6:
            continue
        candidates.append({
            'variable': str(variable),
            'confidence': confidence,
            'impact': impact,
            'uncertainty': uncertainty,
            'priority': uncertainty * impact,
        })
    candidates.sort(key=lambda item: item.get('priority', 0.0), reverse=True)
    return candidates[:6]


def estimate_probe_disambiguation_gain(probe: Any, uncertain_beliefs: List[Dict[str, Any]]) -> float:
    if not uncertain_beliefs:
        return 0.0
    probe_tokens = set(re.findall(r"[a-z0-9_]+", str(getattr(probe, 'target_function', '') or '').lower()))
    test_params = getattr(probe, 'test_params', {})
    if isinstance(test_params, dict):
        probe_tokens.update(re.findall(r"[a-z0-9_]+", str(test_params).lower()))
    hyp_tokens = f"{getattr(probe, 'hypothesis_a', '')} {getattr(probe, 'hypothesis_b', '')}".lower()
    score = 0.0
    for belief in uncertain_beliefs:
        variable = str(belief.get('variable', '') or '').lower()
        variable_tokens = set(re.findall(r"[a-z0-9_]+", variable))
        if not variable_tokens:
            continue
        overlap = len(variable_tokens & probe_tokens)
        semantic_bonus = 0.15 if variable and variable in hyp_tokens else 0.0
        alignment = min(1.0, (overlap / max(1, len(variable_tokens))) + semantic_bonus)
        score += alignment * float(belief.get('priority', 0.0) or 0.0)
    return max(0.0, min(1.0, score))


def annotate_probe_ranking(probe: Any, details: Dict[str, Any]) -> None:
    if not isinstance(details, dict):
        return
    try:
        setattr(probe, 'ranking_details', details)
        setattr(probe, 'expected_information_gain', float(details.get('score', 0.0) or 0.0))
    except Exception:
        pass


def rank_probe_candidates_by_prediction(
    loop: Any,
    probe_candidates: List[Any],
    obs_before: Dict[str, Any],
    surfaced: List[Any],
    frame: Any,
) -> List[Any]:
    if not probe_candidates or not loop._prediction_runtime_active():
        return probe_candidates

    world_model_summary = getattr(frame, 'world_model_summary', {}) if frame is not None else {}
    self_model_summary = getattr(frame, 'self_model_summary', {}) if frame is not None else {}
    uncertain_beliefs = extract_uncertain_high_impact_beliefs(world_model_summary)
    uncertainty_focus = (
        max(0.0, min(1.0, sum(item.get('uncertainty', 0.0) for item in uncertain_beliefs[:3]) / 3.0))
        if uncertain_beliefs
        else 0.0
    )

    ranked = []
    for probe in probe_candidates:
        action = {
            'kind': 'probe',
            '_source': 'probe_gate',
            'payload': {
                'tool_args': {
                    'function_name': getattr(probe, 'target_function', 'wait'),
                    'kwargs': getattr(probe, 'test_params', {}),
                }
            },
        }
        loop._build_action_id(action)
        bundle = loop._prediction_engine.predict_action(
            episode=loop._episode,
            tick=loop._tick,
            action=action,
            obs=obs_before,
            surfaced=surfaced,
            hypotheses=loop._hypotheses.get_active(),
            belief_summary=world_model_summary,
            plan_summary=loop._plan_state.get_plan_summary(),
            step_intent=loop._plan_state.get_intent_for_step(),
            recent_trace=loop._episode_trace[-5:],
            self_model_summary=self_model_summary,
            policy_profile=loop._get_policy_profile(),
            recovery_context=loop._build_recovery_prediction_context(),
        )
        prediction_info_gain = float(bundle.information_gain.value)
        disambiguation_gain = estimate_probe_disambiguation_gain(probe, uncertain_beliefs)
        prediction_weight = 0.55 + (0.25 * uncertainty_focus)
        uncertainty_weight = 1.0 - prediction_weight
        final_score = (prediction_info_gain * prediction_weight) + (disambiguation_gain * uncertainty_weight)
        annotate_probe_ranking(
            probe,
            {
                'prediction_info_gain': round(prediction_info_gain, 4),
                'disambiguation_gain': round(disambiguation_gain, 4),
                'uncertainty_focus': round(uncertainty_focus, 4),
                'score': round(final_score, 4),
                'target_beliefs': [item.get('variable') for item in uncertain_beliefs[:3]],
            },
        )
        ranked.append((final_score, probe))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [probe for _, probe in ranked]
