from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _generic_action_family(function_name: str) -> str:
    fn = str(function_name or "").strip().lower()
    if any(token in fn for token in ("inspect", "probe", "check", "verify", "test")):
        return "probe_interaction"
    if any(token in fn for token in ("confirm", "commit", "seal", "finalize", "submit")):
        return "confirm_interaction"
    if any(token in fn for token in ("move", "advance", "navigate", "step")):
        return "navigation_interaction"
    return "pointer_interaction"


def _phase_mechanism_family(target_phase: str) -> str:
    phase = str(target_phase or "").strip().lower()
    if phase == "committed":
        return "phase_commit_transition"
    if phase == "stabilizing":
        return "phase_stabilization"
    if phase == "disrupted":
        return "instability_trigger"
    return "phase_exploration"


def build_mechanism_graph(
    world_model_summary: Dict[str, Any],
    *,
    object_graph: Optional[Dict[str, Any]] = None,
    recent_trace: Optional[Sequence[Dict[str, Any]]] = None,
    limit: int = 4,
) -> Dict[str, Any]:
    summary = dict(world_model_summary or {})
    existing_rows = [
        dict(row)
        for row in _as_list(summary.get("mechanism_hypotheses_summary", summary.get("mechanism_hypotheses", [])))
        if isinstance(row, dict)
    ]
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(existing_rows[: max(0, int(limit))]):
        confidence = _clamp01(row.get("confidence", row.get("base_confidence", 0.0)), 0.0)
        preferred_action_families = [
            str(item or "")
            for item in _as_list(row.get("preferred_action_families", []))
            if str(item or "")
        ]
        discriminating_actions = [
            str(item or "")
            for item in _as_list(row.get("best_discriminating_actions", []))
            if str(item or "")
        ]
        rows.append({
            "mechanism_id": str(row.get("mechanism_id", row.get("id", f"mechanism_{index}")) or f"mechanism_{index}"),
            "hypothesis_id": str(row.get("hypothesis_id", row.get("mechanism_id", row.get("id", f"mechanism_{index}"))) or f"mechanism_{index}"),
            "family": str(row.get("family", "generic_mechanism") or "generic_mechanism"),
            "confidence": confidence,
            "expected_transition": str(row.get("expected_transition", row.get("predicted_success_condition", "")) or ""),
            "preferred_action_families": preferred_action_families,
            "best_discriminating_actions": discriminating_actions,
            "preferred_target_refs": [
                str(item or "")
                for item in _as_list(row.get("preferred_target_refs", []))
                if str(item or "")
            ][:4],
            "expected_information_gain": _clamp01(row.get("expected_information_gain", row.get("compression_gain", confidence * 0.5)), 0.0),
            "status": str(row.get("status", "candidate") or "candidate"),
        })

    if not rows:
        hidden_state = _as_dict(summary.get("hidden_state", {}))
        latent_branches = _as_list(summary.get("latent_branches", hidden_state.get("latent_branches", [])))
        intervention_targets = []
        if isinstance(object_graph, dict):
            intervention_targets = [
                str(item or "")
                for item in _as_list(object_graph.get("intervention_targets", []))
                if str(item or "")
            ]
        for index, branch in enumerate(latent_branches[: max(0, int(limit))]):
            if not isinstance(branch, dict):
                continue
            target_phase = str(branch.get("target_phase", branch.get("current_phase", "exploring")) or "exploring")
            anchor_functions = [
                str(item or "")
                for item in _as_list(branch.get("anchor_functions", []))
                if str(item or "")
            ][:4]
            action_families = list(dict.fromkeys(_generic_action_family(item) for item in anchor_functions if item))
            rows.append({
                "mechanism_id": str(branch.get("branch_id", f"branch_mechanism_{index}") or f"branch_mechanism_{index}"),
                "hypothesis_id": str(branch.get("branch_id", f"branch_mechanism_{index}") or f"branch_mechanism_{index}"),
                "family": _phase_mechanism_family(target_phase),
                "confidence": _clamp01(branch.get("confidence", 0.0), 0.0),
                "expected_transition": f"{branch.get('current_phase', 'exploring')} -> {target_phase}",
                "preferred_action_families": action_families,
                "best_discriminating_actions": anchor_functions[:2],
                "preferred_target_refs": intervention_targets[:3],
                "expected_information_gain": _clamp01(branch.get("transition_score", branch.get("support", 0.0)), 0.0),
                "status": "candidate",
            })

    rows.sort(
        key=lambda item: (
            -float(item.get("confidence", 0.0) or 0.0),
            -float(item.get("expected_information_gain", 0.0) or 0.0),
            str(item.get("mechanism_id", "") or ""),
        )
    )
    rows = rows[: max(0, int(limit))]
    dominant = rows[0] if rows else {}
    preferred_action_families: List[str] = []
    discriminating_actions: List[str] = []
    mechanism_families: List[str] = []
    for row in rows:
        family = str(row.get("family", "") or "")
        if family and family not in mechanism_families:
            mechanism_families.append(family)
        for item in _as_list(row.get("preferred_action_families", [])):
            item = str(item or "")
            if item and item not in preferred_action_families:
                preferred_action_families.append(item)
        for item in _as_list(row.get("best_discriminating_actions", [])):
            item = str(item or "")
            if item and item not in discriminating_actions:
                discriminating_actions.append(item)
    return {
        "mechanism_hypotheses": rows,
        "mechanism_families": mechanism_families,
        "dominant_mechanism_family": str(dominant.get("family", "") or ""),
        "dominant_mechanism_confidence": _clamp01(dominant.get("confidence", 0.0), 0.0),
        "preferred_action_families": preferred_action_families[:4],
        "discriminating_actions": discriminating_actions[:4],
    }
