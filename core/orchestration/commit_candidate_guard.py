from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.orchestration.action_utils import extract_action_function_name, extract_action_kind
from modules.world_model.mechanism_runtime import mechanism_obs_state


_PROBE_TOKENS = ("probe", "inspect", "verify", "check", "test")
_PROCEDURE_GUARD_REASONS = frozenset(
    {
        "latent_procedure_next_step",
        "procedure_next_step",
    }
)
_COMMIT_GUARD_PRIORITIES = {
    "structured_answer": 50,
    "latent_procedure_next_step": 40,
    "procedure_next_step": 30,
    "plan_target_surface": 20,
    "single_visible_surface": 10,
}


def procedure_guard_reasons() -> Set[str]:
    return set(_PROCEDURE_GUARD_REASONS)


def commit_guard_priority(reason: str) -> int:
    return int(_COMMIT_GUARD_PRIORITIES.get(str(reason or "").strip(), 0))


def normalize_available_functions(available_functions: Iterable[Any]) -> Set[str]:
    normalized: Set[str] = set()
    for fn_name in available_functions or ():
        text = str(fn_name or "").strip()
        if text:
            normalized.add(text)
    return normalized


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _mechanism_runtime_annotation(
    action: Any,
    *,
    mechanism_control_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {
            "wait_ready": False,
            "prerequisite_ready": False,
            "recovery_ready": False,
            "release_ready": False,
            "binding_actionable": False,
            "mode_alignment": 0.0,
            "runtime_discriminating_candidate": False,
            "has_runtime": bool(isinstance(mechanism_control_summary, dict) and mechanism_control_summary),
        }

    meta = _as_dict(action.get("_candidate_meta", {}))
    role = str(meta.get("role", "") or "").strip().lower()
    control_summary = _as_dict(mechanism_control_summary)
    obs_state = mechanism_obs_state({}, control_summary) if control_summary else {}
    release_ready = bool(meta.get("mechanism_release_ready", obs_state.get("release_ready", False)))
    wait_ready = bool(meta.get("mechanism_wait_ready", obs_state.get("wait_ready", False)))
    prerequisite_ready = bool(meta.get("mechanism_prerequisite_ready", obs_state.get("prerequisite_ready", False)))
    recovery_ready = bool(meta.get("mechanism_recovery_ready", obs_state.get("recovery_ready", False)))
    precondition_satisfied = bool(meta.get("mechanism_precondition_satisfied", False))
    binding_actionable = bool(meta.get("mechanism_binding_actionable", False))
    if not binding_actionable and role == "commit" and release_ready and precondition_satisfied:
        binding_actionable = True
    mode_alignment = float(meta.get("mechanism_mode_alignment", 0.0) or 0.0)
    if mode_alignment <= 0.0:
        if role == "wait" and wait_ready:
            mode_alignment = 1.0
        elif role in {"prerequisite", "prepare"} and prerequisite_ready:
            mode_alignment = 1.0
        elif role == "recovery" and recovery_ready:
            mode_alignment = 1.0
        elif role == "commit" and binding_actionable:
            mode_alignment = 1.0
    has_runtime = any(
        key in meta
        for key in (
            "mechanism_release_ready",
            "mechanism_binding_actionable",
            "mechanism_mode_alignment",
            "mechanism_wait_ready",
            "mechanism_prerequisite_ready",
            "mechanism_recovery_ready",
            "mechanism_precondition_satisfied",
        )
    ) or bool(control_summary)
    return {
        "wait_ready": wait_ready,
        "prerequisite_ready": prerequisite_ready,
        "recovery_ready": recovery_ready,
        "release_ready": release_ready,
        "binding_actionable": binding_actionable,
        "mode_alignment": float(mode_alignment),
        "runtime_discriminating_candidate": bool(meta.get("runtime_discriminating_candidate", False)),
        "has_runtime": bool(has_runtime),
    }


def is_probe_like(function_name: str, *, kind: str = "") -> bool:
    text = str(function_name or "").strip().lower()
    action_kind = str(kind or "").strip().lower()
    if action_kind == "probe":
        return True
    if not text:
        return False
    return any(token in text for token in _PROBE_TOKENS)


def visible_non_probe_functions(available_functions: Iterable[Any]) -> Set[str]:
    return {
        fn_name
        for fn_name in normalize_available_functions(available_functions)
        if not is_probe_like(fn_name)
    }


def high_confidence_commit_guard_reason(
    action: Any,
    *,
    available_functions: Iterable[Any] = (),
    plan_target_function: str = "",
) -> str:
    if not isinstance(action, dict):
        return ""

    fn_name = extract_action_function_name(action, default="").strip()
    kind = extract_action_kind(action, default="call_tool")
    if not fn_name or fn_name == "wait" or is_probe_like(fn_name, kind=kind):
        return ""

    visible_non_probe = visible_non_probe_functions(available_functions)
    if visible_non_probe and fn_name not in visible_non_probe:
        return ""

    meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
    procedure = meta.get("procedure", {}) if isinstance(meta.get("procedure", {}), dict) else {}
    procedure_guidance = meta.get("procedure_guidance", {}) if isinstance(meta.get("procedure_guidance", {}), dict) else {}
    support_sources = {
        str(item or "").strip()
        for item in list(meta.get("support_sources", []) or [])
        if str(item or "").strip()
    }
    source = str(action.get("_source", "") or "").strip()

    structured_solver_path = str(meta.get("structured_answer_solver_path", "") or "")
    structured_fallback_used = bool(meta.get("structured_answer_fallback_used", False))
    structured_output_count = int(meta.get("structured_answer_candidate_output_count", 0) or 0)
    structured_program_count = int(meta.get("structured_answer_candidate_program_count", 0) or 0)
    if (
        bool(meta.get("structured_answer_synthesized", False))
        and not structured_fallback_used
        and structured_solver_path != "baseline_fallback"
        and (structured_output_count > 0 or structured_program_count > 0)
    ):
        return "structured_answer"

    is_next_step = bool(procedure.get("is_next_step", False) or procedure_guidance.get("active_next_step", False))
    mapping_confidence = float(procedure.get("mapping_confidence", 0.0) or 0.0)
    family_binding_confidence = float(procedure.get("family_binding_confidence", 0.0) or 0.0)
    procedure_bonus = float(procedure.get("procedure_bonus", 0.0) or 0.0)
    alignment_strength = float(procedure_guidance.get("alignment_strength", 0.0) or 0.0)
    hit_source = str(procedure.get("hit_source", "") or "").strip()
    support_count = int(procedure.get("support_count", 0) or 0)
    role_sequence = [
        str(role_name or "").strip().lower()
        for role_name in (procedure.get("role_sequence", []) if isinstance(procedure.get("role_sequence", []), list) else [])
        if str(role_name or "").strip()
    ]
    compressed_latent_family = (
        hit_source == "latent_mechanism_abstraction"
        and support_count >= 3
        and 2 <= len(role_sequence) <= 2
    )
    if is_next_step:
        if hit_source == "latent_mechanism_abstraction":
            latent_binding_strength = (mapping_confidence * 0.6) + (family_binding_confidence * 0.4)
            if (
                (mapping_confidence >= 0.75 and family_binding_confidence >= 0.60)
                or (alignment_strength >= 0.80 and latent_binding_strength >= 0.62)
                or (procedure_bonus >= 0.18 and latent_binding_strength >= 0.60)
                or (
                    compressed_latent_family
                    and mapping_confidence >= 0.30
                    and family_binding_confidence >= 0.45
                    and procedure_bonus >= 0.16
                )
            ):
                return "latent_procedure_next_step"
        elif (
            mapping_confidence >= 0.80
            or alignment_strength >= 0.80
            or procedure_bonus >= 0.18
            or "procedure_reuse" in support_sources
        ):
            return "procedure_next_step"

    plan_target = str(plan_target_function or "").strip()
    if plan_target and fn_name == plan_target:
        return "plan_target_surface"

    if len(visible_non_probe) == 1 and fn_name in visible_non_probe:
        return "single_visible_surface"

    return ""


def _action_guard_strength(action: Any, reason: str) -> float:
    if not isinstance(action, dict):
        return 0.0
    meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
    procedure = meta.get("procedure", {}) if isinstance(meta.get("procedure", {}), dict) else {}
    procedure_guidance = meta.get("procedure_guidance", {}) if isinstance(meta.get("procedure_guidance", {}), dict) else {}
    if reason == "structured_answer":
        output_count = float(meta.get("structured_answer_candidate_output_count", 0) or 0.0)
        program_count = float(meta.get("structured_answer_candidate_program_count", 0) or 0.0)
        return min(1.0, 0.40 + min(0.40, (output_count + program_count) * 0.04))
    if reason in procedure_guard_reasons():
        mapping_confidence = float(procedure.get("mapping_confidence", 0.0) or 0.0)
        family_binding_confidence = float(procedure.get("family_binding_confidence", 0.0) or 0.0)
        alignment_strength = float(procedure_guidance.get("alignment_strength", 0.0) or 0.0)
        procedure_bonus = min(1.0, float(procedure.get("procedure_bonus", 0.0) or 0.0) * 4.0)
        return max(mapping_confidence, family_binding_confidence, alignment_strength, procedure_bonus)
    if reason == "plan_target_surface":
        return 0.55
    if reason == "single_visible_surface":
        return 0.45
    return 0.0


def collect_high_confidence_commit_candidates(
    actions: Sequence[Any],
    *,
    available_functions: Iterable[Any] = (),
    plan_target_function: str = "",
) -> List[Dict[str, Any]]:
    guarded: List[Dict[str, Any]] = []
    for idx, action in enumerate(actions or ()):
        reason = high_confidence_commit_guard_reason(
            action,
            available_functions=available_functions,
            plan_target_function=plan_target_function,
        )
        if not reason:
            continue
        guarded.append(
            {
                "index": idx,
                "action": action,
                "reason": reason,
                "priority": commit_guard_priority(reason),
                "strength": _action_guard_strength(action, reason),
            }
        )
    guarded.sort(
        key=lambda row: (
            int(row.get("priority", 0) or 0),
            float(row.get("strength", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return guarded


def select_high_confidence_commit_candidate(
    actions: Sequence[Any],
    *,
    available_functions: Iterable[Any] = (),
    plan_target_function: str = "",
) -> Tuple[Optional[int], Any, str]:
    guarded = collect_high_confidence_commit_candidates(
        actions,
        available_functions=available_functions,
        plan_target_function=plan_target_function,
    )
    if not guarded:
        return None, None, ""
    best = guarded[0]
    return int(best.get("index", 0) or 0), best.get("action"), str(best.get("reason", "") or "")


def should_override_selected_action_with_commit_guard(
    selected_action: Any,
    override_reason: str,
    *,
    guarded_action: Any = None,
    available_functions: Iterable[Any] = (),
    plan_target_function: str = "",
    mechanism_control_summary: Optional[Dict[str, Any]] = None,
) -> bool:
    override_priority = commit_guard_priority(override_reason)
    if override_priority <= 0:
        return False

    selected_reason = high_confidence_commit_guard_reason(
        selected_action,
        available_functions=available_functions,
        plan_target_function=plan_target_function,
    )
    if commit_guard_priority(selected_reason) >= override_priority:
        return False

    if not isinstance(selected_action, dict):
        return True

    kind = extract_action_kind(selected_action, default="call_tool")
    meta = _as_dict(selected_action.get("_candidate_meta", {}))
    role = str(meta.get("role", "") or "").strip().lower()
    selected_runtime = _mechanism_runtime_annotation(
        selected_action,
        mechanism_control_summary=mechanism_control_summary,
    )
    mechanism_wait_ready = bool(selected_runtime.get("wait_ready", False))
    mechanism_prerequisite_ready = bool(selected_runtime.get("prerequisite_ready", False))
    mechanism_recovery_ready = bool(selected_runtime.get("recovery_ready", False))
    mechanism_release_ready = bool(selected_runtime.get("release_ready", False))
    mechanism_binding_actionable = bool(selected_runtime.get("binding_actionable", False))
    mechanism_mode_alignment = float(selected_runtime.get("mode_alignment", 0.0) or 0.0)
    runtime_discriminating_candidate = bool(selected_runtime.get("runtime_discriminating_candidate", False))

    guarded_runtime = _mechanism_runtime_annotation(
        guarded_action,
        mechanism_control_summary=mechanism_control_summary,
    )
    guarded_release_ready = bool(guarded_runtime.get("release_ready", False))
    guarded_actionable = bool(guarded_runtime.get("binding_actionable", False))
    guarded_has_mechanism_runtime = bool(guarded_runtime.get("has_runtime", False))

    # When the mechanism controller has already identified a critical wait /
    # prerequisite move, don't let a generic commit
    # guard snap us back to a surface-level commit.
    if kind == "wait" and mechanism_wait_ready:
        return False
    if role in {"prerequisite", "prepare"} and mechanism_prerequisite_ready:
        return False
    if role == "commit" and override_reason in {"plan_target_surface", "single_visible_surface"}:
        if mechanism_release_ready or mechanism_binding_actionable or mechanism_mode_alignment >= 0.9:
            return False
    if role in {"wait", "prerequisite", "prepare", "recovery"} and override_reason in {"plan_target_surface", "single_visible_surface"}:
        return False
    if kind == "wait" and override_reason in {"plan_target_surface", "single_visible_surface"}:
        return False
    if kind in {"wait", "probe"}:
        return True

    procedure_guidance = meta.get("procedure_guidance", {}) if isinstance(meta.get("procedure_guidance", {}), dict) else {}
    source = str(selected_action.get("_source", "") or "").strip()
    if guarded_has_mechanism_runtime and not guarded_release_ready and not guarded_actionable:
        return False
    if bool(meta.get("planner_namespace_mismatch", False)):
        return True
    if bool(meta.get("synthetic_support")):
        return True
    if bool(procedure_guidance.get("conflicts_active_procedure", False)):
        return True
    if source in {"surface_generation", "deliberation_probe"}:
        return True
    return False
