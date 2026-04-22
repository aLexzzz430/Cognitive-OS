from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Tuple

from core.learning import OutcomeSignal
from core.orchestration.runtime_stage_contracts import ApplyLearningUpdatesInput
from modules.governance.object_store import ACCEPT_NEW, MERGE_UPDATE_EXISTING


def _aggregate_learning_assignments(
    assignments: List[Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    synthetic_assignments: List[Any] = []
    for assignment in assignments:
        object_id = str(getattr(assignment, "object_id", "") or "")
        if not object_id:
            continue
        if object_id.startswith("action_"):
            synthetic_assignments.append(assignment)
            continue
        bucket = aggregated.setdefault(
            object_id,
            {"credit": 0.0, "evidence": set(), "based_on": set()},
        )
        assignment_credit = float(getattr(assignment, "credit_amount", 0.0) or 0.0)
        if abs(assignment_credit) >= abs(float(bucket["credit"])):
            bucket["credit"] = assignment_credit
        for trace_id in list(getattr(assignment, "evidence_trace_ids", []) or []):
            if trace_id:
                bucket["evidence"].add(str(trace_id))
        based_on = getattr(assignment, "based_on", "")
        if based_on:
            bucket["based_on"].add(str(based_on))
    return aggregated, synthetic_assignments


def _apply_confidence_update(
    loop: Any,
    *,
    object_id: str,
    payload: Dict[str, Any],
    obj: Dict[str, Any],
) -> None:
    current_confidence = float(obj.get("confidence", 0.5) or 0.5)
    credit = max(-1.0, min(1.0, float(payload["credit"])))
    evidence = sorted(payload["evidence"])
    if credit >= 0.0:
        target_confidence = min(
            1.0,
            max(current_confidence, current_confidence * 0.75 + credit * 0.25),
        )
    else:
        negative_signal = abs(credit)
        target_confidence = max(
            0.05,
            min(current_confidence, current_confidence * 0.7 + (1.0 - negative_signal) * 0.3),
        )
    delta = target_confidence - current_confidence
    if abs(delta) < 0.02 or loop._learning_budget_remaining() <= 0:
        return
    result = loop._update_engine.apply_object_update(
        object_id=object_id,
        patch={
            "confidence": target_confidence,
            "memory_metadata": {
                "learning_update": {
                    "reason": "confidence_adjust",
                    "credit": credit,
                    "episode": int(loop._episode),
                    "tick": int(loop._tick),
                    "based_on": sorted(payload["based_on"]),
                }
            },
        },
        reason="learning_confidence_adjust",
        evidence_ids=evidence,
    )
    loop._learning_update_log.append(
        {
            "episode": loop._episode,
            "tick": loop._tick,
            "object_id": object_id,
            "update_type": "confidence_adjust",
            "from_confidence": current_confidence,
            "to_confidence": target_confidence,
            "credit": credit,
            "result": str(result.get("result", "unknown")),
        }
    )
    if result.get("updated"):
        loop._learning_updates_sent_this_episode += 1


def _apply_retrieval_tag_update(
    loop: Any,
    *,
    object_id: str,
    payload: Dict[str, Any],
    obj: Dict[str, Any],
) -> None:
    credit = max(-1.0, min(1.0, float(payload["credit"])))
    if credit < 0.75 or loop._learning_budget_remaining() <= 0:
        return
    existing_tags = set(obj.get("retrieval_tags", []) if isinstance(obj, dict) else [])
    add_tags = [
        tag for tag in ["high_credit", "trace_backed"] if tag not in existing_tags
    ]
    if not add_tags:
        return
    result = loop._update_engine.apply_object_update(
        object_id=object_id,
        patch={
            "retrieval_tags": add_tags,
            "memory_metadata": {
                "learning_update": {
                    "reason": "retrieval_tag_add",
                    "credit": credit,
                    "episode": int(loop._episode),
                    "tick": int(loop._tick),
                    "based_on": sorted(payload["based_on"]),
                }
            },
        },
        reason="learning_retrieval_tag_add",
        evidence_ids=sorted(payload["evidence"]),
    )
    loop._learning_update_log.append(
        {
            "episode": loop._episode,
            "tick": loop._tick,
            "object_id": object_id,
            "update_type": "retrieval_tag_add",
            "add_tags": add_tags,
            "credit": credit,
            "result": str(result.get("result", "unknown")),
        }
    )
    if result.get("updated"):
        loop._learning_updates_sent_this_episode += 1


def run_apply_learning_updates(loop: Any, stage_input: ApplyLearningUpdatesInput) -> Dict[str, Any]:
    assignments = stage_input.assignments
    start_count = len(loop._learning_update_log)
    if not getattr(loop, "_learning_enabled", False):
        return {"updates_logged": 0}

    aggregated, synthetic_assignments = _aggregate_learning_assignments(assignments)
    for object_id, payload in aggregated.items():
        if loop._learning_budget_remaining() <= 0:
            break
        obj = loop._shared_store.get(object_id)
        if not obj:
            continue
        _apply_confidence_update(loop, object_id=object_id, payload=payload, obj=obj)
        _apply_retrieval_tag_update(loop, object_id=object_id, payload=payload, obj=obj)

    for assignment in synthetic_assignments:
        if loop._learning_budget_remaining() <= 0:
            break
        credit = float(getattr(assignment, "credit_amount", 0.0) or 0.0)
        if credit < 0.85:
            continue
        object_id = str(getattr(assignment, "object_id", ""))
        loop._learning_update_log.append(
            {
                "episode": loop._episode,
                "tick": loop._tick,
                "object_id": object_id,
                "update_type": "new_memory_create",
                "credit": credit,
                "result": "deferred_unimplemented",
            }
        )
    return {"updates_logged": len(loop._learning_update_log) - start_count}


def annotate_candidates_with_learning_updates(
    loop: Any,
    candidate_actions: List[Dict[str, Any]],
    decision_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not getattr(loop, "_learning_enabled", False):
        return candidate_actions
    if not isinstance(loop._learning_policy_snapshot, dict):
        loop._refresh_learning_policy_snapshot()
    snapshot = (
        loop._learning_policy_snapshot
        if isinstance(loop._learning_policy_snapshot, dict)
        else {}
    )
    selector_bias = (
        snapshot.get("selector_bias", {})
        if isinstance(snapshot.get("selector_bias", {}), dict)
        else {}
    )
    agenda_prior = (
        snapshot.get("agenda_prior", {})
        if isinstance(snapshot.get("agenda_prior", {}), dict)
        else {}
    )
    failure_preference_policy = (
        snapshot.get("failure_preference_policy", {})
        if isinstance(snapshot.get("failure_preference_policy", {}), dict)
        else {}
    )
    retention_failure_policy = (
        snapshot.get("retention_failure_policy", {})
        if isinstance(snapshot.get("retention_failure_policy", {}), dict)
        else {}
    )
    continuity_snapshot = (
        decision_context.get("continuity_snapshot", {})
        if isinstance(decision_context.get("continuity_snapshot", {}), dict)
        else (decision_context if isinstance(decision_context, dict) else {})
    )
    perception_summary = (
        decision_context.get("perception_summary", {})
        if isinstance(decision_context.get("perception_summary", {}), dict)
        else {}
    )
    if not perception_summary and isinstance(decision_context.get("perception", {}), dict):
        perception_summary = dict(decision_context.get("perception", {}))
    world_model_summary = (
        decision_context.get("world_model_summary", {})
        if isinstance(decision_context.get("world_model_summary", {}), dict)
        else {}
    )
    if not world_model_summary and isinstance(decision_context.get("world_model", {}), dict):
        world_model_summary = dict(decision_context.get("world_model", {}))
    task_family = str(
        continuity_snapshot.get("task_family", decision_context.get("task_family", "unknown"))
        if isinstance(continuity_snapshot, dict)
        else decision_context.get("task_family", "unknown")
    )
    phase = str(
        decision_context.get(
            "phase",
            world_model_summary.get(
                "current_phase",
                continuity_snapshot.get("phase", "active")
                if isinstance(continuity_snapshot, dict)
                else "active",
            ),
        )
        or "active"
    )
    observation_mode = str(perception_summary.get("coordinate_type", "unknown") or "unknown")
    resource_band = loop._learning_resource_band(decision_context)
    agenda_delta = float(agenda_prior.get(str(task_family), 0.0) or 0.0)
    context_key = (
        f"task_family={str(task_family or 'unknown')}|phase={phase}"
        f"|observation_mode={observation_mode}|resource_band={resource_band}"
    )
    failure_bucket = (
        dict(failure_preference_policy.get(context_key, {}))
        if isinstance(failure_preference_policy.get(context_key, {}), dict)
        else {}
    )
    retention_matches = [
        dict(bucket)
        for bucket in retention_failure_policy.values()
        if isinstance(bucket, dict)
        and str(bucket.get("base_context_key", "") or "") == context_key
    ]
    dominant_retention = max(
        retention_matches,
        key=lambda bucket: (
            float(bucket.get("severity", 0.0) or 0.0),
            float(bucket.get("confidence", 0.0) or 0.0),
            abs(float(bucket.get("delta", 0.0) or 0.0)),
        ),
        default={},
    )
    candidate_function_universe = [
        loop._extract_action_function_name(action, default="wait")
        for action in candidate_actions
        if loop._extract_action_function_name(action, default="wait") != "wait"
    ]
    competition = loop._learning_world_model_competition_profile(
        decision_context,
        candidate_function_universe=list(dict.fromkeys(candidate_function_universe)),
    )
    required_probes = set(competition.get("required_probes", []) or [])
    dominant_anchor_functions = set(competition.get("dominant_anchor_functions", []) or [])
    dominant_risky_functions = set(competition.get("dominant_risky_functions", []) or [])

    for action in candidate_actions:
        fn_name = loop._extract_action_function_name(action, default="wait")
        fn_delta = float(selector_bias.get(str(fn_name), 0.0) or 0.0)
        action_meta = (
            action.get("_candidate_meta", {})
            if isinstance(action.get("_candidate_meta", {}), dict)
            else {}
        )
        candidate_probe_aliases = set()
        for value in list(action_meta.get("world_model_probe_aliases", []) or []):
            text_value = str(value or "").strip()
            if text_value:
                candidate_probe_aliases.add(text_value)
        for key in ("action_family", "target_family", "surface_click_role"):
            text_value = str(action_meta.get(key, "") or "").strip()
            if text_value:
                candidate_probe_aliases.add(text_value)
        candidate_required_probe_match = bool(
            fn_name in required_probes
            or candidate_probe_aliases.intersection(required_probes)
        )
        is_verification = loop._is_learning_verification_function(fn_name)
        failure_preference_delta = 0.0
        if failure_bucket:
            preferred_verification = (
                set(failure_bucket.get("preferred_verification_functions", [])[:3])
                if isinstance(failure_bucket.get("preferred_verification_functions", []), list)
                else set()
            )
            preferred_fallback = (
                set(failure_bucket.get("preferred_fallback_functions", [])[:3])
                if isinstance(failure_bucket.get("preferred_fallback_functions", []), list)
                else set()
            )
            blocked_actions = (
                set(failure_bucket.get("blocked_action_classes", []))
                if isinstance(failure_bucket.get("blocked_action_classes", []), list)
                else set()
            )
            strategy_mode_hint = str(
                failure_bucket.get("strategy_mode_hint", "balanced") or "balanced"
            )
            if fn_name in preferred_verification:
                failure_preference_delta += 0.12 + min(
                    0.08,
                    float(failure_bucket.get("confidence", 0.0) or 0.0) * 0.08,
                )
            if fn_name in preferred_fallback:
                failure_preference_delta += 0.08
            if fn_name in blocked_actions:
                failure_preference_delta -= 0.18
            if strategy_mode_hint == "verify" and is_verification:
                failure_preference_delta += 0.05
            elif strategy_mode_hint == "recover" and fn_name in preferred_fallback:
                failure_preference_delta += 0.03

        retention_delta = 0.0
        if dominant_retention:
            dominant_failure_type = str(
                dominant_retention.get("dominant_failure_type", "") or ""
            )
            severity = loop._clamp_learning_signal(
                dominant_retention.get("severity", 0.0),
                0.0,
                1.0,
                0.0,
            )
            planner_replan_bias_delta = loop._clamp_learning_signal(
                dominant_retention.get("planner_replan_bias_delta", 0.0),
                0.0,
                0.4,
                0.0,
            )
            probe_bias_delta = loop._clamp_learning_signal(
                dominant_retention.get("probe_bias_delta", 0.0),
                0.0,
                0.4,
                0.0,
            )
            if dominant_failure_type in {
                "prediction_drift",
                "governance_overrule_misfire",
            }:
                if is_verification or candidate_required_probe_match:
                    retention_delta += probe_bias_delta * 0.90 + severity * 0.06
                elif fn_name in dominant_risky_functions:
                    retention_delta -= 0.08 + severity * 0.12
            elif dominant_failure_type in {
                "branch_persistence_collapse",
                "planner_target_switch",
            }:
                if fn_name in dominant_anchor_functions:
                    retention_delta += planner_replan_bias_delta * 0.85 + severity * 0.05
                if (
                    candidate_required_probe_match
                    and int(dominant_retention.get("verification_budget_hint", 0) or 0) > 0
                ):
                    retention_delta += 0.04
                if fn_name in dominant_risky_functions:
                    retention_delta -= 0.10 + severity * 0.14

        world_model_delta = 0.0
        if candidate_required_probe_match:
            world_model_delta += (
                0.12 if bool(competition.get("probe_pressure_active", False)) else 0.05
            )
        if (
            fn_name in dominant_anchor_functions
            and float(competition.get("latent_instability", 0.0) or 0.0) >= 0.55
        ):
            world_model_delta += 0.09
        if is_verification and bool(competition.get("probe_pressure_active", False)):
            world_model_delta += 0.06
        if fn_name in dominant_risky_functions:
            world_model_delta -= 0.10 + float(
                competition.get("latent_instability", 0.0) or 0.0
            ) * 0.10

        learning_bonus = max(
            -0.85,
            min(
                0.85,
                fn_delta
                + agenda_delta * 0.35
                + failure_preference_delta
                + retention_delta
                + world_model_delta,
            ),
        )
        meta = action.setdefault("_candidate_meta", {})
        if not isinstance(meta, dict):
            continue
        meta["learning_bias"] = learning_bonus
        meta["selector_bias"] = fn_delta
        meta["agenda_prior"] = agenda_delta
        meta["learning_context_key"] = context_key
        meta["failure_preference_learning_bias"] = failure_preference_delta
        meta["retention_learning_bonus"] = retention_delta
        meta["world_model_learning_bias"] = world_model_delta
        meta["world_model_required_probes"] = list(
            competition.get("required_probes", []) or []
        )
        meta["world_model_probe_pressure"] = float(
            competition.get("probe_pressure", 0.0) or 0.0
        )
        meta["world_model_latent_instability"] = float(
            competition.get("latent_instability", 0.0) or 0.0
        )
        meta["world_model_dominant_branch_id"] = str(
            competition.get("dominant_branch_id", "") or ""
        )
        meta["world_model_anchor_functions"] = list(
            competition.get("dominant_anchor_functions", []) or []
        )
        meta["world_model_risky_functions"] = list(
            competition.get("dominant_risky_functions", []) or []
        )
        meta["world_model_probe_aliases"] = list(
            dict.fromkeys(
                str(item)
                for item in list(meta.get("world_model_probe_aliases", []) or [])
                if str(item or "")
            )
        )
        meta["world_model_required_probe_match"] = candidate_required_probe_match
        meta["world_model_anchor_match"] = fn_name in dominant_anchor_functions
        meta["world_model_risky_match"] = fn_name in dominant_risky_functions
        if failure_bucket:
            merged_profile = loop._merge_learned_failure_strategy_profile(
                meta.get("failure_strategy_profile", {})
                if isinstance(meta.get("failure_strategy_profile", {}), dict)
                else {},
                failure_bucket,
                competition=competition,
            )
            meta["failure_strategy_profile"] = merged_profile
            if not isinstance(meta.get("global_failure_strategy", {}), dict) or not meta.get(
                "global_failure_strategy"
            ):
                meta["global_failure_strategy"] = dict(merged_profile)
            meta["failure_preference_guidance"] = {
                "strategy_mode": str(
                    merged_profile.get("strategy_mode_hint", "balanced") or "balanced"
                ),
                "branch_budget_hint": int(
                    merged_profile.get("branch_budget_hint", 0) or 0
                ),
                "verification_budget_hint": int(
                    merged_profile.get("verification_budget_hint", 0) or 0
                ),
            }
        if dominant_retention:
            meta["retention_learning_bias_detail"] = {
                "dominant_failure_type": str(
                    dominant_retention.get("dominant_failure_type", "") or ""
                ),
                "severity": float(dominant_retention.get("severity", 0.0) or 0.0),
                "strategy_mode_hint": str(
                    dominant_retention.get("strategy_mode_hint", "balanced")
                    or "balanced"
                ),
            }

    return sorted(
        candidate_actions,
        key=lambda act: float(
            (
                act.get("_candidate_meta", {}) if isinstance(act, dict) else {}
            ).get("learning_bias", 0.0)
            or 0.0
        ),
        reverse=True,
    )


def commit_learning_updates(loop: Any, updates: List[Any]) -> int:
    committed = 0
    for update in updates:
        if loop._learning_budget_remaining() <= 0:
            break
        if not hasattr(update, "to_proposal_content"):
            continue
        content = update.to_proposal_content()
        key = str(content.get("key", "") or "")
        update_type = str(content.get("update_type", "") or "")
        confidence = max(0.1, min(0.95, float(content.get("confidence", 0.5) or 0.5)))
        proposal_object_id = f"learning_update_{uuid.uuid4().hex[:12]}"
        proposal_hash = hashlib.sha1(
            f"{proposal_object_id}:{update_type}:{key}:{loop._episode}:{loop._tick}".encode(
                "utf-8"
            )
        ).hexdigest()
        proposal = {
            "type": "memory_proposal",
            "object_id": proposal_object_id,
            "memory_type": "learning_update",
            "memory_layer": "semantic",
            "retrieval_tags": [
                "learning_update",
                f"learning_kind:{update_type}",
                f"learning_key:{key}",
            ],
            "content": content,
            "content_hash": proposal_hash,
            "confidence": confidence,
            "source_module": "core",
            "source_stage": "post_commit_learning",
            "episode": loop._episode,
        }
        decision = loop._validator.validate(proposal)
        if decision.decision not in (ACCEPT_NEW, MERGE_UPDATE_EXISTING):
            continue
        committed_ids = loop._committer.commit([(proposal, decision)], top_k=1)
        if committed_ids:
            committed += 1
            loop._learning_updates_sent_this_episode += 1
            loop._learning_update_log.append(
                {
                    "episode": loop._episode,
                    "tick": loop._tick,
                    "object_id": committed_ids[0],
                    "update_type": update_type,
                    "key": key,
                    "delta": float(content.get("delta", 0.0) or 0.0),
                    "confidence": confidence,
                    "result": "committed",
                }
            )
    loop._refresh_learning_policy_snapshot()
    return committed


def _recent_retention_signal(loop: Any) -> Dict[str, Any]:
    return next(
        (
            row
            for row in reversed(list(loop._learning_signal_log[-12:]))
            if isinstance(row, dict)
            and str(row.get("retention_failure_type", "") or "")
        ),
        {},
    )


def _retention_policy_bucket(loop: Any, retention_signal: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    retention_failure_policy = (
        loop._learning_policy_snapshot.get("retention_failure_policy", {})
        if isinstance(loop._learning_policy_snapshot, dict)
        and isinstance(loop._learning_policy_snapshot.get("retention_failure_policy", {}), dict)
        else {}
    )
    retention_context = (
        retention_signal.get("retention_failure_context", {})
        if isinstance(retention_signal.get("retention_failure_context", {}), dict)
        else {}
    ) if isinstance(retention_signal, dict) else {}
    retention_key = str(retention_context.get("context_key", "") or "")
    retention_bucket = (
        retention_failure_policy.get(retention_key, {})
        if retention_key and isinstance(retention_failure_policy.get(retention_key, {}), dict)
        else {}
    )
    return retention_context, retention_bucket, retention_key


def _latest_prediction_error_level(loop: Any) -> float:
    latest = loop._prediction_trace_log[-1] if loop._prediction_trace_log else {}
    if not isinstance(latest, dict):
        return 0.0
    error = latest.get("error", {}) if isinstance(latest.get("error", {}), dict) else {}
    return float(error.get("total_error", 0.0) or 0.0)


def run_apply_learning_policy_updates(loop: Any, episode: int) -> None:
    retention_signal = _recent_retention_signal(loop)
    retention_context, retention_bucket, retention_key = _retention_policy_bucket(
        loop,
        retention_signal,
    )
    adaptation_inputs = {
        "episode_trace": {
            "retrieval_miss_ratio": float(
                loop._state_mgr.get_state().get("metrics.retrieval_miss_ratio", 0.0) or 0.0
            ),
        },
        "prediction_error": {
            "error_level": float(_latest_prediction_error_level(loop) or 0.0),
        },
        "recovery_trace": (
            loop._recovery_log[-1]
            if loop._recovery_log and isinstance(loop._recovery_log[-1], dict)
            else {"quality": 1.0, "steps": 0}
        ),
        "mechanism_evidence": {
            "support": 0.8 if loop._recovery_log else 0.4,
        },
        "self_model_failure_summary": (
            loop._continuity_log[-1]
            if loop._continuity_log and isinstance(loop._continuity_log[-1], dict)
            else {}
        ),
        "retention_failure": {
            "failure_type": str(retention_signal.get("retention_failure_type", "") or ""),
            "severity": float(retention_signal.get("retention_failure_severity", 0.0) or 0.0),
            "context_key": retention_key,
            "strategy_mode_hint": str(
                retention_bucket.get(
                    "strategy_mode_hint",
                    retention_context.get("strategy_mode_hint", ""),
                )
                or retention_context.get("strategy_mode_hint", "")
            ),
            "branch_budget_hint": int(
                retention_bucket.get(
                    "branch_budget_hint",
                    retention_context.get("branch_budget_hint", 0),
                )
                or 0
            ),
            "verification_budget_hint": int(
                retention_bucket.get(
                    "verification_budget_hint",
                    retention_context.get("verification_budget_hint", 0),
                )
                or 0
            ),
            "planner_replan_bias_delta": float(
                retention_bucket.get("planner_replan_bias_delta", 0.0) or 0.0
            ),
            "probe_bias_delta": float(
                retention_bucket.get("probe_bias_delta", 0.0) or 0.0
            ),
            "retrieval_pressure_delta": float(
                retention_bucket.get("retrieval_pressure_delta", 0.0) or 0.0
            ),
            "rollout_branch_persistence_ratio": float(
                retention_bucket.get(
                    "rollout_branch_persistence_ratio",
                    retention_context.get("rollout_branch_persistence_ratio", 0.0),
                )
                or 0.0
            ),
            "forced_replan_events": list(
                retention_bucket.get(
                    "forced_replan_events",
                    retention_context.get("forced_replan_events", []),
                )
                or []
            ),
        },
    }
    ablation_cfg = getattr(loop, "_causal_ablation", None)
    update = loop._meta_control.apply_learning_policy_updates(
        episode=episode,
        tick=loop._tick,
        episode_reward=float(loop._episode_reward),
        adaptation_inputs=adaptation_inputs,
        enable_representation_adaptation=bool(
            getattr(ablation_cfg, "enable_representation_adaptation", True)
        ),
        freeze_retrieval_pressure=bool(
            getattr(ablation_cfg, "freeze_retrieval_pressure", False)
        ),
    )
    loop._learning_update_log.append(
        {
            "episode": episode,
            "tick": loop._tick,
            "update_type": "policy_learning_update",
            "reward_delta": float(update.get("reward_delta", 0.0)),
            "policy_state": dict(update.get("policy_state", {})),
            "trigger_evidence": list(
                update.get("policy_update_trace", {}).get("trigger_evidence", [])
            ),
            "effect_evaluation": dict(
                update.get("policy_update_trace", {}).get("effect_evaluation", {})
            ),
            "ablation_flags": loop._ablation_flags_snapshot(),
        }
    )
    loop._learning_update_log.append(
        {
            "episode": episode,
            "tick": loop._tick,
            "update_type": "policy_profile_object_update",
            "policy_object_id": str(update.get("policy_profile_object_id", "")),
            "result": str(update.get("result", "unknown")),
            "ablation_flags": loop._ablation_flags_snapshot(),
        }
    )
    loop._learning_update_log.append(
        {
            "episode": episode,
            "tick": loop._tick,
            "update_type": "representation_adaptation",
            "representation_layer": dict(
                update.get("adaptation_report", {}).get("representation_layer", {})
            ),
            "control_layer": dict(
                update.get("adaptation_report", {}).get("control_layer", {})
            ),
            "representation_state": dict(update.get("representation_state", {})),
            "representation_profile_object_id": str(
                update.get("representation_profile_object_id", "")
            ),
            "ablation_flags": loop._ablation_flags_snapshot(),
        }
    )


def _recent_recovery_event(loop: Any) -> Dict[str, Any]:
    for row in reversed(list(getattr(loop, "_recovery_log", [])[-5:])):
        if isinstance(row, dict) and int(row.get("tick", -1)) == int(loop._tick):
            return row
    return {}


def _recovery_cost_profile(recovery_event: Dict[str, Any]) -> Tuple[bool, str, float]:
    if not isinstance(recovery_event, dict):
        return False, "", 0.0
    path = recovery_event.get("path", {}) if isinstance(recovery_event.get("path", {}), dict) else {}
    recovery_type = str(path.get("recovery_type", "") or "").lower()
    if recovery_type == "request_replan":
        return True, recovery_type, 0.8
    if recovery_type == "request_probe":
        return True, recovery_type, 0.55
    return True, recovery_type, 0.35


def _prediction_mismatch_for_tick(loop: Any) -> float:
    for row in reversed(list(getattr(loop, "_prediction_trace_log", [])[-5:])):
        if not isinstance(row, dict):
            continue
        if int(row.get("episode", -1)) != int(loop._episode) or int(row.get("tick", -1)) != int(loop._tick):
            continue
        err = row.get("error", {}) if isinstance(row.get("error", {}), dict) else {}
        return max(0.0, min(1.0, float(err.get("total_error", 0.0) or 0.0)))
    return 0.0


def collect_outcome_learning_signal(
    loop: Any,
    *,
    action_to_use: Dict[str, Any],
    obs_before: Dict[str, Any],
    result: Dict[str, Any],
    reward: float,
) -> None:
    if not getattr(loop, "_learning_enabled", False):
        return
    function_name = loop._extract_action_function_name(action_to_use, default="wait")
    task_family = loop._infer_task_family(obs_before)
    action_meta = (
        action_to_use.get("_candidate_meta", {})
        if isinstance(action_to_use.get("_candidate_meta", {}), dict)
        else {}
    )
    success = bool(result.get("success", True)) and float(reward) >= 0.0
    recovery_event = _recent_recovery_event(loop)
    recovery_triggered, recovery_type, recovery_cost = _recovery_cost_profile(
        recovery_event
    )
    prediction_mismatch = _prediction_mismatch_for_tick(loop)
    resource_band = (
        loop._resource_state.budget_band()
        if hasattr(loop, "_resource_state") and hasattr(loop._resource_state, "budget_band")
        else "normal"
    )
    retention_failure = loop._classify_retention_failure(
        action_to_use=action_to_use,
        function_name=function_name,
        result=result,
        reward=reward,
        prediction_mismatch=prediction_mismatch,
        task_family=str(task_family or "unknown"),
        phase=str(obs_before.get("phase", "active")) if isinstance(obs_before, dict) else "active",
        observation_mode=str(
            (obs_before.get("perception") or {}).get("coordinate_type", "unknown")
        ) if isinstance(obs_before, dict) else "unknown",
        resource_band=resource_band,
        action_meta=action_meta,
    )
    if (
        hasattr(loop, "_hidden_state_tracker")
        and loop._hidden_state_tracker is not None
        and str(retention_failure.get("failure_type", "") or "")
    ):
        loop._hidden_state_tracker.record_retention_failure(
            str(retention_failure.get("failure_type", "") or ""),
            severity=float(retention_failure.get("severity", 0.0) or 0.0),
            context=(
                retention_failure.get("context", {})
                if isinstance(retention_failure.get("context", {}), dict)
                else {}
            ),
        )
    signal = OutcomeSignal(
        episode=int(loop._episode),
        tick=int(loop._tick),
        function_name=str(function_name),
        task_family=str(task_family or "unknown"),
        success=success,
        reward=float(reward),
        recovery_triggered=bool(recovery_triggered),
        recovery_cost=float(recovery_cost),
        prediction_mismatch=float(prediction_mismatch),
        recovery_type=str(recovery_type),
        phase=str(obs_before.get("phase", "active")) if isinstance(obs_before, dict) else "active",
        observation_mode=str(
            (obs_before.get("perception") or {}).get("coordinate_type", "unknown")
        ) if isinstance(obs_before, dict) else "unknown",
        resource_band=resource_band,
        failure_strategy_profile=dict(action_meta.get("failure_strategy_profile", {}))
        if isinstance(action_meta.get("failure_strategy_profile", {}), dict)
        else {},
        global_failure_strategy=dict(action_meta.get("global_failure_strategy", {}))
        if isinstance(action_meta.get("global_failure_strategy", {}), dict)
        else {},
        world_model_required_probes=list(action_meta.get("world_model_required_probes", []))
        if isinstance(action_meta.get("world_model_required_probes", []), list)
        else [],
        world_model_probe_pressure=float(action_meta.get("world_model_probe_pressure", 0.0) or 0.0),
        world_model_latent_instability=float(
            action_meta.get("world_model_latent_instability", 0.0) or 0.0
        ),
        world_model_dominant_branch_id=str(
            action_meta.get("world_model_dominant_branch_id", "") or ""
        ),
        world_model_anchor_functions=list(action_meta.get("world_model_anchor_functions", []))
        if isinstance(action_meta.get("world_model_anchor_functions", []), list)
        else [],
        world_model_risky_functions=list(action_meta.get("world_model_risky_functions", []))
        if isinstance(action_meta.get("world_model_risky_functions", []), list)
        else [],
        world_model_required_probe_match=bool(
            action_meta.get("world_model_required_probe_match", False)
        ),
        world_model_anchor_match=bool(action_meta.get("world_model_anchor_match", False)),
        world_model_risky_match=bool(action_meta.get("world_model_risky_match", False)),
        selected_source=str(action_to_use.get("_source", "") or ""),
        governance_reason=str(loop._latest_governance_entry_for_tick().get("reason", "") or ""),
        retention_failure_type=str(retention_failure.get("failure_type", "") or ""),
        retention_failure_severity=float(retention_failure.get("severity", 0.0) or 0.0),
        retention_failure_context=dict(retention_failure.get("context", {}))
        if isinstance(retention_failure.get("context", {}), dict)
        else {},
    )
    updates = loop._credit_assignment.assign_from_outcome(signal)
    committed = loop._commit_learning_updates(updates)
    loop._learning_signal_log.append(
        {
            "episode": signal.episode,
            "tick": signal.tick,
            "function_name": signal.function_name,
            "success": signal.success,
            "reward": signal.reward,
            "recovery_cost": signal.recovery_cost,
            "prediction_mismatch": signal.prediction_mismatch,
            "retention_failure_type": signal.retention_failure_type,
            "retention_failure_severity": signal.retention_failure_severity,
            "retention_failure_context": dict(signal.retention_failure_context or {})
            if isinstance(signal.retention_failure_context, dict)
            else {},
            "learning_updates_committed": committed,
        }
    )
    del loop._learning_signal_log[:-200]
