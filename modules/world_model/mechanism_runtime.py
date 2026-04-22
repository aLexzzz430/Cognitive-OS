from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _string_tokens(*values: Any, limit: int = 24) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for token in _string_tokens(*list(value), limit=limit):
                if token not in seen:
                    seen.add(token)
                    ordered.append(token)
                    if len(ordered) >= limit:
                        return ordered[:limit]
            continue
        text = str(value or "").strip().lower()
        if not text:
            continue
        canonical = text.replace("::", "_").replace("-", "_").replace(" ", "_")
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
            if len(ordered) >= limit:
                return ordered[:limit]
        for raw in canonical.split("_"):
            token = str(raw or "").strip().lower()
            if token and token not in seen:
                seen.add(token)
                ordered.append(token)
                if len(ordered) >= limit:
                    return ordered[:limit]
        for segment in str(text.replace("::", " ").replace("-", " ")).split():
            segment_canonical = segment.replace(" ", "_")
            if segment_canonical and segment_canonical not in seen:
                seen.add(segment_canonical)
                ordered.append(segment_canonical)
                if len(ordered) >= limit:
                    return ordered[:limit]
            for raw in segment_canonical.split("_"):
                token = str(raw or "").strip().lower()
                if token and token not in seen:
                    seen.add(token)
                    ordered.append(token)
                    if len(ordered) >= limit:
                        return ordered[:limit]
    return ordered[:limit]


def _obs_bool(obs: Dict[str, Any], key: str, default: bool = False) -> bool:
    if key not in obs:
        return bool(default)
    return bool(obs.get(key, default))


def _obs_has_any(obs: Dict[str, Any], *keys: str) -> bool:
    return any(key in obs for key in keys)


def _normalize_block_entry(entry: Any) -> Dict[str, Any]:
    item = _as_dict(entry)
    scope = str(item.get("scope", "") or "").strip()
    value = str(item.get("value", "") or "").strip()
    if not scope or not value:
        return {}
    return {
        "scope": scope,
        "value": value,
        "action_family": str(item.get("action_family", "") or "").strip(),
        "cooldown_until_tick": int(item.get("cooldown_until_tick", -1) or -1),
        "reason": str(item.get("reason", "") or "").strip(),
        "hard": bool(item.get("hard", False)),
        "policy_source": str(item.get("policy_source", "") or "").strip(),
        "policy_stage": str(item.get("policy_stage", "") or "").strip(),
    }


def _merge_block_entries(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str], Dict[str, Any]] = {}
    for raw in list(entries or []):
        entry = _normalize_block_entry(raw)
        if not entry:
            continue
        key = (str(entry.get("scope", "") or ""), str(entry.get("value", "") or ""))
        existing = dict(merged.get(key, {}))
        if not existing:
            merged[key] = entry
            continue
        merged[key] = {
            "scope": key[0],
            "value": key[1],
            "action_family": str(entry.get("action_family", "") or existing.get("action_family", "") or "").strip(),
            "cooldown_until_tick": max(
                int(existing.get("cooldown_until_tick", -1) or -1),
                int(entry.get("cooldown_until_tick", -1) or -1),
            ),
            "reason": str(entry.get("reason", "") or existing.get("reason", "") or "").strip(),
            "hard": bool(existing.get("hard", False) or entry.get("hard", False)),
            "policy_source": str(entry.get("policy_source", "") or existing.get("policy_source", "") or "").strip(),
            "policy_stage": str(entry.get("policy_stage", "") or existing.get("policy_stage", "") or "").strip(),
        }
    out = list(merged.values())
    out.sort(
        key=lambda item: (
            -int(bool(item.get("hard", False))),
            -int(item.get("cooldown_until_tick", -1) or -1),
            str(item.get("scope", "") or ""),
            str(item.get("value", "") or ""),
        )
    )
    return out


def compile_mechanism_cooldown_state(
    *,
    target_family_state: Dict[str, Any],
    action_target_state: Dict[str, Any],
    tick: int,
    active_commitment: Optional[Dict[str, Any]] = None,
    commitment_revoked: bool = False,
    fast_cooldown_ticks: int = 0,
) -> Dict[str, Any]:
    target_state = _as_dict(target_family_state)
    action_state = _as_dict(action_target_state)
    active = _as_dict(active_commitment)
    active_target_family = str(active.get("target_family", "") or "").strip()

    target_families: Dict[str, Dict[str, Any]] = {}
    action_targets: Dict[str, Dict[str, Any]] = {}
    revoked_targets: Dict[str, Dict[str, Any]] = {}
    blocked_entries: List[Dict[str, Any]] = []

    for family, raw_bucket in target_state.items():
        bucket = _as_dict(raw_bucket)
        until = max(
            int(bucket.get("hard_cooldown_until_tick", -1) or -1),
            int(bucket.get("cooldown_until_tick", -1) or -1),
        )
        if until < int(tick):
            continue
        family_key = str(family or "").strip()
        if not family_key:
            continue
        hard = bool(int(bucket.get("hard_cooldown_until_tick", -1) or -1) >= int(tick))
        target_families[family_key] = {
            "dead_streak": int(bucket.get("dead_streak", 0) or 0),
            "no_state_streak": int(bucket.get("no_state_streak", 0) or 0),
            "no_progress_streak": int(bucket.get("no_progress_streak", 0) or 0),
            "low_info_streak": int(bucket.get("low_info_streak", 0) or 0),
            "cooldown_until_tick": until,
            "hard_cooldown_until_tick": int(bucket.get("hard_cooldown_until_tick", -1) or -1),
            "hard": hard,
            "reason": str(
                (
                    bucket.get("hard_cooldown_reason", "")
                    if hard
                    else bucket.get("cooldown_reason", "")
                )
                or "repeated_no_state_progress_info"
            ).strip(),
            "policy_source": str(bucket.get("policy_source", "") or "").strip(),
            "policy_stage": str(bucket.get("policy_stage", "") or "").strip(),
        }
        blocked_entries.append(
            {
                "scope": "target_family",
                "value": family_key,
                "cooldown_until_tick": until,
                "reason": str(
                    (
                        bucket.get("hard_cooldown_reason", "")
                        if hard
                        else bucket.get("cooldown_reason", "")
                    )
                    or "repeated_no_state_progress_info"
                ).strip(),
                "hard": hard,
                "policy_source": str(bucket.get("policy_source", "") or "").strip(),
                "policy_stage": str(bucket.get("policy_stage", "") or "").strip(),
            }
        )

    for action_target, raw_bucket in action_state.items():
        bucket = _as_dict(raw_bucket)
        until = int(bucket.get("cooldown_until_tick", -1) or -1)
        if until < int(tick):
            continue
        action_target_key = str(action_target or "").strip()
        if not action_target_key:
            continue
        action_family = action_target_key.split("::", 1)[0] if "::" in action_target_key else ""
        action_targets[action_target_key] = {
            "dead_streak": int(bucket.get("dead_streak", 0) or 0),
            "no_state_streak": int(bucket.get("no_state_streak", 0) or 0),
            "no_progress_streak": int(bucket.get("no_progress_streak", 0) or 0),
            "low_info_streak": int(bucket.get("low_info_streak", 0) or 0),
            "cooldown_until_tick": until,
            "hard": bool(bucket.get("hard", False)),
            "reason": str(bucket.get("cooldown_reason", "") or "fast_action_target_cooldown").strip(),
            "action_family": action_family,
            "policy_source": str(bucket.get("policy_source", "") or "").strip(),
            "policy_stage": str(bucket.get("policy_stage", "") or "").strip(),
        }
        blocked_entries.append(
            {
                "scope": "action_target",
                "value": action_target_key,
                "action_family": action_family,
                "cooldown_until_tick": until,
                "reason": str(bucket.get("cooldown_reason", "") or "fast_action_target_cooldown").strip(),
                "hard": bool(bucket.get("hard", False)),
                "policy_source": str(bucket.get("policy_source", "") or "").strip(),
                "policy_stage": str(bucket.get("policy_stage", "") or "").strip(),
            }
        )

    if bool(commitment_revoked) and active_target_family:
        revoked_until = int(tick) + max(1, int(fast_cooldown_ticks or 0))
        revoked_targets[active_target_family] = {
            "cooldown_until_tick": revoked_until,
            "hard": True,
            "reason": "commitment_revoked",
            "policy_source": "active_commitment",
            "policy_stage": "commit",
        }
        blocked_entries.append(
            {
                "scope": "target_family",
                "value": active_target_family,
                "cooldown_until_tick": revoked_until,
                "reason": "commitment_revoked",
                "hard": True,
                "policy_source": "active_commitment",
                "policy_stage": "commit",
            }
        )

    merged_entries = _merge_block_entries(blocked_entries)
    for entry in merged_entries:
        if str(entry.get("scope", "") or "") != "target_family":
            continue
        family_key = str(entry.get("value", "") or "").strip()
        if not family_key:
            continue
        if family_key in revoked_targets:
            target_families.setdefault(family_key, {})
            target_families[family_key]["revoked"] = True
            target_families[family_key]["revoked_cooldown_until_tick"] = int(
                revoked_targets[family_key].get("cooldown_until_tick", -1) or -1
            )

    return {
        "target_families": target_families,
        "action_targets": action_targets,
        "revoked_targets": revoked_targets,
        "blocked_entries": merged_entries,
    }


def mechanism_blocked_entries(mechanism_control: Dict[str, Any], *, tick: Optional[int] = None) -> List[Dict[str, Any]]:
    control = _as_dict(mechanism_control)
    runtime_state = _as_dict(control.get("runtime_state", {}))
    cooldown_state = _as_dict(runtime_state.get("cooldown_state", control.get("cooldown_state", {})))
    raw_entries: List[Dict[str, Any]] = []

    for item in _as_list(runtime_state.get("blocked_entries", [])):
        entry = _normalize_block_entry(item)
        if entry:
            raw_entries.append(entry)
    for item in _as_list(control.get("blocked_entries", [])):
        entry = _normalize_block_entry(item)
        if entry:
            raw_entries.append(entry)

    for family, bucket in _as_dict(cooldown_state.get("target_families", {})).items():
        details = _as_dict(bucket)
        raw_entries.append(
            {
                "scope": "target_family",
                "value": str(family or "").strip(),
                "cooldown_until_tick": max(
                    int(details.get("revoked_cooldown_until_tick", -1) or -1),
                    int(details.get("cooldown_until_tick", -1) or -1),
                ),
                "reason": str(details.get("reason", "repeated_no_state_progress_info") or "repeated_no_state_progress_info"),
                "hard": bool(details.get("hard", False) or details.get("revoked", False)),
                "policy_source": str(details.get("policy_source", "") or "").strip(),
                "policy_stage": str(details.get("policy_stage", "") or "").strip(),
            }
        )
    for action_target, bucket in _as_dict(cooldown_state.get("action_targets", {})).items():
        details = _as_dict(bucket)
        raw_entries.append(
            {
                "scope": "action_target",
                "value": str(action_target or "").strip(),
                "action_family": str(details.get("action_family", "") or "").strip(),
                "cooldown_until_tick": int(details.get("cooldown_until_tick", -1) or -1),
                "reason": str(details.get("reason", "fast_action_target_cooldown") or "fast_action_target_cooldown"),
                "hard": bool(details.get("hard", False)),
                "policy_source": str(details.get("policy_source", "") or "").strip(),
                "policy_stage": str(details.get("policy_stage", "") or "").strip(),
            }
        )
    for family, bucket in _as_dict(cooldown_state.get("revoked_targets", {})).items():
        details = _as_dict(bucket)
        raw_entries.append(
            {
                "scope": "target_family",
                "value": str(family or "").strip(),
                "cooldown_until_tick": int(details.get("cooldown_until_tick", -1) or -1),
                "reason": str(details.get("reason", "commitment_revoked") or "commitment_revoked"),
                "hard": bool(details.get("hard", True)),
                "policy_source": str(details.get("policy_source", "") or "").strip(),
                "policy_stage": str(details.get("policy_stage", "") or "").strip(),
            }
        )

    entries = _merge_block_entries(raw_entries)
    if tick is not None:
        entries = [
            entry
            for entry in entries
            if int(entry.get("cooldown_until_tick", -1) or -1) >= int(tick)
        ]
    return entries


def mechanism_block_lists(mechanism_control: Dict[str, Any], *, tick: Optional[int] = None) -> Dict[str, List[str]]:
    entries = mechanism_blocked_entries(mechanism_control, tick=tick)
    blocked_target_families = [
        str(entry.get("value", "") or "").strip()
        for entry in entries
        if str(entry.get("scope", "") or "") == "target_family" and str(entry.get("value", "") or "").strip()
    ]
    blocked_action_families = [
        str(entry.get("value", "") or "").strip()
        for entry in entries
        if str(entry.get("scope", "") or "") == "action_family" and str(entry.get("value", "") or "").strip()
    ]
    blocked_action_targets = [
        str(entry.get("value", "") or "").strip()
        for entry in entries
        if str(entry.get("scope", "") or "") == "action_target" and str(entry.get("value", "") or "").strip()
    ]
    return {
        "blocked_target_families": blocked_target_families,
        "blocked_action_families": blocked_action_families,
        "blocked_action_targets": blocked_action_targets,
    }


def mechanism_obs_state(obs: Dict[str, Any], mechanism_control: Dict[str, Any]) -> Dict[str, bool]:
    obs = _as_dict(obs)
    control = _as_dict(mechanism_control)
    runtime_state = _as_dict(control.get("runtime_state", {}))
    embedded_obs_state = _as_dict(runtime_state.get("obs_state", {}))
    control_mode = str(control.get("control_mode", "") or "")

    if _obs_has_any(obs, "pending_countdown", "delayed_resolution_pending"):
        wait_ready = bool(
            int(obs.get("pending_countdown", 0) or 0) > 0
            or bool(obs.get("delayed_resolution_pending", False))
        )
    else:
        wait_ready = bool(embedded_obs_state.get("wait_ready", False))

    if _obs_has_any(obs, "prerequisite_missing", "has_prerequisite"):
        prerequisite_ready = bool(obs.get("prerequisite_missing", False)) and not bool(obs.get("has_prerequisite", False))
    else:
        prerequisite_ready = bool(embedded_obs_state.get("prerequisite_ready", False))

    if "recovery_required" in obs:
        recovery_ready = bool(obs.get("recovery_required", False))
    else:
        recovery_ready = bool(embedded_obs_state.get("recovery_ready", False))

    release_ready_current = bool(
        control_mode in {"exploit", "recover", "prepare", "wait"}
        or str(obs.get("revealed_signal_token", "") or "").strip()
        or bool(obs.get("goal_revealed", False))
        or bool(obs.get("has_prerequisite", False))
        or wait_ready
        or recovery_ready
    )
    if control_mode in {"exploit", "recover", "prepare", "wait"} or _obs_has_any(
        obs,
        "revealed_signal_token",
        "goal_revealed",
        "has_prerequisite",
        "pending_countdown",
        "delayed_resolution_pending",
        "recovery_required",
    ):
        release_ready = release_ready_current
    else:
        release_ready = bool(embedded_obs_state.get("release_ready", False) or release_ready_current)

    return {
        "wait_ready": bool(wait_ready),
        "prerequisite_ready": bool(prerequisite_ready),
        "recovery_ready": bool(recovery_ready),
        "release_ready": bool(release_ready),
    }


def build_mechanism_runtime_state(
    obs: Dict[str, Any],
    mechanism_control: Dict[str, Any],
    *,
    action_tokens: Sequence[str] = (),
    target_tokens: Sequence[str] = (),
) -> Dict[str, Any]:
    obs = _as_dict(obs)
    control = _as_dict(mechanism_control)
    runtime_state = _as_dict(control.get("runtime_state", {}))
    embedded_facts = _as_dict(runtime_state.get("facts", {}))
    runtime_graph = _as_dict(control.get("runtime_object_graph", {}))
    obs_state = mechanism_obs_state(obs, control)
    active_commitment = _as_dict(control.get("active_commitment", {}))
    blocked_entries = mechanism_blocked_entries(control)
    blocked_lists = mechanism_block_lists(control)
    cooldown_state = _as_dict(runtime_state.get("cooldown_state", control.get("cooldown_state", {})))
    if blocked_entries and not _as_list(cooldown_state.get("blocked_entries", [])):
        cooldown_state = dict(cooldown_state)
        cooldown_state["blocked_entries"] = list(blocked_entries)

    signal_tokens = set(
        _string_tokens(
            runtime_state.get("signal_tokens", []),
            control.get("signal_tokens", []),
            obs.get("revealed_signal_token", ""),
            obs.get("phase", ""),
            obs.get("status", ""),
            limit=24,
        )
    )
    counterevidence_tokens = set(
        _string_tokens(
            runtime_state.get("counterevidence_tokens", []),
            control.get("counterevidence_tokens", []),
            obs.get("counterevidence_token", ""),
            limit=24,
        )
    )
    active_tokens = set(
        _string_tokens(
            runtime_state.get("active_tokens", []),
            active_commitment.get("binding_tokens", []),
            active_commitment.get("target_family", ""),
            active_commitment.get("anchor_ref", ""),
            active_commitment.get("target_key", ""),
            limit=24,
        )
    )
    focus_tokens = set(
        _string_tokens(
            runtime_state.get("focus_tokens", []),
            runtime_graph.get("focus_object_ids", []),
            limit=16,
        )
    )
    target_token_set = {
        str(item or "").strip().lower()
        for item in list(target_tokens or [])
        if str(item or "").strip()
    }
    action_token_set = {
        str(item or "").strip().lower()
        for item in list(action_tokens or [])
        if str(item or "").strip()
    }
    support_tokens = signal_tokens | active_tokens | focus_tokens
    target_signal_overlap = _clamp01(runtime_state.get("target_signal_overlap", 0.0), 0.0)
    if target_token_set and support_tokens:
        target_signal_overlap = len(target_token_set & support_tokens) / float(max(1, len(target_token_set)))

    phase = str(
        obs.get("phase", "")
        or obs.get("status", "")
        or runtime_state.get("phase", "")
        or ""
    ).strip().lower()
    goal_revealed = _obs_bool(obs, "goal_revealed", bool(runtime_state.get("goal_revealed", False)))
    has_prerequisite = _obs_bool(obs, "has_prerequisite", bool(runtime_state.get("has_prerequisite", False)))
    solved = _obs_bool(obs, "solved", bool(runtime_state.get("solved", False)))

    has_signal = bool(signal_tokens)
    target_signaled = target_signal_overlap >= 0.15 or bool(target_token_set & focus_tokens)
    state_changed_after_probe = bool(
        embedded_facts.get("state_changed_after_probe", False)
        or has_signal
        or target_signaled
        or phase in {"revealed", "configured", "stabilizing", "committed", "resolved"}
    )
    focus_aligned_with_goal = bool(
        embedded_facts.get("focus_aligned_with_goal", False)
        or target_signaled
        or bool(action_token_set & (active_tokens | focus_tokens))
    )
    assembly_ready = bool(
        embedded_facts.get("assembly_ready", False)
        or has_signal
        or target_signaled
        or goal_revealed
        or has_prerequisite
    )
    configuration_valid = bool(
        embedded_facts.get("configuration_valid", False)
        or goal_revealed
        or target_signaled
        or phase in {"configured", "committed", "resolved"}
    )

    facts = {
        "wait_ready": bool(obs_state.get("wait_ready", False)),
        "prerequisite_ready": bool(obs_state.get("prerequisite_ready", False)),
        "recovery_ready": bool(obs_state.get("recovery_ready", False)),
        "release_ready": bool(obs_state.get("release_ready", False)),
        "goal_unknown": not goal_revealed and not has_signal and not active_tokens,
        "target_unresolved": not target_signaled and not goal_revealed and not has_signal,
        "state_changed_after_probe": state_changed_after_probe,
        "goal_not_reached": not goal_revealed and not solved,
        "focus_aligned_with_goal": focus_aligned_with_goal,
        "assembly_ready": assembly_ready,
        "assembly_not_ready": not assembly_ready,
        "configuration_invalid": not has_signal and not target_signaled,
        "configuration_partially_formed": bool(
            embedded_facts.get("configuration_partially_formed", False)
            or (has_signal and not goal_revealed)
        ),
        "configuration_valid": configuration_valid,
        "has_counterevidence": bool(counterevidence_tokens),
    }
    runtime_tokens = set(
        _string_tokens(
            signal_tokens,
            counterevidence_tokens,
            active_tokens,
            focus_tokens,
            phase,
            obs.get("status", ""),
            action_tokens,
            target_tokens,
            [name for name, value in facts.items() if value],
            limit=40,
        )
    )
    return {
        "facts": facts,
        "obs_state": obs_state,
        "blocked_entries": blocked_entries,
        "blocked_target_families": list(blocked_lists.get("blocked_target_families", []) or []),
        "blocked_action_families": list(blocked_lists.get("blocked_action_families", []) or []),
        "blocked_action_targets": list(blocked_lists.get("blocked_action_targets", []) or []),
        "cooldown_state": cooldown_state,
        "runtime_tokens": sorted(runtime_tokens),
        "signal_tokens": sorted(signal_tokens),
        "counterevidence_tokens": sorted(counterevidence_tokens),
        "active_tokens": sorted(active_tokens),
        "focus_tokens": sorted(focus_tokens),
        "target_signal_overlap": round(float(target_signal_overlap), 6),
        "phase": phase,
        "goal_revealed": bool(goal_revealed),
        "has_prerequisite": bool(has_prerequisite),
        "solved": bool(solved),
    }


def evaluate_mechanism_preconditions(
    preconditions: Sequence[Any],
    *,
    runtime_state: Dict[str, Any],
) -> Dict[str, Any]:
    rows = [str(item or "").strip().lower() for item in list(preconditions or []) if str(item or "").strip()]
    if not rows:
        return {
            "has_preconditions": False,
            "satisfied": True,
            "support": 1.0,
            "matched": [],
            "unmet": [],
        }

    facts = _as_dict(runtime_state.get("facts", {}))
    runtime_tokens: Set[str] = {
        str(item or "").strip().lower()
        for item in list(runtime_state.get("runtime_tokens", []) or [])
        if str(item or "").strip()
    }

    matched: List[str] = []
    unmet: List[str] = []
    for text in rows:
        satisfied = False
        if "goal unknown" in text:
            satisfied = bool(facts.get("goal_unknown", False))
        elif "target unresolved" in text:
            satisfied = bool(facts.get("target_unresolved", False))
        elif "state changed after probe" in text:
            satisfied = bool(facts.get("state_changed_after_probe", False))
        elif "goal not reached" in text:
            satisfied = bool(facts.get("goal_not_reached", False))
        elif "focus aligned with goal" in text:
            satisfied = bool(facts.get("focus_aligned_with_goal", False))
        elif text in {"assembly_ready", "assembly ready"}:
            satisfied = bool(facts.get("assembly_ready", False))
        elif "assembly not ready" in text:
            satisfied = bool(facts.get("assembly_not_ready", False))
        elif "configuration invalid" in text:
            satisfied = bool(facts.get("configuration_invalid", False))
        elif "configuration partially formed" in text:
            satisfied = bool(facts.get("configuration_partially_formed", False))
        elif text in {"configuration_valid", "configuration valid"}:
            satisfied = bool(facts.get("configuration_valid", False))
        elif "prerequisite missing" in text:
            satisfied = bool(facts.get("prerequisite_ready", False))
        elif "recovery required" in text:
            satisfied = bool(facts.get("recovery_ready", False))
        elif "wait ready" in text or "delayed resolution pending" in text:
            satisfied = bool(facts.get("wait_ready", False))
        elif "release ready" in text:
            satisfied = bool(facts.get("release_ready", False))
        elif "counterevidence" in text:
            satisfied = bool(facts.get("has_counterevidence", False))
        else:
            tokens = {token for token in _string_tokens(text, limit=8) if token}
            overlap = 0.0
            if tokens and runtime_tokens:
                overlap = len(tokens & runtime_tokens) / float(max(1, len(tokens)))
            satisfied = overlap >= 0.4
        if satisfied:
            matched.append(text)
        else:
            unmet.append(text)

    matched_count = len(matched)
    support = matched_count / float(max(1, len(rows)))
    return {
        "has_preconditions": True,
        "satisfied": matched_count == len(rows),
        "support": round(support, 6),
        "matched": matched,
        "unmet": unmet,
    }


def enrich_mechanism_control_summary(
    mechanism_control: Dict[str, Any],
    obs: Dict[str, Any],
) -> Dict[str, Any]:
    summary = _as_dict(mechanism_control)
    summary["runtime_state"] = build_mechanism_runtime_state(obs, summary)
    return summary
