from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.orchestration.commit_candidate_guard import is_probe_like
from decision.mechanism_decision_context import (
    extract_mechanism_decision_context,
    mechanism_hypothesis_priority,
)
from modules.hypothesis.mechanism_posterior_updater import (
    action_matches_family,
    canonical_target_family,
    extract_target_descriptor,
    infer_action_family,
)
from modules.world_model.mechanism_runtime import mechanism_blocked_entries, mechanism_obs_state


@dataclass
class CandidateCooldownReport:
    filtered_candidates: List[Dict[str, Any]]
    removed_candidates: List[Dict[str, Any]]
    policy_delta: Dict[str, Any]
    diagnostics: Dict[str, Any]


class CandidateCooldownGate:
    """Hard candidate filtering for repeated dead-end families.

    This is not a soft score tweak. It removes short-horizon dead ends before the
    arbiter can fall back into them.
    """

    def __init__(self, *, fast_repeat_threshold: int = 2, min_info_gain_escape: float = 0.18) -> None:
        self._fast_repeat_threshold = int(max(1, fast_repeat_threshold))
        self._min_info_gain_escape = float(min_info_gain_escape)

    def filter_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        decision_context: Optional[Dict[str, Any]],
        episode_trace: Sequence[Dict[str, Any]],
        tick: int,
        obs_before: Optional[Dict[str, Any]] = None,
    ) -> CandidateCooldownReport:
        context = dict(decision_context or {})
        mechanism_control, mechanism_hypotheses, mechanism_context = extract_mechanism_decision_context(context)
        obs_snapshot = dict(obs_before or context.get('obs_before', {}) or {}) if isinstance(obs_before or context.get('obs_before', {}), dict) else {}
        obs_state = mechanism_obs_state(obs_snapshot, mechanism_control)
        blocked_entries = mechanism_blocked_entries(mechanism_control, tick=tick)

        filtered: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        reasons: List[Dict[str, Any]] = []
        preserved: List[Dict[str, Any]] = []
        for action in list(candidates or []):
            if not isinstance(action, dict):
                continue
            meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
            action_family = infer_action_family(action)
            target_desc = extract_target_descriptor(action)
            target_family = canonical_target_family(target_desc.get('target_family', '') or 'generic_target')
            action_target = f'{action_family}::{target_family}'
            reason = self._match_block_reason(
                action,
                action_family=action_family,
                target_family=target_family,
                action_target=action_target,
                blocked_entries=blocked_entries,
                tick=tick,
            )
            if reason:
                removed.append(action)
                reasons.append({'function_name': action.get('payload', {}).get('tool_args', {}).get('function_name', action.get('function_name', '')), 'reason': reason, 'action_family': action_family, 'target_family': target_family})
                continue
            preserve_reason = self._runtime_preserve_reason(
                action,
                mechanism_control=mechanism_control,
                obs_state=obs_state,
            )
            if preserve_reason:
                meta = dict(meta)
                meta['runtime_action_family'] = action_family
                meta['runtime_target_family'] = target_family
                meta['mechanism_runtime_preserved'] = True
                meta['mechanism_runtime_preserve_reason'] = preserve_reason
                action['_candidate_meta'] = meta
                filtered.append(action)
                preserved.append({
                    'function_name': action.get('payload', {}).get('tool_args', {}).get('function_name', action.get('function_name', '')),
                    'reason': preserve_reason,
                    'action_family': action_family,
                    'target_family': target_family,
                })
                continue
            reason = self._fast_local_repeat_reason(
                action,
                action_family=action_family,
                target_family=target_family,
                episode_trace=episode_trace,
            )
            if reason and not bool(meta.get('override_for_discrimination', False)):
                preserve_reason = self._mechanism_preserve_reason(
                    action,
                    target_family=target_family,
                    reason=reason,
                    mechanism_hypotheses=mechanism_hypotheses,
                    episode_trace=episode_trace,
                )
                if preserve_reason:
                    meta = dict(meta)
                    meta['runtime_action_family'] = action_family
                    meta['runtime_target_family'] = target_family
                    meta['mechanism_runtime_preserved'] = True
                    meta['mechanism_runtime_preserve_reason'] = preserve_reason
                    action['_candidate_meta'] = meta
                    filtered.append(action)
                    preserved.append({
                        'function_name': action.get('payload', {}).get('tool_args', {}).get('function_name', action.get('function_name', '')),
                        'reason': preserve_reason,
                        'action_family': action_family,
                        'target_family': target_family,
                    })
                    continue
                removed.append(action)
                reasons.append({'function_name': action.get('payload', {}).get('tool_args', {}).get('function_name', action.get('function_name', '')), 'reason': reason, 'action_family': action_family, 'target_family': target_family})
                continue
            meta = dict(meta)
            meta['runtime_action_family'] = action_family
            meta['runtime_target_family'] = target_family
            action['_candidate_meta'] = meta
            filtered.append(action)

        diagnostics = {
            'input_count': len(list(candidates or [])),
            'kept_count': len(filtered),
            'removed_count': len(removed),
            'removed_reasons': reasons,
            'preserved_runtime_candidates': preserved,
            'blocked_entries': blocked_entries,
            'mechanism_obs_state': dict(obs_state),
            'mechanism_hypothesis_count': int(mechanism_context.get('mechanism_hypothesis_count', 0) or 0),
            'mechanism_context_source': str(mechanism_context.get('mechanism_context_source', '') or ''),
        }
        policy_delta = {
            'blocked_entries': blocked_entries,
            'removed_reasons': reasons,
            'preserved_runtime_candidates': preserved,
        }
        return CandidateCooldownReport(
            filtered_candidates=filtered,
            removed_candidates=removed,
            policy_delta=policy_delta,
            diagnostics=diagnostics,
        )

    def _runtime_preserve_reason(
        self,
        action: Dict[str, Any],
        *,
        mechanism_control: Dict[str, Any],
        obs_state: Dict[str, bool],
    ) -> str:
        if not isinstance(action, dict):
            return ''
        meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
        role = str(meta.get('role', '') or '').strip().lower()
        kind = str(action.get('kind', '') or '').strip().lower()
        function_name = str(
            (action.get('payload', {}).get('tool_args', {}) if isinstance(action.get('payload', {}).get('tool_args', {}), dict) else {}).get('function_name', '')
            or action.get('function_name', '')
            or ''
        ).strip()
        control_mode = str(mechanism_control.get('control_mode', '') or '').strip().lower()
        probe_like = bool(
            meta.get('runtime_discriminating_candidate', False)
            or role == 'discriminate'
            or kind == 'probe'
            or is_probe_like(function_name, kind=kind)
        )
        if (kind == 'wait' or function_name == 'wait' or role == 'wait') and bool(obs_state.get('wait_ready', False)):
            return 'runtime_wait_ready'
        if role in {'prerequisite', 'prepare'} and bool(obs_state.get('prerequisite_ready', False)):
            return 'runtime_prerequisite_ready'
        if role == 'recovery' and bool(obs_state.get('recovery_ready', False)):
            return 'runtime_recovery_ready'
        if probe_like and control_mode == 'discriminate':
            return 'runtime_discriminate_required'
        return ''

    def _match_block_reason(
        self,
        action: Dict[str, Any],
        *,
        action_family: str,
        target_family: str,
        action_target: str,
        blocked_entries: Sequence[Dict[str, Any]],
        tick: int,
    ) -> str:
        for entry in blocked_entries:
            if not isinstance(entry, dict):
                continue
            until = int(entry.get('cooldown_until_tick', -1) or -1)
            if until < tick:
                continue
            scope = str(entry.get('scope', '') or '')
            value = str(entry.get('value', '') or '')
            canonical_value = canonical_target_family(value)
            if scope == 'target_family' and value == target_family:
                return f'blocked_target_family:{value}'
            if scope == 'target_family' and canonical_value == target_family:
                return f'blocked_target_family:{canonical_value}'
            if scope == 'action_target' and value == action_target:
                return f'blocked_action_target:{value}'
            if scope == 'action_family' and action_matches_family(action, value):
                return f'blocked_action_family:{value}'
        return ''

    def _fast_local_repeat_reason(
        self,
        action: Dict[str, Any],
        *,
        action_family: str,
        target_family: str,
        episode_trace: Sequence[Dict[str, Any]],
    ) -> str:
        recent = [row for row in list(episode_trace or [])[-6:] if isinstance(row, dict)]
        if not recent:
            return ''
        dead_count = 0
        target_dead_count = 0
        for row in reversed(recent):
            row_action = row.get('action', {}) if isinstance(row.get('action', {}), dict) else {}
            row_family = infer_action_family(row_action)
            row_target_desc = extract_target_descriptor(row_action)
            row_target_family = canonical_target_family(row_target_desc.get('target_family', '') or 'generic_target')
            info_gain = float(row.get('information_gain', 0.0) or 0.0)
            state_changed = bool(row.get('state_changed', False) or row.get('observation_changed', False))
            progressed = False
            task_progress_seen = False
            goal_stalled = False
            local_only_reaction = False
            for marker in list(row.get('progress_markers', []) or []):
                if not isinstance(marker, dict):
                    continue
                name = str(marker.get('name', '') or '')
                if name in {'goal_progressed', 'positive_reward'}:
                    progressed = True
                    break
                if name == 'task_progressed':
                    task_progress_seen = True
                if name == 'goal_stalled':
                    goal_stalled = True
                if name == 'local_only_reaction':
                    local_only_reaction = True
            if not progressed and task_progress_seen and not goal_stalled and not local_only_reaction:
                progressed = True
            if not progressed and float(row.get('reward', 0.0) or 0.0) > 0.0:
                progressed = True
            dead = (not state_changed) and (not progressed) and info_gain < self._min_info_gain_escape
            if row_family == action_family and row_target_family == target_family:
                if dead:
                    dead_count += 1
                else:
                    break
            if row_target_family == target_family:
                if dead:
                    target_dead_count += 1
                else:
                    break
        if dead_count >= self._fast_repeat_threshold:
            return f'fast_repeat_action_target:{action_family}::{target_family}'
        if target_dead_count >= self._fast_repeat_threshold:
            return f'fast_repeat_target_family:{target_family}'
        return ''

    def _recent_dead_anchor_refs(
        self,
        *,
        target_family: str,
        episode_trace: Sequence[Dict[str, Any]],
    ) -> set[str]:
        refs = set()
        recent = [row for row in list(episode_trace or [])[-6:] if isinstance(row, dict)]
        for row in recent:
            row_action = row.get('action', {}) if isinstance(row.get('action', {}), dict) else {}
            row_target_desc = extract_target_descriptor(row_action)
            row_target_family = canonical_target_family(row_target_desc.get('target_family', '') or 'generic_target')
            if row_target_family != target_family:
                continue
            info_gain = float(row.get('information_gain', 0.0) or 0.0)
            state_changed = bool(row.get('state_changed', False) or row.get('observation_changed', False))
            progressed = False
            for marker in list(row.get('progress_markers', []) or []):
                if not isinstance(marker, dict):
                    continue
                if str(marker.get('name', '') or '') in {'goal_progressed', 'positive_reward', 'task_progressed'}:
                    progressed = True
                    break
            if not progressed and float(row.get('reward', 0.0) or 0.0) > 0.0:
                progressed = True
            dead = bool(row.get('dead_action', False)) or ((not state_changed) and (not progressed) and info_gain < self._min_info_gain_escape)
            anchor_ref = str(row_target_desc.get('anchor_ref', '') or '')
            if dead and anchor_ref:
                refs.add(anchor_ref)
        return refs

    def _mechanism_preserve_reason(
        self,
        action: Dict[str, Any],
        *,
        target_family: str,
        reason: str,
        mechanism_hypotheses: Sequence[Dict[str, Any]],
        episode_trace: Sequence[Dict[str, Any]],
    ) -> str:
        if not (
            reason.startswith('fast_repeat_target_family:')
            or reason.startswith('fast_repeat_action_target:')
        ):
            return ''
        mechanism_priority = mechanism_hypothesis_priority(action, mechanism_hypotheses)
        if not bool(mechanism_priority.get('multi_anchor_support', False)):
            return ''
        if str(mechanism_priority.get('preferred_progress_mode', '') or '') != 'expand_anchor_coverage':
            return ''
        anchor_ref = str(extract_target_descriptor(action).get('anchor_ref', '') or '')
        if not anchor_ref:
            return ''
        if anchor_ref in self._recent_dead_anchor_refs(target_family=target_family, episode_trace=episode_trace):
            return ''
        return 'mechanism_multi_anchor_supported_goal'
