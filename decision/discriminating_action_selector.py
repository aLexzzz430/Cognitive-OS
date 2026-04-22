from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from core.orchestration.action_utils import extract_action_click_identity, extract_action_function_name, extract_available_functions
from decision.mechanism_decision_context import (
    extract_mechanism_decision_context,
    mechanism_hypothesis_priority,
)
from modules.hypothesis.mechanism_posterior_updater import (
    binding_token_frequency,
    candidate_binding_signal,
    extract_target_descriptor,
    infer_action_family,
    normalize_action_family,
    visible_functions_by_family,
)
from modules.world_model.mechanism_runtime import mechanism_blocked_entries, mechanism_obs_state


@dataclass
class DiscriminatingSelectionReport:
    candidates: List[Dict[str, Any]]
    enforced: bool
    diagnostics: Dict[str, Any]


class DiscriminatingActionSelector:
    """Force discriminating steps when mechanism uncertainty remains high."""

    def enforce(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        decision_context: Optional[Dict[str, Any]],
        obs_before: Optional[Dict[str, Any]],
        tick: int,
    ) -> DiscriminatingSelectionReport:
        context = dict(decision_context or {})
        mechanism_control, mechanism_hypotheses, mechanism_context = extract_mechanism_decision_context(context)
        require = bool(mechanism_control.get('require_discriminating_action', False))
        if not require:
            return DiscriminatingSelectionReport(
                candidates=list(candidates or []),
                enforced=False,
                diagnostics={
                    'reason': 'mechanism_control_no_enforce',
                    'mechanism_hypothesis_count': int(mechanism_context.get('mechanism_hypothesis_count', 0) or 0),
                    'mechanism_context_source': str(mechanism_context.get('mechanism_context_source', '') or ''),
                },
            )

        discriminating_families = [normalize_action_family(x) for x in list(mechanism_control.get('discriminating_actions', []) or []) if normalize_action_family(x)]
        preferred_target_refs = [str(x or '') for x in list(mechanism_control.get('preferred_target_refs', []) or []) if str(x or '')]
        control_mode = str(mechanism_control.get('control_mode', '') or '')
        obs_snapshot = dict(obs_before or {})
        obs_state = mechanism_obs_state(obs_snapshot, mechanism_control)
        release_ready = bool(obs_state.get('release_ready', False))
        relevant = [
            action
            for action in list(candidates or [])
            if isinstance(action, dict)
            and str((action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}).get('role', '') or '') in {'commit', 'recovery', 'prerequisite', 'prepare'}
        ]
        token_frequency = binding_token_frequency(relevant or list(candidates or []))
        binding_profiles: Dict[int, Dict[str, Any]] = {}
        commit_scores_by_action: Dict[int, float] = {}
        for action in list(candidates or []):
            if not isinstance(action, dict):
                continue
            profile = candidate_binding_signal(
                action,
                obs_before=obs_before,
                mechanism_control=mechanism_control,
                token_frequency=token_frequency,
            )
            binding_profiles[id(action)] = profile
            role = str((action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}).get('role', '') or '')
            if role == 'commit':
                commit_scores_by_action[id(action)] = float(profile.get('score', 0.0) or 0.0)
        allowed: List[Dict[str, Any]] = []
        promoted: List[Dict[str, Any]] = []
        release_candidates: List[Dict[str, Any]] = []
        click_frontier: List[Dict[str, Any]] = []
        seen_click_identities = set()
        for action in list(candidates or []):
            if not isinstance(action, dict):
                continue
            meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
            role = str(meta.get('role', '') or '')
            action_family = infer_action_family(action)
            target_desc = extract_target_descriptor(action)
            anchor_ref = str(target_desc.get('anchor_ref', '') or '')
            expected_info = float(meta.get('expected_information_gain', 0.0) or 0.0)
            target_match = bool(anchor_ref and anchor_ref in preferred_target_refs)
            binding_profile = dict(binding_profiles.get(id(action), {}))
            own_score = float(binding_profile.get('score', 0.0) or 0.0)
            runner_up = max(
                [score for action_id, score in commit_scores_by_action.items() if action_id != id(action)],
                default=0.0,
            ) if role == 'commit' else 0.0
            binding_profile['score_margin'] = round(float(own_score - runner_up), 6)
            mechanism_priority = mechanism_hypothesis_priority(action, mechanism_hypotheses)
            is_discriminating = bool(
                action_family in discriminating_families
                and (role == 'discriminate' or action.get('kind') == 'probe')
            )
            click_identity = extract_action_click_identity(action)
            explicit_click_frontier = bool(
                click_identity
                and (
                    bool(meta.get('surface_click_candidate', False))
                    or bool(meta.get('explicit_perception_target', False))
                    or str(action.get('_source', '') or '').strip().lower() == 'surface_generation'
                )
            )
            promotable = bool(
                expected_info >= 0.16
                or target_match
                or action.get('kind') == 'probe'
                or float(mechanism_priority.get('score', 0.0) or 0.0) >= 0.22
            )
            if bool(binding_profile.get('blocked_match', False)) or bool(binding_profile.get('revoked_match', False)):
                promotable = False
            if float(binding_profile.get('contradiction_penalty', 0.0) or 0.0) >= 0.45:
                promotable = False
            release_candidate = bool(
                (
                    role == 'commit'
                    and release_ready
                    and _commit_release_candidate(binding_profile, mechanism_control=mechanism_control)
                )
                or (role == 'wait' and (control_mode == 'wait' or bool(obs_state.get('wait_ready', False))))
                or (role == 'prerequisite' and (control_mode == 'prepare' or bool(obs_state.get('prerequisite_ready', False))))
                or (role == 'recovery' and (control_mode == 'recover' or bool(obs_state.get('recovery_ready', False))))
            )
            meta = dict(meta)
            meta['runtime_action_family'] = action_family
            meta['runtime_target_family'] = str(target_desc.get('target_family', '') or 'generic_target')
            meta['runtime_discriminating_candidate'] = bool(is_discriminating)
            meta['mechanism_binding_score'] = round(float(binding_profile.get('score', meta.get('mechanism_binding_score', 0.0)) or 0.0), 4)
            meta['mechanism_binding_margin'] = round(float(binding_profile.get('score_margin', meta.get('mechanism_binding_margin', 0.0)) or 0.0), 4)
            meta['mechanism_binding_specificity'] = round(float(binding_profile.get('specificity', meta.get('mechanism_binding_specificity', 0.0)) or 0.0), 4)
            meta['mechanism_binding_actionable'] = bool(meta.get('mechanism_binding_actionable', False) or _commit_release_candidate(binding_profile, mechanism_control=mechanism_control))
            meta['mechanism_object_priority'] = round(float(mechanism_priority.get('score', 0.0) or 0.0), 4)
            meta['mechanism_matched_object_id'] = str(mechanism_priority.get('matched_object_id', '') or '')
            meta['mechanism_multi_anchor_support'] = bool(mechanism_priority.get('multi_anchor_support', False))
            action['_candidate_meta'] = meta
            if explicit_click_frontier and click_identity not in seen_click_identities:
                seen_click_identities.add(click_identity)
                clone = dict(action)
                clone_meta = dict(meta)
                clone_meta['runtime_discriminating_candidate'] = True
                clone_meta['click_frontier_preserved_for_discrimination'] = True
                clone['_candidate_meta'] = clone_meta
                click_frontier.append(clone)
            if is_discriminating:
                allowed.append(action)
            elif promotable or release_candidate:
                clone = dict(action)
                clone_meta = dict(meta)
                clone_meta['runtime_discriminating_candidate'] = True
                clone_meta['override_for_discrimination'] = True
                clone_meta['synthetic_discriminator_promoted'] = True
                if release_candidate:
                    clone_meta['release_candidate'] = True
                clone['_candidate_meta'] = clone_meta
                promoted.append(clone)
                if release_candidate:
                    release_candidates.append(clone)

        synthesized: List[Dict[str, Any]] = []
        selected: List[Dict[str, Any]] = []
        if release_ready and release_candidates:
            selected = _rank_actions(release_candidates)[:3]
        elif allowed:
            selected.extend(_rank_actions(allowed)[:2])
            if release_ready:
                selected.extend(_rank_actions(release_candidates))
                selected.extend(_rank_actions(promoted))
        else:
            if release_candidates:
                selected = _rank_actions(release_candidates)[:3]
            elif promoted:
                selected = _rank_actions(promoted)[:3]
            elif click_frontier:
                selected = list(click_frontier)
        if not selected:
            if click_frontier:
                selected = list(click_frontier)
            else:
                if not release_ready:
                    synthesized = self._synthesize_discriminators(
                        discriminating_families=discriminating_families,
                        preferred_target_refs=preferred_target_refs,
                        obs_before=obs_before,
                        mechanism_control=mechanism_control,
                        tick=tick,
                    )
                    if synthesized:
                        selected = synthesized[:2]

        if click_frontier and selected:
            frontier_by_click = {
                extract_action_click_identity(action): action
                for action in click_frontier
                if extract_action_click_identity(action)
            }
            click_allowed: List[Dict[str, Any]] = []
            seen_allowed_clicks = set()
            for action in selected:
                click_identity = extract_action_click_identity(action)
                if not click_identity or click_identity in seen_allowed_clicks:
                    continue
                seen_allowed_clicks.add(click_identity)
                click_allowed.append(action)
            if click_allowed:
                for click_identity, action in frontier_by_click.items():
                    if click_identity in seen_allowed_clicks:
                        continue
                    seen_allowed_clicks.add(click_identity)
                    click_allowed.append(action)
                selected = click_allowed
            else:
                selected = list(frontier_by_click.values())

        if not selected:
            return DiscriminatingSelectionReport(
                candidates=list(candidates or []),
                enforced=False,
                diagnostics={
                    'reason': 'require_discrimination_but_no_candidate',
                    'discriminating_families': discriminating_families,
                    'visible_functions': extract_available_functions(obs_before or {}) if isinstance(obs_before, dict) else [],
                    'mechanism_hypothesis_count': int(mechanism_context.get('mechanism_hypothesis_count', 0) or 0),
                    'mechanism_context_source': str(mechanism_context.get('mechanism_context_source', '') or ''),
                },
            )

        return DiscriminatingSelectionReport(
            candidates=_dedupe_actions(selected),
            enforced=True,
            diagnostics={
                'reason': 'click_frontier_preserved_for_discrimination' if click_frontier and not synthesized else 'discriminating_mode_enforced',
                'discriminating_families': discriminating_families,
                'preferred_target_refs': preferred_target_refs,
                'control_mode': control_mode,
                'release_ready': bool(release_ready),
                'release_candidate_count': len(release_candidates),
                'selected_count': len(selected),
                'synthesized_count': len(synthesized),
                'preserved_click_frontier_count': len(click_frontier),
                'mechanism_hypothesis_count': int(mechanism_context.get('mechanism_hypothesis_count', 0) or 0),
                'mechanism_context_source': str(mechanism_context.get('mechanism_context_source', '') or ''),
            },
        )

    def _synthesize_discriminators(
        self,
        *,
        discriminating_families: Sequence[str],
        preferred_target_refs: Sequence[str],
        obs_before: Optional[Dict[str, Any]],
        mechanism_control: Optional[Dict[str, Any]] = None,
        tick: int = 0,
    ) -> List[Dict[str, Any]]:
        synthesized: List[Dict[str, Any]] = []
        seen = set()
        control = dict(mechanism_control or {})
        blocked_entries = mechanism_blocked_entries(control, tick=tick)
        skip_last_selected = bool(
            int(control.get('stagnant_ticks', 0) or 0) >= 1
            or any(
                isinstance(entry, dict)
                and (
                    str(entry.get('value', '') or '') == 'discriminating_probe'
                    or str(entry.get('value', '') or '').endswith('::discriminating_probe')
                )
                for entry in blocked_entries
            )
        )
        last_selected_function = str(control.get('last_selected_function', '') or '').strip()
        for family in discriminating_families or ('pointer_interaction', 'state_transform_interaction'):
            for fn in visible_functions_by_family(obs_before, family):
                fn_name = str(fn or '').strip()
                if not fn_name or fn_name in seen:
                    continue
                if skip_last_selected and last_selected_function and fn_name == last_selected_function:
                    continue
                seen.add(fn_name)
                synthesized.append(self._make_action(fn_name, family=family, preferred_target_refs=preferred_target_refs))
                break
        return synthesized

    def _make_action(self, function_name: str, *, family: str, preferred_target_refs: Sequence[str]) -> Dict[str, Any]:
        anchor_ref = str(preferred_target_refs[0] or '') if preferred_target_refs else ''
        target_kind = 'discriminating_probe' if normalize_action_family(family) != 'confirm_interaction' else 'discriminating_commit_test'
        meta: Dict[str, Any] = {
            'runtime_action_family': normalize_action_family(family),
            'runtime_discriminating_candidate': True,
            'override_for_discrimination': True,
            'synthetic_discriminator': True,
            'forbid_wait_baseline': True,
            'expected_information_gain': 0.34,
            'intervention_target': {
                'anchor_ref': anchor_ref,
                'target_kind': target_kind,
            },
        }
        return {
            'kind': 'call_tool',
            'payload': {
                'tool_name': 'call_hidden_function',
                'tool_args': {
                    'function_name': str(function_name),
                    'kwargs': {},
                },
            },
            '_source': 'mechanism_discriminator',
            '_candidate_meta': meta,
        }


def _release_ready(obs_before: Optional[Dict[str, Any]], *, mechanism_control: Optional[Dict[str, Any]] = None) -> bool:
    return bool(mechanism_obs_state(dict(obs_before or {}), dict(mechanism_control or {})).get('release_ready', False))


def _wait_ready(obs_before: Optional[Dict[str, Any]]) -> bool:
    return bool(mechanism_obs_state(dict(obs_before or {}), {}).get('wait_ready', False))


def _prerequisite_ready(obs_before: Optional[Dict[str, Any]]) -> bool:
    return bool(mechanism_obs_state(dict(obs_before or {}), {}).get('prerequisite_ready', False))


def _recovery_ready(obs_before: Optional[Dict[str, Any]]) -> bool:
    return bool(mechanism_obs_state(dict(obs_before or {}), {}).get('recovery_ready', False))


def _commit_release_candidate(binding_profile: Dict[str, Any], *, mechanism_control: Optional[Dict[str, Any]]) -> bool:
    control = dict(mechanism_control or {})
    commitment_trust = float(control.get('commitment_trust', 0.0) or 0.0)
    commitment_revoked = bool(control.get('commitment_revoked', False))
    if bool(binding_profile.get('revoked_match', False)):
        return False
    if float(binding_profile.get('contradiction_penalty', 0.0) or 0.0) >= 0.45:
        return False
    if bool(binding_profile.get('active_match', False)) and not commitment_revoked and commitment_trust >= 0.58:
        return True
    return bool(
        float(binding_profile.get('evidence_strength', 0.0) or 0.0) >= 0.85
        and float(binding_profile.get('specificity', 0.0) or 0.0) >= 0.45
        and float(binding_profile.get('score_margin', 0.0) or 0.0) >= 0.18
    )


def _dedupe_actions(actions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for action in list(actions or []):
        if not isinstance(action, dict):
            continue
        meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
        click_identity = extract_action_click_identity(action)
        key = (
            str(meta.get('action_id', '') or ''),
            str(action.get('payload', {}).get('tool_args', {}).get('function_name', '') or ''),
            str(meta.get('role', '') or ''),
            str(click_identity or ''),
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(action)
    return ordered


def _rank_actions(actions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = _dedupe_actions(actions)
    deduped.sort(
        key=lambda action: (
            float((action.get('_candidate_meta', {}) or {}).get('mechanism_object_priority', 0.0) or 0.0),
            float((action.get('_candidate_meta', {}) or {}).get('mechanism_binding_score', 0.0) or 0.0),
            float((action.get('_candidate_meta', {}) or {}).get('expected_information_gain', 0.0) or 0.0),
        ),
        reverse=True,
    )
    return deduped
