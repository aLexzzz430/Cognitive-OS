from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.objects import OBJECT_TYPE_HYPOTHESIS, proposal_to_object_record
from core.orchestration.action_utils import (
    extract_action_click_identity,
    extract_action_function_name,
    extract_action_kind,
    extract_available_functions,
)
from core.world_model.object_graph import build_runtime_object_graph
from decision.mode_arbiter import infer_mode
from modules.world_model.mechanism_runtime import (
    compile_mechanism_cooldown_state,
    enrich_mechanism_control_summary,
    mechanism_blocked_entries,
    mechanism_block_lists,
)


ACTION_FAMILY_ORDER = (
    'pointer_interaction',
    'confirm_interaction',
    'navigation_interaction',
    'state_transform_interaction',
    'wait',
)


GROUNDING_STOPWORDS = {
    'a',
    'an',
    'the',
    'and',
    'controls',
    'control',
    'gate',
    'switch',
    'device',
    'lever',
    'orb',
    'door',
    'cluster',
    'panel',
    'family',
    'target',
    'probe',
    'diagnostic',
    'signal',
    'revealed',
    'counterevidence',
}


@dataclass
class MechanismRuntimeView:
    mechanisms: List[Dict[str, Any]]
    control_summary: Dict[str, Any]


@dataclass
class MechanismUpdateResult:
    state: Dict[str, Any]
    control_summary: Dict[str, Any]
    diagnostics: Dict[str, Any]
    mechanisms: List[Dict[str, Any]]
    durable_object_ids: List[str]


def _evidence_slug(value: Any) -> str:
    text = str(value or '').strip().lower()
    if not text:
        return ''
    slug = ''.join(ch if ch.isalnum() else '_' for ch in text).strip('_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug[:64]


def _mechanism_predictions(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'predicted_action_effects': deepcopy(row.get('predicted_action_effects', {}))
        if isinstance(row.get('predicted_action_effects', {}), dict)
        else {},
        'predicted_action_effects_by_signature': deepcopy(row.get('predicted_action_effects_by_signature', {}))
        if isinstance(row.get('predicted_action_effects_by_signature', {}), dict)
        else {},
        'predicted_observation_tokens': [
            str(item or '')
            for item in list(row.get('predicted_observation_tokens', []) or [])
            if str(item or '')
        ],
        'predicted_phase_shift': str(row.get('predicted_phase_shift', '') or ''),
        'predicted_information_gain': float(row.get('predicted_information_gain', 0.0) or 0.0),
    }


def _mechanism_hypothesis_record(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(row)
    payload['type'] = str(payload.get('type', 'mechanism_hypothesis') or 'mechanism_hypothesis')
    payload['object_type'] = OBJECT_TYPE_HYPOTHESIS
    payload['hypothesis_type'] = str(payload.get('hypothesis_type', 'mechanism_hypothesis') or 'mechanism_hypothesis')
    payload['source_stage'] = str(payload.get('source_stage', 'mechanism_runtime') or 'mechanism_runtime')
    payload['support_count'] = int(payload.get('support_count', 0) or 0)
    payload['contradiction_count'] = int(
        payload.get('contradiction_count', payload.get('refute_count', 0)) or 0
    )
    payload['predictions'] = _mechanism_predictions(payload)
    payload['hypothesis_metadata'] = (
        deepcopy(payload.get('metadata', {}))
        if isinstance(payload.get('metadata', {}), dict)
        else {}
    )
    payload.setdefault('scope', 'mechanism_runtime')
    payload.setdefault('source', 'world_model_mechanism')
    payload.setdefault('tags', ['mechanism_hypothesis'])
    payload['supporting_evidence_rows'] = [
        deepcopy(item)
        for item in list(payload.get('supporting_evidence', []) or [])
        if isinstance(item, dict)
    ]
    payload['contradicting_evidence_rows'] = [
        deepcopy(item)
        for item in list(payload.get('contradicting_evidence', []) or [])
        if isinstance(item, dict)
    ]
    object_id = str(payload.get('object_id', payload.get('hypothesis_id', '')) or '').strip()
    return proposal_to_object_record(payload, object_id=object_id)


def _mechanism_additional_content(row: Dict[str, Any]) -> Dict[str, Any]:
    supporting_rows = [
        deepcopy(item)
        for item in list(row.get('supporting_evidence', []) or [])
        if isinstance(item, dict)
    ]
    contradicting_rows = [
        deepcopy(item)
        for item in list(row.get('contradicting_evidence', []) or [])
        if isinstance(item, dict)
    ]
    return {
        'summary': str(row.get('summary', '') or ''),
        'family': str(row.get('family', '') or ''),
        'source_stage': str(row.get('source_stage', 'mechanism_runtime') or 'mechanism_runtime'),
        'surface_priority': float(row.get('surface_priority', 0.55) or 0.55),
        'supporting_evidence': deepcopy(row.get('supporting_evidence', []))
        if isinstance(row.get('supporting_evidence', []), list)
        else [],
        'contradicting_evidence': deepcopy(row.get('contradicting_evidence', []))
        if isinstance(row.get('contradicting_evidence', []), list)
        else [],
        'hypothesis_type': str(row.get('hypothesis_type', 'mechanism_hypothesis') or 'mechanism_hypothesis'),
        'posterior': float(row.get('posterior', row.get('confidence', 0.0)) or 0.0),
        'support_count': int(row.get('support_count', 0) or 0),
        'contradiction_count': int(row.get('contradiction_count', row.get('refute_count', 0)) or 0),
        'scope': str(row.get('scope', 'mechanism_runtime') or 'mechanism_runtime'),
        'source': str(row.get('source', 'world_model_mechanism') or 'world_model_mechanism'),
        'predictions': _mechanism_predictions(row),
        'falsifiers': deepcopy(row.get('falsifiers', [])) if isinstance(row.get('falsifiers', []), list) else [],
        'conflicts_with': deepcopy(row.get('conflicts_with', [])) if isinstance(row.get('conflicts_with', []), list) else [],
        'supporting_evidence_rows': supporting_rows,
        'contradicting_evidence_rows': contradicting_rows,
        'tags': deepcopy(row.get('tags', [])) if isinstance(row.get('tags', []), list) else [],
        'hypothesis_metadata': (
            deepcopy(row.get('metadata', {}))
            if isinstance(row.get('metadata', {}), dict)
            else {}
        ),
        'status': str(row.get('status', '') or ''),
        'type': str(row.get('type', 'mechanism_hypothesis') or 'mechanism_hypothesis'),
        'object_type': OBJECT_TYPE_HYPOTHESIS,
        'predicted_action_effects': deepcopy(row.get('predicted_action_effects', {}))
        if isinstance(row.get('predicted_action_effects', {}), dict)
        else {},
        'predicted_action_effects_by_signature': deepcopy(row.get('predicted_action_effects_by_signature', {}))
        if isinstance(row.get('predicted_action_effects_by_signature', {}), dict)
        else {},
        'predicted_observation_tokens': [
            str(item or '')
            for item in list(row.get('predicted_observation_tokens', []) or [])
            if str(item or '')
        ],
        'predicted_phase_shift': str(row.get('predicted_phase_shift', '') or ''),
        'predicted_information_gain': float(row.get('predicted_information_gain', 0.0) or 0.0),
    }


def _mechanism_evidence_ids(row: Dict[str, Any]) -> List[str]:
    object_id = str(row.get('object_id', row.get('hypothesis_id', '')) or '').strip()
    if not object_id:
        return []
    evidence_ids: List[str] = []
    for kind, entries in (
        ('support', list(row.get('supporting_evidence', []) or [])),
        ('contradiction', list(row.get('contradicting_evidence', []) or [])),
    ):
        for index, entry in enumerate(entries):
            if isinstance(entry, dict):
                token = (
                    str(entry.get('id') or entry.get('evidence_id') or entry.get('event_type') or entry.get('reason') or '')
                    or f'{kind}_{index}'
                )
            else:
                token = str(entry or '') or f'{kind}_{index}'
            evidence_ids.append(f'ev_mechanism::{object_id}::{kind}::{index}::{_evidence_slug(token) or f"{kind}_{index}"}')
    seen = set()
    ordered: List[str] = []
    for evidence_id in evidence_ids:
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        ordered.append(evidence_id)
    return ordered


def _sync_mechanism_records_to_store(object_store: Any, mechanisms: Sequence[Dict[str, Any]]) -> List[str]:
    if object_store is None:
        return []
    restore_records = getattr(object_store, 'restore_records', None)
    merge_update = getattr(object_store, 'merge_update', None)
    get_record = getattr(object_store, 'get', None)
    if not callable(restore_records) and not callable(merge_update):
        return []
    synced: List[str] = []
    for row in list(mechanisms or []):
        if not isinstance(row, dict):
            continue
        object_id = str(row.get('object_id', row.get('hypothesis_id', '')) or '').strip()
        if not object_id:
            continue
        evidence_ids = _mechanism_evidence_ids(row)
        existing = get_record(object_id) if callable(get_record) else None
        if isinstance(existing, dict) and callable(merge_update):
            if merge_update(object_id, evidence_ids, additional_content=_mechanism_additional_content(row)):
                synced.append(object_id)
                continue
        if callable(restore_records):
            seeded_record = _mechanism_hypothesis_record(row)
            seeded_record['evidence_ids'] = evidence_ids
            restored = restore_records([seeded_record], replace=False)
            if object_id in list(restored or []):
                synced.append(object_id)
    return synced


class MechanismPosteriorUpdater:
    """Make mechanism hypotheses operational instead of static narration.

    The updater maintains a runtime posterior over mechanism families, produces
    policy consequences (blocked families / forced discrimination), and updates
    that posterior after every executed action.
    """

    def __init__(
        self,
        *,
        support_gain: float = 0.18,
        refute_gain: float = 0.22,
        rival_refute_gain: float = 0.10,
        fast_cooldown_ticks: int = 2,
        hard_cooldown_ticks: int = 3,
        uncertainty_margin_threshold: float = 0.16,
        low_confidence_threshold: float = 0.58,
    ) -> None:
        self._support_gain = float(support_gain)
        self._refute_gain = float(refute_gain)
        self._rival_refute_gain = float(rival_refute_gain)
        self._fast_cooldown_ticks = int(max(1, fast_cooldown_ticks))
        self._hard_cooldown_ticks = int(max(self._fast_cooldown_ticks, hard_cooldown_ticks))
        self._uncertainty_margin_threshold = float(uncertainty_margin_threshold)
        self._low_confidence_threshold = float(low_confidence_threshold)

    # ------------------------------------------------------------------
    # Public runtime API
    # ------------------------------------------------------------------

    def ensure_state(
        self,
        runtime_state: Optional[Dict[str, Any]],
        mechanism_hypotheses_summary: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        state = dict(runtime_state or {})
        state.setdefault('version', 2)
        state.setdefault('mechanisms', {})
        state.setdefault('target_family_state', {})
        state.setdefault('action_target_state', {})
        state.setdefault('target_bindings', {})
        state.setdefault('active_commitment', {})
        state.setdefault('runtime_object_graph', {})
        state.setdefault('blocked_entries', [])
        state.setdefault('last_policy_delta', {})
        state.setdefault('stagnant_ticks', 0)
        state.setdefault('last_posterior_motion', 0.0)
        state.setdefault('last_tick', -1)
        state.setdefault('last_selected_action_family', '')
        state.setdefault('last_selected_target_family', '')
        state.setdefault('last_selected_function', '')

        mechanisms = state['mechanisms'] if isinstance(state.get('mechanisms'), dict) else {}
        for row in mechanism_hypotheses_summary:
            if not isinstance(row, dict):
                continue
            key = self._mechanism_key(row)
            if not key:
                continue
            existing: Dict[str, Any] = {}
            for candidate in (
                str(row.get('object_id', '') or '').strip(),
                str(row.get('hypothesis_id', '') or '').strip(),
                str(row.get('family', '') or '').strip(),
                key,
            ):
                if candidate and isinstance(mechanisms.get(candidate, {}), dict):
                    existing = dict(mechanisms.get(candidate, {}))
                    break
            base_conf = _clamp01(row.get('confidence', 0.0), 0.0)
            mechanisms[key] = {
                'hypothesis_id': str(row.get('hypothesis_id', key) or key),
                'object_id': str(row.get('object_id', existing.get('object_id', row.get('hypothesis_id', key))) or key),
                'family': str(row.get('family', key) or key),
                'base_confidence': base_conf,
                'posterior': _clamp01(existing.get('posterior', base_conf), base_conf),
                'support_count': int(existing.get('support_count', 0) or 0),
                'refute_count': int(existing.get('refute_count', existing.get('contradiction_count', 0)) or 0),
                'last_supported_tick': int(existing.get('last_supported_tick', -1) or -1),
                'last_refuted_tick': int(existing.get('last_refuted_tick', -1) or -1),
                'preferred_target_refs': [str(x or '') for x in list(row.get('preferred_target_refs', []) or []) if str(x or '')],
                'preferred_action_families': [normalize_action_family(x) for x in list(row.get('preferred_action_families', []) or []) if normalize_action_family(x)],
                'best_discriminating_actions': [normalize_action_family(x) for x in list(row.get('best_discriminating_actions', []) or []) if normalize_action_family(x)],
                'predicted_observation_tokens': [str(x or '').strip().lower() for x in list(row.get('predicted_observation_tokens', []) or []) if str(x or '').strip()],
                'supporting_evidence': list(existing.get('supporting_evidence', []) or []),
                'contradicting_evidence': list(existing.get('contradicting_evidence', []) or []),
                'hypothesis_type': str(row.get('hypothesis_type', existing.get('hypothesis_type', 'mechanism_hypothesis')) or 'mechanism_hypothesis'),
            }
        state['mechanisms'] = mechanisms
        return state

    def build_runtime_view(
        self,
        runtime_state: Optional[Dict[str, Any]],
        mechanism_hypotheses_summary: Sequence[Dict[str, Any]],
        *,
        tick: int,
        obs_before: Optional[Dict[str, Any]] = None,
    ) -> MechanismRuntimeView:
        state = self.ensure_state(runtime_state, mechanism_hypotheses_summary)
        ranked = self._rank_mechanisms(state)
        enhanced_mechanisms: List[Dict[str, Any]] = []
        for row in mechanism_hypotheses_summary:
            if not isinstance(row, dict):
                continue
            key = self._mechanism_key(row)
            slot = state['mechanisms'].get(key, {}) if isinstance(state.get('mechanisms', {}), dict) else {}
            merged = dict(row)
            merged['object_id'] = str(slot.get('object_id', row.get('object_id', row.get('hypothesis_id', key))) or key)
            merged['type'] = str(row.get('type', 'mechanism_hypothesis') or 'mechanism_hypothesis')
            merged['object_type'] = str(row.get('object_type', OBJECT_TYPE_HYPOTHESIS) or OBJECT_TYPE_HYPOTHESIS)
            merged['hypothesis_type'] = str(
                slot.get('hypothesis_type', row.get('hypothesis_type', 'mechanism_hypothesis'))
                or 'mechanism_hypothesis'
            )
            merged['confidence'] = round(float(slot.get('posterior', row.get('confidence', 0.0)) or 0.0), 4)
            merged['posterior'] = round(float(slot.get('posterior', row.get('confidence', 0.0)) or 0.0), 4)
            merged['support_count'] = int(slot.get('support_count', 0) or 0)
            merged['refute_count'] = int(slot.get('refute_count', 0) or 0)
            merged['contradiction_count'] = int(slot.get('refute_count', row.get('contradiction_count', 0)) or 0)
            merged['last_supported_tick'] = int(slot.get('last_supported_tick', -1) or -1)
            merged['last_refuted_tick'] = int(slot.get('last_refuted_tick', -1) or -1)
            merged['supporting_evidence'] = list(slot.get('supporting_evidence', row.get('supporting_evidence', [])) or [])
            merged['contradicting_evidence'] = list(slot.get('contradicting_evidence', row.get('contradicting_evidence', [])) or [])
            enhanced_mechanisms.append(merged)
        enhanced_mechanisms.sort(key=lambda item: (-float(item.get('posterior', item.get('confidence', 0.0)) or 0.0), str(item.get('hypothesis_id', '') or '')))
        control_summary = self._build_control_summary(state, ranked, tick=tick, obs_before=obs_before)
        return MechanismRuntimeView(mechanisms=enhanced_mechanisms, control_summary=control_summary)

    def update_after_action(
        self,
        runtime_state: Optional[Dict[str, Any]],
        mechanism_hypotheses_summary: Sequence[Dict[str, Any]],
        *,
        action: Dict[str, Any],
        result: Dict[str, Any],
        actual_transition: Optional[Dict[str, Any]] = None,
        reward: float,
        information_gain: float,
        progress_markers: Sequence[Dict[str, Any]],
        obs_before: Optional[Dict[str, Any]],
        tick: int,
        object_store: Any = None,
    ) -> MechanismUpdateResult:
        state = self.ensure_state(runtime_state, mechanism_hypotheses_summary)
        ranked_before = self._rank_mechanisms(state)
        previous_posterior = {row['key']: float(row.get('posterior', 0.0) or 0.0) for row in ranked_before}

        action_family = infer_action_family(action)
        target_info = extract_target_descriptor(action)
        target_family = str(target_info.get('target_family', '') or 'generic_target')
        anchor_ref = str(target_info.get('anchor_ref', '') or '')
        action_meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
        action_role = str(action_meta.get('role', '') or '')
        binding_profile = action_binding_profile(action)
        actual_transition_dict = dict(actual_transition or {})
        runtime_object_graph = build_runtime_object_graph(
            obs_before=obs_before,
            action=action,
            actual_transition=actual_transition_dict,
            result=result,
        )
        actual_tokens = _actual_observation_tokens(result, actual_transition_dict)
        actual_token_set = set(actual_tokens)
        actual_ground_tokens = grounding_token_set(actual_tokens)
        actual_phase = _actual_phase(result, actual_transition_dict)
        state_changed = _state_changed(result)
        progress = _has_progress(reward, progress_markers, result)
        informative = bool(state_changed or information_gain >= 0.18 or _observation_changed(result))
        repeated_dead_end = not state_changed and not progress and information_gain < 0.12
        counterevidence_visible = bool('counterevidence' in actual_token_set)
        uses_signal_evidence = bool(
            actual_token_set
            and (
                action_role == 'discriminate'
                or 'signal_revealed' in actual_token_set
                or 'goal_reached' in actual_token_set
                or actual_phase.startswith('revealed::')
            )
        )

        diagnostics: Dict[str, Any] = {
            'selected_action_family': action_family,
            'selected_target_family': target_family,
            'selected_anchor_ref': anchor_ref,
            'selected_role': action_role,
            'selected_target_key': str(binding_profile.get('target_key', '') or ''),
            'selected_binding_tokens': list(binding_profile.get('binding_tokens', []) or []),
            'state_changed': bool(state_changed),
            'progress': bool(progress),
            'informative': bool(informative),
            'reward': float(reward),
            'information_gain': float(information_gain),
            'actual_observation_tokens': list(actual_tokens),
            'support_updates': [],
            'refute_updates': [],
            'rival_refutes': [],
            'blocked_entries_added': [],
            'blocked_entries_released': [],
            'blocked_entries_decayed': [],
        }

        matched_mechanisms: List[str] = []
        rival_candidates: List[Tuple[str, float]] = []
        signal_supported: List[str] = []
        signal_contradicted: List[str] = []
        signal_matches_by_key: Dict[str, List[str]] = {}
        for row in ranked_before:
            slot = state['mechanisms'].get(row['key'], {}) if isinstance(state.get('mechanisms', {}), dict) else {}
            preferred_refs = set(str(x or '') for x in list(slot.get('preferred_target_refs', []) or []) if str(x or ''))
            preferred_families = set(normalize_action_family(x) for x in list(slot.get('preferred_action_families', []) or []) if normalize_action_family(x))
            mech_discriminators = set(normalize_action_family(x) for x in list(slot.get('best_discriminating_actions', []) or []) if normalize_action_family(x))
            signal_tokens = {
                str(x or '').strip().lower()
                for x in list(slot.get('predicted_observation_tokens', []) or [])
                if str(x or '').strip()
            }
            target_match = bool(anchor_ref and anchor_ref in preferred_refs)
            family_match = bool(action_family and action_family in preferred_families)
            discriminator_match = bool(action_family and action_family in mech_discriminators)
            overlaps = sorted(signal_tokens & actual_token_set)
            if uses_signal_evidence and signal_tokens:
                if overlaps:
                    signal_supported.append(row['key'])
                    signal_matches_by_key[row['key']] = overlaps
                else:
                    signal_contradicted.append(row['key'])
            if target_match or family_match or discriminator_match:
                matched_mechanisms.append(row['key'])
            rival_candidates.append((row['key'], float(row.get('posterior', 0.0) or 0.0)))

        if signal_supported:
            matched_mechanisms = list(signal_supported)
        if not matched_mechanisms and ranked_before:
            matched_mechanisms.append(ranked_before[0]['key'])

        positive_strength = max(0.0, min(1.0, (0.55 if state_changed else 0.0) + (0.65 if progress else 0.0) + min(0.40, max(0.0, information_gain)) * 0.6))
        negative_strength = max(0.0, min(1.0, (0.70 if repeated_dead_end else 0.0) + (0.25 if reward < 0.0 else 0.0) + (0.10 if not informative else 0.0)))
        signal_support_strength = max(positive_strength, 0.88 if action_role == 'discriminate' else 0.72) if signal_supported else positive_strength
        signal_refute_strength = max(negative_strength, 0.95 if action_role == 'discriminate' else 0.72) if signal_supported else 0.0

        for key in matched_mechanisms:
            if positive_strength > 0.0:
                support_strength = signal_support_strength if key in signal_supported else positive_strength
                evidence = f'action_family:{action_family}'
                if key in signal_supported:
                    evidence = f'signal_token:{",".join(signal_matches_by_key.get(key, [])[:2])}'
                self._apply_support(state, key, strength=support_strength, tick=tick, evidence=evidence)
                diagnostics['support_updates'].append({'mechanism': key, 'strength': round(support_strength, 4)})
            if negative_strength > 0.0:
                self._apply_refute(state, key, strength=negative_strength, tick=tick, evidence=f'target_family:{target_family}')
                diagnostics['refute_updates'].append({'mechanism': key, 'strength': round(negative_strength, 4)})

        for key in signal_contradicted:
            if signal_refute_strength <= 0.0:
                continue
            self._apply_refute(state, key, strength=signal_refute_strength, tick=tick, evidence=f'signal_mismatch:{actual_phase or "observation"}')
            diagnostics['refute_updates'].append({'mechanism': key, 'strength': round(signal_refute_strength, 4)})

        # Explicit rival pressure: if one mechanism gets supported by a discriminating step,
        # nearby competitors should weaken instead of being left untouched.
        if positive_strength > 0.0 and matched_mechanisms:
            matched_set = set(matched_mechanisms)
            for key, posterior in rival_candidates[:3]:
                if key in matched_set:
                    continue
                weaken = max(0.0, min(1.0, positive_strength * 0.65 * (0.6 + posterior * 0.4)))
                if weaken <= 0.0:
                    continue
                self._apply_refute(state, key, strength=weaken, tick=tick, evidence=f'rival_of:{matched_mechanisms[0]}')
                diagnostics['rival_refutes'].append({'mechanism': key, 'strength': round(weaken, 4)})

        # Fast family cooldowns: action+target pair cools quickly; target family cools a bit broader.
        action_target_key = f'{action_family}::{target_family}'
        action_target_state = state.setdefault('action_target_state', {})
        target_family_state = state.setdefault('target_family_state', {})
        at_bucket = dict(action_target_state.get(action_target_key, {}))
        tf_bucket = dict(target_family_state.get(target_family, {}))
        if repeated_dead_end:
            at_bucket['dead_streak'] = int(at_bucket.get('dead_streak', 0) or 0) + 1
            tf_bucket['dead_streak'] = int(tf_bucket.get('dead_streak', 0) or 0) + 1
            at_bucket['no_state_streak'] = int(at_bucket.get('no_state_streak', 0) or 0) + (0 if state_changed else 1)
            tf_bucket['no_state_streak'] = int(tf_bucket.get('no_state_streak', 0) or 0) + (0 if state_changed else 1)
            at_bucket['no_progress_streak'] = int(at_bucket.get('no_progress_streak', 0) or 0) + (0 if progress else 1)
            tf_bucket['no_progress_streak'] = int(tf_bucket.get('no_progress_streak', 0) or 0) + (0 if progress else 1)
            at_bucket['low_info_streak'] = int(at_bucket.get('low_info_streak', 0) or 0) + (1 if information_gain < 0.12 else 0)
            tf_bucket['low_info_streak'] = int(tf_bucket.get('low_info_streak', 0) or 0) + (1 if information_gain < 0.12 else 0)
        else:
            at_bucket['dead_streak'] = 0
            at_bucket['no_state_streak'] = 0 if state_changed else int(at_bucket.get('no_state_streak', 0) or 0)
            at_bucket['no_progress_streak'] = 0 if progress else int(at_bucket.get('no_progress_streak', 0) or 0)
            at_bucket['low_info_streak'] = 0 if information_gain >= 0.12 else int(at_bucket.get('low_info_streak', 0) or 0)
            tf_bucket['dead_streak'] = 0 if progress or state_changed or information_gain >= 0.18 else int(tf_bucket.get('dead_streak', 0) or 0)
            if state_changed or progress:
                tf_bucket['no_state_streak'] = 0
                tf_bucket['no_progress_streak'] = 0
                tf_bucket['low_info_streak'] = 0

        if int(at_bucket.get('dead_streak', 0) or 0) >= 2:
            at_bucket['cooldown_until_tick'] = max(int(at_bucket.get('cooldown_until_tick', -1) or -1), tick + self._fast_cooldown_ticks)
            diagnostics['blocked_entries_added'].append({'scope': 'action_target', 'key': action_target_key, 'until_tick': at_bucket['cooldown_until_tick']})
        if int(tf_bucket.get('dead_streak', 0) or 0) >= 2:
            tf_bucket['cooldown_until_tick'] = max(int(tf_bucket.get('cooldown_until_tick', -1) or -1), tick + self._fast_cooldown_ticks)
            diagnostics['blocked_entries_added'].append({'scope': 'target_family', 'key': target_family, 'until_tick': tf_bucket['cooldown_until_tick']})
        if int(tf_bucket.get('dead_streak', 0) or 0) >= 3:
            tf_bucket['hard_cooldown_until_tick'] = max(int(tf_bucket.get('hard_cooldown_until_tick', -1) or -1), tick + self._hard_cooldown_ticks)
            diagnostics['blocked_entries_added'].append({'scope': 'target_family_hard', 'key': target_family, 'until_tick': tf_bucket['hard_cooldown_until_tick']})

        target_binding_update = self._update_target_binding_state(
            state,
            binding_profile=binding_profile,
            actual_ground_tokens=actual_ground_tokens,
            runtime_object_graph=runtime_object_graph,
            action_role=action_role,
            state_changed=bool(state_changed),
            progress=bool(progress),
            informative=bool(informative),
            repeated_dead_end=bool(repeated_dead_end),
            counterevidence_visible=bool(counterevidence_visible),
            reward=float(reward),
            information_gain=float(information_gain),
            tick=tick,
        )
        diagnostics['target_binding_update'] = dict(target_binding_update)
        cooldown_release = self._apply_evidence_driven_cooldown_decay(
            action_target_key=action_target_key,
            target_family=target_family,
            action_role=action_role,
            actual_phase=actual_phase,
            actual_token_set=actual_token_set,
            state_changed=bool(state_changed),
            progress=bool(progress),
            informative=bool(informative),
            information_gain=float(information_gain),
            target_binding_update=target_binding_update,
            action_target_bucket=at_bucket,
            target_family_bucket=tf_bucket,
            tick=tick,
        )
        diagnostics['blocked_entries_released'] = list(cooldown_release.get('released', []))
        diagnostics['blocked_entries_decayed'] = list(cooldown_release.get('decayed', []))
        action_target_state[action_target_key] = at_bucket
        target_family_state[target_family] = tf_bucket
        state['runtime_object_graph'] = dict(runtime_object_graph)

        ranked_after = self._rank_mechanisms(state)
        motion = 0.0
        for row in ranked_after:
            motion += abs(float(row.get('posterior', 0.0) or 0.0) - float(previous_posterior.get(row['key'], 0.0) or 0.0))
        state['last_posterior_motion'] = round(motion, 6)
        if motion < 0.05 and not progress:
            state['stagnant_ticks'] = int(state.get('stagnant_ticks', 0) or 0) + 1
        else:
            state['stagnant_ticks'] = 0 if progress or motion >= 0.05 else int(state.get('stagnant_ticks', 0) or 0)
        state['last_tick'] = int(tick)
        state['last_selected_action_family'] = action_family
        state['last_selected_target_family'] = target_family
        state['last_selected_function'] = extract_action_function_name(action, default='')

        runtime_view = self.build_runtime_view(state, mechanism_hypotheses_summary, tick=tick, obs_before=obs_before)
        durable_object_ids = _sync_mechanism_records_to_store(object_store, runtime_view.mechanisms)
        control_summary = dict(runtime_view.control_summary or {})
        if durable_object_ids:
            control_summary['durable_mechanism_object_ids'] = list(durable_object_ids)
        state['last_policy_delta'] = dict(control_summary)
        diagnostics['posterior_motion'] = round(motion, 6)
        diagnostics['dominant_mechanism_family'] = str(control_summary.get('dominant_mechanism_family', '') or '')
        diagnostics['dominant_mechanism_confidence'] = float(control_summary.get('dominant_mechanism_confidence', 0.0) or 0.0)
        diagnostics['require_discriminating_action'] = bool(control_summary.get('require_discriminating_action', False))
        diagnostics['durable_mechanism_object_ids'] = list(durable_object_ids)

        return MechanismUpdateResult(
            state=state,
            control_summary=control_summary,
            diagnostics=diagnostics,
            mechanisms=list(runtime_view.mechanisms or []),
            durable_object_ids=list(durable_object_ids),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_support(self, state: Dict[str, Any], key: str, *, strength: float, tick: int, evidence: str) -> None:
        slot = state['mechanisms'].get(key, {}) if isinstance(state.get('mechanisms', {}), dict) else {}
        if not slot:
            return
        current = float(slot.get('posterior', slot.get('base_confidence', 0.0)) or 0.0)
        base = float(slot.get('base_confidence', current) or current)
        slot['posterior'] = _clamp01(current * 0.72 + base * 0.10 + self._support_gain * max(0.0, min(1.0, strength)), current)
        slot['support_count'] = int(slot.get('support_count', 0) or 0) + 1
        slot['last_supported_tick'] = int(tick)
        if evidence:
            evidences = list(slot.get('supporting_evidence', []) or [])
            if evidence not in evidences:
                evidences.append(evidence)
            slot['supporting_evidence'] = evidences[-6:]
        state['mechanisms'][key] = slot

    def _apply_refute(self, state: Dict[str, Any], key: str, *, strength: float, tick: int, evidence: str) -> None:
        slot = state['mechanisms'].get(key, {}) if isinstance(state.get('mechanisms', {}), dict) else {}
        if not slot:
            return
        current = float(slot.get('posterior', slot.get('base_confidence', 0.0)) or 0.0)
        base = float(slot.get('base_confidence', current) or current)
        slot['posterior'] = _clamp01(current * 0.68 + base * 0.06 - self._refute_gain * max(0.0, min(1.0, strength)), 0.0)
        slot['refute_count'] = int(slot.get('refute_count', 0) or 0) + 1
        slot['last_refuted_tick'] = int(tick)
        if evidence:
            evidences = list(slot.get('contradicting_evidence', []) or [])
            if evidence not in evidences:
                evidences.append(evidence)
            slot['contradicting_evidence'] = evidences[-6:]
        state['mechanisms'][key] = slot

    def _rank_mechanisms(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        for key, slot in (state.get('mechanisms', {}) or {}).items():
            if not isinstance(slot, dict):
                continue
            rows.append({
                'key': str(key),
                'object_id': str(slot.get('object_id', slot.get('hypothesis_id', key)) or key),
                'family': str(slot.get('family', key) or key),
                'hypothesis_id': str(slot.get('hypothesis_id', key) or key),
                'posterior': _clamp01(slot.get('posterior', slot.get('base_confidence', 0.0)), 0.0),
                'base_confidence': _clamp01(slot.get('base_confidence', 0.0), 0.0),
            })
        rows.sort(key=lambda item: (-float(item.get('posterior', 0.0) or 0.0), str(item.get('hypothesis_id', '') or '')))
        return rows

    def _update_target_binding_state(
        self,
        state: Dict[str, Any],
        *,
        binding_profile: Dict[str, Any],
        actual_ground_tokens: Sequence[str],
        runtime_object_graph: Optional[Dict[str, Any]],
        action_role: str,
        state_changed: bool,
        progress: bool,
        informative: bool,
        repeated_dead_end: bool,
        counterevidence_visible: bool,
        reward: float,
        information_gain: float,
        tick: int,
    ) -> Dict[str, Any]:
        target_key = str(binding_profile.get('target_key', '') or '')
        if not target_key:
            return {'target_key': '', 'support': 0.0, 'contradiction': 0.0, 'trust': 0.0, 'active_commitment_revoked': False}

        target_family = str(binding_profile.get('target_family', '') or 'generic_target')
        anchor_ref = str(binding_profile.get('anchor_ref', '') or '')
        action_family = str(binding_profile.get('action_family', '') or '')
        binding_tokens = grounding_token_set(binding_profile.get('binding_tokens', []))
        actual_tokens = grounding_token_set(actual_ground_tokens)
        overlap = sorted(binding_tokens & actual_tokens)
        runtime_graph = dict(runtime_object_graph or {})
        focus_object_ids = [
            str(item or '')
            for item in list(runtime_graph.get('focus_object_ids', []) or [])
            if str(item or '')
        ]
        scene_state = runtime_graph.get('scene_state', {}) if isinstance(runtime_graph.get('scene_state', {}), dict) else {}
        signal_tokens = grounding_token_set(scene_state.get('signal_tokens', []))
        counter_tokens = grounding_token_set(scene_state.get('counterevidence_tokens', []))
        scene_overlap = sorted(binding_tokens & signal_tokens)

        support = 0.0
        contradiction = 0.0
        if overlap:
            support += min(0.55, 0.22 + 0.16 * len(overlap))
        if scene_overlap:
            support += min(0.22, 0.08 + 0.08 * len(scene_overlap))
        if state_changed:
            support += 0.12
        if progress:
            support += 0.22
        if information_gain >= 0.18:
            support += 0.08 if action_role in {'discriminate', 'observe'} else 0.04
        if focus_object_ids:
            support += min(0.14, 0.04 + 0.04 * len(focus_object_ids))

        if repeated_dead_end:
            contradiction += 0.24
        if reward < 0.0:
            contradiction += 0.12
        if counterevidence_visible and not overlap:
            contradiction += 0.42
        if counter_tokens and binding_tokens and not (binding_tokens & counter_tokens):
            contradiction += 0.12 if action_role == 'commit' else 0.06
        elif informative and action_role in {'commit', 'recovery', 'prerequisite'} and actual_tokens and binding_tokens and not overlap:
            contradiction += 0.18

        target_bindings = state.setdefault('target_bindings', {})
        slot = dict(target_bindings.get(target_key, {})) if isinstance(target_bindings.get(target_key, {}), dict) else {}
        support_mass = float(slot.get('support_mass', 0.0) or 0.0) * 0.72 + support
        contradiction_mass = float(slot.get('contradiction_mass', 0.0) or 0.0) * 0.76 + contradiction
        trust = _clamp01((support_mass + 0.18) / max(0.32, support_mass + contradiction_mass + 0.32), 0.0)
        slot.update({
            'target_key': target_key,
            'target_family': target_family,
            'anchor_ref': anchor_ref,
            'action_family': action_family,
            'binding_tokens': sorted(binding_tokens),
            'support_mass': round(support_mass, 6),
            'contradiction_mass': round(contradiction_mass, 6),
            'trust': round(trust, 6),
            'last_tick': int(tick),
            'last_supported_tick': int(tick if support > 0.0 else slot.get('last_supported_tick', -1) or -1),
            'last_refuted_tick': int(tick if contradiction > 0.0 else slot.get('last_refuted_tick', -1) or -1),
            'support_count': int(slot.get('support_count', 0) or 0) + (1 if support > contradiction and support > 0.0 else 0),
            'contradiction_count': int(slot.get('contradiction_count', 0) or 0) + (1 if contradiction > 0.0 else 0),
        })
        target_bindings[target_key] = slot

        active = dict(state.get('active_commitment', {})) if isinstance(state.get('active_commitment', {}), dict) else {}
        active_key = str(active.get('target_key', '') or '')
        active_revoked = bool(active.get('revoked', False))
        contradiction_spike = bool(
            contradiction >= max(0.34, support + 0.14)
            or (counterevidence_visible and not overlap)
            or (repeated_dead_end and support <= 0.06)
        )
        if action_role == 'commit':
            if active_key != target_key:
                active = {
                    'target_key': target_key,
                    'target_family': target_family,
                    'anchor_ref': anchor_ref,
                    'action_family': action_family,
                    'binding_tokens': sorted(binding_tokens),
                    'trust': round(trust, 6),
                    'revoked': False,
                    'revoked_tick': -1,
                    'contradiction_pressure': 0.0,
                    'last_updated_tick': int(tick),
                }
            else:
                active.update({
                    'target_family': target_family,
                    'anchor_ref': anchor_ref,
                    'action_family': action_family,
                    'binding_tokens': sorted(binding_tokens),
                    'trust': round(trust, 6),
                    'last_updated_tick': int(tick),
                })
            if contradiction_spike:
                active['revoked'] = True
                active['revoked_tick'] = int(tick)
                active['contradiction_pressure'] = round(float(active.get('contradiction_pressure', 0.0) or 0.0) * 0.55 + contradiction, 6)
                active_revoked = True
            elif support > contradiction:
                active['revoked'] = False
                active['revoked_tick'] = -1
                active['contradiction_pressure'] = round(max(0.0, float(active.get('contradiction_pressure', 0.0) or 0.0) * 0.5 - support * 0.35), 6)
                active_revoked = False
            state['active_commitment'] = active
        elif active_key == target_key and contradiction_spike:
            active['revoked'] = True
            active['revoked_tick'] = int(tick)
            active['trust'] = round(trust, 6)
            active['contradiction_pressure'] = round(float(active.get('contradiction_pressure', 0.0) or 0.0) * 0.55 + contradiction, 6)
            active['last_updated_tick'] = int(tick)
            state['active_commitment'] = active
            active_revoked = True
        elif active_key == target_key and support > contradiction:
            active['trust'] = round(trust, 6)
            active['contradiction_pressure'] = round(max(0.0, float(active.get('contradiction_pressure', 0.0) or 0.0) * 0.5 - support * 0.25), 6)
            active['last_updated_tick'] = int(tick)
            state['active_commitment'] = active
            active_revoked = bool(active.get('revoked', False))

        return {
            'target_key': target_key,
            'support': round(support, 6),
            'contradiction': round(contradiction, 6),
            'trust': round(trust, 6),
            'matched_tokens': overlap[:4],
            'matched_signal_tokens': scene_overlap[:4],
            'focus_object_ids': focus_object_ids[:4],
            'active_commitment_revoked': bool(active_revoked),
        }

    def _rank_target_bindings(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, slot in (state.get('target_bindings', {}) or {}).items():
            if not isinstance(slot, dict):
                continue
            support_mass = float(slot.get('support_mass', 0.0) or 0.0)
            contradiction_mass = float(slot.get('contradiction_mass', 0.0) or 0.0)
            trust = _clamp01(slot.get('trust', 0.0), 0.0)
            rows.append({
                'target_key': str(key),
                'target_family': str(slot.get('target_family', '') or ''),
                'anchor_ref': str(slot.get('anchor_ref', '') or ''),
                'binding_tokens': list(slot.get('binding_tokens', []) or []),
                'trust': trust,
                'support_mass': support_mass,
                'contradiction_mass': contradiction_mass,
                'last_tick': int(slot.get('last_tick', -1) or -1),
            })
        rows.sort(
            key=lambda item: (
                -float(item.get('trust', 0.0) or 0.0),
                -float(item.get('support_mass', 0.0) or 0.0),
                float(item.get('contradiction_mass', 0.0) or 0.0),
                str(item.get('target_key', '') or ''),
            )
        )
        return rows

    def _build_control_summary(
        self,
        state: Dict[str, Any],
        ranked: Sequence[Dict[str, Any]],
        *,
        tick: int,
        obs_before: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        top = ranked[0] if ranked else {}
        runner_up = ranked[1] if len(ranked) > 1 else {}
        probs = [max(1e-6, float(row.get('posterior', 0.0) or 0.0)) for row in ranked[:5]]
        z = sum(probs) or 1.0
        norm = [p / z for p in probs]
        entropy = 0.0
        if norm:
            entropy = -sum(p * math.log(p, 2) for p in norm if p > 0.0) / max(1.0, math.log(len(norm), 2))
        margin = float(top.get('posterior', 0.0) or 0.0) - float(runner_up.get('posterior', 0.0) or 0.0)
        target_ranked = self._rank_target_bindings(state)
        top_target = target_ranked[0] if target_ranked else {}
        runner_up_target = target_ranked[1] if len(target_ranked) > 1 else {}
        target_margin = float(top_target.get('trust', 0.0) or 0.0) - float(runner_up_target.get('trust', 0.0) or 0.0)
        obs_snapshot = dict(obs_before or {}) if isinstance(obs_before, dict) else {}
        signal_tokens = sorted(grounding_token_set(obs_snapshot.get('revealed_signal_token', '')))
        counterevidence_tokens = sorted(grounding_token_set(obs_snapshot.get('counterevidence_token', '')))
        active_commitment = dict(state.get('active_commitment', {})) if isinstance(state.get('active_commitment', {}), dict) else {}
        active_tokens = grounding_token_set(active_commitment.get('binding_tokens', []))
        active_trust = float(active_commitment.get('trust', 0.0) or 0.0)
        contradiction_pressure = float(active_commitment.get('contradiction_pressure', 0.0) or 0.0)
        current_counterevidence_revocation = bool(counterevidence_tokens and active_tokens and active_tokens.isdisjoint(counterevidence_tokens))
        commitment_revoked = bool(active_commitment.get('revoked', False) or current_counterevidence_revocation)
        runtime_object_graph = state.get('runtime_object_graph', {}) if isinstance(state.get('runtime_object_graph', {}), dict) else {}

        preferred_target_refs: List[str] = []
        preferred_action_families: List[str] = []
        discriminating_actions: List[str] = []
        mechanism_families: List[str] = []
        for row in ranked[:2]:
            slot = state['mechanisms'].get(row['key'], {}) if isinstance(state.get('mechanisms', {}), dict) else {}
            family = str(slot.get('family', row.get('family', '')) or '')
            if family and family not in mechanism_families:
                mechanism_families.append(family)
            for value in list(slot.get('preferred_target_refs', []) or []):
                text = str(value or '').strip()
                if text and text not in preferred_target_refs:
                    preferred_target_refs.append(text)
            for value in list(slot.get('preferred_action_families', []) or []):
                text = normalize_action_family(value)
                if text and text not in preferred_action_families:
                    preferred_action_families.append(text)
            for value in list(slot.get('best_discriminating_actions', []) or []):
                text = normalize_action_family(value)
                if text and text not in discriminating_actions:
                    discriminating_actions.append(text)

        cooldown_state = compile_mechanism_cooldown_state(
            target_family_state=state.get('target_family_state', {}) if isinstance(state.get('target_family_state', {}), dict) else {},
            action_target_state=state.get('action_target_state', {}) if isinstance(state.get('action_target_state', {}), dict) else {},
            tick=int(tick),
            active_commitment=active_commitment,
            commitment_revoked=commitment_revoked,
            fast_cooldown_ticks=self._fast_cooldown_ticks,
        )
        blocked_entries = list(cooldown_state.get('blocked_entries', []) or [])
        blocked_lists = mechanism_block_lists({'blocked_entries': blocked_entries})
        blocked_target_families = list(blocked_lists.get('blocked_target_families', []) or [])
        blocked_action_families = list(blocked_lists.get('blocked_action_families', []) or [])
        blocked_action_targets = list(blocked_lists.get('blocked_action_targets', []) or [])
        visible_functions = extract_available_functions(obs_before or {}) if isinstance(obs_before, dict) else []

        mode_decision = infer_mode(
            ranked_mechanisms=ranked,
            target_ranked=target_ranked,
            active_commitment=active_commitment,
            obs_before=obs_snapshot,
            entropy=float(entropy),
            margin=float(margin),
            stagnant_ticks=int(state.get('stagnant_ticks', 0) or 0),
            low_confidence_threshold=self._low_confidence_threshold,
            uncertainty_margin_threshold=self._uncertainty_margin_threshold,
        )
        control_mode = str(mode_decision.control_mode or 'discriminate')
        require_discriminating = bool(mode_decision.require_discriminating_action)
        unresolved: List[str] = list(mode_decision.unresolved_dimensions or [])
        if not discriminating_actions:
            unresolved.append('no_discriminating_actions')

        control_summary = {
            'dominant_mechanism_family': str(top.get('family', '') or ''),
            'dominant_mechanism_confidence': round(float(top.get('posterior', 0.0) or 0.0), 4),
            'dominant_mechanism_ref': str(top.get('hypothesis_id', '') or ''),
            'dominant_mechanism_object_id': str(top.get('object_id', '') or ''),
            'ranked_mechanism_object_ids': [
                str(item.get('object_id', '') or '')
                for item in ranked[:6]
                if str(item.get('object_id', '') or '')
            ],
            'preferred_target_refs': preferred_target_refs[:6],
            'preferred_action_families': preferred_action_families[:5],
            'discriminating_actions': discriminating_actions[:5] or preferred_action_families[:3],
            'mechanism_families': mechanism_families[:6],
            'mechanism_ready': len(unresolved) == 0,
            'unresolved_mechanism_dimensions': unresolved,
            'posterior_entropy': round(float(entropy), 4),
            'posterior_margin': round(float(margin), 4),
            'control_mode': control_mode,
            'signal_tokens': signal_tokens[:6],
            'counterevidence_tokens': counterevidence_tokens[:6],
            'target_binding_margin': round(float(target_margin), 4),
            'target_binding_leader': str(top_target.get('target_key', '') or ''),
            'target_binding_trust': round(float(top_target.get('trust', 0.0) or 0.0), 4),
            'active_commitment': {
                'target_key': str(active_commitment.get('target_key', '') or ''),
                'target_family': str(active_commitment.get('target_family', '') or ''),
                'anchor_ref': str(active_commitment.get('anchor_ref', '') or ''),
                'binding_tokens': list(active_commitment.get('binding_tokens', []) or [])[:6],
                'trust': round(float(active_trust), 4),
                'revoked': bool(commitment_revoked),
            },
            'commitment_trust': round(float(active_trust), 4),
            'commitment_revoked': bool(commitment_revoked),
            'contradiction_pressure': round(float(contradiction_pressure + (0.55 if current_counterevidence_revocation else 0.0)), 4),
            'require_discriminating_action': bool(require_discriminating),
            'mode_diagnostics': dict(mode_decision.diagnostics or {}),
            'cooldown_state': cooldown_state,
            'blocked_entries': blocked_entries,
            'blocked_target_families': blocked_target_families[:8],
            'blocked_action_families': blocked_action_families[:8],
            'blocked_action_targets': blocked_action_targets[:8],
            'runtime_object_graph': {
                'object_count': int(runtime_object_graph.get('object_count', len(runtime_object_graph.get('objects', []) or [])) or 0),
                'relation_count': int(runtime_object_graph.get('relation_count', len(runtime_object_graph.get('relations', []) or [])) or 0),
                'focus_object_ids': list(runtime_object_graph.get('focus_object_ids', []) or [])[:6],
            },
            'stagnant_ticks': int(state.get('stagnant_ticks', 0) or 0),
            'last_selected_action_family': str(state.get('last_selected_action_family', '') or ''),
            'last_selected_target_family': str(state.get('last_selected_target_family', '') or ''),
            'last_selected_function': str(state.get('last_selected_function', '') or ''),
            'last_posterior_motion': float(state.get('last_posterior_motion', 0.0) or 0.0),
            'visible_functions': visible_functions[:12],
        }
        return enrich_mechanism_control_summary(control_summary, obs_snapshot)

    def _mechanism_key(self, row: Dict[str, Any]) -> str:
        object_id = str(row.get('object_id', '') or '')
        hyp = str(row.get('hypothesis_id', '') or '')
        family = str(row.get('family', '') or '')
        return object_id or hyp or family

    def _apply_evidence_driven_cooldown_decay(
        self,
        *,
        action_target_key: str,
        target_family: str,
        action_role: str,
        actual_phase: str,
        actual_token_set: Sequence[str],
        state_changed: bool,
        progress: bool,
        informative: bool,
        information_gain: float,
        target_binding_update: Dict[str, Any],
        action_target_bucket: Dict[str, Any],
        target_family_bucket: Dict[str, Any],
        tick: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        actual_tokens = {
            str(item or '').strip().lower()
            for item in list(actual_token_set or [])
            if str(item or '').strip()
        }
        support = float(target_binding_update.get('support', 0.0) or 0.0)
        contradiction = float(target_binding_update.get('contradiction', 0.0) or 0.0)
        support_margin = support - contradiction
        contradictory_transition = bool(
            target_binding_update.get('active_commitment_revoked', False)
            or actual_tokens.intersection({'counterevidence', 'contradiction'})
            or contradiction >= max(0.18, support + 0.04)
        )
        phase_progress = str(actual_phase or '').strip().lower().startswith(('revealed', 'configured', 'committed', 'resolved'))
        strong_transition_signal = bool(
            actual_tokens.intersection({'signal_revealed', 'goal_reached', 'goal_progressed', 'terminal_reached'})
            or phase_progress
        )
        strong_release = bool(
            progress
            or strong_transition_signal
            or ((state_changed and information_gain >= 0.18) and not contradictory_transition)
            or ((state_changed and support_margin >= 0.08) and not contradictory_transition)
            or ((action_role in {'wait', 'prerequisite', 'prepare', 'recovery'} and state_changed) and not contradictory_transition)
        )
        soft_decay = not strong_release and not contradictory_transition and bool(
            (informative and support_margin >= -0.02)
            or support_margin > 0.0
            or information_gain >= 0.14
            or actual_tokens.intersection({'signal_partial', 'state_change'})
        )

        released: List[Dict[str, Any]] = []
        decayed: List[Dict[str, Any]] = []
        release_reason = (
            'runtime_progress_release'
            if progress
            else 'runtime_transition_signal_release'
            if strong_transition_signal
            else 'runtime_state_change_release'
            if state_changed
            else 'runtime_binding_support_release'
        )
        decay_reason = (
            'runtime_binding_support_decay'
            if support_margin > 0.0
            else 'runtime_informative_decay'
        )

        if strong_release:
            if self._clear_cooldown_bucket(action_target_bucket, tick=tick, reason=release_reason):
                released.append({'scope': 'action_target', 'key': str(action_target_key or ''), 'reason': release_reason})
            if self._clear_cooldown_bucket(target_family_bucket, tick=tick, reason=release_reason):
                released.append({'scope': 'target_family', 'key': str(target_family or ''), 'reason': release_reason})
        elif soft_decay:
            if self._decay_cooldown_bucket(action_target_bucket, tick=tick, reason=decay_reason):
                decayed.append({'scope': 'action_target', 'key': str(action_target_key or ''), 'reason': decay_reason})

        return {'released': released, 'decayed': decayed}

    def _clear_cooldown_bucket(
        self,
        bucket: Dict[str, Any],
        *,
        tick: int,
        reason: str,
    ) -> bool:
        if not isinstance(bucket, dict):
            return False
        active = bool(
            int(bucket.get('cooldown_until_tick', -1) or -1) >= int(tick)
            or int(bucket.get('hard_cooldown_until_tick', -1) or -1) >= int(tick)
            or int(bucket.get('dead_streak', 0) or 0) > 0
            or int(bucket.get('no_state_streak', 0) or 0) > 0
            or int(bucket.get('no_progress_streak', 0) or 0) > 0
            or int(bucket.get('low_info_streak', 0) or 0) > 0
        )
        if not active:
            return False
        bucket['dead_streak'] = 0
        bucket['no_state_streak'] = 0
        bucket['no_progress_streak'] = 0
        bucket['low_info_streak'] = 0
        bucket['cooldown_until_tick'] = int(tick) - 1
        if 'hard_cooldown_until_tick' in bucket:
            bucket['hard_cooldown_until_tick'] = int(tick) - 1
        bucket['last_evidence_release_tick'] = int(tick)
        bucket['last_evidence_release_reason'] = str(reason or '')
        return True

    def _decay_cooldown_bucket(
        self,
        bucket: Dict[str, Any],
        *,
        tick: int,
        reason: str,
    ) -> bool:
        if not isinstance(bucket, dict):
            return False
        changed = False
        for key in ('dead_streak', 'no_state_streak', 'no_progress_streak', 'low_info_streak'):
            value = int(bucket.get(key, 0) or 0)
            if value > 0:
                bucket[key] = max(0, value - 1)
                changed = True
        cooldown_until = int(bucket.get('cooldown_until_tick', -1) or -1)
        if cooldown_until >= int(tick):
            new_until = min(cooldown_until, int(tick) + 1)
            if new_until != cooldown_until:
                bucket['cooldown_until_tick'] = new_until
                changed = True
        hard_until = int(bucket.get('hard_cooldown_until_tick', -1) or -1)
        if hard_until >= int(tick):
            bucket['hard_cooldown_until_tick'] = hard_until
        if changed:
            bucket['last_evidence_decay_tick'] = int(tick)
            bucket['last_evidence_decay_reason'] = str(reason or '')
        return changed


# ----------------------------------------------------------------------
# Shared helper functions used by selector/cooldown gate/tests
# ----------------------------------------------------------------------


def normalize_action_family(value: Any) -> str:
    text = str(value or '').strip().lower()
    if not text:
        return ''
    if text in ACTION_FAMILY_ORDER:
        return text
    upper = text.upper()
    if upper in {'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'}:
        return 'navigation_interaction'
    if upper in {'ACTION5', 'CONFIRM', 'INTERACT', 'SUBMIT', 'ENTER', 'APPLY'}:
        return 'confirm_interaction'
    if upper in {'ACTION6', 'CLICK', 'TAP', 'POINTER_CLICK', 'POINTER_SELECT', 'POINTER_ACTIVATE', 'SELECT'}:
        return 'pointer_interaction'
    if upper in {'ACTION7', 'PROBE', 'PROBE_STATE_CHANGE', 'PROBE_RELATION', 'DRAG', 'TOGGLE', 'TRANSFORM'}:
        return 'state_transform_interaction'
    if 'nav' in text or text in {'move', 'left', 'right', 'up', 'down', 'focus'}:
        return 'navigation_interaction'
    if 'confirm' in text or 'submit' in text or 'interact' in text or text == 'wait':
        return 'confirm_interaction' if text != 'wait' else 'wait'
    if 'pointer' in text or 'click' in text or 'tap' in text or 'select' in text:
        return 'pointer_interaction'
    if 'probe' in text or 'transform' in text or 'toggle' in text:
        return 'state_transform_interaction'
    return 'state_transform_interaction'


def canonical_target_family(value: Any) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    for prefix in ('target_kind::', 'effect::', 'semantic::', 'role::', 'anchor::', 'goal::'):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def infer_action_family(action: Optional[Dict[str, Any]]) -> str:
    if not isinstance(action, dict):
        return 'wait'
    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    for key in (
        'action_family',
        'solver_dominant_interaction_mode',
    ):
        family = normalize_action_family(meta.get(key))
        if family:
            return family
    mechanism_guidance = meta.get('mechanism_guidance', {}) if isinstance(meta.get('mechanism_guidance', {}), dict) else {}
    discriminators = list(mechanism_guidance.get('discriminating_actions', []) or [])
    if len(discriminators) == 1:
        family = normalize_action_family(discriminators[0])
        if family:
            return family
    return normalize_action_family(extract_action_function_name(action, default='wait'))


def grounding_token_set(*values: Any) -> set[str]:
    ordered: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            ordered.update(grounding_token_set(*list(value)))
            continue
        text = str(value or '').strip().lower()
        if not text:
            continue
        fragments = text.replace('::', '_').replace('-', '_').split()
        raw_parts: List[str] = []
        for fragment in fragments:
            raw_parts.extend(fragment.replace('_', ' ').split())
        for token in raw_parts:
            normalized = str(token or '').strip().lower()
            if not normalized or normalized in GROUNDING_STOPWORDS:
                continue
            ordered.add(normalized)
    return ordered


def action_binding_profile(action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    target_desc = extract_target_descriptor(action)
    meta = action.get('_candidate_meta', {}) if isinstance(action, dict) and isinstance(action.get('_candidate_meta', {}), dict) else {}
    target_family = canonical_target_family(target_desc.get('target_family', '') or 'generic_target')
    anchor_ref = str(target_desc.get('anchor_ref', '') or '')
    if target_family and anchor_ref:
        target_key = f'{target_family}@@{anchor_ref}'
    else:
        target_key = target_family or anchor_ref or 'generic_target'
    binding_tokens = grounding_token_set(
        meta.get('grounded_binding_tokens', []),
        anchor_ref,
        target_family,
    )
    return {
        'target_key': target_key,
        'target_family': target_family,
        'anchor_ref': anchor_ref,
        'action_family': infer_action_family(action),
        'role': str(meta.get('role', '') or ''),
        'binding_tokens': sorted(binding_tokens),
    }


def binding_token_frequency(actions: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for action in list(actions or []):
        profile = action_binding_profile(action)
        for token in set(profile.get('binding_tokens', []) or []):
            freq[token] = int(freq.get(token, 0) or 0) + 1
    return freq


def candidate_binding_signal(
    action: Dict[str, Any],
    *,
    obs_before: Optional[Dict[str, Any]],
    mechanism_control: Optional[Dict[str, Any]],
    token_frequency: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    control = dict(mechanism_control or {})
    obs = dict(obs_before or {})
    profile = action_binding_profile(action)
    binding_tokens = set(profile.get('binding_tokens', []) or [])
    signal_tokens = grounding_token_set(control.get('signal_tokens', []), obs.get('revealed_signal_token', ''))
    counterevidence_tokens = grounding_token_set(control.get('counterevidence_tokens', []), obs.get('counterevidence_token', ''))
    active = dict(control.get('active_commitment', {}) or {}) if isinstance(control.get('active_commitment', {}), dict) else {}
    active_target_key = str(active.get('target_key', '') or '')
    active_target_family = str(active.get('target_family', '') or '')
    blocked_entries = mechanism_blocked_entries(control)
    blocked_lists = mechanism_block_lists(control)
    blocked_target_families = {
        canonical_target_family(value)
        for value in list(blocked_lists.get('blocked_target_families', []) or [])
        if canonical_target_family(value)
    }
    blocked_action_families = {
        normalize_action_family(value)
        for value in list(blocked_lists.get('blocked_action_families', []) or [])
        if normalize_action_family(value)
    }
    blocked_action_targets = {
        str(value or '').strip()
        for value in list(blocked_lists.get('blocked_action_targets', []) or [])
        if str(value or '').strip()
    }
    hard_blocked_targets = {
        canonical_target_family(entry.get('value', ''))
        for entry in blocked_entries
        if isinstance(entry, dict) and entry.get('scope') == 'target_family' and bool(entry.get('hard', False)) and canonical_target_family(entry.get('value', ''))
    }
    hard_blocked_action_targets = {
        str(entry.get('value', '') or '').strip()
        for entry in blocked_entries
        if isinstance(entry, dict) and entry.get('scope') == 'action_target' and bool(entry.get('hard', False))
    }
    action_target_key = ''
    if str(profile.get('action_family', '') or '') and str(profile.get('target_family', '') or ''):
        action_target_key = f"{profile['action_family']}::{profile['target_family']}"
    active_match = bool(
        active_target_key
        and (
            active_target_key == str(profile.get('target_key', '') or '')
            or active_target_family == str(profile.get('target_family', '') or '')
        )
    )
    revoked = bool(active.get('revoked', False) or control.get('commitment_revoked', False))
    blocked_target_match = bool(canonical_target_family(profile.get('target_family', '') or '') in blocked_target_families)
    blocked_action_family_match = bool(str(profile.get('action_family', '') or '') in blocked_action_families)
    blocked_action_target_match = bool(action_target_key and action_target_key in blocked_action_targets)
    hard_blocked_match = bool(
        canonical_target_family(profile.get('target_family', '') or '') in hard_blocked_targets
        or (action_target_key and action_target_key in hard_blocked_action_targets)
    )
    weight = lambda token: 1.0 / max(1, int((token_frequency or {}).get(token, 1) or 1))
    support_matches = sorted(binding_tokens & signal_tokens)
    counter_matches = sorted(binding_tokens & counterevidence_tokens)
    support_score = sum(weight(token) for token in support_matches)
    counter_score = sum(weight(token) * 1.35 for token in counter_matches)
    contradiction_penalty = 0.0
    if active_match and revoked:
        contradiction_penalty += 0.85
    elif active_match and counterevidence_tokens and not counter_matches:
        contradiction_penalty += 0.55
    if hard_blocked_match:
        contradiction_penalty += 0.95
    else:
        if blocked_action_target_match:
            contradiction_penalty += 0.65
        if blocked_target_match:
            contradiction_penalty += 0.75 if bool(active_match or revoked) else 0.55
        if blocked_action_family_match and not blocked_action_target_match:
            contradiction_penalty += 0.35
    trust_bonus = float(active.get('trust', control.get('commitment_trust', 0.0)) or 0.0) * 0.45 if active_match and not revoked else 0.0
    binding_mass = sum(weight(token) for token in binding_tokens) or 1.0
    score = support_score + counter_score + trust_bonus - contradiction_penalty
    specificity = (support_score + counter_score) / binding_mass if binding_tokens else 0.0
    return {
        'target_key': str(profile.get('target_key', '') or ''),
        'target_family': str(profile.get('target_family', '') or ''),
        'anchor_ref': str(profile.get('anchor_ref', '') or ''),
        'binding_tokens': sorted(binding_tokens),
        'signal_tokens': sorted(signal_tokens),
        'counterevidence_tokens': sorted(counterevidence_tokens),
        'support_matches': support_matches,
        'counter_matches': counter_matches,
        'active_match': bool(active_match),
        'revoked_match': bool(active_match and revoked),
        'blocked_target_match': bool(blocked_target_match),
        'blocked_action_family_match': bool(blocked_action_family_match),
        'blocked_action_target_match': bool(blocked_action_target_match),
        'hard_blocked_match': bool(hard_blocked_match),
        'blocked_match': bool(blocked_target_match or blocked_action_family_match or blocked_action_target_match or hard_blocked_match),
        'score': round(float(score), 6),
        'evidence_strength': round(float(support_score + counter_score), 6),
        'specificity': round(float(specificity), 6),
        'contradiction_penalty': round(float(contradiction_penalty), 6),
    }


def extract_target_descriptor(action: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(action, dict):
        return {'anchor_ref': '', 'target_family': 'generic_target'}
    click_identity = extract_action_click_identity(action)
    if click_identity:
        point_id = click_identity.split('@', 1)[1]
        return {
            'anchor_ref': f'click@{point_id}',
            'target_family': f'click_point::{point_id}',
        }
    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    intervention_target = meta.get('intervention_target', {}) if isinstance(meta.get('intervention_target', {}), dict) else {}
    anchor_ref = str(intervention_target.get('anchor_ref', '') or meta.get('anchor_ref', '') or '')
    target_kind = str(intervention_target.get('target_kind', '') or '')
    effect_type = str(meta.get('intervention_expected_effect_type', '') or '')
    semantic_labels = list(meta.get('solver_semantic_labels', []) or []) if isinstance(meta.get('solver_semantic_labels', []), list) else []
    role_labels = list(meta.get('solver_object_roles', []) or []) if isinstance(meta.get('solver_object_roles', []), list) else []
    goal_family = str(meta.get('solver_goal_family', '') or '')
    grounded_binding_tokens = [
        str(item or '').strip().lower()
        for item in list(meta.get('grounded_binding_tokens', []) or [])
        if str(item or '').strip()
    ]
    if target_kind:
        target_family = f'target_kind::{target_kind}'
    elif effect_type:
        target_family = f'effect::{effect_type}'
    elif semantic_labels:
        target_family = f'semantic::{str(semantic_labels[0] or "").strip()}'
    elif role_labels:
        target_family = f'role::{str(role_labels[0] or "").strip()}'
    elif grounded_binding_tokens:
        target_family = f'binding::{"::".join(grounded_binding_tokens[:2])}'
    elif anchor_ref:
        target_family = f'anchor::{anchor_ref}'
    elif goal_family:
        target_family = f'goal::{goal_family}'
    else:
        target_family = 'generic_target'
    return {
        'anchor_ref': anchor_ref,
        'target_family': canonical_target_family(target_family),
    }


def action_matches_family(action: Dict[str, Any], family: str) -> bool:
    normalized = normalize_action_family(family)
    if not normalized:
        return False
    return infer_action_family(action) == normalized


def visible_functions_by_family(obs_before: Optional[Dict[str, Any]], family: str) -> List[str]:
    available = extract_available_functions(obs_before or {}) if isinstance(obs_before, dict) else []
    normalized = normalize_action_family(family)
    ranked: List[str] = []
    for fn in available:
        if normalize_action_family(fn) == normalized and fn not in ranked:
            ranked.append(fn)
    if ranked:
        return ranked
    # fallback buckets for ordinal ARC-like surfaces
    upper = [str(fn or '').strip().upper() for fn in available]
    if normalized == 'confirm_interaction':
        preferred = {'ACTION5', 'INTERACT', 'SUBMIT', 'CONFIRM'}
    elif normalized == 'pointer_interaction':
        preferred = {'ACTION6', 'CLICK', 'TAP', 'SELECT'}
    elif normalized == 'navigation_interaction':
        preferred = {'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'}
    else:
        preferred = {'ACTION7', 'PROBE', 'TOGGLE', 'TRANSFORM'}
    for raw, up in zip(available, upper):
        if up in preferred and raw not in ranked:
            ranked.append(raw)
    return ranked


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _actual_phase(result: Dict[str, Any], actual_transition: Dict[str, Any]) -> str:
    phase_candidates = [
        actual_transition.get('next_phase', ''),
        result.get('belief_phase', ''),
        result.get('status', ''),
    ]
    for candidate in phase_candidates:
        text = str(candidate or '').strip().lower()
        if text:
            return text
    return ''


def _actual_observation_tokens(result: Dict[str, Any], actual_transition: Dict[str, Any]) -> List[str]:
    raw_tokens: List[str] = []
    for key in ('predicted_observation_tokens', 'observation_tokens', 'changed_tokens'):
        value = actual_transition.get(key, [])
        if isinstance(value, list):
            raw_tokens.extend(str(item or '').strip().lower() for item in value if str(item or '').strip())
    if not raw_tokens:
        phase = _actual_phase(result, actual_transition)
        if phase:
            raw_tokens.append(phase)
    seen = set()
    ordered: List[str] = []
    for token in raw_tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered[:6]


def _state_changed(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if bool(result.get('state_changed', False) or result.get('observation_changed', False)):
        return True
    try:
        return float(result.get('changed_pixel_count', 0.0) or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def _observation_changed(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(result.get('observation_changed', False) or result.get('state_changed', False))


def _has_progress(reward: float, progress_markers: Sequence[Dict[str, Any]], result: Dict[str, Any]) -> bool:
    if float(reward) > 0.0:
        return True
    if isinstance(result, dict) and bool(result.get('solved', False)):
        return True
    task_progress_seen = False
    goal_stalled = False
    local_only_reaction = False
    for marker in progress_markers or []:
        if not isinstance(marker, dict):
            continue
        name = str(marker.get('name', '') or '')
        if name in {'goal_progressed', 'positive_reward'}:
            return True
        if name == 'task_progressed':
            task_progress_seen = True
        if name == 'goal_stalled':
            goal_stalled = True
        if name == 'local_only_reaction':
            local_only_reaction = True
        if name == 'terminal_reached' and bool(marker.get('success', False)):
            return True
    return bool(task_progress_seen and not goal_stalled and not local_only_reaction)
