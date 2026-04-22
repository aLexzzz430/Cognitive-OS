"""
decision/latent_transfer_organs.py

System-intrinsic latent transfer organs used by candidate generation:

- latent family induction
- structure extraction
- role alignment
- anti-distractor scoring

Experiment-specific world generation and metric collection should live outside
this module.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


_SEMANTIC_STOPWORDS = {
    'a', 'an', 'and', 'any', 'before', 'by', 'for', 'from', 'if', 'in', 'into',
    'is', 'it', 'of', 'on', 'or', 'the', 'then', 'to', 'use', 'using', 'with',
    'without', 'once', 'after', 'via', 'that', 'this', 'must', 'should',
}

_ROLE_CANONICAL_ORDER = {
    'prepare': 0,
    'verify': 1,
    'stabilize': 2,
}

_ROLE_SEMANTIC_PROTOTYPES = {
    'prepare': {
        'prepare', 'prime', 'align', 'bias', 'charge', 'temper', 'activate',
        'wet', 'ink', 'ready', 'arm', 'tuneup', 'condition', 'seed',
        'precondition', 'initialize', 'stage1', 'prefit',
    },
    'verify': {
        'verify', 'check', 'confirm', 'inspect', 'validate', 'measure',
        'phase', 'current', 'coherence', 'tide', 'sample', 'probe',
        'audit', 'checkpoint', 'attest', 'scan', 'review', 'stage2',
    },
    'stabilize': {
        'stabilize', 'seal', 'secure', 'lock', 'release', 'balance', 'tune',
        'open', 'unlock', 'settle', 'commit', 'finalize', 'lockdown',
        'closeout', 'stage3',
    },
}

_ANTI_MECHANISM_TOKENS = {
    'abort', 'interrupt', 'halt', 'purge', 'shock', 'spill', 'stop', 'erase',
    'scramble', 'dump', 'trap', 'null', 'rupture', 'corrupt', 'break',
    'kill', 'sever', 'crash', 'jam', 'destroy', 'immediately', 'emergency',
}

_NEGATIVE_EFFECT_KEYS = {
    'destructive',
    'irreversible',
    'terminal_failure',
    'invalidates_progress',
    'resets_progress',
    'collapses_state',
}

_NUMERIC_PROGRESS_KEYS = (
    'progress',
    'stability',
    'safety',
    'validity',
    'coherence',
    'integrity',
)


def _semantic_tokens(value: Any) -> Set[str]:
    tokens: Set[str] = set()
    if value is None:
        return tokens
    if isinstance(value, str):
        parts = re.split(r'[^a-z0-9]+', value.lower().replace('_', ' '))
        for part in parts:
            if len(part) < 2 or part in _SEMANTIC_STOPWORDS:
                continue
            tokens.add(part)
        return tokens
    if isinstance(value, dict):
        for item in value.values():
            tokens.update(_semantic_tokens(item))
        return tokens
    if isinstance(value, (list, tuple, set)):
        for item in value:
            tokens.update(_semantic_tokens(item))
        return tokens
    return _semantic_tokens(str(value))


def _normalize_effect_profile(value: Any, *, prefix: str = '') -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    if value is None:
        return normalized
    if isinstance(value, bool):
        normalized[prefix.rstrip('.')] = 1.0 if value else 0.0
        return normalized
    if isinstance(value, (int, float)):
        normalized[prefix.rstrip('.')] = float(value)
        return normalized
    if isinstance(value, str):
        for token in _semantic_tokens(value):
            key = f"{prefix}{token}" if prefix else token
            normalized[key] = 1.0
        return normalized
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}{str(key).strip().lower()}."
            for child_key, child_value in _normalize_effect_profile(item, prefix=child_prefix).items():
                normalized[child_key] = normalized.get(child_key, 0.0) + child_value
        return normalized
    if isinstance(value, (list, tuple, set)):
        for item in value:
            for child_key, child_value in _normalize_effect_profile(item, prefix=prefix).items():
                normalized[child_key] = normalized.get(child_key, 0.0) + child_value
        return normalized
    return _normalize_effect_profile(str(value), prefix=prefix)


def _merge_effect_profiles(target: Dict[str, float], source: Dict[str, float], *, weight: float = 1.0) -> None:
    if not source:
        return
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + (float(value) * weight)


def _average_effect_profiles(profiles: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not profiles:
        return {}
    merged: Dict[str, float] = {}
    for profile in profiles:
        _merge_effect_profiles(merged, profile)
    denominator = float(max(len(profiles), 1))
    return {key: value / denominator for key, value in merged.items()}


def _effect_profile_similarity(expected: Dict[str, float], observed: Dict[str, float]) -> float:
    if not expected or not observed:
        return 0.0
    score_total = 0.0
    key_count = 0
    for key, expected_value in expected.items():
        key_count += 1
        observed_value = observed.get(key)
        if observed_value is None:
            continue
        scale = max(abs(expected_value), abs(observed_value), 1.0)
        score_total += max(0.0, 1.0 - (abs(expected_value - observed_value) / scale))
    if key_count == 0:
        return 0.0
    return score_total / float(key_count)


class LatentTransferOrganSystem:
    def build_candidates(
        self,
        *,
        obs: Dict[str, Any],
        available_functions: Sequence[str],
        episode_trace: Sequence[Dict[str, Any]],
        procedure_objects: Sequence[Dict[str, Any]],
        make_call_action: Callable[..., Dict[str, Any]],
        function_name_from_action: Callable[[Optional[Dict[str, Any]]], str],
    ) -> List[Dict[str, Any]]:
        if not procedure_objects or not available_functions:
            return []

        bundles: Dict[str, Dict[str, Any]] = {}
        for obj in procedure_objects:
            if not isinstance(obj, dict):
                continue
            content = obj.get('content', {})
            if not isinstance(content, dict):
                continue
            record = self._extract_latent_mechanism_record(obj)
            if not isinstance(record, dict):
                continue
            role_sequence = list(record.get('role_sequence', []))
            if len(role_sequence) < 2:
                continue
            role_bindings = dict(record.get('role_bindings', {}))
            role_descriptors = record.get('role_descriptors', {})
            role_effect_profiles = record.get('role_effect_profiles', {})
            latent_key = str(record.get('latent_key') or '')
            source_domain = str(record.get('source_domain') or '')
            induced = bool(record.get('induced', False))
            bundle = bundles.setdefault(
                latent_key,
                {
                    'latent_key': latent_key,
                    'domains': set(),
                    'role_descriptors': {},
                    'role_positions': {},
                    'role_effect_profile_samples': {},
                    'support_records': [],
                    'induced_support_count': 0,
                },
            )
            bundle['domains'].add(source_domain)
            bundle['support_records'].append({
                'source_domain': source_domain,
                'object_id': obj.get('object_id', ''),
                'success_rate': float(content.get('success_rate', obj.get('confidence', 0.5)) or 0.0),
                'failure_rate': float(content.get('failure_rate', 0.0) or 0.0),
                'procedure_bonus': float(content.get('procedure_bonus', 0.08) or 0.08),
                'induced': induced,
            })
            if induced:
                bundle['induced_support_count'] += 1

            role_count = max(len(role_sequence) - 1, 1)
            for idx, role in enumerate(role_sequence):
                position_bucket = bundle['role_positions'].setdefault(role, [])
                position_bucket.append(float(idx) / float(role_count))

                token_bucket = bundle['role_descriptors'].setdefault(role, set())
                token_bucket.update(_semantic_tokens(role))
                if isinstance(role_descriptors, dict):
                    token_bucket.update(_semantic_tokens(role_descriptors.get(role)))

                effect_samples = bundle['role_effect_profile_samples'].setdefault(role, [])
                if isinstance(role_effect_profiles, dict):
                    effect_profile = role_effect_profiles.get(role, {})
                    if isinstance(effect_profile, dict) and effect_profile:
                        effect_samples.append(dict(effect_profile))

                if induced:
                    continue
                for fn_name, bound_role in role_bindings.items():
                    if str(bound_role).strip().lower() != role:
                        continue
                    token_bucket.update(_semantic_tokens(fn_name))

        signatures = obs.get('function_signatures', {}) if isinstance(obs, dict) else {}
        available_fn_tokens: Dict[str, Set[str]] = {}
        available_fn_effect_profiles: Dict[str, Dict[str, float]] = {}
        for fn_name in available_functions:
            if not isinstance(fn_name, str) or not fn_name:
                continue
            signature = signatures.get(fn_name, {}) if isinstance(signatures, dict) else {}
            available_fn_tokens[fn_name] = self._function_semantic_tokens(fn_name, signature)
            available_fn_effect_profiles[fn_name] = self._function_effect_profile(signature)

        if not available_fn_tokens:
            return []

        historical_fn_tokens: Dict[str, Set[str]] = {}
        historical_fn_effect_profiles: Dict[str, Dict[str, float]] = {}
        for row in episode_trace:
            if not isinstance(row, dict):
                continue
            if float(row.get('reward', 0.0) or 0.0) <= 0.0:
                continue
            action = row.get('action', {}) if isinstance(row.get('action', {}), dict) else {}
            fn_name = function_name_from_action(action)
            if not fn_name or fn_name in available_fn_tokens:
                continue
            historical_fn_tokens[fn_name] = self._function_semantic_tokens(fn_name, {})
            historical_fn_effect_profiles[fn_name] = {}

        candidate_fn_tokens = dict(historical_fn_tokens)
        candidate_fn_tokens.update(available_fn_tokens)
        candidate_fn_effect_profiles = dict(historical_fn_effect_profiles)
        candidate_fn_effect_profiles.update(available_fn_effect_profiles)

        structural_positions = self._infer_structural_function_positions(available_functions, signatures)
        anchored_role_mapping = self._anchored_role_mapping_from_trace(episode_trace, function_name_from_action)

        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for bundle in bundles.values():
            domains = bundle.get('domains', set())
            if len(domains) < 2:
                continue

            role_sequence = self._resolve_bundle_role_sequence(bundle.get('role_positions', {}))
            if len(role_sequence) < 2:
                continue

            role_descriptor_map = bundle.get('role_descriptors', {})
            role_effect_profiles = {
                role_name: _average_effect_profiles(bundle.get('role_effect_profile_samples', {}).get(role_name, []))
                for role_name in role_sequence
            }
            mapping, mapping_confidence, evidence = self._infer_role_mapping(
                role_sequence=role_sequence,
                role_descriptor_map=role_descriptor_map,
                role_effect_profiles=role_effect_profiles,
                available_fn_tokens=candidate_fn_tokens,
                available_fn_effect_profiles=candidate_fn_effect_profiles,
                structural_positions=structural_positions,
                anchored_mapping=anchored_role_mapping,
            )
            completed_roles = set(anchored_role_mapping.keys())
            if len(mapping) != len(role_sequence):
                next_role = ''
                for role in role_sequence:
                    if role not in completed_roles:
                        next_role = role
                        break
                if not next_role:
                    continue
                partial_mapping, partial_confidence, partial_evidence = self._infer_role_mapping(
                    role_sequence=[next_role],
                    role_descriptor_map={next_role: set(role_descriptor_map.get(next_role, set()))},
                    role_effect_profiles={next_role: dict(role_effect_profiles.get(next_role, {}))},
                    available_fn_tokens=available_fn_tokens,
                    available_fn_effect_profiles=available_fn_effect_profiles,
                    structural_positions=structural_positions,
                    anchored_mapping={},
                )
                next_fn = partial_mapping.get(next_role, '')
                if not next_fn:
                    continue
                mapping = dict(anchored_role_mapping)
                mapping.update(partial_mapping)
                evidence = dict(partial_evidence)
                mapping_confidence = partial_confidence
            else:
                completed_roles = self._completed_mechanism_roles(
                    episode_trace=episode_trace,
                    mapping=mapping,
                    function_name_from_action=function_name_from_action,
                )
            next_role = ''
            for role in role_sequence:
                if role not in completed_roles:
                    next_role = role
                    break
            if not next_role:
                continue
            next_fn = mapping.get(next_role, '')
            if not next_fn:
                continue

            support_records = list(bundle.get('support_records', []))
            avg_success = sum(float(row.get('success_rate', 0.0) or 0.0) for row in support_records) / max(len(support_records), 1)
            avg_failure = sum(float(row.get('failure_rate', 0.0) or 0.0) for row in support_records) / max(len(support_records), 1)
            avg_bonus = sum(float(row.get('procedure_bonus', 0.0) or 0.0) for row in support_records) / max(len(support_records), 1)

            family_binding_confidence = min(
                1.0,
                (mapping_confidence * 0.55)
                + (min(len(domains), 4) * 0.1)
                + (min(int(bundle.get('induced_support_count', 0) or 0), 3) * 0.05),
            )
            candidate = make_call_action(next_fn, obs, None, episode_trace, None)
            candidate['_source'] = 'procedure_reuse'
            meta = candidate.get('_candidate_meta', {})
            meta['procedure'] = {
                'object_id': f"latent::{bundle['latent_key']}",
                'task_signature': '',
                'action_chain': [mapping.get(role, '') for role in role_sequence if mapping.get(role)],
                'hit_source': 'latent_mechanism_abstraction',
                'is_next_step': True,
                'success_rate': avg_success,
                'failure_rate': avg_failure,
                'procedure_bonus': avg_bonus,
                'support_domains': sorted(str(domain) for domain in domains),
                'support_count': len(domains),
                'latent_mechanism_key': bundle['latent_key'],
                'role_sequence': list(role_sequence),
                'selected_role': next_role,
                'role_bindings': dict(mapping),
                'mapping_confidence': mapping_confidence,
                'family_binding_confidence': family_binding_confidence,
                'family_induced': bool(bundle.get('induced_support_count', 0) > 0),
                'induction_support_count': int(bundle.get('induced_support_count', 0) or 0),
                'role_binding_evidence': deepcopy(evidence),
            }
            candidate['_candidate_meta'] = meta
            support_strength = (
                (len(domains) * 0.25)
                + mapping_confidence
                + (family_binding_confidence * 0.25)
                + (avg_success * 0.2)
                - (avg_failure * 0.1)
            )
            candidates.append((support_strength, candidate))

        ordered = [candidate for _score, candidate in sorted(candidates, key=lambda row: row[0], reverse=True)]
        return ordered[:2]

    def _extract_latent_mechanism_record(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return None
        content = obj.get('content', {})
        if not isinstance(content, dict):
            return None
        chain = content.get('action_chain', [])
        if not isinstance(chain, list) or len(chain) < 2:
            return None

        raw_roles = content.get('mechanism_roles') or content.get('role_sequence') or []
        role_bindings = content.get('role_bindings', {})
        surface_descriptors = content.get('surface_descriptors', {})
        role_descriptors = content.get('role_descriptors', {})
        role_effect_profiles = content.get('role_effect_profiles', {})
        action_effect_profiles = content.get('action_effect_profiles', {})
        induced = False

        if isinstance(raw_roles, list) and raw_roles and isinstance(role_bindings, dict) and role_bindings:
            normalized_bindings = {
                str(fn_name).strip(): str(role_name).strip().lower()
                for fn_name, role_name in role_bindings.items()
                if str(fn_name).strip() and str(role_name).strip()
            }
            bound_sequence: List[str] = []
            for raw_fn_name in chain:
                fn_name = str(raw_fn_name).strip()
                bound_role = str(normalized_bindings.get(fn_name, '') or '').strip().lower()
                if bound_role and bound_role not in bound_sequence:
                    bound_sequence.append(bound_role)
            explicit_roles = [str(role).strip().lower() for role in raw_roles if str(role).strip()]
            role_sequence = list(bound_sequence)
            for role_name in explicit_roles:
                if role_name not in role_sequence:
                    role_sequence.append(role_name)
            normalized_descriptors = {
                str(role_name).strip().lower(): set(_semantic_tokens(descriptor))
                for role_name, descriptor in role_descriptors.items()
            } if isinstance(role_descriptors, dict) else {}
            normalized_role_effects = {
                str(role_name).strip().lower(): _normalize_effect_profile(effect_profile)
                for role_name, effect_profile in role_effect_profiles.items()
                if str(role_name).strip()
            } if isinstance(role_effect_profiles, dict) else {}
            if isinstance(action_effect_profiles, dict):
                for fn_name, bound_role in normalized_bindings.items():
                    normalized_profile = _normalize_effect_profile(action_effect_profiles.get(fn_name))
                    if not normalized_profile:
                        continue
                    bucket = normalized_role_effects.setdefault(bound_role, {})
                    _merge_effect_profiles(bucket, normalized_profile)
        else:
            induced = True
            induced_record = self._induce_latent_mechanism_record(content)
            if not isinstance(induced_record, dict):
                return None
            role_sequence = list(induced_record.get('role_sequence', []))
            normalized_bindings = dict(induced_record.get('role_bindings', {}))
            normalized_descriptors = dict(induced_record.get('role_descriptors', {}))
            normalized_role_effects = dict(induced_record.get('role_effect_profiles', {}))
            if not role_sequence or not normalized_bindings:
                return None

        latent_key = str(content.get('latent_mechanism_key') or self._induced_bundle_key(role_sequence))
        source_domain = str(content.get('source_domain') or obj.get('source_domain') or obj.get('object_id') or latent_key)
        return {
            'latent_key': latent_key,
            'source_domain': source_domain,
            'role_sequence': role_sequence,
            'role_bindings': normalized_bindings,
            'role_descriptors': normalized_descriptors,
            'role_effect_profiles': normalized_role_effects,
            'surface_descriptors': surface_descriptors if isinstance(surface_descriptors, dict) else {},
            'induced': induced,
        }

    def _induce_latent_mechanism_record(self, content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        chain = content.get('action_chain', [])
        if not isinstance(chain, list) or len(chain) < 2:
            return None
        surface_descriptors = content.get('surface_descriptors', {})
        surface_descriptors = surface_descriptors if isinstance(surface_descriptors, dict) else {}
        action_effect_profiles = content.get('action_effect_profiles', {})
        action_effect_profiles = action_effect_profiles if isinstance(action_effect_profiles, dict) else {}

        role_sequence: List[str] = []
        role_bindings: Dict[str, str] = {}
        role_descriptors: Dict[str, Set[str]] = {}
        role_effect_profiles: Dict[str, Dict[str, float]] = {}
        chain_len = len(chain)

        for idx, raw_fn_name in enumerate(chain):
            fn_name = str(raw_fn_name).strip()
            if not fn_name:
                continue
            action_tokens = set(_semantic_tokens(fn_name))
            action_tokens.update(_semantic_tokens(surface_descriptors.get(fn_name)))
            action_effects = _normalize_effect_profile(action_effect_profiles.get(fn_name))
            role_name = self._induce_latent_role(
                action_tokens=action_tokens,
                action_effects=action_effects,
                idx=idx,
                chain_len=chain_len,
            )
            if not role_name:
                continue
            role_bindings[fn_name] = role_name
            if role_name not in role_sequence:
                role_sequence.append(role_name)
            descriptor_bucket = role_descriptors.setdefault(role_name, set())
            prototype_tokens = set(_ROLE_SEMANTIC_PROTOTYPES.get(role_name, set()))
            descriptor_bucket.update(prototype_tokens)
            descriptor_bucket.update(action_tokens & prototype_tokens)
            if action_effects:
                effect_bucket = role_effect_profiles.setdefault(role_name, {})
                _merge_effect_profiles(effect_bucket, action_effects)

        if len(role_sequence) < 2 or len(role_bindings) < 2:
            return None
        return {
            'role_sequence': role_sequence,
            'role_bindings': role_bindings,
            'role_descriptors': role_descriptors,
            'role_effect_profiles': role_effect_profiles,
        }

    def _induce_latent_role(
        self,
        *,
        action_tokens: Set[str],
        action_effects: Dict[str, float],
        idx: int,
        chain_len: int,
    ) -> str:
        best_role = ''
        best_score = float('-inf')
        for role_name, prototype_tokens in _ROLE_SEMANTIC_PROTOTYPES.items():
            overlap_score = float(len(action_tokens & prototype_tokens))
            effect_bias = 0.0
            if role_name == 'prepare' and any('progress' in key and value > 0.0 for key, value in action_effects.items()):
                effect_bias += 0.35
            if role_name == 'verify' and any('verify' in key or 'confirm' in key for key in action_effects.keys()):
                effect_bias += 0.35
            if role_name == 'stabilize' and any('stability' in key or 'lock' in key for key in action_effects.keys()):
                effect_bias += 0.35
            position_bias = 0.0
            if idx == 0 and role_name == 'prepare':
                position_bias += 1.0
            if idx == chain_len - 1 and role_name == 'stabilize':
                position_bias += 1.0
            if 0 < idx < chain_len - 1 and role_name == 'verify':
                position_bias += 1.0
            if chain_len == 2 and idx == 1 and role_name == 'stabilize':
                position_bias += 0.6
            score = overlap_score + effect_bias + position_bias
            if score > best_score:
                best_role = role_name
                best_score = score
        return best_role

    def _induced_bundle_key(self, role_sequence: Sequence[str]) -> str:
        cleaned = [str(role).strip().lower() for role in role_sequence if str(role).strip()]
        if len(cleaned) < 2:
            return 'induced::unknown_family'
        return f"induced::{cleaned[0]}_to_{cleaned[-1]}"

    def _resolve_bundle_role_sequence(self, role_positions: Dict[str, Sequence[float]]) -> List[str]:
        scored_roles: List[Tuple[float, int, str]] = []
        for role_name, samples in role_positions.items():
            clean_role = str(role_name).strip().lower()
            if not clean_role:
                continue
            sample_list = [float(item) for item in samples if isinstance(item, (int, float))]
            if not sample_list:
                continue
            avg_position = sum(sample_list) / max(len(sample_list), 1)
            canonical_idx = int(_ROLE_CANONICAL_ORDER.get(clean_role, 100))
            scored_roles.append((avg_position, canonical_idx, clean_role))
        ordered = [role_name for _avg, _idx, role_name in sorted(scored_roles)]
        deduped: List[str] = []
        for role_name in ordered:
            if role_name not in deduped:
                deduped.append(role_name)
        return deduped

    def _function_semantic_tokens(self, function_name: str, signature: Any) -> Set[str]:
        tokens = set(_semantic_tokens(function_name))
        if isinstance(signature, dict):
            for key in (
                'description',
                'summary',
                'doc',
                'help',
                'title',
                'semantic_hint',
                'semantic_hints',
            ):
                tokens.update(_semantic_tokens(signature.get(key)))
            params = signature.get('parameters')
            if isinstance(params, dict):
                tokens.update(_semantic_tokens(params.get('description')))
                props = params.get('properties')
                if isinstance(props, dict):
                    for spec in props.values():
                        if isinstance(spec, dict):
                            tokens.update(_semantic_tokens(spec.get('description')))
        return tokens

    def _function_effect_profile(self, signature: Any) -> Dict[str, float]:
        if not isinstance(signature, dict):
            return {}
        profile: Dict[str, float] = {}
        for key in (
            'effect_profile',
            'transition_profile',
            'transition_features',
            'observable_outcomes',
            'state_transition',
            'action_effect_profile',
        ):
            _merge_effect_profiles(profile, _normalize_effect_profile(signature.get(key)))
        return profile

    def _signature_text(self, signature: Any) -> str:
        if not isinstance(signature, dict):
            return ''
        parts: List[str] = []
        for key in ('description', 'summary', 'doc', 'help', 'title', 'semantic_hint', 'semantic_hints'):
            value = signature.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        parts.append(item.strip())
        params = signature.get('parameters')
        if isinstance(params, dict):
            desc = params.get('description')
            if isinstance(desc, str) and desc.strip():
                parts.append(desc.strip())
            props = params.get('properties')
            if isinstance(props, dict):
                for spec in props.values():
                    if isinstance(spec, dict):
                        prop_desc = spec.get('description')
                        if isinstance(prop_desc, str) and prop_desc.strip():
                            parts.append(prop_desc.strip())
        return ' '.join(parts)

    def _infer_structural_function_positions(
        self,
        available_functions: Sequence[str],
        signatures: Dict[str, Any],
    ) -> Dict[str, float]:
        names = [str(fn).strip() for fn in available_functions if str(fn).strip()]
        if len(names) < 2:
            return {}

        explicit_positions: Dict[str, float] = {}
        outgoing: Dict[str, Set[str]] = {fn: set() for fn in names}
        incoming: Dict[str, Set[str]] = {fn: set() for fn in names}
        bias: Dict[str, float] = {fn: 0.0 for fn in names}
        evidence: Set[str] = set()

        for fn_name in names:
            signature = signatures.get(fn_name, {}) if isinstance(signatures, dict) else {}
            for numeric_key in ('stage_index', 'step_index', 'structural_position', 'transition_position'):
                numeric_value = signature.get(numeric_key) if isinstance(signature, dict) else None
                if isinstance(numeric_value, (int, float)):
                    explicit_positions[fn_name] = float(numeric_value)
                    evidence.add(fn_name)
                    break

            raw_text = self._signature_text(signature).lower()
            if not raw_text:
                continue
            normalized_text = raw_text.replace('_', ' ')

            if any(phrase in normalized_text for phrase in ('first step', 'initial step', 'start with', 'begin with')):
                bias[fn_name] -= 1.0
                evidence.add(fn_name)
            if any(phrase in normalized_text for phrase in ('final step', 'last step', 'finish with', 'complete with')):
                bias[fn_name] += 1.0
                evidence.add(fn_name)
            if any(phrase in normalized_text for phrase in ('middle step', 'intermediate step', 'checkpoint step')):
                evidence.add(fn_name)

            for other_fn in names:
                if other_fn == fn_name:
                    continue
                aliases = {other_fn.lower(), other_fn.lower().replace('_', ' ')}
                for alias in aliases:
                    if not alias:
                        continue
                    text_variant = normalized_text if ' ' in alias else raw_text
                    if self._description_mentions_relation(text_variant, alias, ('before', 'precede', 'precedes', 'prior to', 'ahead of')):
                        outgoing[fn_name].add(other_fn)
                        incoming[other_fn].add(fn_name)
                        evidence.add(fn_name)
                        evidence.add(other_fn)
                    if self._description_mentions_relation(text_variant, alias, ('after', 'once', 'following', 'upon', 'post')):
                        outgoing[other_fn].add(fn_name)
                        incoming[fn_name].add(other_fn)
                        evidence.add(fn_name)
                        evidence.add(other_fn)

        if len(explicit_positions) >= 2:
            ordered_numeric = sorted(explicit_positions.items(), key=lambda item: (item[1], item[0]))
            denominator = max(len(ordered_numeric) - 1, 1)
            return {
                fn_name: float(idx) / float(denominator)
                for idx, (fn_name, _value) in enumerate(ordered_numeric)
            }

        nodes = [fn for fn in names if fn in evidence]
        if len(nodes) < 2:
            return {}

        indegree: Dict[str, int] = {
            fn: len([src for src in incoming.get(fn, set()) if src in nodes])
            for fn in nodes
        }
        queue: List[str] = sorted(
            [fn for fn in nodes if indegree.get(fn, 0) == 0],
            key=lambda item: (bias.get(item, 0.0), item),
        )
        ordered: List[str] = []
        while queue:
            current = queue.pop(0)
            if current in ordered:
                continue
            ordered.append(current)
            for nxt in sorted(outgoing.get(current, set())):
                if nxt not in indegree:
                    continue
                indegree[nxt] = max(0, indegree[nxt] - 1)
                if indegree[nxt] == 0 and nxt not in ordered and nxt not in queue:
                    queue.append(nxt)
                    queue.sort(key=lambda item: (bias.get(item, 0.0), item))

        if len(ordered) != len(nodes):
            ordered = sorted(
                nodes,
                key=lambda item: ((len(incoming.get(item, set())) - len(outgoing.get(item, set()))) + bias.get(item, 0.0), item),
            )

        denominator = max(len(ordered) - 1, 1)
        return {
            fn_name: float(idx) / float(denominator)
            for idx, fn_name in enumerate(ordered)
        }

    def _description_mentions_relation(
        self,
        text: str,
        alias: str,
        relation_keywords: Sequence[str],
    ) -> bool:
        if not text or not alias:
            return False
        alias_pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        for keyword in relation_keywords:
            keyword_pattern = rf"{re.escape(keyword)}"
            if re.search(rf"{keyword_pattern}.{{0,48}}{alias_pattern}", text):
                return True
        return False

    def _infer_role_mapping(
        self,
        *,
        role_sequence: Sequence[str],
        role_descriptor_map: Dict[str, Set[str]],
        role_effect_profiles: Dict[str, Dict[str, float]],
        available_fn_tokens: Dict[str, Set[str]],
        available_fn_effect_profiles: Dict[str, Dict[str, float]],
        structural_positions: Optional[Dict[str, float]] = None,
        anchored_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, str], float, Dict[str, Dict[str, Any]]]:
        used_functions: Set[str] = set()
        mapping: Dict[str, str] = {}
        evidence: Dict[str, Dict[str, Any]] = {}
        score_total = 0.0

        anchored_mapping = anchored_mapping if isinstance(anchored_mapping, dict) else {}
        for role in role_sequence:
            anchored_fn = str(anchored_mapping.get(role, '') or '')
            if not anchored_fn or anchored_fn not in available_fn_tokens or anchored_fn in used_functions:
                continue
            mapping[role] = anchored_fn
            used_functions.add(anchored_fn)
            score_total += 1.0
            evidence[role] = {
                'function_name': anchored_fn,
                'semantic_score': 1.0,
                'structural_score': 0.0,
                'effect_score': 1.0,
                'counter_signal_penalty': 0.0,
                'total_score': 1.0,
                'anchored': True,
            }

        for role in role_sequence:
            if role in mapping:
                continue
            role_tokens = set(role_descriptor_map.get(role, set()))
            role_tokens.update(_semantic_tokens(role))
            role_effects = dict(role_effect_profiles.get(role, {}))
            expected_position = float(role_sequence.index(role)) / float(max(len(role_sequence) - 1, 1))
            best_fn = ''
            best_score = float('-inf')
            best_evidence: Dict[str, Any] = {}
            for fn_name, fn_tokens in available_fn_tokens.items():
                if fn_name in used_functions:
                    continue
                fn_effects = available_fn_effect_profiles.get(fn_name, {})
                semantic_score = self._semantic_overlap_score(role_tokens, fn_tokens)
                structural_score = self._structural_alignment_score(
                    fn_name,
                    expected_position=expected_position,
                    structural_positions=structural_positions or {},
                )
                effect_score = _effect_profile_similarity(role_effects, fn_effects)
                counter_signal_penalty = self._counter_signal_penalty(fn_tokens, fn_effects)
                score = semantic_score + structural_score + (effect_score * 1.2) - counter_signal_penalty
                if score > best_score:
                    best_fn = fn_name
                    best_score = score
                    best_evidence = {
                        'function_name': fn_name,
                        'semantic_score': semantic_score,
                        'structural_score': structural_score,
                        'effect_score': effect_score,
                        'counter_signal_penalty': counter_signal_penalty,
                        'total_score': score,
                        'anchored': False,
                    }
            if not best_fn or best_score < 0.25:
                remaining_functions = [fn_name for fn_name in available_fn_tokens.keys() if fn_name not in used_functions]
                remaining_roles = [role_name for role_name in role_sequence if role_name not in mapping]
                if len(remaining_roles) == 1 and len(remaining_functions) == 1:
                    best_fn = remaining_functions[0]
                    best_score = max(best_score, 0.6)
                    best_evidence = {
                        'function_name': best_fn,
                        'semantic_score': 0.0,
                        'structural_score': 0.0,
                        'effect_score': 0.0,
                        'counter_signal_penalty': 0.0,
                        'total_score': best_score,
                        'anchored': False,
                        'fallback_remaining_pair': True,
                    }
                else:
                    return {}, 0.0, {}
            mapping[role] = best_fn
            used_functions.add(best_fn)
            score_total += min(1.0, best_score)
            evidence[role] = best_evidence
        return mapping, score_total / max(len(role_sequence), 1), evidence

    def _structural_alignment_score(
        self,
        fn_name: str,
        *,
        expected_position: float,
        structural_positions: Dict[str, float],
    ) -> float:
        if fn_name not in structural_positions:
            return 0.0
        observed_position = float(structural_positions.get(fn_name, 0.0) or 0.0)
        gap = abs(observed_position - expected_position)
        return max(0.0, 1.0 - gap)

    def _counter_signal_penalty(
        self,
        fn_tokens: Set[str],
        effect_profile: Dict[str, float],
    ) -> float:
        penalty = 0.0
        if fn_tokens:
            hits = fn_tokens & _ANTI_MECHANISM_TOKENS
            if hits:
                severe_hits = len(hits)
                penalty += min(0.85, 0.35 + (0.15 * float(severe_hits - 1)))
        for key, value in effect_profile.items():
            clean_key = str(key).strip().lower()
            if not clean_key:
                continue
            if any(fragment in clean_key for fragment in _NEGATIVE_EFFECT_KEYS) and value > 0.0:
                penalty += 0.25
                continue
            if any(fragment in clean_key for fragment in _NUMERIC_PROGRESS_KEYS) and value < 0.0:
                penalty += 0.15
        return min(0.85, penalty)

    def _semantic_overlap_score(self, role_tokens: Set[str], fn_tokens: Set[str]) -> float:
        if not role_tokens or not fn_tokens:
            return 0.0
        overlap = role_tokens & fn_tokens
        if not overlap:
            return 0.0
        return (len(overlap) * 1.0) / max(min(len(role_tokens), 3), 1)

    def _completed_mechanism_roles(
        self,
        *,
        episode_trace: Sequence[Dict[str, Any]],
        mapping: Dict[str, str],
        function_name_from_action: Callable[[Optional[Dict[str, Any]]], str],
    ) -> Set[str]:
        completed: Set[str] = set()
        if not episode_trace or not mapping:
            return completed
        reverse_mapping = {fn_name: role for role, fn_name in mapping.items() if fn_name}
        for row in episode_trace:
            if not isinstance(row, dict):
                continue
            if float(row.get('reward', 0.0) or 0.0) <= 0.0:
                continue
            action = row.get('action', {}) if isinstance(row.get('action', {}), dict) else {}
            fn_name = function_name_from_action(action)
            role = reverse_mapping.get(fn_name, '')
            if role:
                completed.add(role)
        return completed

    def _anchored_role_mapping_from_trace(
        self,
        episode_trace: Sequence[Dict[str, Any]],
        function_name_from_action: Callable[[Optional[Dict[str, Any]]], str],
    ) -> Dict[str, str]:
        anchored: Dict[str, str] = {}
        for row in episode_trace:
            if not isinstance(row, dict):
                continue
            if float(row.get('reward', 0.0) or 0.0) <= 0.0:
                continue
            action = row.get('action', {}) if isinstance(row.get('action', {}), dict) else {}
            fn_name = function_name_from_action(action)
            if not fn_name:
                continue
            meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
            procedure = meta.get('procedure', {}) if isinstance(meta.get('procedure', {}), dict) else {}
            selected_role = str(procedure.get('selected_role', '') or '').strip().lower()
            if selected_role:
                anchored[selected_role] = fn_name
                continue
            role_bindings = procedure.get('role_bindings', {}) if isinstance(procedure.get('role_bindings', {}), dict) else {}
            for role_name, mapped_fn in role_bindings.items():
                if str(mapped_fn).strip() == fn_name:
                    anchored[str(role_name).strip().lower()] = fn_name
        return anchored
