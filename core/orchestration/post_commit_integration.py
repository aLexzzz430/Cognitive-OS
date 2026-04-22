from __future__ import annotations

from typing import Any, Dict, Iterable, List, MutableSet

from core.objects import (
    ALL_COGNITIVE_OBJECT_TYPES,
    OBJECT_TYPE_AUTOBIOGRAPHICAL,
    OBJECT_TYPE_DISCRIMINATING_TEST,
    OBJECT_TYPE_HYPOTHESIS,
    OBJECT_TYPE_IDENTITY,
    OBJECT_TYPE_REPRESENTATION,
    OBJECT_TYPE_SKILL,
    OBJECT_TYPE_TRANSFER,
    record_to_cognitive_object,
    select_surfaced_objects,
)
from modules.governance.family_registry import FamilyCard, FamilyState
from modules.memory.retrieval_surface import RetrievalSurface
from modules.memory.router import MemoryRouter


def _resolve_function_name(obj_id: str, obj: Dict[str, Any]) -> str:
    return (
        obj.get('content', {}).get('tool_args', {}).get('function_name')
        or obj.get('content', {}).get('function_name')
        or f'object_{obj_id}'
    )


def _teacher_label_for_confidence(confidence: float) -> str:
    if confidence >= 0.7:
        return 'high_confidence_validated'
    if confidence >= 0.4:
        return 'medium_confidence_validated'
    return 'low_confidence_validated'


def _register_dynamic_family(fn_name: str, obj: Dict[str, Any], family_registry) -> Dict[str, Any]:
    family_id = f'family_{fn_name}'
    card = family_registry.get(family_id)
    created = False
    graduated = False

    if card is None:
        card = FamilyCard(
            family_id=family_id,
            claim=f"Learned family for function '{fn_name}'",
            mechanism='CoreMainLoop dynamic object family',
            signal_type='runtime_evidence',
            firing_phase='step10_commit',
            protective_primitive='commit',
            scope_boundary='Dynamic family inferred from committed object evidence',
            state=FamilyState.QUALIFYING,
            variants=[fn_name],
            n_seeds_tested=1,
        )
        family_registry.register(card)
        created = True

    evidence = {'confidence': obj.get('confidence', 0.5), 'object_id': obj.get('object_id')}
    if obj.get('confidence', 0.5) >= 0.7 and not card.is_graduated():
        family_registry.graduate(
            family_id=family_id,
            evidence=evidence,
            gates_passed=['G_dynamic_confidence'],
            gates_failed=[],
            variant_scope=fn_name,
            reason='Dynamic family graduated after high-confidence commit',
        )
        graduated = True

    return {'family_id': family_id, 'created': created, 'graduated': graduated}


def _route_name_for_object_type(object_type: str) -> str:
    normalized = str(object_type or '').strip()
    if normalized == OBJECT_TYPE_REPRESENTATION:
        return 'matching_and_surfacing'
    if normalized == OBJECT_TYPE_HYPOTHESIS:
        return 'competition_and_test_designer'
    if normalized == OBJECT_TYPE_DISCRIMINATING_TEST:
        return 'probe_queue'
    if normalized == OBJECT_TYPE_SKILL:
        return 'procedural_store_and_planner_prior'
    if normalized == OBJECT_TYPE_TRANSFER:
        return 'cross_domain_prior'
    if normalized == OBJECT_TYPE_IDENTITY:
        return 'self_model_and_continuity'
    if normalized == OBJECT_TYPE_AUTOBIOGRAPHICAL:
        return 'continuity_and_autobiographical_memory'
    return 'matching_and_surfacing'


def integrate_committed_objects(
    *,
    committed_ids: Iterable[str],
    processed_committed_ids: MutableSet[str],
    shared_store,
    runtime_store,
    family_registry,
    confirmed_functions,
    commit_log: List[dict],
    teacher,
    teacher_log: List[dict],
    teacher_allows_intervention,
    tick: int,
    episode: int,
    obs_before: dict,
    result: dict,
    reward: float,
) -> Dict[str, Any]:
    """Fan out Step10 committed objects with idempotent protection and minimal write-back summary."""
    if not committed_ids:
        return {
            'processed_committed_ids': [],
            'new_family_ids': [],
            'teacher_label_counts': {},
            'activation_summary': {'helpful': 0, 'unhelpful': 0},
            'skipped_duplicate_ids': [],
        }

    helpful = reward > 0
    teacher_label_counts: Dict[str, int] = {}
    activation_summary = {'helpful': 0, 'unhelpful': 0}
    new_family_ids: List[str] = []
    processed_ids: List[str] = []
    skipped_duplicates: List[str] = []
    object_route_ids: Dict[str, List[str]] = {object_type: [] for object_type in ALL_COGNITIVE_OBJECT_TYPES}
    routing_events: List[Dict[str, Any]] = []
    surfaced_records: List[Dict[str, Any]] = []
    object_competitions: List[str] = []
    active_tests: List[str] = []
    candidate_tests: List[Dict[str, Any]] = []
    planner_prior_object_ids: List[str] = []
    cross_domain_prior_object_ids: List[str] = []
    mechanism_object_ids: List[str] = []
    current_identity_snapshot: Dict[str, Any] = {}
    autobiographical_summary: Dict[str, Any] = {}
    processed_records: List[Dict[str, Any]] = []

    for obj_id in committed_ids:
        if obj_id in processed_committed_ids:
            skipped_duplicates.append(obj_id)
            continue

        obj = shared_store.get(obj_id)
        processed_committed_ids.add(obj_id)
        if not obj:
            continue

        typed_obj = record_to_cognitive_object(obj)
        object_type = typed_obj.object_type
        processed_records.append(obj)
        route_name = _route_name_for_object_type(object_type)
        object_route_ids.setdefault(object_type, []).append(obj_id)
        routing_events.append({
            'object_id': obj_id,
            'object_type': object_type,
            'route': route_name,
            'family': typed_obj.family,
        })
        if object_type == OBJECT_TYPE_REPRESENTATION:
            surfaced_records.append(obj)
        elif object_type == OBJECT_TYPE_HYPOTHESIS:
            object_competitions.append(obj_id)
        elif object_type == OBJECT_TYPE_DISCRIMINATING_TEST:
            active_tests.append(obj_id)
            candidate_tests.append(dict(obj))
        elif object_type == OBJECT_TYPE_SKILL:
            planner_prior_object_ids.append(obj_id)
        elif object_type == OBJECT_TYPE_TRANSFER:
            cross_domain_prior_object_ids.append(obj_id)
        elif object_type == OBJECT_TYPE_IDENTITY:
            current_identity_snapshot = {
                'object_id': obj_id,
                'family': typed_obj.family,
                'summary': typed_obj.summary,
                'identity_profile': getattr(typed_obj, 'identity_profile', {}),
            }
        elif object_type == OBJECT_TYPE_AUTOBIOGRAPHICAL:
            autobiographical_summary = {
                'object_id': obj_id,
                'family': typed_obj.family,
                'summary': typed_obj.summary,
                'episode_refs': getattr(typed_obj, 'episode_refs', []),
            }

        fn_name = _resolve_function_name(obj_id, obj)
        confirmed_functions.add(fn_name)
        commit_log.append({
            'tick': tick,
            'episode': episode,
            'object_id': obj_id,
            'function_name': fn_name,
            'object_type': object_type,
        })

        runtime_store.record_support(obj_id, tick, obs_before, result)
        runtime_store.record_activation(obj_id, tick, helpful=helpful)
        if helpful:
            activation_summary['helpful'] += 1
        else:
            activation_summary['unhelpful'] += 1

        obj_confidence = obj.get('confidence', 0.5)
        label = _teacher_label_for_confidence(obj_confidence)

        if teacher_allows_intervention():
            teacher.teacher_labeling(
                target_id=obj_id,
                target_type=f'{object_type}_object',
                labels={label: True, 'helpful': helpful},
                rationale=f'Object committed via Step10 with confidence {obj_confidence:.2f}',
                actor='system_validator',
            )
            teacher_log.append({
                'tick': tick,
                'episode': episode,
                'entry': 'teacher_labeling',
                'target_id': obj_id,
                'label': label,
            })
            teacher_label_counts[label] = teacher_label_counts.get(label, 0) + 1

        family_anchor = typed_obj.family or fn_name
        family_result = _register_dynamic_family(family_anchor, obj, family_registry)
        if family_result['created']:
            new_family_ids.append(family_result['family_id'])

        processed_ids.append(obj_id)

    surfaced_object_ids = [
        str(record.get('object_id') or '')
        for record in select_surfaced_objects(
            surfaced_records,
            object_types=[OBJECT_TYPE_REPRESENTATION],
            limit=10,
        )
        if str(record.get('object_id') or '').strip()
    ]
    memory_surface = RetrievalSurface(shared_store).surface_object_records(processed_records, limit=10)
    mechanism_object_ids = list(memory_surface.get('mechanism_object_ids', []) or [])
    memory_route_ids = MemoryRouter().summarize_ids(processed_records)

    return {
        'processed_committed_ids': processed_ids,
        'new_family_ids': new_family_ids,
        'teacher_label_counts': teacher_label_counts,
        'activation_summary': activation_summary,
        'skipped_duplicate_ids': skipped_duplicates,
        'object_route_counts': {
            object_type: len(object_ids)
            for object_type, object_ids in object_route_ids.items()
            if object_ids
        },
        'object_route_ids': {
            object_type: list(object_ids)
            for object_type, object_ids in object_route_ids.items()
            if object_ids
        },
        'object_routing_log': routing_events,
        'surfaced_object_ids': surfaced_object_ids,
        'object_competitions': object_competitions,
        'active_tests': active_tests,
        'candidate_tests': candidate_tests,
        'planner_prior_object_ids': planner_prior_object_ids,
        'cross_domain_prior_object_ids': cross_domain_prior_object_ids,
        'mechanism_object_ids': mechanism_object_ids,
        'current_identity_snapshot': current_identity_snapshot,
        'autobiographical_summary': autobiographical_summary,
        'memory_surface': memory_surface,
        'memory_route_counts': {
            route: len(object_ids)
            for route, object_ids in memory_route_ids.items()
            if object_ids
        },
        'memory_route_ids': {
            route: list(object_ids)
            for route, object_ids in memory_route_ids.items()
            if object_ids
        },
    }
