from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


def procedure_task_signature(obs_before: Dict[str, Any]) -> str:
    perception = obs_before.get('perception', {}) if isinstance(obs_before.get('perception', {}), dict) else {}
    world_state = obs_before.get('world_state', {}) if isinstance(obs_before.get('world_state', {}), dict) else {}
    parts = [
        str(obs_before.get('task', '')).strip().lower(),
        str(obs_before.get('goal', '')).strip().lower(),
        str(obs_before.get('instruction', '')).strip().lower(),
        str(obs_before.get('query', '')).strip().lower(),
        str(perception.get('goal', '')).strip().lower(),
        str(perception.get('summary', '')).strip().lower(),
        str(world_state.get('task_family', '')).strip().lower(),
        str(world_state.get('phase', '')).strip().lower(),
    ]
    return '|'.join([part for part in parts if part])


def procedure_text_tokens(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", str(text or '').lower())
        if token and len(token) > 1
    }


def procedure_observed_functions(obs_before: Dict[str, Any]) -> List[str]:
    observed: List[str] = []

    def _append(values: Any) -> None:
        if isinstance(values, dict):
            values = list(values.keys())
        if not isinstance(values, list):
            return
        for item in values:
            fn_name = str(item.get('name') if isinstance(item, dict) else item or '').strip()
            if fn_name and fn_name not in observed:
                observed.append(fn_name)

    api_raw = obs_before.get('novel_api', {})
    if hasattr(api_raw, 'raw'):
        api_raw = api_raw.raw
    if isinstance(api_raw, dict):
        _append(api_raw.get('visible_functions', []))
        _append(api_raw.get('discovered_functions', []))
        _append(api_raw.get('available_functions', []))

    _append(obs_before.get('available_functions', []))
    _append((obs_before.get('world_state', {}) if isinstance(obs_before.get('world_state', {}), dict) else {}).get('active_functions', []))
    _append(obs_before.get('backend_functions', {}))
    return observed


def load_procedure_objects(
    shared_store: Any,
    obs_before: Dict[str, Any],
    *,
    selected_limit: int = 8,
    retrieve_limit: int = 400,
) -> List[Dict[str, Any]]:
    task_signature = procedure_task_signature(obs_before)
    task_tokens = procedure_text_tokens(task_signature)
    available_functions = procedure_observed_functions(obs_before)
    ranked: List[Tuple[float, float, int, Dict[str, Any]]] = []
    for idx, obj in enumerate(shared_store.retrieve(sort_by='confidence', limit=retrieve_limit)):
        if str(obj.get('memory_type', '')).lower() != 'procedure_chain':
            continue
        content = obj.get('content', {})
        if not isinstance(content, dict):
            continue
        obj_task = str(content.get('task_signature', '')).lower()
        latent_key = str(content.get('latent_mechanism_key') or '').strip().lower()
        mechanism_roles = [
            str(role).strip().lower()
            for role in list(content.get('mechanism_roles', []) or [])
            if str(role).strip()
        ]
        latent_support_record = bool(latent_key and mechanism_roles)
        object_text = ' '.join(
            str(item).strip().lower()
            for item in (
                content.get('task_signature', ''),
                content.get('mechanism_summary', ''),
                ' '.join(str(tag).strip() for tag in list(obj.get('retrieval_tags', []) or [])),
                ' '.join(str(fn).strip() for fn in list(content.get('action_chain', []) or [])),
                ' '.join(str(fn).strip() for fn in list(content.get('source_surface_functions', []) or [])),
                ' '.join(str(fn).strip() for fn in list(content.get('target_surface_functions', []) or [])),
                content.get('source_domain', ''),
                content.get('target_domain', ''),
            )
            if str(item).strip()
        )
        object_tokens = procedure_text_tokens(object_text)
        chain = [
            str(fn).strip()
            for fn in list(content.get('action_chain', []) or [])
            if str(fn).strip()
        ]
        exact_match = bool(task_signature and obj_task and (task_signature in obj_task or obj_task in task_signature))
        token_overlap = (
            float(len(task_tokens & object_tokens)) / float(max(1, len(task_tokens)))
            if task_tokens and object_tokens else 0.0
        )
        function_overlap = (
            float(len(set(available_functions) & set(chain))) / float(max(1, len(set(available_functions))))
            if available_functions and chain else 0.0
        )
        if (
            task_signature
            and obj_task
            and not exact_match
            and token_overlap <= 0.0
            and function_overlap <= 0.0
            and not latent_support_record
        ):
            continue
        confidence = float(obj.get('confidence', 0.0) or 0.0)
        score = confidence * 0.35
        if exact_match:
            score += 0.45
        score += token_overlap * 0.35
        score += function_overlap * 0.55
        if latent_support_record:
            score += 0.10 + min(0.06, float(len(mechanism_roles)) * 0.03)
        if available_functions and any(fn in available_functions for fn in chain):
            score += 0.08
        ranked.append((score, confidence, -idx, obj))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    selected: List[Dict[str, Any]] = []
    seen_content_hashes: Set[str] = set()
    seen_family_domains: Set[Tuple[str, str]] = set()
    for _score, _confidence, _neg_idx, obj in ranked:
        content = obj.get('content', {}) if isinstance(obj.get('content', {}), dict) else {}
        content_hash = str(obj.get('content_hash') or '').strip()
        latent_key = str(content.get('latent_mechanism_key') or '').strip().lower()
        source_domain = str(content.get('source_domain') or '').strip().lower()
        family_domain_key = (latent_key, source_domain) if latent_key and source_domain else None

        # Duplicate durable snapshots can flood retrieval with the same
        # mechanism/domain record and starve latent transfer of cross-domain support.
        if content_hash and content_hash in seen_content_hashes:
            continue
        if family_domain_key and family_domain_key in seen_family_domains:
            continue

        selected.append(obj)
        if content_hash:
            seen_content_hashes.add(content_hash)
        if family_domain_key:
            seen_family_domains.add(family_domain_key)
        if len(selected) >= selected_limit:
            break
    return selected


def maybe_commit_procedure_chain(
    *,
    committed_ids: List[str],
    obs_before: Dict[str, Any],
    reward: float,
    shared_store: Any,
    validator: Any,
    committer: Any,
    procedure_proposal_log: List[Dict[str, Any]],
    episode: int,
    tick: int,
    reject_decision: Any,
) -> None:
    if float(reward or 0.0) <= 0.0:
        return
    fn_chain: List[str] = []
    for obj_id in committed_ids:
        obj = shared_store.get(obj_id)
        if not isinstance(obj, dict):
            continue
        if str(obj.get('memory_type', '')).lower() == 'procedure_chain':
            continue
        content = obj.get('content', {})
        fn = (
            content.get('tool_args', {}).get('function_name')
            or content.get('function_name')
            or obj.get('function_name')
            or ''
        ) if isinstance(content, dict) else ''
        if fn and fn not in fn_chain:
            fn_chain.append(fn)
    if len(fn_chain) < 2:
        return

    task_signature = procedure_task_signature(obs_before)
    content = {
        'type': 'procedure_chain',
        'task_signature': task_signature,
        'action_chain': fn_chain[:4],
        'source_episode': episode,
        'source_tick': tick,
        'success_rate': 1.0,
        'failure_rate': 0.0,
        'success_count': 1,
        'failure_count': 0,
        'procedure_bonus': 0.1,
    }
    proposal = {
        'content': content,
        'confidence': min(0.95, 0.55 + float(reward or 0.0) * 0.15),
        'content_hash': f"{task_signature}|{'->'.join(content['action_chain'])}",
        'memory_type': 'procedure_chain',
        'memory_layer': 'procedural',
        'retrieval_tags': ['procedure', 'cross_episode', 'post_commit'],
        'source_stage': 'post_commit_integration',
        'source_module': 'core_main_loop',
        'episode': episode,
        'trigger_source': 'high_value_action_chain',
        'trigger_episode': episode,
    }
    decision = validator.validate(proposal)
    if decision.decision == reject_decision:
        return
    committed = committer.commit([(proposal, decision)])
    if committed:
        procedure_proposal_log.append({
            'episode': episode,
            'tick': tick,
            'procedure_object_id': committed[0],
            'task_signature': task_signature,
            'action_chain': list(content['action_chain']),
        })
