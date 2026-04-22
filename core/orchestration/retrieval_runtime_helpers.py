from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from modules.episodic import LLMRetrievalContext
from core.orchestration.action_utils import extract_available_functions
from core.orchestration.retrieval_gate import (
    HypothesisAugmentContext,
    HypothesisAugmentCooldownState,
    HypothesisAugmentSignals,
    should_augment_hypotheses,
)


@dataclass(frozen=True)
class RetrieveResultContract:
    selected_ids: List[str]
    action_influence: str
    candidate_count: int


class RetrievalRuntimeHelpers:
    """Unified helper API for retrieval-runtime contracts and protocol payloads."""

    @staticmethod
    def build_llm_retrieval_context(
        *,
        episode: int,
        tick: int,
        obs: Dict[str, Any],
        continuity_snapshot: Dict[str, Any],
        active_hypotheses: int,
        confirmed_hypotheses: int,
        entropy: float,
        margin: float,
    ) -> LLMRetrievalContext:
        api_raw = obs.get('novel_api', {})
        if hasattr(api_raw, 'raw'):
            api_raw = api_raw.raw

        discovered = list(api_raw.get('discovered_functions', []) if isinstance(api_raw, dict) else [])

        # Prefer real env-available surface instead of inferring available := visible - discovered.
        surface_available = list(extract_available_functions(obs))

        raw_available: List[str] = []
        if isinstance(api_raw, dict):
            raw_available = list(api_raw.get('available_functions', []) or [])

        available_candidates: List[str] = []
        for fn in raw_available + surface_available:
            if fn:
                fn = str(fn)
                if fn not in available_candidates:
                    available_candidates.append(fn)

        # In retrieval context, available_functions means callable opportunities
        # not yet fully accounted for as discovered.
        available_remaining = [fn for fn in available_candidates if fn not in discovered]

        running_experiments = continuity_snapshot.get('running_experiments', 0)
        is_saturated = bool(available_candidates) and running_experiments == 0 and len(available_remaining) == 0

        return LLMRetrievalContext(
            episode=episode,
            tick=tick,
            phase='active',
            discovered_functions=discovered,
            available_functions=available_remaining,
            active_hypotheses=active_hypotheses,
            confirmed_hypotheses=confirmed_hypotheses,
            entropy=entropy,
            margin=margin,
            is_saturated=is_saturated,
        )

    @staticmethod
    def build_surfacing_protocol_payload(*, query: Any, surfaced: List[Any], retrieve_result: Any) -> Dict[str, Any]:
        records: List[Dict[str, Any]] = []
        for candidate in surfaced[:5]:
            obj = getattr(candidate, 'object', {}) if hasattr(candidate, 'object') else {}
            content = obj.get('content', {}) if isinstance(obj, dict) else {}
            tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
            fn = tool_args.get('function_name', obj.get('function_name', ''))
            tags = obj.get('retrieval_tags', []) if isinstance(obj, dict) else []
            records.append(
                {
                    'object_id': getattr(candidate, 'object_id', ''),
                    'function_name': fn,
                    'relevance': float(getattr(candidate, 'relevance_score', getattr(candidate, 'retrieval_score', 0.0)) or 0.0),
                    'failure_mode_hit': any('error' in str(tag).lower() or 'fail' in str(tag).lower() for tag in (tags or [])),
                    'plan_target_hit': query.context.get('plan_target_function') == fn if isinstance(query.context, dict) else False,
                    'belief_overlap': [kw for kw in query.query_text.split() if kw.startswith('belief:') and kw.split(':', 1)[1] in str(content)],
                }
            )
        contract = RetrievalRuntimeHelpers.normalize_retrieve_result_contract(retrieve_result)
        return {
            'query_text': getattr(query, 'query_text', ''),
            'surfaced_count': len(surfaced),
            'records': records,
            'retrieve_result_contract': {
                'selected_ids': contract.selected_ids,
                'action_influence': contract.action_influence,
                'candidate_count': contract.candidate_count,
            },
        }

    @staticmethod
    def build_memory_consumption_proof(
        *,
        tick: int,
        episode: int,
        action_to_use: Dict[str, Any],
        query: Any,
        result_reward: float,
        plan_target: Optional[str],
        active_beliefs: List[str],
    ) -> Dict[str, Any]:
        payload = action_to_use.get('payload', {}) if isinstance(action_to_use, dict) else {}
        tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
        fn_name = tool_args.get('function_name', 'wait') if action_to_use.get('kind') != 'wait' else 'wait'
        kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args, dict) else {}
        return {
            'tick': tick,
            'episode': episode,
            'selected_function': fn_name,
            'selected_kwargs_keys': sorted(list(kwargs.keys()))[:10] if isinstance(kwargs, dict) else [],
            'query_terms': str(getattr(query, 'query_text', '')).split()[:12],
            'plan_target': plan_target,
            'belief_context_used': active_beliefs[:3],
            'result_reward': float(result_reward),
        }

    @staticmethod
    def decide_hypothesis_augment(
        *,
        entropy: float,
        reward_stagnation: bool,
        signature_changed: bool,
        pending_recovery_probe: bool,
        pending_replan: bool,
        cooldown_ready: bool,
        tick: int,
        world_model_required_probe_count: int = 0,
        world_model_control_trust: float = 0.5,
        world_model_transition_confidence: float = 0.5,
        world_model_state_shift_risk: float = 0.0,
        hidden_state_drift_score: float = 0.0,
        hidden_state_uncertainty_score: float = 0.0,
        latent_branch_instability: float = 0.0,
    ) -> Any:
        return should_augment_hypotheses(
            signals=HypothesisAugmentSignals(
                entropy=entropy,
                reward_stagnation=reward_stagnation,
                signature_changed=signature_changed,
                pending_recovery_probe=pending_recovery_probe,
                pending_replan=pending_replan,
                world_model_required_probe_count=world_model_required_probe_count,
                world_model_control_trust=world_model_control_trust,
                world_model_transition_confidence=world_model_transition_confidence,
                world_model_state_shift_risk=world_model_state_shift_risk,
                hidden_state_drift_score=hidden_state_drift_score,
                hidden_state_uncertainty_score=hidden_state_uncertainty_score,
                latent_branch_instability=latent_branch_instability,
            ),
            cooldown_state=HypothesisAugmentCooldownState(cooldown_ready=cooldown_ready),
            context=HypothesisAugmentContext(tick=tick),
        )

    @staticmethod
    def normalize_retrieve_result_contract(retrieve_result: Any) -> RetrieveResultContract:
        raw_contract = getattr(retrieve_result, 'contract', None)
        if isinstance(raw_contract, dict):
            return RetrieveResultContract(
                selected_ids=[str(item) for item in raw_contract.get('selected_ids', []) if str(item)],
                action_influence=str(raw_contract.get('action_influence', 'none') or 'none'),
                candidate_count=int(raw_contract.get('candidate_count', 0) or 0),
            )
        selected_ids = []
        if hasattr(retrieve_result, 'selected_ids') and isinstance(retrieve_result.selected_ids, list):
            selected_ids = [str(item) for item in retrieve_result.selected_ids]
        action_influence = str(getattr(retrieve_result, 'action_influence', 'none') or 'none')
        candidates = getattr(retrieve_result, 'candidates', [])
        candidate_count = len(candidates) if isinstance(candidates, list) else 0
        return RetrieveResultContract(
            selected_ids=selected_ids,
            action_influence=action_influence,
            candidate_count=candidate_count,
        )
