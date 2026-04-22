from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from core.orchestration.continuity_persistence_adapter import ContinuityPersistenceAdapter


class EpisodeLifecycle:
    """Orchestrates boot + episode-end lifecycle hooks."""

    def __init__(self, continuity_persistence: ContinuityPersistenceAdapter):
        self._continuity_persistence = continuity_persistence

    def on_boot(self, loop) -> None:
        self._continuity_persistence._load_continuity(loop)
        self._continuity_persistence._bootstrap_continuity(loop)

    def on_episode_end(self, loop, episode: int) -> None:
        loop._grad_tracker.on_episode_end(episode)
        self._process_representation_candidates(loop)
        self._create_episode_record(loop, episode)

        # Learning budget is owned/reset by lifecycle before learning pass.
        loop._learning_updates_sent_this_episode = 0
        self._run_episode_learning(loop, episode)

        self._update_continuity_progress_from_episode_outcome(loop, episode)
        self._continuity_persistence._save_continuity(loop)

    def _create_episode_record(self, loop, episode: int) -> None:
        from modules.memory.schema import MemoryType, MemoryLayer, MemoryMetadata

        committed_objs = loop.get_committed_objects(100)
        committed_obj_ids = [obj.get('object_id', obj.get('id', '')) for obj in committed_objs]

        episode_record = loop._episode_summarizer.summarize(
            episode_id=episode,
            episode_trace=loop._episode_trace,
            committed_object_ids=committed_obj_ids,
            hypothesis_ids=[h.id for h in loop._hypotheses.iter_hypotheses() if h.created_at_episode == episode],
            belief_delta={'total': loop._belief_ledger.belief_count()},
            teacher_present=(
                episode <= loop._grad_tracker.teacher_exit_episode
                if hasattr(loop._grad_tracker, 'teacher_exit_episode')
                else False
            ),
            teacher_exit_episode=getattr(loop._grad_tracker, 'teacher_exit_episode', None),
            retention_competition_summary=self._build_retention_competition_summary(loop, episode),
        )

        proposal = {
            'type': 'memory_proposal',
            'memory_type': MemoryType.EPISODE_RECORD.value,
            'memory_layer': MemoryLayer.EPISODIC.value,
            'content': episode_record.to_dict(),
            'confidence': 0.8,
            'evidence_ids': [],
        }

        metadata = MemoryMetadata(
            memory_type=MemoryType.EPISODE_RECORD,
            memory_layer=MemoryLayer.EPISODIC,
            confidence=0.8,
            source_episode_ids=[episode],
            retrieval_tags=['episode', f'episode_{episode}'],
        )
        proposal['memory_metadata'] = metadata.to_dict()

        decision = loop._validator.validate(proposal)
        if decision.decision == 'accept_new':
            committed_ids = loop.commit_objects([proposal])
            if committed_ids:
                loop._last_episode_record_id = committed_ids[0]
                if hasattr(loop, '_event_log') and loop._event_log:
                    loop._event_log.append({
                        'event_type': 'episode_record_created',
                        'episode': episode,
                        'tick': loop._tick,
                        'data': {
                            'episode_record_id': loop._last_episode_record_id,
                            'episode_id': episode,
                            'tick_count': episode_record.tick_count,
                            'total_reward': episode_record.total_reward,
                            'retention_competition_summary': dict(episode_record.retention_competition_summary),
                        },
                        'source_module': 'core',
                        'source_stage': 'episode_end',
                    })

    def _build_retention_competition_summary(self, loop, episode: int) -> Dict[str, Any]:
        learning_rows = [
            row for row in getattr(loop, '_learning_signal_log', [])
            if isinstance(row, dict) and int(row.get('episode', -1)) == int(episode)
        ]
        governance_rows = [
            row for row in getattr(loop, '_governance_log', [])
            if (
                isinstance(row, dict)
                and int(row.get('episode', -1)) == int(episode)
                and isinstance(row.get('selected_world_model_competition', {}), dict)
            )
        ]
        competition_rows = [
            row.get('selected_world_model_competition', {})
            for row in governance_rows
            if isinstance(row.get('selected_world_model_competition', {}), dict)
        ]

        failure_counter: Counter[str] = Counter()
        failure_severity: Dict[str, float] = {}
        for row in learning_rows:
            failure_type = str(row.get('retention_failure_type', '') or '')
            if not failure_type:
                continue
            severity = max(0.0, min(1.0, float(row.get('retention_failure_severity', 0.0) or 0.0)))
            failure_counter[failure_type] += 1
            failure_severity[failure_type] = failure_severity.get(failure_type, 0.0) + severity

        dominant_failure_type = ''
        if failure_counter:
            dominant_failure_type = max(
                failure_counter,
                key=lambda name: (int(failure_counter[name]), float(failure_severity.get(name, 0.0))),
            )

        latest_dominant_branch_id = ''
        latest_governance_reason = ''
        for row in reversed(governance_rows):
            summary = row.get('selected_world_model_competition', {}) if isinstance(row.get('selected_world_model_competition', {}), dict) else {}
            if not latest_dominant_branch_id:
                latest_dominant_branch_id = str(summary.get('dominant_branch_id', '') or '')
            if not latest_governance_reason:
                latest_governance_reason = str(row.get('reason', '') or '')
            if latest_dominant_branch_id and latest_governance_reason:
                break

        return {
            'learning_signal_count': len(learning_rows),
            'retention_failure_count': sum(1 for row in learning_rows if str(row.get('retention_failure_type', '') or '')),
            'dominant_retention_failure_type': dominant_failure_type,
            'max_retention_failure_severity': max(
                [max(0.0, min(1.0, float(row.get('retention_failure_severity', 0.0) or 0.0))) for row in learning_rows],
                default=0.0,
            ),
            'governance_decision_count': len(governance_rows),
            'competition_active_count': sum(1 for row in competition_rows if bool(row.get('competition_active', False))),
            'required_probe_preserved_count': sum(1 for row in competition_rows if bool(row.get('preserved_required_probe', False))),
            'anchor_preserved_count': sum(1 for row in competition_rows if bool(row.get('preserved_branch_anchor', False))),
            'risky_branch_avoided_count': sum(1 for row in competition_rows if bool(row.get('avoided_risky_branch_action', False))),
            'max_probe_pressure': max(
                [max(0.0, min(1.0, float(row.get('probe_pressure', 0.0) or 0.0))) for row in competition_rows],
                default=0.0,
            ),
            'max_latent_instability': max(
                [max(0.0, min(1.0, float(row.get('latent_instability', 0.0) or 0.0))) for row in competition_rows],
                default=0.0,
            ),
            'latest_dominant_branch_id': latest_dominant_branch_id,
            'latest_governance_reason': latest_governance_reason,
        }

    def _process_representation_candidates(self, loop) -> None:
        if not loop._episode_trace:
            return

        trajectories = [[
            loop._trajectory_entry_cls(
                tick=entry['tick'],
                observation=entry['observation'],
                action=entry['action'],
                outcome=entry['outcome'],
                reward=entry['reward'],
            )
            for entry in loop._episode_trace
        ]]
        families = [f'episode_{loop._episode}']
        cards = loop._repr_proposer.propose_from_trajectories(trajectories, families)
        for card in cards[:2]:
            obj_id = loop._repr_proposer.commit_via_step10(card, loop._shared_store)
            if not obj_id:
                continue
            loop._runtime_store.record_support(card.card_id, loop._tick, {'episode': loop._episode}, {'object_id': obj_id})
            if loop._teacher_allows_intervention():
                loop._teacher.teacher_proposal(
                    target_id=card.card_id,
                    target_type='representation',
                    content={'pattern': card.proposed_pattern, 'object_id': obj_id},
                    rationale='Episode trajectory distilled into representation card',
                    actor='system_llm',
                )
            loop._representation_log.append({
                'episode': loop._episode,
                'card_id': card.card_id,
                'object_id': obj_id,
            })
        loop._continuity.experiments.record_result('exp_main_loop', {
            'episode': loop._episode,
            'representations': len(cards[:2]),
            'reward': loop._episode_reward,
        })

    def _run_episode_learning(self, loop, episode: int) -> None:
        if not getattr(loop, '_learning_enabled', False):
            return

        trace_assigner = getattr(loop._credit_assignment, 'assign_credit_from_trace', None)
        recent_traces = loop._causal_trace.get_recent_traces(n=10)
        if callable(trace_assigner):
            for trace in recent_traces:
                if not getattr(trace, 'execution', None):
                    continue
                assignments = trace_assigner(
                    trace=trace,
                    outcome_success=trace.execution.success,
                    reward=trace.execution.reward,
                )
                loop._apply_learning_updates(assignments)

        promotion_engine = getattr(loop, '_promotion_engine', None)
        promo_candidates = promotion_engine.check_promotions(episode) if promotion_engine is not None else []
        for candidate in promo_candidates:
            payload = promotion_engine.to_update_payload(candidate, episode=episode, tick=loop._tick)
            promo_result = 'disabled'
            promotion_engine.record_promotion_result(candidate, promo_result)
            loop._learning_update_log.append({
                'episode': episode,
                'tick': loop._tick,
                'object_id': candidate.object_id,
                'update_type': 'asset_status_promote',
                'target_status': payload['asset_status'],
                'evidence': payload['evidence'],
                'result': promo_result,
            })
            if loop.verbose:
                print(
                    f"  Sprint 6: Promotion proposal {candidate.object_id} "
                    f"{candidate.current_status} -> {candidate.target_status} ({promo_result})"
                )

        loop._apply_learning_policy_updates(episode)
        loop._resource_state.update_memory_state(
            objects_count=loop._shared_store.count_objects(),
            utilization=0.0,
        )

    def _update_continuity_progress_from_episode_outcome(self, loop, episode: int) -> None:
        goal = loop._continuity.goals.get_goal('goal_explore')
        if goal is None or goal.status != 'active':
            return
        commits_this_episode = sum(1 for rec in loop._commit_log if int(rec.get('episode') or -1) == int(episode))
        positive_outcome = 1 if float(loop._episode_reward or 0.0) > 0.0 else 0
        progress_delta = min(0.25, commits_this_episode * 0.02 + positive_outcome * 0.05)
        if progress_delta <= 0.0:
            return
        loop._continuity.goals.update_progress(
            'goal_explore',
            min(1.0, float(goal.progress) + progress_delta),
            milestone={'description': f'episode_{episode}_outcome'},
        )
