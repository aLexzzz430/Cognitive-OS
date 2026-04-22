from __future__ import annotations

from typing import Any, Dict, Optional

from modules.world_model.protocol import WorldModelControlProtocol
from modules.world_model.events import EventType, WorldModelEvent


class RecoveryFlow:
    """Recovery decision workflow; side effects are collected as commands."""

    def __init__(self, loop: Any, adjudication_flow: Any):
        self.loop = loop
        self.adjudication_flow = adjudication_flow

    def handle_recovery_if_needed(self, action: dict, result: dict, policy: Any, effects: list) -> Optional[Dict[str, Any]]:
        na_result = result.get('novel_api', {}) if isinstance(result, dict) else {}
        if hasattr(na_result, '_data'):
            na_result = na_result._data
        error_message = na_result.get('error') if isinstance(na_result, dict) else None
        error_context = {
            'failed_action': action,
            'error_result': {'error': error_message} if error_message else {},
            'hypotheses_active': len(self.loop._hypotheses.get_active()),
            'hypotheses_confirmed': len(self.loop._hypotheses.get_confirmed()),
            'recent_commits': len(self.loop._commit_log),
        }
        recovery_gate = self.loop._recovery.should_trigger_recovery(error_context)
        if not recovery_gate.get('should_recover'):
            return None

        diagnosis = self.loop._recovery.diagnose_error(error_context)
        wm_summary = self.loop._build_world_model_context(getattr(self.loop, '_last_perception_summary', {}))
        wm_transition_priors = self.loop._build_world_model_transition_priors(getattr(self.loop, '_last_perception_summary', {}))
        wm_control = WorldModelControlProtocol.from_context({'world_model_summary': wm_summary, 'world_model_transition_priors': wm_transition_priors})
        paths = self.loop._recovery.suggest_recovery_paths(diagnosis, {
            'active_hypotheses': len(self.loop._hypotheses.get_active()),
            'hypotheses_confirmed': len(self.loop._hypotheses.get_confirmed()),
            'entropy': self.loop._hypotheses.entropy(),
            'world_model_control': wm_control.to_dict(),
            'world_model_transition_priors': wm_transition_priors,
        })
        path = paths[0] if paths else None
        if path is None:
            return None

        recovery_task_id = f"recovery_{self.loop._episode}_{self.loop._tick}"
        effects.append({
            'command': 'event_emit',
            'args': {
                'event': WorldModelEvent(
                    event_type=EventType.RECOVERY_EXECUTED,
                    episode=self.loop._episode,
                    tick=self.loop._tick,
                    data={
                        'recovery_task_id': recovery_task_id,
                        'recovery_type': path.recovery_type.value,
                        'function_name': getattr(path, 'suggested_action', None),
                        'estimated_success': path.estimated_success_probability,
                    },
                    source_stage='recovery',
                ),
            },
        })
        effects.append({
            'command': 'continuity_add_task',
            'args': {
                'task_id': recovery_task_id,
                'description': diagnosis.description or f"Recover from {diagnosis.error_type.value}",
                'priority': 0.95,
                'metadata': {
                    'recovery_type': path.recovery_type.value,
                    'estimated_success': path.estimated_success_probability,
                    'suggested_action': getattr(path, 'suggested_action', None),
                },
            },
        })

        pending_probe_patch = None
        pending_replan_patch = None
        recovery_type = str(getattr(path.recovery_type, 'value', path.recovery_type) or '').lower()
        if recovery_type == 'request_probe':
            pending_probe_patch = {'task_id': recovery_task_id, 'diagnosis': diagnosis, 'path': path, 'tick': self.loop._tick}
            self.adjudication_flow.mark_probe_task_injected(recovery_task_id, path.recovery_type.value)
        elif recovery_type == 'request_replan':
            pending_replan_patch = {
                'task_id': recovery_task_id,
                'diagnosis': diagnosis,
                'path': path,
                'tick': self.loop._tick,
                'world_model_control': wm_control.to_dict(),
                'world_model_transition_priors': wm_transition_priors,
                'world_model_summary': wm_summary,
            }

        self.adjudication_flow.maybe_inject_recovery_task(
            recovery_task_id=recovery_task_id,
            description=diagnosis.description,
            path=path,
            policy=policy,
        )

        summary = self.loop._recovery.summarize_failure(
            failure_log=result,
            error_diagnosis=diagnosis,
            recovery_attempted=path.recovery_type,
            recovery_success=False,
        )
        recovery_event = {
            'event_type': 'recovery_decision',
            'tick': self.loop._tick,
            'episode': self.loop._episode,
            'diagnosis': {
                'error_type': diagnosis.error_type.value,
                'description': diagnosis.description,
                'root_cause': diagnosis.root_cause_hypothesis,
            },
            'path': {
                'recovery_type': path.recovery_type.value,
                'estimated_success': path.estimated_success_probability,
                'suggested_action': getattr(path, 'suggested_action', None),
            },
            'summary': summary,
            'pending_probe_patch': pending_probe_patch,
            'pending_replan_patch': pending_replan_patch,
            'path_affected_execution': True,
        }
        self.loop._recovery_log.append(recovery_event)
        return recovery_event
