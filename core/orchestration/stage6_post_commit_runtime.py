from __future__ import annotations

from typing import Any, Dict

from core.orchestration.runtime_stage_contracts import Stage6PostCommitInput


def run_stage6_post_commit(loop: Any, stage_input: Stage6PostCommitInput) -> Dict[str, Any]:
    committed_ids = stage_input.committed_ids
    obs_before = stage_input.obs_before
    result = stage_input.result
    action_to_use = stage_input.action_to_use
    reward = stage_input.reward

    runtime_out = loop._planner_runtime.tick(
        phase='progress',
        obs=obs_before,
        selected_action=action_to_use,
        result=result,
        reward=reward,
    )
    loop._consume_planner_runtime_result(runtime_out, fallback_action=action_to_use)
    integration_summary = loop._post_commit_integration(committed_ids, obs_before, result)
    loop._grad_tracker.on_commit_epoch_end(epoch=loop._tick)
    loop._process_graduation_candidates()
    loop._collect_outcome_learning_signal(
        action_to_use=action_to_use,
        obs_before=obs_before,
        result=result,
        reward=reward,
    )
    loop._state_mgr.commit_tick({
        'episode': loop._episode,
        'tick': loop._tick,
        'action': str(action_to_use)[:80],
        'reward': reward,
        'committed': len(committed_ids),
        'post_commit_integration': integration_summary,
    })
    loop._write_world_model_state()
    return {'integration_summary': integration_summary}
