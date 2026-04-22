from __future__ import annotations

from core.orchestration.stage_types import StateSyncStageInput, StateSyncStageOutput
from modules.world_model.state_sync import emit_state_patch as emit_world_model_state_patch


class StateSyncStage:
    """Stage-4 state sync orchestration for world/self/decision patches and tick finalization."""

    def run(self, loop, stage_input: StateSyncStageInput) -> StateSyncStageOutput:
        belief_count = int(loop._belief_ledger.belief_count())
        active_beliefs = loop._belief_ledger.get_active_beliefs()
        established_beliefs = loop._belief_ledger.get_established_beliefs()
        high_uncertainty_beliefs = [
            belief for belief in active_beliefs
            if float(getattr(belief, 'uncertainty', 0.0) or 0.0) >= 0.6
        ]
        loop._state_sync.sync(
            loop._state_sync_input_cls(
                updates=emit_world_model_state_patch(
                    belief_count=belief_count,
                    active_beliefs=active_beliefs,
                    established_beliefs=established_beliefs,
                    high_uncertainty_beliefs=high_uncertainty_beliefs,
                ),
                reason='world_model_belief_patch',
            )
        )
        next_obs = loop.observe()
        loop._active_tick_context_frame = None
        return StateSyncStageOutput(next_obs=next_obs)
