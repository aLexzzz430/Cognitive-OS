from __future__ import annotations

from core.orchestration.stage_types import RetrievalStageInput, RetrievalStageOutput
from decision.state_sync import emit_state_patch as emit_decision_state_patch, emit_surfacing_patch


class RetrievalStage:
    """Stage-1 orchestration wrapper around retrieval/gating helpers in CoreMainLoop."""

    def run(self, loop, stage_input: RetrievalStageInput) -> RetrievalStageOutput:
        raw = loop._stage1_retrieval(
            stage_input.obs_before,
            stage_input.context,
            stage_input.continuity_snapshot,
        )
        loop._state_sync.sync(
            loop._state_sync_input_cls(
                updates=emit_decision_state_patch(
                    retrieval_aux_decisions=dict(getattr(loop, '_last_retrieval_aux_decisions', {}))
                ),
                reason='retrieval_aux_gate_decisions',
            )
        )
        loop._state_sync.sync(
            loop._state_sync_input_cls(
                updates=emit_surfacing_patch(surfacing_protocol=raw.get('surfacing_protocol', {})),
                reason='step3.5 surfacing protocol sync',
            )
        )
        return RetrievalStageOutput(
            query=raw.get('query'),
            retrieve_result=raw.get('retrieve_result'),
            surfaced=raw.get('surfaced', []),
            surfacing_protocol=raw.get('surfacing_protocol', {}),
            llm_retrieval_ctx=raw.get('llm_retrieval_ctx'),
            budget=raw.get('budget', {}),
        )
