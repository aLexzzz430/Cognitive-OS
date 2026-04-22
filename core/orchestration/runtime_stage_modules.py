from __future__ import annotations

from typing import Any, Callable

from core.orchestration.runtime_stage_contracts import (
    ApplyLearningUpdatesInput,
    ApplyLearningUpdatesOutput,
    PostCommitIntegrationInput,
    PostCommitIntegrationOutput,
    ProcessGraduationCandidatesInput,
    ProcessGraduationCandidatesOutput,
    Stage1RetrievalInput,
    Stage1RetrievalOutput,
    Stage2CandidateGenerationInput,
    Stage2GovernanceInput,
    Stage2PlanConstraintsInput,
    Stage2PredictionBridgeInput,
    Stage2SelfModelSuppressionInput,
    Stage3ExecutionInput,
    Stage3ExecutionOutput,
    Stage5EvidenceCommitInput,
    Stage5EvidenceCommitOutput,
    Stage6PostCommitInput,
    Stage6PostCommitOutput,
)


class _CallableStage:
    def __init__(self, handler: Callable[..., Any]) -> None:
        self._handler = handler


class Stage1RetrievalRuntime(_CallableStage):
    def run(self, stage_input: Stage1RetrievalInput) -> Stage1RetrievalOutput:
        raw = self._handler(stage_input)
        return Stage1RetrievalOutput(**raw)


class Stage2CandidateGenerationRuntime(_CallableStage):
    def run(self, stage_input: Stage2CandidateGenerationInput):
        return self._handler(stage_input)


class Stage2PlanConstraintsRuntime(_CallableStage):
    def run(self, stage_input: Stage2PlanConstraintsInput):
        return self._handler(stage_input)


class Stage2SelfModelSuppressionRuntime(_CallableStage):
    def run(self, stage_input: Stage2SelfModelSuppressionInput):
        return self._handler(stage_input)


class Stage2PredictionBridgeRuntime(_CallableStage):
    def run(self, stage_input: Stage2PredictionBridgeInput):
        return self._handler(stage_input)


class Stage2GovernanceRuntime(_CallableStage):
    def run(self, stage_input: Stage2GovernanceInput):
        return self._handler(stage_input)


class Stage3ExecutionRuntime(_CallableStage):
    def run(self, stage_input: Stage3ExecutionInput) -> Stage3ExecutionOutput:
        raw = self._handler(stage_input)
        return Stage3ExecutionOutput(**raw)


class Stage5EvidenceCommitRuntime(_CallableStage):
    def run(self, stage_input: Stage5EvidenceCommitInput) -> Stage5EvidenceCommitOutput:
        raw = self._handler(stage_input)
        return Stage5EvidenceCommitOutput(**raw)


class Stage6PostCommitRuntime(_CallableStage):
    def run(self, stage_input: Stage6PostCommitInput) -> Stage6PostCommitOutput:
        raw = self._handler(stage_input)
        return Stage6PostCommitOutput(**raw)


class PostCommitIntegrationRuntime(_CallableStage):
    def run(self, stage_input: PostCommitIntegrationInput) -> PostCommitIntegrationOutput:
        raw = self._handler(stage_input)
        return PostCommitIntegrationOutput(**raw)


class ProcessGraduationCandidatesRuntime(_CallableStage):
    def run(self, stage_input: ProcessGraduationCandidatesInput) -> ProcessGraduationCandidatesOutput:
        raw = self._handler(stage_input)
        return ProcessGraduationCandidatesOutput(**raw)


class ApplyLearningUpdatesRuntime(_CallableStage):
    def run(self, stage_input: ApplyLearningUpdatesInput) -> ApplyLearningUpdatesOutput:
        raw = self._handler(stage_input)
        return ApplyLearningUpdatesOutput(**raw)
