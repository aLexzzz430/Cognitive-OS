from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from evolution.patch_scorecard import PatchScorecard, build_patch_scorecard
from evolution.proposal_generator import PatchProposal
from evolution.sandbox_runner import SandboxRunResult, WorkspaceRunner
from evolution.strict_reaudit import StrictReauditVerdict, StrictReauditor


@dataclass(frozen=True)
class ProposalEvaluation:
    proposal: PatchProposal
    sandbox_result: SandboxRunResult
    scorecard: PatchScorecard
    strict_reaudit: StrictReauditVerdict

    def to_dict(self) -> Dict[str, object]:
        return {
            "proposal": self.proposal.to_dict(),
            "sandbox_result": self.sandbox_result.to_dict(),
            "scorecard": self.scorecard.to_dict(),
            "strict_reaudit": self.strict_reaudit.to_dict(),
        }


class ProposalEvaluator:
    def __init__(
        self,
        *,
        sandbox_runner: WorkspaceRunner | None = None,
        strict_reauditor: StrictReauditor | None = None,
    ) -> None:
        self._sandbox_runner = sandbox_runner or WorkspaceRunner()
        self._strict_reauditor = strict_reauditor or StrictReauditor()

    def evaluate(
        self,
        *,
        proposal: PatchProposal,
        source_dir: str,
        commands: Optional[Sequence[str]] = None,
        include_paths: Sequence[str] = (),
        env: Optional[Dict[str, str]] = None,
        task_ref: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> ProposalEvaluation:
        sandbox_result = self._sandbox_runner.run_patch_proposal(
            source_dir=source_dir,
            proposal=proposal,
            commands=commands,
            include_paths=include_paths,
            env=env,
            task_ref=task_ref,
            timeout_sec=timeout_sec,
        )
        scorecard = build_patch_scorecard(proposal, sandbox_result)
        strict_reaudit = self._strict_reauditor.reaudit_patch_proposal(
            proposal=proposal,
            sandbox_result=sandbox_result,
            scorecard=scorecard,
        )
        return ProposalEvaluation(
            proposal=proposal,
            sandbox_result=sandbox_result,
            scorecard=scorecard,
            strict_reaudit=strict_reaudit,
        )
