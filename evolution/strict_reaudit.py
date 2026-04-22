from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from evolution.allowed_surfaces import sanitize_reasoning_payload
from evolution.patch_scorecard import PatchScorecard
from evolution.proposal_generator import PatchProposal
from evolution.sandbox_runner import SandboxRunResult


@dataclass(frozen=True)
class StrictReauditVerdict:
    status: str
    accepted_for_merge: bool
    requires_human_review: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "accepted_for_merge": bool(self.accepted_for_merge),
            "requires_human_review": bool(self.requires_human_review),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ReasoningReauditVerdict:
    accepted: bool
    violations: List[str] = field(default_factory=list)
    sanitized_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "violations": list(self.violations),
        }


class StrictReauditor:
    def reaudit_patch_proposal(
        self,
        *,
        proposal: PatchProposal,
        sandbox_result: SandboxRunResult,
        scorecard: PatchScorecard,
    ) -> StrictReauditVerdict:
        reasons: List[str] = []
        surface_verdict = proposal.surface_verdict
        if surface_verdict is None:
            reasons.append("missing_surface_verdict")
        elif surface_verdict.blocked_targets:
            reasons.extend(f"blocked_target:{target}" for target in surface_verdict.blocked_targets)

        if not sandbox_result.all_passed:
            reasons.append("sandbox_failed")
        if scorecard.verdict == "reject":
            reasons.append("scorecard_rejected")

        requires_human_review = bool(surface_verdict.requires_human_review) if surface_verdict else False
        if reasons:
            return StrictReauditVerdict(
                status="reject",
                accepted_for_merge=False,
                requires_human_review=requires_human_review,
                reasons=reasons,
            )
        if requires_human_review:
            return StrictReauditVerdict(
                status="human_review",
                accepted_for_merge=False,
                requires_human_review=True,
                reasons=["offline_only_patch_requires_human_review"],
            )
        return StrictReauditVerdict(
            status="accept",
            accepted_for_merge=True,
            requires_human_review=False,
            reasons=[],
        )

    def reaudit_reasoning_payload(self, payload: Dict[str, Any]) -> ReasoningReauditVerdict:
        sanitized_payload, violations = sanitize_reasoning_payload(payload if isinstance(payload, dict) else {})
        return ReasoningReauditVerdict(
            accepted=not violations,
            violations=violations,
            sanitized_payload=sanitized_payload,
        )
