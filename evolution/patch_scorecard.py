from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from evolution.proposal_generator import PatchProposal
from evolution.sandbox_runner import SandboxRunResult


@dataclass(frozen=True)
class PatchScorecard:
    verdict: str
    score: float
    tests_passed: bool
    requires_human_review: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": float(self.score),
            "tests_passed": bool(self.tests_passed),
            "requires_human_review": bool(self.requires_human_review),
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


def build_patch_scorecard(
    proposal: PatchProposal,
    sandbox_result: SandboxRunResult,
) -> PatchScorecard:
    total_commands = len(sandbox_result.command_results)
    passed_commands = len([item for item in sandbox_result.command_results if item.passed])
    pass_rate = (passed_commands / total_commands) if total_commands > 0 else 0.0
    surface_verdict = proposal.surface_verdict
    requires_human_review = bool(surface_verdict.requires_human_review) if surface_verdict else False
    surfaces_ok = bool(surface_verdict.accepted) if surface_verdict else False
    blocked_targets = list(surface_verdict.blocked_targets) if surface_verdict else []
    auditability = 1.0 if proposal.patch_text or proposal.file_overrides else 0.55

    score = (
        pass_rate * 0.55
        + (1.0 if surfaces_ok else 0.0) * 0.20
        + (0.70 if requires_human_review else 1.0) * 0.15
        + auditability * 0.10
    )
    score = max(0.0, min(1.0, score))

    reasons: List[str] = []
    if blocked_targets:
        reasons.extend(f"blocked_target:{target}" for target in blocked_targets)
    if not sandbox_result.all_passed:
        reasons.append("sandbox_failed")
    if requires_human_review:
        reasons.append("offline_only_requires_human_review")

    if blocked_targets or not sandbox_result.all_passed or not surfaces_ok:
        verdict = "reject"
    elif requires_human_review:
        verdict = "human_review"
    else:
        verdict = "accept"

    return PatchScorecard(
        verdict=verdict,
        score=score,
        tests_passed=bool(sandbox_result.all_passed),
        requires_human_review=requires_human_review,
        reasons=reasons,
        metrics={
            "pass_rate": round(pass_rate, 4),
            "total_commands": total_commands,
            "passed_commands": passed_commands,
            "touch_count": len(proposal.target_files),
            "blocked_targets": blocked_targets,
            "offline_only_targets": list(surface_verdict.offline_only_targets) if surface_verdict else [],
        },
    )
