from __future__ import annotations

from core.reasoning.backend import DeliberationBudget, ReasoningRequest, ReasoningResult


class LLMReasonerBackend:
    """Optional backend placeholder. It never takes control away from formal routing."""

    name = "llm"

    def deliberate(
        self,
        request: ReasoningRequest,
        budget: DeliberationBudget,
    ) -> ReasoningResult:
        return ReasoningResult(
            budget=budget.to_dict(),
            backend=f"{self.name}_unavailable",
            mode=budget.mode,
            deliberation_trace=[{
                "stage": "llm_backend",
                "status": "skipped",
                "reason": "formal_symbolic_path_preferred",
            }],
        )


LLMReasoningBackend = LLMReasonerBackend
