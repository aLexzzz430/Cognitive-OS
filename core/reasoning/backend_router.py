from __future__ import annotations

from core.reasoning.backend import DeliberationBudget, ReasoningBackend, ReasoningRequest
from core.reasoning.backend_llm import LLMReasonerBackend
from core.reasoning.backend_symbolic import DeterministicReasonerBackend, SearchReasonerBackend, SymbolicReasoningBackend


class ReasoningBackendRouter:
    def __init__(
        self,
        *,
        symbolic_backend: ReasoningBackend | None = None,
        deterministic_backend: ReasoningBackend | None = None,
        search_backend: ReasoningBackend | None = None,
        llm_backend: ReasoningBackend | None = None,
    ) -> None:
        self._symbolic_backend = symbolic_backend or SymbolicReasoningBackend()
        self._deterministic_backend = deterministic_backend or DeterministicReasonerBackend()
        self._search_backend = search_backend or SearchReasonerBackend()
        self._llm_backend = llm_backend or LLMReasonerBackend()

    def route(
        self,
        request: ReasoningRequest,
        budget: DeliberationBudget,
    ) -> ReasoningBackend:
        preference = str((request.workspace or {}).get("reasoning_backend_preference", "") or "").strip().lower()
        if budget.mode == "slow" and request.llm_client is not None and hasattr(request.llm_client, "reason_about_workspace"):
            return self._llm_backend
        if preference == "search":
            return self._search_backend
        if preference == "deterministic":
            return self._deterministic_backend
        return self._symbolic_backend
