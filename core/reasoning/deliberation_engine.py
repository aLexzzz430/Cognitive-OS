from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence

from core.orchestration.action_utils import extract_action_function_name
from core.reasoning.backend import ReasoningRequest, ReasoningResult
from core.reasoning.backend_router import ReasoningBackendRouter
from core.reasoning.budget_router import BudgetRouter
from evolution.strict_reaudit import StrictReauditor


def _safe_action_summary(action: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
    return {
        "function_name": extract_action_function_name(action, default=""),
        "kind": str(action.get("kind", "") or ""),
        "source": str(action.get("_source", "") or ""),
        "deliberation_engine_score": float(meta.get("deliberation_engine_score", 0.0) or 0.0),
        "deliberation_engine_rank": int(meta.get("deliberation_engine_rank", 0) or 0),
    }


def _hypothesis_object_id(row: Dict[str, Any], index: int) -> str:
    for key in ("object_id", "hypothesis_id", "id"):
        text = str(row.get(key, "") or "").strip()
        if text:
            return text
    return f"deliberation_hypothesis_{index + 1}"


def _hypothesis_object_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from core.objects.adapters import proposal_to_object_record

    objects: List[Dict[str, Any]] = []
    for index, row in enumerate(list(rows or [])):
        if not isinstance(row, dict):
            continue
        object_id = _hypothesis_object_id(row, index)
        proposal = dict(row)
        proposal.setdefault("object_type", "hypothesis")
        proposal.setdefault("family", str(row.get("hypothesis_type", row.get("family", "generic")) or "generic"))
        proposal.setdefault(
            "summary",
            str(row.get("summary", row.get("description", row.get("hypothesis_id", object_id))) or object_id),
        )
        proposal.setdefault("confidence", row.get("posterior", row.get("confidence", 0.0)))
        if not proposal.get("source_stage"):
            proposal["source_stage"] = "deliberation"
        objects.append(proposal_to_object_record(proposal, object_id=object_id))
    return objects


class DeliberationEngine:
    def __init__(
        self,
        *,
        budget_router: BudgetRouter | None = None,
        backend_router: ReasoningBackendRouter | None = None,
        strict_reauditor: StrictReauditor | None = None,
    ) -> None:
        self._budget_router = budget_router or BudgetRouter()
        self._backend_router = backend_router or ReasoningBackendRouter()
        self._strict_reauditor = strict_reauditor or StrictReauditor()

    def deliberate(
        self,
        *,
        workspace: Dict[str, Any],
        obs: Dict[str, Any],
        surfaced: Sequence[Any],
        candidate_actions: Sequence[Dict[str, Any]],
        continuity_snapshot: Dict[str, Any],
        task_family: str = "",
        available_functions: Sequence[str] = (),
        llm_client: Any = None,
        structured_answer_synthesizer: Any = None,
    ) -> Dict[str, Any]:
        budget = self._budget_router.route(
            workspace=workspace,
            obs=obs,
            candidate_actions=candidate_actions,
            task_family=task_family,
        )
        request = ReasoningRequest(
            workspace=dict(workspace or {}),
            obs=dict(obs or {}),
            surfaced=list(surfaced or []),
            candidate_actions=[deepcopy(action) for action in candidate_actions if isinstance(action, dict)],
            continuity_snapshot=dict(continuity_snapshot or {}),
            task_family=str(task_family or ""),
            available_functions=list(available_functions or []),
            llm_client=llm_client,
            structured_answer_synthesizer=structured_answer_synthesizer,
        )
        backend = self._backend_router.route(request, budget)
        result = backend.deliberate(request, budget)
        if not isinstance(result, ReasoningResult):
            raise TypeError("Reasoning backend must return ReasoningResult")
        payload = result.to_dict()
        reaudit = self._strict_reauditor.reaudit_reasoning_payload(payload)
        payload = reaudit.sanitized_payload
        trace = list(payload.get("deliberation_trace", []) or [])
        if reaudit.violations:
            trace.append({
                "stage": "strict_reaudit",
                "status": "blocked_hidden_controller",
                "violations": list(reaudit.violations),
            })
        payload["deliberation_trace"] = trace
        payload["budget"] = budget.to_dict()
        payload["backend"] = getattr(backend, "name", payload.get("backend", "symbolic"))
        payload["mode"] = budget.mode
        payload["strict_reaudit"] = reaudit.to_dict()
        payload.setdefault("ranked_discriminating_experiments", [])
        payload.setdefault("posterior_summary", {})
        payload.setdefault("control_policy", {})
        payload.setdefault("active_test_ids", [])
        ranked_actions = [
            deepcopy(action)
            for action in list(payload.get("ranked_candidate_actions", []) or [])
            if isinstance(action, dict)
        ]
        ranked_hypotheses = [
            deepcopy(row)
            for row in list(payload.get("ranked_candidate_hypotheses", []) or [])
            if isinstance(row, dict)
        ]
        hypothesis_objects = _hypothesis_object_records(ranked_hypotheses)
        payload["ranked_candidate_hypothesis_objects"] = hypothesis_objects
        payload["ranked_candidate_hypothesis_refs"] = [
            str(item.get("object_id", "") or "")
            for item in hypothesis_objects
            if str(item.get("object_id", "") or "")
        ]
        posterior_summary = (
            dict(payload.get("posterior_summary", {}) or {})
            if isinstance(payload.get("posterior_summary", {}), dict)
            else {}
        )
        if posterior_summary:
            leading_hypothesis_id = str(posterior_summary.get("leading_hypothesis_id", "") or "")
            if leading_hypothesis_id:
                posterior_summary.setdefault("leading_hypothesis_object_id", leading_hypothesis_id)
            posterior_summary["ranked_hypothesis_object_ids"] = list(payload["ranked_candidate_hypothesis_refs"])
            payload["posterior_summary"] = posterior_summary
        if isinstance(payload.get("ranked_discriminating_experiments"), list):
            if not any(
                isinstance(row, dict) and row.get("stage") == "discriminating_experiment_selection"
                for row in trace
            ):
                trace.append(
                    {
                        "stage": "discriminating_experiment_selection",
                        "ranked_count": len(
                            [
                                row
                                for row in list(payload.get("ranked_discriminating_experiments", []) or [])
                                if isinstance(row, dict)
                            ]
                        ),
                    }
                )
                payload["deliberation_trace"] = trace
        payload["ranked_candidate_action_summaries"] = [
            _safe_action_summary(action)
            for action in ranked_actions
        ]
        payload["_ranked_candidate_actions"] = ranked_actions
        return payload
