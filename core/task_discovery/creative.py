from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from modules.llm.json_adaptor import normalize_llm_output

from core.task_discovery.detectors import DiscoveryContext
from core.task_discovery.models import READ_ONLY_ACTIONS, TASK_SOURCES, TaskCandidate, clamp01, string_list


CREATIVE_TASK_GENERATOR_VERSION = "conos.task_discovery.creative/v1"

_ALLOWED_ACTIONS = {
    "read_logs",
    "read_reports",
    "read_files",
    "run_readonly_analysis",
    "write_report",
    "run_eval",
    "run_tests",
    "propose_patch",
    "edit_in_mirror",
}
_DEFAULT_FORBIDDEN = [
    "modify_core_runtime_without_approval",
    "sync_back_without_verified_patch",
    "spend_api_budget_without_budget_policy",
]


class CreativeTaskGenerator:
    """LLM-assisted task ideation constrained by real evidence and schema gates."""

    def __init__(
        self,
        llm_client: Any,
        *,
        max_candidates: int = 3,
        timeout_sec: float = 60.0,
        max_tokens: int = 1400,
        temperature: float = 0.4,
    ) -> None:
        self.llm_client = llm_client
        self.max_candidates = max(0, min(5, int(max_candidates or 3)))
        self.timeout_sec = float(timeout_sec or 60.0)
        self.max_tokens = int(max_tokens or 1400)
        self.temperature = float(temperature)
        self.last_trace: Dict[str, Any] = {}

    def generate(self, context: DiscoveryContext, seed_candidates: Sequence[TaskCandidate]) -> list[TaskCandidate]:
        self.last_trace = {
            "schema_version": CREATIVE_TASK_GENERATOR_VERSION,
            "enabled": True,
            "generated_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "error": "",
        }
        seeds = list(seed_candidates or [])[:8]
        if self.max_candidates <= 0 or not seeds or self.llm_client is None:
            self.last_trace["error"] = "missing_client_or_seed_candidates"
            return []
        prompt = self._build_prompt(context, seeds)
        try:
            raw = self.llm_client.complete(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                think=True,
                thinking_budget=512,
                timeout_sec=self.timeout_sec,
            )
        except TypeError:
            try:
                raw = self.llm_client.complete(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout_sec=self.timeout_sec,
                )
            except Exception as exc:
                self.last_trace["error"] = f"{type(exc).__name__}: {exc}"
                return []
        except Exception as exc:
            self.last_trace["error"] = f"{type(exc).__name__}: {exc}"
            return []
        result = normalize_llm_output(raw, output_kind="creative_task_candidates", expected_type="list")
        self.last_trace["adapter_trace"] = result.to_trace()
        if not result.ok:
            self.last_trace["error"] = result.error or "parse_failed"
            return []
        known_refs = self._known_refs(seeds)
        accepted: list[TaskCandidate] = []
        rejected: list[Dict[str, Any]] = []
        for index, row in enumerate(result.parsed_list()[: self.max_candidates]):
            if not isinstance(row, Mapping):
                rejected.append({"index": index, "reason": "not_object"})
                continue
            candidate, reason = self._candidate_from_row(dict(row), known_refs=known_refs, index=index)
            if candidate is None:
                rejected.append({"index": index, "reason": reason, "row_excerpt": str(row)[:240]})
                continue
            accepted.append(candidate)
        self.last_trace["generated_count"] = len(result.parsed_list())
        self.last_trace["accepted_count"] = len(accepted)
        self.last_trace["rejected_count"] = len(rejected)
        self.last_trace["rejected"] = rejected
        return accepted

    def _build_prompt(self, context: DiscoveryContext, seeds: Sequence[TaskCandidate]) -> str:
        seed_payload = [
            {
                "task_id": item.task_id,
                "source": item.source,
                "observation": item.observation,
                "gap": item.gap,
                "priority": item.priority,
                "source_refs": item.source_refs,
                "allowed_actions": item.allowed_actions,
                "forbidden_actions": item.forbidden_actions,
            }
            for item in seeds
        ]
        constraints = list(context.goal_ledger.constraints or [])
        return (
            "You are the Creating state of Con OS Task Discovery.\n"
            "Generate novel but evidence-grounded task candidates. Do not execute anything.\n"
            "You may synthesize across seeds, but every candidate must cite seed task_ids or source_refs.\n"
            "Prefer tasks that expose a hidden bottleneck, test an important hypothesis, or create a measurable next step.\n"
            "Avoid cosmetic cleanup, dashboards, broad refactors, or vague optimization tasks.\n"
            "Allowed sources: failure_residue, goal_gap, user_feedback, code_health, hypothesis, opportunity.\n"
            "Allowed actions are only read_logs, read_reports, read_files, run_readonly_analysis, write_report, run_eval, run_tests, propose_patch, edit_in_mirror.\n"
            "Use L0/L1/limited_L2 only. Any sync-back, credentials, network spend, or release action must be forbidden.\n"
            "Return ONLY a JSON list with at most "
            f"{self.max_candidates} objects. Each object must include:\n"
            "source, observation, gap, proposed_task, expected_value, goal_alignment, evidence_strength, feasibility, risk, cost, reversibility, distraction_penalty, evidence_needed, success_condition, allowed_actions, forbidden_actions, evidence_refs, permission_level.\n\n"
            f"North Star: {context.goal_ledger.north_star}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Seed candidates:\n{json.dumps(seed_payload, ensure_ascii=False, indent=2)}\n"
        )

    def _candidate_from_row(
        self,
        row: Mapping[str, Any],
        *,
        known_refs: set[str],
        index: int,
    ) -> tuple[TaskCandidate | None, str]:
        evidence_refs = string_list(row.get("evidence_refs") or row.get("source_refs"))
        if not evidence_refs:
            return None, "missing_evidence_refs"
        if not any(ref in known_refs for ref in evidence_refs):
            return None, "evidence_refs_not_grounded"
        source = str(row.get("source") or "opportunity").strip()
        if source not in TASK_SOURCES:
            return None, "invalid_source"
        allowed = [action for action in string_list(row.get("allowed_actions")) if action in _ALLOWED_ACTIONS]
        if not allowed:
            allowed = list(READ_ONLY_ACTIONS)
        forbidden = string_list(row.get("forbidden_actions"))
        for item in _DEFAULT_FORBIDDEN:
            if item not in forbidden:
                forbidden.append(item)
        permission = str(row.get("permission_level") or "L1").strip()
        if permission not in {"L0", "L1", "limited_L2"}:
            permission = "L1"
        if permission == "limited_L2" and "propose_patch" not in allowed and "edit_in_mirror" not in allowed:
            permission = "L1"
        candidate = TaskCandidate(
            task_id="",
            source=source,
            observation=str(row.get("observation") or "").strip(),
            gap=str(row.get("gap") or "").strip(),
            proposed_task=str(row.get("proposed_task") or "").strip(),
            expected_value=clamp01(row.get("expected_value"), fallback=0.5),
            goal_alignment=clamp01(row.get("goal_alignment"), fallback=0.5),
            evidence_strength=clamp01(row.get("evidence_strength"), fallback=0.4),
            feasibility=clamp01(row.get("feasibility"), fallback=0.5),
            risk=clamp01(row.get("risk"), fallback=0.3),
            cost=clamp01(row.get("cost"), fallback=0.35),
            reversibility=clamp01(row.get("reversibility"), fallback=0.8),
            distraction_penalty=clamp01(row.get("distraction_penalty"), fallback=0.2),
            success_condition=str(row.get("success_condition") or "").strip(),
            evidence_needed=string_list(row.get("evidence_needed")),
            allowed_actions=allowed,
            forbidden_actions=forbidden,
            source_refs=evidence_refs,
            permission_level=permission,
            metadata={
                "creative_generation": True,
                "creative_schema_version": CREATIVE_TASK_GENERATOR_VERSION,
                "creative_index": index,
                "raw_source": str(row.get("source") or ""),
            },
        )
        if not candidate.observation or not candidate.gap or not candidate.proposed_task or not candidate.success_condition:
            return None, "missing_required_text_fields"
        return candidate, ""

    @staticmethod
    def _known_refs(seeds: Sequence[TaskCandidate]) -> set[str]:
        refs: set[str] = set()
        for item in seeds:
            if item.task_id:
                refs.add(item.task_id)
            refs.update(item.source_refs)
        return refs
