from __future__ import annotations

from typing import Any, Dict, Sequence

from core.reasoning.backend import DeliberationBudget


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


class BudgetRouter:
    """Route workspace state into a bounded deliberation budget."""

    def route(
        self,
        *,
        workspace: Dict[str, Any],
        obs: Dict[str, Any],
        candidate_actions: Sequence[Dict[str, Any]],
        task_family: str = "",
    ) -> DeliberationBudget:
        uncertainty = self._uncertainty(workspace)
        novelty = self._novelty(workspace, candidate_actions)
        execution_risk = _clamp01(workspace.get("world_shift_risk", 0.0))
        transfer_uncertainty = self._transfer_uncertainty(workspace)
        compute_budget = self._compute_budget(workspace)
        self_model_confidence = self._self_model_confidence(workspace)
        task = str(task_family or self._task_family_from_workspace(workspace, obs) or "").strip().lower()

        complexity = (
            novelty * 0.22
            + uncertainty * 0.30
            + execution_risk * 0.20
            + transfer_uncertainty * 0.13
            + (1.0 - compute_budget) * 0.05
            + (1.0 - self_model_confidence) * 0.10
        )
        if task.startswith("arc_agi") or "arc" in task or "answer" in str(workspace.get("deliberation_mode", "")):
            complexity += 0.10

        if compute_budget <= 0.2:
            return DeliberationBudget(
                mode="fast",
                depth=1,
                branch_budget=2,
                verification_budget=0,
                hypothesis_limit=3,
                test_limit=2,
                program_limit=2,
                output_limit=2,
            )
        if complexity >= 0.72:
            return DeliberationBudget(
                mode="slow",
                depth=4,
                branch_budget=5,
                verification_budget=2,
                hypothesis_limit=6,
                test_limit=5,
                program_limit=10,
                output_limit=8,
            )
        if complexity >= 0.42:
            return DeliberationBudget(
                mode="medium",
                depth=2,
                branch_budget=3,
                verification_budget=1,
                hypothesis_limit=5,
                test_limit=4,
                program_limit=6,
                output_limit=4,
            )
        return DeliberationBudget(
            mode="fast",
            depth=1,
            branch_budget=2,
            verification_budget=0,
            hypothesis_limit=3,
            test_limit=2,
            program_limit=3,
            output_limit=2,
        )

    def _uncertainty(self, workspace: Dict[str, Any]) -> float:
        vector = workspace.get("uncertainty_vector", {})
        if isinstance(vector, dict) and "overall" in vector:
            return _clamp01(vector.get("overall", 0.0))
        return max(
            _clamp01(workspace.get("world_shift_risk", 0.0)),
            _clamp01(workspace.get("retrieval_pressure", 0.0)),
            _clamp01(workspace.get("probe_pressure", 0.0)),
        )

    def _novelty(self, workspace: Dict[str, Any], candidate_actions: Sequence[Dict[str, Any]]) -> float:
        surfaced = workspace.get("surfaced_representations", [])
        skills = workspace.get("active_skills", [])
        evidence_queue = workspace.get("evidence_queue", [])
        if not isinstance(surfaced, list):
            surfaced = []
        if not isinstance(skills, list):
            skills = []
        if not isinstance(evidence_queue, list):
            evidence_queue = []
        trace_factor = 1.0 if not evidence_queue else max(0.0, 1.0 - min(1.0, len(evidence_queue) / 6.0))
        action_factor = 1.0 if len(candidate_actions) <= 1 else max(0.0, 1.0 - min(1.0, len(candidate_actions) / 6.0))
        structure_factor = 1.0 - min(1.0, len(surfaced) / 4.0)
        skill_factor = 1.0 - min(1.0, len(skills) / 4.0)
        return max(0.0, min(1.0, (trace_factor + action_factor + structure_factor + skill_factor) / 4.0))

    def _transfer_uncertainty(self, workspace: Dict[str, Any]) -> float:
        transfers = workspace.get("transfer_candidates", [])
        if not isinstance(transfers, list) or not transfers:
            return 0.5
        confidences = []
        for row in transfers:
            if isinstance(row, dict):
                confidences.append(_clamp01(row.get("confidence", row.get("score", 0.5)), default=0.5))
        if not confidences:
            return 0.5
        return max(0.0, min(1.0, 1.0 - (sum(confidences) / len(confidences))))

    def _compute_budget(self, workspace: Dict[str, Any]) -> float:
        compute = workspace.get("compute_budget", {})
        base_budget = 1.0
        if isinstance(compute, dict):
            base_budget = _clamp01(compute.get("compute_budget", 1.0), default=1.0)
        self_model = workspace.get("self_model_summary", {})
        if isinstance(self_model, dict):
            capability_envelope = self_model.get("capability_envelope", {})
            capability_envelope = capability_envelope if isinstance(capability_envelope, dict) else {}
            budget_multiplier = _clamp01(
                self_model.get("budget_multiplier", capability_envelope.get("budget_multiplier", 1.0)),
                default=1.0,
            )
            base_budget = _clamp01(base_budget * budget_multiplier, default=base_budget)
            teacher_off_escalation = bool(
                self_model.get(
                    "teacher_off_escalation",
                    self_model.get("planner_control_profile", {}).get("teacher_off_escalation", False)
                    if isinstance(self_model.get("planner_control_profile", {}), dict) else False,
                )
            )
            if teacher_off_escalation:
                base_budget = _clamp01(base_budget * 0.92, default=base_budget)
        return base_budget

    def _self_model_confidence(self, workspace: Dict[str, Any]) -> float:
        self_model = workspace.get("self_model_summary", {})
        if isinstance(self_model, dict):
            return _clamp01(self_model.get("global_reliability", 0.5), default=0.5)
        return 0.5

    def _task_family_from_workspace(self, workspace: Dict[str, Any], obs: Dict[str, Any]) -> str:
        task_frame = workspace.get("task_frame_summary", {})
        if isinstance(task_frame, dict):
            family = str(task_frame.get("task_family", "") or "").strip()
            if family:
                return family
        if "arc_task" in obs:
            return "arc"
        return ""
