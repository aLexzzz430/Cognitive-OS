from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class WebArenaRunAudit:
    task_id: str
    instruction: str
    total_reward: float
    confirmed_functions: List[str]
    recovery_log: List[Dict[str, Any]]
    governance_log: List[Dict[str, Any]]
    candidate_viability_log: List[Dict[str, Any]]
    planner_runtime_log: List[Dict[str, Any]]
    world_provider_source: str
    final_url: str
    raw_audit: Dict[str, Any]


class WebArenaAuditWriter:
    @staticmethod
    def build(task_id: str, audit: Dict[str, Any]) -> WebArenaRunAudit:
        surface_structured = dict(audit.get("final_surface_structured", {}) or {})
        return WebArenaRunAudit(
            task_id=str(task_id or surface_structured.get("task_id", "") or "webarena_task"),
            instruction=str(surface_structured.get("instruction", "") or ""),
            total_reward=float(audit.get("total_reward", 0.0) or 0.0),
            confirmed_functions=list(audit.get("confirmed_functions", []) or []),
            recovery_log=list(audit.get("recovery_log", []) or []),
            governance_log=list(audit.get("governance_log", []) or []),
            candidate_viability_log=list(audit.get("candidate_viability_log", []) or []),
            planner_runtime_log=list(audit.get("planner_runtime_log", []) or []),
            world_provider_source=str(audit.get("world_provider_source", "") or ""),
            final_url=str(surface_structured.get("url", "") or ""),
            raw_audit=dict(audit),
        )

    @staticmethod
    def save(path: str | Path, report: WebArenaRunAudit) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return output


def summarize_audit(audit: Dict[str, Any]) -> Dict[str, Any]:
    governance_log = list(audit.get("governance_log", []) or [])
    candidate_viability_log = list(audit.get("candidate_viability_log", []) or [])
    recovery_log = list(audit.get("recovery_log", []) or [])
    final_surface = dict(audit.get("final_surface_structured", {}) or {})
    return {
        "task_id": str(final_surface.get("task_id", "") or "webarena_task"),
        "instruction": str(final_surface.get("instruction", "") or ""),
        "final_url": str(final_surface.get("url", "") or ""),
        "total_reward": float(audit.get("total_reward", 0.0) or 0.0),
        "confirmed_functions": list(audit.get("confirmed_functions", []) or []),
        "governance_entries": len(governance_log),
        "candidate_viability_entries": len(candidate_viability_log),
        "recovery_events": len(recovery_log),
        "world_provider_source": str(audit.get("world_provider_source", "") or ""),
    }
