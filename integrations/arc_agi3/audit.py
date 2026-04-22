from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ARCAGI3RunAudit:
    game_id: str
    total_reward: float
    confirmed_functions: List[str]
    recovery_log: List[Dict[str, Any]]
    governance_log: List[Dict[str, Any]]
    candidate_viability_log: List[Dict[str, Any]]
    planner_runtime_log: List[Dict[str, Any]]
    world_provider_source: str
    raw_audit: Dict[str, Any]


class ARCAGI3AuditWriter:
    @staticmethod
    def build(game_id: str, audit: Dict[str, Any]) -> ARCAGI3RunAudit:
        return ARCAGI3RunAudit(
            game_id=str(game_id),
            total_reward=float(audit.get("total_reward", 0.0) or 0.0),
            confirmed_functions=list(audit.get("confirmed_functions", []) or []),
            recovery_log=list(audit.get("recovery_log", []) or []),
            governance_log=list(audit.get("governance_log", []) or []),
            candidate_viability_log=list(audit.get("candidate_viability_log", []) or []),
            planner_runtime_log=list(audit.get("planner_runtime_log", []) or []),
            world_provider_source=str(audit.get("world_provider_source", "") or ""),
            raw_audit=dict(audit),
        )

    @staticmethod
    def save(path: str | Path, report: ARCAGI3RunAudit) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return output


def summarize_audit(audit: Dict[str, Any]) -> Dict[str, Any]:
    governance_log = list(audit.get("governance_log", []) or [])
    candidate_viability_log = list(audit.get("candidate_viability_log", []) or [])
    recovery_log = list(audit.get("recovery_log", []) or [])
    return {
        "total_reward": float(audit.get("total_reward", 0.0) or 0.0),
        "confirmed_functions": list(audit.get("confirmed_functions", []) or []),
        "governance_entries": len(governance_log),
        "candidate_viability_entries": len(candidate_viability_log),
        "recovery_events": len(recovery_log),
        "world_provider_source": str(audit.get("world_provider_source", "") or ""),
        "arc3_action_coverage": dict(audit.get("arc3_action_coverage", {}) or {}),
    }
