from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, (tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    out: List[str] = []
    seen = set()
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


@dataclass
class DurableIdentitySnapshot:
    agent_id: str = "unknown"
    identity_markers: Dict[str, Any] = field(default_factory=dict)
    traits: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    active_commitments: List[str] = field(default_factory=list)
    autobiographical_threads: List[str] = field(default_factory=list)
    continuity_confidence: float = 0.5
    narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "identity_markers": dict(self.identity_markers or {}),
            "traits": list(self.traits),
            "values": list(self.values),
            "capabilities": list(self.capabilities),
            "limitations": list(self.limitations),
            "active_commitments": list(self.active_commitments),
            "autobiographical_threads": list(self.autobiographical_threads),
            "continuity_confidence": max(0.0, min(1.0, float(self.continuity_confidence))),
            "narrative": str(self.narrative or ""),
        }


class DurableIdentityLedger:
    """Materialize a stable identity snapshot from continuity + self-model signals."""

    def build(
        self,
        *,
        identity_markers: Dict[str, Any],
        continuity_snapshot: Dict[str, Any],
        active_commitments: List[Dict[str, Any]],
        autobiographical_summary: Dict[str, Any],
        continuity_confidence: float,
    ) -> DurableIdentitySnapshot:
        continuity = continuity_snapshot if isinstance(continuity_snapshot, dict) else {}
        marker_dict = dict(identity_markers or {})
        identity_fields = continuity.get("identity_fields", {})
        identity_fields = identity_fields if isinstance(identity_fields, dict) else {}
        durable_identity = continuity.get("durable_identity", {})
        durable_identity = durable_identity if isinstance(durable_identity, dict) else {}

        traits = _normalize_list(
            durable_identity.get("traits", [])
            or identity_fields.get("traits", [])
            or marker_dict.get("traits", [])
        )
        values = _normalize_list(
            durable_identity.get("values", [])
            or identity_fields.get("values", [])
            or marker_dict.get("values", [])
        )
        capabilities = _normalize_list(
            durable_identity.get("capabilities", [])
            or identity_fields.get("capabilities", [])
            or marker_dict.get("capabilities", [])
        )
        limitations = _normalize_list(
            durable_identity.get("limitations", [])
            or identity_fields.get("limitations", [])
            or marker_dict.get("limitations", [])
        )

        commitment_strings = _normalize_list(
            [
                item.get("commitment", "")
                for item in active_commitments
                if isinstance(item, dict)
            ]
        )
        autobiographical_threads = _normalize_list(
            list(autobiographical_summary.get("identity_implications", []) if isinstance(autobiographical_summary, dict) else [])
            + [autobiographical_summary.get("summary", "")] if isinstance(autobiographical_summary, dict) else []
        )

        agent_id = str(
            durable_identity.get("agent_id")
            or identity_fields.get("name")
            or marker_dict.get("agent_id")
            or "unknown"
        ).strip() or "unknown"

        narrative_parts = []
        if commitment_strings:
            narrative_parts.append(f"commitments={', '.join(commitment_strings[:2])}")
        if autobiographical_threads:
            narrative_parts.append(f"memory={autobiographical_threads[0]}")
        if traits:
            narrative_parts.append(f"traits={', '.join(traits[:2])}")

        snapshot = DurableIdentitySnapshot(
            agent_id=agent_id,
            identity_markers=marker_dict,
            traits=traits,
            values=values,
            capabilities=capabilities,
            limitations=limitations,
            active_commitments=commitment_strings,
            autobiographical_threads=autobiographical_threads,
            continuity_confidence=max(0.0, min(1.0, float(continuity_confidence))),
            narrative="; ".join(part for part in narrative_parts if part),
        )
        return snapshot
