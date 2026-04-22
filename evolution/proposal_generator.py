from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from evolution.allowed_surfaces import SurfaceCheckVerdict, validate_patch_targets


@dataclass(frozen=True)
class PatchProposal:
    proposal_id: str
    summary: str
    rationale: str
    target_files: List[str]
    mode: str = "offline"
    generated_by: str = "system"
    created_at: float = field(default_factory=time.time)
    patch_text: str = ""
    file_overrides: Dict[str, str] = field(default_factory=dict)
    commands: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    surface_verdict: Optional[SurfaceCheckVerdict] = None
    status: str = "ready_for_sandbox"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "summary": self.summary,
            "rationale": self.rationale,
            "target_files": list(self.target_files),
            "mode": self.mode,
            "generated_by": self.generated_by,
            "created_at": float(self.created_at),
            "patch_text": self.patch_text,
            "file_overrides": dict(self.file_overrides),
            "commands": list(self.commands),
            "metadata": dict(self.metadata),
            "surface_verdict": self.surface_verdict.to_dict() if self.surface_verdict else {},
            "status": self.status,
        }


class ProposalGenerator:
    def generate_patch_proposal(
        self,
        *,
        summary: str,
        rationale: str,
        target_files: Sequence[str],
        mode: str = "offline",
        generated_by: str = "system",
        patch_text: str = "",
        file_overrides: Optional[Dict[str, str]] = None,
        commands: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PatchProposal:
        override_map = {
            str(path): str(content)
            for path, content in dict(file_overrides or {}).items()
            if str(path or "").strip()
        }
        normalized_targets: List[str] = []
        for target in list(target_files or []) + list(override_map.keys()):
            target_s = str(target or "").strip()
            if not target_s or target_s in normalized_targets:
                continue
            normalized_targets.append(target_s)

        online = str(mode or "offline").strip().lower() == "online"
        surface_verdict = validate_patch_targets(normalized_targets, online=online)
        status = "ready_for_sandbox" if surface_verdict.accepted else "blocked"

        return PatchProposal(
            proposal_id=f"patch_{uuid.uuid4().hex[:12]}",
            summary=str(summary or "").strip(),
            rationale=str(rationale or "").strip(),
            target_files=normalized_targets,
            mode="online" if online else "offline",
            generated_by=str(generated_by or "system"),
            patch_text=str(patch_text or ""),
            file_overrides=override_map,
            commands=[str(command) for command in list(commands or []) if str(command or "").strip()],
            metadata=dict(metadata or {}),
            surface_verdict=surface_verdict,
            status=status,
        )
