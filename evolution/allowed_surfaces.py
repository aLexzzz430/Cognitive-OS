from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ONLINE_BLOCKED_FILE_TARGETS: Tuple[str, ...] = (
    "core/main_loop.py",
    "modules/state/manager.py",
    "modules/state/schema.py",
    "modules/governance/object_store.py",
    "modules/governance/gate.py",
)
ONLINE_BLOCKED_FILE_PREFIXES: Tuple[str, ...] = (
    ".github/workflows/",
)
ONLINE_WEAK_SURFACE_PREFIXES: Tuple[str, ...] = (
    "runtime/proposals/",
    "runtime/hypotheses/",
    "runtime/skill_priors/",
    "runtime/representation_weights/",
    "runtime/agenda_priors/",
    "runtime/recovery_shortcuts/",
)

FORBIDDEN_REASONING_KEYS = {
    "apply_patch",
    "bypass_router_state",
    "commit_patch",
    "direct_durable_write",
    "direct_write",
    "durable_write",
    "formal_write",
    "hidden_controller",
    "main_loop_override",
    "selected_action_override",
    "shell_command",
    "state_patch",
    "target_files",
    "update_state",
    "write_patch",
}
FORBIDDEN_REASONING_ACTION_KINDS = {
    "apply_patch",
    "commit_patch",
    "direct_write",
    "durable_write",
    "self_modify",
    "shell_command",
}


@dataclass(frozen=True)
class SurfaceCheckVerdict:
    accepted: bool
    online: bool
    weakly_allowed_targets: List[str] = field(default_factory=list)
    offline_only_targets: List[str] = field(default_factory=list)
    blocked_targets: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def requires_human_review(self) -> bool:
        return bool(self.offline_only_targets) and not self.online

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "online": bool(self.online),
            "weakly_allowed_targets": list(self.weakly_allowed_targets),
            "offline_only_targets": list(self.offline_only_targets),
            "blocked_targets": list(self.blocked_targets),
            "reasons": list(self.reasons),
            "requires_human_review": bool(self.requires_human_review),
        }


def _normalize_repo_path(path: str) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    if not raw:
        return ""
    while raw.startswith("./"):
        raw = raw[2:]
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if not parts:
        return ""
    for index in range(len(parts)):
        candidate = "/".join(parts[index:])
        if candidate in ONLINE_BLOCKED_FILE_TARGETS:
            return candidate
        if any(candidate.startswith(prefix) for prefix in ONLINE_BLOCKED_FILE_PREFIXES + ONLINE_WEAK_SURFACE_PREFIXES):
            return candidate
    return "/".join(parts)


def validate_patch_targets(target_paths: Sequence[str], *, online: bool) -> SurfaceCheckVerdict:
    weakly_allowed_targets: List[str] = []
    offline_only_targets: List[str] = []
    blocked_targets: List[str] = []
    reasons: List[str] = []

    normalized_targets: List[str] = []
    for raw_target in list(target_paths or []):
        normalized = _normalize_repo_path(str(raw_target or ""))
        if not normalized:
            continue
        if normalized in normalized_targets:
            continue
        normalized_targets.append(normalized)

    for target in normalized_targets:
        is_blocked_online = target in ONLINE_BLOCKED_FILE_TARGETS or any(
            target.startswith(prefix) for prefix in ONLINE_BLOCKED_FILE_PREFIXES
        )
        is_weak_surface = any(target.startswith(prefix) for prefix in ONLINE_WEAK_SURFACE_PREFIXES)

        if is_weak_surface:
            weakly_allowed_targets.append(target)
            continue

        if online and is_blocked_online:
            blocked_targets.append(target)
            reasons.append(f"online_blocked:{target}")
            continue

        offline_only_targets.append(target)
        reasons.append(f"offline_only:{target}")

    if online:
        accepted = bool(normalized_targets) and not blocked_targets and not offline_only_targets
    else:
        accepted = bool(normalized_targets)

    if not normalized_targets:
        reasons.append("missing_targets")

    return SurfaceCheckVerdict(
        accepted=accepted,
        online=bool(online),
        weakly_allowed_targets=weakly_allowed_targets,
        offline_only_targets=offline_only_targets,
        blocked_targets=blocked_targets,
        reasons=reasons,
    )


def _sanitize_reasoning_node(node: Any, prefix: str, violations: List[str]) -> Any:
    if isinstance(node, dict):
        action_kind = str(node.get("kind", "") or "").strip().lower()
        if action_kind in FORBIDDEN_REASONING_ACTION_KINDS:
            violations.append(prefix or "kind")
            return None

        clean: Dict[str, Any] = {}
        for key, value in node.items():
            normalized_key = str(key or "")
            path = f"{prefix}.{normalized_key}" if prefix else normalized_key
            if normalized_key in FORBIDDEN_REASONING_KEYS:
                violations.append(path)
                continue
            if normalized_key == "kind" and str(value or "").strip().lower() in FORBIDDEN_REASONING_ACTION_KINDS:
                violations.append(path)
                return None
            child = _sanitize_reasoning_node(value, path, violations)
            if child is None and isinstance(value, (dict, list)):
                continue
            clean[normalized_key] = child
        return clean

    if isinstance(node, list):
        items: List[Any] = []
        for index, value in enumerate(node):
            child = _sanitize_reasoning_node(value, f"{prefix}[{index}]" if prefix else f"[{index}]", violations)
            if child is None:
                continue
            items.append(child)
        return items

    return node


def collect_reasoning_controller_violations(payload: Any) -> List[str]:
    violations: List[str] = []
    _sanitize_reasoning_node(payload, "", violations)
    return violations


def sanitize_reasoning_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    violations: List[str] = []
    sanitized = _sanitize_reasoning_node(payload if isinstance(payload, dict) else {}, "", violations)
    return (sanitized if isinstance(sanitized, dict) else {}), violations
