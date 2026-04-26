from __future__ import annotations

from pathlib import Path


def resolve_config_path(base_dir: str | Path, requested: str | Path) -> Path:
    requested_path = Path(requested)
    if requested_path.is_absolute():
        return requested_path
    return Path(base_dir) / requested_path.name
