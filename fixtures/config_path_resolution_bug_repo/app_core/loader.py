from __future__ import annotations

import json
from pathlib import Path

from .config import AppConfig, parse_config
from .paths import resolve_config_path


def load_config(base_dir: str | Path, requested: str | Path) -> AppConfig:
    path = resolve_config_path(base_dir, requested)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_config(payload)
