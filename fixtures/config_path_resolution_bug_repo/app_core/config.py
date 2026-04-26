from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AppConfig:
    name: str
    retries: int


def parse_config(payload: Mapping[str, Any]) -> AppConfig:
    return AppConfig(
        name=str(payload.get("name") or "default"),
        retries=int(payload.get("retries", 0)),
    )
