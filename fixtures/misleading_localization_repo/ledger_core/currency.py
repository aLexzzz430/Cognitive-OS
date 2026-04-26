from __future__ import annotations

from decimal import Decimal


def normalize_currency(value: str) -> int:
    cleaned = str(value).strip().replace("$", "")
    return int(Decimal(cleaned) * 100)
