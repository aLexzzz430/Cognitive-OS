from __future__ import annotations

TAX_BPS_BY_STATE = {
    "standard": 500,
    "reduced": 250,
}


def tax_for_state(amount_cents: int, state: str) -> int:
    return amount_cents * TAX_BPS_BY_STATE[state] // 10_000
