from __future__ import annotations


def cents_to_display(cents: int) -> str:
    dollars = cents / 100
    return f"${dollars:,.2f}"
