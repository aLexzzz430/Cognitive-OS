from __future__ import annotations

from decimal import Decimal, InvalidOperation


def parse_amount(raw: str | int | float | Decimal) -> Decimal:
    """Parse a user-entered currency amount into a Decimal."""
    if isinstance(raw, Decimal):
        return raw
    if raw is None:
        raise ValueError("amount is required")
    text = str(raw)
    cleaned = text.strip().replace("$", "").replace(",", "")
    if not cleaned:
        raise ValueError("amount is empty")
    try:
        return Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"invalid amount: {raw!r}") from exc


def format_amount(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.01'))}"
