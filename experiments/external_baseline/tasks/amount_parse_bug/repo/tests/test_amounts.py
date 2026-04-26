from decimal import Decimal

import pytest

from ledger_core.amounts import parse_amount


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", Decimal("42")),
        (" $19.99 ", Decimal("19.99")),
        ("$1,200", Decimal("1200")),
    ],
)
def test_parse_amount_accepts_common_currency_inputs(raw, expected):
    assert parse_amount(raw) == expected


def test_parse_amount_rejects_blank_input():
    with pytest.raises(ValueError):
        parse_amount("   ")
