from decimal import Decimal

from ledger_core.invoice import invoice_total, is_large_invoice


def test_invoice_total_handles_mixed_amount_formats():
    rows = [
        {"amount": " $1,200 "},
        {"amount": "35.50"},
        {"amount": "$4.50"},
    ]

    assert invoice_total(rows) == Decimal("1240.00")


def test_large_invoice_threshold_accepts_comma_currency():
    rows = [{"amount": "$999.99"}, {"amount": "$0.02"}]

    assert is_large_invoice(rows, threshold="$1,000")
