from ledger_core.invoice import invoice_balance_due, invoice_total


def test_invoice_total_accepts_grouped_currency_amounts() -> None:
    lines = [
        {"description": "annual support", "amount": "$1,200.00"},
        {"description": "setup", "amount": "$50.00"},
    ]

    assert invoice_total(lines) == 125000


def test_invoice_balance_due_applies_credit() -> None:
    lines = [
        {"description": "annual support", "amount": "$1,200.00"},
        {"description": "setup", "amount": "$50.00"},
    ]

    assert invoice_balance_due(lines, credit="$200.00") == 105000
