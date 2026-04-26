from ledger_core.currency import normalize_currency


def test_normalize_currency_accepts_simple_values() -> None:
    assert normalize_currency("$12.50") == 1250


def test_normalize_currency_accepts_grouped_values() -> None:
    assert normalize_currency("$1,200.00") == 120000
