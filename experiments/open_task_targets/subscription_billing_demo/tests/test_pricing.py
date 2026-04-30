from billing.pricing import prorate_cents


def test_prorate_half_month() -> None:
    assert prorate_cents(3000, 15, 30) == 1500


def test_prorate_rounds_to_nearest_cent() -> None:
    assert prorate_cents(1000, 1, 3) == 333

