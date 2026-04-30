from datetime import date

from billing.invoices import build_invoice
from billing.models import Customer, Subscription


def test_invoice_prorates_inclusive_subscription_window() -> None:
    customer = Customer("cus_001", "Acme")
    subscription = Subscription(
        plan_code="team",
        monthly_price_cents=3000,
        starts_on=date(2026, 4, 1),
        ends_on=date(2026, 4, 15),
    )

    invoice = build_invoice(customer, subscription, invoice_month=date(2026, 4, 1), credit_cents=250)

    assert invoice.line_item_cents == 1500
    assert invoice.total_cents == 1250


def test_inactive_subscription_only_applies_credit_floor() -> None:
    customer = Customer("cus_002", "Beta")
    subscription = Subscription(
        plan_code="team",
        monthly_price_cents=3000,
        starts_on=date(2026, 4, 1),
        ends_on=date(2026, 4, 30),
        active=False,
    )

    invoice = build_invoice(customer, subscription, invoice_month=date(2026, 4, 1), credit_cents=500)

    assert invoice.line_item_cents == 0
    assert invoice.total_cents == 0

