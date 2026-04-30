from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .models import Customer, Subscription
from .periods import active_days, days_in_month, same_month
from .pricing import prorate_cents


@dataclass(frozen=True)
class Invoice:
    customer_id: str
    line_item_cents: int
    credit_cents: int
    total_cents: int


def build_invoice(
    customer: Customer,
    subscription: Subscription,
    *,
    invoice_month: date,
    credit_cents: int = 0,
) -> Invoice:
    if not subscription.active:
        return Invoice(customer.customer_id, 0, credit_cents, max(0, -credit_cents))
    if not same_month(subscription.starts_on, subscription.ends_on):
        raise ValueError("demo invoices only support one calendar month")

    billable_days = active_days(subscription.starts_on, subscription.ends_on)
    period_days = days_in_month(invoice_month)
    line_item_cents = prorate_cents(subscription.monthly_price_cents, billable_days, period_days)
    return Invoice(
        customer_id=customer.customer_id,
        line_item_cents=line_item_cents,
        credit_cents=credit_cents,
        total_cents=max(0, line_item_cents - credit_cents),
    )

