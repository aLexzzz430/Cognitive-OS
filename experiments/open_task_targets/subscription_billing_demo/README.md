# Subscription Billing Demo

Small billing library for prorated subscription invoices.

Rules:

- Subscription windows are inclusive: a customer active from April 1 through April 15 is active for 15 days.
- Monthly subscriptions are prorated by active days divided by the days in that month.
- Credits are applied after prorated line items.
- Completed invoices should be deterministic and rounded to cents.

