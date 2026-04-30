from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str


@dataclass(frozen=True)
class Subscription:
    plan_code: str
    monthly_price_cents: int
    starts_on: date
    ends_on: date
    active: bool = True

