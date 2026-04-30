from __future__ import annotations

from .invoices import Invoice


def summarize_invoices(invoices: list[Invoice]) -> dict[str, int]:
    return {
        "invoice_count": len(invoices),
        "gross_cents": sum(invoice.line_item_cents for invoice in invoices),
        "credit_cents": sum(invoice.credit_cents for invoice in invoices),
        "net_cents": sum(invoice.total_cents for invoice in invoices),
    }

