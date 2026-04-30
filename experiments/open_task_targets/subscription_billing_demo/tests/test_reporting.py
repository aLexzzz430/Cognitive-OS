from billing.invoices import Invoice
from billing.reporting import summarize_invoices


def test_summarize_invoices() -> None:
    summary = summarize_invoices(
        [
            Invoice("cus_001", line_item_cents=1500, credit_cents=250, total_cents=1250),
            Invoice("cus_002", line_item_cents=3000, credit_cents=0, total_cents=3000),
        ]
    )

    assert summary == {
        "invoice_count": 2,
        "gross_cents": 4500,
        "credit_cents": 250,
        "net_cents": 4250,
    }

