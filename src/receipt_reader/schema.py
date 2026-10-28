"""The target document schema, plus business-rule checks the LLM's output must pass."""

from __future__ import annotations

import datetime as dt
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    description: str
    quantity: float = 1
    unit_price: float | None = None
    amount: float = Field(description="line total as printed")


class Document(BaseModel):
    vendor: str = Field(description="business name at the top of the document")
    document_type: Literal["receipt", "invoice"]
    invoice_number: str | None = Field(default=None, description="bill / invoice / receipt number")
    date: dt.date | None = Field(default=None, description="YYYY-MM-DD")
    currency: str = Field(default="INR", description="ISO 4217 code, e.g. INR, USD, EUR")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = Field(default=None, description="total of all taxes (GST, CGST+SGST, VAT)")
    discount: float | None = None
    total: float
    payment_method: str | None = Field(
        default=None, max_length=30, description="one or two words, e.g. cash, card, UPI"
    )

    @field_validator("currency")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper().strip()


def check_arithmetic(doc: Document, tolerance: float = 0.02) -> list[str]:
    """Cross-check the numbers the way an accountant would. Returns human-readable problems.

    Tolerance is relative (2%) with a 1-unit floor, to allow for printed rounding."""
    problems = []

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= max(1.0, tolerance * max(abs(a), abs(b)))

    for i, item in enumerate(doc.line_items, 1):
        if item.unit_price is not None and not close(item.quantity * item.unit_price, item.amount):
            problems.append(
                f"line {i} '{item.description}': quantity {item.quantity} x unit_price "
                f"{item.unit_price} = {item.quantity * item.unit_price:.2f}, but amount is {item.amount}"
            )
    items_sum = sum(i.amount for i in doc.line_items)
    if doc.line_items and doc.subtotal is not None and not close(items_sum, doc.subtotal):
        problems.append(f"line items sum to {items_sum:.2f} but subtotal is {doc.subtotal}")
    if doc.subtotal is not None:
        expected = doc.subtotal + (doc.tax or 0) - (doc.discount or 0)
        if not close(expected, doc.total):
            problems.append(
                f"subtotal {doc.subtotal} + tax {doc.tax or 0} - discount {doc.discount or 0} "
                f"= {expected:.2f}, but total is {doc.total}"
            )
    elif doc.line_items and not close(items_sum + (doc.tax or 0) - (doc.discount or 0), doc.total):
        problems.append(
            f"line items + tax = {items_sum + (doc.tax or 0):.2f} but total is {doc.total}"
        )
    if doc.total <= 0:
        problems.append("total must be positive")
    return problems


def check_completeness(doc: Document, text: str) -> list[str]:
    """Catch omissions that arithmetic can't: empty fields the source text clearly contains.
    Without this, an extraction with no line items would pass every arithmetic check."""
    problems = []
    lower = text.lower()
    if not doc.line_items:
        problems.append("line_items is empty, but every receipt/invoice lists what was bought")
    if doc.subtotal is None and re.search(r"sub\s*-?\s*total", lower):
        problems.append("the document prints a Subtotal, but subtotal is null")
    if doc.invoice_number is None and re.search(
        r"\b(bill|invoice|receipt|inv)\s*(no|number|#)", lower
    ):
        problems.append("the document prints a bill/invoice number, but invoice_number is null")
    if doc.tax is None and re.search(r"\b(gst|cgst|sgst|igst|vat|tax)\b", lower):
        problems.append("the document prints tax lines, but tax is null")
    return problems
