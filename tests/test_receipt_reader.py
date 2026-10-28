import json

import pytest

from receipt_reader.extract import extract
from receipt_reader.ocr import OCR, Word, group_rows, render_rows
from receipt_reader.sample import make_receipt
from receipt_reader.schema import Document, LineItem, check_arithmetic, check_completeness


def doc(**overrides) -> Document:
    base = dict(
        vendor="Cafe",
        document_type="receipt",
        line_items=[
            LineItem(description="Dosa", quantity=2, unit_price=90, amount=180),
            LineItem(description="Coffee", quantity=3, unit_price=40, amount=120),
        ],
        subtotal=300,
        tax=15,
        total=315,
    )
    base.update(overrides)
    return Document(**base)


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.messages = list(replies), []

    def chat(self, messages, schema):
        self.messages = messages
        assert schema["title"] == "Document"
        return self.replies.pop(0)


def test_rows_rebuilt_from_scattered_boxes():
    words = [
        Word("120.00", 400, 52, 460, 70, 0.99),
        Word("TOTAL", 10, 100, 80, 118, 0.99),
        Word("Masala Dosa", 10, 50, 140, 68, 0.98),
        Word("315.00", 400, 101, 460, 119, 0.97),
    ]
    rows = group_rows(words)
    assert [[w.text for w in r] for r in rows] == [["Masala Dosa", "120.00"], ["TOTAL", "315.00"]]
    text = render_rows(rows, page_width=480)
    assert text.splitlines()[0].startswith("Masala Dosa") and text.splitlines()[0].endswith(
        "120.00"
    )


def test_arithmetic_checks():
    assert check_arithmetic(doc()) == []
    problems = check_arithmetic(doc(total=351))
    assert len(problems) == 1 and "total is 351" in problems[0]
    bad_line = doc(line_items=[LineItem(description="Dosa", quantity=2, unit_price=90, amount=90)])
    assert any("quantity 2" in p for p in check_arithmetic(bad_line))


def test_repair_loop_fixes_transposed_digits():
    wrong = doc(total=351).model_dump(mode="json")
    right = doc().model_dump(mode="json")
    llm = ScriptedLLM(json.dumps(wrong), json.dumps(right))
    result = extract("receipt text", llm)
    assert result.status == "verified" and result.attempts == 2
    assert "total is 351" in llm.messages[-1]["content"]  # the problem was explained to the model


def test_unfixable_document_is_flagged_not_hidden():
    wrong = json.dumps(doc(total=999).model_dump(mode="json"))
    result = extract("text", ScriptedLLM(wrong, wrong, wrong))
    assert result.status == "needs_review" and result.document.total == 999


def test_invalid_json_then_valid():
    result = extract("text", ScriptedLLM("{not json", json.dumps(doc().model_dump(mode="json"))))
    assert result.status == "verified"


def test_sample_receipt_ground_truth(tmp_path):
    truth = make_receipt(tmp_path / "r.png")
    assert truth["total"] == round(truth["subtotal"] + truth["tax"], 2)
    assert (tmp_path / "r.png").stat().st_size > 10_000


def test_ocr_reads_sample_receipt(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    truth = make_receipt(tmp_path / "r.png", noisy=False)
    text, conf = OCR().read(tmp_path / "r.png")
    assert "SARAVANA" in text and "ST-20481" in text and conf > 0.8
    total_line = next(line for line in text.splitlines() if "TOTAL" in line)
    assert f"{truth['total']:.2f}" in total_line  # label and amount rebuilt onto one row


def test_empty_extraction_is_not_verified():
    """Regression: an extraction with no items/subtotal passed every arithmetic check."""
    text = "Bill No: ST-1\nDosa 2 90.00 180.00\nSubtotal 180.00\nCGST 4.50\nTOTAL 184.50"
    lazy = Document(vendor="X", document_type="receipt", total=184.5)
    problems = check_completeness(lazy, text)
    assert len(problems) == 4
    assert check_completeness(doc(invoice_number="ST-1"), text) == []


def test_decoding_schema_forbids_skipping_fields():
    from receipt_reader.extract import decoding_schema

    schema = decoding_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["line_items"]["minItems"] == 1
