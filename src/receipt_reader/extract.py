"""LLM extraction with schema-constrained decoding, validation, and a repair loop."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from pydantic import ValidationError

from .schema import Document, check_arithmetic, check_completeness

_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = _host if _host.startswith("http") else f"http://{_host}"

PROMPT = """Extract the document below into JSON.
- Copy numbers exactly as printed; do not compute values that are not on the document.
- line_items: only purchased items or services, not subtotal/tax/total rows.
- tax: the sum of every tax line (e.g. CGST + SGST). Use null if no tax is printed.
- date as YYYY-MM-DD. currency as an ISO code (₹ or Rs means INR, $ means USD).

The text was produced by OCR and keeps the original row layout:
<<<
{text}
>>>"""

REPAIR = """Your JSON has problems:
{problems}
Re-read the document text and return corrected JSON. Fix only what is wrong."""


class LLM(Protocol):
    def chat(self, messages: list[dict], schema: dict) -> str: ...


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:3b", url: str = OLLAMA_URL):
        self.model, self.url = model, url

    def chat(self, messages: list[dict], schema: dict) -> str:
        resp = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


@dataclass
class Extraction:
    document: Document | None
    problems: list[str] = field(default_factory=list)
    attempts: int = 0
    ocr_confidence: float | None = None

    @property
    def status(self) -> str:
        if self.document is None:
            return "failed"
        return "verified" if not self.problems else "needs_review"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "problems": self.problems,
            "attempts": self.attempts,
            "ocr_confidence": self.ocr_confidence,
            "document": self.document.model_dump(mode="json") if self.document else None,
        }


def decoding_schema() -> dict:
    """The schema handed to the decoder. Every field is *required* (nullable ones may still be
    null) and line_items needs at least one entry: small models skip any field the grammar
    lets them skip, so we don't let them."""
    schema = Document.model_json_schema()
    schema["required"] = list(schema["properties"])
    schema["properties"]["line_items"]["minItems"] = 1
    item = schema["$defs"]["LineItem"]
    item["required"] = list(item["properties"])
    return schema


def extract(text: str, llm: LLM, max_repairs: int = 2) -> Extraction:
    schema = decoding_schema()
    messages = [{"role": "user", "content": PROMPT.format(text=text)}]
    result = Extraction(document=None)
    for attempt in range(1, max_repairs + 2):
        result.attempts = attempt
        raw = llm.chat(messages, schema)
        try:
            doc = Document.model_validate(json.loads(raw))
            problems = check_completeness(doc, text) + check_arithmetic(doc)
        except (json.JSONDecodeError, ValidationError) as e:
            doc, problems = None, [f"invalid JSON for the schema: {e}"]
        if doc is not None:
            result.document = doc  # keep the best parse even if checks still fail
        result.problems = problems
        if not problems:
            break
        messages += [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": REPAIR.format(problems="\n".join(f"- {p}" for p in problems)),
            },
        ]
    return result
