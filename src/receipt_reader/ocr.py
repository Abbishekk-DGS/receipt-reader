"""OCR with layout reconstruction.

OCR engines return text boxes in arbitrary order. Receipts are tables: "Masala Dosa ... 120.00"
must stay on one line, or the LLM can't tell which price belongs to which item. We rebuild
visual rows by clustering boxes on their vertical centre, then sort each row left-to-right.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


def group_rows(words: list[Word], tolerance: float = 0.5) -> list[list[Word]]:
    """Cluster words into rows: a word joins a row if its centre is within
    tolerance x median-word-height of the row's centre."""
    if not words:
        return []
    med_h = float(np.median([w.h for w in words])) or 1.0
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: w.cy):
        if rows:
            row_cy = np.mean([x.cy for x in rows[-1]])
            if abs(w.cy - row_cy) <= tolerance * med_h:
                rows[-1].append(w)
                continue
        rows.append([w])
    return [sorted(r, key=lambda w: w.x0) for r in rows]


def render_rows(rows: list[list[Word]], page_width: float | None = None, cols: int = 60) -> str:
    """Render rows as monospaced text, keeping rough horizontal position so columns line up."""
    if not rows:
        return ""
    left = min(w.x0 for r in rows for w in r)  # drop the page margin
    width = (page_width or max(w.x1 for r in rows for w in r)) - left
    lines = []
    for row in rows:
        line = ""
        for w in row:
            target = int((w.x0 - left) / width * cols)
            line += " " * max(1 if line else 0, target - len(line)) + w.text
        lines.append(line.rstrip())
    return "\n".join(lines)


def load_images(path: Path, dpi: int = 200):
    """Yield PIL images for an image file or each page of a PDF."""
    from PIL import Image, ImageOps

    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        for page in pdf:
            yield page.render(scale=dpi / 72).to_pil().convert("RGB")
    else:
        with Image.open(path) as img:
            yield ImageOps.exif_transpose(img).convert("RGB")


class OCR:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR  # PaddleOCR models exported to ONNX

        self.engine = RapidOCR()

    def words(self, image) -> list[Word]:
        result, _ = self.engine(np.asarray(image))
        out = []
        for box, text, score in result or []:
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            text = unicodedata.normalize("NFKC", text)  # full-width ，：→ ASCII
            out.append(Word(text, min(xs), min(ys), max(xs), max(ys), float(score)))
        return out

    def read(self, path: Path) -> tuple[str, float]:
        """Return layout-preserving text for all pages and the mean OCR confidence."""
        pages, scores = [], []
        for image in load_images(path):
            words = self.words(image)
            scores += [w.score for w in words]
            pages.append(render_rows(group_rows(words), page_width=image.width))
        confidence = float(np.mean(scores)) if scores else 0.0
        return "\n\n--- page break ---\n\n".join(pages), confidence
