"""Render a realistic synthetic receipt, so demos and tests never need anyone's real receipts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ITEMS = [("Masala Dosa", 2, 90.00), ("Filter Coffee", 3, 40.00), ("Medu Vada", 1, 60.00),
         ("Paneer Butter Masala", 1, 240.00), ("Butter Naan", 4, 45.00)]  # fmt: skip


def _font(size: int) -> ImageFont.ImageFont:
    for name in (
        "Menlo.ttc",
        "DejaVuSansMono.ttf",
        "Courier New.ttf",
        "LiberationMono-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def make_receipt(path: Path, noisy: bool = True) -> dict:
    """Draw the receipt and return its ground truth."""
    subtotal = sum(q * p for _, q, p in ITEMS)
    cgst = round(subtotal * 0.025, 2)
    sgst = cgst
    total = round(subtotal + cgst + sgst, 2)
    lines = [
        ("SARAVANA TIFFIN CENTRE", "c"),
        ("12, Anna Salai, Chennai 600002", "c"),
        ("GSTIN 33ABCDE1234F1Z5", "c"),
        ("", ""),
        ("Bill No: ST-20481        Date: 14/09/2026", "l"),
        ("-" * 44, "l"),
        ("Item                 Qty   Rate    Amount", "l"),
        ("-" * 44, "l"),
    ]
    for name, qty, price in ITEMS:
        lines.append((f"{name:<20} {qty:>3} {price:>7.2f} {qty * price:>9.2f}", "l"))
    lines += [
        ("-" * 44, "l"),
        (f"{'Subtotal':<32}{subtotal:>12.2f}", "l"),
        (f"{'CGST 2.5%':<32}{cgst:>12.2f}", "l"),
        (f"{'SGST 2.5%':<32}{sgst:>12.2f}", "l"),
        (f"{'TOTAL':<32}{total:>12.2f}", "l"),
        ("", ""),
        ("Paid by UPI", "c"),
        ("Thank you! Visit again", "c"),
    ]
    font = _font(22)
    width, line_h = 640, 34
    img = Image.new("RGB", (width, line_h * len(lines) + 80), (250, 248, 240))
    draw = ImageDraw.Draw(img)
    for i, (text, align) in enumerate(lines):
        w = draw.textlength(text, font=font)
        x = (width - w) / 2 if align == "c" else 30
        draw.text((x, 40 + i * line_h), text, fill=(30, 30, 30), font=font)
    if noisy:  # a slightly blurred, rotated phone photo
        img = img.rotate(1.2, expand=True, fillcolor=(200, 200, 200)).filter(
            ImageFilter.GaussianBlur(0.6)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return {
        "vendor": "SARAVANA TIFFIN CENTRE",
        "invoice_number": "ST-20481",
        "date": "2026-09-14",
        "subtotal": subtotal,
        "tax": round(cgst + sgst, 2),
        "total": total,
        "n_items": len(ITEMS),
    }
