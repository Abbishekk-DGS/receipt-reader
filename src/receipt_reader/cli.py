"""receipt-reader: receipts and invoices -> validated JSON, entirely offline.

receipt-reader sample                         # writes samples/receipt.png to try it on
receipt-reader samples/receipt.png
receipt-reader ~/Documents/bills --csv bills.csv    # a whole folder, plus a spreadsheet summary
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from . import __version__
from .extract import OllamaLLM, extract
from .ocr import OCR
from .sample import make_receipt

SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".pdf", ".tif", ".tiff", ".bmp"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="receipt-reader",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} · by Abbishekk DGS · abbishekkdgs.tech",
    )
    parser.add_argument("inputs", nargs="+", help="files or folders, or the word 'sample'")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--csv", help="write a one-row-per-document summary CSV")
    parser.add_argument("--show-text", action="store_true", help="print the OCR layout text")
    args = parser.parse_args(argv)

    if args.inputs == ["sample"]:
        truth = make_receipt(Path("samples/receipt.png"))
        print(f"wrote samples/receipt.png (ground truth: {json.dumps(truth)})")
        return 0

    files: list[Path] = []
    for item in args.inputs:
        p = Path(item).expanduser()
        files += (
            sorted(f for f in p.rglob("*") if f.suffix.lower() in SUFFIXES) if p.is_dir() else [p]
        )

    ocr, llm = OCR(), OllamaLLM(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        t0 = time.time()
        text, conf = ocr.read(f)
        if args.show_text:
            print(text)
        result = extract(text, llm)
        result.ocr_confidence = round(conf, 3)
        data = result.to_dict()
        (out / f"{f.stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        doc = result.document
        print(
            f"[{result.status:>12}] {f.name}: {doc.vendor if doc else '?'} | "
            f"{doc.date if doc else '?'} | {doc.currency if doc else ''} {doc.total if doc else '?'} "
            f"| {time.time() - t0:.1f}s, {result.attempts} attempt(s)"
        )
        for problem in result.problems:
            print(f"               ! {problem}")
        rows.append(
            {
                "file": f.name,
                "status": result.status,
                "vendor": doc.vendor if doc else "",
                "date": doc.date if doc else "",
                "invoice_number": doc.invoice_number if doc else "",
                "currency": doc.currency if doc else "",
                "subtotal": doc.subtotal if doc else "",
                "tax": doc.tax if doc else "",
                "total": doc.total if doc else "",
            }
        )
    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"summary -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
