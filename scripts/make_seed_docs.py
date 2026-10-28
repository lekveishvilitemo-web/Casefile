"""Generate the PDF and DOCX seed documents from the same facts as the text seeds."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import docx
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed"

TIMELINE = """SRE timeline INC-1842
2024-03-02 22:14 UTC: ledger-service v2.4.1 deployed.
2024-03-04 06:12 UTC: first checkout-api latency alert.
2024-03-06 11:40 UTC: rollback to ledger-service v2.3.9 completed.
Services harmed were checkout-api and settlement-worker. merchant-notifier was degraded.
This timeline does not state a revenue loss.
"""

MEMO = """Vendor status memo
Dana Ruiz, 2024-03-07.
The March payment delay was caused by an OrbitalNet regional outage.
The ledger-service was not modified in March and the internal deploy is unrelated.
Checkout traffic should be treated as an external card-network problem.
"""


def write_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    path.write_bytes(bytes(pdf.output()))


def write_docx(path: Path, text: str) -> None:
    document = docx.Document()
    for paragraph in text.splitlines():
        if paragraph.strip():
            document.add_paragraph(paragraph.strip())
    buffer = BytesIO()
    document.save(buffer)
    path.write_bytes(buffer.getvalue())


def main() -> None:
    SEED.mkdir(parents=True, exist_ok=True)
    write_pdf(SEED / "sre-timeline-INC-1842.pdf", TIMELINE)
    write_docx(SEED / "status-memo.docx", MEMO)
    print(f"Wrote {SEED / 'sre-timeline-INC-1842.pdf'}")
    print(f"Wrote {SEED / 'status-memo.docx'}")


if __name__ == "__main__":
    main()
