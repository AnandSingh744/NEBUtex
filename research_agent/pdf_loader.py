"""Extract text from a PDF, page by page, keeping page numbers for citations.

Usage (quick inspection):
    python -m research_agent.pdf_loader data/papers/attention_is_all_you_need.pdf
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Pages with fewer characters than this are probably scanned images or blank.
MIN_CHARS_PER_PAGE = 20


class PDFLoadError(Exception):
    """A PDF that can't be read, with a message that is safe to show the user."""


@dataclass
class Page:
    page_number: int  # 1-based, matching what a reader sees in a PDF viewer
    text: str


@dataclass
class Document:
    source: str  # file name, used in citations
    pages: list[Page]

    @property
    def empty_pages(self) -> list[int]:
        """Page numbers with (almost) no extractable text."""
        return [p.page_number for p in self.pages if len(p.text.strip()) < MIN_CHARS_PER_PAGE]


def load_pdf(path: str | Path) -> Document:
    path = Path(path)
    if not path.exists():
        raise PDFLoadError(f"File not found: {path}")

    try:
        pdf = pymupdf.open(path)
    except Exception as e:  # corrupt file, or not a PDF at all
        # Don't put the library's message in ours: it contains full server paths. It stays attached via `from e`.
        raise PDFLoadError(f"Could not open {path.name} as a PDF. The file may be corrupt.") from e

    with pdf:
        if pdf.needs_pass:
            raise PDFLoadError(f"{path.name} is password-protected.")
        # enumerate from 1 so page numbers match the PDF viewer, not Python's 0-based index.
        pages = [Page(page_number=i, text=page.get_text()) for i, page in enumerate(pdf, start=1)]

    document = Document(source=path.name, pages=pages)
    if len(document.empty_pages) == len(pages):
        raise PDFLoadError(
            f"{path.name} has no extractable text. It is probably a scanned document (needs OCR)."
        )
    return document


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m research_agent.pdf_loader <file.pdf>")

    doc = load_pdf(sys.argv[1])
    total_chars = sum(len(p.text) for p in doc.pages)
    print(f"{doc.source}: {len(doc.pages)} pages, {total_chars:,} characters\n")
    for p in doc.pages:
        print(f"  page {p.page_number:>3}: {len(p.text):>6,} chars")
    if doc.empty_pages:
        print(f"\nWarning: pages with no text (scanned or blank?): {doc.empty_pages}")

    print("\n--- Page 1 (first 1200 characters, exactly as extracted) ---")
    print(doc.pages[0].text[:1200])
