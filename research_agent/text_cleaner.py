"""Clean raw PDF text: remove layout artifacts without destroying meaning or structure.

Every rule here is deliberately narrow. When unsure, leave the text alone:
a little noise is harmless, destroyed content can't be recovered.

Usage (before/after comparison):
    python -m research_agent.text_cleaner data/papers/attention_is_all_you_need.pdf 3
"""

import re
import sys

from research_agent.pdf_loader import Document, Page, load_pdf

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
WORD = re.compile(r"\w+")
HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")
# A single newline followed by a lowercase letter is a line wrap in the middle of a sentence.
WRAPPED_LINE = re.compile(r"(?<!\n)\n(?=[a-z])")


def clean_document(doc: Document) -> Document:
    # Words that appear unbroken anywhere in the document; used to repair hyphenated line breaks.
    vocabulary = {w.lower() for p in doc.pages for w in WORD.findall(p.text)}
    pages = [
        Page(page_number=p.page_number, text=clean_page_text(p.text, p.page_number, vocabulary))
        for p in doc.pages
    ]
    return Document(source=doc.source, pages=pages)


def clean_page_text(text: str, page_number: int, vocabulary: set[str]) -> str:
    for ligature, letters in LIGATURES.items():
        text = text.replace(ligature, letters)
    text = _remove_page_number(text, page_number)
    text = HYPHEN_BREAK.sub(lambda m: _join_hyphenated(m, vocabulary), text)
    text = WRAPPED_LINE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)      # repeated spaces -> one
    text = re.sub(r" ?\n ?", "\n", text)      # spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)   # at most one blank line
    return text.strip()


def _remove_page_number(text: str, page_number: int) -> str:
    """Drop the last line only if it is exactly this page's number."""
    lines = text.rstrip().split("\n")
    if lines and lines[-1].strip() == str(page_number):
        lines.pop()
    return "\n".join(lines)


def _join_hyphenated(match: re.Match, vocabulary: set[str]) -> str:
    first, second = match.group(1), match.group(2)
    if (first + second).lower() in vocabulary:
        return first + second      # word split by line wrap: "transfor-\nmation" -> "transformation"
    return f"{first}-{second}"     # real compound word: "position-\nwise" -> "position-wise"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m research_agent.text_cleaner <file.pdf> <page_number>")

    raw = load_pdf(sys.argv[1])
    clean = clean_document(raw)
    n = int(sys.argv[2])

    raw_chars = sum(len(p.text) for p in raw.pages)
    clean_chars = sum(len(p.text) for p in clean.pages)
    raw_lines = sum(p.text.count("\n") for p in raw.pages)
    clean_lines = sum(p.text.count("\n") for p in clean.pages)
    print(f"{raw.source}: {raw_chars:,} -> {clean_chars:,} chars, {raw_lines:,} -> {clean_lines:,} line breaks\n")

    print(f"--- page {n}, RAW (first 700 chars) ---")
    print(raw.pages[n - 1].text[:700])
    print(f"\n--- page {n}, CLEAN (first 700 chars) ---")
    print(clean.pages[n - 1].text[:700])
