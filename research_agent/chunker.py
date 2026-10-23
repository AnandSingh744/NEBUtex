"""Split cleaned documents into overlapping chunks, each tagged with its page number.

Strategy: split each page into sentences, then pack whole sentences into chunks
of up to CHUNK_SIZE characters. Each new chunk starts with the last ~CHUNK_OVERLAP
characters of the previous one. Chunks never cross pages, so each chunk has
exactly one page number for citations.

Usage (inspection):
    python -m research_agent.chunker data/papers/attention_is_all_you_need.pdf
"""

import re
import sys
from dataclasses import dataclass

from research_agent.pdf_loader import Document, load_pdf
from research_agent.text_cleaner import clean_document

CHUNK_SIZE = 1000     # characters (~250 tokens); fits small embedding models
CHUNK_OVERLAP = 200   # characters repeated between neighbouring chunks

# Sentence boundary: . ! or ? followed by whitespace and a capital, digit, or bracket.
# It will sometimes split after abbreviations ("et al. Then"); that only moves a
# chunk boundary slightly, since sentences get packed back together anyway.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

# A unit is (text, separator to put after it): "\n" at the end of a line, else " ".
Unit = tuple[str, str]


@dataclass
class Chunk:
    chunk_id: str      # e.g. "attention_is_all_you_need.pdf:p3:c1"; deterministic
    source: str
    page_number: int
    text: str


def chunk_document(
    doc: Document, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    chunks = []
    for page in doc.pages:
        for i, text in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}:p{page.page_number}:c{i}",
                    source=doc.source,
                    page_number=page.page_number,
                    text=text,
                )
            )
    return chunks


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    current: list[Unit] = []
    for unit in _split_units(text, chunk_size):
        if current and _length(current + [unit]) > chunk_size:
            chunks.append(_join(current))
            current = _tail(current, overlap)
            # Drop overlap sentences if they'd push the next chunk over the size limit.
            while current and _length(current + [unit]) > chunk_size:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append(_join(current))
    return chunks


def _split_units(text: str, max_len: int) -> list[Unit]:
    """Split text into sentences, keeping line breaks. Over-long sentences are cut at word boundaries."""
    units = []
    for line in text.split("\n"):
        sentences = [s for s in SENTENCE_END.split(line.strip()) if s]
        for i, sentence in enumerate(sentences):
            pieces = _split_long(sentence, max_len)
            for j, piece in enumerate(pieces):
                end_of_line = i == len(sentences) - 1 and j == len(pieces) - 1
                units.append((piece, "\n" if end_of_line else " "))
    return units


def _split_long(sentence: str, max_len: int) -> list[str]:
    if len(sentence) <= max_len:
        return [sentence]
    pieces, current = [], ""
    for word in sentence.split(" "):
        if current and len(current) + 1 + len(word) > max_len:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    pieces.append(current)
    return pieces


def _tail(units: list[Unit], overlap: int) -> list[Unit]:
    """The last whole sentences of a chunk, totalling at most `overlap` characters."""
    tail = []
    for unit in reversed(units):
        if _length(tail + [unit]) > overlap:
            break
        tail.insert(0, unit)
    return tail


def _length(units: list[Unit]) -> int:
    return len(_join(units))


def _join(units: list[Unit]) -> str:
    return "".join(text + sep for text, sep in units).strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m research_agent.chunker <file.pdf>")

    doc = clean_document(load_pdf(sys.argv[1]))
    chunks = chunk_document(doc)
    sizes = [len(c.text) for c in chunks]
    print(f"{doc.source}: {len(doc.pages)} pages -> {len(chunks)} chunks")
    print(f"chunk size: min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)} chars\n")

    # Show two neighbouring chunks so the overlap is visible.
    a, b = next((x, y) for x, y in zip(chunks, chunks[1:]) if x.page_number == y.page_number == 3)
    for c in (a, b):
        print(f"--- {c.chunk_id} (page {c.page_number}, {len(c.text)} chars) ---")
        print(c.text, "\n")
