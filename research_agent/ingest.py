"""Ingestion pipeline: PDF -> load -> clean -> chunk -> embed -> save to the document store.

Processing happens once per document: a PDF whose content (sha256) and settings match what's
already stored is skipped.

Usage:
    python -m research_agent.ingest data/papers                 # every PDF in a folder
    python -m research_agent.ingest paper1.pdf paper2.pdf       # specific files
    python -m research_agent.ingest data/papers --force         # reprocess even if unchanged
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from research_agent import config
from research_agent.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_document
from research_agent.embedder import embed_passages
from research_agent.pdf_loader import PDFLoadError, load_pdf
from research_agent.store import DocumentStore, get_store
from research_agent.text_cleaner import clean_document


@dataclass
class IngestResult:
    source: str
    status: str      # "ingested" (processed now) | "unchanged" (already stored, skipped)
    num_chunks: int


def current_settings() -> dict:
    """Changing any of these makes stored chunks/vectors stale, so they're part of the 'unchanged' check."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
    }


def ingest_pdf(path: str | Path, force: bool = False, store: DocumentStore | None = None) -> IngestResult:
    """Process one PDF and save it to the store, unless an identical version is already there."""
    store = store or get_store()
    path = Path(path)
    if not path.is_file():
        raise PDFLoadError(f"File not found: {path}")
    file_hash = _sha256(path)
    settings = current_settings()

    existing = store.get_document(path.name)
    if not force and existing and existing["sha256"] == file_hash and existing["settings"] == settings:
        print(f"skip    {path.name} (unchanged, {existing['num_chunks']} chunks)")
        return IngestResult(path.name, "unchanged", existing["num_chunks"])

    doc = clean_document(load_pdf(path))
    chunks = chunk_document(doc)
    vectors = embed_passages([c.text for c in chunks])
    store.save_document(path.name, file_hash, len(doc.pages), settings, chunks, vectors)
    print(f"ingest  {path.name}: {len(doc.pages)} pages -> {len(chunks)} chunks -> vectors {vectors.shape}")
    return IngestResult(path.name, "ingested", len(chunks))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in 1 MB blocks so large PDFs don't have to fit in memory at once.
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _find_pdfs(inputs: list[str]) -> list[Path]:
    pdfs = []
    for item in map(Path, inputs):
        pdfs.extend(sorted(item.glob("*.pdf")) if item.is_dir() else [item])
    return pdfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into the document store.")
    parser.add_argument("inputs", nargs="+", help="PDF files and/or folders containing PDFs")
    parser.add_argument("--force", action="store_true", help="reprocess even if unchanged")
    args = parser.parse_args()

    store = get_store()
    print(f"Store: {type(store).__name__}\n")
    failed = 0
    for pdf in _find_pdfs(args.inputs):
        try:
            ingest_pdf(pdf, force=args.force, store=store)
        except PDFLoadError as e:
            failed += 1
            print(f"error   {pdf.name}: {e}")

    print(f"\n{store.count()} chunks available from {len(store.list_documents())} documents"
          + (f" ({failed} failed)" if failed else ""))
