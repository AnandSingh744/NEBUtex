"""Ingestion pipeline: PDF -> load -> clean -> chunk -> embed -> save.

Chunks are saved as JSON Lines, their vectors as a .npy array in the same order.
Processing happens once per document. A manifest records each PDF's content hash
and the settings used, so unchanged documents are skipped on later runs.

Usage:
    python -m research_agent.ingest data/papers                 # every PDF in a folder
    python -m research_agent.ingest paper1.pdf paper2.pdf       # specific files
    python -m research_agent.ingest data/papers --force         # reprocess even if unchanged
"""

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research_agent import config
from research_agent.chunker import CHUNK_OVERLAP, CHUNK_SIZE, Chunk, chunk_document
from research_agent.embedder import embed_passages
from research_agent.pdf_loader import PDFLoadError, load_pdf
from research_agent.text_cleaner import clean_document
from research_agent.vector_store import VectorStore

PROCESSED_DIR = config.PROCESSED_DIR
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"


@dataclass
class IngestResult:
    source: str
    status: str          # "ingested" (processed now) | "unchanged" (skipped) | "reindexed" (index rebuilt from files)
    chunks: list[Chunk]


def ingest_pdf(path: str | Path, force: bool = False, store: VectorStore | None = None) -> IngestResult:
    """Process one PDF (or reuse its saved chunks if nothing changed).

    Also keeps the vector store in sync with the saved files.
    """
    store = store or VectorStore()
    path = Path(path)
    if not path.is_file():
        raise PDFLoadError(f"File not found: {path}")
    file_hash = _sha256(path)
    # Changing any of these makes saved chunks/vectors stale, so they're part of the cache key.
    settings = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
    }
    chunks_path = PROCESSED_DIR / f"{path.stem}.jsonl"
    vectors_path = PROCESSED_DIR / f"{path.stem}.npy"

    manifest = _load_manifest()
    entry = manifest.get(path.name)
    up_to_date = (
        entry is not None
        and entry["sha256"] == file_hash
        and entry["settings"] == settings
        and chunks_path.exists()
        and vectors_path.exists()
    )
    if up_to_date and not force:
        chunks = load_chunks(chunks_path)
        if store.count(path.name) != len(chunks):
            # Index missing or out of sync (e.g. data/chroma deleted): rebuild from saved files, no re-embedding.
            store.replace_document(path.name, chunks, np.load(vectors_path))
            print(f"reindex {path.name} ({len(chunks)} chunks from saved files)")
            return IngestResult(path.name, "reindexed", chunks)
        print(f"skip    {path.name} (unchanged, {len(chunks)} chunks)")
        return IngestResult(path.name, "unchanged", chunks)

    doc = clean_document(load_pdf(path))
    chunks = chunk_document(doc)
    vectors = embed_passages([c.text for c in chunks])
    save_chunks(chunks, chunks_path)
    _save_vectors_atomic(vectors, vectors_path)
    store.replace_document(path.name, chunks, vectors)

    manifest[path.name] = {
        "sha256": file_hash,
        "settings": settings,
        "num_pages": len(doc.pages),
        "num_chunks": len(chunks),
        "chunks_file": chunks_path.name,
        "vectors_file": vectors_path.name,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json_atomic(manifest, MANIFEST_PATH)
    print(f"ingest  {path.name}: {len(doc.pages)} pages -> {len(chunks)} chunks -> vectors {vectors.shape}")
    return IngestResult(path.name, "ingested", chunks)


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    lines = [json.dumps(asdict(c), ensure_ascii=False) for c in chunks]
    _write_text_atomic("\n".join(lines) + "\n", path)


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def load_corpus() -> tuple[list[Chunk], np.ndarray]:
    """Every chunk from every ingested document, plus their vectors (row i belongs to chunk i).

    This is what retrieval searches.
    """
    manifest = _load_manifest()
    chunks, vector_blocks = [], []
    for entry in manifest.values():
        doc_chunks = load_chunks(PROCESSED_DIR / entry["chunks_file"])
        doc_vectors = np.load(PROCESSED_DIR / entry["vectors_file"])
        if len(doc_chunks) != len(doc_vectors):
            raise ValueError(f"{entry['chunks_file']}: {len(doc_chunks)} chunks but {len(doc_vectors)} vectors")
        chunks.extend(doc_chunks)
        vector_blocks.append(doc_vectors)
    vectors = np.vstack(vector_blocks) if vector_blocks else np.empty((0, 0), dtype=np.float32)
    return chunks, vectors


def list_documents() -> dict[str, dict]:
    """Ingested documents: file name -> manifest entry (pages, chunks, processed_at, ...)."""
    return _load_manifest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in 1 MB blocks so large PDFs don't have to fit in memory at once.
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_json_atomic(data: dict, path: Path) -> None:
    _write_text_atomic(json.dumps(data, indent=2) + "\n", path)


def _write_text_atomic(text: str, path: Path) -> None:
    """Write to a temp file, then rename. A crash mid-write never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem


def _save_vectors_atomic(vectors: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        np.save(f, vectors)
    os.replace(f.name, path)


def _find_pdfs(inputs: list[str]) -> list[Path]:
    pdfs = []
    for item in map(Path, inputs):
        pdfs.extend(sorted(item.glob("*.pdf")) if item.is_dir() else [item])
    return pdfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into chunked JSON Lines files.")
    parser.add_argument("inputs", nargs="+", help="PDF files and/or folders containing PDFs")
    parser.add_argument("--force", action="store_true", help="reprocess even if unchanged")
    args = parser.parse_args()

    store = VectorStore()
    failed = 0
    for pdf in _find_pdfs(args.inputs):
        try:
            ingest_pdf(pdf, force=args.force, store=store)
        except PDFLoadError as e:
            failed += 1
            print(f"error   {pdf.name}: {e}")

    chunks, vectors = load_corpus()
    total = len(chunks)
    print(
        f"\n{total} chunks (vectors {vectors.shape}) available from {len(_load_manifest())} documents"
        f" | {store.count()} in vector store" + (f" ({failed} failed)" if failed else "")
    )
