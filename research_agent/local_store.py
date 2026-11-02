"""Documents stored on this computer: chunks as JSON Lines, vectors as .npy, searched with Chroma.

Used when no database is configured. manifest.json records each document; the Chroma index is
derived from the files and is rebuilt automatically if it goes missing or out of sync.
"""

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research_agent import config
from research_agent.chunker import Chunk
from research_agent.vector_store import VectorStore

PROCESSED_DIR = config.PROCESSED_DIR
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"


class LocalStore:
    def __init__(self):
        self._index = VectorStore()
        self._sync_index()

    def save_document(
        self, name: str, sha256: str, num_pages: int, settings: dict,
        chunks: list[Chunk], vectors: np.ndarray,
    ) -> None:
        stem = Path(name).stem
        chunks_path = PROCESSED_DIR / f"{stem}.jsonl"
        vectors_path = PROCESSED_DIR / f"{stem}.npy"
        save_chunks(chunks, chunks_path)
        _save_vectors_atomic(vectors, vectors_path)
        self._index.replace_document(name, chunks, vectors)

        manifest = _load_manifest()
        manifest[name] = {
            "sha256": sha256,
            "settings": settings,
            "num_pages": num_pages,
            "num_chunks": len(chunks),
            "chunks_file": chunks_path.name,
            "vectors_file": vectors_path.name,
            "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _write_json_atomic(manifest, MANIFEST_PATH)

    def get_document(self, name: str) -> dict | None:
        return _load_manifest().get(name)

    def list_documents(self) -> dict[str, dict]:
        return _load_manifest()

    def delete_document(self, name: str) -> bool:
        manifest = _load_manifest()
        entry = manifest.pop(name, None)
        if entry is None:
            return False
        self._index.replace_document(name, [], np.empty((0, config.EMBEDDING_DIM), dtype=np.float32))
        _write_json_atomic(manifest, MANIFEST_PATH)
        for file in (entry["chunks_file"], entry["vectors_file"]):
            (PROCESSED_DIR / file).unlink(missing_ok=True)
        return True

    def count(self, source: str | None = None) -> int:
        return self._index.count(source)

    def search(self, query_vector: np.ndarray, k: int, source: str | None = None) -> list[tuple[Chunk, float]]:
        return self._index.search(query_vector, k, source=source)

    def keyword_search(self, query: str, k: int, source: str | None = None) -> list[tuple[Chunk, float]]:
        return []  # no keyword index for local files: hybrid search falls back to vector search only

    def load_corpus(self) -> tuple[list[Chunk], np.ndarray]:
        chunks, vector_blocks = [], []
        for entry in _load_manifest().values():
            doc_chunks = load_chunks(PROCESSED_DIR / entry["chunks_file"])
            doc_vectors = np.load(PROCESSED_DIR / entry["vectors_file"])
            if len(doc_chunks) != len(doc_vectors):
                raise ValueError(f"{entry['chunks_file']}: {len(doc_chunks)} chunks but {len(doc_vectors)} vectors")
            chunks.extend(doc_chunks)
            vector_blocks.append(doc_vectors)
        vectors = np.vstack(vector_blocks) if vector_blocks else np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)
        return chunks, vectors

    def _sync_index(self) -> None:
        """Rebuild any document missing from the Chroma index, from the saved files (no re-embedding)."""
        for name, entry in _load_manifest().items():
            if self._index.count(name) != entry["num_chunks"]:
                chunks = load_chunks(PROCESSED_DIR / entry["chunks_file"])
                self._index.replace_document(name, chunks, np.load(PROCESSED_DIR / entry["vectors_file"]))
                print(f"reindex {name} ({len(chunks)} chunks from saved files)")


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    lines = [json.dumps(asdict(c), ensure_ascii=False) for c in chunks]
    _write_text_atomic("\n".join(lines) + "\n", path)


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


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
