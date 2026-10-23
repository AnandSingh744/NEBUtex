"""Persistent vector index (Chroma). The only module that talks to the vector database.

Chroma is a derived index: the source of truth is data/processed/*.jsonl + *.npy,
so the index can always be rebuilt from those files without re-embedding.
"""

import re

import chromadb
import numpy as np

from research_agent import config
from research_agent.chunker import Chunk


def _collection_name(model: str) -> str:
    # One collection per embedding model, so vectors from different models never mix.
    return "chunks-" + re.sub(r"[^a-zA-Z0-9._-]", "_", model)


class VectorStore:
    def __init__(self, path: str = config.CHROMA_DIR, model: str = config.EMBEDDING_MODEL):
        client = chromadb.PersistentClient(path=path)
        self._collection = client.get_or_create_collection(
            _collection_name(model),
            embedding_function=None,  # we always pass our own vectors; never let Chroma embed
            configuration={"hnsw": {"space": "cosine"}},
        )

    def replace_document(self, source: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Index a document's chunks, removing any old chunks from the same source first.

        A plain upsert would leave stale chunks behind if the new version has fewer chunks.
        """
        self._collection.delete(where={"source": source})
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page_number": c.page_number} for c in chunks],
        )

    def count(self, source: str | None = None) -> int:
        if source is None:
            return self._collection.count()
        return len(self._collection.get(where={"source": source}, include=[])["ids"])

    def search(
        self, query_vector: np.ndarray, k: int, source: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Approximate nearest neighbours (HNSW). Returns (chunk, cosine similarity) pairs, best first."""
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns one list per query; we sent a single query.
        ids, docs, metas, dists = (result[key][0] for key in ("ids", "documents", "metadatas", "distances"))
        return [
            (
                Chunk(chunk_id=i, source=m["source"], page_number=m["page_number"], text=d),
                1.0 - dist,  # Chroma's cosine distance = 1 - cosine similarity
            )
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]
