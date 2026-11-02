"""Where documents live. Two interchangeable stores with the same methods:

- PostgresStore: a PostgreSQL database with pgvector (used when DATABASE_URL is set; survives restarts)
- LocalStore:    files on this computer + a Chroma index (used otherwise, e.g. offline or a fresh clone)

The rest of the engine only uses the DocumentStore methods, so it never needs to know which one it has.
"""

from typing import Protocol

import numpy as np

from research_agent import config
from research_agent.chunker import Chunk


class DocumentStore(Protocol):
    """The methods every store has. (A Protocol describes a shape; any class with these methods fits.)"""

    def save_document(
        self, name: str, sha256: str, num_pages: int, settings: dict,
        chunks: list[Chunk], vectors: np.ndarray,
    ) -> None:
        """Save or replace a document and all its chunks."""

    def get_document(self, name: str) -> dict | None:
        """Info about one document (sha256, settings, num_pages, num_chunks, processed_at), or None."""

    def list_documents(self) -> dict[str, dict]:
        """name -> info for every document."""

    def delete_document(self, name: str) -> bool:
        """Remove a document and its chunks. False if it didn't exist."""

    def count(self, source: str | None = None) -> int:
        """Number of chunks, overall or for one document."""

    def search(self, query_vector: np.ndarray, k: int, source: str | None = None) -> list[tuple[Chunk, float]]:
        """The k chunks most similar to the query vector, best first, as (chunk, cosine similarity)."""

    def load_corpus(self) -> tuple[list[Chunk], np.ndarray]:
        """Every chunk and its vector (row i belongs to chunk i). For exact-search baselines and evaluation."""


def get_store() -> DocumentStore:
    """PostgreSQL if a database is configured, otherwise local files.

    Each store is imported only when chosen, so a cloud app using Postgres never loads Chroma.
    """
    if config.DATABASE_URL:
        from research_agent.pg_store import PostgresStore
        return PostgresStore()
    from research_agent.local_store import LocalStore
    return LocalStore()
