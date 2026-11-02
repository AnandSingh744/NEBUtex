"""Documents and their chunks (text + vectors) stored in PostgreSQL with pgvector.

Replaces three local things: manifest.json -> `documents` table, .jsonl + .npy -> `chunks` table,
and the Chroma index -> pgvector's HNSW index. Has the same methods as LocalStore (see store.py).

Usage:
    python -m research_agent.pg_store                 # list documents in the database
    python -m research_agent.pg_store --import-local  # copy local papers (data/processed) into it
"""

import argparse
import time

import numpy as np
from psycopg.types.json import Jsonb

from research_agent.chunker import Chunk
from research_agent.db import get_pool

# One row per document, with its chunk count. LEFT JOIN keeps documents that have no chunks.
_DOCUMENT_INFO = """
    SELECT d.name, d.sha256, d.settings, d.num_pages, d.processed_at, count(c.id) AS num_chunks
    FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
"""


class PostgresStore:
    def __init__(self):
        self._pool = get_pool()

    def save_document(
        self, name: str, sha256: str, num_pages: int, settings: dict,
        chunks: list[Chunk], vectors: np.ndarray,
    ) -> None:
        """Save (or replace) a document and all its chunks in ONE transaction: all or nothing."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        with self._pool.connection() as conn, conn.transaction():
            # Upsert: insert the document, or update it if one with this name already exists.
            document_id = conn.execute(
                """
                INSERT INTO documents (name, sha256, num_pages, settings, processed_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (name) DO UPDATE
                    SET sha256 = EXCLUDED.sha256, num_pages = EXCLUDED.num_pages,
                        settings = EXCLUDED.settings, processed_at = now()
                RETURNING id
                """,
                (name, sha256, num_pages, Jsonb(settings)),
            ).fetchone()[0]
            # Remove the previous version's chunks (the new version may have fewer).
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            with conn.cursor() as cur:
                # executemany sends all rows as one pipelined batch, not one round trip per row.
                cur.executemany(
                    "INSERT INTO chunks (id, document_id, page_number, chunk_index, text, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        # PostgreSQL text can't contain NUL; strip it here too as a safety net for
                        # chunks cleaned before text_cleaner learned to remove control characters.
                        (c.chunk_id, document_id, c.page_number, i, c.text.replace("\x00", ""), v)
                        for i, (c, v) in enumerate(zip(chunks, vectors))
                    ],
                )

    def get_document(self, name: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(_DOCUMENT_INFO + " WHERE d.name = %s GROUP BY d.id", (name,)).fetchone()
        return _to_info(row)[1] if row else None

    def list_documents(self) -> dict[str, dict]:
        """name -> info, in the same shape as the old manifest entries."""
        with self._pool.connection() as conn:
            rows = conn.execute(_DOCUMENT_INFO + " GROUP BY d.id ORDER BY d.id").fetchall()
        return dict(_to_info(row) for row in rows)

    def count(self, source: str | None = None) -> int:
        """Number of chunks, overall or for one document."""
        with self._pool.connection() as conn:
            if source is None:
                return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            return conn.execute(
                "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.name = %s",
                (source,),
            ).fetchone()[0]

    def search(self, query_vector: np.ndarray, k: int, source: str | None = None) -> list[tuple[Chunk, float]]:
        """The k most similar chunks, computed inside the database: only the winners travel back."""
        sql = """
            SELECT c.id, d.name, c.page_number, c.text, 1 - (c.embedding <=> %(q)s) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            {where}
            ORDER BY c.embedding <=> %(q)s      -- <=> is cosine distance: smallest = most similar
            LIMIT %(k)s
        """
        params = {"q": query_vector, "k": k, "source": source}
        with self._pool.connection() as conn, conn.transaction():
            if source is None:
                rows = conn.execute(sql.format(where=""), params).fetchall()
            else:
                # The HNSW index finds ~40 nearest candidates from ALL documents, then the WHERE removes
                # other documents', which can leave fewer than k. An iterative scan keeps searching
                # until enough matches are found. SET LOCAL lasts only for this transaction.
                conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                rows = conn.execute(sql.format(where="WHERE d.name = %(source)s"), params).fetchall()
        results = [(Chunk(chunk_id=r[0], source=r[1], page_number=r[2], text=r[3]), float(r[4])) for r in rows]
        # "relaxed_order" may return results slightly out of order; put the best first.
        return sorted(results, key=lambda pair: pair[1], reverse=True)

    def keyword_search(self, query: str, k: int, source: str | None = None) -> list[tuple[Chunk, float]]:
        """Full-text search: chunks sharing the query's words, best first. Strong on exact terms and numbers."""
        sql = """
            SELECT c.id, d.name, c.page_number, c.text, ts_rank_cd(c.tsv, q) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id,
                 -- plainto_tsquery ANDs every word ("a & b & c"); a long question would then match almost
                 -- nothing, so switch to OR ("a | b | c") and let the ranking reward chunks with more words.
                 to_tsquery('english', replace(plainto_tsquery('english', %(query)s)::text, '&', '|')) AS q
            WHERE c.tsv @@ q {source_filter}
            ORDER BY score DESC
            LIMIT %(k)s
        """.format(source_filter="AND d.name = %(source)s" if source else "")
        with self._pool.connection() as conn:
            rows = conn.execute(sql, {"query": query, "k": k, "source": source}).fetchall()
        return [(Chunk(chunk_id=r[0], source=r[1], page_number=r[2], text=r[3]), float(r[4])) for r in rows]

    def load_corpus(self) -> tuple[list[Chunk], np.ndarray]:
        """Every chunk with its vector, in document then chunk order. Used by the exact-search baseline."""
        with self._pool.connection() as conn:
            rows = conn.execute("""
                SELECT c.id, d.name, c.page_number, c.text, c.embedding
                FROM chunks c JOIN documents d ON d.id = c.document_id
                ORDER BY d.id, c.chunk_index
            """).fetchall()
        chunks = [Chunk(chunk_id=r[0], source=r[1], page_number=r[2], text=r[3]) for r in rows]
        # pgvector hands back Vector objects; .to_numpy() turns each into a NumPy array.
        vectors = np.array([r[4].to_numpy() for r in rows], dtype=np.float32).reshape(len(rows), -1)
        return chunks, vectors

    def delete_document(self, name: str) -> bool:
        """Delete a document; its chunks go with it (ON DELETE CASCADE). False if it didn't exist."""
        with self._pool.connection() as conn:
            return conn.execute("DELETE FROM documents WHERE name = %s", (name,)).rowcount > 0


def _to_info(row) -> tuple[str, dict]:
    name, sha256, settings, num_pages, processed_at, num_chunks = row
    return name, {
        "sha256": sha256,
        "settings": settings,
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "processed_at": processed_at.isoformat(timespec="seconds"),
    }


def import_local(store: PostgresStore) -> None:
    """Copy documents already processed on disk into Postgres. No PDF parsing, no re-embedding."""
    from research_agent.local_store import PROCESSED_DIR, _load_manifest, load_chunks

    for name, entry in _load_manifest().items():
        existing = store.get_document(name)
        if (
            existing
            and existing["sha256"] == entry["sha256"]
            and existing["settings"] == entry["settings"]
            and existing["num_chunks"] == entry["num_chunks"]
        ):
            print(f"skip    {name} (already in the database)")
            continue
        chunks = load_chunks(PROCESSED_DIR / entry["chunks_file"])
        vectors = np.load(PROCESSED_DIR / entry["vectors_file"])
        start = time.perf_counter()
        store.save_document(name, entry["sha256"], entry["num_pages"], entry["settings"], chunks, vectors)
        print(f"import  {name}: {len(chunks)} chunks in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Documents stored in PostgreSQL.")
    parser.add_argument("--import-local", action="store_true", help="copy local processed papers into the database")
    args = parser.parse_args()

    store = PostgresStore()
    if args.import_local:
        import_local(store)
        print()
    docs = store.list_documents()
    for name, info in docs.items():
        print(f"{name}: {info['num_pages']} pages, {info['num_chunks']} chunks, processed {info['processed_at']}")
    print(f"\n{len(docs)} documents, {store.count()} chunks in the database")
