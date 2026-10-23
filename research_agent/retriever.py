"""Semantic search over ingested chunks.

Two interchangeable retrievers with the same search() interface:
- ChromaRetriever: approximate search (HNSW index) in the Chroma vector database. Default.
- NumpyRetriever: exact brute-force cosine similarity. Kept as a baseline/fallback.

Usage:
    python -m research_agent.retriever "What is multi-head attention?"
    python -m research_agent.retriever "How is the retriever trained?" -k 3 --source rag_lewis_2020.pdf
    python -m research_agent.retriever "What is multi-head attention?" --backend numpy
    python -m research_agent.retriever "What is multi-head attention?" --compare
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

from research_agent.chunker import Chunk
from research_agent.embedder import embed_query
from research_agent.ingest import load_corpus
from research_agent.vector_store import VectorStore

DEFAULT_K = 5


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity; only meaningful for ranking, not as an absolute threshold

    @property
    def citation(self) -> str:
        return f"{self.chunk.source}, p. {self.chunk.page_number}"


class NumpyRetriever:
    """Exact search: compares the query with every chunk. Holds the index in memory."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = vectors  # shape (n_chunks, dim), each row normalized to length 1

    @classmethod
    def from_disk(cls) -> "NumpyRetriever":
        chunks, vectors = load_corpus()
        if not chunks:
            raise RuntimeError("No documents ingested yet. Run: python -m research_agent.ingest data/papers")
        return cls(chunks, vectors)

    def search(self, query: str, k: int = DEFAULT_K, source: str | None = None) -> list[SearchResult]:
        query_vector = embed_query(query)
        scores = self.vectors @ query_vector  # cosine similarity with every chunk, shape (n_chunks,)
        if source is not None:
            # Exclude other documents by giving them the lowest possible score.
            in_source = np.array([c.source == source for c in self.chunks])
            scores = np.where(in_source, scores, -np.inf)

        k = min(k, int(np.isfinite(scores).sum()))
        if k == 0:
            return []
        # argpartition finds the top k in O(n) without sorting everything; then sort just those k.
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in top]


class ChromaRetriever:
    """Approximate search via the Chroma vector database (HNSW index)."""

    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        if self.store.count() == 0:
            raise RuntimeError("Vector store is empty. Run: python -m research_agent.ingest data/papers")

    def search(self, query: str, k: int = DEFAULT_K, source: str | None = None) -> list[SearchResult]:
        hits = self.store.search(embed_query(query), k, source=source)
        return [SearchResult(chunk=chunk, score=score) for chunk, score in hits]


def _print_results(results: list[SearchResult]) -> None:
    for rank, r in enumerate(results, start=1):
        preview = " ".join(r.chunk.text.split())[:160]
        print(f"{rank}. [{r.score:.3f}] {r.citation}  ({r.chunk.chunk_id})")
        print(f"   {preview}...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search ingested documents.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="number of results")
    parser.add_argument("--source", help="only search this document (file name)")
    parser.add_argument("--backend", choices=["chroma", "numpy"], default="chroma")
    parser.add_argument("--compare", action="store_true", help="compare Chroma (approximate) with NumPy (exact)")
    args = parser.parse_args()

    embed_query("warm-up")  # load the embedding model first, so it isn't counted in search time
    print(f'Query: "{args.query}"' + (f"  (only {args.source})" if args.source else "") + "\n")

    if args.compare:
        exact = NumpyRetriever.from_disk().search(args.query, args.k, args.source)
        approx = ChromaRetriever().search(args.query, args.k, args.source)
        exact_ids = [r.chunk.chunk_id for r in exact]
        approx_ids = [r.chunk.chunk_id for r in approx]
        for rank, (e, a) in enumerate(zip(exact, approx), start=1):
            mark = "=" if e.chunk.chunk_id == a.chunk.chunk_id else "≠"
            print(f"{rank}. numpy {e.chunk.chunk_id} [{e.score:.4f}]  {mark}  chroma {a.chunk.chunk_id} [{a.score:.4f}]")
        overlap = len(set(exact_ids) & set(approx_ids))
        print(f"\nChroma found {overlap}/{len(exact_ids)} of the exact top-{args.k} (recall@{args.k} = {overlap / len(exact_ids):.2f})")
    else:
        retriever = ChromaRetriever() if args.backend == "chroma" else NumpyRetriever.from_disk()
        t0 = time.perf_counter()
        results = retriever.search(args.query, args.k, args.source)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _print_results(results)
        print(f"[{args.backend} search: {elapsed_ms:.1f} ms]")
