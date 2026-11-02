"""Semantic search over ingested chunks.

Two interchangeable retrievers with the same search() interface:
- StoreRetriever: asks the document store (pgvector in Postgres, or Chroma locally). Default; approximate (HNSW).
- NumpyRetriever: exact brute-force cosine similarity over every chunk. A baseline for checking the store.

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

from research_agent import config
from research_agent.chunker import Chunk
from research_agent.embedder import embed_query
from research_agent.store import DocumentStore, get_store

DEFAULT_K = 5


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity; only meaningful for ranking, not as an absolute threshold

    @property
    def citation(self) -> str:
        return f"{self.chunk.source}, p. {self.chunk.page_number}"


RRF_K = 60  # the standard constant in Reciprocal Rank Fusion; dampens the gap between rank 1 and rank 2


def reciprocal_rank_fusion(*ranked_lists: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    """Merge ranked lists by RANK, not score (their scores aren't comparable).

    Each chunk scores sum(1 / (RRF_K + rank)) over the lists it appears in, so chunks ranked
    well by several searches rise to the top.
    """
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            chunks[chunk.chunk_id] = chunk
    return sorted(((chunks[cid], score) for cid, score in fused.items()), key=lambda pair: pair[1], reverse=True)


class StoreRetriever:
    """Search through the document store: vector search, or hybrid (vector + keyword, fused by rank)."""

    def __init__(self, store: DocumentStore, mode: str | None = None):
        self.store = store
        self.mode = mode or config.RETRIEVAL_MODE

    def search(self, query: str, k: int = DEFAULT_K, source: str | None = None) -> list[SearchResult]:
        if not config.QUERY_PLANNING:
            return self._search_one(query, k, source, rerank_query=query)
        from research_agent.query_planner import plan_queries
        documents = [source] if source else list(self.store.list_documents())
        queries = plan_queries(query, documents)
        # The ORIGINAL question's search stays the main result (rewrites can lose specific terms).
        main = self._search_one(query, k, source, rerank_query=query)
        if len(queries) == 1:
            return main
        # Several parts: each part's single best passage is guaranteed a place (replacing the weakest
        # main results), so one document can't crowd out another in a multi-document question.
        extras = []
        for q in queries:
            best = self._search_one(q, 1, source, rerank_query=q)
            if best and all(best[0].chunk.chunk_id != r.chunk.chunk_id for r in main + extras):
                extras.append(best[0])
        return main[: max(k - len(extras), 1)] + extras[: k - 1]

    def _search_one(self, query: str, k: int, source: str | None, rerank_query: str) -> list[SearchResult]:
        # Stage 1: gather candidates (more than k when a reranker will choose among them).
        n = max(k, config.RETRIEVAL_CANDIDATES) if (self.mode == "hybrid" or config.RERANK_MODEL) else k
        if self.mode == "hybrid":
            vector_hits = self.store.search(embed_query(query), n, source=source)
            keyword_hits = self.store.keyword_search(query, n, source=source)
            hits = reciprocal_rank_fusion(vector_hits, keyword_hits)[:n]
        else:
            hits = self.store.search(embed_query(query), n, source=source)
        # Stage 2: let the cross-encoder pick the best k.
        if config.RERANK_MODEL:
            from research_agent.reranker import rerank  # only loaded when reranking is on
            hits = rerank(rerank_query, hits, k)
        return [SearchResult(chunk=chunk, score=score) for chunk, score in hits[:k]]


class NumpyRetriever:
    """Exact search: compares the query with every chunk. Holds all vectors in memory."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = vectors  # shape (n_chunks, dim), each row normalized to length 1

    @classmethod
    def from_store(cls, store: DocumentStore) -> "NumpyRetriever":
        chunks, vectors = store.load_corpus()
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
    parser.add_argument("--backend", choices=["store", "numpy"], default="store")
    parser.add_argument("--compare", action="store_true", help="compare the store's search with exact NumPy search")
    args = parser.parse_args()

    store = get_store()
    embed_query("warm-up")  # load the embedding model first, so it isn't counted in search time
    print(f'Query: "{args.query}"' + (f"  (only {args.source})" if args.source else "") + f"  [store: {type(store).__name__}]\n")

    if args.compare:
        exact = NumpyRetriever.from_store(store).search(args.query, args.k, args.source)
        approx = StoreRetriever(store).search(args.query, args.k, args.source)
        exact_ids = [r.chunk.chunk_id for r in exact]
        approx_ids = [r.chunk.chunk_id for r in approx]
        for rank, (e, a) in enumerate(zip(exact, approx), start=1):
            mark = "=" if e.chunk.chunk_id == a.chunk.chunk_id else "≠"
            print(f"{rank}. numpy {e.chunk.chunk_id} [{e.score:.4f}]  {mark}  store {a.chunk.chunk_id} [{a.score:.4f}]")
        overlap = len(set(exact_ids) & set(approx_ids))
        print(f"\nThe store found {overlap}/{len(exact_ids)} of the exact top-{args.k} (recall@{args.k} = {overlap / len(exact_ids):.2f})")
    else:
        retriever = StoreRetriever(store) if args.backend == "store" else NumpyRetriever.from_store(store)
        t0 = time.perf_counter()
        results = retriever.search(args.query, args.k, args.source)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _print_results(results)
        print(f"[{args.backend} search: {elapsed_ms:.1f} ms]")
