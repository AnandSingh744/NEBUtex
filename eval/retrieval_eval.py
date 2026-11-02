"""Retrieval evaluation on eval/testset.jsonl: did the search find the passages that hold the answer?

Metrics, for questions that have evidence:
- hit@k              at least one evidence location is in the top k   (the LLM CAN answer)
- evidence recall@k  share of the evidence locations found in the top k (matters when the answer needs several)
- MRR                mean of 1/rank of the first hit                    (how high the right passages appear)
- precision@k        share of the top k that come from evidence pages   (how much noise the LLM must read past;
                     underestimates quality, since unlabelled pages can be relevant too)

Usage (from the project folder):
    python -m eval.retrieval_eval
    python -m eval.retrieval_eval --backend numpy     # exact search, for comparison
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from research_agent import config
from research_agent.embedder import embed_query
from research_agent.retriever import NumpyRetriever, StoreRetriever
from research_agent.store import get_store

TESTSET_PATH = config.EVAL_DIR / "testset.jsonl"
K_VALUES = (1, 3, 5, 10)
REPORT_K = 5  # the k used for breakdowns (our RAG_TOP_K)


def load_testset(path=TESTSET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _matches(chunk, evidence: dict) -> bool:
    return chunk.source == evidence["source"] and chunk.page_number in evidence["pages"]


@dataclass
class RetrievalResult:
    item: dict
    retrieved: list            # SearchResult objects, best first
    first_hit: int | None      # 1-based rank of the first chunk matching any evidence
    evidence_ranks: list       # per evidence location: rank of its first matching chunk (None if missed)
    relevant_ranks: list[int]  # ranks of every retrieved chunk that matches some evidence

    def hit(self, k: int) -> bool:
        return self.first_hit is not None and self.first_hit <= k

    def evidence_recall(self, k: int) -> float:
        return sum(1 for r in self.evidence_ranks if r is not None and r <= k) / len(self.evidence_ranks)

    def precision(self, k: int) -> float:
        return sum(1 for r in self.relevant_ranks if r <= k) / k

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.first_hit if self.first_hit else 0.0


def evaluate_retrieval(retriever, items: list[dict], max_k: int = max(K_VALUES)) -> list[RetrievalResult]:
    """Run retrieval for every question that has evidence (not_answerable questions are skipped)."""
    answerable = [item for item in items if item["evidence"]]
    with ThreadPoolExecutor(max_workers=4) as pool:  # searches mostly wait on the database: overlap them
        all_retrieved = list(pool.map(lambda item: retriever.search(item["question"], k=max_k), answerable))
    results = []
    for item, retrieved in zip(answerable, all_retrieved):
        ranks_per_chunk = [
            [i for i, e in enumerate(item["evidence"]) if _matches(r.chunk, e)] for r in retrieved
        ]
        relevant_ranks = [rank for rank, hits in enumerate(ranks_per_chunk, start=1) if hits]
        evidence_ranks = [
            next((rank for rank, hits in enumerate(ranks_per_chunk, start=1) if i in hits), None)
            for i in range(len(item["evidence"]))
        ]
        results.append(RetrievalResult(item, retrieved, relevant_ranks[0] if relevant_ranks else None,
                                       evidence_ranks, relevant_ranks))
    return results


def summarize(results: list[RetrievalResult], k_values=K_VALUES) -> dict:
    """Average every metric over the results."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    summary = {"n": n, "mrr": sum(r.reciprocal_rank for r in results) / n}
    for k in k_values:
        summary[f"hit@{k}"] = sum(r.hit(k) for r in results) / n
        summary[f"recall@{k}"] = sum(r.evidence_recall(k) for r in results) / n
        summary[f"precision@{k}"] = sum(r.precision(k) for r in results) / n
    return summary


def _paper(item: dict) -> str:
    sources = sorted({e["source"].split("_")[0] for e in item["evidence"]})
    return "+".join(sources)


def print_report(results: list[RetrievalResult]) -> None:
    s = summarize(results)
    print(f"{s['n']} questions with evidence\n")
    print(f"{'k':>4} {'hit@k':>7} {'recall@k':>9} {'precision@k':>12}")
    for k in K_VALUES:
        print(f"{k:>4} {s[f'hit@{k}']:>7.2f} {s[f'recall@{k}']:>9.2f} {s[f'precision@{k}']:>12.2f}")
    print(f"MRR = {s['mrr']:.2f}")

    for title, key in (("type", lambda r: r.item["type"]), ("paper", lambda r: _paper(r.item)),
                       ("origin", lambda r: r.item["origin"])):
        groups = defaultdict(list)
        for r in results:
            groups[key(r)].append(r)
        print(f"\nBy {title} (k={REPORT_K}):")
        for name, group in sorted(groups.items()):
            g = summarize(group, (REPORT_K,))
            print(f"  {name:22} n={g['n']:>2}  hit={g[f'hit@{REPORT_K}']:.2f}  "
                  f"recall={g[f'recall@{REPORT_K}']:.2f}  MRR={g['mrr']:.2f}")

    misses = [r for r in results if r.evidence_recall(REPORT_K) < 1]
    if misses:
        print(f"\nNot all evidence in the top {REPORT_K} ({len(misses)}):")
        for r in misses:
            where = ", ".join(f"{e['source'].split('_')[0]} p{e['pages']}->{rank or '>10'}"
                              for e, rank in zip(r.item["evidence"], r.evidence_ranks))
            got = ", ".join(f"{x.chunk.source.split('_')[0]} p{x.chunk.page_number}" for x in r.retrieved[:3])
            print(f"  {r.item['id']} {r.item['question'][:70]}")
            print(f"       evidence rank: {where} | top 3: {got}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the test set.")
    parser.add_argument("--backend", choices=["store", "numpy"], default="store")
    args = parser.parse_args()

    store = get_store()
    retriever = StoreRetriever(store) if args.backend == "store" else NumpyRetriever.from_store(store)
    embed_query("warm-up")
    print(f"Retrieval evaluation: backend={args.backend}, store={type(store).__name__}, "
          f"embedding={config.EMBEDDING_MODEL}\n")
    print_report(evaluate_retrieval(retriever, load_testset()))
