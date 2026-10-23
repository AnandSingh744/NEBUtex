"""Measure retrieval quality on a small set of questions with known answer pages.

For each question we find the rank of the first retrieved chunk from the right
document and page, then report:
- hit@k: fraction of questions with a correct chunk in the top k (what the LLM gets to see)
- MRR:   mean of 1/rank of the first correct chunk (0 if missed); how high correct chunks appear

Usage:
    python check_retrieval.py
    python check_retrieval.py --backend numpy
"""

import argparse
import json
from research_agent import config
from research_agent.embedder import embed_query
from research_agent.retriever import ChromaRetriever, NumpyRetriever

QUESTIONS_PATH = config.EVAL_DIR / "retrieval_questions.jsonl"
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)


def load_questions(path=QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_hit_rank(results, source: str, pages: list[int]) -> int | None:
    """1-based rank of the first result from the right document and page, or None if missed."""
    for rank, r in enumerate(results, start=1):
        if r.chunk.source == source and r.chunk.page_number in pages:
            return rank
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval on known-answer questions.")
    parser.add_argument("--backend", choices=["chroma", "numpy"], default="chroma")
    args = parser.parse_args()

    retriever = ChromaRetriever() if args.backend == "chroma" else NumpyRetriever.from_disk()
    embed_query("warm-up")
    questions = load_questions()

    ranks, failures = [], []
    print(f"{'rank':>4}  question")
    for q in questions:
        results = retriever.search(q["question"], k=MAX_K)
        rank = first_hit_rank(results, q["source"], q["pages"])
        ranks.append(rank)
        print(f"{rank if rank else '-':>4}  {q['question']}")
        if rank is None or rank > 3:
            failures.append((q, rank, results[:3]))

    n = len(questions)
    print(f"\n{n} questions, backend={args.backend}")
    for k in K_VALUES:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"  hit@{k:<2} = {hits / n:.2f}  ({hits}/{n})")
    mrr = sum(1 / r for r in ranks if r is not None) / n
    print(f"  MRR    = {mrr:.2f}")

    if failures:
        print("\nNot in top 3 (inspect these):")
        for q, rank, top in failures:
            print(f"\n  Q: {q['question']}")
            print(f"     expected {q['source']} p. {q['pages']}, first correct at rank {rank or f'>{MAX_K}'}")
            for r in top:
                print(f"     got [{r.score:.3f}] {r.citation}: {' '.join(r.chunk.text.split())[:100]}...")
