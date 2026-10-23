"""Run trap questions (near-misses, false premises, cross-document mix-ups) and check each answer's status.

Usage:
    python check_tricky.py
"""

import json
from research_agent import config
from research_agent.rag import answer
from research_agent.retriever import ChromaRetriever

QUESTIONS_PATH = config.EVAL_DIR / "tricky_questions.jsonl"

if __name__ == "__main__":
    retriever = ChromaRetriever()
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    passed = 0
    for q in questions:
        result = answer(q["question"], retriever)
        ok = result.status in q["expected"]
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  status={result.status:<11} expected={'/'.join(q['expected'])}")
        print(f"  Q: {q['question']}")
        print(f"  A: {' '.join(result.text.split())}")
        if result.cited:
            print("  cited: " + ", ".join(f"[S{n}] {s.citation}" for n, s in result.cited_sources()))
        print(f"  ({q['why']})\n")
    print(f"{passed}/{len(questions)} statuses as expected. Status only checks the type of answer; read the answers too.")
