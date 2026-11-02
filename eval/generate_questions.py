"""Draft test questions from your documents with the LLM, then filter out weak ones.

The output is a list of CANDIDATES for a person to review, not a finished test set:
never trust LLM-written test data without checking it.

Usage (from the project folder):
    python -m eval.generate_questions --per-doc 14
"""

import argparse
import json
import random
import re
from collections import Counter

from research_agent import config
from research_agent.llm_client import LLMError, chat
from research_agent.store import get_store

OUTPUT_PATH = config.EVAL_DIR / "candidates.jsonl"
MIN_CHARS = 400        # very short chunks rarely contain a full fact
MIN_SUPPORT = 0.6      # share of the answer's words that must appear in the passage
COPY_NGRAM = 6         # a question repeating 6+ consecutive passage words is copied, not paraphrased
VAGUE_PHRASES = ("this passage", "the passage", "the text", "the authors", "this paper", "this study")

WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""a an the of in on for to and or is are was were be been by with as at from that this
which what how why when who whom does do did it its into than then there their they them can could
would should about over under used use using per each such these those also""".split())

PROMPT = """You write test questions for a research assistant.

Below is a passage from a research paper (file "{source}", page {page}).

Write ONE question that a researcher might ask, which this passage answers, and its short answer.
Rules:
- The question must make sense on its own. Name the paper's method or topic (for example
  "In the Dense Passage Retrieval paper, ..."). Never say "this passage", "this paper" or "the authors".
- Use your own words. Do not copy phrases from the passage into the question.
- The answer must come from the passage: a short phrase or one sentence.
- Prefer specific facts: numbers, names, methods, settings, results.

Reply with JSON only, like {{"question": "...", "answer": "..."}}

Passage:
\"\"\"
{text}
\"\"\""""


def content_words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS]


def is_reference_list(text: str) -> bool:
    """Bibliography chunks: several lines that start like '[12] Author ...'."""
    return sum(1 for line in text.splitlines() if re.match(r"\s*\[\d+\]", line)) >= 3


def answer_support(answer: str, passage: str) -> float:
    """Share of the answer's content words that appear in the passage (1.0 = fully supported)."""
    words = content_words(answer)
    passage_words = set(content_words(passage))
    return sum(w in passage_words for w in words) / len(words) if words else 0.0


def copies_passage(question: str, passage: str, n: int = COPY_NGRAM) -> bool:
    """Does the question repeat n consecutive words from the passage?"""
    q, p = WORD.findall(question.lower()), WORD.findall(passage.lower())
    passage_ngrams = {tuple(p[i:i + n]) for i in range(len(p) - n + 1)}
    return any(tuple(q[i:i + n]) in passage_ngrams for i in range(len(q) - n + 1))


def parse_json_reply(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)  # models sometimes add words around the JSON
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(data.get("question"), str) and isinstance(data.get("answer"), str):
        return {"question": data["question"].strip(), "answer": data["answer"].strip()}
    return None


def check(draft: dict | None, passage: str) -> str | None:
    """The reason to reject a draft, or None if it passes every filter."""
    if draft is None:
        return "unreadable reply"
    if not draft["question"].endswith("?") or len(draft["question"]) < 15:
        return "not a proper question"
    if any(p in draft["question"].lower() for p in VAGUE_PHRASES):
        return "vague (doesn't name the paper)"
    if answer_support(draft["answer"], passage) < MIN_SUPPORT:
        return "answer not in the passage"
    if copies_passage(draft["question"], passage):
        return "question copies the passage"
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draft and filter test questions.")
    parser.add_argument("--per-doc", type=int, default=14, help="passages to draft from, per document")
    parser.add_argument("--seed", type=int, default=7, help="random seed, so runs are repeatable")
    args = parser.parse_args()

    chunks, _ = get_store().load_corpus()
    eligible = [c for c in chunks if len(c.text) >= MIN_CHARS and not is_reference_list(c.text)]
    rng = random.Random(args.seed)
    by_doc: dict[str, list] = {}
    for c in eligible:
        by_doc.setdefault(c.source, []).append(c)
    picked = [c for doc in by_doc.values() for c in rng.sample(doc, min(args.per_doc, len(doc)))]
    print(f"{len(chunks)} chunks, {len(eligible)} eligible (long enough, not reference lists), drafting from {len(picked)}\n")

    kept, rejections = [], Counter()
    for c in picked:
        try:
            reply = chat([{"role": "user", "content": PROMPT.format(source=c.source, page=c.page_number, text=c.text)}])
        except LLMError as e:
            rejections["LLM error"] += 1
            print(f"  LLM error: {e}")
            continue
        draft = parse_json_reply(reply.text)
        reason = check(draft, c.text)
        if reason:
            rejections[reason] += 1
            print(f"  ✗ {reason:32} {c.chunk_id}  {(draft or {}).get('question', '')[:70]}")
            continue
        kept.append({
            "question": draft["question"], "answer": draft["answer"],
            "source": c.source, "pages": [c.page_number], "chunk_id": c.chunk_id,
            "type": "factual", "origin": "generated",
        })
        print(f"  ✓ {c.chunk_id:34} {draft['question'][:80]}")

    OUTPUT_PATH.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), encoding="utf-8")
    print(f"\nkept {len(kept)} of {len(picked)} drafts -> {OUTPUT_PATH.relative_to(config.PROJECT_ROOT)} (review before use!)")
    for reason, n in rejections.most_common():
        print(f"  rejected, {reason}: {n}")
