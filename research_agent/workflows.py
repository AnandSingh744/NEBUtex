"""Research workflows: several LLM calls in a fixed pattern, each building on the last.

- summarize_document:  map-reduce. Summarise groups of pages, then combine the partial summaries.
- compare_documents:   for each aspect, retrieve passages from each paper and compare them.
- find_contradictions: retrieve passages on a topic from every paper, then pair up and judge the claims.

Usage:
    python -m research_agent.workflows summarize rag_lewis_2020.pdf
    python -m research_agent.workflows compare rag_lewis_2020.pdf dpr_karpukhin_2020.pdf
    python -m research_agent.workflows contradictions "Wikipedia data and number of passages"
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from research_agent import config
from research_agent.llm_client import chat
from research_agent.retriever import StoreRetriever
from research_agent.store import DocumentStore, get_store

GROUP_CHARS = 6000  # text per "map" call: well inside the context window, small enough to read carefully
WORKERS = 2         # LLM calls at the same time (matches Ollama's parallel slots)
COMPARE_ASPECTS = {
    "Problem": "the problem or research question the paper addresses",
    "Method": "the proposed method, model, or architecture",
    "Data": "the datasets, corpora, and knowledge sources used",
    "Results": "the main quantitative results and findings",
    "Limitations": "limitations, weaknesses, or future work",
}

MAP_PROMPT = """Summarise these pages of the research paper "{name}" in 3-6 bullet points.
Keep key facts: methods, numbers, datasets, results. End each bullet with its page, like [p. 4].
Use only the text below.

{text}"""

REDUCE_PROMPT = """Below are partial summaries of the research paper "{name}", in page order.
Write one summary with these sections: **Problem**, **Method**, **Key results**, **Limitations**.
Use only facts from the partial summaries and keep their page citations like [p. 4].
If a section isn't covered, write "Not stated in the paper."

{partials}"""

COMPARE_PROMPT = """Compare two research papers on one aspect: {aspect_description}.

Passages from paper A ({a}):
{a_text}

Passages from paper B ({b}):
{b_text}

In 2-4 sentences, say what each paper does for this aspect and how they differ or relate.
Cite passages as [A1], [B2], etc. Use only the passages. If a paper's passages don't cover the aspect, say so."""

CONTRADICTION_PROMPT = """You check whether research papers agree or contradict each other about: {topic}

Passages (each labelled with its paper and page):
{passages}

Find pairs of claims from DIFFERENT papers that are both about the topic above and about the same thing.
Ignore passages that aren't about the topic. Report at most 5 pairs, the most important ones, and keep each
claim to one short sentence. For each pair, decide:
- "agrees": they say the same thing
- "contradicts": they make incompatible claims about the SAME thing under the SAME conditions
- "different_setting": they look different but concern different tasks, datasets, models or conditions
Be careful: different numbers for different tasks or settings are NOT contradictions.

Reply with JSON only:
{{"pairs": [{{"claim_a": "...", "source_a": "S1", "claim_b": "...", "source_b": "S4",
             "verdict": "agrees" | "contradicts" | "different_setting", "explanation": "..."}}]}}
Use an empty list if no related claims are found."""


def _ask(prompt: str, max_tokens: int = 700, json_mode: bool = False) -> str:
    return chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens,
                json_mode=json_mode).text.strip()


def summarize_document(name: str, store: DocumentStore | None = None) -> str:
    """Map-reduce summary of one document, with page citations."""
    store = store or get_store()
    chunks = store.document_chunks(name)
    if not chunks:
        raise ValueError(f"No document named {name!r}.")

    # Group consecutive pages into batches of about GROUP_CHARS characters (chunk overlap removed by
    # taking each page's text once, from its chunks in order).
    groups, current, size = [], [], 0
    for c in chunks:
        if current and size + len(c.text) > GROUP_CHARS:
            groups.append(current)
            current, size = [], 0
        current.append(c)
        size += len(c.text)
    if current:
        groups.append(current)

    def summarize_group(group) -> str:  # map
        text = "\n\n".join(f"[p. {c.page_number}] {c.text}" for c in group)
        return _ask(MAP_PROMPT.format(name=name, text=text), max_tokens=400)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        partials = list(pool.map(summarize_group, groups))
    # reduce
    return _ask(REDUCE_PROMPT.format(name=name, partials="\n\n".join(partials)), max_tokens=800)


def compare_documents(a: str, b: str, store: DocumentStore | None = None, k: int = 3) -> list[dict]:
    """One row per aspect: {aspect, comparison, sources: {label: citation}}."""
    store = store or get_store()
    retriever = StoreRetriever(store)

    def compare_aspect(item) -> dict:
        aspect, description = item
        hits_a = retriever.search(description, k=k, source=a)
        hits_b = retriever.search(description, k=k, source=b)
        sources = {f"A{i}": h for i, h in enumerate(hits_a, 1)} | {f"B{i}": h for i, h in enumerate(hits_b, 1)}
        text = _ask(COMPARE_PROMPT.format(
            aspect_description=description, a=a, b=b,
            a_text="\n\n".join(f"[A{i}] (p. {h.chunk.page_number}) {h.chunk.text}" for i, h in enumerate(hits_a, 1)),
            b_text="\n\n".join(f"[B{i}] (p. {h.chunk.page_number}) {h.chunk.text}" for i, h in enumerate(hits_b, 1)),
        ), max_tokens=350)
        # Accept [A1] as well as grouped forms like [A1, A2, B3].
        cited = dict.fromkeys(re.findall(r"[AB]\d+", " ".join(re.findall(r"\[([AB]\d+(?:\s*,\s*[AB]\d+)*)\]", text))))
        return {"aspect": aspect, "comparison": text,
                "sources": {label: sources[label].citation for label in cited if label in sources}}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(compare_aspect, COMPARE_ASPECTS.items()))


def find_contradictions(topic: str, documents: list[str] | None = None, store: DocumentStore | None = None,
                        k: int = 3) -> list[dict]:
    """Related claim pairs across documents, each judged agrees / contradicts / different_setting."""
    store = store or get_store()
    documents = documents or list(store.list_documents())
    retriever = StoreRetriever(store)
    hits = [h for doc in documents for h in retriever.search(topic, k=k, source=doc)]  # k per paper: fair coverage
    labelled = {f"S{i}": h for i, h in enumerate(hits, 1)}
    passages = "\n\n".join(f"[{label}] {h.chunk.source}, p. {h.chunk.page_number}: {h.chunk.text}"
                           for label, h in labelled.items())
    reply = _ask(CONTRADICTION_PROMPT.format(topic=topic, passages=passages), max_tokens=1500, json_mode=True)
    try:
        pairs = json.loads(reply).get("pairs", [])
    except (json.JSONDecodeError, AttributeError):
        # Cut off at the token cap: keep every complete pair object (they contain no nested braces).
        pairs = []
        for obj in re.findall(r"\{[^{}]*\}", reply):
            try:
                pairs.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
    results = []
    for p in pairs:
        if not isinstance(p, dict) or p.get("verdict") not in ("agrees", "contradicts", "different_setting"):
            continue
        a, b = labelled.get(str(p.get("source_a"))), labelled.get(str(p.get("source_b")))
        if not a or not b or a.chunk.source == b.chunk.source:
            continue  # unknown labels, or both claims from the same paper: not a cross-document finding
        results.append({**p, "citation_a": a.citation, "citation_b": b.citation})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research workflows.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summarize").add_argument("document")
    cmp = sub.add_parser("compare")
    cmp.add_argument("a")
    cmp.add_argument("b")
    con = sub.add_parser("contradictions")
    con.add_argument("topic")
    con.add_argument("--docs", nargs="+", help="only these documents (default: all)")
    args = parser.parse_args()

    print(f"(model: {config.MODEL})\n")
    if args.command == "summarize":
        print(summarize_document(args.document))
    elif args.command == "compare":
        for row in compare_documents(args.a, args.b):
            print(f"## {row['aspect']}\n{row['comparison']}")
            for label, cite in row["sources"].items():
                print(f"   [{label}] {cite}")
            print()
    else:
        pairs = find_contradictions(args.topic, args.docs)
        print(f"{len(pairs)} related claim pairs found\n")
        for p in pairs:
            print(f"[{p['verdict'].upper()}] {p['claim_a']}  ({p['citation_a']})\n"
                  f"   vs {p['claim_b']}  ({p['citation_b']})\n   -> {p.get('explanation', '')}\n")
