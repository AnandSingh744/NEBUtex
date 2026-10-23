"""RAG: retrieve relevant chunks, have the LLM answer only from them, and resolve its citations.

Usage:
    python -m research_agent.rag "How many attention heads does the base model use?"
    python -m research_agent.rag "How many attention heads does the base model use?" --show-prompt
"""

import argparse
import re
from dataclasses import dataclass

from research_agent import config
from research_agent.llm_client import LLMError, LLMResponse, chat
from research_agent.prompts import NOT_FOUND_MESSAGE, RAG_SYSTEM_PROMPT
from research_agent.retriever import ChromaRetriever, SearchResult

# Our sources are labelled [S1], [S2], ... so they can't be confused with a paper's own
# reference numbers like [12]. Matches [S1], [S2, S3] or [S2,S3]; [S1][S4] is two matches.
CITATION = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")

# Some models write citations differently: OpenAI's gpt-oss uses 【S1】 or 【S1†L10-L20】, some use
# full-width ［S1］. normalize_citations() rewrites these to the standard [S1] before parsing.
ALT_CITATION = re.compile(r"[【［]\s*(S\d+(?:\s*[,，]\s*S\d+)*)[^】］]*[】］]")


@dataclass
class RAGAnswer:
    question: str
    text: str
    sources: list[SearchResult]   # what the model was shown, in [S1], [S2], ... order
    cited: list[int]              # valid source numbers the answer cites
    invalid_citations: list[int]  # cited [S<n>] labels that match no source (made up)
    llm: LLMResponse              # token counts and timings

    @property
    def status(self) -> str:
        """answered | partial | not_found | unsupported (answered without citing any source)."""
        said_not_found = is_not_found(self.text)
        if said_not_found:
            return "partial" if self.cited else "not_found"
        return "answered" if self.cited else "unsupported"

    def cited_sources(self) -> list[tuple[int, SearchResult]]:
        return [(n, self.sources[n - 1]) for n in self.cited]


def select_sources(results: list[SearchResult], max_chars: int = config.RAG_MAX_CONTEXT_CHARS) -> list[SearchResult]:
    """Keep the best-ranked results that fit in the size budget (results arrive best first)."""
    selected, total = [], 0
    for r in results:
        if selected and total + len(r.chunk.text) > max_chars:
            break  # everything after this ranks lower, so stop here
        selected.append(r)
        total += len(r.chunk.text)
    return selected


def build_rag_messages(question: str, sources: list[SearchResult]) -> list[dict]:
    """System rules first, numbered sources in the middle, the question last."""
    blocks = [f"[S{i}] {r.citation}\n{r.chunk.text}" for i, r in enumerate(sources, start=1)]
    user_message = (
        "<sources>\n" + "\n\n".join(blocks) + "\n</sources>\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def normalize_citations(text: str) -> str:
    """【S1】, 【S1†L10-L20】, ［S1］ -> [S1], so every model's citations look and parse the same."""
    return ALT_CITATION.sub(lambda m: "[" + m.group(1).replace("，", ",") + "]", text)


def parse_citations(text: str, num_sources: int) -> tuple[list[int], list[int]]:
    """Split the source numbers cited in the answer into (valid, invalid), each sorted and unique."""
    numbers = {int(label.strip()[1:]) for match in CITATION.findall(text) for label in match.split(",")}
    valid = sorted(n for n in numbers if 1 <= n <= num_sources)
    invalid = sorted(n for n in numbers if not 1 <= n <= num_sources)
    return valid, invalid


def is_not_found(text: str) -> bool:
    """Did the model use the not-found sentence? Tolerant of case, curly quotes, and "could not"."""
    normalized = text.lower().replace("\u2019", "'").replace("could not", "couldn't")
    return NOT_FOUND_MESSAGE.lower().rstrip(".") in normalized


def answer(
    question: str, retriever, k: int = config.RAG_TOP_K, on_token=None, source: str | None = None
) -> RAGAnswer:
    """The full RAG pipeline: retrieve -> select -> prompt -> LLM -> resolve citations.

    `source` limits retrieval to one document (file name).
    """
    sources = select_sources(retriever.search(question, k=k, source=source))
    messages = build_rag_messages(question, sources)
    response = chat(messages, on_token=on_token)
    text = normalize_citations(response.text)
    cited, invalid = parse_citations(text, len(sources))
    return RAGAnswer(
        question=question,
        text=text,
        sources=sources,
        cited=cited,
        invalid_citations=invalid,
        llm=response,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer a question from the ingested documents.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=config.RAG_TOP_K)
    parser.add_argument("--show-prompt", action="store_true", help="print the full prompt first")
    args = parser.parse_args()

    retriever = ChromaRetriever()
    if args.show_prompt:
        for m in build_rag_messages(args.question, select_sources(retriever.search(args.question, k=args.k))):
            print(f"===== {m['role'].upper()} =====\n{m['content']}\n")

    print(f"Q: {args.question}\n\nA: ", end="", flush=True)
    try:
        result = answer(args.question, retriever, k=args.k, on_token=lambda piece: print(piece, end="", flush=True))
    except LLMError as e:
        raise SystemExit(f"\n[error] {e}")
    print("\n")

    status_notes = {
        "partial": "⚠ Only part of the question is answered by your documents.",
        "not_found": "ℹ Not found in your documents. The closest passages searched were:",
        "unsupported": "⚠ This answer cites no sources, so it may not come from your documents. Treat it with caution.",
    }
    if result.status in status_notes:
        print(status_notes[result.status])
    if result.cited:
        print("Sources:")
        for n, source in result.cited_sources():
            print(f"  [S{n}] {source.citation}")
    if result.invalid_citations:
        print(f"  ⚠ cited labels that match no source: {['S' + str(n) for n in result.invalid_citations]}")
    uncited = [i for i in range(1, len(result.sources) + 1) if i not in result.cited]
    if uncited:
        label = "  " if result.status == "not_found" else "Retrieved but not cited: "
        print(label + ", ".join(f"[S{i}] {result.sources[i - 1].citation}" for i in uncited))
    print(f"\n[status: {result.status} | tokens in: {result.llm.input_tokens} | out: {result.llm.output_tokens} | "
          f"first token: {result.llm.time_to_first_token:.2f}s | total: {result.llm.total_seconds:.1f}s]")
