"""Personal Research Agent (CLI): ask questions about your PDFs and get cited answers.

Usage:
    python research.py
Commands:
    /add <pdf or folder>  add documents
    /docs                 list documents
    /sources              show the full passages cited in the last answer
    /k <number>           how many passages to retrieve per question (default from config)
    /help                 show commands
    /quit                 exit

Each question is answered on its own (no memory of earlier questions), because
follow-ups like "what about the big model?" would retrieve the wrong passages.
"""

from research_agent import config
from research_agent.embedder import embed_query
from research_agent.ingest import _find_pdfs, ingest_pdf, list_documents
from research_agent.llm_client import LLMError
from research_agent.pdf_loader import PDFLoadError
from research_agent.rag import RAGAnswer, answer
from research_agent.retriever import ChromaRetriever
from research_agent.vector_store import VectorStore

HELP = """Commands:
  /add <pdf or folder>  add documents
  /docs                 list documents
  /sources              show the full passages cited in the last answer
  /k <number>           passages to retrieve per question
  /help                 show this help
  /quit                 exit
Anything else is treated as a question."""

STATUS_NOTES = {
    "partial": "⚠ Only part of the question is answered by your documents.",
    "not_found": "ℹ Not found in your documents.",
    "unsupported": "⚠ This answer cites no sources, so it may not come from your documents. Treat it with caution.",
}


def show_docs() -> None:
    docs = list_documents()
    if not docs:
        print("No documents yet. Use /add <pdf or folder>.\n")
        return
    for name, info in docs.items():
        print(f"  {name}: {info['num_pages']} pages, {info['num_chunks']} chunks")
    print()


def add_documents(target: str, store: VectorStore) -> None:
    pdfs = _find_pdfs([target])
    if not pdfs:
        print(f"No PDFs found at {target}\n")
        return
    for pdf in pdfs:
        try:
            ingest_pdf(pdf, store=store)
        except PDFLoadError as e:
            print(f"error   {pdf.name}: {e}")
    print()


def show_sources(last: RAGAnswer | None) -> None:
    if last is None or not last.cited:
        print("No cited sources yet. Ask a question first.\n")
        return
    for n, source in last.cited_sources():
        print(f"----- [S{n}] {source.citation} ({source.chunk.chunk_id}) -----")
        print(source.chunk.text + "\n")


def ask(question: str, store: VectorStore, k: int) -> RAGAnswer | None:
    if store.count() == 0:
        print("No documents yet. Use /add <pdf or folder> first.\n")
        return None

    print("\nAssistant: ", end="", flush=True)
    try:
        result = answer(question, ChromaRetriever(store), k=k, on_token=lambda piece: print(piece, end="", flush=True))
    except LLMError as e:
        print(f"\n\n[error] {e}\n")
        return None
    except KeyboardInterrupt:
        print("\n(stopped)\n")
        return None
    print("\n")

    if result.status in STATUS_NOTES:
        print(STATUS_NOTES[result.status])
    if result.cited:
        print("Sources: " + "  ".join(f"[S{n}] {s.citation}" for n, s in result.cited_sources()))
    if result.invalid_citations:
        print(f"⚠ cited labels that match no source: {['S' + str(n) for n in result.invalid_citations]}")
    hint = " | /sources to read the cited passages" if result.cited else ""
    print(f"[{result.status} | {result.llm.total_seconds:.1f}s{hint}]\n")
    return result


def main() -> None:
    store = VectorStore()
    print("Loading...", end="", flush=True)
    embed_query("warm-up")  # load the embedding model once, up front
    k = config.RAG_TOP_K
    print(f"\rPersonal Research Agent ({config.MODEL}). {store.count()} passages indexed. Type /help for commands.\n")

    last: RAGAnswer | None = None
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        command, _, arg = line.partition(" ")
        if command == "/quit":
            break
        elif command == "/help":
            print(HELP + "\n")
        elif command == "/docs":
            show_docs()
        elif command == "/add":
            if arg:
                add_documents(arg.strip(), store)
            else:
                print("Usage: /add <pdf or folder>\n")
        elif command == "/sources":
            show_sources(last)
        elif command == "/k":
            if arg.strip().isdigit() and int(arg) > 0:
                k = int(arg)
                print(f"Now retrieving {k} passages per question.\n")
            else:
                print(f"Usage: /k <number>  (currently {k})\n")
        elif command.startswith("/"):
            print(f"Unknown command {command}. Type /help.\n")
        else:
            result = ask(line, store, k)
            if result is not None:
                last = result

    print("Bye!")


if __name__ == "__main__":
    main()
