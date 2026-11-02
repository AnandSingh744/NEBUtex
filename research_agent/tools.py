"""Tools the research agent can ask to use. Each one is a plain function plus a JSON Schema description
(TOOL_SPECS) that tells the model what the tool does and which arguments it takes.

All free, no API keys: our own documents, arXiv's official API, Wikipedia's official API.
Results from the internet are untrusted text: the agent treats them as data, never as instructions.
"""

import xml.etree.ElementTree as ET

import httpx

from research_agent.retriever import StoreRetriever
from research_agent.store import DocumentStore

HTTP_TIMEOUT = 15
# Wikimedia asks API users to identify themselves with a descriptive User-Agent.
USER_AGENT = "NEBUtex-research-agent/0.1 (https://github.com/AnandSingh744/NEBUtex)"
MAX_TEXT = 1200  # characters of each result shown to the model (keeps the context window in check)

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search the user's uploaded research papers. Use this first for questions about them.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look for, in the papers' own terms"},
            "document": {"type": "string", "description": "optional: only search this file, e.g. 'rag_lewis_2020.pdf'"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_arxiv",
        "description": "Search arXiv for research papers (title, authors, abstract). Use for papers not in the user's documents.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords, e.g. 'dense passage retrieval'"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_wikipedia",
        "description": "Search Wikipedia for background facts and definitions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "a topic, e.g. 'BLEU score'"},
        }, "required": ["query"]}}},
]


def search_documents(store: DocumentStore, query: str, document: str | None = None) -> list[dict]:
    if document and document not in store.list_documents():
        document = None  # the model guessed a file name that doesn't exist: search everything instead
    hits = StoreRetriever(store).search(query, k=4, source=document)
    return [{"kind": "document", "title": h.citation, "url": "", "text": h.chunk.text[:MAX_TEXT],
             "source": h.chunk.source, "page": h.chunk.page_number, "chunk_id": h.chunk.chunk_id} for h in hits]


def search_arxiv(query: str) -> list[dict]:
    response = httpx.get("https://export.arxiv.org/api/query", timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT},
                         params={"search_query": f"all:{query}", "max_results": 3, "sortBy": "relevance"})
    response.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in ET.fromstring(response.text).findall("a:entry", ns):
        title = " ".join((entry.findtext("a:title", "", ns)).split())
        authors = [a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)]
        year = (entry.findtext("a:published", "", ns))[:4]
        summary = " ".join((entry.findtext("a:summary", "", ns)).split())
        byline = f"{', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''} ({year})"
        results.append({"kind": "web", "title": f"arXiv: {title}, {byline}", "url": entry.findtext("a:id", "", ns),
                        "text": summary[:MAX_TEXT]})
    return results


def search_wikipedia(query: str) -> list[dict]:
    headers = {"User-Agent": USER_AGENT}
    api = "https://en.wikipedia.org/w/api.php"
    found = httpx.get(api, timeout=HTTP_TIMEOUT, headers=headers, params={
        "action": "query", "list": "search", "srsearch": query, "srlimit": 2, "format": "json"}).json()
    titles = [r["title"] for r in found.get("query", {}).get("search", [])]
    if not titles:
        return []
    pages = httpx.get(api, timeout=HTTP_TIMEOUT, headers=headers, params={
        "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "titles": "|".join(titles),
        "format": "json"}).json()
    results = []
    for page in pages.get("query", {}).get("pages", {}).values():
        title = page.get("title", "")
        results.append({"kind": "web", "title": f"Wikipedia: {title}",
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "text": " ".join(page.get("extract", "").split())[:MAX_TEXT]})
    return results
