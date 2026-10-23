"""Web API for the research agent. A thin front door: all the real work happens in research_agent.

Run:
    uvicorn api.main:app --reload
Then open http://localhost:8000/docs for interactive documentation.
"""

import json
import queue
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from research_agent import config, llm_client
from research_agent.embedder import embed_query
from research_agent.ingest import list_documents
from research_agent.llm_client import LLMError
from research_agent.rag import RAGAnswer, answer
from research_agent.retriever import ChromaRetriever
from research_agent.uploads import UploadError, add_uploaded_pdf
from research_agent.vector_store import VectorStore

# How each kind of rejected upload is reported over HTTP.
UPLOAD_ERROR_STATUS = {"invalid_name": 400, "empty": 400, "not_pdf": 415, "too_large": 413, "unreadable": 422}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at server start-up: open the vector store and load the embedding model,
    # so individual requests don't pay that cost.
    app.state.store = VectorStore()
    embed_query("warm-up")
    yield
    # (Code after `yield` would run at shutdown; nothing to clean up yet.)


app = FastAPI(
    title="Personal Research Agent API",
    description="Ask questions about your documents and get cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)


# Response models: describe the exact shape of each JSON response. FastAPI uses them to
# validate what we return and to generate the /docs page.
class HealthResponse(BaseModel):
    status: str           # "ok" if everything works, "degraded" if something needed is missing
    llm_available: bool   # can we reach the LLM, and is the model downloaded?
    llm_model: str
    embedding_model: str
    documents: int
    passages: int


class UploadResult(BaseModel):
    name: str     # the (sanitized) file name the document is stored under
    status: str   # "ingested" | "unchanged" | "reindexed"
    chunks: int


class DocumentInfo(BaseModel):
    name: str
    pages: int
    chunks: int
    processed_at: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Is the service working? Reports document counts and whether the LLM is reachable."""
    llm_ok = llm_client.is_available()
    passages = app.state.store.count()
    return HealthResponse(
        status="ok" if llm_ok and passages > 0 else "degraded",
        llm_available=llm_ok,
        llm_model=config.MODEL,
        embedding_model=config.EMBEDDING_MODEL,
        documents=len(list_documents()),
        passages=passages,
    )


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    """All ingested documents."""
    return [
        DocumentInfo(name=name, pages=info["num_pages"], chunks=info["num_chunks"], processed_at=info["processed_at"])
        for name, info in list_documents().items()
    ]


@app.post("/documents", response_model=UploadResult, status_code=201)
def upload_document(response: Response, file: UploadFile = File(...)) -> UploadResult:
    """Upload a PDF and add it to the searchable documents.

    Uploading a file with the same name as an existing document replaces it.
    """
    try:
        result = add_uploaded_pdf(file.filename, file.file, app.state.store)
    except UploadError as e:
        raise HTTPException(status_code=UPLOAD_ERROR_STATUS[e.reason], detail=str(e)) from e
    if result.status == "unchanged":
        response.status_code = 200  # nothing new was created
    return UploadResult(name=result.source, status=result.status, chunks=len(result.chunks))


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=config.RAG_TOP_K, ge=1, le=20, description="passages to retrieve")
    source: str | None = Field(default=None, description="only search this document (file name)")


class Citation(BaseModel):
    label: str        # "S1", as written in the answer
    source: str       # file name
    page: int
    chunk_id: str
    text: str         # the full cited passage, so the reader can verify the claim


class AskResponse(BaseModel):
    answer: str
    status: str                    # answered | partial | not_found | unsupported
    citations: list[Citation]
    invalid_citations: list[str]   # labels in the answer that match no source
    input_tokens: int
    output_tokens: int
    seconds: float


def _check_ask_request(req: AskRequest) -> None:
    """Problems we can report with a proper status code before any work starts."""
    if app.state.store.count() == 0:
        raise HTTPException(status_code=400, detail="No documents yet. Upload a PDF first (POST /documents).")
    if req.source is not None and req.source not in list_documents():
        raise HTTPException(status_code=404, detail=f"No document named {req.source!r}.")


def _to_ask_response(result: RAGAnswer) -> AskResponse:
    return AskResponse(
        answer=result.text,
        status=result.status,
        citations=[
            Citation(label=f"S{n}", source=s.chunk.source, page=s.chunk.page_number,
                     chunk_id=s.chunk.chunk_id, text=s.chunk.text)
            for n, s in result.cited_sources()
        ],
        invalid_citations=[f"S{n}" for n in result.invalid_citations],
        input_tokens=result.llm.input_tokens,
        output_tokens=result.llm.output_tokens,
        seconds=round(result.llm.total_seconds, 2),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question from the documents, with citations. Returns when the full answer is ready."""
    _check_ask_request(req)
    try:
        result = answer(req.question, ChromaRetriever(app.state.store), k=req.k, source=req.source)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return _to_ask_response(result)


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event: a type line, a JSON data line, then a blank line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Like /ask, but streams the answer as Server-Sent Events:
    `token` events with pieces of text, then one `done` event with the full result
    (same fields as /ask), or an `error` event if something fails mid-way.
    """
    _check_ask_request(req)
    events: queue.Queue = queue.Queue()  # thread-safe "conveyor belt" from the worker to the response

    def worker() -> None:
        try:
            result = answer(
                req.question, ChromaRetriever(app.state.store), k=req.k, source=req.source,
                on_token=lambda piece: events.put(("token", {"text": piece})),
            )
            events.put(("done", _to_ask_response(result).model_dump()))
        except LLMError as e:
            # The 200 status was already sent when streaming began, so report the failure as an event.
            events.put(("error", {"detail": str(e)}))
        finally:
            events.put(None)  # tells the response that the stream is over

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            yield _sse(*item)

    return StreamingResponse(stream(), media_type="text/event-stream")
