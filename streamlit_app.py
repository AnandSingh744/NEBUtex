"""Web UI for the research agent (Streamlit). Uses the research_agent engine directly.

Run:
    streamlit run streamlit_app.py

Streamlit re-runs this whole script on every interaction, so:
- st.session_state keeps the chat history between re-runs (one per browser tab)
- @st.cache_resource loads heavy things (database, embedding model) only once
"""

import os

import streamlit as st

# On Streamlit Cloud, settings (LLM_PROVIDER, LLM_API_KEY, ...) come from the app's Secrets box.
# The engine reads settings from environment variables, so copy them over BEFORE importing it.
# Locally there is usually no secrets file, and nothing changes.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except FileNotFoundError:
    pass

from research_agent import config, llm_client  # noqa: E402  (must come after the secrets are loaded)
from research_agent.embedder import embed_query  # noqa: E402
from research_agent.ingest import list_documents  # noqa: E402
from research_agent.llm_client import LLMError  # noqa: E402
from research_agent.rag import answer  # noqa: E402
from research_agent.retriever import ChromaRetriever  # noqa: E402
from research_agent.uploads import UploadError, add_uploaded_pdf  # noqa: E402
from research_agent.vector_store import VectorStore  # noqa: E402

ALL_DOCUMENTS = "All documents"
STATUS_NOTES = {
    "partial": ("warning", "Only part of the question is answered by your documents."),
    "not_found": ("info", "Not found in your documents."),
    "unsupported": ("warning", "This answer cites no sources, so it may not come from your documents."),
}

st.set_page_config(page_title="Research Agent", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the search index and embedding model...")
def get_store() -> VectorStore:
    store = VectorStore()
    embed_query("warm-up")
    return store


@st.cache_data(ttl=30, show_spinner=False)
def llm_status() -> tuple[bool, str]:
    return llm_client.check_llm()  # (ok, reason); re-checked at most every 30 seconds, not on every click


def show_markdown(text: str, target=st) -> None:
    # Streamlit treats $...$ as math; escape dollar signs so prices etc. display normally.
    target.markdown(text.replace("$", "\\$"))


def show_assistant_message(msg: dict) -> None:
    show_markdown(msg["content"])
    if msg["status"] in STATUS_NOTES:
        kind, note = STATUS_NOTES[msg["status"]]
        getattr(st, kind)(note)
    for c in msg["citations"]:
        with st.expander(f"[{c['label']}] {c['source']}, p. {c['page']}"):
            st.text(c["text"])


store = get_store()
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", and for answers: "status", "citations"}
if "notices" not in st.session_state:
    st.session_state.notices = []   # upload results to show after the page refreshes

# ---------- Sidebar: library, upload, settings ----------
with st.sidebar:
    st.header("📚 Library")
    docs = list_documents()
    if docs:
        for name, info in docs.items():
            st.markdown(f"**{name}**  \n{info['num_pages']} pages · {info['num_chunks']} passages")
    else:
        st.caption("No documents yet. Add a PDF below.")

    for kind, text in st.session_state.notices:
        getattr(st, kind)(text)
    st.session_state.notices = []

    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
        submitted = st.form_submit_button("Add to library")
    if submitted and files:
        for f in files:
            with st.spinner(f"Processing {f.name}..."):
                try:
                    result = add_uploaded_pdf(f.name, f, store)
                except UploadError as e:
                    st.session_state.notices.append(("error", f"{f.name}: {e}"))
                else:
                    verb = "already in the library" if result.status == "unchanged" else f"added ({len(result.chunks)} passages)"
                    st.session_state.notices.append(("success", f"{result.source}: {verb}"))
        st.rerun()  # redraw so the library list includes the new documents

    st.divider()
    scope = st.selectbox("Search in", [ALL_DOCUMENTS, *docs])
    k = st.slider("Passages per question", min_value=1, max_value=10, value=config.RAG_TOP_K)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    llm_ok, llm_problem = llm_status()
    if not llm_ok:
        st.error(f"The language model ({config.MODEL}) isn't available. {llm_problem}")

# ---------- Main area: chat ----------
st.title("Personal Research Agent")
st.caption("Answers come only from your documents, with page citations. Each question is answered on its own.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            show_markdown(msg["content"])
        else:
            show_assistant_message(msg)

if question := st.chat_input("Ask a question about your documents"):
    if store.count() == 0:
        st.warning("Add a PDF to the library first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        show_markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        parts: list[str] = []

        def on_token(piece: str) -> None:
            parts.append(piece)
            show_markdown("".join(parts) + "▌", target=placeholder)  # the text so far, with a cursor

        try:
            with st.spinner("Searching your documents..."):
                result = answer(
                    question, ChromaRetriever(store), k=k,
                    source=None if scope == ALL_DOCUMENTS else scope, on_token=on_token,
                )
        except LLMError as e:
            placeholder.error(str(e))
            st.session_state.messages.pop()  # forget the unanswered question
            st.stop()

        placeholder.empty()
        msg = {
            "role": "assistant",
            "content": result.text,
            "status": result.status,
            "citations": [
                {"label": f"S{n}", "source": s.chunk.source, "page": s.chunk.page_number, "text": s.chunk.text}
                for n, s in result.cited_sources()
            ],
        }
        show_assistant_message(msg)
    st.session_state.messages.append(msg)
