"""Web UI for the research agent (Streamlit). Uses the research_agent engine directly.

Run:
    streamlit run streamlit_app.py

Streamlit re-runs this whole script on every interaction, so:
- st.session_state keeps the open conversation between re-runs (one per browser tab)
- @st.cache_resource loads heavy things (database, embedding model) only once

Conversations are saved in the database (if DATABASE_URL is set). The open conversation's id
is kept in the page address (?c=<id>), so refreshing the page reopens the same chat.
"""

import logging
import os

import psycopg
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
from research_agent.agent import AgentStep, run_agent  # noqa: E402
from research_agent.chat_store import ChatStore  # noqa: E402
from research_agent.conversation import condense_question  # noqa: E402
from research_agent.report import research_report  # noqa: E402
from research_agent.workflows import compare_documents, find_contradictions, summarize_document  # noqa: E402
from research_agent.db import DatabaseError  # noqa: E402
from research_agent.embedder import embed_query  # noqa: E402
from research_agent.llm_client import LLMError  # noqa: E402
from research_agent.rag import answer  # noqa: E402
from research_agent.retriever import StoreRetriever  # noqa: E402
from research_agent.uploads import UploadError, add_uploaded_pdf  # noqa: E402
from research_agent.store import DocumentStore, get_store  # noqa: E402

ALL_DOCUMENTS = "All documents"
STATUS_NOTES = {
    "partial": ("warning", "Only part of the question is answered by your documents."),
    "not_found": ("info", "Not found in your documents."),
    "unsupported": ("warning", "This answer cites no sources, so it may not come from your documents."),
}

st.set_page_config(page_title="Research Agent", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading your library and the embedding model...")
def load_store() -> DocumentStore:
    store = get_store()  # Postgres if DATABASE_URL is set, otherwise local files
    embed_query("warm-up")
    return store


@st.cache_resource(show_spinner="Connecting to the database...")
def get_chat_store() -> ChatStore | None:
    """The chat history store, or None if there's no database (history then lasts only for this tab)."""
    if not config.DATABASE_URL:
        return None
    try:
        return ChatStore()
    except DatabaseError:
        logging.exception("Chat history disabled: database unavailable")
        return None


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
        where = f"{c['source']}, p. {c['page']}" if c.get("page") else c["source"]  # web results have no page
        with st.expander(f"[{c['label']}] {where}"):
            if c.get("url"):
                st.markdown(f"[Open source]({c['url']})")
            st.text(c["text"])


def open_conversation(conversation_id: str | None) -> None:
    """Switch the page to a conversation (None = a new, empty chat) and redraw."""
    if conversation_id:
        st.query_params["c"] = conversation_id
    else:
        st.query_params.pop("c", None)
    st.rerun()


store = load_store()
chats = get_chat_store()
if "messages" not in st.session_state:
    st.session_state.messages = []      # {"role", "content", and for answers: "status", "citations"}
    st.session_state.loaded_id = None   # which saved conversation `messages` came from
if "notices" not in st.session_state:
    st.session_state.notices = []       # upload results to show after the page refreshes

# Load the conversation named in the page address, if it isn't the one already on screen.
current_id = st.query_params.get("c")
if chats and current_id != st.session_state.loaded_id:
    st.session_state.messages = chats.get_messages(current_id) if current_id else []
    st.session_state.loaded_id = current_id
    if current_id and not st.session_state.messages:
        open_conversation(None)  # unknown or deleted id: start fresh instead of showing an empty "saved" chat

# ---------- Sidebar: conversations, library, upload, settings ----------
with st.sidebar:
    st.header("💬 Conversations")
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        open_conversation(None)
    if chats:
        for conv in chats.list_conversations():
            is_open = conv["id"] == current_id
            if st.button(conv["title"], key=f"conv-{conv['id']}", use_container_width=True,
                         type="primary" if is_open else "secondary"):
                open_conversation(conv["id"])
        if current_id and st.button("🗑 Delete this chat", use_container_width=True):
            chats.delete_conversation(current_id)
            open_conversation(None)
    else:
        st.caption("History isn't saved: no database is configured.")

    st.divider()
    st.header("📚 Library")
    docs = store.list_documents()
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
                    verb = "already in the library" if result.status == "unchanged" else f"added ({result.num_chunks} passages)"
                    st.session_state.notices.append(("success", f"{result.source}: {verb}"))
        st.rerun()  # redraw so the library list includes the new documents

    st.divider()
    workspace = st.radio("Workspace", ["Chat", "Summarize", "Compare", "Contradictions", "Report"], horizontal=True)
    scope = st.selectbox("Search in", [ALL_DOCUMENTS, *docs])
    k = st.slider("Passages per question", min_value=1, max_value=10, value=config.RAG_TOP_K)
    research_mode = st.toggle("🌐 Research mode", help="An agent searches your documents, then arXiv and "
                              "Wikipedia if needed. Slower; works best with a strong model.")

    llm_ok, llm_problem = llm_status()
    if not llm_ok:
        st.error(f"The language model ({config.MODEL}) isn't available. {llm_problem}")

# ---------- Main area ----------
st.title("Personal Research Agent")

VERDICT_STYLE = {"contradicts": ("error", "⚠️ Contradicts"), "agrees": ("success", "✅ Agrees"),
                 "different_setting": ("info", "↔️ Different setting")}


def show_workflow(name: str, names: list[str]) -> None:
    """Summarize / Compare / Contradictions: multi-step research workflows over the library."""
    if not names:
        st.info("Add a PDF to the library first.")
        return
    try:
        if name == "Summarize":
            doc = st.selectbox("Document", names)
            if st.button("Summarize", type="primary"):
                with st.spinner("Summarising each part of the paper, then combining (map-reduce)..."):
                    st.session_state[f"summary:{doc}"] = summarize_document(doc, store)
            if f"summary:{doc}" in st.session_state:
                show_markdown(st.session_state[f"summary:{doc}"])
        elif name == "Compare":
            if len(names) < 2:
                st.info("Add at least two documents to compare.")
                return
            a = st.selectbox("Paper A", names, index=0)
            b = st.selectbox("Paper B", [n for n in names if n != a])
            if st.button("Compare", type="primary"):
                with st.spinner("Retrieving passages from both papers for each aspect..."):
                    st.session_state["comparison"] = (a, b, compare_documents(a, b, store))
            if st.session_state.get("comparison", (None, None))[:2] == (a, b):
                for row in st.session_state["comparison"][2]:
                    st.subheader(row["aspect"])
                    show_markdown(row["comparison"])
                    if row["sources"]:
                        st.caption("  ·  ".join(f"[{label}] {cite}" for label, cite in row["sources"].items()))
        elif name == "Report":
            topic = st.text_input("Report topic", placeholder="e.g. How does retrieval-augmented generation work?")
            if st.button("Write report", type="primary") and topic:
                with st.status("Planning sub-questions...", expanded=True) as progress:
                    def on_step(node: str, update: dict) -> None:
                        if node == "plan":
                            progress.write("📋 **Plan:** " + " · ".join(update["subquestions"]))
                            progress.update(label="Researching each sub-question...")
                        elif node == "research":
                            progress.write("🔎 **Research:** " + ", ".join(f["status"] for f in update["findings"]))
                            progress.update(label="Writing the report...")
                    st.session_state["report"] = research_report(topic, store, on_step=on_step)["report"]
                    progress.update(label="Done", state="complete", expanded=False)
            if "report" in st.session_state:
                show_markdown(st.session_state["report"])
        else:
            topic = st.text_input("Topic to check across papers", placeholder="e.g. Wikipedia corpus used and number of passages")
            chosen = st.multiselect("Papers", names, default=names)
            if st.button("Find agreements and contradictions", type="primary") and topic and len(chosen) >= 2:
                with st.spinner("Comparing claims across papers..."):
                    st.session_state["contradictions"] = find_contradictions(topic, chosen, store)
            for pair in st.session_state.get("contradictions", []):
                kind, label = VERDICT_STYLE[pair["verdict"]]
                getattr(st, kind)(f"**{label}**\n\n**A:** {pair['claim_a']}  *({pair['citation_a']})*\n\n"
                                  f"**B:** {pair['claim_b']}  *({pair['citation_b']})*\n\n{pair.get('explanation', '')}")
    except LLMError as e:
        st.error(str(e))


if workspace != "Chat":
    show_workflow(workspace, list(docs))
    st.stop()

st.caption("Answers come only from your documents, with page citations. Follow-up questions use the conversation.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            show_markdown(msg["content"])
        else:
            show_assistant_message(msg)

def answer_with_rag(question: str) -> dict:
    """Normal mode: retrieve passages and stream a cited answer."""
    placeholder = st.empty()
    parts: list[str] = []

    def on_token(piece: str) -> None:
        parts.append(piece)
        show_markdown("".join(parts) + "▌", target=placeholder)  # the text so far, with a cursor

    try:
        with st.spinner("Searching your documents..."):
            result = answer(question, StoreRetriever(store), k=k,
                            source=None if scope == ALL_DOCUMENTS else scope, on_token=on_token)
    except LLMError as e:
        placeholder.error(str(e))
        st.session_state.messages.pop()  # forget the unanswered question
        st.stop()
    placeholder.empty()
    return {
        "role": "assistant", "content": result.text, "status": result.status,
        "citations": [{"label": f"S{n}", "source": s.chunk.source, "page": s.chunk.page_number, "text": s.chunk.text}
                      for n, s in result.cited_sources()],
    }


def answer_with_agent(question: str) -> dict:
    """Research mode: the agent searches documents, then arXiv/Wikipedia if needed, showing each step."""
    with st.status("Researching...", expanded=True) as progress:
        def on_step(step: AgentStep) -> None:
            found = f"error: {step.error}" if step.error else f"{len(step.labels)} results"
            progress.write(f"🔧 **{step.tool}** `{step.arguments.get('query', '')}` → {found}")

        try:
            res = run_agent(question, store, on_step=on_step)
        except LLMError as e:
            progress.update(label="Failed", state="error")
            st.error(str(e))
            st.session_state.messages.pop()
            st.stop()
        progress.update(label=f"Done: {len(res.steps)} searches", state="complete", expanded=False)
    if res.invalid_citations:
        st.warning(f"The answer cites sources that don't exist: {', '.join(res.invalid_citations)}")
    citations = []
    for label in res.cited:
        src = res.sources[label]
        is_doc = src["kind"] == "document"
        citations.append({"label": label, "source": src["source"] if is_doc else src["title"],
                          "page": src["page"] if is_doc else None, "url": src["url"], "text": src["text"]})
    return {"role": "assistant", "content": res.text, "status": res.status, "citations": citations}


if question := st.chat_input("Ask a question about your documents"):
    if store.count() == 0:
        st.warning("Add a PDF to the library first.")
        st.stop()

    history = list(st.session_state.messages)  # earlier turns, for follow-up questions
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        show_markdown(question)

    with st.chat_message("assistant"):
        standalone = condense_question(question, history) if history else question
        if standalone != question:
            st.caption(f"🔎 Searched for: {standalone}")  # so a bad rewrite is easy to spot
        msg = answer_with_agent(standalone) if research_mode else answer_with_rag(standalone)
        show_assistant_message(msg)
    st.session_state.messages.append(msg)

    # Save the question and answer. A failure here shouldn't lose the answer already on screen.
    if chats:
        try:
            new_conversation = current_id is None
            conversation_id = current_id or chats.create_conversation(question)
            chats.add_message(conversation_id, "user", question)
            chats.add_message(conversation_id, "assistant", msg["content"], msg["status"], msg["citations"])
        except (DatabaseError, psycopg.Error):
            logging.exception("Could not save the conversation")
            st.warning("This answer couldn't be saved to your chat history.")
        else:
            if new_conversation:
                st.session_state.loaded_id = conversation_id  # already on screen; no need to reload it
                open_conversation(conversation_id)            # put its id in the address + list it in the sidebar
