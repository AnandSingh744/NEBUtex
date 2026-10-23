# Personal Research Agent

Ask questions about your PDFs and get answers grounded in them, with page citations.
Built step by step as an AI engineering learning project.

**How it works (RAG):** PDFs are split into passages and embedded locally (fastembed,
`BAAI/bge-small-en-v1.5`), stored in Chroma, and the most relevant passages for each
question are given to an LLM with instructions to answer only from them and cite `[S1]`, `[S2]`...

## Run locally (free, private)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama serve      # local LLM
streamlit run streamlit_app.py               # web UI
```

Other ways in: `python research.py` (terminal), `uvicorn api.main:app` (REST API, docs at `/docs`).

## Configuration

All settings are environment variables (see `research_agent/config.py`). To use a hosted
OpenAI-compatible LLM instead of Ollama (e.g. Groq's free tier):

```
LLM_PROVIDER=openai_compatible
LLM_API_KEY=<your key>          # never commit this
```

## Layout

- `research_agent/` – the engine: PDF loading, cleaning, chunking, embeddings, vector store, retrieval, RAG
- `streamlit_app.py` – web UI · `api/` – FastAPI · `research.py` – CLI · `chat.py` – plain chat
- `eval/` + `check_*.py` – retrieval and behaviour checks

## License

[AGPL-3.0](LICENSE). The project uses PyMuPDF, which is AGPL-licensed: if you run a modified version
as a public web service, you must make its source code available.
