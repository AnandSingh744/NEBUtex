"""Central settings. Every value can be overridden with an environment variable."""

import os
from pathlib import Path

# The project folder (one level above this package). All paths below are anchored to it,
# so they work no matter which folder a command is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"  # chunks (.jsonl), vectors (.npy), manifest.json
PAPERS_DIR = DATA_DIR / "papers"        # original PDFs (uploads are saved here too)
EVAL_DIR = PROJECT_ROOT / "eval"

# Which LLM service to use:
#   "ollama"            - local Ollama (default; free and private, runs on this machine)
#   "openai_compatible" - a hosted API that speaks the OpenAI format, e.g. Groq's free tier (used in the cloud)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
if LLM_PROVIDER not in ("ollama", "openai_compatible"):
    raise ValueError(f"LLM_PROVIDER must be 'ollama' or 'openai_compatible', got {LLM_PROVIDER!r}")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# For openai_compatible only. The key is a secret: set it in the environment, never in code.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Default model per provider. Hosted models get retired (Groq retired llama-3.1-8b-instant in
# Aug 2026), so override with LLM_MODEL when needed. Locally, "llama3.2:3b" is faster but weaker.
MODEL = os.getenv("LLM_MODEL", "llama3.1:8b" if LLM_PROVIDER == "ollama" else "openai/gpt-oss-20b")

# Reasoning models (like gpt-oss) "think" before answering, and that thinking counts toward
# MAX_OUTPUT_TOKENS. "low" keeps it short. Empty = don't send the setting (for other models).
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low" if "gpt-oss" in MODEL else "")

# 0 = always pick the most likely token (deterministic); higher = more varied.
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Hard cap on generated tokens, so one reply can't run forever.
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))

# Max tokens the model can see in one call (input + output combined).
# Ollama uses a small default unless told otherwise, and silently drops the
# oldest text when it overflows. Bigger = more RAM.
CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))

# Retries for transient failures (server down/busy, rate limits, timeouts).
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))  # seconds; doubles each retry

# Give up if the server sends nothing for this many seconds.
REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

# Embedding model (runs locally via fastembed/ONNX; ~67 MB, downloaded once).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# BGE models were trained with this prefix on search queries (not on passages).
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)
# Where downloaded models are kept (fastembed's default is a temp folder wiped on reboot).
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(PROJECT_ROOT / "models"))

# Chroma vector database folder. It's a derived index: rebuildable from data/processed.
CHROMA_DIR = os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma"))

# RAG: how many chunks to retrieve, and the max total size of sources put in the prompt.
# 12,000 chars is ~3,000 tokens, leaving plenty of room in the context window for the answer.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))

# Largest PDF the API accepts.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
