"""Turn text into embedding vectors. The only module that knows which embedding model we use.

Vectors are normalized to length 1, so cosine similarity == dot product.

Usage (similarity demo):
    python -m research_agent.embedder
"""

import numpy as np
from fastembed import TextEmbedding

from research_agent import config

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Load the model on first use only (loading takes time and memory)."""
    global _model
    if _model is None:
        _model = TextEmbedding(config.EMBEDDING_MODEL, cache_dir=config.MODEL_CACHE_DIR)
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed document chunks. Returns an array of shape (len(texts), dim)."""
    vectors = np.array(list(_get_model().passage_embed(texts)), dtype=np.float32)
    return _normalize(vectors)


def embed_query(query: str) -> np.ndarray:
    """Embed a search query (with the model's query prefix). Returns shape (dim,)."""
    vectors = np.array(list(_get_model().query_embed(config.EMBEDDING_QUERY_PREFIX + query)), dtype=np.float32)
    return _normalize(vectors)[0]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


if __name__ == "__main__":
    query = "How do transformers handle long-range dependencies?"
    passages = [
        "Self-attention lets every token attend directly to every other token in the sequence.",
        "Recurrent networks struggle to carry information across many time steps.",
        "Attention mechanisms relate distant words in a sentence.",
        "The recipe needs two cups of flour and a pinch of salt.",
        "The stock market fell sharply on Monday.",
    ]

    q = embed_query(query)
    P = embed_passages(passages)
    print(f"Model: {config.EMBEDDING_MODEL}")
    print(f"Query vector: shape {q.shape}, length {np.linalg.norm(q):.3f}, first 5 values {np.round(q[:5], 3)}\n")

    scores = P @ q  # dot product of each passage with the query == cosine similarity
    print(f'Query: "{query}"\n')
    for score, text in sorted(zip(scores, passages), reverse=True):
        print(f"  {score:.3f}  {text}")
