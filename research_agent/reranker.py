"""Second-stage ranking with a cross-encoder: reads the question and each candidate passage TOGETHER
and scores how well the passage answers it. More accurate than vector similarity, but slower, so it
only re-orders the top candidates from the first-stage search.
"""

from fastembed.rerank.cross_encoder import TextCrossEncoder

from research_agent import config
from research_agent.chunker import Chunk

_model: TextCrossEncoder | None = None


def _get_model() -> TextCrossEncoder:
    """Load on first use only (about 0.1 GB for the default model)."""
    global _model
    if _model is None:
        _model = TextCrossEncoder(config.RERANK_MODEL, cache_dir=config.MODEL_CACHE_DIR)
    return _model


def rerank(query: str, hits: list[tuple[Chunk, float]], k: int) -> list[tuple[Chunk, float]]:
    """Re-order candidates by the cross-encoder's relevance score and keep the best k."""
    if not hits:
        return []
    scores = list(_get_model().rerank(query, [chunk.text for chunk, _ in hits]))
    ranked = sorted(zip((chunk for chunk, _ in hits), scores), key=lambda pair: pair[1], reverse=True)
    return [(chunk, float(score)) for chunk, score in ranked[:k]]
