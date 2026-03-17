"""Dense retrieval utilities (query embedding + index search).

Provides thin wrappers around sentence-transformers encoding and
:class:`~gralc_rag.retrieval.index.VectorIndex` search.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from gralc_rag.retrieval.index import VectorIndex

logger = logging.getLogger(__name__)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Encode *query* with a sentence-transformers model.

    Returns a 1-D, L2-normalised embedding.
    """
    embedding: np.ndarray = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    # model.encode returns (dim,) when given a single string
    embedding = np.atleast_1d(embedding).astype(np.float32)
    return _l2_normalise_1d(embedding)


def dense_retrieve(
    query: str,
    model: SentenceTransformer,
    index: VectorIndex,
    top_k: int = 5,
) -> list[tuple[dict[str, Any], float]]:
    """Embed *query* and search *index*, returning *top_k* results.

    Returns a list of ``(metadata, score)`` tuples sorted by descending
    cosine similarity.
    """
    q_emb = embed_query(query, model)
    return index.search(q_emb, top_k=top_k)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _l2_normalise_1d(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D vector in-place (returns the same array)."""
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec /= norm
    return vec
