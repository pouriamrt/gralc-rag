"""Graph-guided retrieval with KG-proximity re-ranking.

Extends dense retrieval by computing a knowledge-graph proximity score
between query entities and chunk entities (via their SapBERT embeddings)
and blending it with the dense similarity score for final ranking.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from gralc_rag.knowledge.entity_linker import SimpleEntityLinker
from gralc_rag.retrieval.dense import dense_retrieve
from gralc_rag.retrieval.index import VectorIndex

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# KG proximity
# ------------------------------------------------------------------

def compute_kg_proximity(
    query_entities: list[str],
    chunk_entities: list[str],
    entity_embeddings: dict[str, np.ndarray],
) -> float:
    """Compute a KG-proximity score between two entity sets.

    For each query entity, find its maximum cosine similarity with any
    chunk entity (using their SapBERT embeddings), then average across
    query entities.

    Returns a score in ``[0, 1]``.  If either entity list is empty or
    none of the entities have embeddings, returns ``0.0``.
    """
    if not query_entities or not chunk_entities:
        return 0.0

    # Filter to entities that actually have embeddings
    q_embs = {e: entity_embeddings[e] for e in query_entities if e in entity_embeddings}
    c_embs = {e: entity_embeddings[e] for e in chunk_entities if e in entity_embeddings}

    if not q_embs or not c_embs:
        return 0.0

    # Build chunk embedding matrix once (N_c, dim)
    c_ids = list(c_embs.keys())
    c_matrix = np.stack([c_embs[cid] for cid in c_ids])  # (N_c, dim)
    c_norms = np.linalg.norm(c_matrix, axis=1, keepdims=True)
    c_norms = np.clip(c_norms, a_min=1e-10, a_max=None)
    c_matrix_normed = c_matrix / c_norms

    max_sims: list[float] = []
    for q_id, q_vec in q_embs.items():
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-10:
            max_sims.append(0.0)
            continue
        q_normed = q_vec / q_norm

        # Cosine similarities with all chunk entities
        sims = c_matrix_normed @ q_normed  # (N_c,)
        max_sims.append(float(np.max(sims)))

    avg_sim = float(np.mean(max_sims))
    # Clamp to [0, 1] -- cosine similarity can be negative in theory
    return max(0.0, min(1.0, avg_sim))


# ------------------------------------------------------------------
# Graph-guided retrieval
# ------------------------------------------------------------------

def graph_guided_retrieve(
    query: str,
    model: SentenceTransformer,
    index: VectorIndex,
    entity_linker: SimpleEntityLinker,
    entity_embeddings: dict[str, np.ndarray],
    chunk_entity_map: dict[int, list[str]],
    top_k: int = 5,
    beta: float = 0.7,
) -> list[tuple[dict[str, Any], float]]:
    """Retrieve chunks using a hybrid dense + KG-proximity score.

    1. Fetch top-20 candidates via dense retrieval.
    2. Extract entities from the query.
    3. For each candidate, compute:
       ``hybrid_score = beta * dense_score + (1 - beta) * kg_proximity``
    4. Re-rank by *hybrid_score* and return the top *top_k*.

    Parameters
    ----------
    query:
        Natural-language query string.
    model:
        Sentence-transformers model used for dense encoding.
    index:
        Pre-built :class:`VectorIndex`.
    entity_linker:
        :class:`SimpleEntityLinker` for extracting entities from text.
    entity_embeddings:
        Dict mapping entity identifier -> projected embedding (e.g. 384-d).
    chunk_entity_map:
        Dict mapping chunk index (``metadata["chunk_idx"]``) to a list of
        entity identifiers recognised in that chunk.
    top_k:
        Number of results to return after re-ranking.
    beta:
        Blending weight.  ``beta=1`` means pure dense retrieval; ``beta=0``
        means pure KG-proximity.

    Returns
    -------
    List of ``(metadata, hybrid_score)`` tuples sorted descending.
    """
    # Step 1: dense retrieval -- fetch a wider candidate set for re-ranking
    n_candidates = max(top_k, 20)
    candidates = dense_retrieve(query, model, index, top_k=n_candidates)

    if not candidates:
        return []

    # Step 2: extract query entities
    query_entity_hits = entity_linker.find_entities(query)
    query_entity_ids: list[str] = [
        hit["mesh_id"] for hit in query_entity_hits if hit.get("mesh_id")
    ]

    # If no query entities found, fall back to pure dense ranking
    if not query_entity_ids:
        logger.debug(
            "No query entities found; falling back to pure dense retrieval."
        )
        return candidates[:top_k]

    # Step 3: compute hybrid scores
    scored: list[tuple[dict[str, Any], float]] = []
    for meta, dense_score in candidates:
        chunk_idx = meta.get("chunk_idx")
        chunk_ents: list[str] = []
        if chunk_idx is not None:
            chunk_ents = chunk_entity_map.get(int(chunk_idx), [])

        kg_score = compute_kg_proximity(
            query_entity_ids, chunk_ents, entity_embeddings
        )
        hybrid = beta * dense_score + (1.0 - beta) * kg_score
        scored.append((meta, hybrid))

    # Step 4: re-rank
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
