"""Knowledge-graph infusion into token-level embeddings.

Core of **GraLC-RAG**: enrich each token embedding with ontological
knowledge derived from SapBERT entity representations before the
chunk-pooling step.  This file provides:

1. ``load_sapbert_embeddings`` -- encode entity names with SapBERT (768-d).
2. ``project_embeddings`` -- project 768-d -> target dim (e.g. 384).
3. ``infuse_kg_into_tokens`` -- additively fuse projected entity embeddings
   into the corresponding token-span embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from gralc_rag.config import DATA_DIR

logger = logging.getLogger(__name__)

_SAPBERT_CACHE: Path = DATA_DIR / "sapbert_cache.npz"


# ------------------------------------------------------------------
# SapBERT embedding helpers
# ------------------------------------------------------------------

def load_sapbert_embeddings(
    entity_ids: list[str],
    model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
) -> dict[str, np.ndarray]:
    """Encode *entity_ids* (treated as entity names) with SapBERT.

    Parameters
    ----------
    entity_ids:
        List of entity identifiers.  Each string is also used as the
        textual input to SapBERT (i.e. the entity *name*).
    model_name:
        HuggingFace model identifier for SapBERT.

    Returns
    -------
    dict mapping each entity id to its 768-dimensional SapBERT embedding.
    Results are cached to ``data/sapbert_cache.npz``; subsequent calls
    with the same ids return instantly.
    """
    # Try to load from cache first
    cached = _load_cache(entity_ids)
    if cached is not None:
        return cached

    logger.info(
        "Encoding %d entities with SapBERT (%s) ...", len(entity_ids), model_name
    )

    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings: dict[str, np.ndarray] = {}
    batch_size = 64

    for start in range(0, len(entity_ids), batch_size):
        batch = entity_ids[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)

        # CLS-pooling (standard for SapBERT)
        cls_embs: np.ndarray = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for name, emb in zip(batch, cls_embs):
            embeddings[name] = emb.astype(np.float32)

    # Persist cache
    _save_cache(embeddings)
    return embeddings


# ------------------------------------------------------------------
# Projection
# ------------------------------------------------------------------

def project_embeddings(
    entity_embs: dict[str, np.ndarray],
    target_dim: int = 384,
) -> dict[str, np.ndarray]:
    """Linearly project SapBERT embeddings (768-d) to *target_dim*.

    Uses PCA fitted on the provided embeddings.  For a production system
    the projection matrix would be learned end-to-end; PCA serves as a
    solid proof-of-concept baseline.
    """
    if not entity_embs:
        return {}

    ids = list(entity_embs.keys())
    matrix = np.stack([entity_embs[eid] for eid in ids])  # (N, 768)
    source_dim = matrix.shape[1]

    if source_dim <= target_dim:
        logger.info(
            "Source dim (%d) <= target dim (%d); skipping projection.",
            source_dim,
            target_dim,
        )
        return dict(entity_embs)  # shallow copy

    n_samples = matrix.shape[0]
    # PCA requires n_components <= min(n_samples, n_features)
    n_components = min(target_dim, n_samples, source_dim)

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(matrix)  # (N, n_components)

    # If we got fewer components than target_dim, zero-pad
    if n_components < target_dim:
        padding = np.zeros(
            (projected.shape[0], target_dim - n_components), dtype=np.float32
        )
        projected = np.concatenate([projected, padding], axis=1)

    result: dict[str, np.ndarray] = {}
    for idx, eid in enumerate(ids):
        result[eid] = projected[idx].astype(np.float32)

    logger.info(
        "Projected %d entity embeddings from %d-d to %d-d.",
        len(result),
        source_dim,
        target_dim,
    )
    return result


# ------------------------------------------------------------------
# Token-level KG infusion
# ------------------------------------------------------------------

def infuse_kg_into_tokens(
    token_embeddings: np.ndarray,
    entity_spans_tokens: list[tuple[int, int, str]],
    entity_embeddings: dict[str, np.ndarray],
    lambda_weight: float = 0.1,
) -> np.ndarray:
    """Infuse KG entity embeddings into token-level representations.

    For every entity span ``(start_tok, end_tok, entity_id)`` the
    corresponding entity embedding (already projected to the token
    embedding dimension) is added -- scaled by *lambda_weight* -- to
    each token in the span.

    Parameters
    ----------
    token_embeddings:
        Array of shape ``(seq_len, dim)`` with per-token embeddings.
    entity_spans_tokens:
        List of ``(start_tok, end_tok, entity_id)`` tuples.  Indices
        are inclusive-start, exclusive-end.
    entity_embeddings:
        Dict mapping entity id -> projected embedding of shape ``(dim,)``.
    lambda_weight:
        Scalar controlling how much ontological signal to inject.

    Returns
    -------
    Enriched token embeddings of the same shape as the input.
    """
    enriched = token_embeddings.copy()
    dim = enriched.shape[1]

    for start_tok, end_tok, entity_id in entity_spans_tokens:
        if entity_id not in entity_embeddings:
            logger.debug("Entity '%s' not found in embedding dict; skipping.", entity_id)
            continue

        ent_emb = entity_embeddings[entity_id]
        if ent_emb.shape[0] != dim:
            logger.warning(
                "Dimension mismatch: entity embedding has %d dims, tokens have %d.",
                ent_emb.shape[0],
                dim,
            )
            continue

        # Clamp span to valid range
        start_tok = max(0, start_tok)
        end_tok = min(end_tok, enriched.shape[0])

        enriched[start_tok:end_tok] += lambda_weight * ent_emb

    return enriched


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------

def _load_cache(entity_ids: list[str]) -> dict[str, np.ndarray] | None:
    """Return cached embeddings if *all* requested ids are present."""
    if not _SAPBERT_CACHE.exists():
        return None

    data = np.load(_SAPBERT_CACHE, allow_pickle=False)
    cached_ids: set[str] = set(data.files)

    if not set(entity_ids).issubset(cached_ids):
        return None

    return {eid: data[eid] for eid in entity_ids}


def _save_cache(embeddings: dict[str, np.ndarray]) -> None:
    """Persist embeddings to an ``.npz`` archive, merging with any existing cache."""
    existing: dict[str, np.ndarray] = {}
    if _SAPBERT_CACHE.exists():
        data = np.load(_SAPBERT_CACHE, allow_pickle=False)
        existing = {k: data[k] for k in data.files}

    existing.update(embeddings)
    _SAPBERT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_SAPBERT_CACHE, **existing)
    logger.debug("Saved %d embeddings to %s", len(existing), _SAPBERT_CACHE)
