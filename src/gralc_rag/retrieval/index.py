"""FAISS-backed vector index for dense retrieval.

Wraps a ``faiss.IndexFlatIP`` (inner product on L2-normalised vectors,
equivalent to cosine similarity) with metadata storage and persistence.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
    """In-memory FAISS vector index with associated metadata.

    Parameters
    ----------
    dim:
        Dimensionality of the embedding vectors (default 384 for
        ``all-MiniLM-L6-v2``).
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Build the index from *embeddings* and parallel *metadata*.

        Parameters
        ----------
        embeddings:
            Array of shape ``(n, dim)``.  Vectors are L2-normalised
            internally before indexing so that inner product == cosine
            similarity.
        metadata:
            List of dicts (one per embedding) that will be returned
            alongside search results.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dim}), "
                f"got {embeddings.shape}."
            )
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"metadata length ({len(metadata)}) != number of "
                f"embeddings ({embeddings.shape[0]})."
            )

        # L2-normalise so IP == cosine similarity
        normed = _l2_normalise(embeddings.astype(np.float32))

        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(normed)
        self._metadata = list(metadata)
        logger.info("Built FAISS index with %d vectors (dim=%d).", len(metadata), self.dim)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search the index and return the *top_k* closest results.

        Parameters
        ----------
        query_embedding:
            1-D array of shape ``(dim,)`` or 2-D of shape ``(1, dim)``.
            L2-normalised internally.
        top_k:
            Number of results to return.

        Returns
        -------
        List of ``(metadata_dict, score)`` tuples sorted by descending
        cosine similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Search called on an empty index.")
            return []

        query = np.atleast_2d(query_embedding).astype(np.float32)
        query = _l2_normalise(query)

        # Clamp top_k to the number of indexed vectors
        k = min(top_k, self._index.ntotal)

        distances, indices = self._index.search(query, k)

        results: list[tuple[dict[str, Any], float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._metadata[int(idx)], float(dist)))
        return results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        """Serialize the index and metadata to *path*.

        Creates ``<path>.faiss`` for the FAISS index and
        ``<path>.meta.pkl`` for the metadata list.
        """
        if self._index is None:
            raise RuntimeError("Cannot save: index has not been built yet.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".meta.pkl")

        faiss.write_index(self._index, str(faiss_path))
        with open(meta_path, "wb") as fh:
            pickle.dump(self._metadata, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Saved index to %s and metadata to %s.", faiss_path, meta_path)

    def load(self, path: Path) -> None:
        """Deserialize the index and metadata from *path*.

        Expects ``<path>.faiss`` and ``<path>.meta.pkl`` to exist.
        """
        path = Path(path)
        faiss_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".meta.pkl")

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {faiss_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self._index = faiss.read_index(str(faiss_path))
        self.dim = self._index.d

        with open(meta_path, "rb") as fh:
            self._metadata = pickle.load(fh)  # noqa: S301

        logger.info(
            "Loaded index with %d vectors (dim=%d) from %s.",
            self._index.ntotal,
            self.dim,
            faiss_path,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Return a copy of *vectors* with each row L2-normalised."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-10, a_max=None)
    return vectors / norms
