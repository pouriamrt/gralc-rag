"""Naive fixed-window text chunking strategy.

Splits text into fixed-size token windows using whitespace tokenisation
(words as a proxy for tokens) with configurable overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chunk:
    """A single text chunk produced by any chunking strategy.

    Attributes:
        text: The chunk's raw text content.
        embedding: Dense vector representation (filled after encoding).
        metadata: Strategy-specific metadata such as *section_title*,
            *position*, *doc_id*, and *strategy*.
    """

    text: str
    embedding: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def naive_chunk(
    text: str,
    doc_id: str,
    max_tokens: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Split *text* into fixed-size overlapping token windows.

    Parameters:
        text: Source document text.
        doc_id: Unique document identifier stored in each chunk's metadata.
        max_tokens: Maximum number of whitespace tokens per chunk.
        overlap: Number of tokens shared between consecutive chunks.

    Returns:
        Ordered list of :class:`Chunk` objects (embeddings are ``None``).
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    if overlap >= max_tokens:
        overlap = max_tokens // 4  # safety fallback

    stride = max_tokens - overlap
    chunks: list[Chunk] = []

    start = 0
    position = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append(
            Chunk(
                text=chunk_text,
                embedding=None,
                metadata={
                    "strategy": "naive",
                    "position": position,
                    "doc_id": doc_id,
                },
            )
        )

        position += 1

        # If we've consumed all words, stop.
        if end >= len(words):
            break

        start += stride

    return chunks
