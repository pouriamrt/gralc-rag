"""Semantic chunking strategy.

Splits text at sentence boundaries where the cosine similarity between
consecutive sentence embeddings drops below a configurable threshold.
"""

from __future__ import annotations

import re

import numpy as np
from sentence_transformers import SentenceTransformer

from gralc_rag.chunking.naive import Chunk

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY_RE = re.compile(
    r"""(?<=[.!?])   # lookbehind for sentence-ending punctuation
        \s+          # one or more whitespace chars
        (?=[A-Z])    # lookahead for an uppercase letter (new sentence)
    """,
    re.VERBOSE,
)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a simple regex heuristic."""
    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _word_count(text: str) -> int:
    return len(text.split())


def _split_by_max_tokens(text: str, max_tokens: int) -> list[str]:
    """Hard-split *text* into pieces of at most *max_tokens* words."""
    words = text.split()
    pieces: list[str] = []
    for i in range(0, len(words), max_tokens):
        pieces.append(" ".join(words[i : i + max_tokens]))
    return pieces


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def semantic_chunk(
    text: str,
    doc_id: str,
    model: SentenceTransformer,
    threshold: float = 0.75,
    max_chunk_tokens: int = 512,
) -> list[Chunk]:
    """Chunk *text* by detecting semantic shifts between sentences.

    Parameters:
        text: Source document text.
        doc_id: Unique document identifier.
        model: A ``sentence_transformers.SentenceTransformer`` used to encode
            individual sentences.
        threshold: Cosine-similarity threshold below which a split is placed.
        max_chunk_tokens: Maximum word count per chunk; oversized chunks are
            hard-split.

    Returns:
        Ordered list of :class:`Chunk` objects (embeddings are ``None`` —
        the caller is expected to embed the final chunk texts).
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Single sentence → single chunk.
    if len(sentences) == 1:
        return _finalise(sentences, doc_id, max_chunk_tokens, position_offset=0)

    # Encode all sentences in one batch.
    embeddings: np.ndarray = model.encode(
        sentences, show_progress_bar=False, convert_to_numpy=True
    )

    # Compute pairwise cosine similarities between consecutive sentences.
    similarities = [
        _cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(sentences) - 1)
    ]

    # Group sentences into raw segments where consecutive similarity >= threshold.
    groups: list[list[str]] = [[sentences[0]]]
    for idx, sim in enumerate(similarities):
        if sim < threshold:
            groups.append([sentences[idx + 1]])
        else:
            groups[-1].append(sentences[idx + 1])

    # Merge very small groups (< 50 words) with their neighbours.
    groups = _merge_small_groups(groups, min_words=50)

    # Flatten groups into chunk texts and enforce max token limit.
    return _finalise(
        [" ".join(g) for g in groups],
        doc_id,
        max_chunk_tokens,
        position_offset=0,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_small_groups(
    groups: list[list[str]], min_words: int = 50
) -> list[list[str]]:
    """Merge groups with fewer than *min_words* words into neighbours."""
    if len(groups) <= 1:
        return groups

    merged: list[list[str]] = [groups[0]]
    for group in groups[1:]:
        if _word_count(" ".join(merged[-1])) < min_words:
            merged[-1].extend(group)
        else:
            merged.append(group)

    # Final group might still be too small — merge backwards.
    if len(merged) >= 2 and _word_count(" ".join(merged[-1])) < min_words:
        merged[-2].extend(merged.pop())

    return merged


def _finalise(
    texts: list[str],
    doc_id: str,
    max_chunk_tokens: int,
    position_offset: int,
) -> list[Chunk]:
    """Convert raw text segments into :class:`Chunk` objects, enforcing max size."""
    chunks: list[Chunk] = []
    position = position_offset
    for t in texts:
        if _word_count(t) > max_chunk_tokens:
            for piece in _split_by_max_tokens(t, max_chunk_tokens):
                chunks.append(
                    Chunk(
                        text=piece,
                        embedding=None,
                        metadata={
                            "strategy": "semantic",
                            "position": position,
                            "doc_id": doc_id,
                        },
                    )
                )
                position += 1
        else:
            chunks.append(
                Chunk(
                    text=t,
                    embedding=None,
                    metadata={
                        "strategy": "semantic",
                        "position": position,
                        "doc_id": doc_id,
                    },
                )
            )
            position += 1
    return chunks
