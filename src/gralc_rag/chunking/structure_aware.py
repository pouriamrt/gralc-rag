"""Structure-aware late chunking -- the GraLC-RAG innovation.

Combines structural cues (section / subsection / paragraph boundaries),
semantic dissimilarity between local token windows, and biomedical entity
span preservation to place chunk boundaries that respect the logical
structure of a scientific article.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import find_peaks
from sentence_transformers import SentenceTransformer

from gralc_rag.chunking.naive import Chunk
from gralc_rag.corpus.parser import Paragraph, ParsedArticle

# ---------------------------------------------------------------------------
# Sliding-window token embeddings (shared logic with late.py)
# ---------------------------------------------------------------------------


def _triangular_weights(length: int) -> np.ndarray:
    """Return a triangular weight vector peaking at the centre."""
    if length <= 1:
        return np.ones(length, dtype=np.float64)
    half = length / 2.0
    return 1.0 - np.abs(np.arange(length, dtype=np.float64) - half + 0.5) / half


def _compute_token_embeddings(
    text: str,
    model: SentenceTransformer,
    tokenizer,
    max_tokens: int = 512,
    overlap: int = 128,
) -> tuple[np.ndarray, list[int]]:
    """Produce per-token contextualised embeddings for the entire *text*.

    Uses a sliding window over the transformer with triangular weighting for
    overlapping regions.

    Returns:
        token_embeddings: ``(n_tokens, hidden_dim)`` float32 array.
        token_to_char: list mapping each token index to its start character
            offset in *text*.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )

    input_ids: torch.Tensor = encoding["input_ids"].squeeze(0)
    offsets: list[tuple[int, int]] = encoding["offset_mapping"].squeeze(0).tolist()
    total_tokens = input_ids.shape[0]
    token_to_char = [start for start, _end in offsets]

    transformer = model[0].auto_model
    device = next(transformer.parameters()).device

    hidden_dim: int | None = None
    weighted_sum: np.ndarray | None = None
    weight_sum: np.ndarray | None = None

    stride = max_tokens - overlap
    if stride <= 0:
        stride = max_tokens // 2

    with torch.no_grad():
        for win_start in range(0, total_tokens, stride):
            win_end = min(win_start + max_tokens, total_tokens)
            win_ids = input_ids[win_start:win_end].unsqueeze(0).to(device)
            attention_mask = torch.ones_like(win_ids)

            outputs = transformer(input_ids=win_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            win_len = hidden.shape[0]
            if hidden_dim is None:
                hidden_dim = hidden.shape[1]
                weighted_sum = np.zeros((total_tokens, hidden_dim), dtype=np.float64)
                weight_sum = np.zeros(total_tokens, dtype=np.float64)

            weights = _triangular_weights(win_len)
            weighted_sum[win_start:win_end] += hidden * weights[:, None]
            weight_sum[win_start:win_end] += weights

            if win_end >= total_tokens:
                break

    safe_denom = np.maximum(weight_sum[:, None], 1e-12)
    token_embeddings = (weighted_sum / safe_denom).astype(np.float32)
    return token_embeddings, token_to_char


# ---------------------------------------------------------------------------
# Boundary scoring
# ---------------------------------------------------------------------------

def _cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _map_char_offset_to_token(
    char_offset: int,
    token_to_char: list[int],
) -> int:
    """Return the token index whose character span contains *char_offset*.

    Uses binary search for efficiency.
    """
    lo, hi = 0, len(token_to_char) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_to_char[mid] <= char_offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


def compute_boundary_scores(
    paragraphs: list[Paragraph],
    token_embeddings: np.ndarray,
    token_to_char: list[int],
    text: str,
    entity_spans: list[tuple[int, int]] | None = None,
    alpha1: float = 0.5,
    alpha2: float = 0.3,
    window_size: int = 64,
) -> np.ndarray:
    """Score every inter-token position for its suitability as a chunk boundary.

    For each gap between token *i* and token *i+1* (``i`` in
    ``0 .. n_tokens-2``), the score is:

        ``alpha1 * structural + alpha2 * semantic + entity_penalty``

    where
        * **structural** is 1.0 at section boundaries (``section_level == 0``
          and the paragraph is the first in its section), 0.7 at subsection
          boundaries (``section_level > 0``), 0.4 at plain paragraph
          boundaries, and 0.0 elsewhere.
        * **semantic** is ``1 - cosine_similarity(left_window, right_window)``
          with each window being the mean of *window_size* token embeddings on
          that side of the gap.
        * **entity_penalty** is -0.5 if the gap falls inside a named-entity
          span (discouraging splits mid-entity).

    Parameters:
        paragraphs: Ordered paragraphs from a :class:`ParsedArticle`.
        token_embeddings: ``(n_tokens, dim)`` contextualised embeddings.
        token_to_char: Token index -> character offset mapping.
        text: The full concatenated document text.
        entity_spans: Optional list of ``(char_start, char_end)`` entity spans.
        alpha1: Weight for the structural component.
        alpha2: Weight for the semantic component.
        window_size: Token window size for local semantic dissimilarity.

    Returns:
        1-D float array of length ``n_tokens - 1``.
    """
    n_tokens = token_embeddings.shape[0]
    if n_tokens <= 1:
        return np.zeros(0, dtype=np.float64)

    num_gaps = n_tokens - 1
    structural = np.zeros(num_gaps, dtype=np.float64)
    semantic = np.zeros(num_gaps, dtype=np.float64)
    entity_pen = np.zeros(num_gaps, dtype=np.float64)

    # --- Structural scores ------------------------------------------------
    # Identify the character offset where each paragraph starts in *text*.
    running_offset = 0
    para_char_starts: list[int] = []
    for para in paragraphs:
        idx = text.find(para.text, running_offset)
        if idx == -1:
            # Fallback: approximate.
            idx = running_offset
        para_char_starts.append(idx)
        running_offset = idx + len(para.text)

    # Determine which paragraphs are section / subsection starts.
    seen_sections: set[str] = set()
    for i, para in enumerate(paragraphs):
        if i == 0:
            # First paragraph -- no boundary *before* it.
            seen_sections.add(para.section_title)
            continue

        char_start = para_char_starts[i]
        tok_idx = _map_char_offset_to_token(char_start, token_to_char)
        gap_idx = max(0, tok_idx - 1)  # gap just before this token

        if gap_idx >= num_gaps:
            gap_idx = num_gaps - 1

        # Section-level heuristic.
        prev_para = paragraphs[i - 1]
        if para.section_title != prev_para.section_title and para.section_level == 0:
            structural[gap_idx] = max(structural[gap_idx], 1.0)
            seen_sections.add(para.section_title)
        elif para.section_title != prev_para.section_title and para.section_level > 0:
            structural[gap_idx] = max(structural[gap_idx], 0.7)
        else:
            # Plain paragraph boundary within the same (sub)section.
            structural[gap_idx] = max(structural[gap_idx], 0.4)

    # --- Semantic scores --------------------------------------------------
    half_w = max(1, window_size // 2)
    for g in range(num_gaps):
        left_start = max(0, g + 1 - half_w)
        left_end = g + 1
        right_start = g + 1
        right_end = min(n_tokens, g + 1 + half_w)

        if left_end <= left_start or right_end <= right_start:
            semantic[g] = 0.0
            continue

        left_mean = token_embeddings[left_start:left_end].mean(axis=0)
        right_mean = token_embeddings[right_start:right_end].mean(axis=0)

        sim = _cosine_similarity_vectors(left_mean, right_mean)
        semantic[g] = 1.0 - sim

    # --- Entity penalty ---------------------------------------------------
    if entity_spans:
        for char_start, char_end in entity_spans:
            tok_start = _map_char_offset_to_token(char_start, token_to_char)
            tok_end = _map_char_offset_to_token(char_end, token_to_char)
            # Penalise all gaps *inside* the entity span.
            for g in range(tok_start, min(tok_end, num_gaps)):
                entity_pen[g] = -0.5

    # --- Combine ----------------------------------------------------------
    scores = alpha1 * structural + alpha2 * semantic + entity_pen
    return scores


# ---------------------------------------------------------------------------
# Peak selection with min / max chunk constraints
# ---------------------------------------------------------------------------


def _select_boundaries(
    scores: np.ndarray,
    threshold: float,
    min_chunk: int,
    max_chunk: int,
    total_tokens: int,
) -> list[int]:
    """Select boundary positions from *scores* using peak detection.

    Parameters:
        scores: 1-D boundary score array (length ``total_tokens - 1``).
        threshold: Minimum score for a peak to be considered.
        min_chunk: Minimum tokens between consecutive boundaries.
        max_chunk: Maximum tokens between consecutive boundaries (force-split
            if exceeded).
        total_tokens: Total number of tokens in the document.

    Returns:
        Sorted list of token indices where chunks start (token 0 is implicit).
    """
    if scores.size == 0:
        return []

    # find_peaks with distance = min_chunk enforces minimum spacing.
    peak_indices, properties = find_peaks(
        scores, height=threshold, distance=min_chunk
    )

    boundaries = sorted(peak_indices.tolist())

    # Enforce max_chunk: insert forced splits where gaps are too large.
    final: list[int] = []
    prev = 0
    for b in boundaries:
        # Fill forced splits if gap is too large.
        while b - prev > max_chunk:
            forced = prev + max_chunk
            final.append(forced)
            prev = forced
        final.append(b)
        prev = b

    # Handle tail.
    while total_tokens - prev > max_chunk:
        forced = prev + max_chunk
        if forced < total_tokens:
            final.append(forced)
        prev = forced

    return sorted(set(final))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def structure_aware_chunk(
    article: ParsedArticle,
    model: SentenceTransformer,
    tokenizer,
    entity_spans: list[tuple[int, int]] | None = None,
    min_chunk: int = 128,
    max_chunk: int = 512,
    threshold: float = 0.3,
) -> list[Chunk]:
    """Produce structure-aware late-chunked embeddings for a parsed article.

    This is the core GraLC-RAG chunking strategy.  It combines three signals
    -- document structure, local semantic dissimilarity, and entity span
    preservation -- to decide where to split the token-level embedding
    sequence produced by a contextualised encoder.

    Parameters:
        article: A :class:`ParsedArticle` with populated paragraphs.
        model: A ``SentenceTransformer`` (first module must be a Transformer).
        tokenizer: The model's tokenizer (``model.tokenizer``).
        entity_spans: Optional list of ``(char_start, char_end)`` biomedical
            entity spans in the full text.
        min_chunk: Minimum tokens per chunk.
        max_chunk: Maximum tokens per chunk.
        threshold: Minimum boundary score for peak selection.

    Returns:
        List of :class:`Chunk` objects with pre-computed embeddings and rich
        metadata including *section_title* and *strategy*.
    """
    if not article.paragraphs:
        return []

    # Build the full document text by joining paragraphs.
    full_text = "\n\n".join(p.text for p in article.paragraphs)

    if not full_text.strip():
        return []

    # Compute contextualised token embeddings via sliding window.
    token_embeddings, token_to_char = _compute_token_embeddings(
        full_text, model, tokenizer, max_tokens=max_chunk, overlap=min_chunk
    )
    total_tokens = token_embeddings.shape[0]

    if total_tokens == 0:
        return []

    # Score every inter-token gap.
    scores = compute_boundary_scores(
        paragraphs=article.paragraphs,
        token_embeddings=token_embeddings,
        token_to_char=token_to_char,
        text=full_text,
        entity_spans=entity_spans,
    )

    # Select boundary peaks.
    boundaries = _select_boundaries(
        scores, threshold, min_chunk, max_chunk, total_tokens
    )

    # Build (start, end) spans.
    starts = [0] + boundaries
    ends = boundaries + [total_tokens]
    spans = list(zip(starts, ends))

    # Pre-compute a mapping from character offset -> section title for
    # metadata enrichment.
    para_char_ranges: list[tuple[int, int, str]] = []
    running = 0
    for para in article.paragraphs:
        idx = full_text.find(para.text, running)
        if idx == -1:
            idx = running
        para_char_ranges.append((idx, idx + len(para.text), para.section_title))
        running = idx + len(para.text)

    chunks: list[Chunk] = []
    for position, (s, e) in enumerate(spans):
        if s >= e:
            continue

        # Mean-pool token embeddings within this span.
        chunk_emb = token_embeddings[s:e].mean(axis=0)

        # Recover chunk text from character offsets.
        char_start = token_to_char[s]
        if e < total_tokens:
            char_end = token_to_char[e]
        else:
            char_end = len(full_text)
        chunk_text = full_text[char_start:char_end].strip()

        # Determine the dominant section title for this chunk.
        section_title = _dominant_section(char_start, char_end, para_char_ranges)

        chunks.append(
            Chunk(
                text=chunk_text,
                embedding=chunk_emb,
                metadata={
                    "strategy": "structure_aware",
                    "position": position,
                    "doc_id": article.pmid,
                    "section_title": section_title,
                    "token_span": (s, e),
                },
            )
        )

    return chunks


def _dominant_section(
    char_start: int,
    char_end: int,
    para_char_ranges: list[tuple[int, int, str]],
) -> str:
    """Return the section title that covers the most characters in the
    ``[char_start, char_end)`` range.
    """
    overlap_by_section: dict[str, int] = {}
    for p_start, p_end, title in para_char_ranges:
        ol_start = max(char_start, p_start)
        ol_end = min(char_end, p_end)
        if ol_end > ol_start:
            overlap_by_section[title] = (
                overlap_by_section.get(title, 0) + (ol_end - ol_start)
            )
    if not overlap_by_section:
        return ""
    return max(overlap_by_section, key=overlap_by_section.get)
