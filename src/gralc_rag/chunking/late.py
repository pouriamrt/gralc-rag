"""Late chunking strategy (Günther et al. 2024).

The key idea: instead of encoding each chunk independently, first produce
token-level embeddings for the *entire* document (using a sliding window
to handle long texts), then partition the token sequence into chunks and
mean-pool each span.  This lets every chunk embedding benefit from full
document context.
"""

from __future__ import annotations

import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from gralc_rag.chunking.naive import Chunk

# ---------------------------------------------------------------------------
# Sliding-window token embeddings
# ---------------------------------------------------------------------------


def _compute_token_embeddings(
    text: str,
    model: SentenceTransformer,
    tokenizer,
    max_tokens: int = 512,
    overlap: int = 128,
) -> tuple[np.ndarray, list[int]]:
    """Produce a token-level embedding for every token in *text*.

    Uses a sliding window of *max_tokens* with *overlap* tokens of context.
    Overlapping regions are combined via a triangular weighting scheme that
    favours centre tokens.

    Returns:
        token_embeddings: ``(n_tokens, hidden_dim)`` array.
        token_to_char: mapping from token index → start character offset
            in *text* (useful for aligning chunks back to the source).
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )

    input_ids: torch.Tensor = encoding["input_ids"].squeeze(0)  # (total_tokens,)
    offsets: list[tuple[int, int]] = encoding["offset_mapping"].squeeze(0).tolist()
    total_tokens = input_ids.shape[0]
    token_to_char = [start for start, _end in offsets]

    # Access the underlying transformer model.
    transformer = model[0].auto_model
    device = next(transformer.parameters()).device

    # We'll accumulate weighted embeddings and total weights.
    hidden_dim: int | None = None
    weighted_sum: np.ndarray | None = None
    weight_sum: np.ndarray | None = None

    stride = max_tokens - overlap
    if stride <= 0:
        stride = max_tokens // 2  # safety fallback

    with torch.no_grad():
        for win_start in range(0, total_tokens, stride):
            win_end = min(win_start + max_tokens, total_tokens)
            win_ids = input_ids[win_start:win_end].unsqueeze(0).to(device)
            attention_mask = torch.ones_like(win_ids)

            outputs = transformer(input_ids=win_ids, attention_mask=attention_mask)
            # last_hidden_state → (1, win_len, hidden_dim)
            hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            win_len = hidden.shape[0]
            if hidden_dim is None:
                hidden_dim = hidden.shape[1]
                weighted_sum = np.zeros((total_tokens, hidden_dim), dtype=np.float64)
                weight_sum = np.zeros(total_tokens, dtype=np.float64)

            # Triangular weight: highest at centre, lowest at edges.
            weights = _triangular_weights(win_len)

            weighted_sum[win_start:win_end] += hidden * weights[:, None]
            weight_sum[win_start:win_end] += weights

            if win_end >= total_tokens:
                break

    # Normalise.
    safe_denom = np.maximum(weight_sum[:, None], 1e-12)
    token_embeddings = (weighted_sum / safe_denom).astype(np.float32)

    return token_embeddings, token_to_char


def _triangular_weights(length: int) -> np.ndarray:
    """Return a triangular weight vector peaking at the centre."""
    if length <= 1:
        return np.ones(length, dtype=np.float64)
    half = length / 2.0
    return 1.0 - np.abs(np.arange(length, dtype=np.float64) - half + 0.5) / half


# ---------------------------------------------------------------------------
# Boundary detection helpers
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"[.!?]\s")


def _auto_boundaries(
    token_to_char: list[int],
    text: str,
    target_size: int = 256,
) -> list[int]:
    """Create chunk boundaries at roughly every *target_size* tokens,
    aligned to the nearest sentence ending.

    Returns a sorted list of token indices where each boundary **starts** a
    new chunk (the first chunk implicitly starts at token 0).
    """
    total = len(token_to_char)
    if total <= target_size:
        return []

    # Pre-compute character positions of sentence endings.
    sentence_ends: set[int] = set()
    for m in _SENTENCE_END_RE.finditer(text):
        sentence_ends.add(m.start() + 1)  # char index right after punctuation

    boundaries: list[int] = []
    pos = target_size

    while pos < total:
        # Search ±target_size//4 around *pos* for the nearest sentence end.
        search_radius = target_size // 4
        best: int | None = None
        best_dist = search_radius + 1

        lo = max(0, pos - search_radius)
        hi = min(total, pos + search_radius)
        for t in range(lo, hi):
            char_pos = token_to_char[t]
            if char_pos in sentence_ends:
                dist = abs(t - pos)
                if dist < best_dist:
                    best = t
                    best_dist = dist

        split = best if best is not None else pos
        if split > 0 and split < total:
            boundaries.append(split)
        pos = split + target_size

    return sorted(set(boundaries))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def late_chunk(
    text: str,
    doc_id: str,
    model: SentenceTransformer,
    tokenizer,
    chunk_boundaries: list[int] | None = None,
    max_tokens: int = 512,
    overlap: int = 128,
) -> list[Chunk]:
    """Produce contextualised chunk embeddings via late chunking.

    Parameters:
        text: Full document text.
        doc_id: Unique document identifier.
        model: A ``SentenceTransformer`` whose first module is a Transformer.
        tokenizer: The model's tokenizer (``model.tokenizer``).
        chunk_boundaries: Optional list of token indices where chunks start
            (token 0 is always the implicit first boundary).  When ``None``,
            boundaries are placed every ~256 tokens aligned to sentence ends.
        max_tokens: Sliding window size for the transformer forward pass.
        overlap: Token overlap between consecutive windows.

    Returns:
        List of :class:`Chunk` objects **with embeddings already computed**
        (mean-pooled from contextualised token embeddings).
    """
    if not text or not text.strip():
        return []

    token_embeddings, token_to_char = _compute_token_embeddings(
        text, model, tokenizer, max_tokens=max_tokens, overlap=overlap
    )
    total_tokens = token_embeddings.shape[0]

    if total_tokens == 0:
        return []

    # Determine boundaries.
    if chunk_boundaries is None:
        boundaries = _auto_boundaries(token_to_char, text, target_size=256)
    else:
        boundaries = sorted(b for b in chunk_boundaries if 0 < b < total_tokens)

    # Build (start, end) spans.
    starts = [0] + boundaries
    ends = boundaries + [total_tokens]
    spans = list(zip(starts, ends))

    chunks: list[Chunk] = []
    for position, (s, e) in enumerate(spans):
        if s >= e:
            continue

        # Mean-pool token embeddings within this span.
        chunk_emb = token_embeddings[s:e].mean(axis=0)

        # Recover chunk text from character offsets.
        char_start = token_to_char[s]
        char_end = (
            token_to_char[e - 1] + len(_token_text_at(text, token_to_char, e - 1))
            if e <= total_tokens
            else len(text)
        )
        chunk_text = text[char_start:char_end].strip()

        chunks.append(
            Chunk(
                text=chunk_text,
                embedding=chunk_emb,
                metadata={
                    "strategy": "late_chunking",
                    "position": position,
                    "doc_id": doc_id,
                    "token_span": (s, e),
                },
            )
        )

    return chunks


def _token_text_at(text: str, token_to_char: list[int], idx: int) -> str:
    """Best-effort extraction of the text corresponding to token *idx*."""
    start = token_to_char[idx]
    # Next token start (or end of text) gives us the end.
    if idx + 1 < len(token_to_char):
        end = token_to_char[idx + 1]
    else:
        end = len(text)
    return text[start:end]
