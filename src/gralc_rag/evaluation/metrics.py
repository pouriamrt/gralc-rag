"""Retrieval and generation evaluation metrics for GraLC-RAG.

All functions handle edge cases (empty inputs, ``None`` values) gracefully
and return ``0.0`` when no meaningful computation is possible.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def mean_reciprocal_rank(rankings: list[int | None]) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Parameters:
        rankings: For each query, the 1-indexed rank of the first relevant
            result, or ``None`` if no relevant result was found.

    Returns:
        MRR as a float in [0, 1].  Returns 0.0 for empty input.
    """
    if not rankings:
        return 0.0

    total = 0.0
    for rank in rankings:
        if rank is not None and rank > 0:
            total += 1.0 / rank

    return total / len(rankings)


def recall_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[list[str]],
    k: int,
) -> float:
    """Compute Recall@k averaged across queries.

    For each query, a hit is scored if *any* relevant ID appears in the
    top-k retrieved IDs.

    Parameters:
        retrieved_ids: Per-query ordered lists of retrieved document IDs.
        relevant_ids: Per-query lists of ground-truth relevant document IDs.
        k: Cutoff depth.

    Returns:
        Average recall in [0, 1].  Returns 0.0 when inputs are empty.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    n_queries = min(len(retrieved_ids), len(relevant_ids))
    hits = 0

    for i in range(n_queries):
        top_k = set(retrieved_ids[i][:k])
        gold = set(relevant_ids[i]) if relevant_ids[i] else set()
        if top_k & gold:
            hits += 1

    return hits / n_queries


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain up to position *k*."""
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)  # i+2 because positions are 1-indexed
    return score


def ndcg_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[list[str]],
    k: int,
) -> float:
    """Compute Normalised Discounted Cumulative Gain at *k* (binary relevance).

    Parameters:
        retrieved_ids: Per-query ordered lists of retrieved document IDs.
        relevant_ids: Per-query lists of ground-truth relevant document IDs.
        k: Cutoff depth.

    Returns:
        Average NDCG@k in [0, 1].  Returns 0.0 when inputs are empty.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    n_queries = min(len(retrieved_ids), len(relevant_ids))
    total_ndcg = 0.0

    for i in range(n_queries):
        gold = set(relevant_ids[i]) if relevant_ids[i] else set()
        if not gold:
            continue

        # Binary relevance vector for the retrieved list.
        rels = [1.0 if doc_id in gold else 0.0 for doc_id in retrieved_ids[i][:k]]

        dcg = _dcg(rels, k)

        # Ideal relevance: all relevant docs ranked first.
        ideal_rels = sorted(rels, reverse=True)
        # If fewer relevant docs than k, pad with the ideal (all gold at top).
        n_relevant_in_k = min(len(gold), k)
        ideal_rels_full = [1.0] * n_relevant_in_k + [0.0] * (k - n_relevant_in_k)
        idcg = _dcg(ideal_rels_full, k)

        if idcg > 0:
            total_ndcg += dcg / idcg

    return total_ndcg / n_queries


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------


def answer_f1(predicted: str, gold: str) -> float:
    """Token-level F1 between *predicted* and *gold* answer strings.

    Tokenisation is by whitespace after lowercasing.

    Returns:
        F1 score in [0, 1].  Returns 0.0 for empty inputs.
    """
    pred_tokens = predicted.lower().split()
    gold_tokens = gold.lower().split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = 0
    gold_remaining = list(gold_tokens)
    for token in pred_tokens:
        if token in gold_remaining:
            common += 1
            gold_remaining.remove(token)

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def accuracy(predictions: list[str], golds: list[str]) -> float:
    """Exact-match accuracy after lowercasing and stripping.

    Parameters:
        predictions: Predicted answer strings.
        golds: Gold-standard answer strings.

    Returns:
        Accuracy in [0, 1].  Returns 0.0 when inputs are empty.
    """
    if not predictions or not golds:
        return 0.0

    n = min(len(predictions), len(golds))
    correct = sum(
        1
        for pred, gold in zip(predictions[:n], golds[:n])
        if pred.strip().lower() == gold.strip().lower()
    )

    return correct / n
