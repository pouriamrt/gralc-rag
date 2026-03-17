"""Cross-section retrieval metrics for full-text evaluation.

Measures whether retrieved chunks cover multiple required sections,
testing the structural awareness of different chunking strategies.
"""

from __future__ import annotations


def cross_section_recall(
    retrieved_sections: list[list[str]],
    required_sections: list[list[str]],
) -> float:
    """Compute cross-section recall averaged across queries.

    For each query, a "hit" is scored if the retrieved chunks collectively
    cover ALL required sections.  Binary per-query metric.

    Parameters:
        retrieved_sections: Per-query lists of section labels from retrieved
            chunks.
        required_sections: Per-query lists of section labels that must all
            appear among the retrieved chunks.

    Returns:
        Average in [0, 1].  Returns 0.0 for empty input.
    """
    if not retrieved_sections or not required_sections:
        return 0.0

    n = min(len(retrieved_sections), len(required_sections))
    hits = 0

    for i in range(n):
        covered = set(retrieved_sections[i])
        required = set(required_sections[i])
        if required and required.issubset(covered):
            hits += 1

    return hits / n


def section_coverage_at_k(
    retrieved_sections: list[list[str]],
    k: int = 5,
) -> float:
    """Average number of distinct sections in top-k retrieved chunks.

    Parameters:
        retrieved_sections: Per-query lists of section labels from retrieved
            chunks (ordered by retrieval rank).
        k: Cutoff depth.

    Returns:
        Average distinct-section count.  Returns 0.0 for empty input.
    """
    if not retrieved_sections:
        return 0.0

    total = 0.0
    for sections in retrieved_sections:
        distinct = len(set(sections[:k]))
        total += distinct

    return total / len(retrieved_sections)
