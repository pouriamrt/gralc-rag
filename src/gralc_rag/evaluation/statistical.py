"""Statistical significance testing for retrieval metric comparisons.

Implements paired bootstrap resampling with Holm-Bonferroni correction
for multiple comparisons.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired bootstrap test for the difference in means.

    Tests H0: mean(scores_a) == mean(scores_b).

    Parameters:
        scores_a: Per-query scores for system A.
        scores_b: Per-query scores for system B (same length as *scores_a*).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys ``observed_diff``, ``p_value``, ``ci_lower``,
        ``ci_upper``.
    """
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    assert len(a) == len(b), "Score lists must have equal length"

    observed_diff = float(np.mean(a) - np.mean(b))
    n = len(a)

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        diffs[i] = np.mean(a[indices]) - np.mean(b[indices])

    centered = diffs - np.mean(diffs)
    p_value = float(np.mean(np.abs(centered) >= abs(observed_diff)))

    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction.

    Parameters:
        p_values: Unadjusted p-values from multiple hypothesis tests.

    Returns:
        Adjusted p-values in the original order, each >= the original.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0

    for rank, (orig_idx, pval) in enumerate(indexed):
        corrected = pval * (n - rank)
        corrected = min(corrected, 1.0)
        cummax = max(cummax, corrected)
        adjusted[orig_idx] = cummax

    return adjusted
