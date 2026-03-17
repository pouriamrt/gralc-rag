"""Tests for paired bootstrap and Holm-Bonferroni correction."""

import numpy as np
from gralc_rag.evaluation.statistical import (
    paired_bootstrap_test,
    holm_bonferroni_correction,
)


def test_bootstrap_identical_scores_high_pvalue():
    scores_a = [0.5, 0.6, 0.7, 0.8, 0.9]
    scores_b = [0.5, 0.6, 0.7, 0.8, 0.9]
    result = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=1000, seed=42)
    assert result["p_value"] > 0.3


def test_bootstrap_clearly_different():
    rng = np.random.RandomState(42)
    scores_a = rng.normal(0.5, 0.05, size=200).tolist()
    scores_b = rng.normal(0.8, 0.05, size=200).tolist()
    result = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=5000, seed=42)
    assert result["p_value"] < 0.05
    assert "ci_lower" in result
    assert "ci_upper" in result


def test_holm_bonferroni_basic():
    p_values = [0.01, 0.04, 0.03, 0.20]
    adjusted = holm_bonferroni_correction(p_values)
    assert len(adjusted) == 4
    assert adjusted[0] == min(0.01 * 4, 1.0)
    for orig, adj in zip(p_values, adjusted):
        assert adj >= orig
