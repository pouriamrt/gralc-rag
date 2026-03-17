"""Tests for cross-section retrieval metrics."""

from gralc_rag.evaluation.crosssection_metrics import (
    cross_section_recall,
    section_coverage_at_k,
)


def test_cross_section_recall_both_sections_hit():
    retrieved_sections = [["Methods", "Results", "Methods"]]
    required_sections = [["Methods", "Results"]]
    assert cross_section_recall(retrieved_sections, required_sections) == 1.0


def test_cross_section_recall_one_section_missing():
    retrieved_sections = [["Methods", "Methods", "Methods"]]
    required_sections = [["Methods", "Results"]]
    assert cross_section_recall(retrieved_sections, required_sections) == 0.0


def test_cross_section_recall_multiple_queries():
    retrieved = [
        ["Methods", "Results"],
        ["Methods", "Methods"],
    ]
    required = [
        ["Methods", "Results"],
        ["Methods", "Discussion"],
    ]
    assert cross_section_recall(retrieved, required) == 0.5


def test_section_coverage_at_k():
    retrieved_sections = [
        ["Methods", "Methods", "Results", "Introduction", "Discussion"],
    ]
    assert section_coverage_at_k(retrieved_sections, k=3) == 2.0
    assert section_coverage_at_k(retrieved_sections, k=5) == 4.0


def test_section_coverage_empty():
    assert section_coverage_at_k([], k=5) == 0.0
