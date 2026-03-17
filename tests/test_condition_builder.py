"""Tests for section normalizer and condition extraction."""

from gralc_rag.corpus.condition_builder import (
    normalize_section_title,
    IMRaDLabel,
    extract_conditions,
)
from gralc_rag.corpus.parser import ParsedArticle, Section, Paragraph


def test_normalize_introduction_variants():
    assert normalize_section_title("Introduction") == IMRaDLabel.INTRODUCTION
    assert normalize_section_title("1. Introduction") == IMRaDLabel.INTRODUCTION
    assert normalize_section_title("BACKGROUND") == IMRaDLabel.INTRODUCTION
    assert normalize_section_title("1 Background and Objectives") == IMRaDLabel.INTRODUCTION


def test_normalize_methods_variants():
    assert normalize_section_title("Methods") == IMRaDLabel.METHODS
    assert normalize_section_title("Materials and Methods") == IMRaDLabel.METHODS
    assert normalize_section_title("2. Experimental Design") == IMRaDLabel.METHODS
    assert normalize_section_title("Study Design and Procedures") == IMRaDLabel.METHODS


def test_normalize_results_variants():
    assert normalize_section_title("Results") == IMRaDLabel.RESULTS
    assert normalize_section_title("3. Findings") == IMRaDLabel.RESULTS


def test_normalize_results_discussion_merged():
    assert normalize_section_title("Results and Discussion") == IMRaDLabel.RESULTS_DISCUSSION
    assert normalize_section_title("Results & Discussion") == IMRaDLabel.RESULTS_DISCUSSION


def test_normalize_discussion_variants():
    assert normalize_section_title("Discussion") == IMRaDLabel.DISCUSSION
    assert normalize_section_title("4. Discussion") == IMRaDLabel.DISCUSSION
    assert normalize_section_title("Interpretation") == IMRaDLabel.DISCUSSION


def test_normalize_other():
    assert normalize_section_title("Acknowledgments") == IMRaDLabel.OTHER
    assert normalize_section_title("References") == IMRaDLabel.OTHER
    assert normalize_section_title("Supplementary Materials") == IMRaDLabel.OTHER


def test_extract_conditions_full_imrad():
    article = ParsedArticle(
        pmid="123", title="Test", abstract="Abstract text.",
        sections=[
            Section(title="Introduction", level=0, text="Intro text about the study."),
            Section(title="Methods", level=0, text="We used method X on sample Y."),
            Section(title="Results", level=0, text="We found that Z increased."),
            Section(title="Discussion", level=0, text="This confirms hypothesis A."),
        ],
        paragraphs=[
            Paragraph(text="Intro text about the study.", section_title="Introduction", section_level=0, position=0),
            Paragraph(text="We used method X on sample Y.", section_title="Methods", section_level=0, position=1),
            Paragraph(text="We found that Z increased.", section_title="Results", section_level=0, position=2),
            Paragraph(text="This confirms hypothesis A.", section_title="Discussion", section_level=0, position=3),
        ],
    )
    conditions = extract_conditions(article, min_fulltext_words=1)
    assert "intro" in conditions
    assert "partial" in conditions
    assert "fulltext" in conditions
    assert len(conditions["intro"].paragraphs) == 1
    assert conditions["intro"].paragraphs[0].section_title == "Introduction"
    assert len(conditions["partial"].paragraphs) == 3
    assert len(conditions["fulltext"].paragraphs) == 4


def test_extract_conditions_missing_intro():
    article = ParsedArticle(
        pmid="456", title="Test", abstract="Abstract.",
        sections=[
            Section(title="Methods", level=0, text="Method text."),
            Section(title="Results", level=0, text="Result text."),
        ],
        paragraphs=[
            Paragraph(text="Method text.", section_title="Methods", section_level=0, position=0),
            Paragraph(text="Result text.", section_title="Results", section_level=0, position=1),
        ],
    )
    conditions = extract_conditions(article)
    assert conditions.get("intro") is None
