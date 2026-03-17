"""Tests for template-based cross-section QA generation."""

from gralc_rag.benchmark.template_qa import generate_template_questions
from gralc_rag.corpus.parser import ParsedArticle, Paragraph


def _make_article() -> ParsedArticle:
    return ParsedArticle(
        pmid="TEST001",
        title="Test Article",
        abstract="Test abstract.",
        sections=[],
        paragraphs=[
            Paragraph(text="Diabetes affects millions of people worldwide.",
                      section_title="Introduction", section_level=0, position=0),
            Paragraph(text="We measured blood glucose using mass spectrometry.",
                      section_title="Methods", section_level=0, position=1),
            Paragraph(text="Blood glucose levels decreased by 30% in the treatment group.",
                      section_title="Results", section_level=0, position=2),
            Paragraph(text="The reduction in blood glucose confirms our hypothesis about diabetes treatment.",
                      section_title="Discussion", section_level=0, position=3),
        ],
    )


def test_generates_questions():
    article = _make_article()
    questions = generate_template_questions([article])
    assert len(questions) > 0


def test_question_structure():
    article = _make_article()
    questions = generate_template_questions([article])
    q = questions[0]
    assert "question" in q
    assert "article_id" in q
    assert "required_sections" in q
    assert "gold_paragraph_ids" in q
    assert isinstance(q["required_sections"], list)
    assert len(q["required_sections"]) >= 2


def test_questions_require_multiple_sections():
    article = _make_article()
    questions = generate_template_questions([article])
    for q in questions:
        assert len(q["required_sections"]) >= 2, \
            f"Question should require 2+ sections: {q['question']}"
