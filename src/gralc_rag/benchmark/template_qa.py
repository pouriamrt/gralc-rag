"""Template-based cross-section QA generator.

Generates benchmark questions that require information from two or more IMRaD
sections to answer correctly.  Each template pairs a phrase extracted from one
section with a question frame that references another section, ensuring the
retrieval system must surface paragraphs from multiple parts of the article.
"""

from __future__ import annotations

import logging
from typing import Any

from gralc_rag.corpus.condition_builder import IMRaDLabel, normalize_section_title
from gralc_rag.corpus.parser import Paragraph, ParsedArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------
# Each template specifies:
#   - template_type: a short identifier
#   - source_section: where the fill-in phrase is extracted from
#   - required_labels: the IMRaD labels the question spans
#   - template: the question string with a single {phrase} placeholder

_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_type": "method_result",
        "source_section": IMRaDLabel.METHODS,
        "required_labels": [IMRaDLabel.METHODS, IMRaDLabel.RESULTS],
        "template": (
            "What was the outcome when {phrase} was applied, "
            "as reported in the results?"
        ),
    },
    {
        "template_type": "intro_result",
        "source_section": IMRaDLabel.INTRODUCTION,
        "required_labels": [IMRaDLabel.INTRODUCTION, IMRaDLabel.RESULTS],
        "template": "Did the findings support the claim that {phrase}?",
    },
    {
        "template_type": "result_discussion",
        "source_section": IMRaDLabel.RESULTS,
        "required_labels": [IMRaDLabel.RESULTS, IMRaDLabel.DISCUSSION],
        "template": (
            "How did the authors interpret the finding that {phrase}?"
        ),
    },
    {
        "template_type": "method_discussion",
        "source_section": IMRaDLabel.METHODS,
        "required_labels": [IMRaDLabel.METHODS, IMRaDLabel.DISCUSSION],
        "template": (
            "What limitations of {phrase} were discussed by the authors?"
        ),
    },
    {
        "template_type": "intro_discussion",
        "source_section": IMRaDLabel.INTRODUCTION,
        "required_labels": [IMRaDLabel.INTRODUCTION, IMRaDLabel.DISCUSSION],
        "template": (
            "How does the discussion address the initial motivation "
            "regarding {phrase}?"
        ),
    },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_key_phrase(text: str, max_words: int = 15) -> str:
    """Extract a key phrase from *text* for template filling.

    Takes the first sentence (up to the first period) and truncates to
    *max_words* words.  The result is lowercased to read naturally inside
    a question template.
    """
    # Take first sentence
    first_sentence = text.split(".")[0].strip()
    words = first_sentence.split()
    phrase = " ".join(words[:max_words])
    # Lowercase so it fits grammatically in the template
    return phrase[0].lower() + phrase[1:] if phrase else phrase


def _get_section_paragraphs(
    article: ParsedArticle,
) -> dict[IMRaDLabel, list[Paragraph]]:
    """Group an article's paragraphs by their normalized IMRaD label."""
    section_map: dict[IMRaDLabel, list[Paragraph]] = {}
    for para in article.paragraphs:
        label = normalize_section_title(para.section_title)
        section_map.setdefault(label, []).append(para)
    return section_map


def _gold_paragraph_ids(
    section_map: dict[IMRaDLabel, list[Paragraph]],
    labels: list[IMRaDLabel],
) -> list[int]:
    """Collect the position indices of paragraphs belonging to *labels*."""
    ids: list[int] = []
    for label in labels:
        for para in section_map.get(label, []):
            ids.append(para.position)
    return sorted(ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_template_questions(
    articles: list[ParsedArticle],
    max_per_article: int = 5,
) -> list[dict[str, Any]]:
    """Generate cross-section QA questions from templates.

    For each article, the function iterates through the template list and
    fills in phrases extracted from the source section.  A template is only
    used when the article actually contains paragraphs in **all** of the
    required sections.

    Parameters
    ----------
    articles:
        Parsed articles to generate questions from.
    max_per_article:
        Maximum number of questions to produce per article.

    Returns
    -------
    list[dict[str, Any]]:
        Each dict contains ``question``, ``article_id``, ``required_sections``,
        ``gold_paragraph_ids``, and ``template_type``.
    """
    all_questions: list[dict[str, Any]] = []

    for article in articles:
        section_map = _get_section_paragraphs(article)
        article_questions: list[dict[str, Any]] = []

        for tmpl in _TEMPLATES:
            if len(article_questions) >= max_per_article:
                break

            source_label: IMRaDLabel = tmpl["source_section"]
            required_labels: list[IMRaDLabel] = tmpl["required_labels"]

            # Check that the article has paragraphs in all required sections
            if not all(section_map.get(lbl) for lbl in required_labels):
                continue

            # Extract phrase from the first paragraph in the source section
            source_paras = section_map[source_label]
            phrase = _extract_key_phrase(source_paras[0].text)
            if not phrase:
                continue

            question_text = tmpl["template"].format(phrase=phrase)
            gold_ids = _gold_paragraph_ids(section_map, required_labels)

            article_questions.append({
                "question": question_text,
                "article_id": article.pmid,
                "required_sections": [lbl.value for lbl in required_labels],
                "gold_paragraph_ids": gold_ids,
                "template_type": tmpl["template_type"],
            })

        all_questions.extend(article_questions)

    logger.info(
        "Generated %d template questions from %d articles",
        len(all_questions),
        len(articles),
    )
    return all_questions
