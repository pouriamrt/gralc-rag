"""Section normalizer and condition builder for fulltext evaluation.

Normalizes inconsistent PMC section titles to canonical IMRaD labels, then
extracts intro-only, partial (I+M+R), and full-text conditions from parsed
articles.  These conditions drive the ablation study comparing retrieval
quality across different levels of document completeness.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from gralc_rag.corpus.parser import Paragraph, ParsedArticle, Section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IMRaD label enum
# ---------------------------------------------------------------------------

class IMRaDLabel(Enum):
    """Canonical IMRaD section labels."""

    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    RESULTS_DISCUSSION = "results_discussion"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Pattern table — order matters: RESULTS_DISCUSSION checked first
# ---------------------------------------------------------------------------

_LABEL_PATTERNS: list[tuple[IMRaDLabel, re.Pattern[str]]] = [
    # Exclude known non-IMRaD headings early so their substrings don't
    # accidentally match an IMRaD pattern (e.g. "Supplementary Materials"
    # would otherwise match the METHODS "material" keyword).
    (
        IMRaDLabel.OTHER,
        re.compile(
            r"supplementary|acknowledg|reference|funding|"
            r"competing|conflict|author\s+contrib|abbreviation|"
            r"data\s+availab|ethics|appendix|supporting\s+info",
            re.IGNORECASE,
        ),
    ),
    # Must check merged "Results and/& Discussion" before individual labels
    (
        IMRaDLabel.RESULTS_DISCUSSION,
        re.compile(r"results?\s*(?:and|&)\s*discussion", re.IGNORECASE),
    ),
    (
        IMRaDLabel.INTRODUCTION,
        re.compile(r"intro|background|overview|objectives?\b", re.IGNORECASE),
    ),
    (
        IMRaDLabel.METHODS,
        re.compile(
            r"method|material|experimental|procedure|design|participant|setting",
            re.IGNORECASE,
        ),
    ),
    (
        IMRaDLabel.RESULTS,
        re.compile(r"result|finding|outcome", re.IGNORECASE),
    ),
    (
        IMRaDLabel.DISCUSSION,
        re.compile(r"discussion|interpretation|implication|limitation", re.IGNORECASE),
    ),
]

_LEADING_NUMBER_RE = re.compile(r"^\s*\d+[\.\):]?\s*")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_section_title(title: str) -> IMRaDLabel:
    """Map a raw section title to its canonical IMRaD label.

    The function strips leading section numbers (e.g. ``"1. "``, ``"2) "``)
    before matching against known keyword patterns.  The merged
    ``RESULTS_DISCUSSION`` pattern is tested first so that titles like
    *"Results and Discussion"* are not misclassified as plain ``RESULTS``.

    Parameters
    ----------
    title:
        The raw section heading string from a PMC article.

    Returns
    -------
    IMRaDLabel:
        The canonical label, or ``IMRaDLabel.OTHER`` if no pattern matches.
    """
    cleaned = _LEADING_NUMBER_RE.sub("", title).strip()

    for label, pattern in _LABEL_PATTERNS:
        if pattern.search(cleaned):
            return label

    return IMRaDLabel.OTHER


def has_imrad_structure(article: ParsedArticle, min_sections: int = 4) -> bool:
    """Return ``True`` if *article* contains at least *min_sections* distinct
    IMRaD section types (excluding ``OTHER``).
    """
    labels = {
        normalize_section_title(sec.title)
        for sec in article.sections
    }
    labels.discard(IMRaDLabel.OTHER)
    return len(labels) >= min_sections


def word_count(text: str) -> int:
    """Count words in *text* using whitespace splitting."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Condition extraction
# ---------------------------------------------------------------------------

_INTRO_LABELS = frozenset({IMRaDLabel.INTRODUCTION})

_PARTIAL_LABELS = frozenset({
    IMRaDLabel.INTRODUCTION,
    IMRaDLabel.METHODS,
    IMRaDLabel.RESULTS,
    IMRaDLabel.RESULTS_DISCUSSION,
})


def _build_condition_article(
    source: ParsedArticle,
    paragraphs: list[Paragraph],
) -> ParsedArticle:
    """Create a new :class:`ParsedArticle` with *paragraphs* reindexed.

    The returned article shares ``pmid``, ``title``, and ``abstract`` with the
    *source* but has fresh ``sections`` derived from the selected paragraphs
    and zero-based paragraph positions.
    """
    # Reindex paragraph positions
    reindexed: list[Paragraph] = []
    for idx, para in enumerate(paragraphs):
        reindexed.append(
            Paragraph(
                text=para.text,
                section_title=para.section_title,
                section_level=para.section_level,
                position=idx,
                citations=list(para.citations),
            )
        )

    # Derive sections from paragraphs (preserve order, deduplicate by title)
    seen_titles: set[str] = set()
    sections: list[Section] = []
    for para in reindexed:
        if para.section_title not in seen_titles:
            seen_titles.add(para.section_title)
            # Collect all paragraph texts for this section
            section_text = "\n\n".join(
                p.text for p in reindexed if p.section_title == para.section_title
            )
            sections.append(
                Section(
                    title=para.section_title,
                    level=para.section_level,
                    text=section_text,
                )
            )

    return ParsedArticle(
        pmid=source.pmid,
        title=source.title,
        abstract=source.abstract,
        sections=sections,
        paragraphs=reindexed,
    )


def extract_conditions(
    article: ParsedArticle,
    min_fulltext_words: int = 1000,
) -> dict[str, ParsedArticle | None]:
    """Extract intro-only, partial, and full-text conditions from *article*.

    Parameters
    ----------
    article:
        A fully parsed PMC article.
    min_fulltext_words:
        Minimum total word count across all paragraphs for the ``fulltext``
        condition to be produced.  Articles shorter than this threshold will
        have ``fulltext`` set to ``None``.

    Returns
    -------
    dict[str, ParsedArticle | None]:
        A mapping with keys ``"intro"``, ``"partial"``, and ``"fulltext"``.
        Any condition whose required sections are missing is set to ``None``.
    """
    conditions: dict[str, ParsedArticle | None] = {}

    # Classify each paragraph by its section label
    intro_paras: list[Paragraph] = []
    partial_paras: list[Paragraph] = []

    for para in article.paragraphs:
        label = normalize_section_title(para.section_title)
        if label in _INTRO_LABELS:
            intro_paras.append(para)
        if label in _PARTIAL_LABELS:
            partial_paras.append(para)

    # Sort partial paragraphs by their original position
    partial_paras.sort(key=lambda p: p.position)

    # --- intro condition ---
    if intro_paras:
        conditions["intro"] = _build_condition_article(article, intro_paras)
    else:
        conditions["intro"] = None
        logger.debug("Article %s has no introduction section", article.pmid)

    # --- partial condition (I + M + R) ---
    if partial_paras:
        conditions["partial"] = _build_condition_article(article, partial_paras)
    else:
        conditions["partial"] = None
        logger.debug("Article %s has no partial IMRaD sections", article.pmid)

    # --- fulltext condition ---
    total_words = sum(word_count(p.text) for p in article.paragraphs)
    if total_words >= min_fulltext_words:
        conditions["fulltext"] = _build_condition_article(
            article, list(article.paragraphs)
        )
    else:
        conditions["fulltext"] = None
        logger.debug(
            "Article %s has %d words (< %d), skipping fulltext condition",
            article.pmid,
            total_words,
            min_fulltext_words,
        )

    return conditions
