"""LLM-based cross-section QA generator using GPT-4o.

Generates natural-language benchmark questions by prompting an LLM with the
article's section structure.  Each question is designed to require information
from two or more IMRaD sections, complementing the deterministic template
approach in :mod:`gralc_rag.benchmark.template_qa`.

This module is **optional** for development runs -- it returns an empty list
when ``OPENAI_API_KEY`` is not set, allowing the rest of the pipeline to
proceed with template-only questions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from gralc_rag.config import OPENAI_API_KEY
from gralc_rag.corpus.condition_builder import normalize_section_title
from gralc_rag.corpus.parser import ParsedArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a biomedical research question generator.  Given the structured outline
of a scientific article (title, abstract, and section contents), generate
questions that **require information from at least two different sections** to
answer correctly.

For each question, return a JSON array of objects with these fields:
- "question": the natural-language question text
- "required_sections": a list of section labels needed to answer (e.g.
  ["introduction", "results"]).  Must contain at least 2 items.
- "reasoning": a one-sentence explanation of why multiple sections are needed

Return ONLY the JSON array, no other text.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _article_to_prompt(article: ParsedArticle) -> str:
    """Format an article's structure as a prompt for the LLM."""
    parts: list[str] = [
        f"Title: {article.title}",
        f"Abstract: {article.abstract}",
        "",
        "Sections:",
    ]

    # Group paragraphs by section title to present a compact view
    current_section: str | None = None
    for para in article.paragraphs:
        if para.section_title != current_section:
            current_section = para.section_title
            parts.append(f"\n## {current_section}")
        # Truncate long paragraphs to keep prompt manageable
        text = para.text
        if len(text) > 500:
            text = text[:497] + "..."
        parts.append(text)

    return "\n".join(parts)


def _strip_code_fence(text: str) -> str:
    """Remove optional ```json ... ``` fencing from LLM responses."""
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _parse_llm_response(raw: str) -> list[dict[str, Any]]:
    """Parse and validate the LLM's JSON response."""
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON: %.200s", cleaned)
        return []

    if not isinstance(data, list):
        logger.warning("LLM response is not a JSON array")
        return []

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_llm_questions(
    articles: list[ParsedArticle],
    max_per_article: int = 3,
    model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Generate cross-section QA questions using an LLM.

    Calls the OpenAI API to produce natural-language questions that span
    multiple sections of each article.  Returns an empty list if
    ``OPENAI_API_KEY`` is not configured.

    Parameters
    ----------
    articles:
        Parsed articles to generate questions from.
    max_per_article:
        Maximum number of questions to request per article.
    model:
        OpenAI model identifier.

    Returns
    -------
    list[dict[str, Any]]:
        Each dict contains ``question``, ``article_id``, ``required_sections``,
        ``gold_paragraph_ids``, and ``source`` (set to ``"llm"``).
    """
    if not OPENAI_API_KEY:
        logger.info(
            "OPENAI_API_KEY not set -- skipping LLM question generation"
        )
        return []

    # Late import so the module can be loaded without openai installed
    from openai import OpenAI  # noqa: WPS433

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_questions: list[dict[str, Any]] = []

    for article in articles:
        user_prompt = (
            f"{_article_to_prompt(article)}\n\n"
            f"Generate up to {max_per_article} cross-section questions "
            f"for this article."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
        except Exception:
            logger.warning(
                "OpenAI API call failed for article %s",
                article.pmid,
                exc_info=True,
            )
            continue

        raw_content = response.choices[0].message.content or ""
        parsed = _parse_llm_response(raw_content)

        # Build paragraph index for gold_paragraph_ids
        section_para_map: dict[str, list[int]] = {}
        for para in article.paragraphs:
            label = normalize_section_title(para.section_title).value
            section_para_map.setdefault(label, []).append(para.position)

        for item in parsed[:max_per_article]:
            required = item.get("required_sections", [])
            # Normalize section names from the LLM
            normalized = []
            for sec_name in required:
                if isinstance(sec_name, str):
                    label = normalize_section_title(sec_name).value
                    if label != "other":
                        normalized.append(label)

            # Quality filter: must require 2+ sections
            if len(normalized) < 2:
                continue

            # Collect gold paragraph ids from the required sections
            gold_ids: list[int] = []
            for sec_label in normalized:
                gold_ids.extend(section_para_map.get(sec_label, []))

            all_questions.append({
                "question": item.get("question", ""),
                "article_id": article.pmid,
                "required_sections": normalized,
                "gold_paragraph_ids": sorted(gold_ids),
                "source": "llm",
            })

    logger.info(
        "Generated %d LLM questions from %d articles",
        len(all_questions),
        len(articles),
    )
    return all_questions
