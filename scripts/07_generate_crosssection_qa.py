#!/usr/bin/env python
"""Generate cross-section QA benchmark from the full-text corpus.

Usage:
    python scripts/07_generate_crosssection_qa.py [--skip-llm] [--max-articles 200]

Outputs:
    data/fulltext/benchmark/template_qa.json
    data/fulltext/benchmark/llm_qa.json  (unless --skip-llm)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gralc_rag.config import FULLTEXT_CONDITIONS_DIR, FULLTEXT_BENCHMARK_DIR
from gralc_rag.corpus.parser import ParsedArticle, Paragraph
from gralc_rag.benchmark.template_qa import generate_template_questions
from gralc_rag.benchmark.llm_qa import generate_llm_questions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_condition_articles(condition: str = "fulltext") -> list[ParsedArticle]:
    """Load parsed articles from a condition directory, excluding PubMedQA articles."""
    cond_dir = FULLTEXT_CONDITIONS_DIR / condition
    articles: list[ParsedArticle] = []

    if not cond_dir.exists():
        log.warning("Condition directory does not exist: %s", cond_dir)
        return articles

    for json_path in sorted(cond_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Skip PubMedQA articles for QA generation to avoid circularity
        if data.get("is_pubmedqa", False):
            continue

        paragraphs = [
            Paragraph(
                text=p["text"],
                section_title=p["section_title"],
                section_level=p["section_level"],
                position=p["position"],
                citations=p.get("citations", []),
            )
            for p in data.get("paragraphs", [])
        ]

        article = ParsedArticle(
            pmid=data.get("pmid", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            paragraphs=paragraphs,
        )
        articles.append(article)

    return articles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cross-section QA benchmark",
    )
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-generated questions")
    parser.add_argument("--max-articles", type=int, default=200,
                        help="Max articles for LLM QA generation (default: 200)")
    args = parser.parse_args()

    log.info("Loading full-text articles for QA generation...")
    articles = _load_condition_articles("fulltext")
    log.info("Loaded %d articles (excluding PubMedQA)", len(articles))

    if not articles:
        log.error("No articles found. Run 06_build_fulltext_corpus.py first.")
        sys.exit(1)

    # Template QA
    log.info("Generating template-based questions...")
    template_questions = generate_template_questions(articles)
    template_path = FULLTEXT_BENCHMARK_DIR / "template_qa.json"
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(template_questions, f, ensure_ascii=False, indent=2)
    log.info("Saved %d template questions to %s", len(template_questions), template_path)

    # LLM QA (optional)
    if not args.skip_llm:
        log.info(
            "Generating LLM-based questions from %d articles...",
            min(len(articles), args.max_articles),
        )
        llm_articles = articles[: args.max_articles]
        llm_questions = generate_llm_questions(llm_articles)
        llm_path = FULLTEXT_BENCHMARK_DIR / "llm_qa.json"
        with open(llm_path, "w", encoding="utf-8") as f:
            json.dump(llm_questions, f, ensure_ascii=False, indent=2)
        log.info("Saved %d LLM questions to %s", len(llm_questions), llm_path)
    else:
        log.info("Skipping LLM QA generation (--skip-llm)")

    log.info("Done. Benchmark files in %s", FULLTEXT_BENCHMARK_DIR)


if __name__ == "__main__":
    main()
