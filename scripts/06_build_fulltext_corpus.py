#!/usr/bin/env python
"""Build the full-text corpus: download, parse, filter, extract conditions.

Usage:
    python scripts/06_build_fulltext_corpus.py [--max-articles 2000] [--query "biomedical research"]

Outputs:
    data/fulltext/raw/           - Downloaded PMC XML files
    data/fulltext/parsed/        - Parsed JSON per article
    data/fulltext/conditions/    - intro/, partial/, fulltext/ JSON files
    data/fulltext/corpus_stats.json  - Corpus statistics and exclusion rates
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gralc_rag.config import (
    FULLTEXT_RAW_DIR,
    FULLTEXT_PARSED_DIR,
    FULLTEXT_CONDITIONS_DIR,
    FULLTEXT_DIR,
    PUBMEDQA_DIR,
)
from gralc_rag.corpus.downloader import search_pmc_articles, download_pmc_articles
from gralc_rag.corpus.parser import parse_pmc_xml
from gralc_rag.corpus.condition_builder import (
    extract_conditions,
    has_imrad_structure,
    word_count,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_pubmedqa_pmids() -> set[str]:
    """Load PubMedQA PMIDs so we can tag (not exclude) them."""
    for name in ("questions.json", "pqa_labeled.json"):
        path = PUBMEDQA_DIR / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {str(q.get("pubid", q.get("id", ""))) for q in data}
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-text corpus")
    parser.add_argument("--max-articles", type=int, default=2000,
                        help="Maximum number of articles to download (default: 2000)")
    parser.add_argument("--query", type=str, default="biomedical research",
                        help="PMC search query (default: 'biomedical research')")
    args = parser.parse_args()

    pubmedqa_pmids = _load_pubmedqa_pmids()
    log.info("Loaded %d PubMedQA PMIDs for tagging", len(pubmedqa_pmids))

    # Step 1: Search PMC
    log.info("Searching PMC for up to %d articles...", args.max_articles)
    pmc_ids = search_pmc_articles(
        query=args.query,
        max_results=args.max_articles,
    )
    log.info("Found %d PMC IDs", len(pmc_ids))

    # Step 2: Download
    log.info("Downloading articles...")
    paths = download_pmc_articles(
        pmc_ids, save_dir=FULLTEXT_RAW_DIR, max_articles=args.max_articles,
    )
    log.info("Downloaded %d articles", len(paths))

    # Step 3: Parse, filter, extract conditions
    log.info("Parsing articles...")
    stats: dict = {"total_downloaded": len(paths), "parsed": 0, "imrad_filtered": 0}
    condition_counts = {"intro": 0, "partial": 0, "fulltext": 0}

    for condition_name in ("intro", "partial", "fulltext"):
        (FULLTEXT_CONDITIONS_DIR / condition_name).mkdir(parents=True, exist_ok=True)

    for xml_path in paths:
        try:
            article = parse_pmc_xml(xml_path)
        except Exception:
            log.warning("Failed to parse %s", xml_path)
            continue

        # Save parsed article
        parsed_path = FULLTEXT_PARSED_DIR / f"{article.pmid}.json"
        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(asdict(article), f, ensure_ascii=False, indent=2)

        stats["parsed"] += 1

        # Filter: must have IMRaD structure and >= 1000 words
        full_text = " ".join(p.text for p in article.paragraphs)
        if not has_imrad_structure(article) or word_count(full_text) < 1000:
            stats["imrad_filtered"] += 1
            continue

        # Tag PubMedQA articles
        is_pubmedqa = article.pmid in pubmedqa_pmids

        # Extract conditions
        conditions = extract_conditions(article)

        for cond_name, cond_article in conditions.items():
            if cond_article is None:
                continue

            cond_data = asdict(cond_article)
            cond_data["is_pubmedqa"] = is_pubmedqa

            out_path = FULLTEXT_CONDITIONS_DIR / cond_name / f"{article.pmid}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cond_data, f, ensure_ascii=False, indent=2)

            condition_counts[cond_name] += 1

    # Save stats
    stats["condition_counts"] = condition_counts
    accepted = max(stats["parsed"] - stats["imrad_filtered"], 1)
    stats["exclusion_rates"] = {
        k: 1.0 - v / accepted for k, v in condition_counts.items()
    }
    stats_path = FULLTEXT_DIR / "corpus_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("=== Corpus Build Summary ===")
    log.info("Downloaded: %d", stats["total_downloaded"])
    log.info("Parsed: %d", stats["parsed"])
    log.info("IMRaD filtered out: %d", stats["imrad_filtered"])
    log.info(
        "Conditions: intro=%d, partial=%d, fulltext=%d",
        condition_counts["intro"],
        condition_counts["partial"],
        condition_counts["fulltext"],
    )
    log.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
