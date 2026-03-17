#!/usr/bin/env python3
"""Download and parse a biomedical corpus for GraLC-RAG.

Steps:
    1. Load PubMedQA to obtain seed PMIDs.
    2. Expand seeds to related PMC articles via Entrez elink.
    3. Download full-text JATS XML from PMC.
    4. Parse downloaded articles into structured JSON.
    5. Save parsed articles to ``data/parsed/articles.json``.

Usage:
    python scripts/01_download_corpus.py --max-articles 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so the script can be run from the project root.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gralc_rag.config import CORPUS_DIR, DATA_DIR  # noqa: E402
from gralc_rag.corpus.downloader import (  # noqa: E402
    download_pmc_articles,
    download_pubmedqa,
    get_related_pmids,
)
from gralc_rag.corpus.parser import ParsedArticle  # noqa: E402

logger = logging.getLogger("01_download_corpus")


# ---------------------------------------------------------------------------
# JATS XML parsing helper
# ---------------------------------------------------------------------------

def parse_jats_xml(xml_path: Path) -> ParsedArticle | None:
    """Parse a PMC JATS XML file into a :class:`ParsedArticle`.

    Returns ``None`` if the file cannot be parsed.
    """
    from gralc_rag.corpus.parser import Paragraph

    try:
        from lxml import etree
    except ImportError:
        logger.error("lxml is required for XML parsing. Install it with: pip install lxml")
        return None

    try:
        tree = etree.parse(str(xml_path))
    except etree.XMLSyntaxError:
        logger.warning("XML parse error for %s", xml_path.name)
        return None

    root = tree.getroot()

    # Extract article ID.
    doc_id = xml_path.stem

    # Extract title.
    title_el = root.find(".//article-title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""

    # Extract abstract.
    abstract_el = root.find(".//abstract")
    abstract = "".join(abstract_el.itertext()).strip() if abstract_el is not None else ""

    # Extract body paragraphs with section context.
    paragraphs: list[Paragraph] = []
    body = root.find(".//body")
    if body is not None:
        position = 0
        for sec in body.iter("sec"):
            sec_title_el = sec.find("title")
            section_title = (
                "".join(sec_title_el.itertext()).strip()
                if sec_title_el is not None
                else ""
            )

            is_first_in_section = True
            for p_el in sec.findall("p"):
                text = "".join(p_el.itertext()).strip()
                if not text:
                    continue

                paragraphs.append(
                    Paragraph(
                        text=text,
                        section_title=section_title,
                        section_level=0,
                        position=position,
                    )
                )
                position += 1
                is_first_in_section = False

    # If no structured body was found, try to get all <p> tags.
    if not paragraphs:
        position = 0
        for p_el in root.iter("p"):
            text = "".join(p_el.itertext()).strip()
            if not text:
                continue
            paragraphs.append(
                Paragraph(text=text, section_title="", section_level=0, position=position)
            )
            position += 1

    # Extract references.
    references: list[str] = []
    for ref_el in root.iter("ref"):
        ref_text = "".join(ref_el.itertext()).strip()
        if ref_text:
            references.append(ref_text)

    return ParsedArticle(
        pmid=doc_id,
        title=title,
        abstract=abstract,
        paragraphs=paragraphs,
        references=references,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and parse a biomedical corpus for GraLC-RAG."
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=500,
        help="Maximum number of PMC articles to download (default: 500).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
    )

    # ---- Step 1: Load PubMedQA for seed PMIDs ----------------------------
    logger.info("Step 1/5: Loading PubMedQA dataset ...")
    records = download_pubmedqa()
    seed_pmids = [r["pubid"] for r in records if r.get("pubid")]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_pmids: list[str] = []
    for pmid in seed_pmids:
        if pmid not in seen:
            seen.add(pmid)
            unique_pmids.append(pmid)
    seed_pmids = unique_pmids
    logger.info("Obtained %d unique seed PMIDs from PubMedQA.", len(seed_pmids))

    # ---- Step 2: Expand to related PMC articles --------------------------
    logger.info("Step 2/5: Finding related PMC articles via Entrez elink ...")
    related_ids = get_related_pmids(seed_pmids, max_total=args.max_articles)
    logger.info("Expanded to %d PMC IDs.", len(related_ids))

    # ---- Step 3: Download full-text articles -----------------------------
    logger.info("Step 3/5: Downloading up to %d full-text articles ...", args.max_articles)
    saved_paths = download_pmc_articles(
        related_ids,
        save_dir=CORPUS_DIR,
        max_articles=args.max_articles,
    )
    logger.info("Downloaded %d articles to %s.", len(saved_paths), CORPUS_DIR)

    # ---- Step 4: Parse articles ------------------------------------------
    logger.info("Step 4/5: Parsing downloaded articles ...")
    parsed_articles: list[ParsedArticle] = []
    for xml_path in sorted(CORPUS_DIR.glob("*.xml")):
        article = parse_jats_xml(xml_path)
        if article is not None:
            parsed_articles.append(article)

    logger.info("Successfully parsed %d / %d articles.", len(parsed_articles), len(saved_paths))

    # ---- Step 5: Save parsed output --------------------------------------
    parsed_dir = DATA_DIR / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    output_path = parsed_dir / "articles.json"

    # Convert dataclasses to plain dicts for JSON serialisation.
    articles_data = []
    for article in parsed_articles:
        article_dict = asdict(article)
        articles_data.append(article_dict)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(articles_data, fh, ensure_ascii=False, indent=2)

    logger.info("Step 5/5: Saved parsed articles to %s.", output_path)

    # ---- Summary stats ---------------------------------------------------
    total_sections = set()
    total_paragraphs = 0
    for article in parsed_articles:
        for p in article.paragraphs:
            if p.section_title:
                total_sections.add((article.pmid, p.section_title))
            total_paragraphs += 1

    n_articles = len(parsed_articles)
    avg_sections = len(total_sections) / n_articles if n_articles else 0
    avg_paragraphs = total_paragraphs / n_articles if n_articles else 0

    print("\n" + "=" * 60)
    print("Corpus Download Summary")
    print("=" * 60)
    print(f"  Articles downloaded : {len(saved_paths)}")
    print(f"  Articles parsed     : {n_articles}")
    print(f"  Avg sections/article: {avg_sections:.1f}")
    print(f"  Avg paragraphs/article: {avg_paragraphs:.1f}")
    print(f"  Output              : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
