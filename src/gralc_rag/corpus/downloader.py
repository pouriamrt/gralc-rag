"""
Corpus acquisition: PubMedQA dataset and PMC full-text articles.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from Bio import Entrez
from datasets import load_dataset
from tqdm import tqdm

from gralc_rag.config import CORPUS_DIR, PUBMEDQA_DIR, NCBI_API_KEY

logger = logging.getLogger(__name__)

Entrez.email = "gralcrag@research.org"

# Minimum interval between NCBI requests (seconds) for 3 req/s rate limit.
_NCBI_INTERVAL: float = 1.0 / 3.0

if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY
    _NCBI_INTERVAL = 1.0 / 10.0
    logger.info("NCBI API key configured: 10 req/sec rate limit")


# ---------------------------------------------------------------------------
# PubMedQA
# ---------------------------------------------------------------------------

def download_pubmedqa(save_dir: Path | None = None) -> list[dict]:
    """Download the PubMedQA *pqa_labeled* split via HuggingFace ``datasets``.

    Each returned dict contains:
        question, context, long_answer, final_decision

    The raw dataset is also persisted to *save_dir* as ``pqa_labeled.json``.
    """
    save_dir = save_dir or PUBMEDQA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_path = save_dir / "pqa_labeled.json"

    if cache_path.exists():
        logger.info("Loading cached PubMedQA from %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    logger.info("Downloading PubMedQA pqa_labeled split from HuggingFace …")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    records: list[dict] = []
    for row in ds:
        # The HuggingFace version stores contexts as a dict with "contexts"
        # and "labels" keys; we flatten to a single joined string.
        context_parts = row.get("context", {})
        if isinstance(context_parts, dict):
            texts = context_parts.get("contexts", [])
            context_text = "\n".join(texts) if texts else ""
        elif isinstance(context_parts, str):
            context_text = context_parts
        else:
            context_text = str(context_parts)

        records.append(
            {
                "question": row["question"],
                "context": context_text,
                "long_answer": row.get("long_answer", ""),
                "final_decision": row.get("final_decision", ""),
                "pubid": str(row.get("pubid", "")),
            }
        )

    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    logger.info("Saved %d PubMedQA records to %s", len(records), cache_path)
    return records


# ---------------------------------------------------------------------------
# PMC full-text download
# ---------------------------------------------------------------------------

def download_pmc_articles(
    pmids: list[str],
    save_dir: Path | None = None,
    max_articles: int = 1000,
) -> list[Path]:
    """Fetch full-text JATS XML from PMC for each *pmid*.

    Articles are saved to ``data/raw/{pmid}.xml``.  Already-downloaded files
    are skipped.  Returns the list of paths that were successfully saved.

    Rate-limited to ~3 requests / second.
    """
    save_dir = save_dir or CORPUS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    ids_to_fetch = pmids[:max_articles]
    saved_paths: list[Path] = []

    for pmid in tqdm(ids_to_fetch, desc="Downloading PMC articles"):
        dest = save_dir / f"{pmid}.xml"
        if dest.exists():
            saved_paths.append(dest)
            continue

        try:
            handle = Entrez.efetch(
                db="pmc",
                id=pmid,
                rettype="xml",
                retmode="xml",
            )
            xml_bytes: bytes = handle.read()
            handle.close()

            dest.write_bytes(xml_bytes)
            saved_paths.append(dest)
            logger.debug("Downloaded %s", pmid)
        except Exception:
            logger.warning("Failed to download PMC article %s", pmid, exc_info=True)

        # Respect NCBI rate limit
        time.sleep(_NCBI_INTERVAL)

    logger.info(
        "Downloaded %d / %d PMC articles to %s",
        len(saved_paths),
        len(ids_to_fetch),
        save_dir,
    )
    return saved_paths


# ---------------------------------------------------------------------------
# Related-article expansion
# ---------------------------------------------------------------------------

def get_related_pmids(
    seed_pmids: list[str],
    max_total: int = 1000,
) -> list[str]:
    """Use Entrez.elink to expand *seed_pmids* into related PMC IDs.

    Returns a de-duplicated list of PMC IDs (strings) capped at *max_total*.
    """
    related: set[str] = set(seed_pmids)

    # Process seeds in batches of 50 (Entrez guideline).
    batch_size = 50
    for start in tqdm(
        range(0, len(seed_pmids), batch_size),
        desc="Finding related PMIDs",
    ):
        if len(related) >= max_total:
            break

        batch = seed_pmids[start : start + batch_size]

        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=batch,
                linkname="pubmed_pmc",
            )
            link_results = Entrez.read(handle)
            handle.close()

            for result in link_results:
                link_sets = result.get("LinkSetDb", [])
                for link_set in link_sets:
                    for link in link_set.get("Link", []):
                        pmc_id = str(link["Id"])
                        related.add(pmc_id)
                        if len(related) >= max_total:
                            break
                    if len(related) >= max_total:
                        break
                if len(related) >= max_total:
                    break
        except Exception:
            logger.warning(
                "elink failed for batch starting at index %d",
                start,
                exc_info=True,
            )

        time.sleep(_NCBI_INTERVAL)

    result_list = sorted(related)[:max_total]
    logger.info(
        "Expanded %d seed PMIDs to %d related PMC IDs",
        len(seed_pmids),
        len(result_list),
    )
    return result_list


# ---------------------------------------------------------------------------
# PMC search
# ---------------------------------------------------------------------------

def search_pmc_articles(
    query: str = "biomedical research",
    max_results: int = 2000,
    exclude_pmids: set[str] | None = None,
) -> list[str]:
    """Search PMC for open-access articles matching *query*.

    Uses Entrez.esearch to find PMC article IDs.
    Returns a list of PMC ID strings, up to *max_results*.
    """
    all_ids: list[str] = []
    batch_size = 500
    exclude = exclude_pmids or set()

    for retstart in range(0, max_results, batch_size):
        try:
            handle = Entrez.esearch(
                db="pmc",
                term=f"{query} AND open access[filter]",
                retmax=min(batch_size, max_results - len(all_ids)),
                retstart=retstart,
                sort="relevance",
            )
            results = Entrez.read(handle)
            handle.close()

            ids = results.get("IdList", [])
            if not ids:
                break

            for pmcid in ids:
                sid = str(pmcid)
                if sid not in exclude:
                    all_ids.append(sid)
                if len(all_ids) >= max_results:
                    break
        except Exception:
            logger.warning("esearch failed at retstart=%d", retstart, exc_info=True)

        time.sleep(_NCBI_INTERVAL)
        if len(all_ids) >= max_results:
            break

    logger.info("Found %d PMC IDs for query '%s'", len(all_ids), query)
    return all_ids[:max_results]
