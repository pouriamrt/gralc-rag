"""Lightweight biomedical entity linker based on MeSH dictionary matching.

Uses Entrez to download MeSH descriptor names, then performs longest-match-
first dictionary lookup to recognise entities in free text.  No heavy NLP
model is required, so this works on any Python >= 3.11 (including 3.13).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from gralc_rag.config import DATA_DIR
from gralc_rag.knowledge.umls_client import UMLSClient

logger = logging.getLogger(__name__)

_MESH_CACHE_PATH: Path = DATA_DIR / "mesh_terms.json"

# Minimum character length for a term to be considered during matching.
_MIN_ENTITY_LEN: int = 3


# ------------------------------------------------------------------
# MeSH term loading
# ------------------------------------------------------------------

def load_mesh_terms() -> dict[str, str]:
    """Return a mapping of *lowercase term* -> *MeSH UI*.

    On first call the function downloads MeSH descriptors via NCBI Entrez
    and caches the result to ``data/mesh_terms.json``.  Subsequent calls
    read from the cache.
    """
    if _MESH_CACHE_PATH.exists():
        with open(_MESH_CACHE_PATH, encoding="utf-8") as fh:
            cached = json.load(fh)
            if cached:
                logger.info("Loading cached MeSH terms from %s (%d terms)", _MESH_CACHE_PATH, len(cached))
                return cached

    logger.info("Downloading MeSH descriptors via Entrez ...")
    terms = _fetch_mesh_from_entrez()

    if not terms:
        logger.warning("Entrez MeSH download returned empty; falling back to built-in seed terms.")
        terms = _builtin_biomedical_terms()

    _MESH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MESH_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(terms, fh, ensure_ascii=False)
    logger.info("Cached %d MeSH terms to %s", len(terms), _MESH_CACHE_PATH)
    return terms


def _builtin_biomedical_terms() -> dict[str, str]:
    """A curated seed dictionary of common biomedical terms for entity linking.

    This is a fallback when MeSH download fails. Covers common diseases,
    drugs, proteins, genes, anatomical terms, and procedures.
    """
    terms: dict[str, str] = {}
    _seed = {
        # Diseases
        "cancer": "D009369", "diabetes": "D003920", "hypertension": "D006973",
        "asthma": "D001249", "obesity": "D009765", "stroke": "D020521",
        "alzheimer": "D000544", "parkinson": "D010300", "epilepsy": "D004827",
        "tuberculosis": "D014376", "malaria": "D008288", "hepatitis": "D006505",
        "pneumonia": "D011014", "sepsis": "D018805", "anemia": "D000740",
        "leukemia": "D007938", "lymphoma": "D008223", "melanoma": "D008545",
        "carcinoma": "D002277", "sarcoma": "D012509", "glioma": "D005910",
        "fibrosis": "D005355", "cirrhosis": "D008103", "arthritis": "D001168",
        "osteoporosis": "D010024", "atherosclerosis": "D050197",
        "myocardial infarction": "D009203", "heart failure": "D006333",
        "chronic kidney disease": "D051436", "copd": "D029424",
        "covid-19": "D000086382", "sars-cov-2": "D000086402",
        "influenza": "D007251", "hiv": "D015658", "aids": "D000163",
        "depression": "D003863", "schizophrenia": "D012559", "anxiety": "D001007",
        "dementia": "D003704", "migraine": "D008881", "psoriasis": "D011565",
        "eczema": "D004485", "lupus": "D008180", "crohn": "D003424",
        "ulcerative colitis": "D003093", "celiac disease": "D002446",
        "multiple sclerosis": "D009103", "amyotrophic lateral sclerosis": "D000690",
        # Drugs and therapeutics
        "metformin": "D008687", "insulin": "D007328", "aspirin": "D001241",
        "ibuprofen": "D007052", "paracetamol": "D000082", "acetaminophen": "D000082",
        "penicillin": "D010406", "amoxicillin": "D000658",
        "ciprofloxacin": "D002939", "doxycycline": "D004318",
        "methotrexate": "D008727", "cisplatin": "D002945",
        "doxorubicin": "D004317", "tamoxifen": "D013629",
        "warfarin": "D014859", "heparin": "D006493",
        "statin": "D019161", "atorvastatin": "D000069059",
        "omeprazole": "D009853", "prednisone": "D011241",
        "chemotherapy": "D004358", "immunotherapy": "D007167",
        "radiotherapy": "D011878", "antibiotics": "D000900",
        # Genes and proteins
        "p53": "D016159", "brca1": "D019398", "brca2": "D024682",
        "egfr": "D011958", "vegf": "D042461", "tnf": "D014409",
        "interleukin": "D007378", "interferon": "D007372",
        "hemoglobin": "D006454", "albumin": "D000418",
        "cytokine": "D016207", "antibody": "D000906",
        "antigen": "D000941", "receptor": "D011956",
        "kinase": "D010770", "protease": "D010447",
        "collagen": "D003094", "fibronectin": "D005353",
        # Anatomy
        "liver": "D008099", "kidney": "D007668", "lung": "D008168",
        "brain": "D001921", "heart": "D006321", "pancreas": "D010179",
        "colon": "D003106", "stomach": "D013270", "breast": "D001940",
        "prostate": "D011467", "ovary": "D010053", "thyroid": "D013961",
        "spleen": "D013154", "bone marrow": "D001853",
        # Lab / methods
        "pcr": "D016133", "elisa": "D004797", "western blot": "D015153",
        "flow cytometry": "D005434", "mass spectrometry": "D013058",
        "chromatography": "D002845", "sequencing": "D012150",
        "biopsy": "D001706", "mri": "D008279", "ct scan": "D014057",
        "ultrasound": "D014463", "microscopy": "D008853",
        # Clinical
        "mortality": "D009026", "survival": "D016019", "prognosis": "D011379",
        "diagnosis": "D003933", "biomarker": "D015415", "screening": "D008403",
        "prevalence": "D015995", "incidence": "D015994",
        "randomized controlled trial": "D016449", "meta-analysis": "D015201",
        "cohort study": "D015331", "case-control": "D016022",
        "placebo": "D010919", "double-blind": "D004311",
        # Molecular biology
        "dna": "D004247", "rna": "D012313", "mrna": "D012333",
        "protein": "D011506", "gene expression": "D015870",
        "mutation": "D009154", "polymorphism": "D011110",
        "methylation": "D019175", "phosphorylation": "D010766",
        "apoptosis": "D017209", "autophagy": "D001343",
        "inflammation": "D007249", "oxidative stress": "D018384",
        "angiogenesis": "D018919", "metastasis": "D009362",
        "cell proliferation": "D049109", "cell differentiation": "D002454",
        "stem cells": "D013234", "t cells": "D013601", "b cells": "D001402",
        "macrophage": "D008264", "neutrophil": "D009504",
        "platelet": "D001792", "erythrocyte": "D004912",
        "mitochondria": "D008928", "endoplasmic reticulum": "D004721",
        "ribosome": "D012270", "chromosome": "D002875",
        "genome": "D016678", "proteome": "D020543", "metabolome": "D055442",
        "microbiome": "D064307", "epigenetics": "D057890",
        "crispr": "D000071837", "sirna": "D034741",
    }
    for term, mesh_id in _seed.items():
        terms[term.lower()] = mesh_id
    return terms


def _fetch_mesh_from_entrez(
    max_terms: int = 5_000,
) -> dict[str, str]:
    """Download MeSH descriptor names from NCBI Entrez.

    Returns a dict mapping *lowercase name* -> *MeSH UI*.
    Uses esearch + esummary (JSON) to avoid XML binary-mode issues.
    """
    import time
    import requests as _requests

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    email = "gralc-rag@example.com"

    # Step 1: search for MeSH descriptors
    search_url = (
        f"{base}/esearch.fcgi?db=mesh&term=all[sb]"
        f"&retmax={max_terms}&retmode=json&email={email}"
    )
    resp = _requests.get(search_url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    id_list: list[str] = data.get("esearchresult", {}).get("idlist", [])
    logger.info("Entrez returned %d MeSH IDs", len(id_list))

    terms: dict[str, str] = {}

    # Step 2: fetch summaries in batches using esummary (returns JSON)
    batch_size = 100
    for start in range(0, len(id_list), batch_size):
        batch = id_list[start : start + batch_size]
        ids_str = ",".join(batch)
        summary_url = (
            f"{base}/esummary.fcgi?db=mesh&id={ids_str}"
            f"&retmode=json&email={email}"
        )
        try:
            resp = _requests.get(summary_url, timeout=120)
            resp.raise_for_status()
            summary_data = resp.json()
            result = summary_data.get("result", {})
            for uid in batch:
                rec = result.get(uid, {})
                name = rec.get("ds_meshterms", "")
                if isinstance(name, list):
                    for n in name:
                        if n and len(n) >= _MIN_ENTITY_LEN:
                            terms[n.lower()] = f"D{uid.zfill(6)}"
                elif isinstance(name, str) and name:
                    terms[name.lower()] = f"D{uid.zfill(6)}"
        except Exception as exc:
            logger.warning("Error fetching MeSH batch at %d: %s", start, exc)

        # Rate limit: 3 requests/second for NCBI
        time.sleep(0.35)

        if start % 2000 == 0 and start > 0:
            logger.info("  Fetched %d / %d MeSH terms so far ...", len(terms), len(id_list))

    return terms


# ------------------------------------------------------------------
# Entity linker
# ------------------------------------------------------------------

class SimpleEntityLinker:
    """Dictionary-based biomedical entity linker.

    Parameters
    ----------
    umls_client:
        Optional :class:`UMLSClient` instance used for CUI resolution.
    """

    def __init__(self, umls_client: UMLSClient | None = None) -> None:
        self._mesh_terms: dict[str, str] = load_mesh_terms()
        self._umls_client = umls_client

        # Pre-compile a regex alternation sorted longest-first for greedy
        # matching.  We escape every term so special characters are literal.
        sorted_terms = sorted(
            self._mesh_terms.keys(), key=len, reverse=True
        )
        # Only keep terms that meet the minimum length
        sorted_terms = [t for t in sorted_terms if len(t) >= _MIN_ENTITY_LEN]

        if sorted_terms:
            escaped = (re.escape(t) for t in sorted_terms)
            # Word-boundary anchors ensure we don't match inside other words.
            self._pattern: re.Pattern[str] | None = re.compile(
                r"\b(?:" + "|".join(escaped) + r")\b",
                re.IGNORECASE,
            )
        else:
            self._pattern = None
            logger.warning(
                "MeSH term dictionary is empty; entity linking will be a no-op."
            )

    # ------------------------------------------------------------------ #

    def find_entities(self, text: str) -> list[dict[str, Any]]:
        """Find biomedical entities in *text* via MeSH dictionary matching.

        Returns a list of dicts, each with keys:

        * ``text`` -- the matched surface form
        * ``start`` -- character offset start
        * ``end`` -- character offset end
        * ``mesh_id`` -- the MeSH descriptor UI
        """
        if self._pattern is None:
            return []

        entities: list[dict[str, Any]] = []
        for match in self._pattern.finditer(text):
            matched_text = match.group()
            mesh_id = self._mesh_terms.get(matched_text.lower(), "")
            entities.append(
                {
                    "text": matched_text,
                    "start": match.start(),
                    "end": match.end(),
                    "mesh_id": mesh_id,
                }
            )
        return entities

    def get_entity_spans(self, text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` character spans of recognised entities."""
        return [(e["start"], e["end"]) for e in self.find_entities(text)]
