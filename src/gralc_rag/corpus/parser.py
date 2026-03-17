"""
JATS / NLM XML parser for PubMed Central full-text articles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A (possibly nested) document section."""

    title: str
    level: int  # 0 = root / top-level
    text: str
    subsections: list[Section] = field(default_factory=list)


@dataclass
class Paragraph:
    """A single paragraph with its provenance metadata."""

    text: str
    section_title: str
    section_level: int
    position: int  # 0-based index within the entire article
    citations: list[str] = field(default_factory=list)


@dataclass
class ParsedArticle:
    """Complete structured representation of a parsed PMC article."""

    pmid: str
    title: str
    abstract: str
    sections: list[Section] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _text_of(element: etree._Element | None) -> str:
    """Return the full concatenated text content of *element* (including tail
    text of children), stripping leading/trailing whitespace.  Returns ``""``
    if the element is ``None``.
    """
    if element is None:
        return ""
    # itertext() yields all text fragments depth-first.
    return " ".join("".join(element.itertext()).split())


def _extract_citations(para_el: etree._Element) -> list[str]:
    """Pull citation reference ids from ``<xref ref-type="bibr">`` elements
    inside a paragraph element.
    """
    cites: list[str] = []
    for xref in para_el.iter("xref"):
        if xref.get("ref-type") == "bibr":
            rid = xref.get("rid", "")
            if rid:
                cites.append(rid)
            # Also capture the display text (e.g. "[12]") as a fallback id.
            elif xref.text:
                cites.append(xref.text.strip())
    return cites


def _parse_section(
    sec_el: etree._Element,
    level: int,
) -> tuple[Section, list[Paragraph]]:
    """Recursively parse a ``<sec>`` element.

    Returns:
        A ``(Section, list[Paragraph])`` tuple.  The paragraph list is flat
        (depth-first) so callers can easily build the article-wide list.
    """
    # Section title --------------------------------------------------------
    title_el = sec_el.find("title")
    sec_title = _text_of(title_el)

    # Collect text from direct <p> children (not inside nested <sec>).
    direct_paragraphs: list[Paragraph] = []
    direct_text_parts: list[str] = []

    for child in sec_el:
        if child.tag == "p":
            p_text = _text_of(child)
            if p_text:
                direct_text_parts.append(p_text)
                direct_paragraphs.append(
                    Paragraph(
                        text=p_text,
                        section_title=sec_title,
                        section_level=level,
                        position=-1,  # will be set by caller
                        citations=_extract_citations(child),
                    )
                )

    # Recurse into child <sec> elements ------------------------------------
    subsections: list[Section] = []
    child_paragraphs: list[Paragraph] = []
    for child_sec in sec_el.findall("sec"):
        sub_section, sub_paras = _parse_section(child_sec, level + 1)
        subsections.append(sub_section)
        child_paragraphs.extend(sub_paras)

    section = Section(
        title=sec_title,
        level=level,
        text="\n\n".join(direct_text_parts),
        subsections=subsections,
    )

    all_paragraphs = direct_paragraphs + child_paragraphs
    return section, all_paragraphs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pmc_xml(xml_path: Path) -> ParsedArticle:
    """Parse a JATS/NLM XML file and return a :class:`ParsedArticle`.

    Handles common structural variations in PMC XML (e.g. missing body,
    missing title, multiple abstract paragraphs).
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    # Some PMC XMLs wrap everything in <pmc-articleset>; drill down.
    article_el = root.find(".//article")
    if article_el is None:
        article_el = root  # the root *is* the article

    front = article_el.find(".//front")
    body = article_el.find(".//body")
    back = article_el.find(".//back")

    # ------------------------------------------------------------------
    # PMID
    # ------------------------------------------------------------------
    pmid = ""
    if front is not None:
        for art_id in front.iter("article-id"):
            if art_id.get("pub-id-type") in ("pmid", "pmc"):
                pmid = (art_id.text or "").strip()
                break
    if not pmid:
        # Fall back to filename without extension.
        pmid = xml_path.stem

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    title = ""
    if front is not None:
        title_el = front.find(".//article-title")
        title = _text_of(title_el)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    abstract_parts: list[str] = []
    if front is not None:
        abstract_el = front.find(".//abstract")
        if abstract_el is not None:
            # May contain <p> children (structured abstract) or inline text.
            for p_el in abstract_el.iter("p"):
                txt = _text_of(p_el)
                if txt:
                    abstract_parts.append(txt)
            if not abstract_parts:
                txt = _text_of(abstract_el)
                if txt:
                    abstract_parts.append(txt)
    abstract = "\n".join(abstract_parts)

    # ------------------------------------------------------------------
    # Body sections & paragraphs
    # ------------------------------------------------------------------
    sections: list[Section] = []
    paragraphs: list[Paragraph] = []

    if body is not None:
        # Top-level <sec> elements
        top_secs = body.findall("sec")
        if top_secs:
            for sec_el in top_secs:
                section, sec_paras = _parse_section(sec_el, level=0)
                sections.append(section)
                paragraphs.extend(sec_paras)
        else:
            # No <sec> wrappers -- collect bare <p> elements.
            for p_el in body.findall("p"):
                txt = _text_of(p_el)
                if txt:
                    paragraphs.append(
                        Paragraph(
                            text=txt,
                            section_title="",
                            section_level=0,
                            position=-1,
                            citations=_extract_citations(p_el),
                        )
                    )

    # Assign sequential positions.
    for idx, para in enumerate(paragraphs):
        para.position = idx

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------
    references: list[str] = []
    ref_list_el = None
    if back is not None:
        ref_list_el = back.find(".//ref-list")
    # Sometimes <ref-list> lives outside <back>.
    if ref_list_el is None:
        ref_list_el = article_el.find(".//ref-list")

    if ref_list_el is not None:
        for ref_el in ref_list_el.findall("ref"):
            ref_text = _text_of(ref_el)
            if ref_text:
                references.append(ref_text)

    return ParsedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        sections=sections,
        paragraphs=paragraphs,
        references=references,
    )


def parse_all_articles(raw_dir: Path) -> list[ParsedArticle]:
    """Parse every ``*.xml`` file under *raw_dir*.

    Returns a list of :class:`ParsedArticle` instances.  Files that fail to
    parse are logged and skipped.
    """
    xml_files = sorted(raw_dir.glob("*.xml"))
    if not xml_files:
        logger.warning("No XML files found in %s", raw_dir)
        return []

    articles: list[ParsedArticle] = []
    for path in tqdm(xml_files, desc="Parsing PMC XML"):
        try:
            article = parse_pmc_xml(path)
            articles.append(article)
        except Exception:
            logger.warning("Failed to parse %s", path, exc_info=True)

    logger.info("Parsed %d / %d articles from %s", len(articles), len(xml_files), raw_dir)
    return articles
