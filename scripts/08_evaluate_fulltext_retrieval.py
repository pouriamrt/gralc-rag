#!/usr/bin/env python
"""Full-text retrieval evaluation across 6 strategies x 3 corpus conditions.

Evaluates naive, semantic, late_chunking, structure_aware, gralc_rag, and
gralc_rag_graph retrieval strategies on intro-only, partial (I+M+R), and
full-text conditions.  Both PubMedQA and template QA benchmarks are used.

Each strategy x condition pair is checkpointed independently so the evaluation
can be resumed after interruption without re-running completed combinations.

Usage:
    python scripts/08_evaluate_fulltext_retrieval.py [--conditions intro,partial,fulltext]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gralc_rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FULLTEXT_CONDITIONS_DIR,
    FULLTEXT_BENCHMARK_DIR,
    FULLTEXT_RESULTS_DIR,
    KG_WEIGHT_BETA,
    DATA_DIR,
)
from gralc_rag.chunking.naive import Chunk, naive_chunk
from gralc_rag.chunking.semantic import semantic_chunk
from gralc_rag.chunking.late import late_chunk
from gralc_rag.chunking.structure_aware import structure_aware_chunk
from gralc_rag.corpus.parser import ParsedArticle, Paragraph
from gralc_rag.knowledge.entity_linker import SimpleEntityLinker
from gralc_rag.knowledge.kg_infusion import load_sapbert_embeddings, project_embeddings
from gralc_rag.retrieval.index import VectorIndex
from gralc_rag.evaluation.metrics import mean_reciprocal_rank, recall_at_k, ndcg_at_k
from gralc_rag.evaluation.crosssection_metrics import (
    cross_section_recall,
    section_coverage_at_k,
)
from gralc_rag.evaluation.statistical import (
    paired_bootstrap_test,
    holm_bonferroni_correction,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

STRATEGIES = [
    "naive",
    "semantic",
    "late_chunking",
    "structure_aware",
    "gralc_rag",
    "gralc_rag_graph",
]
DEFAULT_CONDITIONS = ["intro", "partial", "fulltext"]
TOP_K = 20  # Retrieve top-20 to measure CS recall at k=5, 10, 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_condition_articles(condition: str) -> list[ParsedArticle]:
    """Load parsed articles from a condition directory."""
    cond_dir = FULLTEXT_CONDITIONS_DIR / condition
    articles: list[ParsedArticle] = []

    if not cond_dir.exists():
        log.warning("Condition directory does not exist: %s", cond_dir)
        return articles

    for json_path in sorted(cond_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

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


def _load_pubmedqa() -> list[dict]:
    """Load PubMedQA questions."""
    path = DATA_DIR / "pubmedqa" / "questions.json"
    if not path.exists():
        path = DATA_DIR / "pubmedqa" / "pqa_labeled.json"
    if not path.exists():
        log.warning("PubMedQA questions not found at %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_template_qa() -> list[dict]:
    """Load template-based cross-section QA questions."""
    path = FULLTEXT_BENCHMARK_DIR / "template_qa.json"
    if not path.exists():
        log.warning("Template QA not found at %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _filter_pubmedqa_for_corpus(
    questions: list[dict], corpus_pmids: set[str]
) -> list[dict]:
    """Keep only PubMedQA questions whose article exists in the corpus."""
    return [
        q
        for q in questions
        if str(q.get("pubid", q.get("id", ""))) in corpus_pmids
    ]


def _filter_template_qa_for_corpus(
    questions: list[dict], corpus_pmids: set[str]
) -> list[dict]:
    """Keep only template QA questions whose article exists in the corpus."""
    return [q for q in questions if q.get("article_id", "") in corpus_pmids]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _checkpoint_path(condition: str, strategy: str) -> Path:
    return FULLTEXT_RESULTS_DIR / f"{condition}_{strategy}.json"


def _checkpoint_path_pubmedqa(condition: str, strategy: str) -> Path:
    return FULLTEXT_RESULTS_DIR / f"{condition}_{strategy}_pubmedqa.json"


def _is_checkpointed(condition: str, strategy: str) -> bool:
    return _checkpoint_path(condition, strategy).exists()


def _is_checkpointed_pubmedqa(condition: str, strategy: str) -> bool:
    return _checkpoint_path_pubmedqa(condition, strategy).exists()


def _save_checkpoint(condition: str, strategy: str, result: dict) -> None:
    path = _checkpoint_path(condition, strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("Saved checkpoint: %s", path)


def _save_checkpoint_pubmedqa(
    condition: str, strategy: str, result: dict
) -> None:
    path = _checkpoint_path_pubmedqa(condition, strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("Saved PubMedQA checkpoint: %s", path)


def _load_checkpoint(condition: str, strategy: str) -> dict | None:
    path = _checkpoint_path(condition, strategy)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_checkpoint_pubmedqa(condition: str, strategy: str) -> dict | None:
    path = _checkpoint_path_pubmedqa(condition, strategy)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Chunking and indexing (follows patterns from 03_evaluate_retrieval_v2.py)
# ---------------------------------------------------------------------------


def _embed_chunks_batch(chunks: list[Chunk], model: Any) -> list[Chunk]:
    """Embed all chunks that don't yet have embeddings."""
    texts = [c.text for c in chunks if c.embedding is None]
    indices = [i for i, c in enumerate(chunks) if c.embedding is None]
    if texts:
        embs = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=128,
        )
        for idx, emb in zip(indices, embs):
            chunks[idx] = Chunk(
                text=chunks[idx].text,
                embedding=emb,
                metadata=chunks[idx].metadata,
            )
    return chunks


def _build_index(chunks: list[Chunk]) -> tuple[VectorIndex, list[dict]]:
    """Build a FAISS index from a list of chunks."""
    embeddings = []
    metadata_list = []
    for c in chunks:
        if c.embedding is not None:
            emb = c.embedding / (np.linalg.norm(c.embedding) + 1e-8)
            embeddings.append(emb)
            metadata_list.append({**c.metadata, "text": c.text})
    if not embeddings:
        return VectorIndex(EMBEDDING_DIM), []
    emb_array = np.array(embeddings, dtype=np.float32)
    idx = VectorIndex(dim=emb_array.shape[1])
    idx.build(emb_array, metadata_list)
    return idx, metadata_list


def _chunk_articles_naive(
    articles: list[ParsedArticle], model: Any
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        text = " ".join(p.text for p in article.paragraphs)
        cs = naive_chunk(text, doc_id=article.pmid, max_tokens=256, overlap=32)
        chunks.extend(cs)
    return _embed_chunks_batch(chunks, model)


def _chunk_articles_semantic(
    articles: list[ParsedArticle], model: Any
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        text = " ".join(p.text for p in article.paragraphs)
        cs = semantic_chunk(
            text, doc_id=article.pmid, model=model, max_chunk_tokens=256
        )
        chunks.extend(cs)
    return _embed_chunks_batch(chunks, model)


def _chunk_articles_late(
    articles: list[ParsedArticle], model: Any, tokenizer: Any
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        text = " ".join(p.text for p in article.paragraphs)
        cs = late_chunk(
            text, doc_id=article.pmid, model=model, tokenizer=tokenizer
        )
        chunks.extend(cs)
    return chunks  # late_chunk already embeds


def _chunk_articles_structure_aware(
    articles: list[ParsedArticle],
    model: Any,
    tokenizer: Any,
    entity_linker: SimpleEntityLinker,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        text = " ".join(p.text for p in article.paragraphs)
        entity_spans = entity_linker.get_entity_spans(text)
        cs = structure_aware_chunk(
            article,
            model=model,
            tokenizer=tokenizer,
            entity_spans=entity_spans,
            min_chunk=64,
            max_chunk=256,
        )
        chunks.extend(cs)
    return chunks  # structure_aware_chunk already embeds


def _chunk_articles_gralc_rag(
    articles: list[ParsedArticle],
    model: Any,
    tokenizer: Any,
    entity_linker: SimpleEntityLinker,
    proj_embs: dict[str, np.ndarray],
) -> tuple[list[Chunk], dict[int, list[str]]]:
    """Build chunks with KG infusion, returning chunks and chunk-entity map."""
    chunks: list[Chunk] = []
    chunk_entity_map: dict[int, list[str]] = {}
    cidx = 0

    for article in articles:
        text = " ".join(p.text for p in article.paragraphs)
        entity_spans = entity_linker.get_entity_spans(text)
        cs = structure_aware_chunk(
            article,
            model=model,
            tokenizer=tokenizer,
            entity_spans=entity_spans,
            min_chunk=64,
            max_chunk=256,
        )
        for c in cs:
            if c.embedding is None:
                cidx += 1
                continue

            c_ents = entity_linker.find_entities(c.text)
            ent_names = [e["text"] for e in c_ents]
            chunk_entity_map[cidx] = ent_names

            # KG infusion: additive fusion of projected SapBERT embeddings
            if ent_names and proj_embs:
                ent_vecs = [proj_embs[n] for n in ent_names if n in proj_embs]
                if ent_vecs:
                    avg_ent = np.mean(ent_vecs, axis=0)
                    enriched = c.embedding + 0.1 * avg_ent
                    enriched = enriched / (np.linalg.norm(enriched) + 1e-8)
                    c = Chunk(
                        text=c.text,
                        embedding=enriched,
                        metadata={
                            **c.metadata,
                            "strategy": "gralc_rag",
                            "chunk_idx": cidx,
                        },
                    )

            if "chunk_idx" not in c.metadata:
                c = Chunk(
                    text=c.text,
                    embedding=c.embedding,
                    metadata={**c.metadata, "chunk_idx": cidx},
                )
            chunks.append(c)
            cidx += 1

    return chunks, chunk_entity_map


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_retrieval(
    strategy_name: str,
    index: VectorIndex,
    questions: list[dict],
    model: Any,
    top_k: int = TOP_K,
    entity_linker: SimpleEntityLinker | None = None,
    entity_embeddings: dict[str, np.ndarray] | None = None,
    chunk_entity_map: dict[int, list[str]] | None = None,
    beta: float = KG_WEIGHT_BETA,
    is_template_qa: bool = False,
) -> dict[str, Any]:
    """Evaluate retrieval for a set of questions against an index.

    Returns a dict with MRR, Recall@1/3/5/10, nDCG@10, cross-section recall,
    section coverage@5, and per-query reciprocal ranks.
    """
    use_graph = (
        "graph" in strategy_name
        and entity_linker is not None
        and entity_embeddings is not None
        and chunk_entity_map is not None
    )

    rankings: list[int | None] = []
    per_query_rr: list[float] = []

    # For recall_at_k and ndcg_at_k metrics
    all_retrieved_ids: list[list[str]] = []
    all_relevant_ids: list[list[str]] = []

    # For cross-section metrics (template QA only)
    all_retrieved_sections: list[list[str]] = []
    all_required_sections: list[list[str]] = []

    n = len(questions)

    for q in questions:
        qtext = q["question"]

        # Determine gold document ID
        if is_template_qa:
            gold_id = str(q.get("article_id", ""))
        else:
            gold_id = str(q.get("pubid", q.get("id", "")))

        query_emb = model.encode(qtext, normalize_embeddings=True)
        results = index.search(query_emb, top_k=top_k)

        # Graph-guided re-ranking
        if use_graph and entity_linker and entity_embeddings:
            q_ents = entity_linker.find_entities(qtext)
            q_ent_names = [e["text"] for e in q_ents]

            reranked = []
            for meta, dense_score in results:
                chunk_idx = meta.get("chunk_idx", -1)
                c_ent_names = chunk_entity_map.get(chunk_idx, [])

                if q_ent_names and c_ent_names:
                    sims = []
                    for qe in q_ent_names:
                        if qe not in entity_embeddings:
                            continue
                        qe_emb = entity_embeddings[qe]
                        max_sim = 0.0
                        for ce in c_ent_names:
                            if ce in entity_embeddings:
                                ce_emb = entity_embeddings[ce]
                                sim = float(
                                    np.dot(qe_emb, ce_emb)
                                    / (
                                        np.linalg.norm(qe_emb)
                                        * np.linalg.norm(ce_emb)
                                        + 1e-8
                                    )
                                )
                                max_sim = max(max_sim, sim)
                        sims.append(max_sim)
                    kg_prox = sum(sims) / len(sims) if sims else 0.0
                else:
                    kg_prox = 0.0

                hybrid = beta * dense_score + (1 - beta) * kg_prox
                reranked.append((meta, hybrid))

            results = sorted(reranked, key=lambda x: x[1], reverse=True)[
                :top_k
            ]

        # Collect retrieved doc IDs and sections
        retrieved_doc_ids: list[str] = []
        retrieved_sections: list[str] = []
        first_rank: int | None = None

        for rank, (meta, score) in enumerate(results, 1):
            doc_id = str(meta.get("doc_id", ""))
            retrieved_doc_ids.append(doc_id)
            if is_template_qa:
                sec = meta.get("section_title", "")
                retrieved_sections.append(sec)
            if doc_id == gold_id and first_rank is None:
                first_rank = rank

        rankings.append(first_rank)
        rr = 1.0 / first_rank if first_rank is not None else 0.0
        per_query_rr.append(rr)

        all_retrieved_ids.append(retrieved_doc_ids)
        all_relevant_ids.append([gold_id])

        if is_template_qa:
            all_retrieved_sections.append(retrieved_sections)
            required = q.get("required_sections", [])
            all_required_sections.append(required)

    # Compute metrics
    mrr = mean_reciprocal_rank(rankings)

    r1_hits = sum(1 for r in rankings if r is not None and r <= 1)
    r3_hits = sum(1 for r in rankings if r is not None and r <= 3)
    r5_hits = sum(1 for r in rankings if r is not None and r <= 5)
    r10_hits = sum(1 for r in rankings if r is not None and r <= 10)

    ndcg = ndcg_at_k(all_retrieved_ids, all_relevant_ids, k=10)

    result: dict[str, Any] = {
        "strategy": strategy_name,
        "MRR": round(mrr, 4),
        "Recall@1": round(r1_hits / n, 4) if n > 0 else 0.0,
        "Recall@3": round(r3_hits / n, 4) if n > 0 else 0.0,
        "Recall@5": round(r5_hits / n, 4) if n > 0 else 0.0,
        "Recall@10": round(r10_hits / n, 4) if n > 0 else 0.0,
        "nDCG@10": round(ndcg, 4),
        "n_questions": n,
        "n_found": sum(1 for r in rankings if r is not None),
        "per_query_rr": per_query_rr,
    }

    # Cross-section metrics for template QA at multiple k values
    if is_template_qa and all_retrieved_sections:
        for k_val in (5, 10, 20):
            cs_r = cross_section_recall(
                all_retrieved_sections, all_required_sections, k=k_val
            )
            sec_c = section_coverage_at_k(all_retrieved_sections, k=k_val)
            result[f"cross_section_recall@{k_val}"] = round(cs_r, 4)
            result[f"section_coverage@{k_val}"] = round(sec_c, 4)
        # Keep backward-compatible keys
        result["cross_section_recall"] = result["cross_section_recall@5"]
        result["section_coverage@5"] = result["section_coverage@5"]

    return result


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------


def _run_significance_tests(
    conditions: list[str],
) -> dict[str, Any]:
    """Compare each strategy vs gralc_rag_graph per condition using bootstrap."""
    baseline = "gralc_rag_graph"
    competitors = [s for s in STRATEGIES if s != baseline]
    results: dict[str, Any] = {}

    for condition in conditions:
        baseline_result = _load_checkpoint(condition, baseline)
        if baseline_result is None:
            log.warning(
                "Skipping significance tests for %s: baseline not found",
                condition,
            )
            continue

        baseline_rr = baseline_result.get("per_query_rr", [])
        if not baseline_rr:
            log.warning(
                "Skipping significance tests for %s: no per-query RR in baseline",
                condition,
            )
            continue

        condition_results: dict[str, Any] = {}
        raw_p_values: list[float] = []
        competitor_names: list[str] = []

        for comp in competitors:
            comp_result = _load_checkpoint(condition, comp)
            if comp_result is None:
                log.warning(
                    "  Skipping %s vs %s (%s): checkpoint missing",
                    baseline,
                    comp,
                    condition,
                )
                continue

            comp_rr = comp_result.get("per_query_rr", [])
            if len(comp_rr) != len(baseline_rr):
                log.warning(
                    "  Skipping %s vs %s (%s): length mismatch (%d vs %d)",
                    baseline,
                    comp,
                    condition,
                    len(baseline_rr),
                    len(comp_rr),
                )
                continue

            test = paired_bootstrap_test(baseline_rr, comp_rr)
            condition_results[comp] = {
                "observed_diff": round(test["observed_diff"], 6),
                "p_value": round(test["p_value"], 6),
                "ci_lower": round(test["ci_lower"], 6),
                "ci_upper": round(test["ci_upper"], 6),
            }
            raw_p_values.append(test["p_value"])
            competitor_names.append(comp)

        # Holm-Bonferroni correction
        if raw_p_values:
            adjusted = holm_bonferroni_correction(raw_p_values)
            for comp_name, adj_p in zip(competitor_names, adjusted):
                condition_results[comp_name]["adjusted_p_value"] = round(
                    adj_p, 6
                )

        results[condition] = condition_results

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _aggregate_results(conditions: list[str]) -> list[dict]:
    """Load all checkpoints and aggregate into a summary list."""
    summary: list[dict] = []
    for condition in conditions:
        for strategy in STRATEGIES:
            result = _load_checkpoint(condition, strategy)
            if result is not None:
                # Exclude per_query_rr from summary (too large)
                entry = {
                    k: v
                    for k, v in result.items()
                    if k != "per_query_rr"
                }
                entry["condition"] = condition
                summary.append(entry)
    return summary


def _print_summary(results: list[dict]) -> None:
    """Print a formatted summary table."""
    header = (
        f"{'Condition':<12s} {'Strategy':<22s} {'MRR':>7s} {'R@1':>7s} "
        f"{'R@5':>7s} {'nDCG':>7s} "
        f"{'CSR@5':>7s} {'CSR@10':>8s} {'CSR@20':>8s} "
        f"{'SC@5':>7s} {'SC@10':>7s} {'SC@20':>7s} {'Found':>8s}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        def _fmt(key: str) -> str:
            v = r.get(key, "")
            return f"{v:7.4f}" if isinstance(v, (int, float)) else f"{'--':>7s}"
        print(
            f"{r.get('condition', ''):12s} {r['strategy']:<22s} "
            f"{r['MRR']:7.4f} {r['Recall@1']:7.4f} "
            f"{r['Recall@5']:7.4f} "
            f"{r.get('nDCG@10', 0):7.4f} "
            f"{_fmt('cross_section_recall@5')} {_fmt('cross_section_recall@10'):>8s} {_fmt('cross_section_recall@20'):>8s} "
            f"{_fmt('section_coverage@5')} {_fmt('section_coverage@10')} {_fmt('section_coverage@20')} "
            f"{r['n_found']:>4d}/{r['n_questions']}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-text retrieval evaluation with checkpointing "
        "and statistical significance tests.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated list of conditions to evaluate "
        "(default: intro,partial,fulltext)",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    log.info("Conditions to evaluate: %s", conditions)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer

    log.info("Initializing entity linker...")
    entity_linker = SimpleEntityLinker()
    log.info("Entity linker: %d terms", len(entity_linker._mesh_terms))

    # ------------------------------------------------------------------
    # Load questions
    # ------------------------------------------------------------------
    log.info("Loading PubMedQA questions...")
    pubmedqa_questions = _load_pubmedqa()
    log.info("Loaded %d PubMedQA questions", len(pubmedqa_questions))

    log.info("Loading template QA questions...")
    template_questions = _load_template_qa()
    log.info("Loaded %d template QA questions", len(template_questions))

    # ------------------------------------------------------------------
    # Evaluate each condition
    # ------------------------------------------------------------------
    for condition in conditions:
        log.info(
            "=" * 70,
        )
        log.info("=== Condition: %s ===", condition)
        log.info("=" * 70)

        # Load condition corpus
        log.info("Loading %s corpus...", condition)
        articles = _load_condition_articles(condition)
        if not articles:
            log.warning(
                "No articles found for condition '%s'. Skipping.", condition
            )
            continue
        log.info("Loaded %d articles for condition '%s'", len(articles), condition)

        corpus_pmids = {a.pmid for a in articles}

        # Filter questions to match corpus
        cond_pubmedqa = _filter_pubmedqa_for_corpus(
            pubmedqa_questions, corpus_pmids
        )
        cond_template = _filter_template_qa_for_corpus(
            template_questions, corpus_pmids
        )
        log.info(
            "Filtered questions: %d PubMedQA, %d template QA",
            len(cond_pubmedqa),
            len(cond_template),
        )

        if not cond_template and not cond_pubmedqa:
            log.warning(
                "No matching questions for condition '%s'. Skipping.",
                condition,
            )
            continue

        # ------------------------------------------------------------------
        # Evaluate each strategy
        # ------------------------------------------------------------------

        # Check which strategies need evaluation
        needs_template = {
            s: not _is_checkpointed(condition, s) for s in STRATEGIES
        }
        needs_pubmedqa = {
            s: not _is_checkpointed_pubmedqa(condition, s) for s in STRATEGIES
        }
        needs_any = {
            s: needs_template[s] or needs_pubmedqa[s] for s in STRATEGIES
        }

        # --- Naive ---
        if needs_any["naive"]:
            log.info("--- Strategy: naive ---")
            t0 = time.time()
            chunks = _chunk_articles_naive(articles, model)
            idx, _ = _build_index(chunks)
            log.info("  %d chunks, indexed in %.1fs", len(chunks), time.time() - t0)

            if needs_template["naive"] and cond_template:
                r = _evaluate_retrieval(
                    "naive", idx, cond_template, model, is_template_qa=True
                )
                _save_checkpoint(condition, "naive", r)
                log.info("  Template QA: MRR=%.4f R@5=%.4f", r["MRR"], r["Recall@5"])

            if needs_pubmedqa["naive"] and cond_pubmedqa:
                r = _evaluate_retrieval(
                    "naive", idx, cond_pubmedqa, model, is_template_qa=False
                )
                _save_checkpoint_pubmedqa(condition, "naive", r)
                log.info("  PubMedQA: MRR=%.4f R@5=%.4f", r["MRR"], r["Recall@5"])
        else:
            log.info("--- Strategy: naive --- (checkpointed, skipping)")

        # --- Semantic ---
        if needs_any["semantic"]:
            log.info("--- Strategy: semantic ---")
            t0 = time.time()
            chunks = _chunk_articles_semantic(articles, model)
            idx, _ = _build_index(chunks)
            log.info("  %d chunks, indexed in %.1fs", len(chunks), time.time() - t0)

            if needs_template["semantic"] and cond_template:
                r = _evaluate_retrieval(
                    "semantic", idx, cond_template, model, is_template_qa=True
                )
                _save_checkpoint(condition, "semantic", r)
                log.info("  Template QA: MRR=%.4f R@5=%.4f", r["MRR"], r["Recall@5"])

            if needs_pubmedqa["semantic"] and cond_pubmedqa:
                r = _evaluate_retrieval(
                    "semantic", idx, cond_pubmedqa, model, is_template_qa=False
                )
                _save_checkpoint_pubmedqa(condition, "semantic", r)
                log.info("  PubMedQA: MRR=%.4f R@5=%.4f", r["MRR"], r["Recall@5"])
        else:
            log.info("--- Strategy: semantic --- (checkpointed, skipping)")

        # --- Late Chunking ---
        if needs_any["late_chunking"]:
            log.info("--- Strategy: late_chunking ---")
            t0 = time.time()
            chunks = _chunk_articles_late(articles, model, tokenizer)
            idx, _ = _build_index(chunks)
            log.info("  %d chunks, indexed in %.1fs", len(chunks), time.time() - t0)

            if needs_template["late_chunking"] and cond_template:
                r = _evaluate_retrieval(
                    "late_chunking",
                    idx,
                    cond_template,
                    model,
                    is_template_qa=True,
                )
                _save_checkpoint(condition, "late_chunking", r)
                log.info("  Template QA: MRR=%.4f R@5=%.4f", r["MRR"], r["Recall@5"])

            if needs_pubmedqa["late_chunking"] and cond_pubmedqa:
                r = _evaluate_retrieval(
                    "late_chunking",
                    idx,
                    cond_pubmedqa,
                    model,
                    is_template_qa=False,
                )
                _save_checkpoint_pubmedqa(condition, "late_chunking", r)
                log.info("  PubMedQA: MRR=%.4f R@5=%.4f", r["MRR"], r["Recall@5"])
        else:
            log.info("--- Strategy: late_chunking --- (checkpointed, skipping)")

        # --- Structure-Aware ---
        if needs_any["structure_aware"]:
            log.info("--- Strategy: structure_aware ---")
            t0 = time.time()
            chunks = _chunk_articles_structure_aware(
                articles, model, tokenizer, entity_linker
            )
            idx, _ = _build_index(chunks)
            log.info("  %d chunks, indexed in %.1fs", len(chunks), time.time() - t0)

            if needs_template["structure_aware"] and cond_template:
                r = _evaluate_retrieval(
                    "structure_aware",
                    idx,
                    cond_template,
                    model,
                    is_template_qa=True,
                )
                _save_checkpoint(condition, "structure_aware", r)
                log.info("  Template QA: MRR=%.4f R@5=%.4f", r["MRR"], r["Recall@5"])

            if needs_pubmedqa["structure_aware"] and cond_pubmedqa:
                r = _evaluate_retrieval(
                    "structure_aware",
                    idx,
                    cond_pubmedqa,
                    model,
                    is_template_qa=False,
                )
                _save_checkpoint_pubmedqa(condition, "structure_aware", r)
                log.info("  PubMedQA: MRR=%.4f R@5=%.4f", r["MRR"], r["Recall@5"])
        else:
            log.info("--- Strategy: structure_aware --- (checkpointed, skipping)")

        # --- GraLC-RAG and GraLC-RAG + Graph ---
        # These two share the same index. Build once if either needs evaluation.
        gralc_needs = needs_any["gralc_rag"] or needs_any["gralc_rag_graph"]

        if gralc_needs:
            log.info("--- Strategy: gralc_rag / gralc_rag_graph ---")
            t0 = time.time()

            # Collect all entities for SapBERT projection
            all_entity_names: set[str] = set()
            for article in articles:
                text = " ".join(p.text for p in article.paragraphs)
                for e in entity_linker.find_entities(text):
                    all_entity_names.add(e["text"])
            log.info(
                "  %d unique entities, loading SapBERT...",
                len(all_entity_names),
            )

            if all_entity_names:
                sapbert_embs = load_sapbert_embeddings(
                    list(all_entity_names)
                )
                proj_embs = project_embeddings(
                    sapbert_embs, target_dim=EMBEDDING_DIM
                )
            else:
                proj_embs = {}

            chunks, chunk_entity_map = _chunk_articles_gralc_rag(
                articles, model, tokenizer, entity_linker, proj_embs
            )
            idx, _ = _build_index(chunks)
            log.info(
                "  %d chunks, indexed in %.1fs",
                len(chunks),
                time.time() - t0,
            )

            # Evaluate gralc_rag (no graph re-ranking)
            if needs_any["gralc_rag"]:
                if needs_template["gralc_rag"] and cond_template:
                    r = _evaluate_retrieval(
                        "gralc_rag",
                        idx,
                        cond_template,
                        model,
                        is_template_qa=True,
                    )
                    _save_checkpoint(condition, "gralc_rag", r)
                    log.info(
                        "  gralc_rag Template QA: MRR=%.4f R@5=%.4f",
                        r["MRR"],
                        r["Recall@5"],
                    )

                if needs_pubmedqa["gralc_rag"] and cond_pubmedqa:
                    r = _evaluate_retrieval(
                        "gralc_rag",
                        idx,
                        cond_pubmedqa,
                        model,
                        is_template_qa=False,
                    )
                    _save_checkpoint_pubmedqa(condition, "gralc_rag", r)
                    log.info(
                        "  gralc_rag PubMedQA: MRR=%.4f R@5=%.4f",
                        r["MRR"],
                        r["Recall@5"],
                    )

            # Evaluate gralc_rag_graph (with graph-guided re-ranking)
            if needs_any["gralc_rag_graph"]:
                if needs_template["gralc_rag_graph"] and cond_template:
                    r = _evaluate_retrieval(
                        "gralc_rag_graph",
                        idx,
                        cond_template,
                        model,
                        entity_linker=entity_linker,
                        entity_embeddings=proj_embs,
                        chunk_entity_map=chunk_entity_map,
                        beta=KG_WEIGHT_BETA,
                        is_template_qa=True,
                    )
                    _save_checkpoint(condition, "gralc_rag_graph", r)
                    log.info(
                        "  gralc_rag_graph Template QA: MRR=%.4f R@5=%.4f",
                        r["MRR"],
                        r["Recall@5"],
                    )

                if needs_pubmedqa["gralc_rag_graph"] and cond_pubmedqa:
                    r = _evaluate_retrieval(
                        "gralc_rag_graph",
                        idx,
                        cond_pubmedqa,
                        model,
                        entity_linker=entity_linker,
                        entity_embeddings=proj_embs,
                        chunk_entity_map=chunk_entity_map,
                        beta=KG_WEIGHT_BETA,
                        is_template_qa=False,
                    )
                    _save_checkpoint_pubmedqa(condition, "gralc_rag_graph", r)
                    log.info(
                        "  gralc_rag_graph PubMedQA: MRR=%.4f R@5=%.4f",
                        r["MRR"],
                        r["Recall@5"],
                    )
        else:
            log.info(
                "--- Strategy: gralc_rag / gralc_rag_graph --- "
                "(checkpointed, skipping)"
            )

    # ------------------------------------------------------------------
    # Statistical significance tests
    # ------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Running statistical significance tests...")
    significance = _run_significance_tests(conditions)
    sig_path = FULLTEXT_RESULTS_DIR / "significance_tests.json"
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(significance, f, indent=2)
    log.info("Saved significance tests to %s", sig_path)

    # Print significance summary
    for condition, tests in significance.items():
        log.info("  Condition: %s", condition)
        for comp, result in tests.items():
            adj_p = result.get("adjusted_p_value", result["p_value"])
            sig_marker = "*" if adj_p < 0.05 else ""
            log.info(
                "    gralc_rag_graph vs %s: diff=%.4f, p=%.4f (adj=%.4f)%s",
                comp,
                result["observed_diff"],
                result["p_value"],
                adj_p,
                sig_marker,
            )

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Aggregating results...")
    summary = _aggregate_results(conditions)
    summary_path = FULLTEXT_RESULTS_DIR / "fulltext_retrieval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary to %s", summary_path)

    # Also aggregate PubMedQA results
    pubmedqa_summary: list[dict] = []
    for condition in conditions:
        for strategy in STRATEGIES:
            result = _load_checkpoint_pubmedqa(condition, strategy)
            if result is not None:
                entry = {
                    k: v for k, v in result.items() if k != "per_query_rr"
                }
                entry["condition"] = condition
                pubmedqa_summary.append(entry)
    pubmedqa_summary_path = (
        FULLTEXT_RESULTS_DIR / "fulltext_retrieval_pubmedqa_summary.json"
    )
    with open(pubmedqa_summary_path, "w", encoding="utf-8") as f:
        json.dump(pubmedqa_summary, f, indent=2)
    log.info("Saved PubMedQA summary to %s", pubmedqa_summary_path)

    # Print summary tables
    if summary:
        print("\n--- Template QA Results ---")
        _print_summary(summary)

    if pubmedqa_summary:
        print("\n--- PubMedQA Results ---")
        _print_summary(pubmedqa_summary)

    log.info("Done. All results in %s", FULLTEXT_RESULTS_DIR)


if __name__ == "__main__":
    main()
