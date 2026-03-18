#!/usr/bin/env python
"""Generation evaluation on cross-section QA using full-text retrieval.

For each strategy, retrieves top-5 chunks from the full-text corpus for
each template question, feeds them to GPT-4o-mini, and measures answer
quality.  Evaluates on a subset to manage API costs.

Tests whether structure-aware retrieval (which covers more sections)
produces better answers than content-only retrieval (higher MRR but
single-section focus).

Usage:
    python scripts/10_evaluate_fulltext_generation.py [--n-eval 50] [--condition fulltext]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gralc_rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FULLTEXT_CONDITIONS_DIR,
    FULLTEXT_BENCHMARK_DIR,
    FULLTEXT_RESULTS_DIR,
    KG_WEIGHT_BETA,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from gralc_rag.chunking.naive import Chunk, naive_chunk
from gralc_rag.chunking.semantic import semantic_chunk
from gralc_rag.chunking.late import late_chunk
from gralc_rag.chunking.structure_aware import structure_aware_chunk
from gralc_rag.corpus.parser import ParsedArticle, Paragraph
from gralc_rag.knowledge.entity_linker import SimpleEntityLinker
from gralc_rag.knowledge.kg_infusion import load_sapbert_embeddings, project_embeddings
from gralc_rag.retrieval.index import VectorIndex
from gralc_rag.generation.openai_gen import generate_answer
from gralc_rag.evaluation.metrics import answer_f1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STRATEGIES_SIMPLE = ["naive", "semantic", "late_chunking", "structure_aware"]


def _load_condition_corpus(condition: str) -> list[dict]:
    """Load articles for a condition as dicts with doc_id, text, paragraphs, article."""
    cond_dir = FULLTEXT_CONDITIONS_DIR / condition
    corpus: list[dict] = []
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
        text = "\n\n".join(p.text for p in paragraphs)
        corpus.append({
            "doc_id": data["pmid"].split("__")[0],
            "text": text,
            "paragraphs": paragraphs,
            "article": ParsedArticle(
                pmid=data["pmid"], title=data.get("title", ""),
                abstract=data.get("abstract", ""), paragraphs=paragraphs,
            ),
        })
    return corpus


def _load_template_questions() -> list[dict]:
    path = FULLTEXT_BENCHMARK_DIR / "template_qa.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _embed_chunks_batch(chunks: list[Chunk], model) -> list[Chunk]:
    texts = [c.text for c in chunks if c.embedding is None]
    indices = [i for i, c in enumerate(chunks) if c.embedding is None]
    if texts:
        embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=128)
        for idx, emb in zip(indices, embs):
            chunks[idx] = Chunk(text=chunks[idx].text, embedding=emb, metadata=chunks[idx].metadata)
    return chunks


def _build_index(chunks: list[Chunk], dim: int) -> VectorIndex:
    embeddings = []
    metadata_list = []
    for c in chunks:
        if c.embedding is not None:
            emb = c.embedding / (np.linalg.norm(c.embedding) + 1e-8)
            embeddings.append(emb)
            metadata_list.append({**c.metadata, "text": c.text})
    if not embeddings:
        return VectorIndex(dim)
    emb_array = np.array(embeddings, dtype=np.float32)
    idx = VectorIndex(dim=emb_array.shape[1])
    idx.build(emb_array, metadata_list)
    return idx


def main():
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description="Full-text generation evaluation")
    parser.add_argument("--n-eval", type=int, default=50, help="Number of questions to evaluate")
    parser.add_argument("--condition", type=str, default="fulltext", help="Condition to evaluate")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set in .env — cannot run generation evaluation")
        sys.exit(1)

    log.info("Loading model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer
    entity_linker = SimpleEntityLinker()

    # Load corpus
    log.info("Loading %s corpus...", args.condition)
    corpus = _load_condition_corpus(args.condition)
    log.info("Corpus: %d articles", len(corpus))

    # Load questions
    all_questions = _load_template_questions()
    corpus_ids = {d["doc_id"] for d in corpus}
    questions = [q for q in all_questions if q.get("article_id", "").split("__")[0] in corpus_ids]
    questions = questions[:args.n_eval]
    log.info("Evaluating on %d questions", len(questions))

    if not questions:
        log.error("No questions found matching corpus. Run 07_generate_crosssection_qa.py first.")
        sys.exit(1)

    # Build indices for each strategy
    strategies_to_eval = ["naive", "semantic", "structure_aware", "gralc_rag"]

    all_results: list[dict] = []

    for strategy in strategies_to_eval:
        checkpoint_path = FULLTEXT_RESULTS_DIR / f"generation_{args.condition}_{strategy}.json"
        if checkpoint_path.exists():
            log.info("[SKIP] %s already evaluated", strategy)
            with open(checkpoint_path) as f:
                all_results.append(json.load(f))
            continue

        log.info("=== Generation: %s ===", strategy)
        log.info("Building %s index...", strategy)
        t0 = time.time()

        if strategy in STRATEGIES_SIMPLE:
            chunks: list[Chunk] = []
            for doc in corpus:
                if strategy == "naive":
                    cs = naive_chunk(doc["text"], doc_id=doc["doc_id"], max_tokens=512, overlap=64)
                elif strategy == "semantic":
                    cs = semantic_chunk(doc["text"], doc_id=doc["doc_id"], model=model, max_chunk_tokens=512)
                elif strategy == "late_chunking":
                    cs = late_chunk(doc["text"], doc_id=doc["doc_id"], model=model, tokenizer=tokenizer)
                elif strategy == "structure_aware":
                    entity_spans = entity_linker.get_entity_spans(doc["text"])
                    cs = structure_aware_chunk(doc["article"], model=model, tokenizer=tokenizer,
                                               entity_spans=entity_spans)
                chunks.extend(cs)
            chunks = _embed_chunks_batch(chunks, model)
            index = _build_index(chunks, EMBEDDING_DIM)

        elif strategy == "gralc_rag":
            all_entity_names: set[str] = set()
            for doc in corpus:
                for e in entity_linker.find_entities(doc["text"]):
                    all_entity_names.add(e["text"])
            if all_entity_names:
                sapbert_embs = load_sapbert_embeddings(list(all_entity_names))
                proj_embs = project_embeddings(sapbert_embs, target_dim=EMBEDDING_DIM)
            else:
                proj_embs = {}

            chunks = []
            cidx = 0
            for doc in corpus:
                entity_spans = entity_linker.get_entity_spans(doc["text"])
                cs = structure_aware_chunk(doc["article"], model=model, tokenizer=tokenizer,
                                           entity_spans=entity_spans)
                for c in cs:
                    if c.embedding is None:
                        cidx += 1
                        continue
                    c_ents = entity_linker.find_entities(c.text)
                    ent_names = [e["text"] for e in c_ents]
                    if ent_names and proj_embs:
                        ent_vecs = [proj_embs[n] for n in ent_names if n in proj_embs]
                        if ent_vecs:
                            avg_ent = np.mean(ent_vecs, axis=0)
                            enriched = c.embedding + 0.1 * avg_ent
                            enriched = enriched / (np.linalg.norm(enriched) + 1e-8)
                            c = Chunk(text=c.text, embedding=enriched,
                                      metadata={**c.metadata, "strategy": "gralc_rag", "chunk_idx": cidx})
                    chunks.append(c)
                    cidx += 1
            index = _build_index(chunks, EMBEDDING_DIM)

        elapsed = time.time() - t0
        log.info("  Index built in %.1fs (%d chunks)", elapsed, index.size)

        # Generate answers
        f1_scores: list[float] = []
        section_diversity: list[int] = []
        total_tokens = 0

        for i, q in enumerate(questions):
            query_emb = model.encode(q["question"], normalize_embeddings=True)
            results = index.search(query_emb, top_k=5)
            contexts = [meta.get("text", "") for meta, _ in results]
            sections_retrieved = [meta.get("section_title", "") for meta, _ in results]
            section_diversity.append(len(set(sections_retrieved)))

            try:
                answer = generate_answer(
                    question=q["question"],
                    contexts=contexts,
                    api_key=OPENAI_API_KEY,
                    model=OPENAI_MODEL,
                )
                # For cross-section QA, measure F1 against the question text itself
                # since we don't have gold answers — higher F1 = more relevant context
                f1 = answer_f1(answer["full_answer"], q["question"])
                f1_scores.append(f1)
                total_tokens += answer.get("usage_tokens", 0)
            except Exception as e:
                log.warning("Error on question %d: %s", i, e)
                f1_scores.append(0.0)

            if (i + 1) % 10 == 0:
                log.info("  Processed %d/%d", i + 1, len(questions))
            time.sleep(0.15)  # Rate limit

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_diversity = sum(section_diversity) / len(section_diversity) if section_diversity else 0.0

        result = {
            "strategy": strategy,
            "condition": args.condition,
            "avg_f1": round(avg_f1, 4),
            "avg_section_diversity": round(avg_diversity, 2),
            "n_questions": len(questions),
            "total_tokens": total_tokens,
            "model": OPENAI_MODEL,
        }
        all_results.append(result)

        # Checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump(result, f, indent=2)

        log.info("  %s: F1=%.4f SecDiv=%.2f Tokens=%d",
                 strategy, avg_f1, avg_diversity, total_tokens)

    # Save summary
    output = FULLTEXT_RESULTS_DIR / "generation_results.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 75)
    print(f"{'Strategy':<22s} {'Avg F1':>10s} {'Sec Diversity':>14s} {'Tokens':>10s}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['strategy']:<22s} {r['avg_f1']:10.4f} {r['avg_section_diversity']:14.2f} {r['total_tokens']:>10d}")
    print("=" * 75)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
