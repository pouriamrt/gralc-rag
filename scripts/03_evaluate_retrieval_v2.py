"""Retrieval evaluation using PubMedQA abstracts as corpus.

For each question, the gold-standard context (abstract) is mixed into a pool
of all 1000 abstracts. The task: retrieve the correct abstract given only the
question text. This is the standard PubMedQA* evaluation protocol.

Each abstract is chunked with all strategies and indexed. Since abstracts are
short (~200 words), this runs fast even on CPU.
"""

from __future__ import annotations

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
    RESULTS_DIR,
    DATA_DIR,
    KG_WEIGHT_BETA,
)
from gralc_rag.chunking.naive import Chunk, naive_chunk
from gralc_rag.chunking.semantic import semantic_chunk
from gralc_rag.chunking.late import late_chunk
from gralc_rag.chunking.structure_aware import structure_aware_chunk
from gralc_rag.corpus.parser import ParsedArticle, Paragraph
from gralc_rag.knowledge.entity_linker import SimpleEntityLinker
from gralc_rag.knowledge.kg_infusion import (
    load_sapbert_embeddings,
    project_embeddings,
)
from gralc_rag.retrieval.index import VectorIndex
from gralc_rag.retrieval.dense import embed_query
from gralc_rag.evaluation.metrics import mean_reciprocal_rank, recall_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_pubmedqa() -> list[dict]:
    path = DATA_DIR / "pubmedqa" / "questions.json"
    if not path.exists():
        path = DATA_DIR / "pubmedqa" / "pqa_labeled.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def embed_chunks_batch(chunks: list[Chunk], model) -> list[Chunk]:
    texts = [c.text for c in chunks if c.embedding is None]
    indices = [i for i, c in enumerate(chunks) if c.embedding is None]
    if texts:
        embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=128)
        for idx, emb in zip(indices, embs):
            chunks[idx] = Chunk(text=chunks[idx].text, embedding=emb, metadata=chunks[idx].metadata)
    return chunks


def index_strategy(name: str, chunks: list[Chunk]) -> tuple[VectorIndex, list[dict]]:
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


def evaluate(
    name: str,
    index: VectorIndex,
    questions: list[dict],
    model,
    top_k: int = 10,
    entity_linker=None,
    entity_embeddings=None,
    chunk_entity_map=None,
    beta: float = 0.7,
) -> dict:
    rankings = []
    r1_hits, r3_hits, r5_hits, r10_hits = 0, 0, 0, 0
    n = len(questions)

    use_graph = (
        "graph" in name
        and entity_linker is not None
        and entity_embeddings is not None
        and chunk_entity_map is not None
    )

    for q in questions:
        qtext = q["question"]
        gold_id = str(q.get("pubid", q.get("id", "")))

        query_emb = model.encode(qtext, normalize_embeddings=True)
        results = index.search(query_emb, top_k=top_k)

        if use_graph and entity_linker and entity_embeddings:
            # Re-rank with KG proximity
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
                                sim = float(np.dot(qe_emb, ce_emb) / (
                                    np.linalg.norm(qe_emb) * np.linalg.norm(ce_emb) + 1e-8))
                                max_sim = max(max_sim, sim)
                        sims.append(max_sim)
                    kg_prox = sum(sims) / len(sims) if sims else 0.0
                else:
                    kg_prox = 0.0

                hybrid = beta * dense_score + (1 - beta) * kg_prox
                reranked.append((meta, hybrid))

            results = sorted(reranked, key=lambda x: x[1], reverse=True)[:top_k]

        # Check if gold document is in results
        first_rank = None
        for rank, (meta, score) in enumerate(results, 1):
            doc_id = str(meta.get("doc_id", ""))
            if doc_id == gold_id:
                if first_rank is None:
                    first_rank = rank
                break

        rankings.append(first_rank)
        if first_rank is not None:
            if first_rank <= 1: r1_hits += 1
            if first_rank <= 3: r3_hits += 1
            if first_rank <= 5: r5_hits += 1
            if first_rank <= 10: r10_hits += 1

    mrr = mean_reciprocal_rank(rankings)
    return {
        "strategy": name,
        "MRR": round(mrr, 4),
        "Recall@1": round(r1_hits / n, 4),
        "Recall@3": round(r3_hits / n, 4),
        "Recall@5": round(r5_hits / n, 4),
        "Recall@10": round(r10_hits / n, 4),
        "n_questions": n,
        "n_found": sum(1 for r in rankings if r is not None),
    }


def main():
    from sentence_transformers import SentenceTransformer

    log.info("Loading model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer

    log.info("Loading PubMedQA questions...")
    questions = load_pubmedqa()
    log.info("Loaded %d questions", len(questions))

    # Build corpus from PubMedQA abstracts
    log.info("Building retrieval corpus from PubMedQA abstracts...")
    abstracts = []
    for q in questions:
        ctx = q.get("context", "")
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        abstracts.append({
            "doc_id": str(q.get("pubid", q.get("id", ""))),
            "text": ctx,
            "question": q["question"],
        })
    log.info("Corpus: %d abstracts", len(abstracts))

    entity_linker = SimpleEntityLinker()
    log.info("Entity linker: %d terms", len(entity_linker._mesh_terms))

    all_results = []

    # === 1. Naive ===
    log.info("=== Strategy: naive ===")
    t0 = time.time()
    chunks = []
    for a in abstracts:
        cs = naive_chunk(a["text"], doc_id=a["doc_id"], max_tokens=256, overlap=32)
        chunks.extend(cs)
    chunks = embed_chunks_batch(chunks, model)
    idx, _ = index_strategy("naive", chunks)
    log.info("  %d chunks, indexed in %.1fs", len(chunks), time.time() - t0)
    r = evaluate("naive", idx, questions, model)
    all_results.append(r)
    log.info("  MRR=%.4f R@1=%.4f R@3=%.4f R@5=%.4f R@10=%.4f",
             r["MRR"], r["Recall@1"], r["Recall@3"], r["Recall@5"], r["Recall@10"])

    # === 2. Semantic ===
    log.info("=== Strategy: semantic ===")
    t0 = time.time()
    chunks = []
    for a in abstracts:
        cs = semantic_chunk(a["text"], doc_id=a["doc_id"], model=model, max_chunk_tokens=256)
        chunks.extend(cs)
    chunks = embed_chunks_batch(chunks, model)
    idx, _ = index_strategy("semantic", chunks)
    log.info("  %d chunks, indexed in %.1fs", len(chunks), time.time() - t0)
    r = evaluate("semantic", idx, questions, model)
    all_results.append(r)
    log.info("  MRR=%.4f R@1=%.4f R@3=%.4f R@5=%.4f R@10=%.4f",
             r["MRR"], r["Recall@1"], r["Recall@3"], r["Recall@5"], r["Recall@10"])

    # === 3. Late Chunking ===
    log.info("=== Strategy: late_chunking ===")
    t0 = time.time()
    chunks = []
    for a in abstracts:
        cs = late_chunk(a["text"], doc_id=a["doc_id"], model=model, tokenizer=tokenizer)
        chunks.extend(cs)
    idx, _ = index_strategy("late_chunking", chunks)
    log.info("  %d chunks, indexed in %.1fs", len(chunks), time.time() - t0)
    r = evaluate("late_chunking", idx, questions, model)
    all_results.append(r)
    log.info("  MRR=%.4f R@1=%.4f R@3=%.4f R@5=%.4f R@10=%.4f",
             r["MRR"], r["Recall@1"], r["Recall@3"], r["Recall@5"], r["Recall@10"])

    # === 4. Structure-Aware (no KG) ===
    log.info("=== Strategy: structure_aware ===")
    t0 = time.time()
    chunks = []
    for a in abstracts:
        parsed = ParsedArticle(
            pmid=a["doc_id"], title="", abstract=a["text"],
            paragraphs=[Paragraph(text=a["text"], section_title="Abstract",
                                  section_level=0, position=0)],
        )
        entity_spans = entity_linker.get_entity_spans(a["text"])
        cs = structure_aware_chunk(
            parsed, model=model, tokenizer=tokenizer,
            entity_spans=entity_spans, min_chunk=64, max_chunk=256,
        )
        chunks.extend(cs)
    idx, _ = index_strategy("structure_aware", chunks)
    log.info("  %d chunks, indexed in %.1fs", len(chunks), time.time() - t0)
    r = evaluate("structure_aware", idx, questions, model)
    all_results.append(r)
    log.info("  MRR=%.4f R@1=%.4f R@3=%.4f R@5=%.4f R@10=%.4f",
             r["MRR"], r["Recall@1"], r["Recall@3"], r["Recall@5"], r["Recall@10"])

    # === 5. GraLC-RAG (structure + KG infusion) ===
    log.info("=== Strategy: gralc_rag ===")
    t0 = time.time()

    # Collect entities for SapBERT
    all_entity_names = set()
    for a in abstracts:
        for e in entity_linker.find_entities(a["text"]):
            all_entity_names.add(e["text"])
    log.info("  %d unique entities, loading SapBERT...", len(all_entity_names))

    if all_entity_names:
        sapbert_embs = load_sapbert_embeddings(list(all_entity_names))
        proj_embs = project_embeddings(sapbert_embs, target_dim=EMBEDDING_DIM)
    else:
        proj_embs = {}

    chunks = []
    chunk_entity_map = {}
    cidx = 0
    for a in abstracts:
        parsed = ParsedArticle(
            pmid=a["doc_id"], title="", abstract=a["text"],
            paragraphs=[Paragraph(text=a["text"], section_title="Abstract",
                                  section_level=0, position=0)],
        )
        entity_spans = entity_linker.get_entity_spans(a["text"])
        cs = structure_aware_chunk(
            parsed, model=model, tokenizer=tokenizer,
            entity_spans=entity_spans, min_chunk=64, max_chunk=256,
        )
        for c in cs:
            if c.embedding is None:
                cidx += 1
                continue
            c_ents = entity_linker.find_entities(c.text)
            ent_names = [e["text"] for e in c_ents]
            chunk_entity_map[cidx] = ent_names

            # KG infusion
            if ent_names and proj_embs:
                ent_vecs = [proj_embs[n] for n in ent_names if n in proj_embs]
                if ent_vecs:
                    avg_ent = np.mean(ent_vecs, axis=0)
                    enriched = c.embedding + 0.1 * avg_ent
                    enriched = enriched / (np.linalg.norm(enriched) + 1e-8)
                    c = Chunk(text=c.text, embedding=enriched,
                              metadata={**c.metadata, "strategy": "gralc_rag", "chunk_idx": cidx})

            if "chunk_idx" not in c.metadata:
                c = Chunk(text=c.text, embedding=c.embedding,
                          metadata={**c.metadata, "chunk_idx": cidx})
            chunks.append(c)
            cidx += 1

    idx, _ = index_strategy("gralc_rag", chunks)
    log.info("  %d chunks, indexed in %.1fs", len(chunks), time.time() - t0)

    # Evaluate without graph-guided retrieval
    r = evaluate("gralc_rag", idx, questions, model)
    all_results.append(r)
    log.info("  MRR=%.4f R@1=%.4f R@3=%.4f R@5=%.4f R@10=%.4f",
             r["MRR"], r["Recall@1"], r["Recall@3"], r["Recall@5"], r["Recall@10"])

    # === 6. GraLC-RAG + graph-guided retrieval ===
    r = evaluate("gralc_rag_graph", idx, questions, model,
                 entity_linker=entity_linker, entity_embeddings=proj_embs,
                 chunk_entity_map=chunk_entity_map, beta=KG_WEIGHT_BETA)
    all_results.append(r)
    log.info("  MRR=%.4f R@1=%.4f R@3=%.4f R@5=%.4f R@10=%.4f",
             r["MRR"], r["Recall@1"], r["Recall@3"], r["Recall@5"], r["Recall@10"])

    # === Summary ===
    output = RESULTS_DIR / "retrieval_results_v2.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 85)
    print(f"{'Strategy':<25s} {'MRR':>8s} {'R@1':>8s} {'R@3':>8s} {'R@5':>8s} {'R@10':>8s} {'Found':>8s}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['strategy']:<25s} {r['MRR']:8.4f} {r['Recall@1']:8.4f} "
              f"{r['Recall@3']:8.4f} {r['Recall@5']:8.4f} {r['Recall@10']:8.4f} "
              f"{r['n_found']:>4d}/{r['n_questions']}")
    print("=" * 85)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
