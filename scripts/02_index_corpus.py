"""Step 2: Index the corpus with all chunking strategies.

Produces a FAISS index + metadata for each strategy:
  - naive
  - semantic
  - late_chunking
  - structure_aware (GraLC-RAG without KG)
  - structure_aware_kg (GraLC-RAG with KG infusion)
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
    CORPUS_DIR,
    DATA_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    MAX_TOKENS_PER_CHUNK,
    CHUNK_OVERLAP,
    RESULTS_DIR,
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
    infuse_kg_into_tokens,
)
from gralc_rag.retrieval.index import VectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_articles() -> list[dict]:
    path = DATA_DIR / "parsed" / "articles.json"
    if not path.exists():
        log.error("No parsed articles found. Run 01_download_corpus.py first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dict_to_parsed_article(d: dict) -> ParsedArticle:
    paragraphs = [
        Paragraph(
            text=p["text"],
            section_title=p.get("section_title", ""),
            section_level=p.get("section_level", 0),
            position=p.get("position", i),
            citations=p.get("citations", []),
        )
        for i, p in enumerate(d.get("paragraphs", []))
    ]
    return ParsedArticle(
        pmid=d.get("pmid", ""),
        title=d.get("title", ""),
        abstract=d.get("abstract", ""),
        sections=[],
        paragraphs=paragraphs,
        references=d.get("references", []),
    )


def get_full_text(article: dict) -> str:
    parts = []
    if article.get("title"):
        parts.append(article["title"])
    if article.get("abstract"):
        parts.append(article["abstract"])
    for p in article.get("paragraphs", []):
        parts.append(p["text"])
    return "\n\n".join(parts)


def embed_chunks(chunks: list[Chunk], model) -> list[Chunk]:
    """Embed chunks that don't already have embeddings."""
    texts_to_embed = []
    indices = []
    for i, c in enumerate(chunks):
        if c.embedding is None:
            texts_to_embed.append(c.text)
            indices.append(i)

    if texts_to_embed:
        embeddings = model.encode(texts_to_embed, show_progress_bar=False,
                                   normalize_embeddings=True, batch_size=64)
        for idx, emb in zip(indices, embeddings):
            chunks[idx] = Chunk(
                text=chunks[idx].text,
                embedding=emb,
                metadata=chunks[idx].metadata,
            )
    return chunks


def build_index_for_strategy(
    strategy_name: str,
    all_chunks: list[Chunk],
) -> tuple[VectorIndex, list[dict]]:
    embeddings = []
    metadata_list = []
    for c in all_chunks:
        if c.embedding is not None:
            emb = c.embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
            metadata_list.append({**c.metadata, "text": c.text})

    if not embeddings:
        log.warning(f"No embeddings for strategy {strategy_name}")
        return VectorIndex(EMBEDDING_DIM), []

    emb_array = np.array(embeddings, dtype=np.float32)
    index = VectorIndex(dim=emb_array.shape[1])
    index.build(emb_array, metadata_list)
    return index, metadata_list


def main():
    from sentence_transformers import SentenceTransformer

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer

    log.info("Loading parsed articles...")
    articles = load_articles()
    log.info("Loaded %d articles", len(articles))

    entity_linker = SimpleEntityLinker()
    log.info("Loaded entity linker with %d MeSH terms", len(entity_linker._mesh_terms))

    index_dir = RESULTS_DIR / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)

    strategies = {}

    # --- 1. Naive chunking ---
    log.info("=== Naive Chunking ===")
    t0 = time.time()
    naive_chunks = []
    for art in articles:
        text = get_full_text(art)
        if len(text.strip()) < 50:
            continue
        chunks = naive_chunk(text, doc_id=art.get("pmid", ""), max_tokens=MAX_TOKENS_PER_CHUNK, overlap=CHUNK_OVERLAP)
        naive_chunks.extend(chunks)
    naive_chunks = embed_chunks(naive_chunks, model)
    idx, meta = build_index_for_strategy("naive", naive_chunks)
    idx.save(index_dir / "naive")
    strategies["naive"] = {"n_chunks": len(meta), "time": time.time() - t0}
    log.info("Naive: %d chunks in %.1fs", len(meta), strategies["naive"]["time"])

    # --- 2. Semantic chunking ---
    log.info("=== Semantic Chunking ===")
    t0 = time.time()
    sem_chunks = []
    for art in articles:
        text = get_full_text(art)
        if len(text.strip()) < 50:
            continue
        chunks = semantic_chunk(text, doc_id=art.get("pmid", ""), model=model)
        sem_chunks.extend(chunks)
    sem_chunks = embed_chunks(sem_chunks, model)
    idx, meta = build_index_for_strategy("semantic", sem_chunks)
    idx.save(index_dir / "semantic")
    strategies["semantic"] = {"n_chunks": len(meta), "time": time.time() - t0}
    log.info("Semantic: %d chunks in %.1fs", len(meta), strategies["semantic"]["time"])

    # --- 3. Late chunking ---
    log.info("=== Late Chunking ===")
    t0 = time.time()
    late_chunks = []
    for art in articles:
        text = get_full_text(art)
        if len(text.strip()) < 50:
            continue
        chunks = late_chunk(text, doc_id=art.get("pmid", ""), model=model, tokenizer=tokenizer)
        late_chunks.extend(chunks)
    idx, meta = build_index_for_strategy("late_chunking", late_chunks)
    idx.save(index_dir / "late_chunking")
    strategies["late_chunking"] = {"n_chunks": len(meta), "time": time.time() - t0}
    log.info("Late chunking: %d chunks in %.1fs", len(meta), strategies["late_chunking"]["time"])

    # --- 4. Structure-aware late chunking (GraLC-RAG without KG) ---
    log.info("=== Structure-Aware Late Chunking ===")
    t0 = time.time()
    struct_chunks = []
    for art in articles:
        parsed = dict_to_parsed_article(art)
        if not parsed.paragraphs:
            continue
        entity_spans = entity_linker.get_entity_spans(get_full_text(art))
        chunks = structure_aware_chunk(
            parsed, model=model, tokenizer=tokenizer,
            entity_spans=entity_spans,
            min_chunk=128, max_chunk=MAX_TOKENS_PER_CHUNK,
        )
        struct_chunks.extend(chunks)
    idx, meta = build_index_for_strategy("structure_aware", struct_chunks)
    idx.save(index_dir / "structure_aware")
    strategies["structure_aware"] = {"n_chunks": len(meta), "time": time.time() - t0}
    log.info("Structure-aware: %d chunks in %.1fs", len(meta), strategies["structure_aware"]["time"])

    # --- 5. Structure-aware + KG infusion (full GraLC-RAG) ---
    log.info("=== Structure-Aware + KG Infusion (GraLC-RAG) ===")
    t0 = time.time()

    # Collect all entity names across corpus for SapBERT encoding
    all_entity_names = set()
    for art in articles:
        text = get_full_text(art)
        entities = entity_linker.find_entities(text)
        for e in entities:
            all_entity_names.add(e["text"])

    log.info("Found %d unique entity names, loading SapBERT embeddings...", len(all_entity_names))
    if all_entity_names:
        sapbert_embs = load_sapbert_embeddings(list(all_entity_names))
        projected_embs = project_embeddings(sapbert_embs, target_dim=EMBEDDING_DIM)
    else:
        projected_embs = {}

    kg_chunks = []
    chunk_entity_map = {}  # chunk_index -> list of entity names
    chunk_idx = 0

    for art in articles:
        parsed = dict_to_parsed_article(art)
        if not parsed.paragraphs:
            continue
        full_text = get_full_text(art)
        entities = entity_linker.find_entities(full_text)
        entity_char_spans = [(e["start"], e["end"]) for e in entities]

        chunks = structure_aware_chunk(
            parsed, model=model, tokenizer=tokenizer,
            entity_spans=entity_char_spans,
            min_chunk=128, max_chunk=MAX_TOKENS_PER_CHUNK,
        )

        # KG infusion: enrich chunk embeddings with entity knowledge
        for c in chunks:
            if c.embedding is None:
                continue
            # Find entities within this chunk's text
            chunk_entities = entity_linker.find_entities(c.text)
            ent_names = [e["text"] for e in chunk_entities]
            chunk_entity_map[chunk_idx] = ent_names

            if ent_names and projected_embs:
                # Simple infusion: add weighted average of entity embeddings
                ent_embs = [projected_embs[n] for n in ent_names if n in projected_embs]
                if ent_embs:
                    avg_ent_emb = np.mean(ent_embs, axis=0)
                    enriched = c.embedding + 0.1 * avg_ent_emb
                    enriched = enriched / (np.linalg.norm(enriched) + 1e-8)
                    c = Chunk(text=c.text, embedding=enriched,
                              metadata={**c.metadata, "strategy": "gralc_rag"})

            kg_chunks.append(c)
            chunk_idx += 1

    idx, meta = build_index_for_strategy("gralc_rag", kg_chunks)
    idx.save(index_dir / "gralc_rag")
    strategies["gralc_rag"] = {"n_chunks": len(meta), "time": time.time() - t0}
    log.info("GraLC-RAG: %d chunks in %.1fs", len(meta), strategies["gralc_rag"]["time"])

    # Save chunk entity map for graph-guided retrieval
    with open(index_dir / "chunk_entity_map.json", "w") as f:
        json.dump({str(k): v for k, v in chunk_entity_map.items()}, f)

    # Save entity embeddings
    if projected_embs:
        np.savez(index_dir / "entity_embeddings.npz",
                 names=list(projected_embs.keys()),
                 embeddings=np.array(list(projected_embs.values())))

    # Summary
    log.info("\n=== Indexing Summary ===")
    for name, info in strategies.items():
        log.info("  %-20s: %5d chunks in %6.1fs", name, info["n_chunks"], info["time"])

    with open(RESULTS_DIR / "indexing_summary.json", "w") as f:
        json.dump(strategies, f, indent=2)

    log.info("All indices saved to %s", index_dir)


if __name__ == "__main__":
    main()
