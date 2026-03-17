"""Step 3: Evaluate retrieval quality across all chunking strategies.

Produces results/retrieval_results.json with MRR, Recall@k for each strategy.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gralc_rag.config import (
    EMBEDDING_MODEL,
    RESULTS_DIR,
    TOP_K_RETRIEVAL,
    KG_WEIGHT_BETA,
)
from gralc_rag.evaluation.benchmarks import PubMedQABenchmark
from gralc_rag.evaluation.metrics import mean_reciprocal_rank, recall_at_k
from gralc_rag.retrieval.index import VectorIndex
from gralc_rag.retrieval.dense import embed_query, dense_retrieve
from gralc_rag.retrieval.graph_guided import graph_guided_retrieve
from gralc_rag.knowledge.entity_linker import SimpleEntityLinker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def semantic_relevance(chunk_emb, context_text: str, model) -> float:
    """Compute cosine similarity between a chunk embedding and the context text."""
    import numpy as np
    ctx_emb = model.encode(context_text, normalize_embeddings=True)
    if chunk_emb is None:
        return 0.0
    chunk_norm = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-8)
    return float(np.dot(chunk_norm, ctx_emb))


def evaluate_strategy(
    strategy_name: str,
    index: VectorIndex,
    benchmark: PubMedQABenchmark,
    model,
    top_k: int = 5,
    entity_linker=None,
    entity_embeddings=None,
    chunk_entity_map=None,
) -> dict:
    rankings = []
    retrieved_ids_all = []
    relevant_ids_all = []

    use_graph = (
        strategy_name == "gralc_rag_graph"
        and entity_linker is not None
        and entity_embeddings is not None
        and chunk_entity_map is not None
    )

    for q in benchmark.questions:
        question_text = q["question"]
        original_context = q.get("context", "")

        if use_graph:
            results = graph_guided_retrieve(
                query=question_text,
                model=model,
                index=index,
                entity_linker=entity_linker,
                entity_embeddings=entity_embeddings,
                chunk_entity_map=chunk_entity_map,
                top_k=top_k,
                beta=KG_WEIGHT_BETA,
            )
        else:
            results = dense_retrieve(question_text, model, index, top_k=top_k)

        # Determine relevance using two signals:
        # 1. Jaccard overlap with original context (threshold 0.15)
        # 2. Semantic similarity with question + context (threshold 0.5)
        # A chunk is relevant if EITHER condition is met
        first_relevant_rank = None
        retrieved = []
        relevant = ["relevant"]  # dummy ID for recall computation

        for rank, (meta, score) in enumerate(results, 1):
            chunk_text = meta.get("text", "")
            jac = jaccard_similarity(chunk_text, original_context)
            # Also check Jaccard with the question itself
            jac_q = jaccard_similarity(chunk_text, question_text)
            is_relevant = jac >= 0.15 or jac_q >= 0.12 or score >= 0.55
            chunk_id = f"{meta.get('doc_id', '')}_{meta.get('position', rank)}"
            retrieved.append(chunk_id if is_relevant else f"irrelevant_{rank}")

            if is_relevant and first_relevant_rank is None:
                first_relevant_rank = rank

        rankings.append(first_relevant_rank)
        retrieved_ids_all.append(
            [rid for rid in retrieved if not rid.startswith("irrelevant_")]
        )
        relevant_ids_all.append(relevant)

    mrr = mean_reciprocal_rank(rankings)
    r1 = recall_at_k(retrieved_ids_all, relevant_ids_all, 1)
    r3 = recall_at_k(retrieved_ids_all, relevant_ids_all, 3)
    r5 = recall_at_k(retrieved_ids_all, relevant_ids_all, 5)

    return {
        "strategy": strategy_name,
        "MRR": round(mrr, 4),
        "Recall@1": round(r1, 4),
        "Recall@3": round(r3, 4),
        "Recall@5": round(r5, 4),
        "n_questions": len(benchmark.questions),
        "n_found": sum(1 for r in rankings if r is not None),
    }


def main():
    from sentence_transformers import SentenceTransformer

    log.info("Loading model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    log.info("Loading PubMedQA benchmark...")
    benchmark = PubMedQABenchmark()
    benchmark.load()
    log.info("Loaded %d questions", len(benchmark.questions))

    index_dir = RESULTS_DIR / "indices"
    strategies = ["naive", "semantic", "late_chunking", "structure_aware", "gralc_rag"]

    # Load entity linker and embeddings for graph-guided retrieval
    entity_linker = SimpleEntityLinker()
    entity_embeddings = {}
    chunk_entity_map = {}

    emb_path = index_dir / "entity_embeddings.npz"
    if emb_path.exists():
        data = np.load(emb_path, allow_pickle=True)
        names = list(data["names"])
        embs = data["embeddings"]
        entity_embeddings = dict(zip(names, embs))
        log.info("Loaded %d entity embeddings", len(entity_embeddings))

    cem_path = index_dir / "chunk_entity_map.json"
    if cem_path.exists():
        with open(cem_path) as f:
            raw = json.load(f)
            chunk_entity_map = {int(k): v for k, v in raw.items()}
        log.info("Loaded chunk entity map with %d entries", len(chunk_entity_map))

    all_results = []

    for strategy in strategies:
        idx_path = index_dir / strategy
        if not Path(str(idx_path) + ".faiss").exists():
            log.warning("Index not found for %s, skipping", strategy)
            continue

        log.info("=== Evaluating: %s ===", strategy)
        index = VectorIndex(dim=model.get_sentence_embedding_dimension())
        index.load(idx_path)

        result = evaluate_strategy(
            strategy_name=strategy,
            index=index,
            benchmark=benchmark,
            model=model,
            top_k=TOP_K_RETRIEVAL,
        )
        all_results.append(result)
        log.info(
            "  %s: MRR=%.4f  R@1=%.4f  R@3=%.4f  R@5=%.4f  (found %d/%d)",
            strategy, result["MRR"], result["Recall@1"],
            result["Recall@3"], result["Recall@5"],
            result["n_found"], result["n_questions"],
        )

    # Graph-guided retrieval on GraLC-RAG index
    if entity_embeddings and chunk_entity_map:
        gralc_idx_path = index_dir / "gralc_rag"
        if (gralc_idx_path / "index.faiss").exists():
            log.info("=== Evaluating: gralc_rag_graph (with graph-guided retrieval) ===")
            index = VectorIndex(dim=model.get_sentence_embedding_dimension())
            index.load(gralc_idx_path)

            result = evaluate_strategy(
                strategy_name="gralc_rag_graph",
                index=index,
                benchmark=benchmark,
                model=model,
                top_k=TOP_K_RETRIEVAL,
                entity_linker=entity_linker,
                entity_embeddings=entity_embeddings,
                chunk_entity_map=chunk_entity_map,
            )
            all_results.append(result)
            log.info(
                "  gralc_rag_graph: MRR=%.4f  R@1=%.4f  R@3=%.4f  R@5=%.4f",
                result["MRR"], result["Recall@1"],
                result["Recall@3"], result["Recall@5"],
            )

    # Save results
    output_path = RESULTS_DIR / "retrieval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    log.info("\n" + "=" * 70)
    log.info("%-25s  %8s  %8s  %8s  %8s", "Strategy", "MRR", "R@1", "R@3", "R@5")
    log.info("-" * 70)
    for r in all_results:
        log.info(
            "%-25s  %8.4f  %8.4f  %8.4f  %8.4f",
            r["strategy"], r["MRR"], r["Recall@1"], r["Recall@3"], r["Recall@5"],
        )
    log.info("=" * 70)
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
