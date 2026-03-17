"""Step 4: Evaluate end-to-end generation with OpenAI on PubMedQA.

Uses retrieved chunks from each strategy as context for GPT-4o-mini generation.
Produces results/generation_results.json with accuracy and F1 for each strategy.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gralc_rag.config import (
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RESULTS_DIR,
    TOP_K_RETRIEVAL,
    KG_WEIGHT_BETA,
)
from gralc_rag.evaluation.benchmarks import PubMedQABenchmark
from gralc_rag.evaluation.metrics import accuracy, answer_f1
from gralc_rag.generation.openai_gen import generate_answer
from gralc_rag.retrieval.index import VectorIndex
from gralc_rag.retrieval.dense import dense_retrieve
from gralc_rag.retrieval.graph_guided import graph_guided_retrieve
from gralc_rag.knowledge.entity_linker import SimpleEntityLinker

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    from sentence_transformers import SentenceTransformer

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set in .env")
        sys.exit(1)

    log.info("Loading model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    log.info("Loading PubMedQA benchmark...")
    benchmark = PubMedQABenchmark()
    benchmark.load()
    log.info("Loaded %d questions", len(benchmark.questions))

    index_dir = RESULTS_DIR / "indices"

    # Load entity linker + embeddings for graph-guided strategy
    entity_linker = SimpleEntityLinker()
    entity_embeddings = {}
    chunk_entity_map = {}

    emb_path = index_dir / "entity_embeddings.npz"
    if emb_path.exists():
        data = np.load(emb_path, allow_pickle=True)
        entity_embeddings = dict(zip(data["names"], data["embeddings"]))

    cem_path = index_dir / "chunk_entity_map.json"
    if cem_path.exists():
        with open(cem_path) as f:
            chunk_entity_map = {int(k): v for k, v in json.load(f).items()}

    strategies_to_eval = ["naive", "late_chunking", "gralc_rag"]
    all_results = []

    for strategy in strategies_to_eval:
        idx_path = index_dir / strategy
        if not (idx_path / "index.faiss").exists():
            log.warning("Index not found for %s, skipping", strategy)
            continue

        log.info("=== Generating answers: %s ===", strategy)
        index = VectorIndex(dim=model.get_sentence_embedding_dimension())
        index.load(idx_path)

        use_graph = (
            strategy == "gralc_rag"
            and entity_embeddings
            and chunk_entity_map
        )

        predictions = []
        labels = []
        f1_scores = []
        total_tokens = 0

        for i, q in enumerate(benchmark.questions):
            if use_graph:
                results = graph_guided_retrieve(
                    query=q["question"],
                    model=model,
                    index=index,
                    entity_linker=entity_linker,
                    entity_embeddings=entity_embeddings,
                    chunk_entity_map=chunk_entity_map,
                    top_k=TOP_K_RETRIEVAL,
                    beta=KG_WEIGHT_BETA,
                )
            else:
                results = dense_retrieve(q["question"], model, index, top_k=TOP_K_RETRIEVAL)

            contexts = [meta.get("text", "") for meta, _ in results]

            try:
                answer = generate_answer(
                    question=q["question"],
                    contexts=contexts,
                    api_key=OPENAI_API_KEY,
                    model=OPENAI_MODEL,
                )
                predictions.append(answer["decision"])
                labels.append(q["label"])
                f1 = answer_f1(answer["full_answer"], q.get("long_answer", ""))
                f1_scores.append(f1)
                total_tokens += answer.get("usage_tokens", 0)
            except Exception as e:
                log.warning("Error on question %d: %s", i, e)
                predictions.append("unknown")
                labels.append(q["label"])
                f1_scores.append(0.0)

            if (i + 1) % 50 == 0:
                log.info("  Processed %d/%d questions", i + 1, len(benchmark.questions))

            # Rate limiting
            time.sleep(0.2)

        acc = accuracy(predictions, labels)
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        result = {
            "strategy": strategy + ("_graph" if use_graph else ""),
            "accuracy": round(acc, 4),
            "avg_f1": round(avg_f1, 4),
            "n_questions": len(benchmark.questions),
            "total_tokens": total_tokens,
            "model": OPENAI_MODEL,
        }
        all_results.append(result)
        log.info("  %s: Accuracy=%.4f  F1=%.4f  Tokens=%d",
                 result["strategy"], acc, avg_f1, total_tokens)

    # Save results
    output_path = RESULTS_DIR / "generation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("%-25s  %10s  %10s", "Strategy", "Accuracy", "Avg F1")
    log.info("-" * 60)
    for r in all_results:
        log.info("%-25s  %10.4f  %10.4f", r["strategy"], r["accuracy"], r["avg_f1"])
    log.info("=" * 60)
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
