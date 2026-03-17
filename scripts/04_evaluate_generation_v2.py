"""Generation evaluation using PubMedQA abstracts + OpenAI GPT-4o-mini.

Builds indices from PubMedQA abstracts (same as retrieval v2), retrieves
top-5 chunks per question, then calls OpenAI to generate yes/no/maybe answers.
Evaluates on 100 questions (subset) to manage API costs.
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
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from gralc_rag.chunking.naive import Chunk, naive_chunk
from gralc_rag.chunking.late import late_chunk
from gralc_rag.retrieval.index import VectorIndex
from gralc_rag.generation.openai_gen import generate_answer
from gralc_rag.evaluation.metrics import accuracy, answer_f1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_EVAL = 100  # Evaluate on first 100 questions to manage API costs


def load_pubmedqa() -> list[dict]:
    path = DATA_DIR / "pubmedqa" / "questions.json"
    if not path.exists():
        path = DATA_DIR / "pubmedqa" / "pqa_labeled.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: list[Chunk]) -> VectorIndex:
    embeddings = []
    metadata_list = []
    for c in chunks:
        if c.embedding is not None:
            emb = c.embedding / (np.linalg.norm(c.embedding) + 1e-8)
            embeddings.append(emb)
            metadata_list.append({**c.metadata, "text": c.text})
    emb_array = np.array(embeddings, dtype=np.float32)
    idx = VectorIndex(dim=emb_array.shape[1])
    idx.build(emb_array, metadata_list)
    return idx


def main():
    from sentence_transformers import SentenceTransformer

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set in .env")
        sys.exit(1)

    log.info("Loading model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer

    log.info("Loading PubMedQA...")
    all_questions = load_pubmedqa()
    questions = all_questions[:N_EVAL]
    log.info("Evaluating generation on %d questions (of %d total)", len(questions), len(all_questions))

    # Build corpus from ALL abstracts (not just eval subset)
    abstracts = []
    for q in all_questions:
        ctx = q.get("context", "")
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        abstracts.append({
            "doc_id": str(q.get("pubid", q.get("id", ""))),
            "text": ctx,
        })

    # --- Build naive index ---
    log.info("Building naive index...")
    naive_chunks = []
    for a in abstracts:
        cs = naive_chunk(a["text"], doc_id=a["doc_id"], max_tokens=256, overlap=32)
        naive_chunks.extend(cs)
    texts = [c.text for c in naive_chunks if c.embedding is None]
    if texts:
        embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=128)
        for i, c in enumerate(naive_chunks):
            if c.embedding is None:
                naive_chunks[i] = Chunk(text=c.text, embedding=embs[i], metadata=c.metadata)
    naive_idx = build_index(naive_chunks)

    # --- Build late chunking index ---
    log.info("Building late chunking index...")
    late_chunks = []
    for a in abstracts:
        cs = late_chunk(a["text"], doc_id=a["doc_id"], model=model, tokenizer=tokenizer)
        late_chunks.extend(cs)
    late_idx = build_index(late_chunks)

    strategies = {
        "naive": naive_idx,
        "late_chunking": late_idx,
    }

    all_results = []

    for strategy_name, index in strategies.items():
        log.info("=== Generation: %s ===", strategy_name)

        predictions = []
        labels = []
        f1_scores = []
        total_tokens = 0

        for i, q in enumerate(questions):
            query_emb = model.encode(q["question"], normalize_embeddings=True)
            results = index.search(query_emb, top_k=5)
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

            if (i + 1) % 25 == 0:
                log.info("  Processed %d/%d", i + 1, len(questions))

            time.sleep(0.15)  # Rate limit

        acc = accuracy(predictions, labels)
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        result = {
            "strategy": strategy_name,
            "accuracy": round(acc, 4),
            "avg_f1": round(avg_f1, 4),
            "n_questions": len(questions),
            "total_tokens": total_tokens,
            "model": OPENAI_MODEL,
        }
        all_results.append(result)
        log.info("  %s: Accuracy=%.4f  F1=%.4f  Tokens=%d",
                 strategy_name, acc, avg_f1, total_tokens)

    # Save
    output = RESULTS_DIR / "generation_results_v2.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'Strategy':<25s} {'Accuracy':>10s} {'Avg F1':>10s}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['strategy']:<25s} {r['accuracy']:10.4f} {r['avg_f1']:10.4f}")
    print("=" * 60)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
