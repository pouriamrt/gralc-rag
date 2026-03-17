"""PubMedQA benchmark harness for retrieval and generation evaluation.

Loads the PubMedQA labelled split, provides helpers to run retrieval and
generation functions against it, and computes standard metrics.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from gralc_rag.config import PUBMEDQA_DIR
from gralc_rag.evaluation.metrics import (
    accuracy,
    answer_f1,
    mean_reciprocal_rank,
    recall_at_k,
)

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    """Lowercase whitespace tokenisation for Jaccard overlap."""
    return set(text.lower().split())


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two strings (whitespace-tokenised)."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class PubMedQABenchmark:
    """Benchmark wrapper around the PubMedQA *pqa_labeled* dataset.

    Parameters:
        data_dir: Directory where PubMedQA data is cached.  Defaults to
            ``data/pubmedqa/``.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or PUBMEDQA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.questions: list[dict] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load PubMedQA questions from local cache or HuggingFace.

        Persists the processed data to ``data/pubmedqa/questions.json`` for
        fast subsequent loads.
        """
        cache_path = self.data_dir / "questions.json"

        if cache_path.exists():
            logger.info("Loading cached PubMedQA questions from %s", cache_path)
            with open(cache_path, "r", encoding="utf-8") as fh:
                self.questions = json.load(fh)
            logger.info("Loaded %d questions", len(self.questions))
            return

        logger.info("Downloading PubMedQA pqa_labeled from HuggingFace ...")
        from datasets import load_dataset

        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

        self.questions = []
        for row in ds:
            # Flatten the nested context structure.
            context_parts = row.get("context", {})
            if isinstance(context_parts, dict):
                texts = context_parts.get("contexts", [])
                context_text = "\n".join(texts) if texts else ""
            elif isinstance(context_parts, str):
                context_text = context_parts
            else:
                context_text = str(context_parts)

            pubid = str(row.get("pubid", ""))

            self.questions.append(
                {
                    "id": pubid,
                    "question": row["question"],
                    "context": context_text,
                    "long_answer": row.get("long_answer", ""),
                    "label": row.get("final_decision", "").lower(),
                    "pmid": pubid,
                }
            )

        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(self.questions, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Saved %d PubMedQA questions to %s", len(self.questions), cache_path
        )

    # ------------------------------------------------------------------
    # Retrieval evaluation
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        retrieval_fn: Callable[[str], list[dict]],
        corpus_chunks: list[dict],
        jaccard_threshold: float = 0.3,
    ) -> dict[str, float]:
        """Evaluate a retrieval function on PubMedQA questions.

        Parameters:
            retrieval_fn: Accepts a question string, returns a ranked list of
                chunk dicts (must have at least a ``"text"`` key).
            corpus_chunks: Full corpus chunk list (unused directly but kept
                for API symmetry with other benchmarks).
            jaccard_threshold: Minimum Jaccard similarity between a chunk and
                the original abstract for the chunk to be considered relevant.

        Returns:
            Dictionary of metric name to value: ``mrr``, ``recall@1``,
            ``recall@3``, ``recall@5``.
        """
        if not self.questions:
            logger.warning("No questions loaded — call load() first.")
            return {}

        rankings: list[int | None] = []
        all_retrieved: list[list[str]] = []
        all_relevant: list[list[str]] = []

        for q in self.questions:
            question_text = q["question"]
            original_context = q["context"]

            retrieved_chunks = retrieval_fn(question_text)

            # Determine which retrieved chunks are relevant via Jaccard overlap
            # with the original context.
            first_relevant_rank: int | None = None
            retrieved_ids: list[str] = []
            relevant_ids: list[str] = []

            for rank_idx, chunk in enumerate(retrieved_chunks):
                chunk_text = chunk.get("text", "")
                chunk_id = chunk.get("id", str(rank_idx))
                retrieved_ids.append(chunk_id)

                sim = _jaccard(chunk_text, original_context)
                if sim >= jaccard_threshold:
                    relevant_ids.append(chunk_id)
                    if first_relevant_rank is None:
                        first_relevant_rank = rank_idx + 1  # 1-indexed

            rankings.append(first_relevant_rank)
            all_retrieved.append(retrieved_ids)
            # For recall computation, the "relevant" set is the chunks that
            # matched the original context.  If none matched, we still add
            # the list (empty) so the query is counted.
            all_relevant.append(relevant_ids if relevant_ids else [])

        return {
            "mrr": mean_reciprocal_rank(rankings),
            "recall@1": recall_at_k(all_retrieved, all_relevant, k=1),
            "recall@3": recall_at_k(all_retrieved, all_relevant, k=3),
            "recall@5": recall_at_k(all_retrieved, all_relevant, k=5),
        }

    # ------------------------------------------------------------------
    # Generation evaluation
    # ------------------------------------------------------------------

    def evaluate_generation(
        self,
        generation_fn: Callable[[str], dict],
    ) -> dict[str, float]:
        """Evaluate a generation function on PubMedQA questions.

        Parameters:
            generation_fn: Accepts a question string, returns a dict with
                at least ``"decision"`` (yes/no/maybe) and ``"full_answer"``.

        Returns:
            Dictionary with ``accuracy`` (decision vs label) and
            ``avg_f1`` (full_answer vs long_answer).
        """
        if not self.questions:
            logger.warning("No questions loaded — call load() first.")
            return {}

        predictions: list[str] = []
        golds: list[str] = []
        f1_scores: list[float] = []

        for q in self.questions:
            question_text = q["question"]

            try:
                result = generation_fn(question_text)
            except Exception:
                logger.warning(
                    "Generation failed for question %s", q.get("id", "?"),
                    exc_info=True,
                )
                predictions.append("")
                golds.append(q["label"])
                f1_scores.append(0.0)
                continue

            decision = result.get("decision", "unknown").lower()
            full_answer = result.get("full_answer", "")

            predictions.append(decision)
            golds.append(q["label"])
            f1_scores.append(answer_f1(full_answer, q.get("long_answer", "")))

        acc = accuracy(predictions, golds)
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {
            "accuracy": acc,
            "avg_f1": avg_f1,
        }
