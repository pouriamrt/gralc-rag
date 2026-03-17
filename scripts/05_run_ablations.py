"""Step 5: Run ablation study and compile all results.

Compiles retrieval + generation results into a single summary.
Produces results/ablation_summary.json and prints the paper-ready tables.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gralc_rag.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def main():
    retrieval_results = load_json(RESULTS_DIR / "retrieval_results.json")
    generation_results = load_json(RESULTS_DIR / "generation_results.json")
    indexing_summary = load_json(RESULTS_DIR / "indexing_summary.json")

    if not retrieval_results:
        log.error("No retrieval results found. Run 03_evaluate_retrieval.py first.")
        sys.exit(1)

    # Build retrieval lookup
    ret_lookup = {r["strategy"]: r for r in retrieval_results}
    gen_lookup = {r["strategy"]: r for r in generation_results}

    # === Table 1: Main retrieval results ===
    print("\n" + "=" * 80)
    print("TABLE 1: Retrieval Performance on PubMedQA*")
    print("=" * 80)
    print(f"{'Strategy':<28s} {'MRR':>8s} {'R@1':>8s} {'R@3':>8s} {'R@5':>8s} {'Found':>8s}")
    print("-" * 80)

    strategy_order = ["naive", "semantic", "late_chunking", "structure_aware",
                      "gralc_rag", "gralc_rag_graph"]

    for s in strategy_order:
        if s in ret_lookup:
            r = ret_lookup[s]
            print(f"{s:<28s} {r['MRR']:8.4f} {r['Recall@1']:8.4f} "
                  f"{r['Recall@3']:8.4f} {r['Recall@5']:8.4f} "
                  f"{r['n_found']:>4d}/{r['n_questions']}")
    print("=" * 80)

    # === Table 2: Generation results ===
    if generation_results:
        print("\n" + "=" * 60)
        print("TABLE 2: Generation Performance on PubMedQA*")
        print("=" * 60)
        print(f"{'Strategy':<28s} {'Accuracy':>10s} {'Avg F1':>10s}")
        print("-" * 60)
        for s in ["naive", "late_chunking", "gralc_rag_graph"]:
            if s in gen_lookup:
                r = gen_lookup[s]
                print(f"{r['strategy']:<28s} {r['accuracy']:10.4f} {r['avg_f1']:10.4f}")
        print("=" * 60)

    # === Table 3: Ablation (retrieval) ===
    print("\n" + "=" * 80)
    print("TABLE 3: Ablation Study — Retrieval MRR on PubMedQA*")
    print("=" * 80)
    ablation_configs = [
        ("Late Chunking (base)", "late_chunking"),
        ("+ Structure Boundaries", "structure_aware"),
        ("+ KG Infusion", "gralc_rag"),
        ("+ Graph-Guided Retrieval (full)", "gralc_rag_graph"),
    ]
    base_mrr = ret_lookup.get("late_chunking", {}).get("MRR", 0)
    for label, key in ablation_configs:
        if key in ret_lookup:
            mrr = ret_lookup[key]["MRR"]
            delta = mrr - base_mrr
            print(f"  {label:<40s}  MRR={mrr:.4f}  (Delta={delta:+.4f})")
    print("=" * 80)

    # === Table 4: Indexing efficiency ===
    if indexing_summary:
        print("\n" + "=" * 60)
        print("TABLE 4: Indexing Efficiency")
        print("=" * 60)
        print(f"{'Strategy':<28s} {'Chunks':>8s} {'Time (s)':>10s}")
        print("-" * 60)
        for name, info in indexing_summary.items():
            print(f"{name:<28s} {info['n_chunks']:8d} {info['time']:10.1f}")
        print("=" * 60)

    # Save compiled results
    compiled = {
        "retrieval": retrieval_results,
        "generation": generation_results,
        "indexing": indexing_summary,
    }
    output_path = RESULTS_DIR / "ablation_summary.json"
    with open(output_path, "w") as f:
        json.dump(compiled, f, indent=2)

    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
