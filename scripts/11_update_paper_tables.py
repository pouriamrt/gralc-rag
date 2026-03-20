#!/usr/bin/env python
"""Auto-update paper.tex with results from fulltext evaluation.

Reads all 18 result JSON files from results/fulltext/ and generates:
1. Updated LaTeX table for tab:fulltext (retrieval performance)
2. Updated LaTeX table for tab:crosssection (section coverage + CS recall)
3. Updated discussion text with correct numbers
4. Summary of all results to stdout

Usage:
    python scripts/11_update_paper_tables.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "fulltext"
PAPER_PATH = PROJECT_DIR / "paper.tex"

STRATEGIES = [
    "naive", "semantic", "late_chunking",
    "structure_aware", "gralc_rag", "gralc_rag_graph",
]
CONDITIONS = ["intro", "partial", "fulltext"]

STRATEGY_LABELS = {
    "naive": "Naive",
    "semantic": "Semantic",
    "late_chunking": "Late Chunking",
    "structure_aware": "Structure-Aware",
    "gralc_rag": r"\gralcrag{} (KG)",
    "gralc_rag_graph": r"\gralcrag{} (+Graph)",
}

CONDITION_LABELS = {
    "intro": "Introduction",
    "partial": "Partial",
    "fulltext": "Full-text",
}


def load_results() -> dict[str, dict[str, dict]]:
    """Load all result files. Returns {condition: {strategy: data}}."""
    results: dict[str, dict[str, dict]] = {}
    missing = []

    for cond in CONDITIONS:
        results[cond] = {}
        for strat in STRATEGIES:
            path = RESULTS_DIR / f"{cond}_{strat}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    results[cond][strat] = json.load(f)
            else:
                missing.append(f"{cond}_{strat}")

    if missing:
        print(f"WARNING: {len(missing)} result files missing:")
        for m in missing:
            print(f"  - {m}.json")
        print()

    return results


def find_best(results: dict, cond: str, metric: str) -> str | None:
    """Find the strategy with the best value for a metric in a condition."""
    best_val = -1
    best_strat = None
    for strat in STRATEGIES:
        if strat in results[cond] and metric in results[cond][strat]:
            val = results[cond][strat][metric]
            if val > best_val:
                best_val = val
                best_strat = strat
    return best_strat


def fmt(val: float, bold: bool = False) -> str:
    """Format a float to 3 decimal places, optionally bold."""
    s = f"{val:.3f}"
    return rf"\textbf{{{s}}}" if bold else s


def generate_fulltext_table(results: dict) -> str:
    """Generate LaTeX content for the fulltext retrieval performance table."""
    lines = []

    for i, cond in enumerate(CONDITIONS):
        if i > 0:
            lines.append(r"\midrule")

        label = CONDITION_LABELS[cond]
        best_mrr = find_best(results, cond, "MRR")
        best_r1 = find_best(results, cond, "Recall@1")
        best_r5 = find_best(results, cond, "Recall@5")
        best_ndcg = find_best(results, cond, "nDCG@10")

        for j, strat in enumerate(STRATEGIES):
            if strat not in results[cond]:
                continue

            d = results[cond][strat]
            prefix = rf"\multirow{{6}}{{*}}{{{label}}}" if j == 0 else ""

            mrr = fmt(d["MRR"], strat == best_mrr)
            r1 = fmt(d["Recall@1"], strat == best_r1)
            r5 = fmt(d["Recall@5"], strat == best_r5)
            ndcg = fmt(d["nDCG@10"], strat == best_ndcg)
            slabel = STRATEGY_LABELS[strat]

            lines.append(
                f"{prefix}\n  & {slabel:<22s} & {mrr} & {r1} & {r5} & {ndcg} \\\\"
            )

    return "\n".join(lines)


def generate_crosssection_table(results: dict) -> str:
    """Generate LaTeX content for the cross-section table with SecCov@5/10/20."""
    lines = []

    for strat in STRATEGIES:
        slabel = STRATEGY_LABELS[strat]
        row_parts = [slabel]

        # SecCov@5 for each condition
        for cond in CONDITIONS:
            if strat in results[cond]:
                d = results[cond][strat]
                val = d.get("section_coverage@5", 0)
                # Bold if structure-aware and fulltext condition
                bold = (strat in ("structure_aware", "gralc_rag", "gralc_rag_graph")
                        and cond == "fulltext")
                row_parts.append(fmt(val, bold) if val != 1.0 else "1.00")
            else:
                row_parts.append("---")

        # SecCov@20 for each condition
        for cond in CONDITIONS:
            if strat in results[cond]:
                d = results[cond][strat]
                val = d.get("section_coverage@20", 0)
                bold = (strat in ("structure_aware", "gralc_rag", "gralc_rag_graph")
                        and cond == "fulltext")
                row_parts.append(fmt(val, bold) if val != 1.0 else "1.00")
            else:
                row_parts.append("---")

        # CS Recall@20 for each condition
        for cond in CONDITIONS:
            if strat in results[cond]:
                d = results[cond][strat]
                val = d.get("cross_section_recall@20", 0)
                row_parts.append(f"{val:.3f}")
            else:
                row_parts.append("---")

        line = f"{row_parts[0]:<22s}"
        for p in row_parts[1:]:
            line += f" & {p}"
        line += r" \\"
        lines.append(line)

    return "\n".join(lines)


def print_summary(results: dict) -> None:
    """Print a human-readable summary of all results."""
    print("=" * 80)
    print("FULLTEXT EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    for cond in CONDITIONS:
        print(f"\n--- {CONDITION_LABELS[cond]} ---")
        print(f"  {'Strategy':<20s} {'MRR':>8s} {'R@1':>8s} {'R@5':>8s} {'nDCG@10':>8s} {'SecCov@5':>9s} {'SecCov@20':>10s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")

        for strat in STRATEGIES:
            if strat not in results[cond]:
                print(f"  {STRATEGY_LABELS[strat]:<20s} --- MISSING ---")
                continue

            d = results[cond][strat]
            seccov5 = d.get("section_coverage@5", 0)
            seccov20 = d.get("section_coverage@20", 0)
            print(
                f"  {STRATEGY_LABELS[strat]:<20s} "
                f"{d['MRR']:8.4f} {d['Recall@1']:8.4f} {d['Recall@5']:8.4f} "
                f"{d['nDCG@10']:8.4f} {seccov5:9.4f} {seccov20:10.4f}"
            )


def update_paper(results: dict, dry_run: bool = False) -> None:
    """Replace table content in paper.tex with new results."""
    paper = PAPER_PATH.read_text(encoding="utf-8")
    original = paper

    # --- Update fulltext table (tab:fulltext) ---
    table_content = generate_fulltext_table(results)
    # Match from first \multirow to last \\ before \bottomrule
    pattern = r"(\\multirow\{6\}\{\*\}\{Introduction\}.*?)(\\bottomrule)"
    match = re.search(pattern, paper, re.DOTALL)
    if match:
        paper = paper[:match.start()] + table_content + "\n" + match.group(2) + paper[match.end():]
        print("[OK] Updated tab:fulltext table content")
    else:
        print("[WARN] Could not find tab:fulltext table content to replace")

    # --- Update cross-section table (tab:crosssection) ---
    # Replace the table body between \midrule and \bottomrule in the crosssection table
    cs_content = generate_crosssection_table(results)
    cs_pattern = r"(\\label\{tab:crosssection\}.*?\\midrule\n)(.*?)(\\bottomrule)"
    cs_match = re.search(cs_pattern, paper, re.DOTALL)
    if cs_match:
        paper = paper[:cs_match.start(2)] + cs_content + "\n" + paper[cs_match.start(3):]
        print("[OK] Updated tab:crosssection table content")
    else:
        print("[WARN] Could not find tab:crosssection table content to replace")

    # --- Update discussion numbers ---
    # Update "Semantic chunking achieves the highest MRR" numbers in discussion
    if "intro" in results and "semantic" in results["intro"]:
        intro_sem = results["intro"]["semantic"]["MRR"]
        partial_sem = results["partial"]["semantic"]["MRR"] if "semantic" in results.get("partial", {}) else None
        fulltext_sem = results["fulltext"]["semantic"]["MRR"] if "semantic" in results.get("fulltext", {}) else None

        if partial_sem and fulltext_sem:
            old_discussion = (
                r"Semantic chunking achieves the highest MRR across all document-length "
                r"conditions (0.586 on introduction, 0.735 on partial, 0.736 on full-text)"
            )
            new_discussion = (
                r"Semantic chunking achieves the highest MRR across all document-length "
                f"conditions ({intro_sem:.3f} on introduction, {partial_sem:.3f} on partial, "
                f"{fulltext_sem:.3f} on full-text)"
            )
            if old_discussion in paper:
                paper = paper.replace(old_discussion, new_discussion)
                print("[OK] Updated discussion MRR numbers")

    # Update GraLC-RAG section coverage numbers in discussion
    if "intro" in results and "gralc_rag_graph" in results["intro"]:
        intro_gg = results["intro"]["gralc_rag_graph"].get("section_coverage@5", 0)
        fulltext_gg = (results["fulltext"]["gralc_rag_graph"].get("section_coverage@5", 0)
                       if "gralc_rag_graph" in results.get("fulltext", {}) else None)

        if fulltext_gg:
            old_seccov = (
                r"\gralcrag{} (+Graph) retrieves from 2.20 sections on introduction-only "
                r"documents and 4.10 sections on full-text articles"
            )
            new_seccov = (
                rf"\gralcrag{{}} (+Graph) retrieves from {intro_gg:.2f} sections on "
                f"introduction-only documents and {fulltext_gg:.2f} sections on full-text articles"
            )
            if old_seccov in paper:
                paper = paper.replace(old_seccov, new_seccov)
                print("[OK] Updated discussion SecCov numbers")

    if paper != original:
        if dry_run:
            print("\n[DRY RUN] Would write updated paper.tex")
        else:
            PAPER_PATH.write_text(paper, encoding="utf-8")
            print(f"\n[OK] Updated {PAPER_PATH}")
    else:
        print("\n[INFO] No changes to paper.tex")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update paper.tex with evaluation results")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    results = load_results()

    # Check completeness
    total = sum(len(results[c]) for c in CONDITIONS)
    expected = len(STRATEGIES) * len(CONDITIONS)
    print(f"Loaded {total}/{expected} result files\n")

    print_summary(results)

    if total < expected:
        print(f"\n[WARN] Only {total}/{expected} results available. Tables will have gaps.")
        response = input("Continue with partial update? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    print()
    update_paper(results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
