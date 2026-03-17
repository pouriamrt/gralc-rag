"""Generate publication-quality figures for full-text evaluation results.

Produces:
    fig6_crossover_plot.{pdf,png}    — MRR vs document length (crossover plot)
    fig7_crosssection_heatmap.{pdf,png} — Cross-section recall heatmap
    fig8_section_coverage.{pdf,png}  — Section coverage@5 bar chart
"""

from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── Paths ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_DIR / "results" / "fulltext"

CROSSOVER_PATH = RESULTS_DIR / "fulltext_retrieval_pubmedqa_summary.json"
SUMMARY_PATH = RESULTS_DIR / "fulltext_retrieval_summary.json"

# ── Publication style (matches generate_figures.py) ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# ── Strategies ──
STRATEGY_KEYS = [
    "naive", "semantic", "late_chunking",
    "structure_aware", "gralc_rag", "gralc_rag_graph",
]

STRATEGY_LABELS = {
    "naive": "Naive",
    "semantic": "Semantic",
    "late_chunking": "Late Chunking",
    "structure_aware": "Structure-Aware",
    "gralc_rag": "GraLC-RAG (KG)",
    "gralc_rag_graph": "GraLC-RAG (+Graph)",
}

COLORS = {
    "naive": "#4C72B0",
    "semantic": "#55A868",
    "late_chunking": "#C44E52",
    "structure_aware": "#8172B2",
    "gralc_rag": "#CCB974",
    "gralc_rag_graph": "#64B5CD",
}

MARKERS = {
    "naive": "o",
    "semantic": "s",
    "late_chunking": "^",
    "structure_aware": "D",
    "gralc_rag": "v",
    "gralc_rag_graph": "P",
}

LINESTYLES = {
    "naive": "-",
    "semantic": "--",
    "late_chunking": "-.",
    "structure_aware": ":",
    "gralc_rag": "-",
    "gralc_rag_graph": "--",
}

# ── Document-length conditions for crossover plot ──
CONDITIONS = ["abstract", "intro", "partial", "fulltext"]
CONDITION_LABELS = {
    "abstract": "Abstract\n(~200 w)",
    "intro": "Intro\n(~500\u20131K w)",
    "partial": "Partial\n(~2\u20134K w)",
    "fulltext": "Full-text\n(~5\u20138K w)",
}

# ── Hardcoded abstract baseline MRR ──
ABSTRACT_MRR = {
    "naive": 0.9787,
    "semantic": 0.9802,
    "late_chunking": 0.9768,
    "structure_aware": 0.9765,
    "gralc_rag": 0.9687,
    "gralc_rag_graph": 0.9502,
}

# Non-abstract conditions used for heatmap and section coverage
NON_ABSTRACT_CONDITIONS = ["intro", "partial", "fulltext"]
NON_ABSTRACT_LABELS = {
    "intro": "Intro",
    "partial": "Partial",
    "fulltext": "Full-text",
}


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None with a warning if it doesn't exist."""
    if not path.exists():
        warnings.warn(f"Results file not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_crossover_mrr() -> dict[str, dict[str, float | None]]:
    """Build {strategy: {condition: mrr}} for the crossover plot.

    Abstract values are always hardcoded.  Other conditions come from
    ``pubmedqa_crossover_results.json`` (preferred) falling back to
    ``fulltext_results_summary.json``.
    """
    mrr: dict[str, dict[str, float | None]] = {
        s: {"abstract": ABSTRACT_MRR[s]} for s in STRATEGY_KEYS
    }

    # Try crossover results first (list of dicts with strategy/condition/MRR)
    crossover = _load_json(CROSSOVER_PATH)
    if crossover is not None and isinstance(crossover, list) and len(crossover) > 0:
        for entry in crossover:
            strat = entry.get("strategy", "")
            cond = entry.get("condition", "")
            if strat in mrr and cond in NON_ABSTRACT_CONDITIONS:
                mrr[strat][cond] = entry.get("MRR")
        return mrr

    # Fallback to summary (list of dicts with strategy/condition/MRR)
    summary = _load_json(SUMMARY_PATH)
    if summary is not None and isinstance(summary, list) and len(summary) > 0:
        for entry in summary:
            strat = entry.get("strategy", "")
            cond = entry.get("condition", "")
            if strat in mrr and cond in NON_ABSTRACT_CONDITIONS:
                mrr[strat][cond] = entry.get("MRR")
        return mrr

    # No data available — fill with None
    for strategy in STRATEGY_KEYS:
        for cond in NON_ABSTRACT_CONDITIONS:
            mrr[strategy][cond] = None
    return mrr


def _get_summary_data() -> dict[str, dict[str, dict]] | None:
    """Load full-text results summary, indexed by (condition, strategy)."""
    raw = _load_json(SUMMARY_PATH)
    if raw is None:
        return None
    if isinstance(raw, list):
        indexed: dict[str, dict[str, dict]] = {}
        for entry in raw:
            cond = entry.get("condition", "")
            strat = entry.get("strategy", "")
            indexed.setdefault(cond, {})[strat] = entry
        return indexed
    return raw


# ════════════════════════════════════════════════════════════
# Figure 6: Crossover Plot (SIGNATURE FIGURE)
# ════════════════════════════════════════════════════════════

def generate_fig6() -> None:
    mrr_data = _get_crossover_mrr()

    fig, ax = plt.subplots(figsize=(7, 4))
    x_positions = np.arange(len(CONDITIONS))

    has_any_data = False
    for strategy in STRATEGY_KEYS:
        ys = []
        xs = []
        for i, cond in enumerate(CONDITIONS):
            val = mrr_data[strategy].get(cond)
            if val is not None:
                xs.append(x_positions[i])
                ys.append(val)
                has_any_data = True

        if xs:
            ax.plot(
                xs, ys,
                color=COLORS[strategy],
                marker=MARKERS[strategy],
                linestyle=LINESTYLES[strategy],
                linewidth=1.8,
                markersize=7,
                label=STRATEGY_LABELS[strategy],
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in CONDITIONS], fontsize=8.5,
    )
    ax.set_ylabel("Mean Reciprocal Rank (MRR)")
    ax.set_title("Retrieval MRR vs. Document Length Condition")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="best", framealpha=0.9, fontsize=8)

    if has_any_data:
        all_vals = [
            v for s in mrr_data.values()
            for v in s.values() if v is not None
        ]
        ymin = min(all_vals) - 0.02
        ymax = max(all_vals) + 0.01
        ax.set_ylim(max(0, ymin), min(1.0, ymax))
    else:
        ax.set_ylim(0.90, 1.0)
        ax.text(
            0.5, 0.5,
            "No results data available yet",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12, color="gray", fontstyle="italic",
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_crossover_plot.pdf")
    fig.savefig(FIGURES_DIR / "fig6_crossover_plot.png")
    plt.close(fig)
    print("Figure 6: Crossover plot saved.")


# ════════════════════════════════════════════════════════════
# Figure 7: Cross-Section Recall Heatmap
# ════════════════════════════════════════════════════════════

def generate_fig7() -> None:
    summary = _get_summary_data()

    # Build matrix: rows = strategies, cols = non-abstract conditions
    matrix = np.full((len(STRATEGY_KEYS), len(NON_ABSTRACT_CONDITIONS)), np.nan)
    has_data = False

    if summary is not None:
        for i, strategy in enumerate(STRATEGY_KEYS):
            for j, cond in enumerate(NON_ABSTRACT_CONDITIONS):
                cond_data = summary.get(cond, {}).get(strategy, {})
                val = cond_data.get("cross_section_recall")
                if val is not None:
                    matrix[i, j] = val
                    has_data = True

    fig, ax = plt.subplots(figsize=(5.5, 4))

    if has_data:
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    else:
        # Show empty grid with neutral color
        im = ax.imshow(
            np.zeros_like(matrix), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto",
        )

    # Annotate cells
    for i in range(len(STRATEGY_KEYS)):
        for j in range(len(NON_ABSTRACT_CONDITIONS)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "—"
                color = "gray"
            else:
                text = f"{val:.3f}"
                # Dark text on light cells, light text on dark cells
                color = "white" if val > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(np.arange(len(NON_ABSTRACT_CONDITIONS)))
    ax.set_xticklabels(
        [NON_ABSTRACT_LABELS[c] for c in NON_ABSTRACT_CONDITIONS], fontsize=9,
    )
    ax.set_yticks(np.arange(len(STRATEGY_KEYS)))
    ax.set_yticklabels(
        [STRATEGY_LABELS[s] for s in STRATEGY_KEYS], fontsize=9,
    )
    ax.set_title("Cross-Section Recall by Strategy and Condition")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cross-Section Recall", fontsize=9)

    if not has_data:
        ax.text(
            0.5, 0.5,
            "No results data available yet",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12, color="gray", fontstyle="italic",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_crosssection_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "fig7_crosssection_heatmap.png")
    plt.close(fig)
    print("Figure 7: Cross-section heatmap saved.")


# ════════════════════════════════════════════════════════════
# Figure 8: Section Coverage@5 Bar Chart
# ════════════════════════════════════════════════════════════

def generate_fig8() -> None:
    summary = _get_summary_data()

    # Collect section_coverage_at_5 values
    coverage: dict[str, dict[str, float | None]] = {
        s: {} for s in STRATEGY_KEYS
    }
    has_data = False

    if summary is not None:
        for strategy in STRATEGY_KEYS:
            for cond in NON_ABSTRACT_CONDITIONS:
                cond_data = summary.get(cond, {}).get(strategy, {})
                val = cond_data.get("section_coverage@5") or cond_data.get("section_coverage_at_5")
                coverage[strategy][cond] = val
                if val is not None:
                    has_data = True
    else:
        for strategy in STRATEGY_KEYS:
            for cond in NON_ABSTRACT_CONDITIONS:
                coverage[strategy][cond] = None

    fig, ax = plt.subplots(figsize=(7, 4))

    n_conditions = len(NON_ABSTRACT_CONDITIONS)
    n_strategies = len(STRATEGY_KEYS)
    x = np.arange(n_conditions)
    width = 0.12
    offsets = np.arange(n_strategies) - (n_strategies - 1) / 2

    for i, strategy in enumerate(STRATEGY_KEYS):
        vals = []
        for cond in NON_ABSTRACT_CONDITIONS:
            v = coverage[strategy].get(cond)
            vals.append(v if v is not None else 0.0)
        ax.bar(
            x + offsets[i] * width,
            vals,
            width * 0.9,
            label=STRATEGY_LABELS[strategy],
            color=COLORS[strategy],
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [NON_ABSTRACT_LABELS[c] for c in NON_ABSTRACT_CONDITIONS], fontsize=9,
    )
    ax.set_ylabel("Avg. Distinct Sections in Top-5")
    ax.set_title("Section Coverage@5 by Condition")
    ax.legend(ncol=3, loc="upper left", framealpha=0.9, fontsize=7.5)

    if has_data:
        all_vals = [
            coverage[s][c]
            for s in STRATEGY_KEYS
            for c in NON_ABSTRACT_CONDITIONS
            if coverage[s].get(c) is not None
        ]
        ax.set_ylim(0, max(all_vals) * 1.2 if all_vals else 5)
    else:
        ax.set_ylim(0, 5)
        ax.text(
            0.5, 0.5,
            "No results data available yet",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12, color="gray", fontstyle="italic",
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_section_coverage.pdf")
    fig.savefig(FIGURES_DIR / "fig8_section_coverage.png")
    plt.close(fig)
    print("Figure 8: Section coverage saved.")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    generate_fig6()
    generate_fig7()
    generate_fig8()
    print(f"\nAll full-text figures saved to {FIGURES_DIR}/")
