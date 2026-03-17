"""Generate publication-quality figures for the GraLC-RAG paper."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Publication style
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

# ── Data ──
strategies = [
    "Naive",
    "Semantic",
    "Late\nChunking",
    "Structure-\nAware",
    "GraLC-RAG\n(KG)",
    "GraLC-RAG\n(+Graph)",
]
strategies_short = ["Naive", "Semantic", "Late Chunk.", "Struct.-Aware", "GraLC (KG)", "GraLC (+Graph)"]

mrr   = [0.9787, 0.9802, 0.9768, 0.9765, 0.9687, 0.9502]
r1    = [0.9690, 0.9710, 0.9660, 0.9660, 0.9520, 0.9260]
r3    = [0.9880, 0.9880, 0.9860, 0.9850, 0.9830, 0.9700]
r5    = [0.9900, 0.9910, 0.9920, 0.9890, 0.9880, 0.9820]
r10   = [0.9960, 0.9950, 0.9960, 0.9940, 0.9930, 0.9930]

index_strategies = ["Naive", "Semantic", "Late Chunking", "Struct.-Aware", "GraLC-RAG"]
index_chunks = [1109, 6765, 3055, 2139, 2139]
index_time   = [44.6, 535.1, 313.7, 1044.1, 2566.1]

# Ablation
abl_configs = ["Late Chunk.\n(base)", "+ Structure\nBoundaries", "+ KG\nInfusion", "+ Graph\nRetrieval"]
abl_mrr     = [0.9768, 0.9765, 0.9687, 0.9502]
abl_delta   = [0.0, -0.0003, -0.0081, -0.0266]

# Colors
COLORS = {
    "naive": "#4C72B0",
    "semantic": "#55A868",
    "late": "#C44E52",
    "struct": "#8172B2",
    "gralc_kg": "#CCB974",
    "gralc_graph": "#64B5CD",
}
color_list = list(COLORS.values())


# ════════════════════════════════════════════════════════════
# Figure 1: Retrieval MRR comparison (bar chart)
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(strategies_short))
bars = ax.bar(x, mrr, color=color_list, edgecolor="white", linewidth=0.8, width=0.65)

# Value labels on bars
for bar, val in zip(bars, mrr):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Mean Reciprocal Rank (MRR)")
ax.set_xticks(x)
ax.set_xticklabels(strategies_short, fontsize=8.5)
ax.set_ylim(0.94, 0.99)
ax.set_title("Retrieval Performance on PubMedQA* (1,000 Questions)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

# Highlight best
bars[1].set_edgecolor("black")
bars[1].set_linewidth(1.5)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_mrr_comparison.pdf")
fig.savefig(FIGURES_DIR / "fig1_mrr_comparison.png")
plt.close(fig)
print("Figure 1: MRR comparison saved.")


# ════════════════════════════════════════════════════════════
# Figure 2: Recall@k grouped bar chart
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 3.8))
k_labels = ["R@1", "R@3", "R@5", "R@10"]
recall_data = np.array([r1, r3, r5, r10]).T  # shape (6, 4)

x = np.arange(len(k_labels))
width = 0.12
offsets = np.arange(len(strategies_short)) - (len(strategies_short) - 1) / 2

for i, (strat, color) in enumerate(zip(strategies_short, color_list)):
    ax.bar(x + offsets[i] * width, recall_data[i], width * 0.9,
           label=strat, color=color, edgecolor="white", linewidth=0.4)

ax.set_ylabel("Recall")
ax.set_xticks(x)
ax.set_xticklabels(k_labels)
ax.set_ylim(0.92, 1.005)
ax.set_title("Recall@k Across Chunking Strategies")
ax.legend(ncol=3, loc="lower right", framealpha=0.9, fontsize=7.5)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig2_recall_at_k.pdf")
fig.savefig(FIGURES_DIR / "fig2_recall_at_k.png")
plt.close(fig)
print("Figure 2: Recall@k saved.")


# ════════════════════════════════════════════════════════════
# Figure 3: Ablation waterfall chart
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.5))

abl_colors = ["#55A868", "#8172B2", "#CCB974", "#64B5CD"]
x = np.arange(len(abl_configs))
bars = ax.bar(x, abl_mrr, color=abl_colors, edgecolor="white", linewidth=0.8, width=0.55)

for bar, val, delta in zip(bars, abl_mrr, abl_delta):
    label = f"{val:.4f}"
    if delta != 0:
        label += f"\n({delta:+.4f})"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0008,
            label, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_ylabel("MRR")
ax.set_xticks(x)
ax.set_xticklabels(abl_configs, fontsize=8)
ax.set_ylim(0.94, 0.985)
ax.set_title("Ablation: Incremental Effect of GraLC-RAG Components")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

# Dashed line at base
ax.axhline(y=abl_mrr[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig3_ablation.pdf")
fig.savefig(FIGURES_DIR / "fig3_ablation.png")
plt.close(fig)
print("Figure 3: Ablation saved.")


# ════════════════════════════════════════════════════════════
# Figure 4: Indexing efficiency (dual-axis: chunks + time)
# ════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(6, 3.5))

x = np.arange(len(index_strategies))
width = 0.35

eff_colors_chunks = "#4C72B0"
eff_colors_time = "#C44E52"

bars1 = ax1.bar(x - width / 2, index_chunks, width, label="Chunks",
                color=eff_colors_chunks, edgecolor="white", alpha=0.85)
ax1.set_ylabel("Number of Chunks", color=eff_colors_chunks)
ax1.tick_params(axis="y", labelcolor=eff_colors_chunks)
ax1.set_ylim(0, 8000)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width / 2, index_time, width, label="Index Time (s)",
                color=eff_colors_time, edgecolor="white", alpha=0.85)
ax2.set_ylabel("Index Time (seconds)", color=eff_colors_time)
ax2.tick_params(axis="y", labelcolor=eff_colors_time)
ax2.set_ylim(0, 3000)

ax1.set_xticks(x)
ax1.set_xticklabels(index_strategies, fontsize=8.5)
ax1.set_title("Indexing Efficiency (200 Full-Text Articles, CPU)")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig4_efficiency.pdf")
fig.savefig(FIGURES_DIR / "fig4_efficiency.png")
plt.close(fig)
print("Figure 4: Efficiency saved.")


# ════════════════════════════════════════════════════════════
# Figure 5: Framework architecture diagram (simplified)
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("GraLC-RAG Framework Architecture", fontsize=12, fontweight="bold", pad=10)

box_props = dict(boxstyle="round,pad=0.4", facecolor="#E8EAF6", edgecolor="#3F51B5", linewidth=1.2)
box_kg    = dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.2)
box_out   = dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.2)
arrow = dict(arrowstyle="->,head_width=0.3", color="#333333", lw=1.5)

# Boxes
ax.text(1.5, 5.2, "Biomedical\nDocument", ha="center", va="center", fontsize=9, bbox=box_props)
ax.text(4.2, 5.2, "Document\nStructure Graph", ha="center", va="center", fontsize=8.5, bbox=box_props)
ax.text(7.2, 5.2, "UMLS Knowledge\nSubgraph", ha="center", va="center", fontsize=8.5, bbox=box_kg)

ax.text(1.5, 3.5, "Full-Document\nTransformer\nEncoding", ha="center", va="center", fontsize=8.5, bbox=box_props)
ax.text(4.2, 3.5, "Structure-Aware\nBoundary\nDetection", ha="center", va="center", fontsize=8.5, bbox=box_props)
ax.text(7.2, 3.5, "KG Infusion\n(GAT + Fusion)", ha="center", va="center", fontsize=8.5, bbox=box_kg)

ax.text(4.2, 1.8, "Graph-Enriched\nChunk Embeddings", ha="center", va="center", fontsize=9, fontweight="bold", bbox=box_out)
ax.text(8.0, 1.8, "Graph-Guided\nRetrieval", ha="center", va="center", fontsize=8.5, bbox=box_out)

# Arrows
ax.annotate("", xy=(1.5, 4.7), xytext=(1.5, 4.1), arrowprops=arrow)
ax.annotate("", xy=(4.2, 4.7), xytext=(4.2, 4.1), arrowprops=arrow)
ax.annotate("", xy=(7.2, 4.7), xytext=(7.2, 4.1), arrowprops=arrow)
ax.annotate("", xy=(2.7, 3.5), xytext=(3.0, 3.5), arrowprops=arrow)
ax.annotate("", xy=(5.5, 3.5), xytext=(5.9, 3.5), arrowprops=arrow)
ax.annotate("", xy=(4.2, 3.0), xytext=(4.2, 2.3), arrowprops=arrow)
ax.annotate("", xy=(5.5, 1.8), xytext=(6.7, 1.8), arrowprops=arrow)

# Stage labels
ax.text(0.3, 5.2, "1", fontsize=14, fontweight="bold", color="#3F51B5", ha="center")
ax.text(0.3, 3.5, "2", fontsize=14, fontweight="bold", color="#3F51B5", ha="center")
ax.text(9.3, 3.5, "3", fontsize=14, fontweight="bold", color="#E65100", ha="center")
ax.text(2.8, 1.8, "4", fontsize=14, fontweight="bold", color="#2E7D32", ha="center")
ax.text(9.3, 1.8, "5", fontsize=14, fontweight="bold", color="#2E7D32", ha="center")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig5_architecture.pdf")
fig.savefig(FIGURES_DIR / "fig5_architecture.png")
plt.close(fig)
print("Figure 5: Architecture diagram saved.")


print(f"\nAll figures saved to {FIGURES_DIR}/")
