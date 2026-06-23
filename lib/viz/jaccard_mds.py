"""Jaccard MDS v2 — fix Both(ABC) label below dot, extend y-axis to -0.4."""
import json
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path(__file__).parent
REPO = Path("/Users/elliottower/Documents/GitHub/factorization-circuits")

SUBTASKS = ["s2_io_flip", "s1_io_flip", "abc", "random_names", "full_flip"]
SHORT = {"s2_io_flip": "IO name", "s1_io_flip": "Subject",
         "abc": "Both (ABC)", "random_names": "Random", "full_flip": "Full flip"}
COLORS = {"s2_io_flip": "#e41a1c", "s1_io_flip": "#377eb8", "abc": "#4daf4a",
          "random_names": "#984ea3", "full_flip": "#ff7f00"}

LABEL_BELOW = {"abc"}

das = json.load(open(REPO / "artifacts/per_variable_das/per_variable_das_exploration.json"))
layer5 = das["per_layer"]["5"]
TOP_K = 20

factor_sets = {}
for st in SUBTASKS:
    factor_sets[st] = set(t["factor"] for t in layer5[st]["top_factors"][:TOP_K])

n = len(SUBTASKS)

jaccard_mat = np.zeros((n, n))
for i, a in enumerate(SUBTASKS):
    for j, b in enumerate(SUBTASKS):
        if i == j:
            jaccard_mat[i, j] = 0
        else:
            inter = len(factor_sets[a] & factor_sets[b])
            union = len(factor_sets[a] | factor_sets[b])
            jaccard_mat[i, j] = 1 - inter / union

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
          normalized_stress="auto")
coords = mds.fit_transform(jaccard_mat)

fig, ax = plt.subplots(figsize=(7, 7))

for i in range(n):
    for j in range(i + 1, n):
        inter = len(factor_sets[SUBTASKS[i]] & factor_sets[SUBTASKS[j]])
        if inter == 0:
            continue
        similarity = 1 - jaccard_mat[i, j]
        lw = 1 + inter * 0.8
        alpha = min(0.3 + similarity * 2, 0.9)
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color="#555555", linewidth=lw, alpha=alpha, zorder=1)
        mid_x = (coords[i, 0] + coords[j, 0]) / 2
        mid_y = (coords[i, 1] + coords[j, 1]) / 2
        ax.text(mid_x, mid_y, str(inter), ha="center", va="center",
                fontsize=9, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8),
                zorder=4)

for i, st in enumerate(SUBTASKS):
    ax.scatter(coords[i, 0], coords[i, 1], c=COLORS[st], s=300, zorder=3,
               edgecolors="black", linewidths=1.2)
    if st in LABEL_BELOW:
        ax.text(coords[i, 0], coords[i, 1] - 0.06, SHORT[st],
                ha="center", va="top", fontsize=10, fontweight="bold",
                color=COLORS[st])
    else:
        ax.text(coords[i, 0], coords[i, 1] + 0.06, SHORT[st],
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=COLORS[st])

ax.set_xlabel("MDS dimension 1", fontsize=9)
ax.set_ylabel("MDS dimension 2", fontsize=9)
ax.set_aspect("equal")

xlim = ax.get_xlim()
ylim = ax.get_ylim()
y_lo = -0.8
y_hi = 0.65
x_lo = xlim[0] - 0.15
x_hi = xlim[1] + 0.15
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)

legend_elements = [
    Line2D([0], [0], color="#555555", linewidth=3, alpha=0.6,
           label="Shared factors (number on edge)"),
    Line2D([0], [0], color="white", marker="o", markerfacecolor="#999999",
           markeredgecolor="black", markersize=8,
           label="No edge = 0 shared (chance expects <1)"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
          framealpha=0.9, edgecolor="#cccccc")

ax.set_title("Factor Sharing Between IOI Subtasks\n(MDS of Jaccard distance, DAS top-20, Layer 5)",
             fontsize=11, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "12a_jaccard_mds_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / '12a_jaccard_mds_v2.png'}")
