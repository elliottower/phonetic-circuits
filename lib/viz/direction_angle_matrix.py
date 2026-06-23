"""Viz #6 — Cross-task DAS direction cosine/angle matrix.
Two panels: (a) geodesic distance heatmap, (b) principal angle profile per pair.
Shows trained causal directions are near-orthogonal for unrelated tasks."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
REPO = Path("/Users/elliottower/Documents/GitHub/factorization-circuits")

SUBTASKS = ["s2_io_flip", "s1_io_flip", "abc", "random_names", "full_flip"]
SHORT = {"s2_io_flip": "IO name", "s1_io_flip": "Subject",
         "abc": "Both (ABC)", "random_names": "Random", "full_flip": "Full flip"}

das = json.load(open(REPO / "artifacts/per_variable_das/per_variable_das_exploration.json"))

# --- Panel (a): Geodesic distance heatmap at layer 5 ---
fig, (ax_heat, ax_angles) = plt.subplots(1, 2, figsize=(14, 5.5),
                                          gridspec_kw={"width_ratios": [1, 1.3]})

layer5_angles = das["per_layer"]["5"]["subspace_angles"]
n = len(SUBTASKS)
geo_mat = np.zeros((n, n))
for i, a in enumerate(SUBTASKS):
    for j, b in enumerate(SUBTASKS):
        if i == j:
            geo_mat[i, j] = 0
        else:
            key = f"{a}_vs_{b}" if f"{a}_vs_{b}" in layer5_angles else f"{b}_vs_{a}"
            geo_mat[i, j] = layer5_angles[key]["geodesic_distance"]

im = ax_heat.imshow(geo_mat, cmap="RdYlGn_r", vmin=0, vmax=2.5)
ax_heat.set_xticks(range(n))
ax_heat.set_xticklabels([SHORT[s] for s in SUBTASKS], fontsize=9, rotation=30, ha="right")
ax_heat.set_yticks(range(n))
ax_heat.set_yticklabels([SHORT[s] for s in SUBTASKS], fontsize=9)

for i in range(n):
    for j in range(n):
        color = "white" if geo_mat[i, j] > 1.5 else "black"
        ax_heat.text(j, i, f"{geo_mat[i, j]:.2f}", ha="center", va="center",
                     fontsize=10, fontweight="bold", color=color)

cb = plt.colorbar(im, ax=ax_heat, shrink=0.8)
cb.set_label("Geodesic distance (rad)", fontsize=9)
cb.ax.axhline(np.pi / 2, color="black", linewidth=1.5, linestyle="--")
cb.ax.text(1.5, np.pi / 2, "π/2", fontsize=8, va="center", fontweight="bold")

ax_heat.set_title("DAS Subspace Geodesic Distances\n(Layer 5, k=4)", fontsize=11, fontweight="bold")

# --- Panel (b): Principal angle profiles ---
# Show each pair's 4 principal angles as connected dots
# Highlight IO↔Full (most aligned) and Subject↔Full (most orthogonal)
pairs_to_show = [
    ("s2_io_flip", "full_flip", "#e41a1c", "IO ↔ Full", 2.5),
    ("s2_io_flip", "abc", "#4daf4a", "IO ↔ ABC", 1.5),
    ("s2_io_flip", "s1_io_flip", "#377eb8", "IO ↔ Subj", 1.5),
    ("s1_io_flip", "full_flip", "#984ea3", "Subj ↔ Full", 1.5),
    ("abc", "random_names", "#ff7f00", "ABC ↔ Rand", 1.5),
    ("s1_io_flip", "random_names", "#888888", "Subj ↔ Rand", 1.2),
]

for a, b, color, label, lw in pairs_to_show:
    key = f"{a}_vs_{b}" if f"{a}_vs_{b}" in layer5_angles else f"{b}_vs_{a}"
    angles = layer5_angles[key]["principal_angles_deg"]
    ax_angles.plot(range(1, len(angles) + 1), angles, "o-", color=color,
                   label=label, linewidth=lw, markersize=6)

ax_angles.axhline(90, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax_angles.text(4.15, 90, "orthogonal", fontsize=7, va="bottom", color="#666666")
ax_angles.axhline(45, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)

ax_angles.set_xlabel("Principal angle index", fontsize=10)
ax_angles.set_ylabel("Principal angle (degrees)", fontsize=10)
ax_angles.set_xticks([1, 2, 3, 4])
ax_angles.set_ylim(0, 95)
ax_angles.set_xlim(0.7, 4.3)
ax_angles.legend(fontsize=8, loc="lower right", ncol=2)
ax_angles.set_title("Principal Angles Between DAS Directions\n(Layer 5, k=4)",
                     fontsize=11, fontweight="bold")
ax_angles.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / "10_das_direction_angles.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / '10_das_direction_angles.png'}")
