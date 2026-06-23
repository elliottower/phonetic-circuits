"""Generate hypocorism-oronym circuit overlap figure for slides."""
import json
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def get_top_heads(task, n=30):
    data = json.load(open(f"results/circuits/{task}_gpt2.json"))
    head_scores = defaultdict(float)
    for edge_name, edge_info in data["edges"].items():
        score = abs(edge_info["score"])
        m = re.search(r"a(\d+)\.h(\d+)", edge_name.split("->")[-1])
        if m:
            head_scores[(int(m.group(1)), int(m.group(2)))] += score
    ranked = sorted(head_scores.items(), key=lambda x: -x[1])
    return dict(ranked[:n])


hypo = get_top_heads("op1_hypocorism")
orny = get_top_heads("op4_oronym")

shared = set(hypo) & set(orny)
hypo_only = set(hypo) - set(orny)
orny_only = set(orny) - set(hypo)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

# Left: layer-head grid showing which heads belong to which set
ax = axes[0]
n_layers = 12
n_heads = 12

grid = np.zeros((n_layers, n_heads))  # 0=neither, 1=hypo_only, 2=orny_only, 3=shared

for layer in range(n_layers):
    for head in range(n_heads):
        h = (layer, head)
        if h in shared:
            grid[layer, head] = 3
        elif h in hypo_only:
            grid[layer, head] = 1
        elif h in orny_only:
            grid[layer, head] = 2

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#f0f0f0", "#4ECDC4", "#FF6B6B", "#8B5CF6"])
im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=3)

ax.set_xticks(range(n_heads))
ax.set_yticks(range(n_layers))
ax.set_xticklabels([f"H{i}" for i in range(n_heads)], fontsize=9)
ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)
ax.set_xlabel("Head", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_title("Circuit heads: Hypocorism vs Oronym (top-30 each)", fontsize=13, fontweight="bold")

legend_patches = [
    mpatches.Patch(color="#4ECDC4", label=f"Hypocorism only ({len(hypo_only)})"),
    mpatches.Patch(color="#FF6B6B", label=f"Oronym only ({len(orny_only)})"),
    mpatches.Patch(color="#8B5CF6", label=f"Shared ({len(shared)})"),
    mpatches.Patch(color="#f0f0f0", label="Not in either circuit"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)

for layer in range(n_layers):
    for head in range(n_heads):
        if grid[layer, head] > 0:
            ax.text(head, layer, f"{int(grid[layer, head])}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

# Right: layer distribution bar chart
ax2 = axes[1]
layers = list(range(n_layers))
shared_by_layer = Counter(h[0] for h in shared)
hypo_by_layer = Counter(h[0] for h in hypo_only)
orny_by_layer = Counter(h[0] for h in orny_only)

shared_counts = [shared_by_layer.get(l, 0) for l in layers]
hypo_counts = [hypo_by_layer.get(l, 0) for l in layers]
orny_counts = [orny_by_layer.get(l, 0) for l in layers]

bar_width = 0.6
bottoms_h = [0] * n_layers
bottoms_o = [h for h in hypo_counts]
bottoms_s = [h + o for h, o in zip(hypo_counts, orny_counts)]

ax2.barh(layers, hypo_counts, bar_width, color="#4ECDC4", label="Hypo only")
ax2.barh(layers, orny_counts, bar_width, left=hypo_counts, color="#FF6B6B", label="Orny only")
ax2.barh(layers, shared_counts, bar_width, left=bottoms_s, color="#8B5CF6", label="Shared")

ax2.set_yticks(layers)
ax2.set_yticklabels([f"L{l}" for l in layers], fontsize=9)
ax2.set_xlabel("Number of heads", fontsize=12)
ax2.set_title("Heads per layer", fontsize=13, fontweight="bold")
ax2.legend(fontsize=8)

# Add counts on bars
for l in layers:
    total = hypo_counts[l] + orny_counts[l] + shared_counts[l]
    if total > 0:
        ax2.text(total + 0.1, l, str(total), va="center", fontsize=9)

plt.tight_layout()
plt.savefig("circuit_overlap_hypo_orny.png", dpi=200, bbox_inches="tight")
print("Saved circuit_overlap_hypo_orny.png")
