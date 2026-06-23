"""
Comprehensive circuit analysis experiments for phonetic-circuits slides.

Experiments:
1. Full pairwise circuit overlap (all 15 task pairs, head + edge level)
2. Act patching vs EAP-IG head ranking correlation
3. Causal discovery edge overlap with EAP-IG circuits
4. Sub-circuit containment analysis (which circuits contain which)
5. Layer-resolved Jaccard (per-layer overlap between tasks)
6. Hub head analysis: how many tasks use each head
"""
import json
import re
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

TASKS = [
    ("op1_hypocorism", "Hypo"),
    ("op2_clipping", "Clip"),
    ("op3_initialism", "Init"),
    ("op4_oronym", "Orny"),
    ("op5_homophone", "Homo"),
    ("op6_folk_etym", "Folk"),
]
TASK_IDS = [t[0] for t in TASKS]
TASK_LABELS = [t[1] for t in TASKS]


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


def get_top_edges(task, n=200):
    data = json.load(open(f"results/circuits/{task}_gpt2.json"))
    edges = [(k, abs(v["score"])) for k, v in data["edges"].items()]
    edges.sort(key=lambda x: -x[1])
    return set(e[0] for e in edges[:n])


def get_act_patching_ranking(task):
    data = json.load(open(f"results/act_patching/node/{task}_gpt2.json"))
    head_effects = {}
    for head_name, info in data.items():
        m = re.match(r"a(\d+)\.h(\d+)", head_name)
        if m:
            head_effects[(int(m.group(1)), int(m.group(2)))] = abs(info["effect"])
    return head_effects


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# =========================================
# Experiment 1: Full pairwise Jaccard matrix
# =========================================
print("=== Experiment 1: Full pairwise Jaccard (head + edge) ===")
all_heads = {tid: set(get_top_heads(tid, 30).keys()) for tid in TASK_IDS}
all_edges = {tid: get_top_edges(tid, 200) for tid in TASK_IDS}

n = len(TASKS)
head_jaccard = np.zeros((n, n))
edge_jaccard = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        head_jaccard[i, j] = jaccard(all_heads[TASK_IDS[i]], all_heads[TASK_IDS[j]])
        edge_jaccard[i, j] = jaccard(all_edges[TASK_IDS[i]], all_edges[TASK_IDS[j]])

print("Head-level Jaccard:")
for i in range(n):
    row = " ".join(f"{head_jaccard[i,j]:.2f}" for j in range(n))
    print(f"  {TASK_LABELS[i]:5s}: {row}")

print("\nEdge-level Jaccard:")
for i in range(n):
    row = " ".join(f"{edge_jaccard[i,j]:.2f}" for j in range(n))
    print(f"  {TASK_LABELS[i]:5s}: {row}")

# =========================================
# Experiment 2: EAP-IG vs Act Patching rank correlation
# =========================================
print("\n=== Experiment 2: EAP-IG vs Act Patching rank correlation ===")
from scipy.stats import spearmanr, kendalltau

for tid, label in TASKS:
    eap_heads = get_top_heads(tid, 144)  # all heads
    ap_heads = get_act_patching_ranking(tid)

    common_heads = set(eap_heads.keys()) & set(ap_heads.keys())
    eap_scores = [eap_heads[h] for h in common_heads]
    ap_scores = [ap_heads[h] for h in common_heads]

    rho, p_rho = spearmanr(eap_scores, ap_scores)
    tau, p_tau = kendalltau(eap_scores, ap_scores)

    # Top-30 overlap
    eap_top30 = set(list(get_top_heads(tid, 30).keys()))
    ap_ranked = sorted(ap_heads.items(), key=lambda x: -x[1])
    ap_top30 = set(h for h, _ in ap_ranked[:30])
    overlap = len(eap_top30 & ap_top30)

    print(f"  {label}: Spearman={rho:.3f} (p={p_rho:.1e}), Kendall={tau:.3f}, Top-30 overlap={overlap}/30")


# =========================================
# Experiment 3: Causal discovery edges vs EAP-IG
# =========================================
print("\n=== Experiment 3: Causal discovery hub heads vs EAP-IG top heads ===")
cd_data = json.load(open("results/causal_discovery/causal_discovery_gpt2.json"))

for tid, label in TASKS:
    if tid not in cd_data:
        continue
    cd_task = cd_data[tid]
    cd_heads = set()
    for edge in cd_task["pc"]["edges"]:
        for node in edge[:2]:
            m = re.match(r"a(\d+)\.h(\d+)", node)
            if m:
                cd_heads.add((int(m.group(1)), int(m.group(2))))

    eap_top30 = set(get_top_heads(tid, 30).keys())
    overlap = cd_heads & eap_top30
    print(f"  {label}: CD has {len(cd_heads)} heads in graph, {len(overlap)} overlap with EAP top-30 ({len(overlap)/max(1,len(cd_heads))*100:.0f}%)")


# =========================================
# Experiment 4: Sub-circuit containment
# =========================================
print("\n=== Experiment 4: Sub-circuit containment (A ⊂ B analysis) ===")
for i, (tid_a, label_a) in enumerate(TASKS):
    for j, (tid_b, label_b) in enumerate(TASKS):
        if i == j:
            continue
        heads_a = all_heads[tid_a]
        heads_b = all_heads[tid_b]
        contained = len(heads_a & heads_b) / len(heads_a)
        if contained > 0.6:
            print(f"  {label_a} ⊂ {label_b}: {len(heads_a & heads_b)}/{len(heads_a)} = {contained:.0%}")


# =========================================
# Experiment 5: Layer-resolved Jaccard
# =========================================
print("\n=== Experiment 5: Layer-resolved overlap ===")
layer_heads = {}
for tid in TASK_IDS:
    heads = set(get_top_heads(tid, 30).keys())
    by_layer = defaultdict(set)
    for l, h in heads:
        by_layer[l].add((l, h))
    layer_heads[tid] = by_layer

# For each layer, compute average Jaccard across all task pairs
for layer in range(12):
    jaccards = []
    for i, j in combinations(range(n), 2):
        a = layer_heads[TASK_IDS[i]].get(layer, set())
        b = layer_heads[TASK_IDS[j]].get(layer, set())
        if a or b:
            jaccards.append(jaccard(a, b))
    if jaccards:
        print(f"  Layer {layer:2d}: mean Jaccard={np.mean(jaccards):.3f} (n_pairs={len(jaccards)})")


# =========================================
# Experiment 6: Universal head usage frequency
# =========================================
print("\n=== Experiment 6: Head usage frequency across tasks ===")
head_count = Counter()
for tid in TASK_IDS:
    for h in all_heads[tid]:
        head_count[h] += 1

print("Heads in ALL 6 tasks:")
universal = [(h, c) for h, c in head_count.items() if c == 6]
for h, c in sorted(universal):
    print(f"  L{h[0]}.H{h[1]}")

print(f"\nHeads in 5+ tasks: {sum(1 for c in head_count.values() if c >= 5)}")
print(f"Heads in 4+ tasks: {sum(1 for c in head_count.values() if c >= 4)}")
print(f"Heads in 3+ tasks: {sum(1 for c in head_count.values() if c >= 3)}")
print(f"Heads in exactly 1 task: {sum(1 for c in head_count.values() if c == 1)}")

# Which heads are unique to each task?
print("\nTask-unique heads (in exactly 1 task):")
for tid, label in TASKS:
    unique = [h for h in all_heads[tid] if head_count[h] == 1]
    if unique:
        layers = [f"L{h[0]}.H{h[1]}" for h in sorted(unique)]
        print(f"  {label}: {', '.join(layers)}")


# =========================================
# Experiment 7: DAS IIA comparison across tasks/layers
# =========================================
print("\n=== Experiment 7: DAS IIA by task and layer ===")
das = json.load(open("results/das/das_k1_gpt2.json"))
for tid, label in TASKS:
    if tid not in das:
        continue
    task_das = das[tid]
    layers = task_das["layers"]
    best_layer = max(layers.keys(), key=lambda l: layers[l]["das_iia"])
    best_iia = layers[best_layer]["das_iia"]
    random_iia = layers[best_layer]["random_iia"]
    pca_iia = layers[best_layer]["pca_iia"]
    print(f"  {label}: best DAS IIA={best_iia:.3f} at layer {best_layer} (random={random_iia:.3f}, PCA={pca_iia:.3f})")


# =========================================
# FIGURES
# =========================================

# Figure 1: Head usage heatmap (which heads appear in how many tasks)
fig, ax = plt.subplots(figsize=(10, 5))
grid = np.zeros((12, 12))
for (layer, head), count in head_count.items():
    grid[layer, head] = count
im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="lower", vmin=0, vmax=6)
ax.set_xticks(range(12))
ax.set_yticks(range(12))
ax.set_xticklabels([f"H{i}" for i in range(12)], fontsize=9)
ax.set_yticklabels([f"L{i}" for i in range(12)], fontsize=9)
ax.set_xlabel("Head", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_title("Head usage across 6 phonological tasks (top-30 per task)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Number of tasks using this head")
for layer in range(12):
    for head in range(12):
        if grid[layer, head] > 0:
            color = "white" if grid[layer, head] >= 4 else "black"
            ax.text(head, layer, f"{int(grid[layer, head])}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
plt.tight_layout()
plt.savefig("slides/figures/head_usage_heatmap.png", dpi=200, bbox_inches="tight")
print("\nSaved head_usage_heatmap.png")

# Figure 2: EAP vs Act Patching scatter for each task
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for idx, (tid, label) in enumerate(TASKS):
    ax = axes[idx // 3][idx % 3]
    eap_heads = get_top_heads(tid, 144)
    ap_heads = get_act_patching_ranking(tid)
    common = sorted(set(eap_heads.keys()) & set(ap_heads.keys()))

    eap_vals = np.array([eap_heads[h] for h in common])
    ap_vals = np.array([ap_heads[h] for h in common])

    # Normalize
    eap_norm = eap_vals / (eap_vals.max() + 1e-10)
    ap_norm = ap_vals / (ap_vals.max() + 1e-10)

    # Color by layer
    layers = np.array([h[0] for h in common])
    scatter = ax.scatter(eap_norm, ap_norm, c=layers, cmap="viridis", s=20, alpha=0.7, vmin=0, vmax=11)

    rho, _ = spearmanr(eap_vals, ap_vals)
    ax.set_title(f"{label} (ρ={rho:.2f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("EAP-IG (normalized)", fontsize=9)
    ax.set_ylabel("Act Patching (normalized)", fontsize=9)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

plt.colorbar(scatter, ax=axes, label="Layer", shrink=0.6)
fig.suptitle("EAP-IG vs Activation Patching head importance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("slides/figures/eap_vs_actpatch.png", dpi=200, bbox_inches="tight")
print("Saved eap_vs_actpatch.png")

# Figure 3: Layer-resolved Jaccard heatmap
fig, ax = plt.subplots(figsize=(8, 5))
layer_jaccard_matrix = np.zeros((12, 12))
layer_count = np.zeros((12, 12))
for layer in range(12):
    for i, j in combinations(range(n), 2):
        a = layer_heads[TASK_IDS[i]].get(layer, set())
        b = layer_heads[TASK_IDS[j]].get(layer, set())
        if a or b:
            layer_jaccard_matrix[layer, layer] += jaccard(a, b)
            layer_count[layer, layer] += 1

# Actually make it: for each layer, for each task pair
layer_pair_jaccard = np.zeros((12, len(list(combinations(range(n), 2)))))
pair_labels = []
for pidx, (i, j) in enumerate(combinations(range(n), 2)):
    pair_labels.append(f"{TASK_LABELS[i]}-{TASK_LABELS[j]}")
    for layer in range(12):
        a = layer_heads[TASK_IDS[i]].get(layer, set())
        b = layer_heads[TASK_IDS[j]].get(layer, set())
        if a or b:
            layer_pair_jaccard[layer, pidx] = jaccard(a, b)

im = ax.imshow(layer_pair_jaccard, cmap="YlOrRd", aspect="auto", origin="lower", vmin=0, vmax=1)
ax.set_yticks(range(12))
ax.set_yticklabels([f"L{i}" for i in range(12)], fontsize=9)
ax.set_xticks(range(len(pair_labels)))
ax.set_xticklabels(pair_labels, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("Layer", fontsize=12)
ax.set_title("Layer-resolved Jaccard overlap between task pairs", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Jaccard similarity")
plt.tight_layout()
plt.savefig("slides/figures/layer_resolved_jaccard.png", dpi=200, bbox_inches="tight")
print("Saved layer_resolved_jaccard.png")

# Figure 4: Containment matrix (what % of circuit A is in circuit B)
fig, ax = plt.subplots(figsize=(7, 6))
containment = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            containment[i, j] = 1.0
        else:
            containment[i, j] = len(all_heads[TASK_IDS[i]] & all_heads[TASK_IDS[j]]) / len(all_heads[TASK_IDS[i]])

im = ax.imshow(containment, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(TASK_LABELS, fontsize=11)
ax.set_yticklabels(TASK_LABELS, fontsize=11)
ax.set_xlabel("Circuit B (containing)", fontsize=12)
ax.set_ylabel("Circuit A (contained)", fontsize=12)
ax.set_title("Sub-circuit containment: % of A's heads in B", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Fraction of A in B")
for i in range(n):
    for j in range(n):
        color = "white" if containment[i, j] > 0.6 else "black"
        ax.text(j, i, f"{containment[i, j]:.0%}", ha="center", va="center", fontsize=10, color=color)
plt.tight_layout()
plt.savefig("slides/figures/containment_matrix.png", dpi=200, bbox_inches="tight")
print("Saved containment_matrix.png")

# Figure 5: DAS IIA by layer for each task
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.Set2(np.linspace(0, 1, len(TASKS)))
for idx, (tid, label) in enumerate(TASKS):
    if tid not in das:
        continue
    task_das = das[tid]
    layers_data = task_das["layers"]
    layer_nums = sorted(int(l) for l in layers_data.keys())
    iias = [layers_data[str(l)]["das_iia"] for l in layer_nums]
    ax.plot(layer_nums, iias, "o-", color=colors[idx], label=label, linewidth=2, markersize=6)

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("DAS IIA", fontsize=12)
ax.set_title("DAS Interchange Intervention Accuracy by layer", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xticks(range(0, 12, 2))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("slides/figures/das_iia_by_layer.png", dpi=200, bbox_inches="tight")
print("Saved das_iia_by_layer.png")

print("\n=== All experiments complete ===")
