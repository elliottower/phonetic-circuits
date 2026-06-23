"""CMD comparison across decomposition methods and tasks.
Panel 1: PCA CMD vs k (line plot per task, showing optimal k)
Panel 2: Best-of-each-method bar chart across tasks
Panel 3: Method ranking heatmap (normalized CMD)"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
REPO = Path("/Users/elliottower/Documents/GitHub/factorization-circuits")

d = json.load(open(REPO / "artifacts/tucker_results/all_tasks_sweep.json"))

TASKS = ["ioi", "sva", "greater_than", "gender_bias", "capital_country"]
TASK_LABELS = {"ioi": "IOI", "sva": "SVA", "greater_than": "Greater-than",
               "gender_bias": "Gendered pronoun", "capital_country": "Capital country"}
TASK_COLORS = {"ioi": "#e41a1c", "sva": "#377eb8", "greater_than": "#4daf4a",
               "gender_bias": "#984ea3", "capital_country": "#ff7f00"}

PCA_KS = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# Method groups for best-of comparison
METHOD_GROUPS = {
    "EAP-IG\n(baseline)": lambda entries: [e for e in entries if e["label"] == "standard_eap_ig"],
    "PCA\n(best k)": lambda entries: [e for e in entries if e["label"].startswith("factor_pca_k")],
    "Tucker": lambda entries: [e for e in entries if "tucker" in e["label"]],
    "Soft threshold\n(best σ)": lambda entries: [e for e in entries if e["label"].startswith("soft_thresh")],
    "Top-k/edge\n(best k)": lambda entries: [e for e in entries if e["label"].startswith("topk_per_edge")],
}

fig = plt.figure(figsize=(16, 10))

# --- Panel 1: PCA CMD vs k per task ---
ax1 = fig.add_subplot(2, 2, (1, 2))

for task in TASKS:
    entries = {e["label"]: e for e in d[task]}
    cmds = []
    ks = []
    for k in PCA_KS:
        label = f"factor_pca_k{k}"
        if label in entries:
            cmds.append(entries[label]["cmd"])
            ks.append(k)

    ax1.plot(ks, cmds, "o-", color=TASK_COLORS[task], label=TASK_LABELS[task],
             linewidth=2, markersize=6)

    # Mark the baseline
    baseline = entries["standard_eap_ig"]["cmd"]
    ax1.axhline(baseline, color=TASK_COLORS[task], linewidth=0.8,
                linestyle="--", alpha=0.3)

ax1.set_xscale("log", base=2)
ax1.set_xticks(PCA_KS)
ax1.set_xticklabels([str(k) for k in PCA_KS], fontsize=9)
ax1.set_xlabel("PCA subspace dimension (k)", fontsize=10)
ax1.set_ylabel("CMD (lower = better)", fontsize=10)
ax1.set_title("PCA Circuit Discovery: CMD vs Subspace Dimension",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(axis="y", alpha=0.2)
ax1.set_ylim(0, 0.16)

# Annotate optimal k per task
for task in TASKS:
    entries = {e["label"]: e for e in d[task]}
    best_cmd = 1.0
    best_k = None
    for k in PCA_KS:
        label = f"factor_pca_k{k}"
        if label in entries and entries[label]["cmd"] < best_cmd:
            best_cmd = entries[label]["cmd"]
            best_k = k
    if best_k:
        ax1.annotate(f"k={best_k}", xy=(best_k, best_cmd),
                     xytext=(best_k * 1.3, best_cmd + 0.005),
                     fontsize=7, color=TASK_COLORS[task], fontweight="bold")

# --- Panel 2: Best-of-each-method bar chart ---
ax2 = fig.add_subplot(2, 2, 3)

method_names = list(METHOD_GROUPS.keys())
n_methods = len(method_names)
n_tasks = len(TASKS)
x = np.arange(n_methods)
width = 0.15

for ti, task in enumerate(TASKS):
    best_cmds = []
    for method_name, selector in METHOD_GROUPS.items():
        candidates = selector(d[task])
        if candidates:
            best = min(candidates, key=lambda e: e["cmd"])
            best_cmds.append(best["cmd"])
        else:
            best_cmds.append(np.nan)

    offset = (ti - n_tasks / 2 + 0.5) * width
    bars = ax2.bar(x + offset, best_cmds, width, color=TASK_COLORS[task],
                   edgecolor="black", linewidth=0.3, label=TASK_LABELS[task])

ax2.set_xticks(x)
ax2.set_xticklabels(method_names, fontsize=8)
ax2.set_ylabel("CMD (lower = better)", fontsize=10)
ax2.set_title("Best CMD per Method Family", fontsize=12, fontweight="bold")
ax2.legend(fontsize=7, loc="upper right", ncol=2)
ax2.grid(axis="y", alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# --- Panel 3: Method ranking heatmap ---
ax3 = fig.add_subplot(2, 2, 4)

rank_mat = np.zeros((n_tasks, n_methods))
cmd_mat = np.zeros((n_tasks, n_methods))

for ti, task in enumerate(TASKS):
    best_cmds = []
    for method_name, selector in METHOD_GROUPS.items():
        candidates = selector(d[task])
        if candidates:
            best = min(candidates, key=lambda e: e["cmd"])
            best_cmds.append(best["cmd"])
        else:
            best_cmds.append(np.nan)
    cmd_mat[ti] = best_cmds
    order = np.argsort(best_cmds)
    for rank, idx in enumerate(order):
        rank_mat[ti, idx] = rank + 1

im = ax3.imshow(rank_mat, cmap="RdYlGn_r", vmin=1, vmax=n_methods, aspect="auto")

for i in range(n_tasks):
    for j in range(n_methods):
        rank = int(rank_mat[i, j])
        cmd = cmd_mat[i, j]
        text = f"#{rank}\n{cmd:.3f}"
        color = "white" if rank >= 4 else "black"
        ax3.text(j, i, text, ha="center", va="center", fontsize=8,
                 fontweight="bold" if rank == 1 else "normal", color=color)

ax3.set_xticks(range(n_methods))
ax3.set_xticklabels(method_names, fontsize=7)
ax3.set_yticks(range(n_tasks))
ax3.set_yticklabels([TASK_LABELS[t] for t in TASKS], fontsize=9)
ax3.set_title("Method Ranking per Task\n(#1 = best CMD)", fontsize=12, fontweight="bold")

fig.suptitle("Circuit Discovery Method Comparison (CMD)",
             fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "14_cmd_method_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / '14_cmd_method_comparison.png'}")
