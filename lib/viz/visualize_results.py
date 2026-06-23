"""Visualize phonetic circuit EAP-IG results.

Generates 4 figures from the Modal run results:
1. Summary bar chart (accuracy, CMD, CPR per task)
2. Margin (logit diff) distributions per task
3. Jaccard MDS of top circuit edges across tasks
4. Cross-task circuit overlap heatmap

Usage:
    PYTHONPATH=. python -m lib.viz.visualize_results
    PYTHONPATH=. python -m lib.viz.visualize_results --results-dir results_modal
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.manifold import MDS

REPO = Path(__file__).resolve().parent.parent.parent

TASK_ORDER = ["op1_hypocorism", "op2_clipping", "op3_initialism",
              "op4_oronym", "op5_homophone", "op7_folk_etym"]

TASK_SHORT = {
    "op1_hypocorism": "Hypocorism",
    "op2_clipping": "Clipping",
    "op3_initialism": "Initialism",
    "op4_oronym": "Oronym",
    "op5_homophone": "Homophone",
    "op7_folk_etym": "Folk etym.",
}

TASK_COLORS = {
    "op1_hypocorism": "#e41a1c",
    "op2_clipping": "#377eb8",
    "op3_initialism": "#4daf4a",
    "op4_oronym": "#984ea3",
    "op5_homophone": "#ff7f00",
    "op7_folk_etym": "#a65628",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.6,
})


def load_summary(results_dir: Path) -> dict:
    return json.load(open(results_dir / "summary.json"))


def load_circuit_edges(results_dir: Path, task: str, model: str = "gpt2") -> dict:
    p = results_dir / "circuits" / "EAP-IG_patching_edge" / f"{task}_{model}.json"
    return json.load(open(p))


def get_top_edges(circuit: dict, top_k: int = 200) -> set:
    edges = circuit["edges"]
    scored = [(name, abs(e["score"])) for name, e in edges.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {name for name, _ in scored[:top_k]}


def fig_summary_bars(summary: dict, out_dir: Path):
    tasks = [t for t in TASK_ORDER if t in summary["tasks"]]
    n = len(tasks)
    accs = [summary["tasks"][t]["behavioral"]["accuracy"] for t in tasks]
    cmds = [summary["tasks"][t]["cmd"] for t in tasks]
    mean_lds = [summary["tasks"][t]["behavioral"]["mean_logit_diff"] for t in tasks]
    ns = [summary["tasks"][t]["behavioral"]["n"] for t in tasks]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(n)
    colors = [TASK_COLORS[t] for t in tasks]
    labels = [f"{TASK_SHORT[t]}\n(n={ns[i]})" for i, t in enumerate(tasks)]

    ax = axes[0]
    bars = ax.bar(x, accs, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(n - 0.5, 0.71, "70% gate", fontsize=7, color="gray", ha="right")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Behavioral accuracy", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    bars = ax.bar(x, cmds, color=colors, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(cmds):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("CMD (lower = more faithful)")
    ax.set_title("Circuit minimality distance", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    bars = ax.bar(x, mean_lds, color=colors, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(mean_lds):
        ax.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean logit difference")
    ax.set_title("Mean logit diff (target - foil)", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Phonetic Circuit Discovery — GPT-2 Baseline (EAP-IG)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / "summary_bars.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig_margin_distributions(summary: dict, results_dir: Path, out_dir: Path):
    import sys
    sys.path.insert(0, str(REPO / "reference" / "mib" / "EAP-IG" / "src"))
    sys.path.insert(0, str(REPO / "reference" / "mib"))
    import torch
    from transformer_lens import HookedTransformer
    from lib.dataset import PhoneticDataset

    tasks = [t for t in TASK_ORDER if t in summary["tasks"]]

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    all_margins = {}
    for task in tasks:
        dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer)
        margins = []
        for i in range(len(dataset)):
            clean, corrupted, (tid, fid) = dataset[i]
            tokens = model.to_tokens(clean)
            with torch.no_grad():
                logits = model(tokens)
            margins.append((logits[0, -1, tid] - logits[0, -1, fid]).item())
        all_margins[task] = np.array(margins)

    n = len(tasks)
    fig, axes = plt.subplots(n, 1, figsize=(5, 1.4 * n), sharex=True)
    x_grid = np.linspace(-8, 15, 500)
    threshold = 1.0

    for i, task in enumerate(tasks):
        ax = axes[i]
        margins = all_margins[task]
        color = TASK_COLORS[task]
        n_narrow = int(np.sum(margins < threshold))
        pct = 100 * n_narrow / len(margins)

        kde = gaussian_kde(margins, bw_method=0.25)
        density = kde(x_grid)
        narrow_mask = x_grid <= threshold

        ax.fill_between(x_grid[narrow_mask], density[narrow_mask], alpha=0.55, color=color)
        ax.plot(x_grid, density, color=color, linewidth=0.9, alpha=0.7)
        ax.axvline(threshold, color="#444444", linestyle="--", linewidth=0.5, alpha=0.5)

        handle = mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.55)
        label = f"{TASK_SHORT[task]}  {n_narrow}/{len(margins)} hard ({pct:.0f}%)"
        ax.legend([handle], [label], loc="upper right", fontsize=7,
                  framealpha=0.95, edgecolor="#cccccc", fancybox=False,
                  borderpad=0.3, handlelength=1.0, handletextpad=0.4)

        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel(r"Logit margin ($\mathrm{logit}_{\mathrm{target}} - \mathrm{logit}_{\mathrm{foil}}$)")
    axes[0].set_title("Margin distributions by task", fontsize=11, fontweight="bold")

    fig.tight_layout(h_pad=0.2, pad=0.5)
    out = out_dir / "margin_distributions.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig_jaccard_mds(summary: dict, results_dir: Path, out_dir: Path, top_k: int = 200):
    from matplotlib.lines import Line2D

    tasks = [t for t in TASK_ORDER if t in summary["tasks"]]
    edge_sets = {}
    for task in tasks:
        circuit = load_circuit_edges(results_dir, task)
        edge_sets[task] = get_top_edges(circuit, top_k=top_k)

    n = len(tasks)
    jaccard_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            inter = len(edge_sets[tasks[i]] & edge_sets[tasks[j]])
            union = len(edge_sets[tasks[i]] | edge_sets[tasks[j]])
            jaccard_mat[i, j] = 1 - inter / union if union > 0 else 1.0

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto")
    coords = mds.fit_transform(jaccard_mat)

    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(n):
        for j in range(i + 1, n):
            inter = len(edge_sets[tasks[i]] & edge_sets[tasks[j]])
            if inter == 0:
                continue
            similarity = 1 - jaccard_mat[i, j]
            lw = 1 + inter * 0.15
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

    LABEL_OFFSETS = {
        "op3_initialism": (0, -0.07, "center", "top"),
        "op4_oronym": (-0.07, 0, "right", "center"),
        "op2_clipping": (0, -0.07, "center", "top"),
        "op5_homophone": (0.07, 0, "left", "center"),
    }
    DEFAULT_OFFSET = (0, 0.07, "center", "bottom")

    for i, task in enumerate(tasks):
        ax.scatter(coords[i, 0], coords[i, 1], c=TASK_COLORS[task], s=350, zorder=3,
                   edgecolors="black", linewidths=1.2)
        dx, dy, ha, va = LABEL_OFFSETS.get(task, DEFAULT_OFFSET)
        ax.text(coords[i, 0] + dx, coords[i, 1] + dy, TASK_SHORT[task],
                ha=ha, va=va, fontsize=10, fontweight="bold",
                color=TASK_COLORS[task])

    ax.set_xlabel("MDS dimension 1", fontsize=9)
    ax.set_ylabel("MDS dimension 2", fontsize=9)
    ax.set_aspect("equal")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] - 0.2, xlim[1] + 0.2)
    ax.set_ylim(ylim[0] - 0.2, ylim[1] + 0.1)

    legend_elements = [
        Line2D([0], [0], color="#555555", linewidth=3, alpha=0.6,
               label=f"Shared edges (top-{top_k}, count on line)"),
        Line2D([0], [0], color="white", marker="o", markerfacecolor="#999999",
               markeredgecolor="black", markersize=8,
               label="No line = 0 shared edges"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
              framealpha=0.9, edgecolor="#cccccc")
    ax.set_title(f"Circuit Sharing Between Phonetic Tasks\n(MDS of Jaccard distance, EAP-IG top-{top_k} edges)",
                 fontsize=11, fontweight="bold")

    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(top=0.92)
    out = out_dir / "jaccard_mds.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig_overlap_heatmap(summary: dict, results_dir: Path, out_dir: Path, top_k: int = 200):
    tasks = [t for t in TASK_ORDER if t in summary["tasks"]]
    edge_sets = {}
    for task in tasks:
        circuit = load_circuit_edges(results_dir, task)
        edge_sets[task] = get_top_edges(circuit, top_k=top_k)

    n = len(tasks)
    overlap_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(edge_sets[tasks[i]] & edge_sets[tasks[j]])
            overlap_mat[i, j] = inter / top_k

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(overlap_mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            v = overlap_mat[i, j]
            color = "white" if v > 0.6 else "black"
            count = int(v * top_k)
            ax.text(j, i, f"{v:.0%}\n({count})", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if i == j else "normal")

    short_labels = [TASK_SHORT[t] for t in tasks]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Overlap fraction (top-{top_k} edges)", fontsize=9)

    ax.set_title(f"Cross-Task Circuit Overlap\n(EAP-IG top-{top_k} edges, GPT-2)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "overlap_heatmap.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig_top_edges_by_layer(summary: dict, results_dir: Path, out_dir: Path, top_k: int = 100):
    tasks = [t for t in TASK_ORDER if t in summary["tasks"]]
    n_layers = 12

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        circuit = load_circuit_edges(results_dir, task)
        edges = circuit["edges"]
        scored = [(name, abs(e["score"])) for name, e in edges.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        layer_counts = np.zeros(n_layers)
        for name, _ in top:
            parts = name.split("->")
            dst = parts[1] if len(parts) > 1 else parts[0]
            for l in range(n_layers):
                if f"a{l}." in dst or f"m{l}" in dst:
                    layer_counts[l] += 1
                    break

        ax.bar(range(n_layers), layer_counts, color=TASK_COLORS[task],
               edgecolor="black", linewidth=0.3)
        ax.set_title(TASK_SHORT[task], fontsize=10, fontweight="bold",
                     color=TASK_COLORS[task])
        ax.set_xlabel("Layer" if idx >= 3 else "")
        if idx % 3 == 0:
            ax.set_ylabel(f"Edges in top-{top_k}")
        ax.set_xticks(range(n_layers))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Layer Distribution of Top-{top_k} Circuit Edges (EAP-IG, GPT-2)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "top_edges_by_layer.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_modal")
    parser.add_argument("--out-dir", type=str, default="results_modal/plots")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--skip-margins", action="store_true",
                        help="Skip margin distributions (requires model load)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(results_dir)

    print("=== Generating visualizations ===")
    fig_summary_bars(summary, out_dir)
    fig_overlap_heatmap(summary, results_dir, out_dir, top_k=args.top_k)
    fig_jaccard_mds(summary, results_dir, out_dir, top_k=args.top_k)
    fig_top_edges_by_layer(summary, results_dir, out_dir, top_k=min(args.top_k, 100))

    if not args.skip_margins:
        fig_margin_distributions(summary, results_dir, out_dir)

    print(f"\nAll plots saved to {out_dir}/")
