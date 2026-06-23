"""Figure 1: Method progression bar chart with IOI + numeric labels.

Visual categories:
  - Untrained (Constr. PCA, Δ-PCA): hatched bars, light color
  - Baseline (Vanilla DAS): gray
  - Ours trained (CPCA unconstr., Δ-init fac., CPCA-init fac.): solid task color

Skips methods with no data (IOI missing some ablations).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path("/Users/elliottower/Documents/GitHub/factorization-circuits")
OUT = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

TASK_LABELS = {
    "ioi": "IOI",
    "sva": "SVA",
    "gender_bias": "Gender Bias",
    "capital_country": "Capital–Country",
}

METHOD_COLORS = {
    "constrained_pca": "#1f77b4",
    "vanilla_das": "#ff7f0e",
    "cpca_init_fac": "#2ca02c",
}

METHODS = [
    ("constrained_pca", "CPCA"),
    ("vanilla_das", "Vanilla DAS"),
    ("cpca_init_fac", "CPCA-init DAS"),
]


def load_data():
    data = {}

    for task in ["sva", "gender_bias", "capital_country"]:
        path = REPO / f"artifacts/cpca_das_v3/{task}.json"
        d = json.load(open(path))
        k1 = d["per_k"]["1"]
        data[task] = {
            "n_hard": d["n_hard"],
            "constrained_pca": k1["constrained_pca"]["iia_mean"],
            "delta_pca": k1["delta_pca"]["iia_mean"],
            "vanilla_das": k1["vanilla_das_200"]["iia_mean"],
            "cpca_unconstrained": k1["cpca_init_unconstrained_100"]["iia_mean"],
            "delta_init_fac": k1["factorized_das_delta_s200_l0.1"]["iia_mean"],
            "cpca_init_fac": k1["cpca_init_fac_s200_l0.05"]["iia_mean"],
        }

    ioi_hard = json.load(open(REPO / "artifacts/cpca_hard_ioi/cpca_hard_ioi/cpca_init_multitask_hard.json"))
    ioi_v4 = json.load(open(REPO / "artifacts/factorized_das_hard_v4/ioi.json"))
    ioi_pt = ioi_hard["per_task"]["ioi"]
    data["ioi"] = {
        "n_hard": ioi_pt["n_hard"],
        "constrained_pca": ioi_pt["per_k"]["1"]["constrained_pca"]["iia"],
        "delta_pca": None,
        "vanilla_das": ioi_v4["per_k"]["1"]["vanilla_das"]["iia"],
        "cpca_unconstrained": None,
        "delta_init_fac": None,
        "cpca_init_fac": ioi_pt["per_k"]["1"]["cpca_init_das"]["iia"],
    }

    return data


def main():
    data = load_data()
    tasks = ["ioi", "sva", "gender_bias", "capital_country"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), sharey=True)

    for ax_i, task in enumerate(tasks):
        ax = axes[ax_i]
        td = data[task]

        present = [(key, label) for key, label in METHODS if td.get(key) is not None]

        x = np.arange(len(present))
        colors = [METHOD_COLORS[key] for key, _ in present]
        vals = [td[key] for key, _ in present]

        ax.bar(x, vals, width=0.65, color=colors,
               edgecolor="black", linewidth=0.5)

        for i, val in enumerate(vals):
            ax.text(i, val + 0.015, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in present], fontsize=8)
        ax.set_title(f"{TASK_LABELS[task]} (n={td['n_hard']} hard)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("IIA (hard examples, k=1)", fontsize=10)
    axes[0].set_ylim(0, 1.12)

    legend_handles = [
        mpatches.Patch(facecolor=METHOD_COLORS["constrained_pca"], edgecolor="black",
                       label="CPCA (untrained, on manifold)"),
        mpatches.Patch(facecolor=METHOD_COLORS["vanilla_das"], edgecolor="black",
                       label="Vanilla DAS (trained, random init)"),
        mpatches.Patch(facecolor=METHOD_COLORS["cpca_init_fac"], edgecolor="black",
                       label="CPCA-init DAS (trained, manifold init)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Hard-example IIA across tasks (k=1)",
                 fontsize=12, y=1.07, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_method_comparison_v2.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig1_method_comparison_v2.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved fig1_method_comparison_v2")


if __name__ == "__main__":
    main()
