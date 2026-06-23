"""Margin distributions — only narrow-margin region shaded, wide-margin as outline only."""
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

REPO = Path("/Users/elliottower/Documents/GitHub/factorization-circuits")
OUT = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "mathtext.fontset": "cm",
})

TASK_COLORS = {
    "ioi": "#2166ac",
    "sva": "#b2182b",
    "gender_bias": "#762a83",
    "capital_country": "#e08214",
}
TASK_LABELS = {
    "ioi": "IOI",
    "sva": "SVA",
    "gender_bias": "Gender Bias",
    "capital_country": "Capital–Country",
}
THRESHOLD = 1.0
TASKS = ["ioi", "sva", "gender_bias", "capital_country"]


def main():
    all_margins = {}
    for task in TASKS:
        p = REPO / f"artifacts/hard_example_cache/{task}/hard_examples.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = d["all_margins"]
        if hasattr(m, "numpy"):
            m = m.numpy()
        all_margins[task] = np.array(m, dtype=float)

    n = len(all_margins)
    fig, axes = plt.subplots(n, 1, figsize=(3.25, 1.2 * n), sharex=True)
    x_grid = np.linspace(-5, 15, 500)

    for i, task in enumerate(TASKS):
        if task not in all_margins:
            continue
        ax = axes[i]
        margins = all_margins[task]
        n_narrow = int(np.sum(margins < THRESHOLD))
        pct = 100 * n_narrow / len(margins)
        color = TASK_COLORS[task]

        kde = gaussian_kde(margins, bw_method=0.2)
        density = kde(x_grid)
        narrow_mask = x_grid <= THRESHOLD

        ax.fill_between(x_grid[narrow_mask], density[narrow_mask], alpha=0.55, color=color)
        ax.plot(x_grid, density, color=color, linewidth=0.9, alpha=0.7)
        ax.axvline(THRESHOLD, color="#444444", linestyle="--", linewidth=0.5, alpha=0.5)

        handle = mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.55)
        label = f"{TASK_LABELS[task]}  {n_narrow}/{len(margins)} ({pct:.0f}%)"
        ax.legend([handle], [label], loc="upper right", fontsize=6.5,
                  framealpha=0.95, edgecolor="#cccccc", fancybox=False,
                  borderpad=0.3, handlelength=1.0, handletextpad=0.4)

        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(-5, 15)

    axes[-1].set_xlabel(r"Model margin ($\mathrm{logit}_{\mathrm{correct}} - \mathrm{logit}_{\mathrm{incorrect}}$)")

    y_top = axes[0].get_ylim()[1]
    axes[0].annotate("narrow-margin", xy=(THRESHOLD - 0.3, y_top * 1.18),
                     ha="right", fontsize=6, color="#777777")
    axes[0].annotate("", xy=(THRESHOLD - 4.0, y_top * 1.12),
                     xytext=(THRESHOLD - 0.3, y_top * 1.12),
                     arrowprops=dict(arrowstyle="<-", color="#999999", lw=0.6))

    fig.tight_layout(h_pad=0.15, pad=0.5)
    fig.savefig(OUT / "02_margin_distributions_v7.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "02_margin_distributions_v7.pdf", bbox_inches="tight")
    plt.close()
    print("Saved 02_margin_distributions_v7")


if __name__ == "__main__":
    main()
