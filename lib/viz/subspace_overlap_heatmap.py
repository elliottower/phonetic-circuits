"""5 different visualizations of factor-subtask membership, same data."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent
REPO = Path("/Users/elliottower/Documents/GitHub/factorization-circuits")

SUBTASKS = ["s2_io_flip", "s1_io_flip", "abc", "random_names", "full_flip"]
SHORT = {"s2_io_flip": "IO name", "s1_io_flip": "Subject",
         "abc": "Both (ABC)", "random_names": "Random", "full_flip": "Full flip"}
COLORS = {"s2_io_flip": "#e41a1c", "s1_io_flip": "#377eb8", "abc": "#4daf4a",
          "random_names": "#984ea3", "full_flip": "#ff7f00"}

das = json.load(open(REPO / "artifacts/per_variable_das/per_variable_das_exploration.json"))
layer5 = das["per_layer"]["5"]
TOP_K = 20

factor_sets = {}
all_scores = {}
for st in SUBTASKS:
    factor_sets[st] = set()
    for t in layer5[st]["top_factors"][:TOP_K]:
        factor_sets[st].add(t["factor"])
        all_scores.setdefault(t["factor"], {})[st] = t["score"]

all_factors = sorted(all_scores.keys())

def primary_task(f):
    scores = all_scores[f]
    return max(scores, key=scores.get)

def sort_key(f):
    scores = all_scores[f]
    best = max(scores, key=scores.get)
    return (SUBTASKS.index(best), -scores[best])

all_factors.sort(key=sort_key)


# =========================================================================
# VIZ A: Binary dot matrix — colored circles on a grid
# =========================================================================
def viz_a_dot_matrix():
    fig, ax = plt.subplots(figsize=(16, 4))

    for j, f in enumerate(all_factors):
        for i, st in enumerate(SUBTASKS):
            if st in all_scores[f]:
                ax.scatter(j, i, c=COLORS[st], s=50, edgecolors="none", zorder=3)
            else:
                ax.scatter(j, i, c="#f0f0f0", s=15, edgecolors="none", zorder=2)

    # Group dividers
    current = None
    for j, f in enumerate(all_factors):
        pt = primary_task(f)
        if pt != current:
            if current is not None:
                ax.axvline(j - 0.5, color="#aaaaaa", linewidth=0.8, linestyle="-")
            current = pt

    ax.set_yticks(range(len(SUBTASKS)))
    ax.set_yticklabels([SHORT[s] for s in SUBTASKS], fontsize=10)
    ax.set_xlim(-1, len(all_factors))
    ax.set_ylim(-0.8, len(SUBTASKS) - 0.2)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlabel(f"{len(all_factors)} factors (grouped by primary task)", fontsize=10)
    ax.set_title("Factor Usage by IOI Subtask (DAS top-20, Layer 5)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "11a_dot_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 11a_dot_matrix.png")


# =========================================================================
# VIZ B: Waffle chart — each factor is a colored tile
# =========================================================================
def viz_b_waffle():
    n = len(all_factors)
    cols = 10
    rows = int(np.ceil(n / cols))

    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, f in enumerate(all_factors):
        row = idx // cols
        col = idx % cols
        pt = primary_task(f)
        n_users = len(all_scores[f])

        rect = plt.Rectangle((col, rows - 1 - row), 0.9, 0.9,
                              facecolor=COLORS[pt],
                              edgecolor="white" if n_users == 1 else "black",
                              linewidth=1.5 if n_users > 1 else 0.5,
                              zorder=2)
        ax.add_patch(rect)

        if n_users > 1:
            ax.text(col + 0.45, rows - 1 - row + 0.45, str(n_users),
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white", zorder=3)

    ax.set_xlim(-0.2, cols + 0.2)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    legend = [mpatches.Patch(facecolor=COLORS[s], edgecolor="black", linewidth=0.5,
                             label=f"{SHORT[s]} ({sum(1 for f in all_factors if primary_task(f) == s)})")
              for s in SUBTASKS]
    legend.append(mpatches.Patch(facecolor="gray", edgecolor="black", linewidth=2,
                                 label="Shared (black border)"))
    ax.legend(handles=legend, loc="lower center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.08))

    ax.set_title(f"82 Factors Colored by Primary Task\n(number = how many tasks share it)",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "11b_waffle.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 11b_waffle.png")


# =========================================================================
# VIZ C: Horizontal stacked bars — private vs shared breakdown per task
# =========================================================================
def viz_c_stacked_bars():
    fig, ax = plt.subplots(figsize=(10, 4))

    y_pos = np.arange(len(SUBTASKS))
    bar_data = {}

    for st in SUBTASKS:
        private = sum(1 for f in factor_sets[st] if len(all_scores[f]) == 1)
        shared_with = {}
        for f in factor_sets[st]:
            if len(all_scores[f]) > 1:
                for other_st in all_scores[f]:
                    if other_st != st:
                        shared_with[other_st] = shared_with.get(other_st, 0) + 1
        bar_data[st] = {"private": private, "shared": shared_with}

    # Draw private bars
    privates = [bar_data[st]["private"] for st in SUBTASKS]
    bars = ax.barh(y_pos, privates, color=[COLORS[s] for s in SUBTASKS],
                   edgecolor="black", linewidth=0.5, label="Private", height=0.6)

    # Draw shared segments stacked after private
    for i, st in enumerate(SUBTASKS):
        left = privates[i]
        for other in SUBTASKS:
            if other == st:
                continue
            count = bar_data[st]["shared"].get(other, 0)
            if count > 0:
                ax.barh(i, count, left=left, color=COLORS[other], alpha=0.5,
                        edgecolor="black", linewidth=0.5, height=0.6,
                        hatch="//")
                if count >= 2:
                    ax.text(left + count / 2, i, str(count), ha="center", va="center",
                            fontsize=8, fontweight="bold")
                left += count

    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT[s] for s in SUBTASKS], fontsize=10)
    ax.set_xlabel("Number of factors in top-20", fontsize=10)
    ax.set_title("Private vs Shared Factors per Subtask", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(20, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(20.3, -0.6, "top-20 limit", fontsize=7, color="gray")

    legend = [mpatches.Patch(facecolor="gray", edgecolor="black", label="Private"),
              mpatches.Patch(facecolor="gray", edgecolor="black", hatch="//",
                             alpha=0.5, label="Shared (color = partner)")]
    ax.legend(handles=legend, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT / "11c_stacked_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 11c_stacked_bars.png")


# =========================================================================
# VIZ D: Network graph — factors as nodes, task membership as edges
# =========================================================================
def viz_d_network():
    fig, ax = plt.subplots(figsize=(8, 8))

    # Place task nodes in a circle
    n_tasks = len(SUBTASKS)
    task_angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False) - np.pi / 2
    task_x = 3.5 * np.cos(task_angles)
    task_y = 3.5 * np.sin(task_angles)

    # Place factor nodes around their primary task
    factor_positions = {}
    for st_idx, st in enumerate(SUBTASKS):
        my_factors = [f for f in all_factors if primary_task(f) == st]
        n_f = len(my_factors)
        spread = 1.2
        for fi, f in enumerate(my_factors):
            angle = task_angles[st_idx] + (fi - n_f / 2) * 0.12
            r = 1.5 + (fi % 3) * 0.4
            fx = task_x[st_idx] + r * np.cos(angle) * 0.5
            fy = task_y[st_idx] + r * np.sin(angle) * 0.5
            factor_positions[f] = (fx, fy)

    # Draw edges for shared factors
    for f in all_factors:
        users = [st for st in SUBTASKS if st in all_scores[f]]
        if len(users) > 1:
            fx, fy = factor_positions[f]
            for st in users:
                if st != primary_task(f):
                    st_idx = SUBTASKS.index(st)
                    ax.plot([fx, task_x[st_idx]], [fy, task_y[st_idx]],
                            color=COLORS[st], alpha=0.2, linewidth=0.8, zorder=1)

    # Draw factor nodes
    for f in all_factors:
        pt = primary_task(f)
        fx, fy = factor_positions[f]
        n_users = len(all_scores[f])
        size = 30 if n_users == 1 else 60
        edge = "none" if n_users == 1 else "black"
        ax.scatter(fx, fy, c=COLORS[pt], s=size, edgecolors=edge,
                   linewidths=1, zorder=3, alpha=0.8)

    # Draw task nodes
    for i, st in enumerate(SUBTASKS):
        count = sum(1 for f in all_factors if primary_task(f) == st)
        ax.scatter(task_x[i], task_y[i], c=COLORS[st], s=500,
                   edgecolors="black", linewidths=2, zorder=5, marker="s")
        ax.text(task_x[i], task_y[i] - 0.4, f"{SHORT[st]}\n({count})",
                ha="center", va="top", fontsize=8, fontweight="bold",
                color=COLORS[st], zorder=6)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Factor-Subtask Network\n(edges = shared factors)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "11d_network.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 11d_network.png")


# =========================================================================
# VIZ E: Pie/donut per task showing private vs shared composition
# =========================================================================
def viz_e_donuts():
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))

    for idx, st in enumerate(SUBTASKS):
        ax = axes[idx]
        private = sum(1 for f in factor_sets[st] if len(all_scores[f]) == 1)
        shared = TOP_K - private

        wedges, texts = ax.pie(
            [private, shared],
            colors=[COLORS[st], "#dddddd"],
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
        )

        ax.text(0, 0, f"{private}/{TOP_K}\nprivate",
                ha="center", va="center", fontsize=10, fontweight="bold")
        ax.set_title(SHORT[st], fontsize=11, fontweight="bold", color=COLORS[st])

    fig.suptitle("Factor Privacy per IOI Subtask (DAS top-20, Layer 5)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "11e_donuts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 11e_donuts.png")


if __name__ == "__main__":
    viz_a_dot_matrix()
    viz_b_waffle()
    viz_c_stacked_bars()
    viz_d_network()
    viz_e_donuts()
    print(f"\nAll 5 variants saved to {OUT}/")
