"""Plot MIB faithfulness sweeps written by ``run_evaluation.py``.

Each input is a pickle produced by ``MIB-circuit-track/run_evaluation.py``
(``evaluate_area_under_curve``), holding a single faithfulness sweep::

    {"sizes": [...], "area_under": float, "area_from_1": float,
     "average": float, "faithfulnesses": [f_0, ..., f_9]}

The ten ``faithfulnesses`` line up with the fixed circuit-fraction grid
``PERCENTAGES``.  ``area_under`` is the integrated faithfulness (CPR-style,
higher is better); ``area_from_1`` is the integrated gap to 1 (CMD, lower is
better).  These are the honest scalar summaries — there is no separate
"CPR curve" vs "CMD curve"; both are read off the *one* faithfulness curve.

Two figures:

* **area chart** (one pickle): log-x faithfulness curve with the region under
  the curve filled as CPR and the gap up to 1 filled as CMD — couples both in
  one plot.
* **overlay** (several pickles): the faithfulness curves on shared axes,
  labeled by level / method (or via ``--labels``), CMD/CPR annotated per curve.

Usage::

    python -m lib.checkpoint_eval.plot \
        results/EAP_patching_edge/ioi_gpt2_validation_abs-False_caf-False.pkl

    python -m lib.checkpoint_eval.plot a.pkl b.pkl \
        --labels "EAP edge,EAP-IG factor" -o compare.png
"""
from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Circuit fractions swept by evaluate_area_under_curve (MIB_circuit_track).
PERCENTAGES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

_PALETTE = [
    "#E8913A",
    "#5B9BD5",
    "#70AD47",
    "#C0504D",
    "#7030A0",
    "#00B0F0",
    "#FF6699",
    "#996633",
    "#339966",
    "#FFC000",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "p"]

# Fill colors for the area chart (CPR under the curve, CMD as the gap to 1).
_CPR_FILL = "#f2b282"
_CMD_FILL = "#9fcbed"
_CMD_HATCH = _CMD_FILL
_LINE = "#444444"

plt.rcParams["hatch.linewidth"] = 2.0

# Trailing "_{split}_abs-{bool}_caf-{bool}" tail of an eval filename.
_TAIL = re.compile(r"_(train|validation|test)_abs-(True|False)_caf-(True|False)$")


@dataclass
class Curve:
    """One faithfulness sweep loaded from an eval pickle."""

    path: Path
    faithfulness: np.ndarray  # (10,), aligned to PERCENTAGES
    area_under: float  # CPR-style integrated faithfulness
    area_from_1: float  # CMD: integrated |1 - f|
    average: float
    label: str


def auto_label(path: Path) -> str:
    """Readable label from the eval filename + its method/ablation/level dir."""
    core = _TAIL.sub("", path.stem)  # drops split/abs/caf tail -> "{task}_{model}"
    return f"{core} · {path.parent.name}"


def load_curve(path: str | Path, label: str | None = None) -> Curve:
    path = Path(path)
    with open(path, "rb") as f:
        d = pickle.load(f)
    if "faithfulnesses" not in d:
        raise ValueError(
            f"{path}: not a faithfulness pickle (keys={sorted(d)}). "
            "Expected the output of run_evaluation.py's evaluate_area_under_curve."
        )
    faith = np.asarray(d["faithfulnesses"], dtype=float)
    if faith.shape != (len(PERCENTAGES),):
        raise ValueError(
            f"{path}: expected {len(PERCENTAGES)} faithfulness points, got {faith.shape}."
        )
    return Curve(
        path=path,
        faithfulness=faith,
        area_under=float(d["area_under"]),
        area_from_1=float(d["area_from_1"]),
        average=float(d["average"]),
        label=label or auto_label(path),
    )


def _insert_baseline_crossings(
    x: np.ndarray, f: np.ndarray, log_x: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Add synthetic points where f crosses 1 (interpolated in x, matching the
    axis scale) so the CPR / CMD fills tile without gaps."""
    xs, fs = list(x), list(f)
    for i in range(len(x) - 1):
        if (f[i] < 1.0) != (f[i + 1] < 1.0):
            t = (1.0 - f[i]) / (f[i + 1] - f[i])
            if log_x:
                lx0, lx1 = np.log10(x[i]), np.log10(x[i + 1])
                xs.append(float(10 ** (lx0 + t * (lx1 - lx0))))
            else:
                xs.append(float(x[i] + t * (x[i + 1] - x[i])))
            fs.append(1.0)
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(fs)[order]


def _style_x_axis(ax: plt.Axes, log_x: bool = True) -> None:
    ax.set_xlim(PERCENTAGES[0], PERCENTAGES[-1])
    ax.set_xlabel("Proportion of circuit kept (k)", fontsize=14)
    ax.set_ylabel("Normalized faithfulness", fontsize=14)
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks([0.001, 0.01, 0.1, 1.0])
        ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.9, alpha=0.85)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def area_chart(curve: Curve, title: str | None = None, log_x: bool = True) -> plt.Figure:
    """One faithfulness curve with CPR (under-curve) and CMD (gap-to-1) fills."""
    x = np.asarray(PERCENTAGES)
    f = curve.faithfulness
    xd, fd = _insert_baseline_crossings(x, f, log_x=log_x)
    capped = np.minimum(fd, 1.0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    _style_x_axis(ax, log_x=log_x)

    ax.fill_between(
        xd,
        0.0,
        capped,
        color=_CPR_FILL,
        alpha=0.7,
        linewidth=0,
        zorder=1,
        label=f"CPR (area under = {curve.area_under:.3f})",
    )
    ax.fill_between(
        xd,
        capped,
        1.0,
        color=_CMD_FILL,
        alpha=0.7,
        linewidth=0,
        zorder=2,
        label=f"CMD (area from 1 = {curve.area_from_1:.3f})",
    )
    # Overshoot (f > 1): counted in BOTH integrals — it is the 1->f slice of
    # CPR (area under) and the f-1 = |1-f| of CMD (area from 1).  Show it as
    # orange filled with a blue hatch to reflect the double membership.
    over = np.maximum(fd, 1.0)
    ax.fill_between(xd, 1.0, over, facecolor=_CPR_FILL, alpha=0.7, linewidth=0, zorder=2)
    ax.fill_between(
        xd, 1.0, over, facecolor="none", edgecolor=_CMD_HATCH, hatch="////", linewidth=0.0, zorder=3
    )

    ax.axhline(1.0, color=_LINE, linewidth=1.1, linestyle=":", alpha=0.75, zorder=4)
    ax.plot(xd, fd, color=_LINE, linewidth=2.0, zorder=5)
    ax.plot(
        x,
        f,
        color=_LINE,
        linewidth=0,
        marker="o",
        markersize=5.5,
        markerfacecolor=_LINE,
        markeredgecolor="white",
        markeredgewidth=1.1,
        zorder=6,
    )

    ax.set_ylim(min(0.0, f.min()) - 0.05, max(1.4, f.max() + 0.1))
    ax.set_title(title or curve.label, fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=11)
    fig.tight_layout()
    return fig


def overlay(curves: list[Curve], title: str | None = None, log_x: bool = True) -> plt.Figure:
    """Overlay several faithfulness curves on shared axes."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _style_x_axis(ax, log_x=log_x)

    ymax = 1.4
    for i, c in enumerate(curves):
        color = _PALETTE[i % len(_PALETTE)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(
            PERCENTAGES,
            c.faithfulness,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=2.0,
            alpha=0.9,
            label=f"{c.label}  (CMD={c.area_from_1:.3f}, CPR={c.area_under:.3f})",
        )
        ymax = max(ymax, c.faithfulness.max() + 0.1)

    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_ylim(-0.05, ymax)
    ax.set_title(title or "Faithfulness comparison", fontsize=15, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("pickles", nargs="+", help="Eval pickle(s) from run_evaluation.py")
    p.add_argument(
        "--labels",
        default=None,
        help="Comma-separated labels, one per pickle (overrides auto labels)",
    )
    p.add_argument("-o", "--out", default=None, help="Output PNG path")
    p.add_argument("--title", default=None, help="Figure title")
    p.add_argument(
        "--linear-x",
        action="store_true",
        help="Use a linear x-axis instead of the default log scale",
    )
    args = p.parse_args()

    labels = args.labels.split(",") if args.labels else [None] * len(args.pickles)
    if len(labels) != len(args.pickles):
        p.error(f"got {len(args.pickles)} pickles but {len(labels)} labels")

    curves = [load_curve(path, lbl) for path, lbl in zip(args.pickles, labels)]

    if len(curves) == 1:
        fig = area_chart(curves[0], title=args.title, log_x=not args.linear_x)
        out = args.out or str(curves[0].path.with_suffix(".png"))
    else:
        fig = overlay(curves, title=args.title, log_x=not args.linear_x)
        out = args.out or "faithfulness_overlay.png"

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
