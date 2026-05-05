"""Generate the LW-post charts from the canonical judged sweep.

Reads:
  results/lindsey_full_*.judged.json  (auto-picks latest)

Writes:
  figures/structural_finding.png  — id × coh trade-off, lindsey/all_caps
  figures/all_conditions.png      — 4-panel grid across (vector × scaffold)

Both PNGs are 144 dpi, wide format, sans-serif. The structural-finding chart
is the lead figure for the post; the all-conditions chart is the robustness
follow-up.

Usage:
    python make_figures.py
    python make_figures.py path/to/lindsey_full_*.judged.json
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_aggregates(path):
    d = json.load(open(path))
    by_label = defaultdict(list)  # label -> sorted list of (mag, rates_dict)
    for a in d["aggregates"]:
        by_label[a["label"]].append((a["magnitude"], a["rates"]))
    for label in by_label:
        by_label[label].sort(key=lambda x: x[0])
    return by_label, d


def line_kwargs(metric):
    """Consistent style across panels."""
    if metric == "identifies":
        return dict(color="#c0392b", marker="o", linewidth=2.4, label="identifies")
    if metric == "coherent":
        return dict(color="#2980b9", marker="s", linewidth=2.4, label="coherent")
    if metric == "immediate":
        return dict(color="#7f8c8d", marker="^", linewidth=1.6,
                    linestyle="--", label="immediate")
    if metric == "affirmative":
        return dict(color="#27ae60", marker="v", linewidth=1.6,
                    linestyle=":", label="affirmative")
    raise ValueError(metric)


def plot_panel(ax, cells, *, title, metrics, show_xlabel=True, show_ylabel=True,
               show_legend=False, annotate_crossover=False):
    if not cells:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#888", fontsize=11)
        ax.set_axis_off()
        return
    mags = [c[0] for c in cells]
    for m in metrics:
        ys = [c[1].get(m, 0) for c in cells]
        ax.plot(mags, ys, **line_kwargs(m))
    ax.set_xlim(min(mags) - 0.5, max(mags) + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(mags)
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    if show_xlabel:
        ax.set_xlabel("steering magnitude (norm-matched)")
    if show_ylabel:
        ax.set_ylabel("rate")
    ax.set_title(title, fontsize=11, loc="left")
    if show_legend:
        ax.legend(loc="center left", frameon=False, fontsize=10)

    if annotate_crossover and len(metrics) == 2:
        # Mark the empty top-right region where both ≥ 0.5 — visual proof of "no overlap"
        ax.axhspan(0.5, 1.05, alpha=0.0)  # no-op, kept for clarity
        ax.fill_between([min(mags) - 0.5, max(mags) + 0.5], 0.5, 1.05,
                        color="#888", alpha=0.0)
        ax.axhline(0.5, color="#aaa", linewidth=0.6, linestyle=":")


def figure_structural_finding(by_label, out_path):
    """Single-panel: lindsey/all_caps, identifies vs coherent — the post's lead chart."""
    cells = by_label["lindsey_all_caps"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=144)
    plot_panel(
        ax, cells,
        title="Identification and coherence trade off — lindsey scaffold, all_caps vector",
        metrics=["identifies", "coherent"],
        show_legend=True,
        annotate_crossover=True,
    )
    ax.text(
        0.5, 0.50,
        "no magnitude where both ≥ 0.5",
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=10, color="#555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#ccc", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def figure_all_conditions(by_label, out_path):
    """2x2 grid: (lindsey | alt) × (all_caps | love), id and coh per panel."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), dpi=144, sharey=True)
    panels = [
        ("lindsey_all_caps", axes[0][0],
         "Lindsey scaffold · all_caps", True,  True,  True),
        ("lindsey_love",     axes[0][1],
         "Lindsey scaffold · love",     True,  False, False),
        ("alt_all_caps",     axes[1][0],
         "Alt prompt · all_caps",       True,  True,  False),
        ("alt_love",         axes[1][1],
         "Alt prompt · love",           True,  False, False),
    ]
    for label, ax, title, _, show_y, show_legend in panels:
        plot_panel(
            ax, by_label.get(label, []),
            title=title,
            metrics=["identifies", "coherent"],
            show_xlabel=ax in axes[1],     # bottom row only
            show_ylabel=show_y,
            show_legend=show_legend,
        )
    fig.suptitle(
        "Identification × coherence trade-off across all conditions "
        "(Llama-3.1-405B-Instruct, layer 84)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        candidates = sorted(
            glob.glob("results/**/lindsey_full_*.judged.json", recursive=True)
            + glob.glob("results/lindsey_full_*.judged.json"),
            key=os.path.getmtime,
        )
        if not candidates:
            sys.exit("ERROR: no lindsey_full_*.judged.json found in results/")
        path = candidates[-1]
        print(f"[auto] using: {path}")

    by_label, _ = load_aggregates(path)
    Path("figures").mkdir(exist_ok=True)
    figure_structural_finding(by_label, "figures/structural_finding.png")
    figure_all_conditions(by_label, "figures/all_conditions.png")


if __name__ == "__main__":
    main()
