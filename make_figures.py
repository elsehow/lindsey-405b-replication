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
    python make_figures.py --layer-stack DIR1 DIR2 ...   # per-layer crossover stack

For --layer-stack, pass any number of result-dirs each containing one
lindsey_full_*.judged.json. The script extracts the layer number from each
file's "layer" field and stacks them into one figure (one panel per layer,
in depth order).
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


def merge_aggregates(paths):
    """Combine multiple judged JSONs into one by_label dict, merging by
    (label, magnitude). Later paths override earlier paths at the same key.
    Useful for stitching the dense layer-84 grid into the canonical run."""
    merged = defaultdict(dict)  # label -> mag -> rates
    for p in paths:
        d = json.load(open(p))
        for a in d["aggregates"]:
            merged[a["label"]][a["magnitude"]] = a["rates"]
    by_label = defaultdict(list)
    for label, mag_map in merged.items():
        by_label[label] = sorted(mag_map.items(), key=lambda x: x[0])
    return by_label


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
    """Two-panel stacked: lindsey/all_caps (top) + lindsey/love (bottom),
    identifies vs coherent — the post's lead chart.

    Same x-axis (magnitude) on both panels so the curves are visually matched.
    Panels share the y-axis too. Annotate the empty top-right region to flag
    the absence of any cell where both ≥ 0.5."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), dpi=144,
                             sharex=True, sharey=True)
    panels = [
        ("lindsey_all_caps", axes[0], "all_caps vector"),
        ("lindsey_love",     axes[1], "love vector"),
    ]
    for label, ax, title in panels:
        cells = by_label.get(label, [])
        plot_panel(
            ax, cells,
            title=title,
            metrics=["identifies", "coherent"],
            show_xlabel=(ax is axes[1]),
            show_ylabel=True,
            show_legend=(ax is axes[0]),
            annotate_crossover=True,
        )
    fig.suptitle(
        "Identification × coherence trade-off (Llama-3.1-405B-Instruct, layer 84, lindsey scaffold)",
        fontsize=12, y=0.995,
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


def figure_layer_stack(judged_paths, out_path, *, label="lindsey_all_caps"):
    """Stacked panels (rows × 1), one per layer, showing id × coh as crossing
    line plots (matches the lead structural_finding.png style).

    Layer order: shallowest at top, deepest at bottom — standard transformer
    convention (layer 0 = input embedding, layer N = final output). Reading
    top-to-bottom traces the signal flowing through the network.

    If multiple judged JSONs share the same `layer`, their aggregates are
    merged by (label, magnitude) — useful for combining the May-3 canonical
    grid [5, 10, 12, 15, 18] with the May-6 dense grid [10, 10.5, 11, 11.5, 12]
    at layer 84."""
    by_layer = defaultdict(lambda: defaultdict(dict))  # layer -> label -> mag -> rates
    for p in judged_paths:
        d = json.load(open(p))
        layer = d.get("layer")
        if layer is None:
            sys.exit(f"ERROR: no 'layer' field in {p}")
        for a in d["aggregates"]:
            # Later files in the iteration override earlier ones at the same (layer,label,mag).
            by_layer[int(layer)][a["label"]][a["magnitude"]] = a["rates"]

    panels = []
    for layer, lab_map in by_layer.items():
        cells = sorted(lab_map.get(label, {}).items(), key=lambda x: x[0])
        panels.append((layer, cells, None))
    # Shallowest at top, deepest at bottom (standard transformer convention).
    panels.sort(key=lambda x: x[0])

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 2.6 * n + 0.6), dpi=144,
                             sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for i, (layer, cells, _) in enumerate(panels):
        ax = axes[i]
        plot_panel(
            ax, cells,
            title=f"Layer {layer}",
            metrics=["identifies", "coherent"],
            show_xlabel=(i == n - 1),  # only on bottom panel
            show_ylabel=True,
            show_legend=(i == 0),       # only on top panel
            annotate_crossover=True,
        )

    fig.suptitle(
        f"Identification × coherence by layer — {label}\n"
        "(Llama-3.1-405B-Instruct FP8, lindsey scaffold, dense magnitudes)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--layer-stack":
        judged = []
        for arg in sys.argv[2:]:
            ap = Path(arg)
            if ap.is_dir():
                hits = list(ap.glob("lindsey_full_*.judged.json"))
                if not hits:
                    sys.exit(f"no lindsey_full_*.judged.json in {ap}")
                judged.append(hits[0])
            elif ap.is_file():
                judged.append(ap)
            else:
                sys.exit(f"not found: {arg}")
        Path("figures").mkdir(exist_ok=True)
        figure_layer_stack(judged, "figures/layer_stack_all_caps.png", label="lindsey_all_caps")
        figure_layer_stack(judged, "figures/layer_stack_love.png", label="lindsey_love")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        # Multiple judged JSONs merged by (label, mag). Lead chart only.
        paths = sys.argv[2:]
        if not paths:
            sys.exit("ERROR: --merge needs ≥1 judged JSON path")
        by_label = merge_aggregates(paths)
        Path("figures").mkdir(exist_ok=True)
        figure_structural_finding(by_label, "figures/structural_finding.png")
        return

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
