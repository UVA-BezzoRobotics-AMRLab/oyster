"""
plot_gap_worlds.py

Finds and plots the N worlds where file1 performs best *relative to* file2
(i.e. maximises the success-rate gap:  rate1 - rate2).

Usage:
    python plot_gap_worlds.py results1.txt results2.txt
    python plot_gap_worlds.py results1.txt results2.txt --labels "CBF-QP" "CBF-CLF"
    python plot_gap_worlds.py results1.txt results2.txt --top_n 20
    python plot_gap_worlds.py results1.txt results2.txt --top_n 15 --task_num 6
    python plot_gap_worlds.py results1.txt results2.txt --top_n 15 --output gap.png
    python plot_gap_worlds.py results1.txt results2.txt --top_n 15 --min_rate 80

Data format (tab or space separated):
    WORLD  TASK  SUCCESS  STEPS  CLEARANCE
    0      6     1        238    0.42
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from pathlib import Path


# ── Parsing ───────────────────────────────────────────────────────────────────

def load_results(filepath, task_num=None):
    world_successes = defaultdict(list)
    world_clearance = defaultdict(list)
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.upper().startswith("WORLD"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                if task_num is not None and int(parts[1]) != task_num:
                    continue
                world     = int(parts[0])
                success   = int(parts[2])
                steps     = int(parts[3])
                clearance = float(parts[4])
                if clearance < 0.075 or steps >= 250:
                    success = 0
                world_successes[world].append(success)
                if len(parts) >= 5:
                    world_clearance[world].append(float(parts[4]))
            except ValueError:
                continue
    return world_successes, world_clearance


# ── Core gap logic ────────────────────────────────────────────────────────────

def compute_gap(ws1, ws2, max_world=None, min_rate1=0.0):
    """
    For every world present in BOTH files compute:
        gap = success_rate_1 - success_rate_2   (percentage points)

    Worlds where rate1 < min_rate1 are excluded entirely.
    Returns a list of (world, rate1, rate2, gap) sorted by gap descending.
    """
    common_worlds = sorted(set(ws1.keys()) & set(ws2.keys()))
    if max_world is not None:
        common_worlds = [w for w in common_worlds if w < max_world]

    rows    = []
    skipped = 0
    for w in common_worlds:
        r1 = np.mean(ws1[w]) * 100
        r2 = np.mean(ws2[w]) * 100
        if r1 < min_rate1:
            skipped += 1
            continue
        rows.append((w, r1, r2, r1 - r2))

    if skipped:
        print(f"Skipped {skipped} worlds where file1 rate < {min_rate1:.1f}%")

    rows.sort(key=lambda x: x[3], reverse=True)
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def _set_xticks(ax, worlds):
    worlds = np.asarray(worlds)
    if len(worlds) > 0:
        step = max(1, len(worlds) // 30)
        ax.set_xticks(worlds[::step])
        ax.tick_params(axis="x", labelsize=8, rotation=45)


def plot_gap(file1, file2, label1="File 1", label2="File 2",
             top_n=10, max_world=None, task_num=None, output=None,
             dump=False, min_rate1=0.0):

    ws1, wc1 = load_results(file1, task_num)
    ws2, wc2 = load_results(file2, task_num)

    gap_rows = compute_gap(ws1, ws2, max_world, min_rate1=min_rate1)

    if not gap_rows:
        print("No worlds found in common between the two files "
              "(check --min_rate threshold).")
        return

    # ── Top-N worlds by gap ───────────────────────────────────────────────────
    top_rows   = gap_rows[:top_n]
    top_worlds = [r[0] for r in top_rows]
    top_r1     = np.array([r[1] for r in top_rows])
    top_r2     = np.array([r[2] for r in top_rows])
    top_gaps   = np.array([r[3] for r in top_rows])

    c1 = plt.cm.tab10.colors[0]   # blue   – file 1
    c2 = plt.cm.tab10.colors[1]   # orange – file 2

    x     = np.arange(len(top_worlds))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    title = f"Top-{top_n} worlds where '{label1}' most outperforms '{label2}'"
    if min_rate1 > 0:
        title += f"  (file1 rate ≥ {min_rate1:.0f}%)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # ── Panel 1: side-by-side success rates ───────────────────────────────────
    ax = axes[0]
    ax.bar(x - width/2, top_r1, width, color=c1, alpha=0.85,
           edgecolor="white", linewidth=0.4, label=label1)
    ax.bar(x + width/2, top_r2, width, color=c2, alpha=0.85,
           edgecolor="white", linewidth=0.4, label=label2)
    ax.set_title("Success Rate on Top-N Worlds", fontsize=11,
                 fontweight="semibold", loc="left")
    ax.set_ylabel("Success Rate (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_xlabel("World #", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(top_worlds, rotation=45, fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: gap bar chart ────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(x, top_gaps, width=0.6, alpha=0.85,
                  edgecolor="white", linewidth=0.4)
    for bar, gap in zip(bars, top_gaps):
        bar.set_color(c1 if gap >= 0 else c2)
        bar.set_alpha(0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    mean_gap = float(np.mean(top_gaps))
    ax.axhline(mean_gap, color="gray", linewidth=1.2, linestyle="-.",
               label=f"Mean gap = {mean_gap:.1f} pp")
    ax.set_title(f"Gap ({label1} − {label2})", fontsize=11,
                 fontweight="semibold", loc="left")
    ax.set_ylabel("Gap (percentage points)", fontsize=10)
    ax.set_xlabel("World #", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(top_worlds, rotation=45, fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 3: full gap distribution (all common worlds after filter) ────────
    ax = axes[2]
    all_worlds = np.array([r[0] for r in gap_rows])
    all_gaps   = np.array([r[3] for r in gap_rows])

    bar_colors = [c1 if g >= 0 else c2 for g in all_gaps]
    ax.bar(all_worlds, all_gaps, width=0.8, color=bar_colors, alpha=0.75,
           edgecolor="none")
    for tw in top_worlds:
        ax.axvline(tw, color="gold", linewidth=0.8, alpha=0.6)

    ax.axhline(0, color="black", linewidth=0.8)
    overall_mean = float(np.mean(all_gaps))
    ax.axhline(overall_mean, color="gray", linewidth=1.2, linestyle="-.",
               label=f"Overall mean = {overall_mean:.1f} pp")
    ax.set_title(
        f"Gap Across Eligible Worlds  (top-{top_n} highlighted)"
        + (f"\n[file1 rate ≥ {min_rate1:.0f}%]" if min_rate1 > 0 else ""),
        fontsize=11, fontweight="semibold", loc="left"
    )
    ax.set_ylabel("Gap (percentage points)", fontsize=10)
    ax.set_xlabel("World #", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    _set_xticks(ax, all_worlds)

    from matplotlib.patches import Patch
    axes[2].legend(handles=[
        Patch(color=c1, alpha=0.85, label=f"{label1} leads"),
        Patch(color=c2, alpha=0.85, label=f"{label2} leads"),
        plt.Line2D([0], [0], color="gray", linestyle="-.", linewidth=1.2,
                   label=f"Overall mean = {overall_mean:.1f} pp"),
        plt.Line2D([0], [0], color="gold", linewidth=0.8,
                   label=f"Top-{top_n} worlds"),
    ], fontsize=8, loc="lower right")

    plt.tight_layout()

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\nTop-{top_n} worlds where '{label1}' outperforms '{label2}'"
          + (f"  (file1 rate ≥ {min_rate1:.0f}%)" if min_rate1 > 0 else "") + ":")
    print(f"  {'World':>6}  {'Rate1':>8}  {'Rate2':>8}  {'Gap':>8}")
    print("  " + "-" * 38)
    for w, r1, r2, g in top_rows:
        print(f"  {w:>6}  {r1:>7.1f}%  {r2:>7.1f}%  {g:>+7.1f} pp")

    if dump:
        for w, *_ in top_rows:
            print(w)

    if output:
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to: {output}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find worlds where file1 most outperforms file2 on success rate."
    )
    parser.add_argument("file1", help="First result file (the 'better' one)")
    parser.add_argument("file2", help="Second result file (the baseline)")
    parser.add_argument("--labels", nargs=2, default=None,
                        metavar=("LABEL1", "LABEL2"),
                        help="Display names for the two files")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of worlds to highlight (default: 10)")
    parser.add_argument("--max_world", type=int, default=None,
                        help="Limit to worlds 0..max_world-1")
    parser.add_argument("--task_num", type=int, default=None,
                        help="Filter to a specific task number")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to this path (e.g. gap.png)")
    parser.add_argument("--dump", action="store_true",
                        help="Dump top-N world indices to stdout (one per line)")
    parser.add_argument("--min_rate", type=float, default=0.0,
                        metavar="PCT",
                        help="Exclude worlds where file1 success rate is below "
                             "this threshold in percent (e.g. 80 means ≥80%%). "
                             "Default: 0 (no filter).")
    args = parser.parse_args()

    label1, label2 = args.labels if args.labels else \
        (Path(args.file1).stem, Path(args.file2).stem)

    plot_gap(
        args.file1, args.file2,
        label1=label1, label2=label2,
        top_n=args.top_n,
        max_world=args.max_world,
        task_num=args.task_num,
        output=args.output,
        dump=args.dump,
        min_rate1=args.min_rate,
    )
