"""
plot_success_rate.py

Plots two subplots per scheme:
  Left:  Success rate per world (STEPS=250 counted as failure)
  Right: Avg min clearance per world with +/- std error bars (all trials included)

Usage:
    python plot_success_rate.py results.txt
    python plot_success_rate.py results.txt --label "CBF-QP" --max_world 50
    python plot_success_rate.py results1.txt results2.txt --labels "CBF-QP" "CBF-CLF"
    python plot_success_rate.py results.txt --output comparison.png

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
    """Parse a results file.

    Returns:
        world_successes : {world: [0/1 ints]}   -- STEPS=250 forced to 0
        world_clearance : {world: [float]}       -- all trials included
    """
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
                if task_num is not None:
                    if int(parts[1]) != task_num:
                        continue
                    
                world   = int(parts[0])
                success = int(parts[2])
                steps   = int(parts[3])
                clearance = float(parts[4])

                if clearance < 0.09:
                    success = 0
                # Timeout = implicit failure
                # if steps == 250:
                #     success = 0

                world_successes[world].append(success)

                # Clearance (5th column, index 4) -- all trials included
                if len(parts) >= 5:
                    world_clearance[world].append(float(parts[4]))
            except ValueError:
                continue
    return world_successes, world_clearance


# ── Statistics ────────────────────────────────────────────────────────────────

def _filter_worlds(worlds_sorted, max_world):
    if max_world is None:
        return worlds_sorted
    return [w for w in worlds_sorted if w < max_world]


def compute_success_rates(world_successes, max_world=None):
    worlds = _filter_worlds(sorted(world_successes), max_world)
    rates  = [np.mean(world_successes[w]) * 100 for w in worlds]
    return np.array(worlds), np.array(rates)


def compute_clearance_stats(world_clearance, max_world=None):
    worlds = _filter_worlds(sorted(world_clearance), max_world)
    means  = [np.mean(world_clearance[w]) for w in worlds]
    stds   = [np.std(world_clearance[w])  for w in worlds]
    return np.array(worlds), np.array(means), np.array(stds)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_xticks(ax, worlds):
    if len(worlds) > 0:
        step = max(1, len(worlds) // 30)
        ax.set_xticks(worlds[::step])
        ax.tick_params(axis="x", labelsize=8, rotation=45)


def _mean_line(ax, value, color, label):
    ax.axhline(value, color=color, linewidth=1.2, linestyle="-.",
               alpha=0.9, label=label)


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_success_rates(files, labels=None, max_world=None, output=None, task_num=None):
    if labels is None:
        labels = [Path(f).stem for f in files]

    n_schemes = len(files)
    fig, axes = plt.subplots(
        n_schemes, 2,
        figsize=(18, 3.8 * n_schemes),
        squeeze=False,
    )
    fig.suptitle("Per-World Results by Scheme", fontsize=15, fontweight="bold", y=1.01)

    colors = plt.cm.tab10.colors

    for idx, (fpath, label) in enumerate(zip(files, labels)):
        color = colors[idx % len(colors)]
        world_successes, world_clearance = load_results(fpath, task_num)

        # ── Left: success rate ─────────────────────────────────────────────
        ax_s = axes[idx][0]
        worlds_s, rates = compute_success_rates(world_successes, max_world)

        ax_s.bar(worlds_s, rates, width=0.8, color=color, alpha=0.85,
                 edgecolor="white", linewidth=0.4)
        ax_s.axhline(100, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
        mean_rate = float(np.mean(rates)) if len(rates) > 0 else 0.0
        _mean_line(ax_s, mean_rate, color, f"Mean = {mean_rate:.1f}%")

        ax_s.set_title(f"{label} -- Success Rate", fontsize=11,
                       fontweight="semibold", loc="left")
        ax_s.set_ylabel("Success Rate (%)", fontsize=10)
        ax_s.set_ylim(0, 110)
        ax_s.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax_s.set_xlabel("World #", fontsize=10)
        ax_s.legend(fontsize=9, loc="lower right")
        ax_s.grid(axis="y", linestyle="--", alpha=0.4)
        ax_s.spines[["top", "right"]].set_visible(False)
        _set_xticks(ax_s, worlds_s)

        # ── Right: clearance avg +/- std ──────────────────────────────────
        ax_c = axes[idx][1]
        worlds_c, means, stds = compute_clearance_stats(world_clearance, max_world)

        ax_c.bar(worlds_c, means, width=0.8, color=color, alpha=0.75,
                 edgecolor="white", linewidth=0.4)
        ax_c.errorbar(worlds_c, means, yerr=stds, fmt="none",
                      ecolor="black", elinewidth=0.8, capsize=2, alpha=0.55)
        mean_clr = float(np.mean(means)) if len(means) > 0 else 0.0
        _mean_line(ax_c, mean_clr, color, f"Mean = {mean_clr:.3f}")

        ax_c.set_title(f"{label} -- Min Clearance to Obstacle", fontsize=11,
                       fontweight="semibold", loc="left")
        ax_c.set_ylabel("Avg Min Clearance (+/- std)", fontsize=10)
        ax_c.set_xlabel("World #", fontsize=10)
        ax_c.legend(fontsize=9, loc="lower right")
        ax_c.grid(axis="y", linestyle="--", alpha=0.4)
        ax_c.spines[["top", "right"]].set_visible(False)
        _set_xticks(ax_c, worlds_c)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {output}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot per-world success rates and clearance.")
    parser.add_argument("files", nargs="+", help="Result file(s) to plot")
    parser.add_argument("--task_num", type=int, default=None,
                        help="Filter data across tasks")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Scheme names (one per file)")
    parser.add_argument("--max_world", type=int, default=None,
                        help="Limit to worlds 0..max_world-1")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to this path (e.g. results.png)")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        parser.error("--labels must have the same count as files")

    plot_success_rates(args.files, args.labels, args.max_world, args.output, args.task_num)
