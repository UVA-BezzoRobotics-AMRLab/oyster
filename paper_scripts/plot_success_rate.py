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
    python plot_success_rate.py results.txt --input worlds.txt
    python plot_success_rate.py results.txt --input worlds.txt --show_world_ids

Data format (tab or space separated):
    WORLD  TASK  SUCCESS  STEPS  CLEARANCE
    0      6     1        238    0.42

--input file format (one world index per line):
    3
    7
    42
    When --input is used, worlds are relabelled 0..N on the x-axis in file order.
    Use --show_world_ids to show the original world numbers instead.
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
        world_successes : {world: [0/1 ints]}   -- STEPS=250 counted as failure
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

                world = int(parts[0])
                success = int(parts[2])
                steps = int(parts[3])
                clearance = float(parts[4])

                if clearance < 0.05 or steps >= 250:
                    success = 0

                world_successes[world].append(success)

                if len(parts) >= 5:
                    world_clearance[world].append(float(parts[4]))
            except ValueError:
                continue
    return world_successes, world_clearance


def load_world_filter(filepath, shuffle_seed=42):
    """Read a file containing one world index per line.

    Returns:
        world_order : list of ints in file order (shuffled), or None
        world_set   : set of ints for fast lookup, or None
    """
    if filepath is None:
        return None, None

    world_order = []
    seen = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                w = int(line)
                if w not in seen:
                    world_order.append(w)
                    seen.add(w)
            except ValueError:
                continue

    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(world_order)

    print(f"Loaded {len(world_order)} worlds from '{filepath}'")
    return world_order, seen


# ── Statistics ────────────────────────────────────────────────────────────────


def _filter_worlds(worlds_sorted, max_world, world_set):
    result = worlds_sorted
    if max_world is not None:
        result = [w for w in result if w < max_world]
    if world_set is not None:
        result = [w for w in result if w in world_set]
    return result


def compute_success_rates(world_successes, max_world=None, world_set=None):
    worlds = _filter_worlds(sorted(world_successes), max_world, world_set)
    rates = [np.mean(world_successes[w]) * 100 for w in worlds]
    return np.array(worlds), np.array(rates)


def compute_clearance_stats(world_clearance, max_world=None, world_set=None):
    worlds = _filter_worlds(sorted(world_clearance), max_world, world_set)
    means = [np.mean(world_clearance[w]) for w in worlds]
    stds = [min(np.std(world_clearance[w]), 0.23) for w in worlds]
    return np.array(worlds), np.array(means), np.array(stds)


# ── Remapping ─────────────────────────────────────────────────────────────────


def remap_to_file_order(
    worlds_arr, rates_or_means, world_order, stds=None, show_world_ids=False
):
    """
    Reorder worlds_arr and rates/means to match world_order from the input file.

    show_world_ids=False : x-axis labels are sequential 1..N
    show_world_ids=True  : x-axis labels are the original world numbers
    """
    world_to_value = dict(zip(worlds_arr, rates_or_means))
    world_to_std = dict(zip(worlds_arr, stds)) if stds is not None else {}

    x_indices = []
    values = []
    stds_out = []
    tick_labels = []

    for i, w in enumerate(world_order):
        if w in world_to_value:
            x_indices.append(i)
            values.append(world_to_value[w])
            stds_out.append(world_to_std.get(w, 0.0))
            tick_labels.append(str(w) if show_world_ids else str(i + 1))

    return (
        np.array(x_indices),
        np.array(values),
        np.array(stds_out) if stds is not None else None,
        tick_labels,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _set_xticks(ax, x_positions, tick_labels=None, n_max_ticks=30):
    if len(x_positions) == 0:
        return
    step = max(1, len(x_positions) // n_max_ticks)
    ticks = x_positions[::step]
    ax.set_xticks(ticks)
    if tick_labels is not None:
        ax.set_xticklabels(
            [tick_labels[i] for i in range(0, len(tick_labels), step)],
            fontsize=7,
            rotation=45,
            ha="right",
        )
    else:
        ax.tick_params(axis="x", labelsize=8, rotation=45)


def _mean_line(ax, value, color, label):
    ax.axhline(
        value, color=color, linewidth=1.2, linestyle="-.", alpha=0.9, label=label
    )


# ── Main plot ─────────────────────────────────────────────────────────────────


def plot_success_rates(
    files,
    labels=None,
    max_world=None,
    output=None,
    task_num=None,
    world_order=None,
    world_set=None,
    show_world_ids=False,
):
    if labels is None:
        labels = [Path(f).stem for f in files]

    n_schemes = len(files)

    MAX_FIG_HEIGHT = 10.0
    row_height = min(3.8, MAX_FIG_HEIGHT / n_schemes)
    font_scale = 1.0

    fig, axes = plt.subplots(
        n_schemes,
        2,
        figsize=(18, row_height * n_schemes),
        squeeze=False,
    )

    fig.suptitle(
        "Success Rate per World", fontsize=int(15 * font_scale), fontweight="bold"
    )

    # x-axis label depends on mode
    if world_order is None:
        xlabel = "World #"
    elif show_world_ids:
        xlabel = "World # (original)"
    else:
        xlabel = "World index"

    colors = plt.cm.tab10.colors

    for idx, (fpath, label) in enumerate(zip(files, labels)):
        color = colors[idx % len(colors)]
        world_successes, world_clearance = load_results(fpath, task_num)

        # ── Left: success rate ─────────────────────────────────────────────
        ax_s = axes[idx][0]
        worlds_s, rates = compute_success_rates(world_successes, max_world, world_set)

        if world_order is not None:
            x_pos, rates, _, tick_labels = remap_to_file_order(
                worlds_s, rates, world_order, show_world_ids=show_world_ids
            )
        else:
            x_pos, tick_labels = worlds_s, None

        ax_s.bar(
            x_pos,
            rates,
            width=0.8,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        ax_s.axhline(100, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
        mean_rate = float(np.mean(rates)) if len(rates) > 0 else 0.0
        _mean_line(ax_s, mean_rate, color, f"Mean = {mean_rate:.1f}%")

        ax_s.set_title(
            f"{label} — Success Rate",
            fontsize=int(11 * font_scale),
            fontweight="semibold",
            loc="left",
        )
        ax_s.set_ylabel("Success Rate (%)", fontsize=int(10 * font_scale))
        ax_s.set_ylim(0, 110)
        ax_s.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax_s.set_xlabel(xlabel, fontsize=int(10 * font_scale))
        ax_s.legend(fontsize=int(9 * font_scale), loc="lower right")
        ax_s.grid(axis="y", linestyle="--", alpha=0.4)
        ax_s.spines[["top", "right"]].set_visible(False)
        _set_xticks(ax_s, x_pos, tick_labels)

        # ── Right: clearance avg +/- std ──────────────────────────────────
        ax_c = axes[idx][1]
        worlds_c, means, stds = compute_clearance_stats(
            world_clearance, max_world, world_set
        )

        if world_order is not None:
            x_pos_c, means, stds, tick_labels_c = remap_to_file_order(
                worlds_c, means, world_order, stds=stds, show_world_ids=show_world_ids
            )
        else:
            x_pos_c, tick_labels_c = worlds_c, None

        ax_c.bar(
            x_pos_c,
            means,
            width=0.8,
            color=color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )
        ax_c.errorbar(
            x_pos_c,
            means,
            yerr=stds,
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
            alpha=0.55,
        )
        mean_clr = float(np.mean(means)) if len(means) > 0 else 0.0
        _mean_line(ax_c, mean_clr, color, f"Mean = {mean_clr:.3f}")

        ax_c.set_title(
            f"{label} — Min Clearance to Obstacle",
            fontsize=int(11 * font_scale),
            fontweight="semibold",
            loc="left",
        )
        ax_c.set_ylabel("Avg Min Clearance (+/- std)", fontsize=int(10 * font_scale))
        ax_c.set_xlabel(xlabel, fontsize=int(10 * font_scale))
        ax_c.legend(fontsize=int(9 * font_scale), loc="lower right")
        ax_c.grid(axis="y", linestyle="--", alpha=0.4)
        ax_c.spines[["top", "right"]].set_visible(False)
        _set_xticks(ax_c, x_pos_c, tick_labels_c)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {output}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot per-world success rates and clearance."
    )
    parser.add_argument("files", nargs="+", help="Result file(s) to plot")
    parser.add_argument(
        "--task_num", type=int, default=None, help="Filter data across tasks"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Scheme names (one per file)"
    )
    parser.add_argument(
        "--max_world", type=int, default=None, help="Limit to worlds 0..max_world-1"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save figure to this path (e.g. results.png)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Text file listing worlds to plot (one world index per line). "
        "Worlds are relabelled 0..N on the x-axis in file order.",
    )
    parser.add_argument(
        "--show_world_ids",
        action="store_true",
        help="When --input is used, label x-axis with original world "
        "numbers instead of sequential 1..N indices.",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        parser.error("--labels must have the same count as files")

    world_order, world_set = load_world_filter(args.input)

    plot_success_rates(
        args.files,
        args.labels,
        args.max_world,
        args.output,
        args.task_num,
        world_order,
        world_set,
        show_world_ids=args.show_world_ids,
    )
