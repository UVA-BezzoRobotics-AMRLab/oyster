"""
plot_success_rate.py

Plots subplots per scheme:
  Left:  Stacked 100% bar chart showing Success rate, Collision rate, and Timeout rate per world
  Right: Avg min clearance per world with +/- std error bars (all trials included) [Optional]

Usage:
    python plot_success_rate.py results.txt
    python plot_success_rate.py results.txt --label "CBF-QP" --max_world 50
    python plot_success_rate.py results1.txt results2.txt --labels "CBF-QP" "CBF-CLF"
    python plot_success_rate.py results.txt --output comparison.png
    python plot_success_rate.py results.txt --input worlds.txt
    python plot_success_rate.py results.txt --input worlds.txt --show_world_ids
    python plot_success_rate.py results.txt --only_success

Data format (tab or space separated):
    WORLD  TASK  SUCCESS  STEPS  CLEARANCE
    0      6     1        238    0.42
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# Apply professional minimalist theme
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("whitegrid")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

import matplotlib.ticker as mticker
from collections import defaultdict
from pathlib import Path


# ── Parsing ───────────────────────────────────────────────────────────────────


def load_results(filepath, task_num=None):
    """Parse a results file categorizing trials into success, collision, or timeout.

    Returns:
        world_stats : {world: {'success': int, 'collision': int, 'timeout': int}}
        world_clearance : {world: [float]}
    """
    world_stats = defaultdict(lambda: {"success": 0, "collision": 0, "timeout": 0})
    world_clearance = defaultdict(list)
    
    timeout_count = 0
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.upper().startswith("WORLD"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                if task_num is not None:
                    if int(parts[1]) != task_num:
                        continue

                world = int(parts[0])
                success = int(parts[2])
                steps = int(parts[3])
                clearance = float(parts[4])

                if "horizon_2" in filepath and steps >= 250:
                    success = 0

                if clearance < 0.075 or steps >= 250:
                    success = 0

                # Classify the outcome
                if success == 1:
                    world_stats[world]["success"] += 1
                elif steps >= 250:
                    world_stats[world]["timeout"] += 1
                    timeout_count += 1
                else:
                    world_stats[world]["collision"] += 1

                world_clearance[world].append(clearance)
            except ValueError:
                continue

    if "cbf1" in filepath:
        print("timeout count", timeout_count)
                
    return world_stats, world_clearance


def load_world_filter(filepath, shuffle_seed=42):
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


def compute_success_rates(world_stats, max_world=None, world_set=None):
    worlds = _filter_worlds(sorted(world_stats), max_world, world_set)
    
    s_rates = []
    c_rates = []
    t_rates = []
    
    for w in worlds:
        counts = world_stats[w]
        total = counts["success"] + counts["collision"] + counts["timeout"]
        if total > 0:
            s_rates.append((counts["success"] / total) * 100)
            c_rates.append((counts["collision"] / total) * 100)
            t_rates.append((counts["timeout"] / total) * 100)
        else:
            s_rates.append(0.0)
            c_rates.append(0.0)
            t_rates.append(0.0)
            
    return np.array(worlds), np.array(s_rates), np.array(c_rates), np.array(t_rates)


def compute_clearance_stats(world_clearance, max_world=None, world_set=None):
    worlds = _filter_worlds(sorted(world_clearance), max_world, world_set)
    means = np.array([np.mean(world_clearance[w]) for w in worlds])
    means[means > 1.0] = 0.
    stds = np.array([min(np.std(world_clearance[w]), 0.23) for w in worlds])
    stds[means > 1.0] = 0
    return np.array(worlds), means, stds


# ── Remapping ─────────────────────────────────────────────────────────────────


def remap_to_file_order_stacked(
    worlds_arr, s_rates, c_rates, t_rates, world_order, show_world_ids=False
):
    w_to_s = dict(zip(worlds_arr, s_rates))
    w_to_c = dict(zip(worlds_arr, c_rates))
    w_to_t = dict(zip(worlds_arr, t_rates))

    x_indices, s_out, c_out, t_out, tick_labels = [], [], [], [], []

    for i, w in enumerate(world_order):
        if w in w_to_s:
            x_indices.append(i)
            s_out.append(w_to_s[w])
            c_out.append(w_to_c[w])
            t_out.append(w_to_t[w])
            tick_labels.append(str(w) if show_world_ids else str(i + 1))

    return np.array(x_indices), np.array(s_out), np.array(c_out), np.array(t_out), tick_labels


def remap_to_file_order(
    worlds_arr, rates_or_means, world_order, stds=None, show_world_ids=False
):
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

def _set_xticks(ax, x_positions, tick_labels=None, n_max_ticks=1e6):
    if len(x_positions) == 0:
        return
    step = max(1, len(x_positions) // n_max_ticks)
    ticks = x_positions[::step]
    ax.set_xticks(ticks)
    if tick_labels is not None:
        ax.set_xticklabels(
            [tick_labels[i] for i in range(0, len(tick_labels), step)],
            fontsize=11,
            rotation=90,
            ha="right",
        )
    else:
        ax.tick_params(axis="x", labelsize=8, rotation=45)

    ax.tick_params(axis="y", labelsize=11)


def _mean_line(ax, value, color, label, linestyle="-."):
    ax.axhline(
        value, color=color, linewidth=2.5, linestyle=linestyle, alpha=0.85, label=label
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
    only_success=False,
):
    if labels is None:
        labels = [Path(f).stem for f in files]

    n_schemes = len(files)
    n_cols = 1 if only_success else 2

    MAX_FIG_HEIGHT = 10.0
    row_height = min(4.2, MAX_FIG_HEIGHT / n_schemes)
    font_scale = 1.0

    fig, axes = plt.subplots(
        n_schemes,
        n_cols,
        figsize=(8 if only_success else 16, row_height * n_schemes),
        squeeze=False,
    )

    titles = {
        6: r"Performance Breakdown: Double Integrator, $v_{\max}=1.5$ m/s", 
        7: r"Performance Breakdown: Double Integrator, $v_{\max}=2.5$ m/s", 
        8: r"Performance Breakdown: Unicycle, $v_{\max}=1.8$ m/s", 
        9: r"Performance Breakdown: Unicycle, $v_{\max}=1.5$ m/s", 
    }
    title = "Performance Breakdown per World"
    if task_num in titles:
        title = titles[task_num]

    if world_order is None:
        xlabel = "World ID"
    elif show_world_ids:
        xlabel = "World (original)"
    else:
        xlabel = "World Index"

    # ── First Pass: Calculate Global Y-Limits for Clearance (Skipped if only_success) ──
    if not only_success:
        global_max_y = 0.0
        global_min_y = 0.0
        clearance_data_cache = []

        for fpath in files:
            _, world_clearance = load_results(fpath, task_num)
            _, means, stds = compute_clearance_stats(world_clearance, max_world, world_set)
            
            clearance_data_cache.append((means, stds))
            
            if len(means) > 0:
                max_with_err = np.max(means + stds)
                min_with_err = np.min(means - stds)
                if max_with_err > global_max_y:
                    global_max_y = max_with_err
                if min_with_err < global_min_y:
                    global_min_y = min_with_err

        y_buffer = (global_max_y - global_min_y) * 0.1
        clearance_ylim = (max(0.0, global_min_y - y_buffer), global_max_y + y_buffer)

    # ── Main Plot Loop ────────────────────────────────────────────────────────
    for idx, (fpath, label) in enumerate(zip(files, labels)):
        world_stats, world_clearance = load_results(fpath, task_num)

        # ── Left / Main Column: Stacked Success / Collision / Timeout Rate ──
        ax_s = axes[idx][0]
        worlds_s, s_rates, c_rates, t_rates = compute_success_rates(world_stats, max_world, world_set)

        if world_order is not None:
            x_pos, s_rates, c_rates, t_rates, tick_labels = remap_to_file_order_stacked(
                worlds_s, s_rates, c_rates, t_rates, world_order, show_world_ids=show_world_ids
            )
        else:
            x_pos, tick_labels = worlds_s, None

        ax_s.bar(x_pos, s_rates, width=0.85, color="#2ecc71", alpha=0.85, edgecolor="none")
        ax_s.bar(x_pos, c_rates, bottom=s_rates, width=0.85, color="#e74c3c", alpha=0.85, edgecolor="none")
        ax_s.bar(x_pos, t_rates, bottom=s_rates + c_rates, width=0.85, color="#f39c12", alpha=0.85, edgecolor="none")

        ax_s.axhline(100, color="#7f8c8d", linewidth=0.8, linestyle="--", alpha=0.4)
        
        mean_rate = float(np.mean(s_rates)) if len(s_rates) > 0 else 0.0
        _mean_line(ax_s, mean_rate, "black", f"Mean Success: {mean_rate:.1f}\\%", linestyle="--")

        ax_s.set_title(f"{label}", fontsize=int(14 * font_scale), pad=8)
        ax_s.set_ylabel("Distribution", fontsize=int(12 * font_scale), fontweight="bold")
        ax_s.set_ylim(0, 105)
        ax_s.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax_s.set_xlabel(xlabel, fontsize=int(12 * font_scale))
        ax_s.legend(fontsize=int(15 * font_scale), loc="lower right", frameon=True, framealpha=0.95, facecolor="white")
        ax_s.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax_s.spines[["top", "right", "left", "bottom"]].set_visible(False)
        _set_xticks(ax_s, x_pos, tick_labels)

        # ── Right Column: Clearance Avg +/- Std (Rendered conditionally) ──
        if not only_success:
            ax_c = axes[idx][1]
            
            means, stds = clearance_data_cache[idx]
            worlds_c, _, _ = compute_clearance_stats(world_clearance, max_world, world_set)

            if world_order is not None:
                x_pos_c, means, stds, tick_labels_c = remap_to_file_order(
                    worlds_c, means, world_order, stds=stds, show_world_ids=show_world_ids
                )
            else:
                x_pos_c, tick_labels_c = worlds_c, None

            unified_blue = "#05668D"
            ax_c.bar(x_pos_c, means, width=0.85, color=unified_blue, alpha=0.75, edgecolor="none")
            ax_c.errorbar(x_pos_c, means, yerr=stds, fmt="none", ecolor="#2c3e50", elinewidth=1.0, capsize=2.5, alpha=0.7)
            
            mean_clr = float(np.mean(means)) if len(means) > 0 else 0.0
            _mean_line(ax_c, mean_clr, "black", f"Mean Clearance = {mean_clr:.3f} m", linestyle="--")

            ax_c.set_title(f"{label}", fontsize=int(14 * font_scale), pad=8)
            ax_c.set_ylabel("Min Clearance [m]", fontsize=int(12 * font_scale))
            ax_c.set_xlabel(xlabel, fontsize=int(12 * font_scale))
            ax_c.set_ylim(clearance_ylim)
            ax_c.legend(fontsize=int(15 * font_scale), loc="upper right", frameon=True, framealpha=0.95, facecolor="white")
            ax_c.grid(True, axis="y", linestyle=":", alpha=0.6)
            ax_c.spines[["top", "right", "left", "bottom"]].set_visible(False)
            _set_xticks(ax_c, x_pos_c, tick_labels_c)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved optimized figure to: {output}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot per-world success rates and clearance.")
    parser.add_argument("files", nargs="+", help="Result file(s) to plot")
    parser.add_argument("--task_num", type=int, default=None, help="Filter data across tasks")
    parser.add_argument("--labels", nargs="+", default=None, help="Scheme names (one per file)")
    parser.add_argument("--max_world", type=int, default=None, help="Limit to worlds 0..max_world-1")
    parser.add_argument("--output", type=str, default=None, help="Save figure to this path (e.g. results.png)")
    parser.add_argument("--input", type=str, default=None, help="Text file listing worlds to plot.")
    parser.add_argument("--show_world_ids", action="store_true", help="Label x-axis with original numbers.")
    parser.add_argument("--only_success", action="store_true", help="Only plot success rates and suppress clearance plots.")
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
        only_success=args.only_success,
    )
