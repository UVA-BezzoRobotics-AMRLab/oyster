"""
fill_world_list.py

Takes a pre-computed list of "gap worlds" (where file1 outperforms file2)
and fills remaining slots with the highest success-rate worlds from file1
that are not already in the gap list, until a target total count is reached.

Output is one world index per line to stdout (pipe-able to a file or
directly into plot_success_rate.py --input).

Usage:
    python fill_world_list.py file1.txt --input gap_worlds.txt --total 30
    python fill_world_list.py file1.txt --input gap_worlds.txt --total 30 --task_num 6
    python fill_world_list.py file1.txt --input gap_worlds.txt --total 30 > final_worlds.txt

Data format (tab or space separated):
    WORLD  TASK  SUCCESS  STEPS  CLEARANCE
    0      6     1        238    0.42

--input file format (one world index per line):
    3
    7
    42
"""

import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


# ── Parsing ───────────────────────────────────────────────────────────────────

def load_results(filepath, task_num=None):
    world_successes = defaultdict(list)
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
            except ValueError:
                continue
    return world_successes


def load_world_list(filepath):
    """Read a file of world indices, one per line. Returns ordered list."""
    worlds = []
    seen   = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                w = int(line)
                if w not in seen:
                    worlds.append(w)
                    seen.add(w)
            except ValueError:
                continue
    return worlds


# ── Core logic ────────────────────────────────────────────────────────────────

def fill_world_list(file1, gap_worlds, total, task_num=None, max_world=None):
    """
    Start with gap_worlds, then fill remaining slots (up to total)
    with the highest success-rate worlds from file1 not already included.

    Returns the final ordered list of world indices:
        [gap worlds (in original order)] + [filler worlds (sorted by rate desc)]
    """
    ws1 = load_results(file1, task_num)

    # Filter by max_world if specified
    if max_world is not None:
        ws1 = {w: v for w, v in ws1.items() if w < max_world}

    gap_set = set(gap_worlds)
    n_slots = total - len(gap_worlds)

    if n_slots < 0:
        print(f"# WARNING: gap list ({len(gap_worlds)} worlds) already exceeds "
              f"total ({total}) — returning gap list as-is.", flush=True)
        for w in gap_worlds:
            print(w)
        return gap_worlds

    # Compute success rates for all worlds NOT in gap list
    candidates = []
    for w, results in ws1.items():
        if w in gap_set:
            continue
        rate = np.mean(results) * 100
        candidates.append((w, rate))

    # Sort by rate descending, then world index ascending for determinism
    candidates.sort(key=lambda x: (-x[1], x[0]))

    filler_worlds = [w for w, _ in candidates[:n_slots]]

    final = gap_worlds + filler_worlds
    return final, candidates


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fill a gap-world list with high success-rate worlds from file1."
    )
    parser.add_argument("file1",
                        help="Result file to source filler worlds from")
    parser.add_argument("--input", required=True,
                        help="Text file of pre-computed gap worlds (one per line)")
    parser.add_argument("--total", type=int, required=True,
                        help="Target total number of worlds in final list")
    parser.add_argument("--task_num", type=int, default=None,
                        help="Filter to a specific task number")
    parser.add_argument("--max_world", type=int, default=None,
                        help="Limit to worlds 0..max_world-1")
    parser.add_argument("--verbose", action="store_true",
                        help="Print a summary table to stderr instead of just world indices")
    args = parser.parse_args()

    gap_worlds = load_world_list(args.input)

    if not gap_worlds:
        raise SystemExit("ERROR: --input file is empty or could not be parsed.")

    if len(gap_worlds) >= args.total:
        import sys
        print(f"# WARNING: gap list already has {len(gap_worlds)} worlds "
              f">= total {args.total}. Truncating.", file=sys.stderr)
        gap_worlds = gap_worlds[:args.total]
        for w in gap_worlds:
            print(w)
        raise SystemExit(0)

    final, candidates = fill_world_list(
        args.file1,
        gap_worlds,
        args.total,
        task_num=args.task_num,
        max_world=args.max_world,
    )

    if args.verbose:
        import sys
        ws1 = load_results(args.file1, args.task_num)
        n_gap    = len(gap_worlds)
        n_filler = len(final) - n_gap

        print(f"# Gap worlds  : {n_gap}", file=sys.stderr)
        print(f"# Filler worlds: {n_filler}", file=sys.stderr)
        print(f"# Total        : {len(final)}", file=sys.stderr)
        print("#", file=sys.stderr)
        print(f"# {'World':>6}  {'Type':>8}  {'Rate (file1)':>12}", file=sys.stderr)
        print("# " + "-" * 32, file=sys.stderr)

        gap_set = set(gap_worlds)
        for w in final:
            rate    = np.mean(ws1[w]) * 100 if w in ws1 else float("nan")
            kind    = "gap" if w in gap_set else "filler"
            print(f"# {w:>6}  {kind:>8}  {rate:>11.1f}%", file=sys.stderr)
        print("#", file=sys.stderr)

    # Clean stdout output — one world per line, pipe-able
    for w in final:
        print(w)
