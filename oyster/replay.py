import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap
from pathlib import Path

from py_mpcc import Spline, Trajectory

from oyster.MapLoader import (
    parse_xml_file,
    generate_map_from_cylinders,
    OccupancyGrid,
    Pose
)

# ── Style & Palette ───────────────────────────────────────────────────────────
DARK_BG      = "#EDE8D0"
PANEL_BG     = "#EDE8D0"
BORDER       = "#30363d"
TEXT_PRIMARY = "#594423"
TEXT_MUTED   = "#82796B"
C_UPPER      = "#ff6b6b"
C_LOWER      = "#4ecdc4"
C_HORIZON    = "#FFAE03"
C_ROBO       = "#FFAE03"
GRID_COLOR   = "#D1CDBC"
TRAIL_CMAP   = plt.get_cmap("magma")

OCC_COLORS = np.zeros((256, 4))
OCC_COLORS[253] = [0.2, 0.2, 0.2, 0.1]
OCC_COLORS[254] = [0.1, 0.1, 0.1, 0.2]
OCC_CMAP = ListedColormap(OCC_COLORS)

def _apply_global_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   PANEL_BG,
        "axes.edgecolor":   BORDER,
        "axes.labelcolor":  TEXT_PRIMARY,
        "axes.titlecolor":  TEXT_PRIMARY,
        "xtick.color":      TEXT_MUTED,
        "ytick.color":      TEXT_MUTED,
        "grid.color":       GRID_COLOR,
        "grid.linestyle":   "--",
        "grid.linewidth":   0.5,
        "font.family":      "monospace",
    })


def build_trajectory(knots, xs, ys):
    """Reconstruct a py_mpcc Trajectory from logged knots and control points."""
    knots = np.array(knots)
    xs    = np.array(xs)
    ys    = np.array(ys)
    spline_x = Spline(knots, xs)
    spline_y = Spline(knots, ys)
    return Trajectory(spline_x, spline_y)


def compute_tubes(trajectory, upper_coeffs, lower_coeffs, n_pts=100):
    """
    Compute upper and lower tube boundary points along the trajectory.
    Matches the logic from the working live plotter exactly.
    """
    extended_len = trajectory.get_arclen()
    ss           = np.linspace(0, extended_len, n_pts)
    tau_norm     = ss / extended_len

    upper_d = np.polyval(np.array(upper_coeffs)[::-1], tau_norm)
    lower_d = np.polyval(np.array(lower_coeffs)[::-1], tau_norm)

    traj     = np.vstack([trajectory(s) for s in ss])
    tangents = np.vstack([trajectory(s, 1) for s in ss])
    norms    = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (norms + 1e-6)
    normals  = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    upper = traj + upper_d[:, None] * normals
    lower = traj + lower_d[:, None] * normals

    return upper, lower


class ReplayPlotter:
    def __init__(self, json_path, mode="animate", fps=30, use_cmap_telemetry=False, save_dir=None):
        self.json_path = Path(json_path)
        with open(json_path) as f:
            self.data = json.load(f)

        self.start    = self.data["start_pos"]
        self.goal     = self.data["goal_pos"]
        self.frames   = self.data["frames"]
        self.mode     = mode
        self.fps      = fps
        self.use_cmap = use_cmap_telemetry
        self.save_dir = Path(save_dir) if save_dir else None

        self.obstacles = []
        match = re.search(r'world_(\d+)', self.json_path.name)
        if match:
            world_id  = match.group(1)
            barn_path = os.getenv("BARN_DATASET_PATH")
            if barn_path:
                w_file = Path(barn_path) / "world_files" / f"world_{world_id}.world"
                if w_file.exists():
                    self.obstacles = parse_xml_file(str(w_file), offset=-5)

        self.occ_grid = generate_map_from_cylinders(self.obstacles)

        _apply_global_style()
        self._build_figure()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 2, figure=self.fig, width_ratios=[2.2, 1], wspace=0.3, hspace=0.4)

        # ── Main map ──────────────────────────────────────────────────────────
        self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax.set_title(f"Replay: {self.json_path.name}", pad=20, fontsize=12, fontweight="bold")
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.grid(True, zorder=0)

        g = self.occ_grid
        w, h, res = g.info.width, g.info.height, g.info.resolution
        ox, oy    = g.info.origin.position[0], g.info.origin.position[1]
        extent    = (ox, ox + w * res, oy, oy + h * res)
        self.ax.imshow(
            np.array(g.data).reshape((h, w)),
            origin="lower", cmap=OCC_CMAP,
            extent=extent, interpolation="nearest", zorder=0.1
        )

        for obs in self.obstacles:
            self.ax.add_patch(Circle(
                (obs[0], obs[1]), obs[2],
                facecolor="#d1d5da", edgecolor="#24292e", zorder=1
            ))

        # Animated map elements
        self.path_coll = LineCollection([], cmap=TRAIL_CMAP, linewidth=3, zorder=4)
        self.path_coll.set_clim(0, 1)
        self.ax.add_collection(self.path_coll)

        self.robot_patch = Circle((0, 0), 0.07, facecolor=C_ROBO, edgecolor="black", zorder=7, lw=2)
        self.ax.add_patch(self.robot_patch)

        self.traj_line,    = self.ax.plot([], [], color=TEXT_PRIMARY, lw=1, alpha=0.2, zorder=3)
        self.horizon_line, = self.ax.plot([], [], color=C_HORIZON, lw=3, zorder=5)
        self.u_tube,       = self.ax.plot([], [], color=C_UPPER, lw=2, alpha=0.85, zorder=6)
        self.l_tube,       = self.ax.plot([], [], color=C_LOWER, lw=2, alpha=0.85, zorder=6)

        # ── Telemetry subplots ────────────────────────────────────────────────
        self.ax_a = self.fig.add_subplot(gs[0, 1])
        self.ax_h = self.fig.add_subplot(gs[1, 1])
        self.ax_v = self.fig.add_subplot(gs[2, 1])

        self.colls = {}
        telemetry_meta = [
            ('au', self.ax_a, "Alpha Scaling (α)",      "Value", C_UPPER),
            ('al', self.ax_a, "Alpha Scaling (α)",      "Value", C_LOWER),
            ('hu', self.ax_h, "Safety Function h(x)",   "h(x)",  C_UPPER),
            ('hl', self.ax_h, "Safety Function h(x)",   "h(x)",  C_LOWER),
            ('v',  self.ax_v, "Robot Linear Velocity",  "m/s",   None),
        ]
        for name, ax, title, ylabel, color in telemetry_meta:
            ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel("Time Step", fontsize=8)
            ax.grid(True)
            use_grad = self.use_cmap or color is None
            lc = LineCollection(
                [],
                cmap=TRAIL_CMAP if use_grad else None,
                color=color if not use_grad else None,
                lw=2.5
            )
            ax.add_collection(lc)
            self.colls[name] = lc

        # State buffers
        self.steps = []
        self.tx, self.ty = [], []
        self.vh, self.au, self.al, self.hu, self.hl = [], [], [], [], []

        # rx, ry = -2.25, -2.5
        # self.ax.set_xlim(rx - 5, rx + 5)
        # self.ax.set_ylim(ry - 2, ry + 12)
        min_x = min(self.goal[0], self.start[0])
        max_x = max(self.goal[0], self.start[0])
        min_y = min(self.goal[1], self.start[1])
        max_y = max(self.goal[1], self.start[1])

        self.ax.set_xlim(min_x - 5, max_x + 5)
        self.ax.set_ylim(min_y - 1, max_y + 1)
        self.ax.set_aspect("equal")

    def _update_tubes(self, fr):
        t     = fr.get("trajectory", {})
        tb    = fr.get("tubes")
        knots = t.get("knots")
        xs    = t.get("xs")
        ys    = t.get("ys")

        if not (knots and xs and ys and tb):
            self.traj_line.set_data([], [])
            self.u_tube.set_data([], [])
            self.l_tube.set_data([], [])
            return

        try:
            trajectory = build_trajectory(knots, xs, ys)

            # Full reference trajectory line (static, just the spline shape)
            arclen = trajectory.get_arclen()
            ss     = np.linspace(0, arclen, 200)
            pts    = np.vstack([trajectory(s) for s in ss])
            self.traj_line.set_data(pts[:, 0], pts[:, 1])

            # Find where the robot is on the trajectory and get the adjusted view
            # This makes tubes slide with the robot, matching the live plotter
            robot_pos  = fr["robot_pos"][:2]
            len_start  = trajectory.get_closest_s(robot_pos)
            ref_samples = len(fr.get("mpc_horizon", [])) or 20
            adjusted   = trajectory.get_adjusted_traj(len_start, int(ref_samples))
            view       = adjusted.view()

            # Evaluate tubes along the adjusted trajectory view
            extended_len = adjusted.get_arclen()
            ss_adj       = np.linspace(0, extended_len, 100)
            tau_norm     = ss_adj / extended_len

            upper_d = np.polyval(np.array(tb["top"])[::-1],    tau_norm)
            lower_d = np.polyval(np.array(tb["bottom"])[::-1], tau_norm)

            traj_pts = np.vstack([adjusted(s) for s in ss_adj])
            tangents = np.vstack([adjusted(s, 1) for s in ss_adj])
            norms    = np.linalg.norm(tangents, axis=1, keepdims=True)
            tangents = tangents / (norms + 1e-6)
            normals  = np.column_stack([-tangents[:, 1], tangents[:, 0]])

            upper = traj_pts + upper_d[:, None] * normals
            lower = traj_pts + lower_d[:, None] * normals

            self.u_tube.set_data(upper[:, 0], upper[:, 1])
            self.l_tube.set_data(lower[:, 0], lower[:, 1])

        except Exception as e:
            print(f"[warn] tube update failed: {e}")
            self.traj_line.set_data([], [])
            self.u_tube.set_data([], [])
            self.l_tube.set_data([], [])

    def _update_frame(self, i):
        fr = self.frames[i]
        rx, ry = fr["robot_pos"][:2]
        self.robot_patch.center = (rx, ry)

        self.tx.append(rx)
        self.ty.append(ry)
        self.steps.append(i)
        self.vh.append(fr.get("velocity", 0))
        self.au.append(fr.get("alpha_upper", 0))
        self.al.append(fr.get("alpha_lower", 0))
        self.hu.append(fr.get("cbf_upper", 0))
        self.hl.append(fr.get("cbf_lower", 0))

        # Robot trail
        if len(self.steps) > 1:
            segs = np.array([self.tx, self.ty]).T.reshape(-1, 1, 2)
            self.path_coll.set_segments(np.concatenate([segs[:-1], segs[1:]], axis=1))
            self.path_coll.set_array(np.linspace(0, i / len(self.frames), len(self.steps)))

            for k, d in zip(['au', 'al', 'hu', 'hl', 'v'],
                             [self.au, self.al, self.hu, self.hl, self.vh]):
                pts = np.array([self.steps, d]).T.reshape(-1, 1, 2)
                self.colls[k].set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
                if self.colls[k].get_cmap():
                    self.colls[k].set_array(np.linspace(0, 1, len(self.steps)))

        for ax, d in zip([self.ax_a, self.ax_h, self.ax_v],
                         [self.au + self.al, self.hu + self.hl, self.vh]):
            ax.set_xlim(0, max(i, 10))
            if d:
                s = max(max(d) - min(d), 0.1)
                ax.set_ylim(min(d) - s * 0.2, max(d) + s * 0.2)

        # MPC horizon — read directly from logged data, updates every frame
        mpc_horizon = fr.get("mpc_horizon", [])
        if len(mpc_horizon) > 1:
            hx = [p[0] for p in mpc_horizon]
            hy = [p[1] for p in mpc_horizon]
            self.horizon_line.set_data(hx, hy)
        else:
            self.horizon_line.set_data([], [])

        # Tubes — rebuilds when trajectory spline changes
        self._update_tubes(fr)

        return self.robot_patch,

    def run(self):
        if self.mode == "static":
            for i in range(len(self.frames)):
                self._update_frame(i)
            plt.show()
        else:
            self.anim = FuncAnimation(
                self.fig,
                self._update_frame,
                frames=len(self.frames),
                interval=1000 / self.fps,
                repeat=False,
            )
            if self.save_dir:
                p = self.save_dir / self.json_path.with_suffix(".mp4").name
                self.anim.save(p, writer=FFMpegWriter(fps=self.fps))
                print(f"Saved: {p}")
            else:
                plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("--mode", choices=["animate", "static"], default="animate")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--use-cmap-telemetry", action="store_true")
    parser.add_argument("--save-dir", type=str)
    args = parser.parse_args()
    ReplayPlotter(
        args.json_file,
        mode=args.mode,
        fps=args.fps,
        use_cmap_telemetry=args.use_cmap_telemetry,
        save_dir=args.save_dir
    ).run()
