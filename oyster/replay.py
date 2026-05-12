import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap
from pathlib import Path

from py_mpcc import Spline, Trajectory

from oyster.MapLoader import (
    parse_xml_file,
    generate_map_from_cylinders,
    OccupancyGrid,
    Pose,
)

# ── Style & Palette ───────────────────────────────────────────────────────────
DARK_BG      = "#FFFFFF"
PANEL_BG     = "#FFFFFF"
BORDER       = "#30363d"
TEXT_PRIMARY = "#594423"
TRAJECTORY   = "#00FF00"
TEXT_MUTED   = "#82796B"
C_UPPER      = "#c00000"
C_LOWER      = "#fb8500"
C_HORIZON    = "#222222"
C_ROBO       = "#FFAE03"
GRID_COLOR   = "#D1CDBC"
TRAIL_CMAP   = plt.get_cmap("cool")

OCC_COLORS = np.zeros((256, 4))
OCC_COLORS[253] = [0.2, 0.2, 0.2, 0.1]
OCC_COLORS[254] = [0.1, 0.1, 0.1, 0.2]
OCC_CMAP = ListedColormap(OCC_COLORS)

FOLLOW_WINDOW = 5.0


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
    knots = np.array(knots)
    xs    = np.array(xs)
    ys    = np.array(ys)
    return Trajectory(Spline(knots, xs), Spline(knots, ys))


# ── Base class — owns all figure/drawing logic ────────────────────────────────

class BasePlotter:
    """
    Owns the figure layout, robot patch, telemetry subplots, and per-frame
    drawing logic.  Does not know anything about where obstacles or maps come
    from; subclasses handle that.

    Subclasses must call _build_figure() after their own __init__ sets:
        self.start, self.goal, self.meta_data, self.follow
    """

    def _build_figure(self):
        self._closed = False
        self._map_img = None

        # telemetry buffers (frames intentionally excluded — subclasses own it)
        self.steps  = []
        self.tx, self.ty                            = [], []
        self.vh, self.au, self.al, self.hu, self.hl = [], [], [], [], []

        _apply_global_style()

        self.fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(
            3, 2, figure=self.fig,
            width_ratios=[1.6, 1],
            wspace=0.15, hspace=0.55,
            left=0.05, right=0.97, top=0.93, bottom=0.08,
        )

        # ── Main map axes ─────────────────────────────────────────────────────
        self.ax = self.fig.add_subplot(gs[:, 0])
        title = (
            f"{self.meta_data['model']}: {self.meta_data['max_speed']} m/s"
            if self.meta_data else "Live Plotter"
        )
        self.ax.set_title(title, pad=12, fontsize=12, fontweight="bold")
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.grid(True, zorder=0)
        self.ax.set_aspect("equal")

        if not self.follow:
            min_x = min(self.goal[0], self.start[0])
            max_x = max(self.goal[0], self.start[0])
            min_y = min(self.goal[1], self.start[1])
            max_y = max(self.goal[1], self.start[1])
            self.ax.set_xlim(min_x - 5, max_x + 5)
            self.ax.set_ylim(min_y - 1, max_y + 1)

        # Trail
        self.path_coll = LineCollection([], cmap=TRAIL_CMAP, linewidth=3, zorder=4)
        self.path_coll.set_clim(0, 1)
        self.ax.add_collection(self.path_coll)

        # Robot patch
        robot_size = 0.07
        theta = np.pi / 3
        if self.meta_data is None or self.meta_data.get("model") == "Double Integrator":
            self.robot_patch     = Circle((0, 0), robot_size,
                                          facecolor=C_ROBO, edgecolor="black",
                                          zorder=7, lw=2)
            self.robot_footprint = []
        else:
            robot_size *= 2
            self.robot_footprint = [
                ( robot_size, 0),
                (-robot_size * np.sin(theta), -robot_size * np.cos(theta)),
                (-robot_size * np.sin(theta),  robot_size * np.cos(theta)),
            ]
            self.robot_patch = Polygon(
                self.robot_footprint, closed=True,
                facecolor=C_ROBO, edgecolor="black",
                linewidth=1.2, alpha=0.9, zorder=7,
            )
        self.ax.add_patch(self.robot_patch)

        (self.traj_line,)    = self.ax.plot([], [], color=TRAJECTORY, lw=4, alpha=1.0, zorder=3)
        (self.horizon_line,) = self.ax.plot([], [], color=C_HORIZON,  lw=3, zorder=5)
        (self.u_tube,)       = self.ax.plot([], [], color=C_UPPER,    lw=2, alpha=0.85, zorder=6)
        (self.l_tube,)       = self.ax.plot([], [], color=C_LOWER,    lw=2, alpha=0.85, zorder=6)

        # ── Telemetry subplots ────────────────────────────────────────────────
        self.ax_a = self.fig.add_subplot(gs[0, 1])
        self.ax_h = self.fig.add_subplot(gs[1, 1])
        self.ax_v = self.fig.add_subplot(gs[2, 1])

        self.colls = {}
        for name, ax, ttl, ylabel, color in [
            ("au", self.ax_a, "Alpha Scaling (α)",     "Value", C_UPPER),
            ("al", self.ax_a, "Alpha Scaling (α)",     "Value", C_LOWER),
            ("hu", self.ax_h, "Safety Function h(x)",  "h(x)",  C_UPPER),
            ("hl", self.ax_h, "Safety Function h(x)",  "h(x)",  C_LOWER),
            ("v",  self.ax_v, "Robot Linear Velocity", "m/s",   None),
        ]:
            ax.set_title(ttl, loc="left", fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel("Time Step", fontsize=8)
            ax.grid(True)
            use_grad = color is None
            lc = LineCollection(
                [],
                cmap=TRAIL_CMAP if use_grad else None,
                color=color if not use_grad else None,
                lw=2.5,
            )
            ax.add_collection(lc)
            self.colls[name] = lc

    def _update_frame(self, i):
        fr     = self.frames[i]
        rx, ry = fr["robot_pos"][:2]

        if not self.robot_footprint:
            self.robot_patch.center = (rx, ry)
        else:
            state = np.array(fr["robot_pos"][:3])
            th    = state[2]
            R     = np.array([[np.cos(th), -np.sin(th)],
                               [np.sin(th),  np.cos(th)]])
            fp    = np.dot(self.robot_footprint, R.T) + state[:2]
            self.robot_patch.set_xy(fp.reshape(-1, 2))

        if self.follow:
            self.ax.set_xlim(rx - FOLLOW_WINDOW, rx + FOLLOW_WINDOW)
            self.ax.set_ylim(ry - FOLLOW_WINDOW, ry + FOLLOW_WINDOW)

        self.tx.append(rx);  self.ty.append(ry);  self.steps.append(i)
        self.vh.append(fr.get("velocity",    0))
        self.au.append(fr.get("alpha_upper", 0))
        self.al.append(fr.get("alpha_lower", 0))
        self.hu.append(fr.get("cbf_upper",   0))
        self.hl.append(fr.get("cbf_lower",   0))

        if len(self.steps) > 1:
            segs = np.array([self.tx, self.ty]).T.reshape(-1, 1, 2)
            self.path_coll.set_segments(np.concatenate([segs[:-1], segs[1:]], axis=1))
            self.path_coll.set_array(
                np.linspace(0, i / max(len(self.frames), 1), len(self.steps))
            )
            for k, d in zip(
                ["au", "al", "hu", "hl", "v"],
                [self.au, self.al, self.hu, self.hl, self.vh],
            ):
                pts = np.array([self.steps, d]).T.reshape(-1, 1, 2)
                self.colls[k].set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
                if self.colls[k].get_cmap():
                    self.colls[k].set_array(np.linspace(0, 1, len(self.steps)))

        for ax, d in zip(
            [self.ax_a, self.ax_h, self.ax_v],
            [self.au + self.al, self.hu + self.hl, self.vh],
        ):
            ax.set_xlim(0, max(i, 10))
            if d:
                s = max(max(d) - min(d), 0.1)
                ax.set_ylim(min(d) - s * 0.2, max(d) + s * 0.2)

        # MPC horizon
        horizon = fr.get("mpc_horizon", [])
        if len(horizon) > 1:
            self.horizon_line.set_data(
                [p[0] for p in horizon], [p[1] for p in horizon]
            )
        else:
            self.horizon_line.set_data([], [])

        # Trajectory spline
        t = fr.get("trajectory", {})
        knots, xs, ys = t.get("knots"), t.get("xs"), t.get("ys")
        if knots and xs and ys:
            traj = build_trajectory(knots, xs, ys)
            ss   = np.linspace(0, traj.get_arclen(), 200)
            pts  = np.vstack([traj(s) for s in ss])
            self.traj_line.set_data(pts[:, 0], pts[:, 1])

        # Tubes
        tube = fr.get("tubes")
        if tube:
            self.u_tube.set_data(*np.array(tube["top"]).T)
            self.l_tube.set_data(*np.array(tube["bottom"]).T)
        else:
            self.u_tube.set_data([], [])
            self.l_tube.set_data([], [])

        return (self.robot_patch,)

    def push_frame(self, frame):
        self.frames.append(frame)
        self._update_frame(len(self.frames) - 1)
        plt.pause(0.001)

    def close(self):
        self._closed = True
        plt.close(self.fig)


# ── World-file plotter (original behaviour, unchanged) ────────────────────────

class ReplayPlotter(BasePlotter):
    def __init__(
        self,
        json_path,
        mode="animate",
        fps=30,
        use_cmap_telemetry=False,
        save_dir=None,
        follow=False,
    ):
        self.json_path = Path(json_path)
        with open(json_path) as f:
            self.data = json.load(f)

        self.meta_data = self.data.get("metadata")
        self.start     = self.data["start_pos"]
        self.goal      = self.data["goal_pos"]
        self.frames    = self.data["frames"]
        self.mode      = mode
        self.fps       = fps
        self.use_cmap  = use_cmap_telemetry
        self.save_dir  = Path(save_dir) if save_dir else None
        self.follow    = follow

        self.load_obstacles()
        self._build_figure()
        self._draw_obstacles()

    def load_obstacles(self):
        self.obstacles = []
        if not self.data.get("world_nums"):
            match = re.search(r"world_(\d+)", self.json_path.name)
            world_ids = [match.group(1)] if match else []
        else:
            world_ids = self.data["world_nums"]

        offsets = [-5 + 8 * i for i in range(len(world_ids))]
        for i, world in enumerate(world_ids):
            barn_path = os.getenv("BARN_DATASET_PATH")
            if barn_path:
                w_file = Path(barn_path) / "world_files" / f"world_{world}.world"
                if w_file.exists():
                    obs = parse_xml_file(str(w_file), offset=offsets[i])
                    self.obstacles = (obs if len(self.obstacles) == 0
                                      else np.concatenate((obs, self.obstacles)))

        if len(self.obstacles) == 0:
            raise Exception("Could not figure out what world to load obstacles from!")

        self.occ_grid = generate_map_from_cylinders(self.obstacles)

    def _draw_obstacles(self):
        """Render the static cylinder map onto self.ax after _build_figure."""
        g = self.occ_grid
        w, h, res = g.info.width, g.info.height, g.info.resolution
        ox, oy    = g.info.origin.position[0], g.info.origin.position[1]
        extent    = (ox, ox + w * res, oy, oy + h * res)
        self.ax.imshow(
            np.array(g.data).reshape((h, w)),
            origin="lower", cmap=OCC_CMAP,
            extent=extent, interpolation="nearest", zorder=0.1,
        )
        for obs in self.obstacles:
            self.ax.add_patch(Circle(
                (obs[0], obs[1]), obs[2],
                facecolor="#d1d5da", edgecolor="#24292e", zorder=1,
            ))

    @classmethod
    def live(
        cls,
        fps=30,
        use_cmap_telemetry=False,
        meta_data=None,
        start_pos=None,
        goal_pos=None,
        obstacles=None,
        follow=False,
    ):
        instance = cls.__new__(cls)
        instance.json_path  = None
        instance.frames     = []
        instance.mode       = "live"
        instance.fps        = fps
        instance.use_cmap   = use_cmap_telemetry
        instance.meta_data  = meta_data
        instance.save_dir   = None
        instance.obstacles  = obstacles
        instance.start      = start_pos
        instance.goal       = goal_pos
        instance.follow     = follow
        instance.occ_grid   = generate_map_from_cylinders(instance.obstacles)
        instance._build_figure()
        instance._draw_obstacles()
        plt.ion()
        plt.show(block=False)
        return instance

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
    parser.add_argument("--mode",   choices=["animate", "static"], default="animate")
    parser.add_argument("--fps",    type=float, default=30)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--use-cmap-telemetry", action="store_true")
    parser.add_argument("--save-dir", type=str)
    args = parser.parse_args()
    ReplayPlotter(
        args.json_file,
        mode=args.mode,
        fps=args.fps,
        use_cmap_telemetry=args.use_cmap_telemetry,
        save_dir=args.save_dir,
        follow=args.follow,
    ).run()
