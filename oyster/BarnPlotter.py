import os
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from matplotlib.cm import Greys

from pathlib import Path
from datetime import datetime
from oyster.RobotMPC import Dynamics
from oyster.MapLoader import OccupancyGrid
from py_planner import MapLayer
from py_mpcc import extend_trajectory

# ── Palette ────────────────────────────────────────────────────────────────────
DARK_BG = "#EDE8D0"
PANEL_BG = "#EDE8D0"
BORDER = "#30363d"
TEXT_PRIMARY = "#594423"
TEXT_MUTED = "#82796B"

C_UPPER = "#ff6b6b"  # upper / warning
C_LOWER = "#4ecdc4"  # lower / safe
C_HORIZON = "#FFAE03"  # MPC horizon
C_PATH = "#413C58"  # path trail
C_ROBO = "#FFAE03"
WHITE = "#f0f6fc"
GREEN = "#3fb950"  # positive reward
GRID_LINE = "#21262d"


# DARK_BG      = "#0d1117"
# PANEL_BG     = "#161b22"
# BORDER       = "#30363d"
# TEXT_PRIMARY = "#e6edf3"
# TEXT_MUTED   = "#7d8590"
#
# C_UPPER        = "#ff6b6b"   # upper / warning
# C_LOWER         = "#4ecdc4"   # lower / safe
# C_HORIZON         = "#ffd93d"   # MPC horizon
# C_PATH       = "#c084fc"   # path trail
# WHITE        = "#f0f6fc"
# GREEN        = "#3fb950"   # positive reward
# GRID_LINE    = "#21262d"

# Custom SDF colormap: dark-blue → transparent → warm-red
_sdf_colors = ["#353E43", "#628699", "#B9D9EB"]
SDF_CMAP = LinearSegmentedColormap.from_list("sdf", _sdf_colors)

# Custom occupancy colormap: free (dark) → occupied (off-white)
OCC_CMAP = LinearSegmentedColormap.from_list("occ", [DARK_BG, "#c9d1d9"])


def _apply_global_style():
    """Apply a coherent dark theme globally."""
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT_MUTED,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID_LINE,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.facecolor": PANEL_BG,
            "legend.edgecolor": BORDER,
            "legend.labelcolor": TEXT_PRIMARY,
            "legend.fontsize": 8,
            "font.family": "monospace",
            "text.color": TEXT_PRIMARY,
            "lines.antialiased": True,
            "patch.antialiased": True,
        }
    )


def _style_ax(ax, title="", xlabel="", ylabel="", zero_line=False):
    """Consistently style a subplot."""
    ax.set_title(
        title, fontsize=15, color=TEXT_PRIMARY, pad=6, fontweight="bold", loc="left"
    )
    ax.set_xlabel(xlabel, fontsize=12, color=TEXT_MUTED, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=12, color=TEXT_MUTED, labelpad=4)
    ax.tick_params(axis="both", which="both", length=3, color=BORDER)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    if zero_line:
        ax.axhline(0, color=BORDER, linewidth=0.8, zorder=0)


class BarnPlotter:
    def __init__(
        self,
        curve,
        occ_grid,
        map_util,
        upper_coeffs,
        lower_coeffs,
        mpc,
        dynamics=Dynamics.DOUBLE_INTEGRATOR,
        robot_size=0.2,
        theta=np.pi / 3,
        render=True,
        save_video=False,
        fps=30,
    ):
        self.dynamics = dynamics
        self.robot_size = robot_size
        self.robot_footprint = [
            (robot_size, 0),
            (-robot_size * np.sin(theta), -robot_size * np.cos(theta)),
            (-robot_size * np.sin(theta), robot_size * np.cos(theta)),
        ]

        self.prev_states = []
        self.save_video = save_video
        self.fps = fps
        self.writer = None
        self.map_util = map_util
        self.sdf_im = None

        if render:
            _apply_global_style()
            self.init_plot(curve, occ_grid, upper_coeffs, lower_coeffs, mpc)

    # ── Layout ─────────────────────────────────────────────────────────────────

    def init_plot(self, curve, occ_grid, upper_coeffs, lower_coeffs, mpc):
        self.fig = plt.figure(figsize=(18, 9), facecolor=DARK_BG)
        self.fig.subplots_adjust(
            left=0.04, right=0.97, top=0.93, bottom=0.08, wspace=0.32, hspace=0.55
        )

        # 2-col layout: map (wide) | 3 stacked telemetry panels
        gs = gridspec.GridSpec(
            3,
            2,
            figure=self.fig,
            width_ratios=[2.2, 1],
            height_ratios=[1, 1, 1],
        )

        # ── Main map axes (spans all rows, col 0) ───────────────────────────
        self.ax = self.fig.add_subplot(gs[:, 0])
        _style_ax(self.ax)

        max_vel = mpc.get_params().constraints.max_linvel
        self.fig.suptitle(
            f"BARN  ·  {self.dynamics.name}  ·  v_max = {max_vel:.2f} m/s",
            fontsize=13,
            fontweight="bold",
            color=TEXT_PRIMARY,
            x=0.5,
            y=0.98,
        )

        # thin accent border around map
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(1.2)

        # ── Telemetry axes (stacked in col 1) ───────────────────────────────
        self.ax_alphas = self.fig.add_subplot(gs[0, 1])
        self.ax_cbfs = self.fig.add_subplot(gs[1, 1])
        self.ax_reward = self.fig.add_subplot(gs[2, 1])

        _style_ax(
            self.ax_alphas,
            title="α  values",
            xlabel="step",
            ylabel="α",
            zero_line=False,
        )
        _style_ax(
            self.ax_cbfs,
            title="CBF  h(x)",
            xlabel="step",
            ylabel="h(x)",
            zero_line=True,
        )
        _style_ax(
            self.ax_reward,
            title="velocity",
            xlabel="step",
            ylabel="v  (m/s)",
            zero_line=False,
        )

        # ── Telemetry lines ─────────────────────────────────────────────────
        lw = 4.0

        (self.alpha_upper_line,) = self.ax_alphas.plot(
            [],
            [],
            color=C_UPPER,
            linewidth=lw,
            label="α upper",
            path_effects=[
                pe.Stroke(linewidth=lw + 1.5, foreground=DARK_BG),
                pe.Normal(),
            ],
        )
        (self.alpha_lower_line,) = self.ax_alphas.plot(
            [],
            [],
            color=C_LOWER,
            linewidth=lw,
            label="α lower",
            path_effects=[
                pe.Stroke(linewidth=lw + 1.5, foreground=DARK_BG),
                pe.Normal(),
            ],
        )

        self.ax_alphas.legend(loc="upper right", framealpha=0.6)

        (self.cbf_upper_line,) = self.ax_cbfs.plot(
            [],
            [],
            color=C_UPPER,
            linewidth=lw,
            label="h above",
            path_effects=[
                pe.Stroke(linewidth=lw + 1.5, foreground=DARK_BG),
                pe.Normal(),
            ],
        )
        (self.cbf_lower_line,) = self.ax_cbfs.plot(
            [],
            [],
            color=C_LOWER,
            linewidth=lw,
            label="h below",
            path_effects=[
                pe.Stroke(linewidth=lw + 1.5, foreground=DARK_BG),
                pe.Normal(),
            ],
        )

        # zero-crossing shading (filled between h and 0)
        self.cbf_upper_fill = self.ax_cbfs.fill_between(
            [], [], 0, color=C_UPPER, alpha=0.12, zorder=0
        )
        self.cbf_lower_fill = self.ax_cbfs.fill_between(
            [], [], 0, color=C_LOWER, alpha=0.12, zorder=0
        )

        self.ax_cbfs.legend(loc="upper right", framealpha=0.6)

        (self.reward_line,) = self.ax_reward.plot(
            [],
            [],
            color=GREEN,
            linewidth=lw,
            label="velocity",
            path_effects=[
                pe.Stroke(linewidth=lw + 1.5, foreground=DARK_BG),
                pe.Normal(),
            ],
        )

        # gradient fill under reward
        self.reward_fill = self.ax_reward.fill_between(
            [], [], 0, color=GREEN, alpha=0.15, zorder=0
        )

        self.ax_reward.legend(loc="upper right", framealpha=0.6)

        # histories
        self.reward_hist = []
        self.total_reward_hist = []
        self.alpha_upper_hist = []
        self.alpha_lower_hist = []
        self.cbf_upper_hist = []
        self.cbf_lower_hist = []
        self.steps = []

        # ── Map elements ────────────────────────────────────────────────────
        self._init_robot_patch()
        self.init_curve(curve)
        self.init_sdf_layer()
        self.init_grid(occ_grid)
        self.ax.add_patch(self.robot_patch)
        self.init_tubes(curve, upper_coeffs, lower_coeffs)

        # reference point
        (self.ref_point,) = self.ax.plot(
            [], [], "o", color=C_HORIZON, markersize=5, label="reference", zorder=6
        )

        # robot path trail
        (self.path_line,) = self.ax.plot(
            [],
            [],
            color=C_PATH,
            linewidth=1.5,
            alpha=0.7,
            linestyle="-",
            zorder=4,
            label="trail",
        )

        # MPC horizon
        (self.mpc_horizon,) = self.ax.plot(
            [],
            [],
            color=C_HORIZON,
            linewidth=4.2,
            alpha=0.8,
            zorder=5,
            label="MPC horizon",
        )

        # reference trajectory from MPC belief
        (self.mpc_trajectory_belief,) = self.ax.plot(
            [], [], color=WHITE, linewidth=1.2, alpha=0.4, linestyle="--", zorder=3
        )

        # map legend
        self.ax.legend(loc="upper left", framealpha=0.5, fontsize=7)

        # ── Video writer ────────────────────────────────────────────────────
        if self.save_video:
            self.writer = FFMpegWriter(fps=self.fps, bitrate=1800)
            script_dir = Path(__file__).parent.absolute()
            video_folder = os.path.join(script_dir, "videos")
            os.makedirs(video_folder, exist_ok=True)
            now = datetime.now()
            t_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            v_max = round(mpc.get_params().constraints.max_linvel, 2)
            v_str = str(v_max).replace(".", "_")
            dyn_str = str(self.dynamics).split(".")[-1]
            fname = f"{dyn_str}_{v_str}_{t_str}.mp4"
            self.writer.setup(self.fig, os.path.join(video_folder, fname), dpi=120)

    # ── Robot patch ─────────────────────────────────────────────────────────────

    def _init_robot_patch(self):
        if self.dynamics == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_patch = Circle(
                (0, 0),
                self.robot_size,
                facecolor=C_ROBO,
                edgecolor=WHITE,
                linewidth=1.2,
                alpha=0.9,
                label="robot",
                zorder=7,
            )
        else:
            self.robot_patch = Polygon(
                self.robot_footprint,
                closed=True,
                facecolor=C_ROBO,
                edgecolor=WHITE,
                linewidth=1.2,
                alpha=0.9,
                label="robot",
                zorder=7,
            )

    # ── Map layers ──────────────────────────────────────────────────────────────

    def init_sdf_layer(self):
        sdf = np.zeros((50, 50))
        self.sdf_im = self.ax.imshow(
            sdf,
            origin="lower",
            cmap=SDF_CMAP,
            alpha=0.25,
            zorder=1,
            interpolation="bilinear",
        )

    def init_grid(self, grid, ax=None):
        if ax is None:
            ax = self.ax

        w, h = grid.info.width, grid.info.height
        res = grid.info.resolution
        ox = grid.info.origin.position[0]
        oy = grid.info.origin.position[1]

        data = np.array(grid.data).reshape((h, w))
        vis = np.where(data < 0, 0, data)

        x_min, x_max = ox, ox + w * res
        y_min, y_max = oy, oy + h * res

        self.grid_im = ax.imshow(
            vis,
            origin="lower",
            cmap=OCC_CMAP,
            extent=(x_min, x_max, y_min, y_max),
            interpolation="nearest",
            zorder=0,
        )

    def init_curve(self, curve, ax=None):
        if ax is None:
            ax = self.ax
        (self.traj_line,) = ax.plot(
            curve.xs,
            curve.ys,
            color=TEXT_MUTED,
            linewidth=5.0,
            linestyle="-",
            alpha=0.6,
            zorder=2,
            label="global path",
        )

    def init_tubes(self, curve, upper_coeffs, lower_coeffs, ax=None):
        if ax is None:
            ax = self.ax

        tube_kw = dict(linewidth=4.0, alpha=0.75, zorder=3)
        belief_kw = dict(linewidth=1.2, alpha=0.4, linestyle="--", zorder=3)

        (self.upper_tube_line,) = ax.plot(
            [], [], color=C_UPPER, **tube_kw, label="tube upper"
        )
        (self.lower_tube_line,) = ax.plot(
            [], [], color=C_LOWER, **tube_kw, label="tube lower"
        )
        (self.upper_tube_belief,) = ax.plot([], [], color=C_UPPER, **belief_kw)
        (self.lower_tube_belief,) = ax.plot([], [], color=C_LOWER, **belief_kw)
        (self.upper_tube_pts,) = ax.plot(
            [], [], "o", color=C_UPPER, markersize=5, zorder=4
        )
        (self.lower_tube_pts,) = ax.plot(
            [], [], "o", color=C_LOWER, markersize=5, zorder=4
        )

    # ── Logging ─────────────────────────────────────────────────────────────────

    def log_reward(self, reward):
        self.reward_hist.append(reward)

    def add_state_to_path(self, robot_state):
        if len(self.prev_states) > 100:
            self.prev_states.pop(0)
        self.prev_states.append(list(robot_state))

    def clear_state_path(self):
        self.prev_states.clear()

    # ── Tube geometry ───────────────────────────────────────────────────────────

    def plot_tubes(self, curve, robot_state, mpc, upper_coeffs, lower_coeffs):
        trajectory = mpc.get_trajectory()
        len_start = trajectory.get_closest_s(robot_state[:2])
        adjusted_traj = trajectory.get_adjusted_traj(
            len_start, int(mpc.get_params().ref_samples)
        )
        traj_view = adjusted_traj.view()
        xs, ys = traj_view.xs, traj_view.ys

        state = mpc.get_state_from_horizon(0)
        args = {
            "i0": state,
            "i1": mpc.get_input_from_horizon(0),
            "i2": xs,
            "i3": ys,
            "i4": upper_coeffs,
            "i5": lower_coeffs,
            "i6": mpc.get_params().clf.w_lag_e,
            "i7": mpc.get_params().clf.w_contour_e,
            "i8": mpc.get_params().clf.gamma,
            "i9": traj_view.arclen,
        }

        extended_len = adjusted_traj.get_arclen()
        tau = np.linspace(0, extended_len, 100)
        upper_d = np.zeros(100)
        lower_d = np.zeros(100)

        for i in range(100):
            state[4] = tau[i]
            upper_d[i] = mpc.debug_fns["d_abv"](**args)
            lower_d[i] = mpc.debug_fns["d_blw"](**args)

        upper_d_actual = np.polyval(upper_coeffs[::-1], tau / extended_len)
        lower_d_actual = np.polyval(lower_coeffs[::-1], tau / extended_len)

        ss = np.linspace(0, extended_len, 100)
        traj = np.vstack([adjusted_traj(s) for s in ss])
        tangents = np.vstack([adjusted_traj(s, 1) for s in ss])
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        upper = traj + upper_d[:, None] * normals
        lower = traj + lower_d[:, None] * normals
        upper_actual = traj + upper_d_actual[:, None] * normals
        lower_actual = traj + lower_d_actual[:, None] * normals

        self.upper_tube_line.set_data(upper[:, 0], upper[:, 1])
        self.lower_tube_line.set_data(lower[:, 0], lower[:, 1])
        self.upper_tube_belief.set_data(upper_actual[:, 0], upper_actual[:, 1])
        self.lower_tube_belief.set_data(lower_actual[:, 0], lower_actual[:, 1])

    def plot_vector(self, start, vec):
        end = np.array(start) + np.array(vec)
        self.ax.plot([start[0], end[0]], [start[1], end[1]])

    # ── Main render ─────────────────────────────────────────────────────────────

    def render(self, robot_state, current_ref, curve, mpc, upper_coeffs, lower_coeffs):
        # ── Robot pose ──────────────────────────────────────────────────────
        if self.dynamics == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_patch.center = (robot_state[0], robot_state[1])
        else:
            theta = robot_state[2]
            R = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            footprint = np.dot(self.robot_footprint, R.T)
            self.robot_patch.set_xy((footprint + robot_state[:2]).reshape(-1, 2))

        # ── Trail ───────────────────────────────────────────────────────────
        if len(self.prev_states) > 1:
            path = np.array(self.prev_states)
            self.path_line.set_data(path[:, 0], path[:, 1])

        # ── MPC horizon ─────────────────────────────────────────────────────
        horiz_obj = mpc.get_horizon()
        horiz_len = horiz_obj.length
        horizon = np.array([mpc.get_state_from_horizon(i) for i in range(horiz_len)])
        self.mpc_horizon.set_data(horizon[:, 0], horizon[:, 1])

        # ── Trajectory reference ─────────────────────────────────────────────
        state = mpc.get_state_from_horizon(0)
        u = mpc.get_input_from_horizon(0)
        trajectory = mpc.get_trajectory()
        len_start = trajectory.get_closest_s(state[:2])
        adjusted_traj = trajectory.get_adjusted_traj(
            len_start, int(mpc.get_params().ref_samples)
        )
        traj_view = adjusted_traj.view()

        tau = np.linspace(0, adjusted_traj.get_arclen(), 30)
        self.traj_line.set_data(
            [adjusted_traj(s)[0] for s in tau],
            [adjusted_traj(s)[1] for s in tau],
        )

        non_ext = mpc.get_non_extended_trajectory()
        tau2 = np.linspace(0, non_ext.get_arclen(), 30)
        self.mpc_trajectory_belief.set_data(
            [non_ext(s)[0] for s in tau2],
            [non_ext(s)[1] for s in tau2],
        )

        self.plot_tubes(curve, robot_state, mpc, upper_coeffs, lower_coeffs)

        # ── Camera follow ────────────────────────────────────────────────────
        x, y = robot_state[:2]
        half_sz = 2
        self.ax.set_xlim(x - half_sz, x + half_sz)
        self.ax.set_ylim(y - half_sz, y + half_sz)

        # ── SDF heatmap ──────────────────────────────────────────────────────
        N = 60
        xs_grid = np.linspace(x - half_sz, x + half_sz, N)
        ys_grid = np.linspace(y - half_sz, y + half_sz, N)
        X, Y = np.meshgrid(xs_grid, ys_grid)
        vec_sdf = np.vectorize(self.map_util.sdf_dist)
        sdf_vals = vec_sdf(X, Y, MapLayer.kInflated)

        self.sdf_im.set_data(sdf_vals)
        self.sdf_im.set_extent((x - half_sz, x + half_sz, y - half_sz, y + half_sz))
        self.sdf_im.set_clim(vmin=0, vmax=0.5)

        # ── Step counter — append ONCE, all hists use the same index ─────────
        t = len(self.steps)
        self.steps.append(t)

        # ── Velocity — append immediately so len matches steps ────────────────
        _s = mpc.get_state_from_horizon(0)
        if self.dynamics == Dynamics.DOUBLE_INTEGRATOR:
            vel = float(np.linalg.norm(_s[2:4]))
        else:
            vel = float(_s[3])
        self.reward_hist.append(vel)

        # ── Alpha — append then plot (same length as steps) ───────────────────
        params = mpc.get_params()
        self.alpha_upper_hist.append(params.cbf.alpha_abv)
        self.alpha_lower_hist.append(params.cbf.alpha_blw)

        self.alpha_upper_line.set_data(
            self.steps, self.alpha_upper_hist[: len(self.steps)]
        )
        self.alpha_lower_line.set_data(
            self.steps, self.alpha_lower_hist[: len(self.steps)]
        )
        self.ax_alphas.relim()
        self.ax_alphas.autoscale_view()

        # ── CBF — append then plot ─────────────────────────────────────────────
        cbf_abv, cbf_blw = self.get_cbfs(mpc, upper_coeffs, lower_coeffs)
        self.cbf_upper_hist.append(cbf_abv)
        self.cbf_lower_hist.append(cbf_blw)

        self.cbf_upper_line.set_data(self.steps, self.cbf_upper_hist[: len(self.steps)])
        self.cbf_lower_line.set_data(self.steps, self.cbf_lower_hist[: len(self.steps)])
        self.ax_cbfs.relim()
        self.ax_cbfs.autoscale_view()

        # ── Velocity line + fill ─────────────────────────────────────────────
        self.reward_line.set_data(self.steps, self.reward_hist[: len(self.steps)])
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()

        # ── Shaded fills — all hists same length now, safe to recreate ───────
        self.cbf_upper_fill.remove()
        self.cbf_lower_fill.remove()
        self.cbf_upper_fill = self.ax_cbfs.fill_between(
            self.steps,
            self.cbf_upper_hist,
            0,
            color=C_UPPER,
            alpha=0.12,
            zorder=0,
        )
        self.cbf_lower_fill = self.ax_cbfs.fill_between(
            self.steps,
            self.cbf_lower_hist,
            0,
            color=C_LOWER,
            alpha=0.12,
            zorder=0,
        )
        self.reward_fill.remove()
        self.reward_fill = self.ax_reward.fill_between(
            self.steps,
            self.reward_hist[: len(self.steps)],
            0,
            color=GREEN,
            alpha=0.15,
            zorder=0,
        )

        plt.pause(0.001)

        if self.save_video:
            self.writer.grab_frame()

    # ── CBF helper ──────────────────────────────────────────────────────────────

    def get_cbfs(self, mpc, upper_coeffs, lower_coeffs):
        state = mpc.get_state_from_horizon(0)
        u = mpc.get_input_from_horizon(0)
        trajectory = mpc.get_trajectory()
        len_start = max(trajectory.get_closest_s(state[:2]), 1e-2)
        state[-2] = max(state[-2], 1e-2)
        adj = trajectory.get_adjusted_traj(
            len_start, int(mpc.get_params().ref_samples)
        ).view()

        args = {
            "i0": state,
            "i1": u,
            "i2": adj.xs,
            "i3": adj.ys,
            "i4": upper_coeffs,
            "i5": lower_coeffs,
            "i6": mpc.get_params().clf.w_lag_e,
            "i7": mpc.get_params().clf.w_contour_e,
            "i8": mpc.get_params().clf.gamma,
            "i9": trajectory.get_arclen(),
        }
        return float(mpc.debug_fns["h_abv"](**args)), float(
            mpc.debug_fns["h_blw"](**args)
        )

    # ── Cleanup ─────────────────────────────────────────────────────────────────

    def close(self):
        if self.save_video and self.writer is not None:
            self.writer.finish()
        plt.close("all")
