import numpy as np
import matplotlib
# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from RobotMPC import Dynamics
from matplotlib.patches import Circle, Polygon


class Plotter:
    def __init__(
        self,
        curve,
        upper_coeffs,
        lower_coeffs,
        obstacles,
        dynamics=Dynamics.DOUBLE_INTEGRATOR,
        robot_size=0.2,
        theta=np.pi / 3,
        render=True
    ):
        self.dynamics = dynamics
        self.robot_size = robot_size
        self.robot_footprint = [
            (robot_size, 0),
            (
                -robot_size * np.sin(theta),
                -robot_size * np.cos(theta),
            ),
            (
                -robot_size * np.sin(theta),
                robot_size * np.cos(theta),
            ),
        ]

        self.prev_states = []

        if render:
            self.init_plot(curve, upper_coeffs, lower_coeffs, obstacles)

    def init_plot(self, curve, upper_coeffs, lower_coeffs, obstacles):

        # visualization setup
        self.fig = plt.figure(figsize=(16, 8))

        self.ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
        # self.ax.set_aspect("equal", adjustable="box")

        self.ax_alpha_upper = plt.subplot2grid((2, 2), (0, 1))
        self.ax_alpha_lower = plt.subplot2grid((2, 2), (1, 1), sharex=self.ax_alpha_upper)
        # self.fig, (self.ax, self.ax_alpha_col) = plt.subplots(
        #         1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [3, 1]}
        # )
        # self.ax.set_aspect("equal")

        # self.ax_alpha_upper, self.ax_alpha_lower = self.ax_alpha_col.figure.subplots(2, 1, sharex=True)

        (self.alpha_upper_line,) = self.ax_alpha_upper.plot([], [], "r-", label="alpha_upper")
        (self.alpha_lower_line,) = self.ax_alpha_lower.plot([], [], "b-", label="alpha_lower")

        self.ax_alpha_upper.set_ylabel(r"$\alpha_{upper}$")
        self.ax_alpha_lower.set_ylabel(r"$\alpha_{lower}$")
        self.ax_alpha_lower.set_xlabel("Time step")
        self.ax_alpha_upper.grid(True, linestyle="--", alpha=0.5)
        self.ax_alpha_lower.grid(True, linestyle="--", alpha=0.5)

        self.alpha_upper_hist = []
        self.alpha_lower_hist = []
        self.alpha_t = []

        if self.dynamics == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_patch = Circle(
                (0, 0), self.robot_size, color="blue", label="robot"
            )
        else:
            self.robot_patch = Polygon(
                self.robot_footprint,
                closed=True,
                color="blue",
                label="robot",
            )
        
        self.init_curve(curve)

        (self.ref_point,) = self.ax.plot([], [], "ro", label="reference")

        (self.path_line,) = self.ax.plot([], [], "b-", linewidth=1.5)

        self.init_tubes(curve, upper_coeffs, lower_coeffs)
        self.init_obs(obstacles)

        self.ax.add_patch(self.robot_patch)

    def init_curve(self, curve, ax=None):
        if ax is None:
            ax = self.ax

        margin = 2.0

        p_min = min(np.min(curve.xs), np.min(curve.ys))
        p_max = max(np.max(curve.xs), np.max(curve.ys))

        x_min, y_min = p_min, p_min
        x_max, y_max = p_max, p_max

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        (self.traj_line,) = ax.plot(curve.xs, curve.ys, "k--", label="trajectory")


    def init_tubes(self, curve, upper_coeffs, lower_coeffs, ax=None):
        if ax is None:
            ax = self.ax

        # tubes (initialized as empty lines)
        (self.upper_tube_line,) = ax.plot(
            [], [], "r-", label="upper tube", linewidth=2.5
        )
        (self.lower_tube_line,) = ax.plot(
            [], [], "b-", label="lower tube", linewidth=2.5
        )

        ss = np.linspace(0, curve.get_arclen(), 100)
        traj = np.vstack([curve.trajx(ss), curve.trajy(ss)]).T

        tangents = np.column_stack([curve.trajx_d(ss), curve.trajy_d(ss)])
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        # compute normals
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        upper_d = np.polyval(upper_coeffs[::-1], ss)
        lower_d = np.polyval(lower_coeffs[::-1], ss)

        # offset trajectory
        upper = traj + upper_d[:, None] * normals
        lower = traj - lower_d[:, None] * normals

        ax.plot(
            upper[:, 0], upper[:, 1], "r-", label="upper tube", alpha=0.3, linewidth=2.5
        )
        ax.plot(
            lower[:, 0], lower[:, 1], "r-", label="lower tube", alpha=0.3, linewidth=2.5
        )

    def init_obs(self, obstacles, ax=None):
        if ax is None:
            ax = self.ax

        ax.plot(
            obstacles[:, 0],
            obstacles[:, 1],
            "ko",
            label="obstacles_p",
        )

        ax.plot(
            obstacles[:, 0],
            obstacles[:, 1],
            "bo",
            label="obstacles_n",
        )

    def plot_tubes(self, curve, tube_gen, robot_state, mpc):

        # figure out where to start and stop the tubes
        len_start = mpc.get_s_from_pose(robot_state[:2])
        curve_len = curve.get_arclen()
        if len_start > curve_len - 1e-2:
            return

        ref_len = mpc.get_params()["REF_LENGTH"]
        len_stop = min(len_start + ref_len, curve_len)

        ss = np.linspace(len_start, len_stop, 100)
        traj = np.vstack([curve.trajx(ss), curve.trajy(ss)]).T

        tangents = np.column_stack([curve.trajx_d(ss), curve.trajy_d(ss)])
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        # compute normals
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        upper_coeffs, lower_coeffs = tube_gen.shift_poly_parameter(len_start, ref_len)

        tau = np.linspace(0, len_stop - len_start, 100)
        upper_d = np.polyval(upper_coeffs[::-1], tau)
        lower_d = np.polyval(lower_coeffs[::-1], tau)

        # offset trajectory
        upper = traj + upper_d[:, None] * normals
        lower = traj - lower_d[:, None] * normals

        # plot tubes
        self.upper_tube_line.set_data(upper[:, 0], upper[:, 1])
        self.lower_tube_line.set_data(lower[:, 0], lower[:, 1])

    def render(self, robot_state, current_ref, curve, tube_gen, mpc):
        if self.dynamics == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_patch.center = (robot_state[0], robot_state[1])
        else:
            theta = robot_state[2]
            R = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            footprint = np.dot(self.robot_footprint, R.T)
            # print((footprint + robot_state[:2]).reshape(-1, 2))
            self.robot_patch.set_xy((footprint + robot_state[:2]).reshape(-1, 2))

        self.ref_point.set_data([current_ref[0]], [current_ref[1]])

        if len(self.prev_states) > 1:
            path = np.array(self.prev_states)
            # if len(self.prev_states) == 100:
            #     print(path)
            #     exit(0)
            self.path_line.set_data(path[:, 0], path[:, 1])

        params = mpc.get_params()
        alpha_upper = params["CBF_ALPHA_ABV"]
        alpha_lower = params["CBF_ALPHA_BLW"]

        t = len(self.alpha_t)
        self.alpha_t.append(t)
        self.alpha_upper_hist.append(alpha_upper)
        self.alpha_lower_hist.append(alpha_lower)

        self.alpha_upper_line.set_data(self.alpha_t, self.alpha_upper_hist)
        self.alpha_lower_line.set_data(self.alpha_t, self.alpha_lower_hist)

        for ax in [self.ax_alpha_upper, self.ax_alpha_lower]:
            ax.relim()
            ax.autoscale_view()

        self.plot_tubes(
            curve,
            tube_gen,
            robot_state,
            mpc,
        )

        plt.pause(0.001)

    def add_state_to_path(self, robot_state):
        # if more than 60 points, remove oldest
        if len(self.prev_states) > 100:
            self.prev_states.pop(0)
        self.prev_states.append(list(robot_state))

    def clear_state_path(self):
        self.prev_states.clear()

    def close(self):
        plt.close("all")
