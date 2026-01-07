import numpy as np
import matplotlib

# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from oyster.RobotMPC import Dynamics
from oyster.MapLoader import OccupancyGrid
from matplotlib.patches import Circle, Polygon


class BarnPlotter:
    def __init__(
        self,
        curve,
        occ_grid,
        upper_coeffs,
        lower_coeffs,
        mpc,
        dynamics=Dynamics.DOUBLE_INTEGRATOR,
        robot_size=0.2,
        theta=np.pi / 3,
        render=True,
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
            self.init_plot(curve, occ_grid, upper_coeffs, lower_coeffs, mpc)

    def init_plot(self, curve, occ_grid, upper_coeffs, lower_coeffs, mpc):

        # visualization setup
        self.fig = plt.figure(figsize=(16, 8))
        # fig_manager = plt.get_current_fig_manager()
        # self.fig.canvas.manager.full_screen_toggle()

        max_vel = mpc.get_params()["LINVEL"]
        self.ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
        self.ax.set_title(f"{self.dynamics.name}:   max_vel={max_vel} m/s")
        # self.ax.set_aspect("equal", adjustable="box")

        self.ax_alphas = plt.subplot2grid((2, 2), (0, 1))
        self.ax_cbfs = plt.subplot2grid((2, 2), (1, 1))

        (self.alpha_upper_line,) = self.ax_alphas.plot(
            [], [], "r-", label="alpha_upper", linewidth=2
        )
        (self.alpha_lower_line,) = self.ax_alphas.plot(
            [], [], "b-", label="alpha_lower", linewidth=2
        )

        self.ax_alphas.set_ylabel(r"$\alpha$ values")
        self.ax_alphas.set_xlabel("Time step")
        self.ax_alphas.grid(True, linestyle="--", alpha=0.5)

        self.alpha_upper_hist = []
        self.alpha_lower_hist = []
        self.steps = []

        (self.cbf_upper_line,) = self.ax_cbfs.plot(
            [], [], "r-", label="cbf_abv", linewidth=2
        )
        (self.cbf_lower_line,) = self.ax_cbfs.plot(
            [], [], "b-", label="cbf_blw", linewidth=2
        )

        self.ax_cbfs.set_ylabel(r"$h(x)$ values")
        self.ax_cbfs.set_xlabel("Time step")
        self.ax_cbfs.grid(True, linestyle="--", alpha=0.5)

        self.cbf_upper_hist = []
        self.cbf_lower_hist = []

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

        (self.mpc_horizon,) = self.ax.plot([], [], "g", linewidth=3.0)

        self.init_grid(occ_grid)

        self.ax.add_patch(self.robot_patch)

        self.init_tubes(curve, upper_coeffs, lower_coeffs)

    def init_grid(self, grid, ax=None):
        if ax is None:
            ax = self.ax

        w = grid.info.width
        h = grid.info.height
        res = grid.info.resolution
        ox = grid.info.origin.position[0]
        oy = grid.info.origin.position[1]

        data = np.array(grid.data).reshape((h, w))
        vis = np.where(data < 0, 0, data)

        # world coordinate bounds
        # x_min = ox
        # x_max = ox + w * res
        # y_min = oy
        # y_max = oy + h * res
        x_min = ox
        x_max = ox + w * res
        y_min = oy
        y_max = oy + h * res

        print(x_min, x_max, y_min, y_max)

        # draw onto main axis exactly once
        self.grid_im = self.ax.imshow(
            vis,
            origin="lower",
            cmap="gray_r",
            extent=(x_min, x_max, y_min, y_max),
            interpolation="nearest",
            zorder=0,             # keep it behind everything else
        )

    def init_curve(self, curve, ax=None):
        if ax is None:
            ax = self.ax

        # margin = 2.0

        # p_min = min(np.min(curve.xs), np.min(curve.ys))
        # p_max = max(np.max(curve.xs), np.max(curve.ys))
        #
        # x_min, y_min = p_min, p_min
        # x_max, y_max = p_max, p_max

        # ax.set_xlim(x_min - margin, x_max + margin)
        # ax.set_ylim(y_min - margin, y_max + margin)

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

        # markers for sampled tube points
        (self.upper_tube_pts,) = ax.plot([], [], "ro", markersize=6)
        (self.lower_tube_pts,) = ax.plot([], [], "bo", markersize=6)

    def plot_tubes(self, curve, robot_state, mpc, upper_coeffs, lower_coeffs):

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

        tau = np.linspace(0, len_stop - len_start, 100)
        upper_d = np.polyval(upper_coeffs[::-1], tau)
        lower_d = np.polyval(lower_coeffs[::-1], tau)


        # offset trajectory
        upper = traj + upper_d[:, None] * normals
        lower = traj + lower_d[:, None] * normals

        # plot tubes
        self.upper_tube_line.set_data(upper[:, 0], upper[:, 1])
        self.lower_tube_line.set_data(lower[:, 0], lower[:, 1])

        sample_tau = np.array([0.0, 0.5, 1.0])
        remaining_len = len_stop - len_start
        sample_tau = np.clip(sample_tau, 0.0, remaining_len)

        idx = np.searchsorted(tau, sample_tau)

        du = np.polyval(upper_coeffs[::-1], sample_tau)
        dl = np.polyval(lower_coeffs[::-1], sample_tau)

        upper_pts = traj[idx] + du[:, None] * normals[idx]
        lower_pts = traj[idx] + dl[:, None] * normals[idx]

        self.upper_tube_pts.set_data(upper_pts[:, 0], upper_pts[:, 1])
        self.lower_tube_pts.set_data(lower_pts[:, 0], lower_pts[:, 1])


    def render(self, robot_state, current_ref, curve, mpc, upper_coeffs, lower_coeffs):
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


        self.traj_line.set_data(curve.xs, curve.ys)

        horizon = np.array(mpc.get_horizon())
        self.mpc_horizon.set_data(horizon[:, 1], horizon[:, 2])

        self.plot_tubes(curve, robot_state, mpc, upper_coeffs, lower_coeffs)

        x,y = robot_state[:2]
        half_sz = 4
        self.ax.set_xlim(x-half_sz, x+half_sz)
        self.ax.set_ylim(y-half_sz, y+half_sz)

        # alpha 
        params = mpc.get_params()
        alpha_upper = params["CBF_ALPHA_ABV"]
        alpha_lower = params["CBF_ALPHA_BLW"]

        t = len(self.steps)
        self.steps.append(t)
        self.alpha_upper_hist.append(alpha_upper)
        self.alpha_lower_hist.append(alpha_lower)

        self.alpha_upper_line.set_data(self.steps, self.alpha_upper_hist)
        self.alpha_lower_line.set_data(self.steps, self.alpha_lower_hist)

        self.ax_alphas.relim()
        self.ax_alphas.autoscale_view()

        # cbf
        cbf_abv, cbf_blw = self.get_cbfs(mpc, upper_coeffs, lower_coeffs)
        self.cbf_upper_hist.append(cbf_abv)
        self.cbf_lower_hist.append(cbf_blw)

        self.cbf_upper_line.set_data(self.steps, self.cbf_upper_hist)
        self.cbf_lower_line.set_data(self.steps, self.cbf_lower_hist)

        self.ax_cbfs.relim()
        self.ax_cbfs.autoscale_view()

        plt.pause(0.001)

    def get_cbfs(self, mpc, upper_coeffs, lower_coeffs):

        horizon = np.array(mpc.get_horizon())
        len_start = mpc.get_s_from_pose(horizon[0, 1:3])
        xs, ys = mpc.compute_adjusted_ref(len_start)

        ref_len = mpc.get_params()["REF_LENGTH"]

        cbf_abv = mpc.get_cbf_abv(horizon[0,1:7], upper_coeffs, xs, ys)
        cbf_blw = mpc.get_cbf_blw(horizon[0,1:7], lower_coeffs, xs, ys)

        return float(cbf_abv), float(cbf_blw)

    def add_state_to_path(self, robot_state):
        # if more than 60 points, remove oldest
        if len(self.prev_states) > 100:
            self.prev_states.pop(0)
        self.prev_states.append(list(robot_state))

    def clear_state_path(self):
        self.prev_states.clear()

    def close(self):
        plt.close("all")

