import os
import copy
import numpy as np
import matplotlib

# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from oyster.RobotMPC import Dynamics
from oyster.MapLoader import OccupancyGrid
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FFMpegWriter

from py_planner import MapLayer

from py_mpcc import extend_trajectory


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

        self.save_video = save_video
        self.fps = fps
        self.writer = None

        self.map_util = map_util
        self.sdf_im = None

        if render:
            self.init_plot(curve, occ_grid, upper_coeffs, lower_coeffs, mpc)


    def init_plot(self, curve, occ_grid, upper_coeffs, lower_coeffs, mpc):

        # visualization setup
        self.fig = plt.figure(figsize=(16, 8))
        # fig_manager = plt.get_current_fig_manager()
        # self.fig.canvas.manager.full_screen_toggle()

        max_vel = mpc.get_params()["LINVEL"]
        self.ax = plt.subplot2grid((3, 2), (0, 0), rowspan=3, colspan=1)
        self.ax.set_title(f"{self.dynamics.name}:   max_vel={max_vel} m/s")
        # self.ax.set_aspect("equal", adjustable="box")

        self.ax_alphas = plt.subplot2grid((3, 2), (0, 1))
        self.ax_cbfs = plt.subplot2grid((3, 2), (1, 1))
        self.ax_reward = plt.subplot2grid((3, 2), (2, 1))

        self.ax_reward.set_ylabel("Velocity")
        self.ax_reward.set_xlabel("Time step")
        self.ax_reward.grid(True, linestyle="--", alpha=0.5)

        (self.reward_line,) = self.ax_reward.plot([], [], "k-", label="reward", linewidth=3)
        (self.total_reward_line,) = self.ax_reward.plot([], [], "r-", label="total_reward", linewidth=3)

        self.reward_hist = []
        self.total_reward_hist = []

        (self.alpha_upper_line,) = self.ax_alphas.plot(
            [], [], "r-", label="alpha_upper", linewidth=3
        )
        (self.alpha_lower_line,) = self.ax_alphas.plot(
            [], [], "b-", label="alpha_lower", linewidth=3
        )

        self.ax_alphas.set_ylabel(r"$\alpha$ values")
        self.ax_alphas.set_xlabel("Time step")
        self.ax_alphas.grid(True, linestyle="--", alpha=0.5)

        self.alpha_upper_hist = []
        self.alpha_lower_hist = []
        self.steps = []

        (self.cbf_upper_line,) = self.ax_cbfs.plot(
            [], [], "r-", label="cbf_abv", linewidth=3
        )
        (self.cbf_lower_line,) = self.ax_cbfs.plot(
            [], [], "b-", label="cbf_blw", linewidth=3
        )

        self.ax_cbfs.set_ylabel(r"$h(x)$ values")
        self.ax_cbfs.set_xlabel("Time step")
        self.ax_cbfs.grid(True, linestyle="--", alpha=0.5)

        self.cbf_upper_hist = []
        self.cbf_lower_hist = []

        if self.dynamics == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_patch = Circle(
                (0, 0), self.robot_size, color="blue", label="robot", zorder=0
            )
            # self.robot_patch = Polygon(
            #     self.robot_footprint,
            #     closed=True,
            #     color="blue",
            #     label="robot",
            # )
        else:
            self.robot_patch = Polygon(
                self.robot_footprint,
                closed=True,
                color="blue",
                label="robot",
            )

        self.init_curve(curve)
        self.init_sdf_layer()

        (self.ref_point,) = self.ax.plot([], [], "ro", label="reference")

        (self.path_line,) = self.ax.plot([], [], "k-", linewidth=1.5, alpha=0.5)

        (self.mpc_horizon,) = self.ax.plot([], [], "g", linewidth=3.0, alpha=0.5)

        (self.mpc_trajectory_belief,) = self.ax.plot([], [], "r-", linewidth=1.5, alpha=0.5)

        self.init_grid(occ_grid)

        self.ax.add_patch(self.robot_patch)

        self.init_tubes(curve, upper_coeffs, lower_coeffs)

        if self.save_video:
            self.writer = FFMpegWriter(
                fps=self.fps,
                bitrate=1800,
            )

            script_dir = Path(__file__).parent.absolute()
            video_folder = os.path.join(script_dir, "videos")
            if not os.path.exists(video_folder):
                os.mkdir(video_folder)

            now = datetime.now()
            t_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            v_max = round(mpc.get_params()["LINVEL"], 2)
            v_max_str = str(v_max).replace('.', '_')
            dyn_str = str(self.dynamics).split('.')[-1]
            fname = f"{dyn_str}_{v_max_str}_{t_str}.mp4"
            self.writer.setup(self.fig, os.path.join(video_folder, fname), dpi=100)

    def init_sdf_layer(self):
        # placeholder 2D array (will be updated in render)
        sdf = np.zeros((50, 50))

        self.sdf_im = self.ax.imshow(
            sdf,
            origin="lower",
            cmap="coolwarm",
            alpha=0.6,
            zorder=1,     # above occupancy, below robot
            interpolation="bilinear"
        )

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
            [], [], "r-", label="upper tube", linewidth=2.5, alpha=0.5
        )
        (self.lower_tube_line,) = ax.plot(
            [], [], "b-", label="lower tube", linewidth=2.5, alpha=0.5
        )

        (self.upper_tube_belief,) = ax.plot(
            [], [], "r--", label="upper tube", linewidth=2, alpha=0.5,
        )
        (self.lower_tube_belief,) = ax.plot(
            [], [], "b--", label="lower tube", linewidth=2, alpha=0.5,
        )

        # markers for sampled tube points
        (self.upper_tube_pts,) = ax.plot([], [], "ro", markersize=6)
        (self.lower_tube_pts,) = ax.plot([], [], "bo", markersize=6)

    def log_reward(self, reward):
        self.reward_hist.append(reward)

    def plot_tubes(self, curve, robot_state, mpc, upper_coeffs, lower_coeffs):

        trajectory = mpc.get_trajectory()
        # trajectory = extend_trajectory(mpc.get_trajectory(), mpc.get_params()["REF_LENGTH"])
        # trajectory = extend_trajectory(mpc.get_trajectory(), mpc.get_trajectory().get_arclen() + 2)
        len_start = trajectory.get_closest_s(robot_state[:2])

        adjusted_traj = trajectory.get_adjusted_traj(len_start, int(mpc.get_params()["REF_SAMPLES"]))
        traj_view = adjusted_traj.view()
        xs = traj_view.xs
        ys = traj_view.ys
        state = mpc.get_state_from_horizon(0)
        args = {"i0": state, "i1": mpc.get_input_from_horizon(0), "i2": xs, "i3": ys, "i4": upper_coeffs, "i5": lower_coeffs, "i6": mpc.get_params()["CLF_W_LAG"], "i7": mpc.get_params()["CLF_W_CONTOUR"], "i8": mpc.get_params()["CLF_GAMMA"], "i9": traj_view.arclen}

        # horizon = mpc.get_params()["REF_LENGTH"]
        # if len_start + horizon > trajectory.get_arclen():
        #     horizon = trajectory.get_arclen() - len_start
        extended_len = adjusted_traj.get_arclen()

        trajectory = mpc.get_non_extended_trajectory()
        adjusted_len_start = min(len_start, trajectory.get_arclen())
        non_ex_adj = trajectory.get_adjusted_traj(adjusted_len_start, int(mpc.get_params()["REF_SAMPLES"]))
        # horizon = non_ex_adj.get_arclen()
        horizon = extended_len


        tau = np.linspace(0, horizon, 100)
        upper_d = np.zeros((100,))
        lower_d = np.zeros((100,))

        for i in range(100):
            state[4] = tau[i]
            upper_d[i] = mpc.debug_fns["d_abv"](**args)
            lower_d[i] = mpc.debug_fns["d_blw"](**args)

        upper_d_actual = np.polyval(upper_coeffs[::-1], tau / extended_len)
        lower_d_actual = np.polyval(lower_coeffs[::-1], tau / extended_len)

        ss = np.linspace(0, horizon, 100)
        # traj = np.vstack([curve.trajx(ss), curve.trajy(ss)]).T
        traj = np.vstack([adjusted_traj(s) for s in ss])

        # tangents = np.column_stack([curve.trajx_d(ss), curve.trajy_d(ss)])
        tangents = np.vstack([adjusted_traj(s, 1) for s in ss])
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        # compute normals
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        # offset trajectory
        upper = traj + upper_d[:, None] * normals
        lower = traj + lower_d[:, None] * normals

        upper_actual = traj + upper_d_actual[:, None] * normals
        lower_actual = traj + lower_d_actual[:, None] * normals

        # plot tubes
        self.upper_tube_line.set_data(upper[:, 0], upper[:, 1])
        self.lower_tube_line.set_data(lower[:, 0], lower[:, 1])

        self.upper_tube_belief.set_data(upper_actual[:, 0], upper_actual[:, 1])
        self.lower_tube_belief.set_data(lower_actual[:, 0], lower_actual[:, 1])

        # sample_tau = np.array([0.0, 0.5, 1.0])
        # remaining_len = len_stop - len_start
        # sample_tau = np.clip(sample_tau, 0.0, remaining_len)
        #
        # idx = np.searchsorted(tau, sample_tau)
        #
        # du = np.polyval(upper_coeffs[::-1], sample_tau / trajectory.get_arclen())
        # dl = np.polyval(lower_coeffs[::-1], sample_tau / trajectory.get_arclen())
        #
        # upper_pts = traj[idx] + du[:, None] * normals[idx]
        # lower_pts = traj[idx] + dl[:, None] * normals[idx]
        #
        # self.upper_tube_pts.set_data(upper_pts[:, 0], upper_pts[:, 1])
        # self.lower_tube_pts.set_data(lower_pts[:, 0], lower_pts[:, 1])

    def plot_vector(self, start, vec):
        end = np.array(start) + np.array(vec)
        print(start)
        self.ax.plot([start[0], end[0]], [start[1], end[1]])

    def render(self, robot_state, current_ref, curve, mpc, upper_coeffs, lower_coeffs):
        if self.dynamics == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_patch.center = (robot_state[0], robot_state[1])
            # theta = np.atan2(robot_state[3], robot_state[2])
            # R = np.array(
            #     [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            # )
            # footprint = np.dot(self.robot_footprint, R.T)
            # print((footprint + robot_state[:2]).reshape(-1, 2))
            # self.robot_patch.set_xy((footprint + robot_state[:2]).reshape(-1, 2))
        else:
            theta = robot_state[2]
            R = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            footprint = np.dot(self.robot_footprint, R.T)
            # print((footprint + robot_state[:2]).reshape(-1, 2))
            self.robot_patch.set_xy((footprint + robot_state[:2]).reshape(-1, 2))

        # self.ref_point.set_data([current_ref[0]], [current_ref[1]])

        if len(self.prev_states) > 1:
            path = np.array(self.prev_states)
            # if len(self.prev_states) == 100:
            #     print(path)
            #     exit(0)
            self.path_line.set_data(path[:, 0], path[:, 1])


        horizon = mpc.get_horizon()
        horiz_len = horizon.length
        horizon = np.array([mpc.get_state_from_horizon(i) for i in range(horiz_len)])
        self.mpc_horizon.set_data(horizon[:, 0], horizon[:, 1])

        #################### TRAJ REFERENCE FROM MPC PERSPECTIVE ####################
        state = mpc.get_state_from_horizon(0)
        u = mpc.get_input_from_horizon(0)

        trajectory = mpc.get_trajectory()
        # trajectory = extend_trajectory(mpc.get_trajectory(), mpc.get_params()["REF_LENGTH"])
        # trajectory = extend_trajectory(mpc.get_trajectory(), mpc.get_trajectory().get_arclen() + 2)
        len_start = trajectory.get_closest_s(state[:2])
        adjusted_traj = trajectory.get_adjusted_traj(len_start, int(mpc.get_params()["REF_SAMPLES"]))
        traj_view = adjusted_traj.view()
        xs = traj_view.xs
        ys = traj_view.ys
        args = {"i0": state, "i1": u, "i2": xs, "i3": ys, "i4": upper_coeffs, "i5": lower_coeffs, "i6": mpc.get_params()["CLF_W_LAG"], "i7": mpc.get_params()["CLF_W_CONTOUR"], "i8": mpc.get_params()["CLF_GAMMA"], "i9": traj_view.arclen}

        pts = np.zeros((30, 2))
        count = 0
        for i in np.linspace(0, adjusted_traj.get_arclen(), 30):
            state[4] = i
            pts[count,:] = [mpc.debug_fns["xr"](**args), mpc.debug_fns["yr"](**args)]
            count += 1

        # non_extended_traj = mpc.get_non_extended_trajectory()
        # non_ex_adj = non_extended_traj.get_adjusted_traj(len_start, int(mpc.get_params()["REF_SAMPLES"]))
        # print("true ARCLEN:", non_ex_adj.get_arclen())
        tau = np.linspace(0, adjusted_traj.get_arclen(), 30)
        xs = [adjusted_traj(s)[0] for s in tau]
        ys = [adjusted_traj(s)[1] for s in tau]
        self.traj_line.set_data(xs, ys)

        non_extended_traj = mpc.get_non_extended_trajectory()
        tau = np.linspace(0, non_extended_traj.get_arclen(), 30)
        xs = [non_extended_traj(s)[0] for s in tau]
        ys = [non_extended_traj(s)[1] for s in tau]
        self.mpc_trajectory_belief.set_data(xs, ys)

        self.plot_tubes(curve, robot_state, mpc, upper_coeffs, lower_coeffs)

        x,y = robot_state[:2]
        half_sz = 2
        # half_sz = 4
        self.ax.set_xlim(x-half_sz, x+half_sz)
        self.ax.set_ylim(y-half_sz, y+half_sz)

        # reward 
        if len(self.reward_hist) > 0:
            reward_tot = np.sum(self.reward_hist)
            self.total_reward_hist.append(reward_tot)
            self.reward_line.set_data(self.steps, self.reward_hist)
            # self.total_reward_line.set_data(self.steps, self.total_reward_hist)

            self.ax_reward.relim()
            self.ax_reward.autoscale_view()

        t = len(self.steps)
        self.steps.append(t)

        # alpha 
        params = mpc.get_params()
        alpha_upper = params["CBF_ALPHA_ABV"]
        alpha_lower = params["CBF_ALPHA_BLW"]

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

        # sdf
        N = 60  # grid resolution for heatmap
        xs = np.linspace(x - half_sz, x + half_sz, N)
        ys = np.linspace(y - half_sz, y + half_sz, N)

        X, Y = np.meshgrid(xs, ys)
        sdf_vals = np.zeros_like(X)


        vec_sdf = np.vectorize(self.map_util.sdf_dist)
        sdf_vals = vec_sdf(X, Y, MapLayer.kInflated)
        # for i in range(N):
        #     for j in range(N):
        #         sdf_vals[i, j] = self.map_util.sdf_dist(X[i, j], Y[i, j], MapLayer.kInflated)

        self.sdf_im.set_data(sdf_vals)
        self.sdf_im.set_extent((x - half_sz, x + half_sz,
                                y - half_sz, y + half_sz))

        # optional: dynamic color scaling
        self.sdf_im.set_clim(vmin=0, vmax=0.5)

        plt.pause(0.001)

        if self.save_video:
            self.writer.grab_frame()

    def get_cbfs(self, mpc, upper_coeffs, lower_coeffs):

        state = mpc.get_state_from_horizon(0)
        u = mpc.get_input_from_horizon(0)

        trajectory = mpc.get_trajectory()
        # trajectory = extend_trajectory(mpc.get_trajectory(), mpc.get_trajectory().get_arclen() + 2)
        len_start = trajectory.get_closest_s(state[:2])
        adjusted_traj = trajectory.get_adjusted_traj(len_start, int(mpc.get_params()["REF_SAMPLES"])).view()
        xs, ys = adjusted_traj.xs, adjusted_traj.ys

        ref_len = mpc.get_params()["REF_LENGTH"]

        Q_l = mpc.get_params()["CLF_W_LAG"]
        Q_c = mpc.get_params()["CLF_W_CONTOUR"]
        gamma = mpc.get_params()["CLF_GAMMA"]

        args = {"i0": state, "i1": u, "i2": xs, "i3": ys, "i4": upper_coeffs, "i5": lower_coeffs, "i6": Q_l, "i7": Q_c, "i8": gamma, "i9": trajectory.get_arclen()}
        cbf_abv = mpc.debug_fns["h_abv"](**args)
        cbf_blw = mpc.debug_fns["h_blw"](**args)

        return float(cbf_abv), float(cbf_blw)

    def add_state_to_path(self, robot_state):
        # if more than 60 points, remove oldest
        if len(self.prev_states) > 100:
            self.prev_states.pop(0)
        self.prev_states.append(list(robot_state))

    def clear_state_path(self):
        self.prev_states.clear()

    def close(self):
        if self.save_video and self.writer is not None:
            self.writer.finish()

        plt.close("all")

