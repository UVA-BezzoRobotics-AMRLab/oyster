import os
import csv
import gymnasium as gym
import time
import copy
import click
import numpy as np

from enum import IntEnum, auto
from gymnasium import spaces
from oyster.BarnPlotter import BarnPlotter
from oyster.Planner import PyPlanner
from oyster.Bezier import BezierCurve
from oyster.ParamLoader import ParameterLoader
from oyster.RobotMPC import RobotMPC, Dynamics
from oyster.MapLoader import parse_xml_file, generate_map_from_cylinders

from py_planner import Planner
from py_planner import RFNState
from py_planner import vec_Vec2d, vec_MatX4d
from py_planner import PlannerStatus
from py_planner import PlannerParams
from py_planner import OccupancyGrid

from py_mpcc import Polynomial
from py_mpcc import extend_trajectory


class RLObs(IntEnum):
    CBF_ABV = 0
    LFH_LGH_ABV = auto()
    CBF_BLW = auto()
    LFH_LGH_BLW = auto()


def action_unnormalize(val, min, max):
    return (val + 1.0) * (max - min) / 2.0 + min


class CBFEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        world_num=[0],
        N=3,
        manual_step=False,
        normalize_obs=True,
        save_video=False,
        max_step_count=250,
    ):

        super(CBFEnv, self).__init__()

        self.world_nums = world_num
        self.max_step_count = max_step_count
        self.save_video = save_video
        self.should_normalize = normalize_obs
        self.manual_step = manual_step
        self.N_alpha = 2
        self.N_horizon = N

        self.state_dim = len(RLObs) * self.N_horizon + self.N_alpha
        self.action_dim = 2

        # each task is loaded from parameter list
        self.task_loader = self.load_tasks()
        self.traj_schedule = [15, 5]

        self.did_collide = False

        self.epoch = 0
        self.epoch_incremented = False
        # 282 is a really nice world!!!
        # large alpha trouble from 140 to 285
        # mid alpha trouble from 255 to  285
        # include some other worlds for training as well...
        self.worlds = [280, 290, 140, 245, 285, 255, 265, 275, 5, 282, 296, 111]

        self.planner = PyPlanner()
        self.step_count = 0

        self.total_reward = 0
        self.traj_planner_success = False

        self.planner_params = PlannerParams()
        # self.planner_params.SOLVER = "faster"
        self.planner_params.SOLVER = "gcopter"
        self.planner_params.W_MAX = 1.8
        self.planner_params.V_MAX = 50
        self.planner_params.A_MAX = 60
        self.planner_params.J_MAX = 150
        self.planner_params.DT_FACTOR_INIT = 1.0
        self.planner_params.DT_FACTOR_FINAL = 10.0
        self.planner_params.DT_FACTOR_INCREMENT = 1.0
        self.planner_params.SOLVER_TRAJ_DT = 0.05
        self.planner_params.TRIM_DIST = -100

        self.planner_params.N_SEGMENTS = 6
        self.planner_params.MAX_POLYS = 3
        self.planner_params.N_THREADS = 0
        self.planner_params.FORCE_FINAL_CONSTRAINT = True
        self.planner_params.VERBOSE = False
        self.planner_params.USE_MINVO = False
        self.planner_params.PLAN_IN_FREE = False
        self.planner_params.SIMPLIFY_JPS = True
        self.planner_params.MAX_SOLVE_TIME = 0.25

        self.max_plan_horizon = 3.0
        self.curr_horizon = 3.0

        self.planner.set_params(self.planner_params)
        self.set_world(self.world_nums)

        if len(self.task_loader) == 0:
            raise ValueError("No parameter files found!")

        self.task_idx = -1

        # i dont think these bounds every get used by the SAC algorithm
        self.low = np.array(
            -10 * np.ones(self.state_dim),
            dtype=np.float64,
        )

        self.high = np.array(
            10 * np.ones(self.state_dim),
            dtype=np.float64,
        )

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-np.ones(self.action_dim),
            high=np.ones(self.action_dim),
            dtype=np.float64,
        )

        # for some reason _goal is needed for PEARL, but it can be
        # set to anything
        self._start = None
        self._goal = None

        self.plotter = None
        self.reset()

    def set_world(self, world_num):
        try:
            obstacles = []
            # offsets = [-5, 3]
            offsets = [-5 + 8 * i for i in range(len(world_num))]
            print(world_num)
            for i, world in enumerate(world_num):
                print(i)
                path = os.path.join(
                    os.getenv("BARN_DATASET_PATH"),
                    "world_files",
                    f"world_{world}.world",
                )
                obs = parse_xml_file(path, offsets[i])
                if len(obstacles) == 0:
                    obstacles = obs
                else:
                    obstacles = np.concatenate((obstacles, obs))

        except Exception as e:
            obstacles = None
            print(
                "[ERROR] Loading obstacles did not work, has BARN dataset been installed and the BARN_DATASET_PATH",
                "environment variable been set to it's top level directory location?",
            )
            print(str(e))
            exit(0)

        self.occupancy_grid = generate_map_from_cylinders(obstacles, 0, 0, 0)
        self.map_util = OccupancyGrid(
            self.occupancy_grid.info.width,
            self.occupancy_grid.info.height,
            self.occupancy_grid.info.resolution,
            float(self.occupancy_grid.info.origin.position[0]),
            float(self.occupancy_grid.info.origin.position[1]),
            np.array(self.occupancy_grid.data),
            np.array([253, 254]),
            np.array([255]),
        )

    @staticmethod
    def get_mu_and_std(N_horizon, N_alpha, params):
        # These values were computed empirically over 100k samples
        obs_mu_single = np.zeros(len(RLObs))
        obs_mu_single[RLObs.CBF_ABV] = 0.2953
        obs_mu_single[RLObs.LFH_LGH_ABV] = -0.0163
        obs_mu_single[RLObs.CBF_BLW] = 0.3125
        obs_mu_single[RLObs.LFH_LGH_BLW] = -0.0195
        # obs_mu[RLObs.CBF_ABV] = 0.423
        # obs_mu[RLObs.LFH_LGH_ABV] = 1.0155
        # obs_mu[RLObs.CBF_BLW] = 0.1753
        # obs_mu[RLObs.LFH_LGH_BLW] = -0.17

        obs_std_single = np.zeros(len(RLObs))
        obs_std_single[RLObs.CBF_ABV] = 0.1011
        obs_std_single[RLObs.LFH_LGH_ABV] = 0.1497
        obs_std_single[RLObs.CBF_BLW] = 0.105
        obs_std_single[RLObs.LFH_LGH_BLW] = 0.1687
        # obs_std[RLObs.CBF_ABV] = 0.2371
        # obs_std[RLObs.LFH_LGH_ABV] = 1.1442
        # obs_std[RLObs.CBF_BLW] = 0.1505
        # obs_std[RLObs.LFH_LGH_BLW] = 0.2709

        obs_mu = np.zeros(obs_mu_single.shape[0] * N_horizon + N_alpha)
        obs_std = np.zeros(obs_std_single.shape[0] * N_horizon + N_alpha)

        obs_mu[:-N_alpha] = np.tile(obs_mu_single, N_horizon)
        obs_std[:-N_alpha] = np.tile(obs_std_single, N_horizon)

        for i in range(N_alpha):
            obs_mu[len(RLObs) * N_horizon + i] = (
                params["MIN_ALPHA"] + params["MAX_ALPHA"]
            ) / 2

        for i in range(N_alpha):
            obs_std[len(RLObs) * N_horizon + i] = (
                params["MIN_ALPHA"] + params["MAX_ALPHA"]
            ) / 4

        return obs_mu, obs_std

    def set_mpc(self, params):

        self.params = copy.deepcopy(params)
        self.dynamic_model = self.params["DYNAMIC_MODEL"]
        # self.params["USE_CBF"] = True

        init_theta = np.atan2(
            self._goal[1, 0] - self._start[1, 0], self._goal[0, 0] - self._start[0, 0]
        )

        init_pose = np.concatenate((self._start[:2, 0], [init_theta]))
        self.mpc = RobotMPC(init_pose, self.params)
        self.mpc.set_occ_map(self.occupancy_grid)
        self.robot_state = self.mpc.get_robot_state()

        self.upper_coeffs = np.array([0] * (self.params["TUBE_DEGREE"] + 1))
        self.upper_coeffs[0] = self.params["MAX_TUBE_WIDTH"] / 2.0
        self.lower_coeffs = np.array([0] * (self.params["TUBE_DEGREE"] + 1))
        self.lower_coeffs[0] = -self.params["MAX_TUBE_WIDTH"] / 2.0

        if not self.params["USE_CBF"]:
            self.upper_coeffs[0] = 100
            self.lower_coeffs[0] = -100

        self.obs_mu, self.obs_std = CBFEnv.get_mu_and_std(
            self.N_horizon, self.N_alpha, self.params
        )

    def load_tasks(self):
        param_path = os.path.join(os.path.dirname(__file__), "configs")
        fnames = []
        for file in os.listdir(param_path):
            if file.endswith(".yaml"):
                fnames.append(os.path.join(param_path, file))

        return ParameterLoader(fnames)

    def normalize_obs(self, obs):
        # 95% of observed data will be within -1 to 1
        z = (obs - self.obs_mu) / (2 * self.obs_std)
        z = np.clip(z, self.low, self.high)
        return z

    def step(self, action):
        self.step_count += 1

        self.update_trajectory()
        trajectory = self.mpc.get_trajectory()

        self.params = self.mpc.get_params()

        len_start = trajectory.get_closest_s(self.robot_state[:2])
        print("len start: ", len_start)
        print("knots[-1]: ", self.knots[-1])
        # self.get_and_set_tubes(len_start)

        action = np.array(action)
        action = action_unnormalize(
            action, self.params["MIN_ALPHA_DOT"], self.params["MAX_ALPHA_DOT"]
        )

        dt = self.params["DT"]
        alpha_abv = self.params["CBF_ALPHA_ABV"] + action[0] * dt
        alpha_blw = self.params["CBF_ALPHA_BLW"] + action[1] * dt

        self.params["CBF_ALPHA_ABV"] = alpha_abv
        self.params["CBF_ALPHA_BLW"] = alpha_blw

        # if self.params["USE_CBF"]:
        #     self.mpc.load_params(self.params)

        u = self.mpc.get_control(len_start)
        # print("ROBOT_STATE before:", self.robot_state)
        self.robot_state = self.mpc.apply_control(u)
        tube = self.mpc.get_tube()
        self.upper_coeffs = tube[0].get_coeffs()
        self.lower_coeffs = tube[1].get_coeffs()

        # print(self.upper_coeffs)
        # print(self.lower_coeffs)
        # print("ROBOT_STATE after:", self.robot_state)

        self.current_ref = trajectory(len_start)

        obs = self.get_obs()
        print("obs:", obs)
        # print("obs:", self.normalize_obs(obs))

        v_max = self.params["LINVEL"]

        is_colliding = self.map_util.is_occupied(
            self.robot_state[0], self.robot_state[1], "inflated"
        )

        self.did_collide = is_colliding

        # ---------------- reward ----------------
        reward = CBFEnv.get_reward(obs, is_colliding, self.params, self.N_horizon)

        is_done = (
            self.step_count >= self.max_step_count
            or (len_start >= self.knots[-1] - 0.2)
            or is_colliding
        )

        self.total_reward += reward
        # print("step reward:", reward)
        # print("total reward:", self.total_reward)
        if self.plotter is not None:
            # self.plotter.log_reward(reward)
            vel = self.robot_state[3]
            if self.dynamic_model == Dynamics.DOUBLE_INTEGRATOR:
                vel = np.linalg.norm(self.robot_state[2:4])
            self.plotter.log_reward(vel)

        return (
            self.normalize_obs(obs) if self.should_normalize else obs,
            reward,
            is_done,
            {},
        )

    def reset(self):
        self.did_collide = False
        self.total_reward = 0
        self.traj_planner_success = False
        self.step_count = 0

        if self.plotter:
            self.plotter.close()

        params = copy.deepcopy(self.task_loader[self.task_idx])
        min_alpha = params["MIN_ALPHA"]
        max_alpha = params["MAX_ALPHA"]

        self._goal = np.zeros((3, 4))
        self._goal[:2, 0] = [-2.25, -2.5]
        self._start = np.zeros((3, 4))
        self._start[:2, 0] = [-2.25, -2.5 + (10 * len(self.world_nums))]

        tmp = self._goal
        self._goal = self._start
        self._start = tmp

        if self.epoch >= self.traj_schedule[0]:
            params["CBF_ALPHA_ABV"] = np.random.uniform(min_alpha, max_alpha)
            params["CBF_ALPHA_BLW"] = np.random.uniform(min_alpha, max_alpha)

            # randomly swap the two goals...
            r = np.random.randint(2)
            if r == 1:
                tmp = self._goal
                self._goal = self._start
                self._start = tmp

        delta = self.epoch - self.traj_schedule[0]
        if delta >= 0 and delta % self.traj_schedule[1] == 0:
            world_num = self.worlds[delta // self.traj_schedule[1]]
            self.set_world([world_num])

        # params["USE_CBF"] = False
        params["CBF_ALPHA_ABV"] = 2.0
        params["CBF_ALPHA_BLW"] = 2.0

        self.set_mpc(params)
        print("params:", params)

        self.planner.has_trajectory = False
        self.plotter = None

        # self.robot_state = np.zeros(4, dtype=np.float64)
        # self.mpc.set_mpc_state(self.robot_state)

        obs = np.zeros(self.state_dim, dtype=np.float64)
        obs[-self.N_alpha :] = [params["CBF_ALPHA_ABV"], params["CBF_ALPHA_BLW"]]
        return self.normalize_obs(obs) if self.should_normalize else obs

    def get_obs(self):
        horizon = self.mpc.get_horizon()
        horizon_len = horizon.length
        if horizon_len < self.N_horizon:
            raise ValueError(
                "Horizon shape",
                horizon.shape[0],
                "is smaller than N_horizon set",
                self.N_horizon,
            )

        trajectory = self.mpc.get_trajectory()
        # trajectory = extend_trajectory(self.mpc.get_trajectory(), self.mpc.get_params()["REF_LENGTH"])
        # trajectory = extend_trajectory(self.mpc.get_trajectory(), self.mpc.get_trajectory().get_arclen() + 2)

        state = self.mpc.get_state_from_horizon(1)
        len_start = max(trajectory.get_closest_s(state[:2]), 1e-6)
        adjusted_traj = trajectory.get_adjusted_traj(
            len_start, int(self.mpc.get_params()["REF_SAMPLES"])
        )
        xs = adjusted_traj.get_ctrls_x()
        ys = adjusted_traj.get_ctrls_y()

        # go to length-2 because N-1 inputs in horizon of size N
        inds = np.linspace(1, horizon_len - 2, self.N_horizon, dtype=int)
        obs = np.zeros(len(RLObs) * self.N_horizon + self.N_alpha)
        for i in range(self.N_horizon):
            state = self.mpc.get_state_from_horizon(inds[i])
            # some numerical issue is causing s state to be -1e-<large num> for some reason
            # Casadi doesnt like that so enforcing a strict positive minimum.
            state[-2] = max(state[-2], 1e-6)
            u = self.mpc.get_input_from_horizon(inds[i])
            acc = u[:2]

            args = {
                "i0": state,
                "i1": u,
                "i2": xs,
                "i3": ys,
                "i4": self.upper_coeffs,
                "i5": self.lower_coeffs,
                "i6": self.params["CLF_W_LAG"],
                "i7": self.params["CLF_W_CONTOUR"],
                "i8": self.params["CLF_GAMMA"],
                "i9": adjusted_traj.get_arclen(),
            }

            # cbf_abv = self.mpc.get_cbf_abv(state, self.upper_coeffs, xs, ys)
            # lfh_abv = self.mpc.get_lfh_abv(state, self.upper_coeffs, xs, ys)
            # lgh_abv = self.mpc.get_lgh_abv(state, self.upper_coeffs, xs, ys)
            # lghu_abv = lgh_abv[:2] @ acc

            cbf_abv = self.mpc.debug_fns["h_abv"](**args)
            lfh_abv = self.mpc.debug_fns["Lfh_abv"](**args)
            lghu_abv = self.mpc.debug_fns["Lghu_abv"](**args)

            # cbf_blw = self.mpc.get_cbf_blw(state, self.lower_coeffs, xs, ys)
            # lfh_blw = self.mpc.get_lfh_blw(state, self.lower_coeffs, xs, ys)
            # lgh_blw = self.mpc.get_lgh_blw(state, self.lower_coeffs, xs, ys)
            # lghu_blw = lgh_blw[:2] @ acc

            cbf_blw = self.mpc.debug_fns["h_blw"](**args)
            lfh_blw = self.mpc.debug_fns["Lfh_blw"](**args)
            lghu_blw = self.mpc.debug_fns["Lghu_blw"](**args)

            obs[i * len(RLObs) + RLObs.CBF_ABV] = float(cbf_abv)
            obs[i * len(RLObs) + RLObs.LFH_LGH_ABV] = float(lfh_abv + lghu_abv)
            obs[i * len(RLObs) + RLObs.CBF_BLW] = float(cbf_blw)
            obs[i * len(RLObs) + RLObs.LFH_LGH_BLW] = float(lfh_blw + lghu_blw)

            if np.any(np.isnan(obs)):
                # # args = {"i0": state, "i1": xs}
                print(i, "state:", state)
                signed_d = self.mpc.debug_fns("signed_d",args)
                print(i, "signed_d", signed_d)
                print(i, "xr:", self.mpc.debug_fns["xr"](**args))
                print(i, "yr:", self.mpc.debug_fns["yr"](**args))
                print(i, "xr_dot:", self.mpc.debug_fns["xr_dot"](**args))
                print(i, "yr_dot:", self.mpc.debug_fns["yr_dot"](**args))
                print(i, "phi_r:", self.mpc.debug_fns["phi_r"](**args))
                print(i, "theta:", np.atan2(state[3], state[2]))
                print(i, "p_abv", self.mpc.debug_fns["p_abv"](**args))
                print(i, "p_blw", self.mpc.debug_fns["p_blw"](**args))
                print(i, "h_abv:", self.mpc.debug_fns["h_abv"](**args))
                print(i, "h_blw:", self.mpc.debug_fns["h_blw"](**args))

                # print("e_c:", self.mpc.debug_fns["e_c"](**args))
                # print("e_l:", self.mpc.debug_fns["e_l"](**args))
                # h_abv = self.mpc.debug_fns["h_abv"](**args)
                # lfh_abv = self.mpc.debug_fns["Lfh_abv"](**args)
                # lghu_abv = self.mpc.debug_fns["Lghu_abv"](**args)
                # h_blw= self.mpc.debug_fns["h_blw"](**args)
                # lfh_blw= self.mpc.debug_fns["Lfh_blw"](**args)
                # lghu_blw= self.mpc.debug_fns["Lghu_blw"](**args)

                # print("signed d", self.mpc.debug_fns["signed_d"](**args))
                phi_r = self.mpc.debug_fns["phi_r"](**args)
                #
                # print("clf_dot", self.mpc.debug_fns["lyap_dot"](**args))
                # print("Lfv", self.mpc.debug_fns["Lfv"](**args))
                # # print("Lgv", self.mpc.debug_fns["Lgv"](**args))
                # print("Lgvu", self.mpc.debug_fns["Lgvu"](**args))
                # print("clf_const", self.mpc.debug_fns["lyap_const"](**args))
                # #
                # print("lfh_abv:", self.mpc.debug_fns["Lfh_abv"](**args))
                # print("lfh_blw:", self.mpc.debug_fns["Lfh_blw"](**args))
                # print("lghu_abv:", self.mpc.debug_fns["Lghu_abv"](**args))
                # print("lghu_blw:", self.mpc.debug_fns["Lghu_blw"](**args))
                #
                # print("const_abv", lfh_abv + lghu_abv + self.params["CBF_ALPHA_ABV"] * h_abv)
                # print("const_blw", lfh_blw + lghu_blw + self.params["CBF_ALPHA_BLW"] * h_blw)

        obs[-self.N_alpha :] = [
            self.params["CBF_ALPHA_ABV"],
            self.params["CBF_ALPHA_BLW"],
        ]
        # p_blw = self.mpc.debug_fns["p_blw"](**args)
        # d_blw = self.mpc.debug_fns["d_blw"](**args)
        # h_blw = self.mpc.debug_fns["h_blw"](**args)

        # signed_d = self.mpc.get_signed_d(state, xs, ys)
        # d_blw = self.mpc.get_d_blw(state, self.lower_coeffs)
        # h_blw = (signed_d - d_blw) * np.exp(-p_blw)

        # tmp_p = trajectory(len_start)
        # adj_tmp_p = np.array([np.polyval(xs[::-1], 0), np.polyval(ys[::-1], 0)])
        # adj_man_sd = np.linalg.norm(adj_tmp_p-state[:2])
        # man_signed_d =  np.linalg.norm(tmp_p-state[:2])
        # man_d_blw = np.polyval(self.lower_coeffs[::-1], state[-2])

        # print("inputs:")
        # for i in range(horizon.length-1):
        #     print(horizon.get_input_at_step(i))

        state = self.mpc.get_state_from_horizon(0)
        len_start = trajectory.get_closest_s(state[:2])
        adjusted_traj = trajectory.get_adjusted_traj(
            len_start, int(self.mpc.get_params()["REF_SAMPLES"])
        )
        xs = adjusted_traj.get_ctrls_x()
        ys = adjusted_traj.get_ctrls_y()

        args = {
            "i0": state,
            "i1": u,
            "i2": xs,
            "i3": ys,
            "i4": self.upper_coeffs,
            "i5": self.lower_coeffs,
            "i6": self.params["CLF_W_LAG"],
            "i7": self.params["CLF_W_CONTOUR"],
            "i8": self.params["CLF_GAMMA"],
            "i9": adjusted_traj.get_arclen(),
        }

        # print("ADJUSTED LENGTH: ", adjusted_traj.get_arclen())
        # print("state:", state)
        # signed_d = self.mpc.debug_fns["signed_d"](**args)
        # print("signed_d", signed_d)
        # print("xr:", self.mpc.debug_fns["xr"](**args))
        # print("yr:", self.mpc.debug_fns["yr"](**args))
        # print("xr_dot:", self.mpc.debug_fns["xr_dot"](**args))
        # print("yr_dot:", self.mpc.debug_fns["yr_dot"](**args))
        # print("phi_r:", self.mpc.debug_fns["phi_r"](**args))
        # print("theta:", np.atan2(state[3], state[2]))
        # print("p_abv", self.mpc.debug_fns["p_abv"](**args))
        # print("p_blw", self.mpc.debug_fns["p_blw"](**args))
        # print("h_abv:", self.mpc.debug_fns["h_abv"](**args))
        # print("lfh_abv:", self.mpc.debug_fns["Lfh_abv"](**args))
        # print("lghu_abv:", self.mpc.debug_fns["Lghu_abv"](**args))
        # print("alpha_abv:", self.params["CBF_ALPHA_ABV"])
        # print("h_blw:", self.mpc.debug_fns["h_blw"](**args))
        # print("lfh_blw:", self.mpc.debug_fns["Lfh_blw"](**args))
        # print("lghu_blw:", self.mpc.debug_fns["Lghu_blw"](**args))
        # print("alpha_blw:", self.params["CBF_ALPHA_BLW"])
        # print("const_abv", self.mpc.debug_fns["Lfh_abv"](**args) + self.mpc.debug_fns["Lghu_abv"](**args) + self.params["CBF_ALPHA_ABV"]* self.mpc.debug_fns["h_abv"](**args))
        # print("const_blw", self.mpc.debug_fns["Lfh_blw"](**args) + self.mpc.debug_fns["Lghu_blw"](**args) + self.params["CBF_ALPHA_BLW"] * self.mpc.debug_fns["h_blw"](**args))

        # lfv = self.mpc.debug_fns["Lfv"](**args) 
        # lgv = self.mpc.debug_fns["Lgv"](**args)
        # lgvu = self.mpc.debug_fns["Lgvu"](**args)
        # lyap_con = self.mpc.debug_fns["lyap_const"](**args) 
        # print("vdot: ", lfv + lgvu)
        # print("lgv:", lgv)
        # print("lgvu:", lgvu)
        # print("lyap_cons", lyap_con)
        return obs

    def set_epoch(self, epoch):
        self.epoch_incremented = epoch != self.epoch
        self.epoch = epoch

    def render(self, mode="human", close=False):
        if close:
            if self.plotter:
                self.plotter.close()
                self.plotter = None
            return None

        trajectory = self.mpc.get_trajectory()
        # trajectory = extend_trajectory(self.mpc.get_trajectory(), self.params["REF_LENGTH"])
        # trajectory = extend_trajectory(self.mpc.get_trajectory(), self.mpc.get_trajectory().get_arclen() + 2)
        if self.plotter is None:
            self.plotter = BarnPlotter(
                trajectory.view(),
                self.occupancy_grid,
                self.upper_coeffs,
                self.lower_coeffs,
                self.mpc,
                self.dynamic_model,
                0.1,
                save_video=self.save_video,
            )

        self.plotter.add_state_to_path(self.robot_state[:2])
        self.plotter.render(
            self.robot_state,
            self.current_ref,
            trajectory.view(),
            self.mpc,
            self.upper_coeffs,
            self.lower_coeffs,
        )

        if self.manual_step:
            a = input()

        return self.plotter.ax

    def get_all_task_idx(self):
        return range(len(self.task_loader))

    def reset_task(self, idx):
        if self.plotter:
            self.plotter.close()

        self.task_idx = idx
        self.reset()

    @staticmethod
    def get_reward(obs, is_colliding, params, N_horizon):
        reward = 0

        # min_const_abv = 1e6
        # min_const_blw = 1e6
        constraints = []
        for i in range(N_horizon):
            cbf_abv = obs[i * len(RLObs) + RLObs.CBF_ABV]
            lfh_lghu_abv = obs[i * len(RLObs) + RLObs.LFH_LGH_ABV]

            cbf_blw = obs[i * len(RLObs) + RLObs.CBF_BLW]
            lfh_lghu_blw = obs[i * len(RLObs) + RLObs.LFH_LGH_BLW]

            const_abv = lfh_lghu_abv + cbf_abv * obs[-2]
            const_blw = lfh_lghu_blw + cbf_blw * obs[-1]

            # print(i, "cbf_abv", cbf_abv)
            # print(i, "cbf_blw", cbf_blw)
            # print(i, "lfh_lgh_abv", lfh_lghu_abv)
            # print(i, "lfh_lgh_blw", lfh_lghu_blw)
            # print(i, "const_abv", const_abv)
            # print(i, "const_blw", const_blw)

            # min_const_abv = min(min_const_abv, const_abv)
            # min_const_blw = min(min_const_blw, const_blw)
            constraints.append(const_abv)
            constraints.append(const_blw)

        rho = 100
        cons = np.array(constraints)

        c_min = np.min(cons)
        exp_cons = np.exp(-rho * (cons - c_min))
        worst_const = c_min - (1 / rho) * np.log(np.mean(np.sum(exp_cons)))

        # print(constraints)
        # print(c_min)
        # print("worst_constraint:", worst_const)

        # reward model for having large constraint values
        # worst_const = min(min_const_abv, min_const_blw)
        a = 7.0
        reward += (np.log(2.0) - np.log1p(np.exp(-a * worst_const))) / a
        reward = np.clip(reward, -5 * np.log(2.0)/a, np.log(2.0)/a)
        print("CONSTRAINT_REWARD", reward)

        avg = (params["MIN_ALPHA"] + params["MAX_ALPHA"]) / 2
        alphas = (obs[-2:] - avg) / avg
        d = np.minimum(alphas - (-1.), 1. - alphas)
        # print(alphas,d)

        # penalize alpha being too large
        # penalize alpha leaving prescribed bounds
        alpha_reward = 0
        # alpha_reward = -.1 * np.sum((obs[-self.N_alpha:] - self.params["MIN_ALPHA"]) / (self.params["MAX_ALPHA"] - self.params["MIN_ALPHA"]))

        # print("largeness:", alpha_reward)
        # alpha_reward -= (1./2) * np.sum((d[d < 0])**2)
        alpha_reward -= np.sum((d[d<0])**2)

        print("ALPHA_REWARD:", alpha_reward)

        reward += alpha_reward

        # solver_status = self.mpc.get_solver_status()
        # reward += -0.05 if not solver_status else 0.0

        if is_colliding:
            # print("total reward before collision:", self.total_reward)
            reward = -5.0

        print("REWARD", reward)

        return reward

    def update_trajectory(self):

        status = None
        jpsPath = vec_Vec2d()
        polys = vec_MatX4d()

        # goal = np.zeros((3, 4))
        # gy = -2.5 + (10 * len(self.world_nums))
        # goal[:2, 0] = [-2.25, gy]
        if not self.planner.has_trajectory:
            start = np.zeros((3, 4))
            start[:2, 0] = self.robot_state[:2]

            status, jpsPath, polys = self._plan(start, self._goal)

            if not status:
                print("failed to find initial trajectory")
                exit(-1)

        elif self.step_count % 5 == 0 or not self.traj_planner_success:
            # horizon = self.mpc.get_horizon()

            init_state = self.mpc.get_state_from_horizon(2)
            start = np.zeros((3, 4))
            # start[:2,0] = self.current_ref
            # start[:2, 0] = self.robot_state[:2]
            start[:2, 0] = init_state[:2]
            if self.dynamic_model == Dynamics.UNICYCLE:
                theta = init_state[2]
                start[:2, 1] = init_state[3] * np.array([np.cos(theta), np.sin(theta)])
            elif self.dynamic_model == Dynamics.DOUBLE_INTEGRATOR:
                print("state is: ", init_state)
                start[:2, 1] = np.array([init_state[2], init_state[3]])
                print(start[:2, 1])

            self.planner.set_costmap(self.map_util)
            self.planner.set_start(start)
            self.planner.set_goal(self._goal)

            status = self.planner.plan(jpsPath, polys)

        if status == PlannerStatus.SUCCESS:
            self.knots, self.xs, self.ys = self.planner.get_arclen_traj()
            print("setting trajectory")
            self.mpc.set_trajectory(
                self.xs,
                self.ys,
                self.knots,
            )
            print("done")

        if status is not None:
            self.traj_planner_success = (
                True if status == PlannerStatus.SUCCESS else False
            )

    def _plan(self, start, goal):

        self.planner.set_costmap(self.map_util)
        self.planner.set_start(start)
        self.planner.set_goal(goal)

        jpsPath = vec_Vec2d()
        polys = vec_MatX4d()
        status = self.planner.plan(jpsPath, polys)

        count = 1
        while status != PlannerStatus.SUCCESS and count < 10:
            self.planner.set_start(start)
            self.planner.set_goal(goal)
            self.planner.set_costmap(self.map_util)

            status = self.planner.plan(jpsPath, polys)
            count += 1

        return status, jpsPath, polys


class RunningStats:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)

    def update(self, x):
        if np.any(np.abs(x) > 10):
            return

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / max(self.n - 1, 1)

    @property
    def std(self):
        return np.sqrt(self.var)


@click.command()
@click.option("--world_num", default=0)
@click.option("--task_num", default=0)
@click.option("--world_sweep", is_flag=True, default=False)
@click.option("--record_data", is_flag=True, default=False)
@click.option("--manual_step", is_flag=True, default=False)
def main(record_data, manual_step, task_num, world_sweep, world_num):

    if not record_data and not world_sweep:
        env = CBFEnv(world_num=[world_num], manual_step=manual_step)
        env.reset_task(task_num)
        done = False
        world_count = 0
        obs_count = 0
        while not done:
            obs, reward, done, _ = env.step([0, 0])
            env.render()

    elif not world_sweep:
        world_count = 0
        obs_count = 0
        n_samples = 1e5
        # n_samples = 1e2
        done = False
        stats = RunningStats(12)
        with open("obs_log.csv", "w", newline="") as f:
            writer = csv.writer(f)

            for i in range(0, 300):
                # obvsiouly cant normalize if we are trying to collect
                # distribution statistics
                env = CBFEnv(world_num=[i], normalize_obs=False)

                done = False
                for j in range(6):

                    env.reset_task(j)
                    done = False

                    while not done:
                        obs, reward, done, _ = env.step([0, 0])
                        obs = obs[:-2]

                        if obs_count > n_samples:
                            break

                        obs_count += 1

                        if not np.any(np.isnan(obs)):
                            writer.writerow(obs)
                            stats.update(obs)

                    if done:
                        world_count += 1

        cbf_abv_mu = [mu for i, mu in enumerate(stats.mean) if i % 4 == 0]
        cbf_dot_abv_mu = [mu for i, mu in enumerate(stats.mean) if i % 4 == 1]

        cbf_blw_mu = [mu for i, mu in enumerate(stats.mean) if i % 4 == 2]
        cbf_dot_blw_mu = [mu for i, mu in enumerate(stats.mean) if i % 4 == 3]

        cbf_abv_std = [std for i, std in enumerate(stats.std) if i % 4 == 0]
        cbf_dot_abv_std = [std for i, std in enumerate(stats.std) if i % 4 == 1]

        cbf_blw_std = [std for i, std in enumerate(stats.std) if i % 4 == 2]
        cbf_dot_blw_std = [std for i, std in enumerate(stats.std) if i % 4 == 3]

        print("CBF_ABV_MEAN:", np.average(cbf_abv_mu))
        print("CBF_DOT_ABV_MEAN:", np.average(cbf_dot_abv_mu))
        print("CBF_BLW_MEAN:", np.average(cbf_blw_mu))
        print("CBF_DOT_BLW_MEAN:", np.average(cbf_dot_blw_mu))

        print("CBF_ABV_STD:", np.average(cbf_abv_std))
        print("CBF_DOT_ABV_STD:", np.average(cbf_dot_abv_std))
        print("CBF_BLW_STD:", np.average(cbf_blw_std))
        print("CBF_DOT_BLW_STD:", np.average(cbf_dot_blw_std))

        print("Code -------------")
        print("obs_mu[RLObs.CBF_ABV]", np.round(np.average(cbf_abv_mu), 4))
        print("obs_mu[RLObs.LFH_LGH_ABV]", np.round(np.average(cbf_dot_abv_mu), 4))
        print("obs_mu[RLObs.CBF_BLW]", np.round(np.average(cbf_blw_mu), 4))
        print("obs_mu[RLObs.LFH_LGH_BLW]:", np.round(np.average(cbf_dot_blw_mu), 4))

        print("obs_std[RLObs.CBF_ABV]", np.round(np.average(cbf_abv_std), 4))
        print("obs_std[RLObs.LFH_LGH_ABV]", np.round(np.average(cbf_dot_abv_std), 4))
        print("obs_std[RLObs.CBF_BLW]", np.round(np.average(cbf_blw_std), 4))
        print("obs_std[RLObs.LFH_LGH_BLW]", np.round(np.average(cbf_dot_blw_std), 4))

        print("ROW_COUNT:", obs_count)

    else:
        outfile = "out.txt"
        with open(outfile, "w", newline="") as f:
            f.write("Below are the simulation results for the test trials\n")
            f.write("WORLD\tTASK\tSUCCESS\tSTEPS\n")

        env = CBFEnv()
        for task in range(6):

            for world_n in range(0, 300, 5):
                env.set_world([world_n])

                for i in range(3):
                    env.reset_task(task)

                    done = False
                    while not done:
                        obs, reward, done, _ = env.step([0, 0])
                        obs = obs[:-2]

                    with open(outfile, "a", newline="") as f:
                        f.write(
                            "%d\t%d\t%d\t%d\t\n"
                            % (
                                world_n,
                                task,
                                not env.did_collide,
                                env.step_count,
                            )
                        )


if __name__ == "__main__":
    main()
