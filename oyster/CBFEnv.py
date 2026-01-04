import os
import csv
import gym
import time
import copy
import click
import numpy as np

from enum import IntEnum, auto
from gym import spaces
from oyster.BarnPlotter import BarnPlotter
from oyster.Planner import PyPlanner
from oyster.Bezier import BezierCurve
from oyster.Tubes import TubeGenerator
from oyster.TrajLibGen import TrajLibLoader
from oyster.ParamLoader import ParameterLoader
from oyster.RobotMPC import RobotMPC, Dynamics
from MapLoader import parse_xml_file, generate_map_from_cylinders

from py_planner import Planner
from py_planner import RFNState
from py_planner import vec_Vec2d, vec_MatX4d
from py_planner import PlannerStatus
from py_planner import PlannerParams
from py_planner import OccupancyGrid
from MapLoader import parse_xml_file, generate_map_from_cylinders


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
        world_num=0,
        N = 3
    ):

        super(CBFEnv, self).__init__()

        self.N_alpha = 2
        self.N_horizon = N

        self.state_dim = len(RLObs) * self.N_horizon + self.N_alpha
        self.action_dim = 2

        # each task is loaded from parameter list
        self.task_loader = self.load_tasks()
        self.traj_schedule = [30, 10]

        self.did_collide = False

        self.epoch = 0
        self.epoch_incremented = False

        self.planner = PyPlanner()
        self.step_count = 0

        self.total_reward = 0
        self.is_success = False

        self.planner_params = PlannerParams()
        self.planner_params.SOLVER = "faster"
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

        try:
            obstacles = parse_xml_file(os.path.join(os.getenv("BARN_DATASET_PATH"), "world_files", f"world_{world_num}.world"))
        except:
            obstacles = None
            print("[ERROR] Loading obstacles did not work, has BARN dataset been installed and the BARN_DATASET_PATH",
                  "environment variable been set to it's top level directory location?")
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

        if len(self.task_loader) == 0:
            raise ValueError("No parameter files found!")

        self.task_idx = 0

        # i dont think these bounds every get used by the SAC algorithm
        self.low = np.array(
            -2 * np.ones(self.state_dim),
            dtype=np.float64,
        )

        self.high = np.array(
            2 * np.ones(self.state_dim),
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
        self._goal = None
        self.curve = None

        self.plotter = None
        self.reset()


    def set_mpc(self, params):

        self.params = copy.deepcopy(params)
        self.dynamic_model = self.params["DYNAMIC_MODEL"]
        self.params["USE_CBF"] = True

        init_pose = np.concatenate(([-2.25, -2.5], [np.pi / 2]))
        self.mpc = RobotMPC(init_pose, self.params)
        self.mpc.set_occ_map(self.occupancy_grid)
        self.robot_state = self.mpc.get_robot_state()

        self.upper_coeffs = np.array([0] * (self.params["TUBE_DEGREE"] + 1))
        self.upper_coeffs[0] = self.params["MAX_TUBE_WIDTH"] / 2.0
        self.lower_coeffs = np.array([0] * (self.params["TUBE_DEGREE"] + 1))
        self.lower_coeffs[0] = -self.params["MAX_TUBE_WIDTH"] / 2.0

        # These values were computed empirically over 50k samples
        obs_mu = np.zeros(len(RLObs))
        obs_mu[RLObs.LFH_LGH_ABV] = -0.188003803
        obs_mu[RLObs.CBF_ABV] = 0.121208923
        obs_mu[RLObs.LFH_LGH_BLW] = -0.061815101
        obs_mu[RLObs.CBF_BLW] = -0.306548636


        obs_std = np.zeros(len(RLObs))
        obs_std[RLObs.LFH_LGH_ABV] = 0.458712499
        obs_std[RLObs.CBF_ABV] = 0.21908311
        obs_std[RLObs.LFH_LGH_BLW] = 0.571774786
        obs_std[RLObs.CBF_BLW] = 0.121597948

        self.obs_mu = np.zeros(obs_mu.shape[0] * self.N_horizon + self.N_alpha)
        self.obs_std = np.zeros(obs_std.shape[0] * self.N_horizon + self.N_alpha)

        self.obs_mu[:-self.N_alpha] = np.tile(obs_mu, self.N_horizon)
        self.obs_std[:-self.N_alpha] = np.tile(obs_std, self.N_horizon)

        for i in range(self.N_alpha):
            self.obs_mu[len(RLObs) * self.N_horizon + i] = (self.params["MIN_ALPHA"] + self.params["MAX_ALPHA"])/2

        for i in range(self.N_alpha):
            self.obs_std[len(RLObs) * self.N_horizon + i] = (self.params["MIN_ALPHA"] + self.params["MAX_ALPHA"])/4


    def load_tasks(self):
        param_path = os.path.join(os.path.dirname(__file__), "configs")
        fnames = []
        for file in os.listdir(param_path):
            if file.endswith(".yaml"):
                fnames.append(os.path.join(param_path, file))

        return ParameterLoader(fnames)

    def normalize_obs(self, obs):
        # 95% of observed data will be within -1 to 1
        z = (obs - self.obs_mu) / (2*self.obs_std)
        z = np.clip(z, self.low, self.high)
        return z

    def step(self, action):
        self.step_count += 1

        self.update_trajectory()

        self.params = self.mpc.get_params()

        len_start = self.mpc.get_s_from_pose(self.robot_state[:2])
        self.get_and_set_tubes(len_start)

        action = np.array(action)
        action = action_unnormalize(
            action, self.params["MIN_ALPHA_DOT"], self.params["MAX_ALPHA_DOT"]
        )

        dt = self.params["DT"]
        alpha_abv = self.params["CBF_ALPHA_ABV"] + action[0] * dt
        alpha_blw = self.params["CBF_ALPHA_BLW"] + action[1] * dt

        self.params["CBF_ALPHA_ABV"] = alpha_abv
        self.params["CBF_ALPHA_BLW"] = alpha_blw

        if self.params["USE_CBF"]:
            self.mpc.load_params(self.params)

        u = self.mpc.get_control(len_start)
        self.robot_state = self.mpc.apply_control(u)

        idx = (np.abs(self.curve.knots.flatten() - len_start)).argmin()
        self.current_ref = (self.curve.xs[idx], self.curve.ys[idx])

        obs = self.get_obs()
        # print("obs:", self.normalize_obs(obs))

        v_max = self.params["LINVEL"]

        is_colliding = self.map_util.is_occupied(self.robot_state[0], self.robot_state[1], "inflated")

        # ---------------- reward ----------------
        reward = self.get_reward(obs, is_colliding)

        is_done = self.step_count >= 190 or (len_start >= self.curve.knots[-1] - 1) or is_colliding

        self.total_reward += reward
        return (
            # self.normalize_obs(obs),
            obs,
            reward,
            is_done,
            {},
        )

    def reset(self):
        self.did_collide = False
        self.total_reward = 0
        self.is_success = False
        self.step_count = 0

        if self.plotter:
            self.plotter.close()

        params = copy.deepcopy(self.task_loader[self.task_idx])
        min_alpha = params["MIN_ALPHA"]
        max_alpha = params["MAX_ALPHA"]

        if self.epoch >= self.traj_schedule[0]:
            params["CBF_ALPHA_ABV"] = np.random.uniform(min_alpha, max_alpha)
            params["CBF_ALPHA_BLW"] = np.random.uniform(min_alpha, max_alpha)

        params["CBF_ALPHA_ABV"] = 2.5
        params["CBF_ALPHA_BLW"] = 2.5

        self.set_mpc(params)
        self.curve = None

        self.plotter = None

        # self.robot_state = np.zeros(4, dtype=np.float64)
        # self.mpc.set_mpc_state(self.robot_state)

        obs = np.zeros(self.state_dim, dtype=np.float64)
        obs[-self.N_alpha:] = [params["CBF_ALPHA_ABV"], params["CBF_ALPHA_BLW"]]
        return self.normalize_obs(obs)

    def get_obs(self):
        horizon = np.array(self.mpc.get_horizon())
        if horizon.shape[0] < self.N_horizon:
            raise ValueError("Horizon shape", horizon.shape[0], "is smaller than N_horizon set", 
                             self.N_horizon)

        xs, ys = self.curve.compute_adjusted_ref(0)

        obs = np.zeros(len(RLObs) * self.N_horizon + self.N_alpha)
        for i in range(self.N_horizon):
            state = horizon[i,1:7]
            acc = horizon[i,7:]

            cbf_abv = self.mpc.get_cbf_abv(state, self.upper_coeffs, xs, ys)
            lfh_abv = self.mpc.get_lfh_abv(state, self.upper_coeffs, xs, ys)
            lgh_abv = self.mpc.get_lgh_abv(state, self.upper_coeffs, xs, ys)
            lghu_abv = lgh_abv[:2] @ acc

            cbf_blw = self.mpc.get_cbf_blw(state, self.lower_coeffs, xs, ys)
            lfh_blw = self.mpc.get_lfh_blw(state, self.lower_coeffs, xs, ys)
            lgh_blw = self.mpc.get_lgh_blw(state, self.lower_coeffs, xs, ys)
            lghu_blw = lgh_blw[:2] @ acc

            if np.isnan(cbf_abv) or np.isnan(cbf_blw):
                print("STATE", state)
                print("upper", self.upper_coeffs)
                print("lower", self.lower_coeffs)
                print(xs)
                print(ys)

                print("xr_dot", self.mpc.get_xrdot(state, xs, ys))
                print("yr_dot", self.mpc.get_yrdot(state, xs, ys))

                print(i, "cbf_abv", cbf_abv)
                print(i, "cbf_blw", cbf_blw)
                print(i, "lfh_lgh_abv", lfh_abv + lghu_abv)
                print(i, "lfh_lgh_blw", lfh_blw + lghu_blw)

            obs[i * len(RLObs) + RLObs.CBF_ABV] = float(cbf_abv)
            obs[i * len(RLObs) + RLObs.LFH_LGH_ABV] = float(lfh_abv + lghu_abv)
            obs[i * len(RLObs) + RLObs.CBF_BLW] = float(cbf_blw)
            obs[i * len(RLObs) + RLObs.LFH_LGH_BLW] = float(lfh_blw + lghu_blw)

            # obs[i * len(RLObs) : (i+1)*len(RLObs)] = [
            #     float(cbf_abv), 
            #     float(lfh_abv + lghu_abv),
            #     float(cbf_blw),
            #     float(lfh_blw + lghu_blw),
            # ]

        obs[-self.N_alpha:] = [self.params["CBF_ALPHA_ABV"], self.params["CBF_ALPHA_BLW"]]

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

        if self.plotter is None:
            self.plotter = BarnPlotter(
                self.curve,
                self.occupancy_grid,
                self.upper_coeffs,
                self.lower_coeffs,
                self.dynamic_model,
                0.25,
            )

        self.plotter.add_state_to_path(self.robot_state[:2])
        self.plotter.render(
            self.robot_state,
            self.current_ref,
            self.curve,
            self.mpc,
            self.upper_coeffs,
            self.lower_coeffs,
        )

        return self.plotter.ax

    def get_all_task_idx(self):
        return range(len(self.task_loader))

    def reset_task(self, idx):
        if self.plotter:
            self.plotter.close()

        self.task_idx = idx
        self.reset()

    def get_reward(self, obs, is_colliding):
        reward = 0

        min_const_abv = 1e6
        min_const_blw = 1e6
        for i in range(self.N_horizon):
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

            min_const_abv = min(min_const_abv, const_abv)
            min_const_blw = min(min_const_blw, const_blw)

        # reward model for having large constraint values
        worst_const = min(min_const_abv, min_const_blw)
        # print("WORST_CONST", worst_const)
        reward += 0.7 * np.tanh(worst_const)
        # print("CONST REWARD", reward)

        avg = (self.params["MIN_ALPHA"] + self.params["MAX_ALPHA"]) / 2
        alphas = (obs[-self.N_alpha:] - avg) / avg
        d = np.minimum(alphas - (-1.), 1. - alphas)

        # penalize alpha being too large
        # penalize alpha leaving prescribed bounds
        alpha_reward = -.1 * np.sum((obs[-self.N_alpha:] - self.params["MIN_ALPHA"]) / (self.params["MAX_ALPHA"] - self.params["MIN_ALPHA"]))
        # print("largeness:", alpha_reward)
        alpha_reward -= np.sum((d[d < 0])**2)

        # print("ALPHA_REWARD:", alpha_reward)

        reward += alpha_reward

        if is_colliding:
            reward = -1.0

        # print("REWARD", reward)

        return reward

    def get_and_set_tubes(self, len_start):
        if not hasattr(self, "knots"):
            return

        ref_len = self.params["REF_LENGTH"]
        horizon = ref_len
        if len_start + horizon > self.knots[-1]:
            horizon = self.knots[-1] - len_start

        if self.params["USE_CBF"]:
            new_tubes = self.mpc.get_tubes(
                self.params["TUBE_DEGREE"],
                100,
                self.params["MAX_TUBE_WIDTH"] / 2.0,
                self.xs,
                self.ys,
                3,
                self.knots,
                len_start,
                horizon,
            )

            if len(new_tubes[0]) > 0:
                self.upper_coeffs = new_tubes[0]
                self.lower_coeffs = new_tubes[1]

        else:
            self.upper_coeffs = [0] * int(self.params["TUBE_DEGREE"]+1)
            self.upper_coeffs[0] = 100
            self.lower_coeffs = [0] * int(self.params["TUBE_DEGREE"]+1)
            self.lower_coeffs[0] = -100

        self.mpc.set_tubes(self.upper_coeffs, self.lower_coeffs)

    def update_trajectory(self):

        status = None
        jpsPath = vec_Vec2d()
        polys = vec_MatX4d()

        if self.curve is None:
            start = np.zeros((3, 4))
            start[:2, 0] = self.robot_state[:2]

            goal = np.zeros((3, 4))
            goal[:2, 0] = [-2.25, 8.5]

            status, jpsPath, polys = self._plan(start, goal)

            if not status:
                print("failed to find initial trajectory")
                exit(-1)

        elif self.step_count % 5 == 0:
            horizon = self.mpc.get_horizon()

            start = np.zeros((3, 4))
            # start[:2,0] = self.current_ref
            start[:2, 0] = self.robot_state[:2]

            goal = np.zeros((3, 4))
            goal[:2, 0] = [-2.25, 8.5]

            self.planner.set_costmap(self.map_util)
            self.planner.set_start(start)
            self.planner.set_goal(goal)

            status = self.planner.plan(jpsPath, polys)

        if status == PlannerStatus.SUCCESS:
            self.knots, self.xs, self.ys = self.planner.get_arclen_traj()
            self.curve = BezierCurve(knots=self.knots, xs=self.xs, ys=self.ys)

            self.mpc.set_trajectory(
                self.curve.xs,
                self.curve.ys,
                self.curve.knots,
            )

    def _dist_from_traj(self, point):
        dists = np.linalg.norm(self.curve.pts - point[None, :], axis=1)
        return np.min(dists)

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
@click.option("--record_data", is_flag=True, default=False)
def main(record_data):

    if not record_data:
        env = CBFEnv(world_num = 296)
        env.reset_task(1)
        done = False
        world_count = 0
        obs_count = 0
        while not done:
            obs, reward, done, _ = env.step([0, 0])
            env.render()

        env.reset_task(4)
        done = False
        while not done:
            obs, reward, done, _ = env.step([0, 0])
            env.render()

    else:
        world_count = 0
        obs_count = 0
        n_samples = 1e5
        done = False
        stats = RunningStats(14)
        with open("obs_log.csv", "w", newline="") as f:
            writer = csv.writer(f)

            for i in range(0, 300, 3):
                env = CBFEnv(world_num = i)

                done = False
                for j in range(6):

                    env.reset_task(j)
                    done = False

                    while not done:
                        try:
                            obs, reward, done, _ = env.step([0, 0])
                        except:
                            break

                        if obs_count > n_samples:
                            break

                        obs_count += 1
                        writer.writerow(obs)
                        stats.update(obs)

                    if done:
                        world_count += 1

                print("MEAN:", stats.mean)
                print("STD:", stats.std)
                print("ROW_COUNT:", obs_count)

if __name__ == "__main__":
    main()

