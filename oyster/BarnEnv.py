import os
import gym
import time
import copy
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
    # VEL_T = 0
    # VEL_N = auto()
    # D_ABV1 = auto()
    # D_BLW1 = auto()
    # D_ABV2 = auto()
    # D_BLW2 = auto()
    # D_ABV3 = auto()
    # D_BLW3 = auto()
    # D_SIGNED = auto()
    # SIN_OBS_HEADING = auto()
    # COS_OBS_HEADING = auto()
    # CBF_ABV = auto()
    # CBF_BLW = auto()

    CBF_ABV = 0
    CBF_BLW = auto()

def normalize(val, min, max):
    return (val - min) / (max - min)


def action_unnormalize(val, min, max):
    return (val + 1.0) * (max - min) / 2.0 + min


class BarnEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, task={}, n_tasks=2, randomize_tasks=False, n_obs=100, traj_id=None, randomize_traj=False
    ):

        super(BarnEnv, self).__init__()

        self.state_dim = len(RLObs)
        self.action_dim = 2

        # each task is loaded from parameter list
        self.task_loader = self.load_tasks()
        self.traj_schedule = [30, 10]

        self.randomize_traj = randomize_traj
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

        obstacles = parse_xml_file("/home/bezzo/Programs/BARN_dataset/world_files/world_296.world")
        # obstacles = parse_xml_file("/Users/nickmohammad/Programs/BARN_dataset/world_files/world_1.world")
        # obstacles = parse_xml_file("/Users/nickmohammad/Programs/BARN_dataset/world_files/world_1.world")
        self.occupancy_grid = generate_map_from_cylinders(obstacles, 0, 0, 0)
        self.map_util = OccupancyGrid(self.occupancy_grid.info.width, self.occupancy_grid.info.height, self.occupancy_grid.info.resolution, float(self.occupancy_grid.info.origin.position[0]), float(self.occupancy_grid.info.origin.position[1]), np.array(self.occupancy_grid.data), np.array([253, 254]), np.array([255]))

        self.planner.set_costmap(self.map_util)


        if len(self.task_loader) == 0:
            raise ValueError("No parameter files found!")

        self.task_idx = 0

        self.low = np.array(
            np.zeros(self.state_dim),
            dtype=np.float64,
        )

        self.high = np.array(
            np.ones(self.state_dim),
            dtype=np.float64,
        )

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=np.zeros(self.action_dim),
            high=np.ones(self.action_dim),
            dtype=np.float64,
        )

        # for some reason this is needed for PEARL, but it can be
        # set to anything
        self._goal = None
        self.curve = None

        self.plotter = None
        self.reset()


    def set_mpc(self, params):

        self.params = copy.deepcopy(params)
        self.dynamic_model = self.params["DYNAMIC_MODEL"]
        # self.params["USE_CBF"] = False
        # self.params["CBF_ALPHA_ABV"] = 2.5
        # self.params["CBF_ALPHA_BLW"] = 2.5

        init_pose = np.concatenate(([-2.25, -2.5], [np.pi/2]))
        self.mpc = RobotMPC(init_pose, self.params)
        self.mpc.set_occ_map(self.occupancy_grid)
        self.robot_state = self.mpc.get_robot_state()

        self.upper_coeffs = np.array([0] * (self.params["TUBE_DEGREE"]+1))
        self.upper_coeffs[0] = self.params["MAX_TUBE_WIDTH"]/2.0
        self.lower_coeffs = np.array([0] * (self.params["TUBE_DEGREE"]+1))
        self.lower_coeffs[0] = self.params["MAX_TUBE_WIDTH"]/ 2.0



    def load_tasks(self):
        param_path = os.path.join(os.path.dirname(__file__), "configs")
        fnames = []
        for file in os.listdir(param_path):
            if file.endswith(".yaml"):
                fnames.append(os.path.join(param_path, file))

        return ParameterLoader(fnames)

    def step(self, action):
        self.step_count += 1

        status = None
        jpsPath = vec_Vec2d()
        polys = vec_MatX4d()

        if self.curve is None:
            start = np.zeros((3,4))
            start[:2,0] = self.robot_state[:2]

            goal = np.zeros((3,4))
            goal[:2,0] = [-2.25, 8.5]

            status, jpsPath, polys = self._plan(start, goal)

            if not status:
                print("failed to find initial trajectory")
                exit(-1)

        elif self.step_count % 5 == 0:
            horizon = self.mpc.get_horizon()

            self.planner.set_costmap(self.map_util)

            start = np.zeros((3,4))
            # start[:2,0] = self.current_ref
            start[:2,0] = self.robot_state[:2]

            goal = np.zeros((3,4))
            goal[:2,0] = [-2.25, 8.5]

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

        self.params = self.mpc.get_params()

        len_start = self.mpc.get_s_from_pose(self.robot_state[:2])

        ref_len = self.params["REF_LENGTH"]
        horizon = ref_len
        if len_start + horizon > self.knots[-1]:
            horizon = self.knots[-1] - len_start;

        if self.params["USE_CBF"]:
            new_tubes = self.mpc.get_tubes(self.params["TUBE_DEGREE"], 100, self.params["MAX_TUBE_WIDTH"] / 2.0, self.xs, self.ys, 3, self.knots, len_start, horizon)

            if len(new_tubes[0]) > 0:
                self.upper_coeffs = new_tubes[0]
                self.lower_coeffs = -new_tubes[1]

            self.mpc.set_tubes(self.upper_coeffs, self.lower_coeffs)

        action = action_unnormalize(
            action, self.params["MIN_ALPHA_DOT"], self.params["MAX_ALPHA_DOT"]
        )

        dt = self.params["DT"]
        alpha_abv = self.params["CBF_ALPHA_ABV"] + action[0] * dt
        alpha_blw = self.params["CBF_ALPHA_BLW"] + action[1] * dt

        exceed_count = 0
        if alpha_abv < self.params["MIN_ALPHA"] or alpha_abv > self.params["MAX_ALPHA"]:
            exceed_count += 1

        if alpha_blw < self.params["MIN_ALPHA"] or alpha_blw > self.params["MAX_ALPHA"]:
            exceed_count += 1


        # self.params["CBF_ALPHA_ABV"] = np.clip(
        #     alpha_abv, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
        # )
        # self.params["CBF_ALPHA_BLW"] = np.clip(
        #     alpha_blw, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
        # )
        self.params["CBF_ALPHA_ABV"] = alpha_abv
        self.params["CBF_ALPHA_BLW"] = alpha_blw

        if self.params["USE_CBF"]:
            self.mpc.load_params(self.params)

        u = self.mpc.get_control(len_start)
        self.robot_state = self.mpc.apply_control(u)

        idx = (np.abs(self.curve.knots.flatten() - len_start)).argmin()
        self.current_ref = (self.curve.xs[idx], self.curve.ys[idx])

        obs = self._get_obs()

        v_max = self.params["LINVEL"]
        solver_status = self.mpc.get_solver_status()
        mpc_state = self.mpc.get_mpc_state()
        progress = mpc_state[5] / np.sqrt(2 * v_max **2)

        # distance to nearest bound (per alpha)
        rl_min_alpha = self.params["MIN_ALPHA"] - 1.0
        rl_max_alpha = self.params["MAX_ALPHA"] + 1.0
        norm_min_alpha = normalize(self.params["MIN_ALPHA"], rl_min_alpha, rl_max_alpha)
        norm_max_alpha = normalize(self.params["MAX_ALPHA"], rl_min_alpha, rl_max_alpha)
        d = np.minimum(
            obs - norm_min_alpha,
            norm_max_alpha - obs,
        )

        # ---------------- reward ----------------
        # interior reward (maximize margin)
        reward = float(np.sum(d))

        # smooth penalty when violated
        violation = d < 0
        if np.any(violation):
            reward -= .1 * np.sum((-d[violation]) ** 2)

        if np.any(obs < -1.0) or np.any(obs > 1.0):
            reward = -5.0
            done = True
            obs = obs.clip(-1, 1)

        is_done = self.step_count >= 190

        self.total_reward += reward
        return (
            obs,
            reward,
            is_done,
            {},
        )

    # def step(self, action):
    #
    #     status = None
    #     jpsPath = vec_Vec2d()
    #     polys = vec_MatX4d()
    #
    #     if self.curve is None:
    #         start = np.zeros((3,4))
    #         start[:2,0] = self.robot_state[:2]
    #
    #         goal = np.zeros((3,4))
    #         goal[:2,0] = [-2.25, 8.5]
    #
    #         status, jpsPath, polys = self._plan(start, goal)
    #
    #         if not status:
    #             print("failed to find initial trajectory")
    #             exit(-1)
    #
    #     elif self.step_count % 5 == 0:
    #         horizon = self.mpc.get_horizon()
    #
    #         self.planner.set_costmap(self.map_util)
    #
    #         start = np.zeros((3,4))
    #         # start[:2,0] = self.current_ref
    #         start[:2,0] = self.robot_state[:2]
    #
    #         goal = np.zeros((3,4))
    #         goal[:2,0] = [-2.25, 8.5]
    #
    #         self.planner.set_costmap(self.map_util)
    #         self.planner.set_start(start)
    #         self.planner.set_goal(goal)
    #
    #         status = self.planner.plan(jpsPath, polys)
    #
    #     if status == PlannerStatus.SUCCESS:
    #         self.knots, self.xs, self.ys = self.planner.get_arclen_traj()
    #         self.curve = BezierCurve(knots=self.knots, xs=self.xs, ys=self.ys)
    #
    #         self.mpc.set_trajectory(
    #             self.curve.xs,
    #             self.curve.ys,
    #             self.curve.knots,
    #         )
    #
    #     self.params = self.mpc.get_params()
    #
    #     len_start = self.mpc.get_s_from_pose(self.robot_state[:2])
    #
    #     ref_len = self.params["REF_LENGTH"]
    #     horizon = ref_len
    #     if len_start + horizon > self.knots[-1]:
    #         horizon = self.knots[-1] - len_start;
    #
    #     if self.params["USE_CBF"]:
    #         new_tubes = self.mpc.get_tubes(self.params["TUBE_DEGREE"], 100, self.params["MAX_TUBE_WIDTH"] / 2.0, self.xs, self.ys, 3, self.knots, len_start, horizon)
    #
    #         if len(new_tubes[0]) > 0:
    #             self.upper_coeffs = new_tubes[0]
    #             self.lower_coeffs = -new_tubes[1]
    #
    #         self.mpc.set_tubes(self.upper_coeffs, self.lower_coeffs)
    #
    #     action[0] = action_unnormalize(
    #         action[0], self.params["MIN_ALPHA_DOT"], self.params["MAX_ALPHA_DOT"]
    #     )
    #     action[1] = action_unnormalize(
    #         action[1], self.params["MIN_ALPHA_DOT"], self.params["MAX_ALPHA_DOT"]
    #     )
    #
    #     dt = self.params["DT"]
    #     alpha_abv = self.params["CBF_ALPHA_ABV"] + action[0] * dt
    #     alpha_blw = self.params["CBF_ALPHA_BLW"] + action[1] * dt
    #
    #     exceed_count = 0
    #     if alpha_abv < self.params["MIN_ALPHA"] or alpha_abv > self.params["MAX_ALPHA"]:
    #         exceed_count += 1
    #
    #     if alpha_blw < self.params["MIN_ALPHA"] or alpha_blw > self.params["MAX_ALPHA"]:
    #         exceed_count += 1
    #
    #
    #     # self.params["CBF_ALPHA_ABV"] = np.clip(
    #     #     alpha_abv, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
    #     # )
    #     # self.params["CBF_ALPHA_BLW"] = np.clip(
    #     #     alpha_blw, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
    #     # )
    #     self.params["CBF_ALPHA_ABV"] = alpha_abv
    #     self.params["CBF_ALPHA_BLW"] = alpha_blw
    #
    #     if self.params["USE_CBF"]:
    #         self.mpc.load_params(self.params)
    #
    #     u = self.mpc.get_control(len_start)
    #     self.robot_state = self.mpc.apply_control(u)
    #
    #     idx = (np.abs(self.curve.knots.flatten() - len_start)).argmin()
    #     self.current_ref = (self.curve.xs[idx], self.curve.ys[idx])
    #
    #     obs = self._get_obs()
    #
    #     # if exceed_count > 0:
    #     #     reward = BarnEnv.get_reward(
    #     #             obs, 
    #     #             True,
    #     #             0,
    #     #             action,
    #     #             exceed_count,
    #     #             False,
    #     #             self.params,
    #     #     )
    #     #     self.total_reward += reward
    #     #     return (
    #     #         obs, 
    #     #         reward,
    #     #         True,
    #     #         {},
    #     #     )
    #
    #     # check if done
    #     # figure out which side of trajectory we are on
    #     tangent = self.curve.vel(len_start)
    #     tangent /= np.linalg.norm(tangent)
    #     normal = np.array([-tangent[1], tangent[0]])
    #
    #     traj_p = np.array([self.curve.trajx(len_start), self.curve.trajy(len_start)])
    #     to_robot = self.robot_state[:2] - traj_p
    #     side = np.sign(np.dot(to_robot, normal))
    #
    #     dist = self._dist_from_traj(self.robot_state[:2])
    #
    #     # is_colliding = False
    #     # if side < 0:
    #     #     is_colliding = dist > np.polyval(self.lower_coeffs[::-1], len_start)
    #     # else:
    #     #     is_colliding = dist > np.polyval(self.upper_coeffs[::-1], len_start)
    #     is_colliding = self.map_util.is_occupied(self.robot_state[0], self.robot_state[1], "inflated")
    #
    #     len_start = self.mpc.get_s_from_pose(self.robot_state[:2])
    #     is_done = len_start > self.curve.knots[-1] - 2e-1
    #
    #     if is_colliding:
    #         self.did_collide = True
    #
    #     self.step_count += 1
    #
    #     v_max = self.params["LINVEL"]
    #     solver_status = self.mpc.get_solver_status()
    #     mpc_state = self.mpc.get_mpc_state()
    #     progress = mpc_state[5] / np.sqrt(2 * v_max **2)
    #
    #     if is_done:
    #         self.is_success = True
    #
    #     # if np.any(obs > 1.0) or np.any(obs < -1.0):
    #     #     print("d abv:", np.polyval(self.upper_coeffs[::-1], 0.))
    #     #     print("d blw:", np.polyval(self.lower_coeffs[::-1], 0.))
    #     #     print(obs)
    #     #     exit(0)
    #     reward = BarnEnv.get_reward(
    #             obs,
    #             solver_status,
    #             progress,
    #             action,
    #             exceed_count,
    #             is_colliding,
    #             self.params,
    #         )
    #
    #     self.total_reward += reward
    #     return (
    #         obs,
    #         reward,
    #         is_colliding or is_done,
    #         {},
    #     )

    def reset(self):
        self.did_collide = False
        self.total_reward = 0
        self.is_success = False

        if self.plotter:
            self.plotter.close()

        params = copy.deepcopy(self.task_loader[self.task_idx])
        min_alpha = params["MIN_ALPHA"]
        max_alpha = params["MAX_ALPHA"]
        if self.epoch >= self.traj_schedule[0]:

            params["CBF_ALPHA_ABV"] = np.random.uniform(min_alpha, max_alpha)
            params["CBF_ALPHA_BLW"] = np.random.uniform(min_alpha, max_alpha)

        self.set_mpc(params)
        self.curve = None

        self.plotter = None

        # self.robot_state = np.zeros(4, dtype=np.float64)
        # self.mpc.set_mpc_state(self.robot_state)

        obs = np.zeros(len(RLObs), dtype=np.float64)
        # obs[RLObs.D_ABV1] = 1.0
        # obs[RLObs.D_BLW1] = 0.
        # obs[RLObs.D_ABV2] = 1.0
        # obs[RLObs.D_BLW2] = 0.
        # obs[RLObs.D_ABV3] = 1.0
        # obs[RLObs.D_BLW3] = 0.
        obs[RLObs.CBF_ABV] = normalize(params["CBF_ALPHA_ABV"], min_alpha, max_alpha)
        obs[RLObs.CBF_BLW] = normalize(params["CBF_ALPHA_BLW"], min_alpha, max_alpha)

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
            self.robot_state, self.current_ref, self.curve, self.mpc, self.upper_coeffs, self.lower_coeffs
        )

        return self.plotter.ax

    def get_all_task_idx(self):
        return range(len(self.task_loader))

    def reset_task(self, idx):
        if self.plotter:
            self.plotter.close()

        self.task_idx = idx
        self.reset()

    def _get_obs(self):
        mpc_state = self.mpc.get_mpc_state()
        mpc_input = self.mpc.get_mpc_command()
        solver_status = self.mpc.get_solver_status()

        cbf_data_abv = self.mpc.get_cbf_data(mpc_state, mpc_input, True)
        cbf_data_blw = self.mpc.get_cbf_data(mpc_state, mpc_input, False)

        params = self.mpc.get_params()
        alpha_abv = params["CBF_ALPHA_ABV"]
        alpha_blw = params["CBF_ALPHA_BLW"]

        min_alpha = params["MIN_ALPHA"]
        max_alpha = params["MAX_ALPHA"]
        rl_min_alpha = min_alpha - 1.0
        rl_max_alpha = max_alpha + 1.0

        v_max = params["LINVEL"]
        curr_progress = mpc_state[5] / np.sqrt(2 * v_max**2)

        state_limits = self.mpc.get_state_limits()
        input_limits = self.mpc.get_input_limits()

        obs = np.zeros(len(RLObs), dtype=np.float64)

        len_start = self.mpc.get_s_from_pose(self.robot_state[:2])

        tangent = self.curve.vel(len_start)
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])

        traj_p = np.array([self.curve.trajx(len_start), self.curve.trajy(len_start)])
        to_robot = self.robot_state[:2] - traj_p
        side = np.sign(np.dot(to_robot, normal))

        signed_dist = side * self._dist_from_traj(self.robot_state[:2])

        # technically maximum distance is sum of both... 
        max_width = self.params["MAX_TUBE_WIDTH"]

        remaining_len = min(self.curve.get_arclen() - len_start, self.params["REF_LENGTH"])

        vel = np.array(mpc_state[2:4])
        # obs[RLObs.VEL_T] = normalize(np.dot(vel, tangent), state_limits[0][2], state_limits[1][2])
        # obs[RLObs.VEL_N] = normalize(np.dot(vel, normal), state_limits[0][3], state_limits[1][3])
        # obs[RLObs.D_ABV1] = normalize(
        #         np.polyval(self.upper_coeffs[::-1], 0), 0, max_width)
        # obs[RLObs.D_BLW1] = normalize(
        #         np.polyval(self.lower_coeffs[::-1], 0), 0, max_width)
        # obs[RLObs.D_ABV2] = normalize(
        #         np.polyval(self.upper_coeffs[::-1], min(.5, remaining_len)), 0, max_width)
        # obs[RLObs.D_BLW2] = normalize(
        #         np.polyval(self.lower_coeffs[::-1], min(.5, remaining_len)), 0, max_width)
        # obs[RLObs.D_ABV3] = normalize(
        #         np.polyval(self.upper_coeffs[::-1], min(1.0, remaining_len)), 0, max_width)
        # obs[RLObs.D_BLW3] = normalize(
        #         np.polyval(self.lower_coeffs[::-1], min(1.0, remaining_len)), 0, max_width)
        #
        # obs[RLObs.D_SIGNED] = normalize(signed_dist, -max_width, max_width)
        # obs[RLObs.SIN_OBS_HEADING] = np.sin(cbf_data_abv[2])
        # obs[RLObs.COS_OBS_HEADING] = np.cos(cbf_data_abv[2])
        obs[RLObs.CBF_ABV] = normalize(alpha_abv, rl_min_alpha, rl_max_alpha)
        obs[RLObs.CBF_BLW] = normalize(alpha_blw, rl_min_alpha, rl_max_alpha)

        return obs

    @staticmethod
    def safety_penalty(h, min_val=-10.0, max_val=1.0):
        penalty = -np.exp(-10 * (h - 0.5)) + 1
        return np.clip(penalty, min_val, max_val)

    @staticmethod
    def get_reward(obs, solver_status, progress, action, exceed_count, is_done, params):
        # progress = obs[7]

        # h_abv = obs[8]
        # h_blw = obs[9]

        alpha_abv = params["CBF_ALPHA_ABV"]
        alpha_blw = params["CBF_ALPHA_BLW"]

        a_max = params["MAX_ALPHA"]
        a_min = params["MIN_ALPHA"]
        avg = (a_max + a_min) / 2.0

        # mpc_success = bool(obs[12])

        # mpc_failed = bool(obs[12])
        # weights
        w_feas = 0
        # w_safety = 6
        w_progress = 3
        w_collision = 150
        w_alpha_exceeded = 150
        w_alpha_reg = 1

        # safety_abv = BarnEnv.safety_penalty(h_abv)
        # safety_blw = BarnEnv.safety_penalty(h_blw)

        # bounds_penalty = -exceed_count * w_alpha_exceeded
        # bounds_penalty = 0
        # if exceed_count > 0:
        #     bounds_penalty = -w_alpha_exceeded

        collision = -w_collision if is_done else 0

        feasibility = 0
        if not solver_status:
            feasibility = -w_feas

        reward = float(
            # w_safety * safety_abv
            # + w_safety * safety_blw
            # +
            # w_progress * (1 - progress) +

            # w_progress * progress
            + bounds_penalty
            # + collision
            # + feasibility

            # - w_alpha_reg * mid_abv
            # - w_alpha_reg * mid_blw
            # - w_alpha_reg * reg_abv
            # - w_alpha_reg * reg_blw
        )

        # print('reward:')
        # print('\tsafety_abv:', w_safety * safety_abv)
        # print('\tsafety_blw:', w_safety * safety_blw)
        # print('\tprogress:', w_progress * progress)
        # print('\tbounds_penalty:', bounds_penalty)
        # print('\tcollision:', collision)
        # print('\tfeasibility:', feasibility)
        # print('\talpha_abv_mid:', w_alpha_reg * mid_abv)
        # print('\talpha_blw_mid:', w_alpha_reg * mid_blw)
        # print('\talpha_abv_reg:', w_alpha_reg * reg_abv)
        # print('\talpha_blw_reg:', w_alpha_reg * reg_blw)
        # print('total:', reward)


        return reward

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


if __name__ == "__main__":
    env = BarnEnv(n_obs=150, randomize_traj=True)

    i = 0
    done = False
    env.reset_task(1)
    while not done:
        _, reward, done, _ = env.step([0, 0])
        obs = env._get_obs()
        # print("signed:", obs[RLObs.D_SIGNED])
        # print("vel T:", obs[RLObs.VEL_T])
        # print("vel N:", obs[RLObs.VEL_N])
        env.render()
        i += 1
        # time.sleep(.2)

    # i = 0
    # done = False
    # env.reset_task(3)
    # while not done and i < 200:
    #     _, _, done, _ = env.step([0, 0])
    #     env.render()
    #     i += 1

