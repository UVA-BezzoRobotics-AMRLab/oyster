import os
import gym
import copy
import numpy as np

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

        self.state_dim = 13
        self.action_dim = 2

        # each task is loaded from parameter list
        self.task_loader = self.load_tasks()
        self.traj_schedule = [30, 10]

        self.randomize_traj = randomize_traj
        self.traj_ids = [
            94,
            38,
            20,
            7,
            41,
            43,
            48,
            24,
            11,
            31,
            40,
            86,
            53,
            61,
            57,
            71,
            15,
            13,
            79,
            66,
            3,
            92,
            96,
        ]
        self.curr_traj_idx = 0
        self.did_collide = False

        self.epoch = 0
        self.epoch_incremented = False

        self.planner = PyPlanner()
        self.step_count = 0

        params = PlannerParams()
        params.SOLVER = "faster"
        params.W_MAX = 1.8
        params.V_MAX = 50
        params.A_MAX = 60
        params.J_MAX = 150
        params.DT_FACTOR_INIT = 1.0
        params.DT_FACTOR_FINAL = 10.0
        params.DT_FACTOR_INCREMENT = 1.0
        params.SOLVER_TRAJ_DT = 0.05
        params.TRIM_DIST = -100

        params.N_SEGMENTS = 6
        params.MAX_POLYS = 3
        params.N_THREADS = 0
        params.FORCE_FINAL_CONSTRAINT = True
        params.VERBOSE = False
        params.USE_MINVO = False
        params.PLAN_IN_FREE = False
        params.SIMPLIFY_JPS = True
        params.MAX_SOLVE_TIME = 0.25

        self.planner.set_params(params)

        obstacles = parse_xml_file("/Users/nickmohammad/Programs/BARN_dataset/world_files/world_290.world")
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
        self.params["CBF_ALPHA_ABV"] = 5.0
        self.params["CBF_ALPHA_BLW"] = 5.0

        init_pose = np.concatenate(([-2.25, -2.5], [np.pi/2]))
        self.mpc = RobotMPC(init_pose, self.params)
        self.mpc.set_occ_map(self.occupancy_grid)
        self.robot_state = self.mpc.get_robot_state()

        self.upper_coeffs = np.array([0] * (self.params["TUBE_DEGREE"]+1))
        self.upper_coeffs[0] = 100
        self.lower_coeffs = np.array([0] * (self.params["TUBE_DEGREE"]+1))
        self.lower_coeffs[0] = 100



    def load_tasks(self):
        param_path = os.path.join(os.path.dirname(__file__), "configs")
        fnames = []
        for file in os.listdir(param_path):
            if file.endswith(".yaml"):
                fnames.append(os.path.join(param_path, file))

        return ParameterLoader(fnames)

    def step(self, action):

        if self.curve is None:
            start = np.zeros((3,4))
            start[:2,0] = self.robot_state[:2]
            self.planner.set_start(start)

            goal = np.zeros((3,4))
            goal[:2,0] = [-2.25, 8.5]
            self.planner.set_goal(goal)

            jpsPath = vec_Vec2d()
            polys = vec_MatX4d()
            self.planner.plan(6, jpsPath, polys)
            self.knots, self.xs, self.ys = self.planner.get_arclen_traj()
            self.curve = BezierCurve(knots=self.knots, xs=self.xs, ys=self.ys)

            self.mpc.set_trajectory(
                self.curve.xs,
                self.curve.ys,
                self.curve.knots,
            )

        elif self.step_count % 5 == 0:
            horizon = self.mpc.get_horizon()

            self.planner.set_costmap(self.map_util)

            start = np.zeros((3,4))
            start[:2,0] = self.current_ref
            self.planner.set_start(start)

            goal = np.zeros((3,4))
            goal[:2,0] = [-2.25, 8.5]
            self.planner.set_goal(goal)

            jpsPath = vec_Vec2d()
            polys = vec_MatX4d()
            self.planner.plan(6, jpsPath, polys)
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

        action[0] = action_unnormalize(
            action[0], self.params["MIN_ALPHA_DOT"], self.params["MAX_ALPHA_DOT"]
        )
        action[1] = action_unnormalize(
            action[1], self.params["MIN_ALPHA_DOT"], self.params["MAX_ALPHA_DOT"]
        )

        dt = self.params["DT"]
        alpha_abv = self.params["CBF_ALPHA_ABV"] + action[0] * dt
        alpha_blw = self.params["CBF_ALPHA_BLW"] + action[1] * dt

        exceed_count = 0
        if alpha_abv < self.params["MIN_ALPHA"] or alpha_abv > self.params["MAX_ALPHA"]:
            exceed_count += 1

        if alpha_blw < self.params["MIN_ALPHA"] or alpha_blw > self.params["MAX_ALPHA"]:
            exceed_count += 1

        self.params["CBF_ALPHA_ABV"] = np.clip(
            alpha_abv, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
        )
        self.params["CBF_ALPHA_BLW"] = np.clip(
            alpha_blw, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
        )

        if self.params["USE_CBF"]:
            self.mpc.load_params(self.params)

        u = self.mpc.get_control(len_start)
        self.robot_state = self.mpc.apply_control(u)

        idx = (np.abs(self.curve.knots.flatten() - len_start)).argmin()
        self.current_ref = (self.curve.xs[idx], self.curve.ys[idx])

        obs = self._get_obs()

        # check if done
        # figure out which side of trajectory we are on
        tangent = self.curve.vel(len_start)
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])

        traj_p = np.array([self.curve.trajx(len_start), self.curve.trajy(len_start)])
        to_robot = self.robot_state[:2] - traj_p
        side = np.sign(np.dot(to_robot, normal))

        dist = self._dist_from_traj(self.robot_state[:2])

        is_colliding = False
        if side < 0:
            is_colliding = dist > np.polyval(self.lower_coeffs[::-1], len_start)
        else:
            is_colliding = dist > np.polyval(self.upper_coeffs[::-1], len_start)

        len_start = self.mpc.get_s_from_pose(self.robot_state[:2])
        is_done = len_start > self.curve.knots[-1] - 2e-1

        if is_colliding:
            self.did_collide = True

        self.step_count += 1

        return (
            obs,
            BarnEnv.get_reward(
                obs,
                action,
                exceed_count,
                is_colliding,
                self.params,
            ),
            # self._get_reward(obs, exceeded_bounds_blw or exceeded_bounds_abv, is_colliding),
            is_colliding or is_done,
            {},
        )

    def reset(self):
        self.did_collide = False

        if self.plotter:
            self.plotter.close()

        params = copy.deepcopy(self.task_loader[self.task_idx])
        min_alpha = params["MIN_ALPHA"]
        max_alpha = params["MAX_ALPHA"]
        if self.epoch >= self.traj_schedule[0]:

            params["CBF_ALPHA_ABV"] = np.random.uniform(min_alpha, max_alpha)
            params["CBF_ALPHA_BLW"] = np.random.uniform(min_alpha, max_alpha)

        self.set_mpc(params)

        self.plotter = None

        # self.robot_state = np.zeros(4, dtype=np.float64)
        # self.mpc.set_mpc_state(self.robot_state)

        obs = np.zeros(13, dtype=np.float64)
        obs[4] = 1.0
        obs[5] = -1.0
        obs[10] = normalize(params["CBF_ALPHA_ABV"], min_alpha, max_alpha)
        obs[11] = normalize(params["CBF_ALPHA_BLW"], min_alpha, max_alpha)
        # obs[10] = 0.0
        # obs[11] = 0.0

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

        v_max = params["LINVEL"]
        curr_progress = mpc_state[5] / np.sqrt(2 * v_max**2)

        state_limits = self.mpc.get_state_limits()
        input_limits = self.mpc.get_input_limits()

        obs = np.zeros(13, dtype=np.float64)
        obs[0] = normalize(mpc_state[2], state_limits[0][2], state_limits[1][2])
        obs[1] = normalize(mpc_state[3], state_limits[0][3], state_limits[1][3])
        obs[2] = normalize(mpc_input[0], input_limits[0][0], input_limits[1][0])
        obs[3] = normalize(mpc_input[1], input_limits[0][1], input_limits[1][0])
        obs[4] = normalize(cbf_data_abv[1], 0, 1.0)
        obs[5] = normalize(cbf_data_blw[1], 0, 1.0)
        obs[6] = normalize(cbf_data_abv[2], -np.pi, np.pi)
        obs[7] = curr_progress
        obs[8] = normalize(cbf_data_abv[0], -100, 100)
        obs[9] = normalize(cbf_data_blw[0], -100, 100)
        obs[10] = normalize(alpha_abv, min_alpha, max_alpha)
        obs[11] = normalize(alpha_blw, min_alpha, max_alpha)
        obs[12] = float(solver_status)

        return obs

    @staticmethod
    def safety_penalty(h, min_val=-10.0, max_val=1.0):
        penalty = -np.exp(-10 * (h - 0.5)) + 1
        return np.clip(penalty, min_val, max_val)

    @staticmethod
    def get_reward(obs, action, exceed_count, is_done, params):
        progress = obs[7]

        h_abv = obs[8]
        h_blw = obs[9]

        alpha_abv = params["CBF_ALPHA_ABV"]
        alpha_blw = params["CBF_ALPHA_BLW"]

        exceeded_bounds_abv = False
        exceeded_bounds_blw = False
        if alpha_abv < params["MIN_ALPHA"] or alpha_abv > params["MAX_ALPHA"]:
            exceeded_bounds_abv = True

        if alpha_blw < params["MIN_ALPHA"] or alpha_blw > params["MAX_ALPHA"]:
            exceeded_bounds_blw = True

        a_max = params["MAX_ALPHA"]
        a_min = params["MIN_ALPHA"]
        avg = (a_max + a_min) / 2.0

        mpc_success = bool(obs[12])

        # mpc_failed = bool(obs[12])
        # weights
        w_feas = 10
        w_safety = 6
        w_progress = 10
        w_collision = 75
        w_alpha_exceeded = 15
        w_alpha_reg = 13

        safety_abv = BarnEnv.safety_penalty(h_abv)
        safety_blw = BarnEnv.safety_penalty(h_blw)

        bounds_penalty = -exceed_count * w_alpha_exceeded
        collision = -w_collision if is_done else 0

        feasibility = 0
        if not mpc_success:
            feasibility = -w_feas

        mid_abv = (alpha_abv - avg) ** 2
        mid_blw = (alpha_blw - avg) ** 2

        reg_abv = action[0] ** 2
        reg_blw = action[1] ** 2

        return float(
            w_safety * safety_abv
            + w_safety * safety_blw
            +
            # w_progress * (1 - progress) +
            w_progress * progress
            + bounds_penalty
            + collision
            + feasibility
            - w_alpha_reg * mid_abv
            - w_alpha_reg * mid_blw
            - w_alpha_reg * reg_abv
            - w_alpha_reg * reg_blw
        )

    def _obs_init(self, n_obs, min_dist):
        # obs = [[6.12, 2.3], [2.75, 4.45]]
        obs = []

        # get initial obstacles
        d_min = 0.1
        d_max = 0.3
        n = np.random.randint(0, 4) + 1
        ss = []
        for i in range(0, n):

            s = np.random.rand() * self.curve.get_arclen()
            while len(ss) > 0 and np.abs(np.min(np.array(ss) - s)) < 1:
                s = np.random.rand() * self.curve.get_arclen()

            d = np.random.rand() * (d_max - d_min) + d_min
            d = -d if np.random.rand() > 0.5 else d
            pos = self.curve.pos(s)
            tan = self.curve.vel(s)
            tan = tan / np.linalg.norm(tan)
            normal = np.array([-tan[1], tan[0]])

            obs.append((pos + normal * d).tolist())

        # obs = [[7.54, 9.32]]
        needed = n_obs

        # get trajectory points
        traj = self.curve.fill(np.linspace(0, 1, 100))

        while len(obs) < n_obs:
            p_min = min(np.min(self.curve.xs), np.min(self.curve.ys))
            p_max = max(np.max(self.curve.xs), np.max(self.curve.ys))

            x_min, y_min = p_min, p_min
            x_max, y_max = p_max, p_max
            # x_min, x_max = float(np.min(self.curve.xs)), float(np.max(self.curve.xs))
            # y_min, y_max = float(np.min(self.curve.ys)), float(np.max(self.curve.ys))

            # oversample to reduce resampling loops
            cand_x = np.random.rand(5 * needed) * (x_max - x_min) + x_min
            cand_y = np.random.rand(5 * needed) * (y_max - y_min) + y_min
            cand = np.vstack([cand_x, cand_y]).T

            # compute distances to trajectory (brute force)
            # None index adds a new axis
            dists = np.min(
                np.linalg.norm(cand[:, None, :] - traj[None, :, :], axis=2), axis=1
            )
            valid = cand[dists > min_dist]

            obs.extend(valid.tolist())
            needed = n_obs - len(obs)

        self.obstacles = np.array(obs[:n_obs])

    def _dist_from_traj(self, point):
        dists = np.linalg.norm(self.curve.pts - point[None, :], axis=1)
        return np.min(dists)


if __name__ == "__main__":
    env = BarnEnv(n_obs=150, randomize_traj=True)

    i = 0
    done = False
    env.reset_task(3)
    while not done:
        _, reward, done, _ = env.step([0, 0])
        env.render()
        i += 1

    # i = 0
    # done = False
    # env.reset_task(2)
    # while not done and i < 200:
    #     _, _, done, _ = env.step([0, 0])
    #     env.render()
    #     i += 1

