import numpy as np
import matplotlib.pyplot as plt

from py_planner import Planner
from py_planner import RFNState
from py_planner import vec_Vec2d, vec_MatX4d
from py_planner import PlannerStatus
from py_planner import PlannerParams
from py_planner import OccupancyGrid
from oyster.MapLoader import parse_xml_file, generate_map_from_cylinders

from scipy.interpolate import UnivariateSpline


class PyPlanner:

    def __init__(self):
        self.params = None
        self.planner = Planner()

        self.max_horizon = 3.0
        self.curr_horizon = self.max_horizon
        self.has_trajectory = True

    def set_params(self, params):
        self.params = params
        self.planner.set_params(params)

    def set_start(self, start):
        self.planner.set_start(start)

    def set_goal(self, goal):
        self.planner.set_goal(goal)

    def set_costmap(self, cmap):
        self.planner.set_costmap(cmap)

    def plan(self, path, polys):
        status = self.planner.plan(self.curr_horizon, path, polys)

        if status:
            self.curr_horizon = self.max_horizon
            self.has_trajectory = True
        else:
            self.curr_horizon /= 0.9

        return status

    def get_arclen_traj(self, refine=False):
        # if refine:
        #     traj = self.planner.refine_traj()
        # else:
        traj = self.planner.get_arclen_traj(refine)

        knots = np.array([x.t for x in traj])
        xs = np.array([x.pos[0] for x in traj])
        ys = np.array([x.pos[1] for x in traj])

        # trajx = UnivariateSpline(knots, xs, k=3, s=0)
        # trajy = UnivariateSpline(knots, ys, k=3, s=0)
        #
        # trajx_d = trajx.derivative(n=1)
        # trajy_d = trajy.derivative(n=1)
        #
        # for s in np.linspace(0, knots[-1], 100):
        #     mag = np.sqrt(trajx_d(s)**2 + trajy_d(s)**2)
        #     print(mag)
        #
        # exit(0)

        return knots, xs, ys


def plot_jps(jps):
    plt.plot(jps[:, 0], jps[:, 1])


def plot_traj(traj):
    ps = np.array([[x.pos[0], x.pos[1]] for x in traj])
    plt.plot(ps[:, 0], ps[:, 1])


if __name__ == "__main__":

    planner = PyPlanner()

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

    planner.set_params(params)

    obstacles = parse_xml_file(
        "/Users/nickmohammad/Programs/BARN_dataset/world_files/world_5.world"
    )
    occupancy_grid = generate_map_from_cylinders(obstacles, 0, 0, 0)

    map_util = OccupancyGrid(
        occupancy_grid.info.width,
        occupancy_grid.info.height,
        occupancy_grid.info.resolution,
        float(occupancy_grid.info.origin.position[0]),
        float(occupancy_grid.info.origin.position[1]),
        np.array(occupancy_grid.data),
        np.array([253, 254]),
        np.array([255]),
    )

    planner.set_costmap(map_util)

    start = np.zeros((3, 4))
    start[:, 0] = np.array([-2.25, -2.5, 0])
    planner.set_start(start)

    goal = np.zeros((3, 4))
    goal[:, 0] = np.array([-2.1, 2.25, 0])
    planner.set_goal(goal)

    jpsPath = vec_Vec2d()
    polys = vec_MatX4d()
    planner.plan(6, jpsPath, polys)
    jpsPath = np.array(jpsPath[:]).reshape(len(jpsPath), 2)

    traj = planner.get_arclen_traj()

    plot_grid(occupancy_grid)
    plot_jps(jpsPath)
    plot_traj(traj)
    plt.show()
