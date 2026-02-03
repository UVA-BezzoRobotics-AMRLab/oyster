import os
import pathlib
import numpy as np
from casadi import external
from pathlib import Path
from py_mpcc import MPCCore
from py_mpcc import vec_VecXd
from py_mpcc import OccupancyGrid
from py_mpcc import MPCType as Dynamics
from py_mpcc  import MapConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CASADI_LIB_DIR = REPO_ROOT / "build" / "acados_code_gen" / "casadi"


def _unwrap(out):

    if isinstance(out, dict):
        vals = list(out.values())
        if not vals[0].is_scalar():
            return vals[0]

        return float(vals[0]) if len(vals) == 1 else np.concatenate(
            [np.asarray(v).ravel() for v in vals]
        )

def load_casadi_functions(mpc_type):
    fn_names = [
        "xr", 
        "yr",
        "xr_dot",
        "yr_dot",
        "phi_r",
        "e_c",
        "e_l",
        "signed_d",
        "p_abv",
        "p_blw",
        "d_abv",
        "d_blw", 
        "h_abv",
        "h_blw", 
        "Lfh_abv",
        "Lfh_blw",
        "Lghu_abv",
        "Lghu_blw",
        "Lfv",
        "Lgv",
        "Lgvu",
        "lyap_const",
    ]

    so_file = "mpcc_casadi_double_integrator_internals.so"
    if mpc_type == Dynamics.UNICYCLE:
        so_file = "mpcc_casadi_unicycle_internals.so"

    so_path = os.path.join(CASADI_LIB_DIR, so_file)
    print("SO_PATH", so_path)
    fns = {}
    for name in fn_names:
        f = external(name, so_path)
        # print(name, external(name, so_path).name_in())
        # fns[name] = (lambda **args: _unwrap(external(name, so_path)(**args)))
        fns[name] = (lambda f=f: (lambda **args: _unwrap(f(**args))))()

    return fns

class RobotMPC:

    def __init__(self, init_pos, params):

        self.dyn_model = params["DYNAMIC_MODEL"]
        print("dynamic model: ", self.dyn_model)


        # robot state: x, y, vx, vy
        self.robot_state = np.zeros(4, dtype=np.float64)
        self.robot_state[:2] = init_pos[:2]

        if self.dyn_model == Dynamics.UNICYCLE:
            self.robot_state[2] = init_pos[2]

        self.dt = params["DT"]
        self.v_max = params["LINVEL"]
        self.ref_len = params["REF_LENGTH"]

        self.prev_s = 0.0

        # self.mpc = MPCCore(Dynamics.DOUBLE_INTEGRATOR)
        self.mpc = MPCCore(self.dyn_model)
        self.mpc.load_params(params)
        self.params = self.mpc.get_params()

        self.debug_fns = load_casadi_functions(self.dyn_model)

    def set_trajectory(self, traj_x, traj_y, knots):
        self.knots = knots
        self.traj_x = traj_x
        self.traj_y = traj_y

        self.mpc.set_trajectory(self.traj_x, self.traj_y, self.knots)

    def set_occ_map(self, occupancy_grid):
        config = MapConfig()
        config.width = occupancy_grid.info.width
        config.height = occupancy_grid.info.height
        config.resolution = occupancy_grid.info.resolution
        config.origin = occupancy_grid.info.origin.position[:2]
        config.occupied_values = np.array([253, 254])
        config.no_information_values = np.array([255])
        self.mpc.set_map(config, np.array(occupancy_grid.data))
        # self.map_util = OccupancyGrid(config, np.array(occupancy_grid.data))
        # self.map_util = OccupancyGrid(occupancy_grid.info.width, occupancy_grid.info.height, occupancy_grid.info.resolution, float(occupancy_grid.info.origin.position[0]), float(occupancy_grid.info.origin.position[1]), np.array(occupancy_grid.data), np.array([253, 254]), np.array([255]))

    def load_params(self, params):
        self.mpc.load_params(params)
        self.params = self.mpc.get_params()

    def get_control(self, len_start):

        # u = [0, 0]
        # if len_start <= self.knots[-1] - 1e-2:
        #     s_dot = min(
        #         max((len_start - self.prev_s) / self.dt, 0),
        #         np.sqrt(2 * self.v_max**2),
        #     )
        #     self.prev_s = len_start
        #
        #     state = np.concatenate((self.robot_state, np.array([0, s_dot])))
        #     # if self.dyn_model == Dynamics.UNICYCLE:
        #     #     v = self.robot_state[3]
        #     #     state[2] = v * np.cos(self.robot_state[2])
        #     #     state[3] = v * np.sin(self.robot_state[2])
        #
        #     # if self.dyn_model == Dynamics.DOUBLE_INTEGRATOR:
        #     #     v = np.linalg.norm(state[2:4])
        #     #     if v < 1e-3:
        #     #         state[2] = 1e-2
        #
        #     state[4] = max(state[4], 1e-6)
        #     print("solve state", state)
        #     u = self.mpc.solve(state, False)
        # else:
        #     print("[RobotMPC] start length exceeds maximum length")
        u = self.mpc.solve(self.robot_state, False)

        return u

    def get_trajectory(self):
        return self.mpc.get_trajectory()

    def apply_control(self, u):

        if self.dyn_model == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_state[2] = u[0]
            self.robot_state[3] = u[1]

            self.robot_state[0] += self.robot_state[2] * self.dt
            self.robot_state[1] += self.robot_state[3] * self.dt

        elif self.dyn_model == Dynamics.UNICYCLE:
            # u_uni = self._di_to_uni_cmd_mapper(self.robot_state, u)
            u_uni = u
            # u_uni[0] = max(min(u_uni[0], self.v_max), -self.v_max)
            # print("mapped u:", u_uni)

            self.robot_state[0] += u_uni[0] * np.cos(self.robot_state[2]) * self.dt
            self.robot_state[1] += u_uni[0] * np.sin(self.robot_state[2]) * self.dt
            self.robot_state[2] += u_uni[1] * self.dt
            self.robot_state[2] = np.arctan2(
                np.sin(self.robot_state[2]), np.cos(self.robot_state[2])
            )
            self.robot_state[3] = u_uni[0]

        # elif self.dyn_model == Dynamics.BICYCLE:
        #     # print("initial u:", u)
        #     u_uni = self._di_to_uni_cmd_mapper(self.robot_state, u)
        #     L = 0.5
        #     if u_uni[0] > 1e-3:
        #         delta = np.arctan2(L * u_uni[1], u_uni[0])
        #     elif u_uni[1] > 1e-2:
        #         u_uni[0] = 0.1
        #         delta = np.arctan2(L * u_uni[1], u_uni[0])
        #     else:
        #         delta = 0.0

        #     delta = np.clip(delta, -np.pi / 6, np.pi / 6)
        #     # print("mapped u:", u_uni)

        #     self.robot_state[0] += u_uni[0] * np.cos(self.robot_state[2]) * self.dt
        #     self.robot_state[1] += u_uni[0] * np.sin(self.robot_state[2]) * self.dt
        #     self.robot_state[2] += u_uni[0] * np.tan(delta) / L * self.dt
        #     self.robot_state[3] = u_uni[0]

        return self.robot_state

    def get_len_start(self):
        return self.mpc.get_s_from_pose()

    def get_robot_state(self):
        return self.robot_state

    def get_mpc_state(self):
        return self.mpc.get_state()

    def set_mpc_state(self, state):
        self.robot_state = state

    def get_acc_command(self):
        return self.mpc.get_mpc_command()

    def compute_adjusted_ref(self, s):
        return self.mpc.compute_adjusted_ref(s)

    def get_solver_status(self):
        return self.mpc.get_solver_status()

    def get_cbf_data(self, state, input, is_abv):
        return self.mpc.get_cbf_data(state, input, is_abv)

    def get_params(self):
        return self.mpc.get_params()

    def get_s_from_pose(self, pose):
        return self.mpc.get_s_from_pose(pose)

    def get_state_limits(self):
        return self.mpc.get_state_limits()

    def get_input_limits(self):
        return self.mpc.get_input_limits()

    def get_horizon(self):
        return self.mpc.get_horizon()

    def get_state_from_horizon(self, ind):
        horizon = self.mpc.get_horizon()
        # state = [
        #     horizon.states.xs[ind],
        #     horizon.states.ys[ind],
        #     horizon.states.vs_x[ind],
        #     horizon.states.vs_y[ind],
        #     horizon.states.arclens[ind],
        #     horizon.states.arclens_dot[ind],
        # ]

        return horizon.get_state_at_step(ind)


    def get_input_from_horizon(self, ind):
        horizon = self.mpc.get_horizon()
        # input_ = [
        #     horizon.inputs.accs_x[ind],
        #     horizon.inputs.accs_y[ind],
        #     horizon.inputs.arclens_ddot[ind],
        # ]

        return horizon.get_input_at_step(ind)

    def get_tube(self):
        return self.mpc.get_tube()


    def _di_to_uni_cmd_mapper(self, state, u, kp=10.0):

        params = self.get_params()
        v_max = params["LINVEL"]
        w_max = params["ANGVEL"]

        theta_v = np.arctan2(u[1], u[0])
        error = theta_v - state[2]

        # bound to -pi and pi
        error = np.arctan2(np.sin(error), np.cos(error))

        u_new = [0.0, 0.0]

        # if error is too high, turn in place (20 degrees threshold)
        v = np.linalg.norm(u)
        if np.abs(error) > np.pi / 9:
            u_new[0] = 0
        else:
            u_new[0] = v

        if v > 1e-2:
            u_new[1] = kp * error

        u_new[0] = np.clip(u_new[0], -v_max, v_max)
        u_new[1] = np.clip(u_new[1], -w_max, w_max)

        return u_new
