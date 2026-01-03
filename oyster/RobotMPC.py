import os
import pathlib
import numpy as np
import casadi as ca

from pathlib import Path
from py_mpcc import MPCCore
from py_mpcc import get_tubes
from py_mpcc import vec_VecXd
from py_mpcc import OccupancyGrid
from py_mpcc import get_cbf_abv
from py_mpcc import MPCType as Dynamics


class RobotMPC:

    def __init__(self, init_pos, params):

        self.dyn_model = params["DYNAMIC_MODEL"]
        print("dynamic model: ", self.dyn_model)

        self.map_util = None

        # robot state: x, y, vx, vy
        self.robot_state = np.zeros(4, dtype=np.float64)
        self.robot_state[:2] = init_pos[:2]

        if self.dyn_model == Dynamics.UNICYCLE:
            print(init_pos[2])
            self.robot_state[2] = init_pos[2]

        self.dt = params["DT"]
        self.v_max = params["LINVEL"]
        self.ref_len = params["REF_LENGTH"]

        self.prev_s = 0.0

        self.mpc = MPCCore(Dynamics.DOUBLE_INTEGRATOR)
        self.mpc.load_params(params)
        self.params = self.mpc.get_params()

        script_dir = Path(__file__).parent.absolute()
        og_dir = os.getcwd()
        cpp_dir = os.path.join(script_dir, "cpp")
        os.chdir(cpp_dir)
        cpp_files = pathlib.Path(cpp_dir).glob('*.cpp')
        so_files = pathlib.Path(cpp_dir).glob('*.so')

        if len(list(so_files)) == 0:
            # assuming one . in fname
            for file in cpp_files:
                fname = str(file).split(".")[0]
                os.system(f"gcc -fPIC -shared {fname}.cpp -o {fname}.so")

        self.get_cbf_abv = ca.external('h_abv', os.path.join(cpp_dir, "./compute_cbf_abv.so"))
        self.get_lfh_abv = ca.external('lfh_abv', os.path.join(cpp_dir, "./compute_lfh_abv.so"))
        self.get_lgh_abv = ca.external('lgh_abv', os.path.join(cpp_dir, "./compute_lgh_abv.so"))

        self.get_cbf_blw = ca.external('h_blw', os.path.join(cpp_dir, "./compute_cbf_blw.so"))
        self.get_lfh_blw = ca.external('lfh_blw', os.path.join(cpp_dir, "./compute_lfh_blw.so"))
        self.get_lgh_blw = ca.external('lgh_blw', os.path.join(cpp_dir, "./compute_lgh_blw.so"))

        os.chdir(og_dir)

    def set_trajectory(self, traj_x, traj_y, knots):
        self.knots = knots
        self.traj_x = traj_x
        self.traj_y = traj_y

        self.mpc.set_trajectory(self.traj_x, self.traj_y, 3, self.knots)

    def set_occ_map(self, occupancy_grid):
        self.map_util = OccupancyGrid(occupancy_grid.info.width, occupancy_grid.info.height, occupancy_grid.info.resolution, float(occupancy_grid.info.origin.position[0]), float(occupancy_grid.info.origin.position[1]), np.array(occupancy_grid.data), np.array([253, 254]), np.array([255]))

    def set_tubes(self, upper_coeffs, lower_coeffs):

        if bool(self.mpc.get_params()["USE_CBF"]) is False:
            upper_coeffs = np.zeros(7)
            upper_coeffs[0] = 100
            lower_coeffs = np.zeros(7)
            lower_coeffs[0] = 100

        self.mpc.set_tubes([upper_coeffs, lower_coeffs])

    def load_params(self, params):
        self.mpc.load_params(params)
        self.params = self.mpc.get_params()

    def get_control(self, len_start):

        u = [0, 0]
        if len_start <= self.knots[-1] - 1e-2:
            s_dot = min(
                max((len_start - self.prev_s) / self.dt, 0),
                np.sqrt(2 * self.v_max**2),
            )
            self.prev_s = len_start

            state = np.concatenate((self.robot_state, np.array([0, s_dot])))
            if self.dyn_model == Dynamics.UNICYCLE:
                v = self.robot_state[3]
                state[2] = v * np.cos(self.robot_state[2])
                state[3] = v * np.sin(self.robot_state[2])

            # if self.dyn_model == Dynamics.DOUBLE_INTEGRATOR:
            #     v = np.linalg.norm(state[2:4])
            #     if v < 1e-3:
            #         # print("velocity really small, clipping")
            #         state[2] = 1e-2

            u = self.mpc.solve(state, False)
        else:
            print("[RobotMPC] start length exceeds maximum length")

        u[0] = max(min(u[0], self.v_max), -self.v_max)
        u[1] = max(min(u[1], self.v_max), -self.v_max)

        return u

    def apply_control(self, u):

        if self.dyn_model == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_state[2] = u[0]
            self.robot_state[3] = u[1]

            self.robot_state[0] += self.robot_state[2] * self.dt
            self.robot_state[1] += self.robot_state[3] * self.dt

        elif self.dyn_model == Dynamics.UNICYCLE:
            # print("initial u:", u)
            u_uni = self._di_to_uni_cmd_mapper(self.robot_state, u)
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

    def get_tubes(self, d, N, max_dist, xs, ys, degree, knots, len_start, horizon):
        tubes = vec_VecXd()
        get_tubes(int(d), N, max_dist, xs, ys, degree, knots, knots[-1],
                  len_start, horizon, self.map_util, tubes)
        return tubes

    # def get_cbf_abv(self, x, d_abv_coeff, x_coeff, y_coeff):
    #     return get_cbf_abv(x, d_abv_coeff, x_coeff, y_coeff)
        

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
