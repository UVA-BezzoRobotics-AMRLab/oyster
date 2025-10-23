import numpy as np

from py_mpcc import MPCCore


class Dynamics:
    DOUBLE_INTEGRATOR = 0
    UNICYCLE = 1
    BICYCLE = 2
    int2str = ["double_integrator", "unicycle", "bicycle"]


class RobotMPC:

    def __init__(self, init_pos, params):
        # params = {
        #     "DT": 0.1,
        #     "STEPS": 10,
        #     "ANGVEL": 3.2,
        #     "LINVEL": 1.0,
        #     "MAX_LINACC": 1.5,
        #     "MAX_ANGA": 6.28,
        #     "BOUND": 1e3,
        #     "W_ANGVEL": 0.15,
        #     "W_DANGVEL": 0.7,
        #     "W_DA": 0.1,
        #     "W_LAG": 100,
        #     "W_CONTOUR": 0.8,
        #     "W_SPEED": 0.1,
        #     "REF_LENGTH": 6,
        #     "REF_SAMPLES": 11,
        #     "CLF_GAMMA": 0.5,
        #     "CLF_W_LAG": 1,
        #     "USE_CBF": False,
        #     "CBF_ALPHA_ABV": 5.0,
        #     "CBF_ALPHA_BLW": 5.0,
        #     # "USE_CBF": True,
        #     # "CBF_ALPHA_ABV": 1.05,
        #     # "CBF_ALPHA_BLW": 1.05,
        #     "CBF_COLINEAR": 0.1,
        #     "CBF_PADDING": 0.1,
        # }

        self.dyn_model = params["DYNAMIC_MODEL"]

        # robot state: x, y, vx, vy
        self.robot_state = np.zeros(4, dtype=np.float64)
        self.robot_state[:2] = init_pos[:2]

        if self.dyn_model in [Dynamics.UNICYCLE, Dynamics.BICYCLE]:
            print(init_pos[2])
            self.robot_state[2] = init_pos[2]

        self.dt = params["DT"]
        self.v_max = params["LINVEL"]
        self.ref_len = params["REF_LENGTH"]

        self.prev_s = 0.0

        self.mpc = MPCCore(Dynamics.int2str[Dynamics.DOUBLE_INTEGRATOR])
        self.mpc.load_params(params)
        self.params = self.mpc.get_params()

    def set_trajectory(self, traj_x, traj_y, knots):
        self.knots = knots
        self.traj_x = traj_x
        self.traj_y = traj_y

        self.mpc.set_trajectory(self.traj_x, self.traj_y, 3, self.knots)

    def set_tubes(self, upper_coeffs, lower_coeffs):

        if bool(self.mpc.get_params()["USE_CBF"]) is False:
            upper_coeffs = np.zeros(7)
            upper_coeffs[0] = 100
            lower_coeffs = np.zeros(7)
            lower_coeffs[0] = 100

        self.mpc.set_tubes([upper_coeffs, -lower_coeffs])

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
            if self.dyn_model in [Dynamics.UNICYCLE, Dynamics.BICYCLE]:
                v = self.robot_state[3]
                state[2] = v * np.cos(self.robot_state[2])
                state[3] = v * np.sin(self.robot_state[2])

            u = self.mpc.solve(state, False)

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
            self.robot_state[3] = u_uni[0]

        elif self.dyn_model == Dynamics.BICYCLE:
            # print("initial u:", u)
            u_uni = self._di_to_uni_cmd_mapper(self.robot_state, u)
            L = 0.5
            if u_uni[0] > 1e-3:
                delta = np.arctan2(L * u_uni[1], u_uni[0])
            elif u_uni[1] > 1e-2:
                u_uni[0] = 0.1
                delta = np.arctan2(L * u_uni[1], u_uni[0])
            else:
                delta = 0.

            delta = np.clip(delta, -np.pi / 6, np.pi / 6)
            # print("mapped u:", u_uni)

            self.robot_state[0] += u_uni[0] * np.cos(self.robot_state[2]) * self.dt
            self.robot_state[1] += u_uni[0] * np.sin(self.robot_state[2]) * self.dt
            self.robot_state[2] += u_uni[0] * np.tan(delta) / L * self.dt
            self.robot_state[3] = u_uni[0]

        return self.robot_state

    def get_len_start(self):
        return self.mpc.get_s_from_pose()

    def get_robot_state(self):
        return self.robot_state

    def get_mpc_state(self):
        return self.mpc.get_state()

    def set_mpc_state(self, state):
        self.robot_state = state

    def get_mpc_command(self):
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

    def _di_to_uni_cmd_mapper(self, state, u, kp=10.0):

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

        return u_new
