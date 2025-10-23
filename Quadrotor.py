import numpy as np
import matplotlib.pyplot as plt

class QuadState:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.roll = 0
        self.pitch = 0
        self.p = 0
        self.q = 0

class Quadrotor2D:

    def __init__(self, dt):
        self.dt = dt

        self.state = QuadState()
        self.reference = [0., 0.]
        self.mass = 4.34
        self.gravity = 9.81

        self.max_roll = np.pi/6.
        self.max_pitch = np.pi/6.

    def get_acc(self):
        if np.abs(np.cos(self.state.roll)) < 1e-3 or \
                np.abs(np.cos(self.state.pitch)) < 1e-3:
                print("somehow the pitch has gotten way too large")
                exit(0)

        ax = self.gravity * np.tan(self.state.pitch)
        ay = -self.gravity * np.tan(self.state.roll) * 1 / (np.cos(self.state.pitch))

        return ax, ay

    def step(self, body_rates):
        p, q = body_rates

        self.state.roll += p * self.dt
        self.state.pitch += q * self.dt

        self.state.roll = np.clip(self.state.roll, -self.max_roll, self.max_roll)
        self.state.pitch = np.clip(self.state.pitch, -self.max_pitch, self.max_pitch)

        ax, ay = self.get_acc()

        self.state.x += self.state.vx * self.dt
        self.state.y += self.state.vy * self.dt
        self.state.vx += ax * self.dt
        self.state.vy += ay * self.dt

        self.state.p = p
        self.state.q = q

        return self.state

    # assuming yaw is 0
    def bz_in_world(self, roll, pitch):
        R13 = np.sin(pitch)*np.cos(roll)
        R23 = -np.sin(roll)
        R33 = np.cos(pitch) * np.cos(roll)

        return np.array([R13, R23, R33])

    def compute_thrust_vec(self):
        thrust = self.mass * 9.81 / (np.cos(self.state.roll) * np.cos(self.state.pitch))
        rot = self.bz_in_world(self.state.roll, self.state.pitch)
        return rot * thrust

class AttitudePDController:

    def __init__(self, kp_roll=100.0, kd_roll=0, kp_pitch=10.0, kd_pitch=0.0, 
                 dt=0.01, max_rate=np.radians(90)):
        self.kp_roll = kp_roll
        self.kd_roll = kd_roll

        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch

        self.dt = dt
        self.prev_roll_err = None
        self.prev_pitch_err = None
        self.max_rate = max_rate

    def reset(self):
        self.prev_roll_err = 0.0
        self.prev_pitch_err = 0.0

    def step(self, roll_d, pitch_d, roll, pitch, p, q):
        # errors
        e_roll = roll_d - roll
        e_pitch = pitch_d - pitch

        # derivative (finite difference)
        if self.prev_roll_err is not None:
            # de_roll = (e_roll - self.prev_roll_err) / self.dt
            # de_pitch = (e_pitch - self.prev_pitch_err) / self.dt
            de_roll = -p
            de_pitch = -q
        else:
            de_roll = 0
            de_pitch = 0

        # PD -> commanded rates
        u_roll = self.kp_roll * e_roll + self.kd_roll * de_roll
        u_pitch = self.kp_pitch * e_pitch + self.kd_pitch * de_pitch
        print("kp:", round(self.kp_roll * e_roll, 3),"\tkd:",self.kd_roll* de_roll)
        # print("kp:", round(self.kp_pitch * e_pitch, 3),"\tkd:",self.kd_pitch * de_pitch)

        # clip
        u_roll = float(np.clip(u_roll, -self.max_rate, self.max_rate))
        u_pitch = float(np.clip(u_pitch, -self.max_rate, self.max_rate))

        # save error history
        self.prev_roll_err = e_roll
        self.prev_pitch_err = e_pitch

        return u_roll, u_pitch


def desired_attitude_from_acc(ax_d, ay_d, g=9.81):
    pitch = np.arctan2(ax_d , g)
    roll = np.arctan2(-ay_d * np.cos(pitch) , g)

    return roll, pitch


def main():
    dt = 0.01
    sim_time = 3.0
    quad = Quadrotor2D(dt / 10.)
    ctrl = AttitudePDController(dt=dt)

    # desired accelerations
    ax_d = 0.0  # m/s^2 forward
    ay_d = 1.0  # m/s^2 right

    # desired attitude from desired accelerations
    roll_d, pitch_d = desired_attitude_from_acc(ax_d, ay_d)

    history = {
        "roll": [],
        "pitch": [],
        "roll_d": [],
        "pitch_d": [],
        "x": [],
        "y": [],
    }

    n_steps = int(sim_time / dt)
    for i in range(n_steps):
        # desired angles from acceleration commands
        phi_d, theta_d = desired_attitude_from_acc(ax_d, ay_d)

        # body rate commands from PD controller
        u_roll, u_pitch = ctrl.step(phi_d, theta_d, quad.state.roll, quad.state.pitch,
                                    quad.state.p, quad.state.q)

        # integrate dynamics
        for i in range(10):
            state = quad.step((u_roll, u_pitch))

        # save for plotting
        history["roll"].append(state.roll)
        history["pitch"].append(state.pitch)
        history["roll_d"].append(phi_d)
        history["pitch_d"].append(theta_d)
        history["x"].append(state.x)
        history["y"].append(state.y)

    t = np.arange(n_steps) * dt

    # plot roll/pitch
    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(t, history["roll"], label="roll")
    plt.plot(t, history["roll_d"], "--", label="roll_d")
    plt.ylabel("Roll [rad]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t, history["pitch"], label="pitch")
    plt.plot(t, history["pitch_d"], "--", label="pitch_d")
    plt.ylabel("Pitch [rad]")
    plt.grid(True)
    plt.legend()

    # plot x/y trajectory
    plt.subplot(3,1,3)
    plt.plot(history["x"], history["y"], label="position")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
