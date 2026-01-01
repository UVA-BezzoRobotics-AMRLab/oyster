import gym
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces

def normalize_alpha(alpha, alpha_min, alpha_max):
    return 2.0 * (alpha - alpha_min) / (alpha_max - alpha_min) - 1.0

class TwoAlphaBoundsEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.dt = 0.1

        self.min_alpha = .05
        self.max_alpha = 5

        self.rl_min_alpha = -1.0
        self.rl_max_alpha = 7.5

        self.norm_min_alpha = normalize_alpha(self.min_alpha, self.rl_min_alpha, self.rl_max_alpha)
        self.norm_max_alpha = normalize_alpha(self.max_alpha, self.rl_min_alpha, self.rl_max_alpha)

        self.min_alpha_dot = -3
        self.max_alpha_dot = 3

        self.horizon = 200
        self.step_count = 0

        # state: [alpha_abv, alpha_blw]
        self.observation_space = spaces.Box(
            low=np.array([self.rl_min_alpha, self.rl_min_alpha]),
            high=np.array([self.rl_max_alpha, self.rl_max_alpha]),
            dtype=np.float32,
        )

        # render state
        self._fig = None
        self._ax = None
        self._lines = None
        self._history = []

        self.reward = 0

        # action: normalized alpha_dot
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.alpha = None
        self._goal = None

    def get_all_task_idx(self):
        return [i for i in range(10)]

    def reset_task(self, num):
        return self.reset()

    def set_epoch(self, epoch):
        pass

    def reset(self):
        self.step_count = 0
        self.reward = 0

        # randomize initial alphas inside bounds
        self.alpha = np.random.uniform(
            self.min_alpha,
            self.max_alpha,
            size=(2,),
        )

        self._history = []

        return self.alpha.astype(np.float32)

    def render(self):
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots()
            self._lines = [
                self._ax.plot([], [], label="alpha_1")[0],
                self._ax.plot([], [], label="alpha_2")[0],
            ]
            self._ax.legend()
            self._ax.set_xlabel("t")
            self._ax.set_ylabel("alpha")
            self._ax.set_ylim(-10, 10)

        data = np.array(self._history)
        t = np.arange(len(data))

        for i, line in enumerate(self._lines):
            line.set_data(t, data[:, i])

        self._ax.set_xlim(0, max(50, len(data)))
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def step(self, action):
        self.step_count += 1

        # unnormalize actions (EXACTLY like your env)
        alpha_dot = (
            self.min_alpha_dot
            + 0.5 * (action + 1.0)
            * (self.max_alpha_dot - self.min_alpha_dot)
        )

        # integrator
        self.alpha += alpha_dot * self.dt
        self._history.append(self.alpha.copy())

        obs = normalize_alpha(self.alpha, self.rl_min_alpha, self.rl_max_alpha)

        # distance to nearest bound (per alpha)
        d = np.minimum(
            obs - self.norm_min_alpha,
            self.norm_max_alpha - obs,
        )

        # ---------------- reward ----------------
        # interior reward (maximize margin)
        reward = float(np.sum(d))

        # smooth penalty when violated
        violation = d < 0
        if np.any(violation):
            reward -= .1 * np.sum((-d[violation]) ** 2)


        done = self.step_count >= self.horizon

        if np.any(obs < -1.0) or np.any(obs > 1.0):
            reward = -5.0
            done = True
            obs = obs.clip(-1, 1)

        self.reward += reward

        return obs, reward, done, {}

