import json
import numpy as np

from oyster.RobotMPC import RobotMPC, Dynamics


def _ensure_list(data):
    if isinstance(data, (np.ndarray, np.generic)):
        return data.tolist()
    if isinstance(data, (list, tuple)):
        return [_ensure_list(item) for item in data]
    return data


class Logger:
    def __init__(self, filename="sim_data.json"):
        self.filename = filename
        self.data = {
            "static_obstacles": [],  # List of [x, y, radius]
            "world_nums": [],
            "start_pos": [],
            "goal_pos": [],
            "frames": [],
            "metadata": {
                "model": "",
                "max_speed": "",
            },
        }

    def log_world_nums(self, nums):
        self.data["world_nums"] = _ensure_list(nums)

    def log_meta_data(self, params):
        model_str = (
            "Unicycle"
            if params.input_type == Dynamics.UNICYCLE
            else "Double Integrator"
        )
        self.data["metadata"]["model"] = model_str
        self.data["metadata"]["max_speed"] = params.constraints.max_linvel

    def log_static_obstacles(self, obstacle_list):
        self.data["static_obstacles"] = _ensure_list(obstacle_list)

    def log_start_and_goal(self, start_pvaj, goal_pvaj):
        # start, goal coming in as pos, vel, acc, jerk
        self.data["start_pos"] = _ensure_list(start_pvaj[:2, 0])
        self.data["goal_pos"] = _ensure_list(goal_pvaj[:2, 0])

    @staticmethod
    def generate_json_frame(
        robot_pos,
        velocity,
        obs,
        mpc_horizon,
        trajectory_tuple,
        tube_top,
        tube_bottom,
    ):
        knots, xs, ys = trajectory_tuple

        frame = {
            "robot_pos": _ensure_list(robot_pos),
            "velocity": velocity,
            "cbf_upper": obs[0],
            "cbf_lower": obs[2],
            "alpha_upper": obs[-2],
            "alpha_lower": obs[-1],
            "mpc_horizon": _ensure_list(mpc_horizon),
            "trajectory": {
                "knots": _ensure_list(knots),
                "xs": _ensure_list(xs),
                "ys": _ensure_list(ys),
            },
            "tubes": {
                "top": _ensure_list(tube_top),
                "bottom": _ensure_list(tube_bottom),
            },
        }

        return frame

    def log_frame(
        self,
        robot_pos,
        velocity,
        obs,
        mpc_horizon,
        trajectory_tuple,
        tube_top,
        tube_bottom,
    ):
        """
        robot_pos: [x, y] or [x, y, z]
        trajectory_tuple: (knots, xs, ys)
        tube_top/bottom: Lists of coefficients or evaluated points
        """
        frame = self.generate_json_frame(
            robot_pos,
            velocity,
            obs,
            mpc_horizon,
            trajectory_tuple,
            tube_top,
            tube_bottom,
        )
        self.data["frames"].append(frame)

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Data saved. Total frames: {len(self.data['frames'])}")
