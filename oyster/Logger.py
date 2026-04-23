import json
import numpy as np

class Logger:
    def __init__(self, filename="sim_data.json"):
        self.filename = filename
        self.data = {
            "static_obstacles": [], # List of [x, y, radius]
            "frames": []
        }

    def _ensure_list(self, data):
        """Recursively converts numpy arrays/tuples to Python lists for JSON."""
        if isinstance(data, (np.ndarray, np.generic)):
            return data.tolist()
        if isinstance(data, (list, tuple)):
            return [self._ensure_list(item) for item in data]
        return data

    def log_static_obstacles(self, obstacle_list):
        """
        Takes a list of triples: [(x1, y1, r1), (x2, y2, r2), ...]
        """
        self.data["static_obstacles"] = self._ensure_list(obstacle_list)

    def log_frame(self, robot_pos, trajectory_tuple, tube_top, tube_bottom):
        """
        robot_pos: [x, y] or [x, y, z]
        trajectory_tuple: (knots, xs, ys)
        tube_top/bottom: Lists of coefficients or evaluated points
        """
        knots, xs, ys = trajectory_tuple
        
        frame = {
            "robot_pos": self._ensure_list(robot_pos),
            "trajectory": {
                "knots": self._ensure_list(knots),
                "xs": self._ensure_list(xs),
                "ys": self._ensure_list(ys)
            },
            "tubes": {
                "top": self._ensure_list(tube_top),
                "bottom": self._ensure_list(tube_bottom)
            }
        }
        self.data["frames"].append(frame)

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Data saved. Total frames: {len(self.data['frames'])}")
