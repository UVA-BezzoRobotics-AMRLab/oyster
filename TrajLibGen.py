import os
import ast
import csv
import numpy as np

from tqdm import tqdm
from Plotter import Plotter
from Bezier import BezierCurve
from Tubes import TubeGenerator


class TrajLibGen:

    def __init__(self, seed = None):

        if seed != None:
            np.random.seed(seed)

        self.tube_generator = TubeGenerator()

    def create_library(self, n_traj, n_obs, min_dist, max_dist, out_dir):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        i = 0
        pbar = tqdm(total=n_traj)
        rows = []
        while i < n_traj:
            curve = self.gen_traj()
            if curve is None:
                continue

            obs = self.gen_obs(curve, n_obs, min_dist, max_dist)

            knots = (curve.knots, curve.xs, curve.ys)
            self.tube_generator.set_curve(knots)
            self.tube_generator.set_obstacles(obs)

            # try to generate corridor and if it fails, skip
            try:
                upper_coeffs, lower_coeffs = self.tube_generator.generate_corridor()

                if upper_coeffs is None or lower_coeffs is None:
                    continue

            except Exception as e:
                print(e)
                continue

            fig, ax = plt.subplots()


            plotter = Plotter(curve, upper_coeffs, lower_coeffs, obs, render=False)
            plotter.init_curve(curve, ax=ax)
            plotter.init_tubes(curve, upper_coeffs, lower_coeffs, ax=ax)
            plotter.init_obs(obs, ax=ax)

            fname = f"traj_{i:04d}.jpg"
            fig.savefig(os.path.join(out_dir, fname))

            np.savez_compressed(
                os.path.join(out_dir, fname),
                p0=curve.p0,
                p1=curve.p1,
                p2=curve.p2,
                p3=curve.p3,
                obs=obs,
            )

            plt.close(fig)

            i += 1
            pbar.update(1)

    def gen_traj(self):
        r = 10

        # random goal point
        p0 = np.array([0,0])
        theta = 2 * np.pi * np.random.rand()
        p3 = p0 + r * np.array([np.cos(theta), np.sin(theta)])

        # random p1 p2
        k = 1e6
        i = 0
        while k > 2 and i < 100:
            p1 = r * np.random.rand(1,2)
            p2 = r * np.random.rand(1,2)

            curve = BezierCurve(p0, p1, p2, p3)
            k = curve.get_max_k_appx()
            
            i += 1

        if i >= 100:
            return None

        return curve


    # horrendous naming scheme...
    def gen_obs(self, curve, n_obs, d_min, d_max, min_dist=0.6):
        # obs = [[6.12, 2.3], [2.75, 4.45]]
        obs = []

        # get initial obstacles
        n = np.random.randint(0,4) + 1
        ss = []
        for i in range(0, n):

            s = np.random.rand() * (curve.get_arclen() - 2) + 1
            while len(ss) > 0 and np.abs(np.min(np.array(ss) - s)) < 1:
                s = np.random.rand() * (curve.get_arclen() - 2) + 1
            
            d = np.random.rand() * (d_max-d_min) + d_min
            d = -d if np.random.rand() > 0.5 else d
            pos = curve.pos(s)
            tan = curve.vel(s)
            tan = tan / np.linalg.norm(tan)
            normal = np.array([-tan[1],tan[0]])

            obs.append((pos + normal * d).tolist())

        # obs = [[7.54, 9.32]]
        needed = n_obs

        # get trajectory points
        traj = curve.fill(np.linspace(0,1,100))

        while len(obs) < n_obs:
            p_min = min(np.min(curve.xs), np.min(curve.ys))
            p_max = max(np.max(curve.xs), np.max(curve.ys))

            x_min, y_min = p_min, p_min
            x_max, y_max = p_max, p_max

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

        return np.array(obs[:n_obs])


class TrajLibLoader:
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])
        self.n = len(self.files)

    def __len__(self):
        return self.n

    def get(self, idx):
        """Return trajectory and obstacles for a given index."""
        path = os.path.join(self.folder, self.files[idx])
        data = np.load(path)
        curve = BezierCurve(data["p0"], data["p1"], data["p2"], data["p3"])
        return {
            "traj_id": idx,
            "curve": curve,
            "obs_points": data["obs"],
        }

    def sample(self):
        """Randomly sample one trajectory."""
        idx = np.random.randint(self.n)
        return self.get(idx)


if __name__ == "__main__":

    if not os.path.exists("./envs"):
        os.makedirs("./envs")

    out_dir = "./envs"
    libgen = TrajLibGen()
    libgen.create_library(100, 150, 0.1, 0.3, out_dir)
