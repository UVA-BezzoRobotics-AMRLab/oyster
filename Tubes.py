import copy
import numpy as np
import cvxpy as cp

from scipy.interpolate import UnivariateSpline


class TubeGenerator:
    def __init__(self, obstacles=None, curve=None, tube_degree=6):
        self.degree = tube_degree

        if curve is not None:
            self.set_curve(curve)

        if obstacles is not None:
            self.set_obstacles(obstacles)

    def set_curve(self, curve):
        knots, xs, ys = curve
        self.curve_len = knots[-1]
        self.curve_x = UnivariateSpline(knots, xs, k=3, s=0)
        self.curve_y = UnivariateSpline(knots, ys, k=3, s=0)

        self.curve_xd = self.curve_x.derivative(n=1)
        self.curve_yd = self.curve_y.derivative(n=1)

        self.curve = lambda s: np.stack((self.curve_x(s), self.curve_y(s)), axis=-1)
        self.curve_d = lambda s: np.stack((self.curve_xd(s), self.curve_yd(s)), axis=-1)

        self.coeffs_p = None
        self.coeffs_n = None

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def get_dists(self):
        ss = np.linspace(0, self.curve_len, 100)
        traj = self.curve(ss)
        diff = self.obstacles[:, None, :] - traj[None, :, :]

        # creates array of shape (n_obs, n_traj)
        dists = np.sum(diff**2, axis=-1)
        min_inds = np.argmin(dists, axis=-1)

        # perp distances are obs - traj projected onto traj normal
        tangents = self.curve_d(ss[min_inds])
        tangents = tangents / np.linalg.norm(tangents, axis=-1, keepdims=True)
        normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=-1)
        perp_dists = np.sum((self.obstacles - traj[min_inds]) * normals, axis=-1)

        # split up depending on side of curve
        d_parallels = ss[min_inds]
        d_parallel_p = d_parallels[perp_dists >= 0]
        d_parallel_n = d_parallels[perp_dists < 0]

        perp_dists_p = perp_dists[perp_dists >= 0]
        perp_dists_n = np.abs(perp_dists[perp_dists < 0])

        return d_parallel_p, perp_dists_p, d_parallel_n, perp_dists_n

    def generate_corridor(self):
        d_parallel_p, perp_dists_p, d_parallel_n, perp_dists_n = self.get_dists()

        dp_min = np.min(perp_dists_p)
        dp_max = np.max(perp_dists_p)
        dn_min = np.min(perp_dists_n)
        dn_max = np.max(perp_dists_n)

        # solve LP
        prob_p, x_p = self.corridor_LP(d_parallel_p, perp_dists_p, dp_min, dp_max)
        prob_n, x_n = self.corridor_LP(d_parallel_n, perp_dists_n, dn_min, dn_max)

        prob_p.solve(solver=cp.CLARABEL)
        prob_n.solve(solver=cp.CLARABEL)

        if prob_p.status != 'optimal' or prob_n.status != 'optimal':
            return None, None

        self.coeffs_p = x_p.value
        self.coeffs_n = x_n.value

        return self.coeffs_p, self.coeffs_n

    def corridor_LP(self, d_parallel, d_perp, d_min, d_max):
        n_constraints = d_parallel.shape[0]
        n_vars = self.degree + 1

        # decision vars
        x = cp.Variable(n_vars)

        cost = 0
        for k in range(n_vars):
            cost += x[k] * (self.curve_len ** (k + 1)) / (k + 1)

        # constraints
        constraints = []

        n_sweep = 100
        xi_sweep = np.linspace(0, self.curve_len, n_sweep)
        for i in range(n_sweep):
            xi = xi_sweep[i]
            poly = 0
            for k in range(n_vars):
                poly += x[k] * (xi**k)

            constraints.append(poly >= d_min)
            constraints.append(poly <= d_max)

        for i in range(n_constraints):
            di = d_parallel[i]
            dpi = d_perp[i]
            poly = 0
            for k in range(n_vars):
                poly += x[k] * (di**k)
            constraints.append(poly <= dpi)

        prob = cp.Problem(cp.Minimize(-cost), constraints)
        return prob, x

    def shift_poly_parameter(self, s0, ref_len):

        if self.coeffs_p is None or self.coeffs_n is None:
            raise ValueError("Must generate corridor before shifting")

        tau = np.linspace(0, ref_len, 100)
        s = s0 + tau

        vals_x = np.polyval(self.coeffs_p[::-1], s)
        vals_y = np.polyval(self.coeffs_n[::-1], s)

        shifted_coeffs_x = np.polyfit(tau, vals_x, self.degree)
        shifted_coeffs_y = np.polyfit(tau, vals_y, self.degree)

        # need to reverse because np.polyfit order is opposit of our LP
        return shifted_coeffs_x[::-1], shifted_coeffs_y[::-1]
