import yaml
import numpy as np

from py_mpcc import MPCConfig
from oyster.RobotMPC import Dynamics


class ParameterLoader:
    def __init__(self, yaml_files):
        """
        yaml_files: list of paths to YAML files
        """
        self.param_structs = []

        # sort yaml file by name to ensure consistent order
        yaml_files = sorted(yaml_files)
        for f in yaml_files:
            with open(f, "r") as stream:
                try:
                    params = yaml.safe_load(stream)
                    self._add_to_list(params)
                except yaml.YAMLError as e:
                    raise RuntimeError(f"Error loading {f}: {e}")

    def get(self, params, key, default):
        if key in params:
            return params[key]

        return default

    def _add_to_list(self, params):
        cfg = MPCConfig()

        # Core MPC params
        cfg.steps = self.get(params, "mpc_steps", 10)
        cfg.dt = 1.0 / self.get(params, "controller_frequency", 10)
        cfg.ref_samples = self.get(params, "mpc_ref_samples", 100)
        cfg.input_type = Dynamics(self.get(params, "mpc_input_type", 1))

        # Cost weights
        cfg.weights.w_vel = self.get(params, "w_vel", 1.0)
        cfg.weights.w_angvel = self.get(params, "w_angvel", 1.0)
        cfg.weights.w_linvel = self.get(params, "w_linvel", 1.0)
        cfg.weights.w_angvel_d = self.get(params, "w_angvel_d", 1.0)
        cfg.weights.w_linvel_d = self.get(params, "w_linvel_d", 0.5)
        cfg.weights.w_etheta = self.get(params, "w_etheta", 0.5)
        cfg.weights.w_cte = self.get(params, "w_cte", 1.0)
        cfg.weights.w_lag_e = self.get(params, "w_lag_e", 50.0)
        cfg.weights.w_contour_e = self.get(params, "w_contour_e", 0.1)
        cfg.weights.w_speed = self.get(params, "w_speed", 0.3)

        # Constraints
        cfg.constraints.max_angvel = self.get(params, "max_angvel", 3.0)
        cfg.constraints.max_linvel = self.get(params, "max_linvel", 2.0)
        cfg.constraints.max_linacc = self.get(params, "max_linacc", 3.0)
        cfg.constraints.max_angacc = self.get(params, "max_angacc", 2 * np.pi)
        cfg.constraints.bound_value = self.get(params, "bound_value", 1e19)

        # CBF
        cfg.cbf.use_cbf = self.get(params, "use_cbf", False)
        cfg.cbf.alpha_abv = self.get(params, "cbf_alpha_abv", 0.5)
        cfg.cbf.alpha_blw = self.get(params, "cbf_alpha_blw", 0.5)
        cfg.cbf.colinear = self.get(params, "cbf_colinear", 0.1)
        cfg.cbf.padding = self.get(params, "cbf_padding", 0.1)
        cfg.cbf.dynamic_alpha = self.get(params, "dynamic_alpha", False)
        cfg.cbf.min_alpha = self.get(params, "min_alpha", 0.1)
        cfg.cbf.max_alpha = self.get(params, "max_alpha", 5.0)
        cfg.cbf.min_alpha_dot = self.get(params, "min_alpha_dot", -3.0)
        cfg.cbf.max_alpha_dot = self.get(params, "max_alpha_dot", 3.0)
        cfg.cbf.min_h_val = self.get(params, "min_h_val", -100.0)
        cfg.cbf.max_h_val = self.get(params, "max_h_val", 100.0)

        # CLF
        cfg.clf.w_lag_e = self.get(params, "w_lyap_lag_e", 1.0)
        cfg.clf.w_contour_e = self.get(params, "w_lyap_contour_e", 1.0)
        cfg.clf.gamma = self.get(params, "clf_gamma", 0.5)

        # Prop controller params
        cfg.prop.gain = self.get(params, "prop_gain", 0.5)
        cfg.prop.gain_thresh = self.get(
            params, "prop_gain_thresh", 30.0 * np.pi / 180.0
        )

        # Tube Generation (for CBF)
        cfg.tube.poly_degree = self.get(params, "tube_poly_degree", 6)
        cfg.tube.num_samples = self.get(params, "tube_num_samples", 50)
        cfg.tube.max_width = self.get(params, "max_tube_width", 2.0)

        cfg.cbf.min_alpha = 0.5
        cfg.cbf.max_alpha = 6.0

        self.param_structs.append(cfg)

    def __len__(self):
        return len(self.param_structs)

    def __getitem__(self, idx):
        """Allow bracket access like loader[0]."""
        return self.param_structs[idx]

    def get_params(self, idx):
        """Explicit method to get params by index."""
        if idx < 0 or idx >= len(self.param_structs):
            raise IndexError(
                f"Index {idx} out of range (have {len(self.param_structs)})."
            )
        return self.param_structs[idx]
