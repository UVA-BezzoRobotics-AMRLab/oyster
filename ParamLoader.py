import yaml


class ParameterLoader:
    def __init__(self, yaml_files):
        """
        yaml_files: list of paths to YAML files
        """
        self.param_dicts = []

        # sort yaml file by name to ensure consistent order
        yaml_files = sorted(yaml_files)
        for f in yaml_files:
            with open(f, "r") as stream:
                try:
                    params = yaml.safe_load(stream)
                    self._add_to_list(params)
                except yaml.YAMLError as e:
                    raise RuntimeError(f"Error loading {f}: {e}")

    def _add_to_list(self, params):
        ps = {
            "DT": 1.0 / params["controller_frequency"],
            "STEPS": params["mpc_steps"],
            "ANGVEL": params["max_angvel"],
            "LINVEL": params["max_linvel"],
            "MAX_LINACC": params["max_linacc"],
            "MAX_ANGA": params["max_angacc"],
            "BOUND": 1e3,
            "W_ANGVEL": params["w_angvel"],
            "W_DANGVEL": params["w_angvel_d"],
            "W_DA": 0.1,
            "W_LAG": params["w_lag_e"],
            "W_CONTOUR": params["w_contour_e"],
            "W_SPEED": params["w_speed"],
            "REF_LENGTH": params["ref_length_size"],
            "REF_SAMPLES": params["mpc_ref_samples"],
            "CLF_GAMMA": params["clf_gamma"],
            "CLF_W_LAG": params["w_lyap_lag_e"],
            "USE_CBF": params["use_cbf"],
            "CBF_ALPHA_ABV": params["cbf_alpha_abv"],
            "CBF_ALPHA_BLW": params["cbf_alpha_blw"],
            "CBF_COLINEAR": params["cbf_colinear"],
            "CBF_PADDING": params["cbf_padding"],
            "DYNAMIC_MODEL": params["mpc_input_type"],
            "MIN_ALPHA": params["min_alpha"],
            "MAX_ALPHA": params["max_alpha"],
            "MIN_ALPHA_DOT": params["min_alpha_dot"],
            "MAX_ALPHA_DOT": params["max_alpha_dot"],
            "MIN_H_VAL": params["min_h_val"],
            "MAX_H_VAL": params["max_h_val"],
        }

        self.param_dicts.append(ps)

    def __len__(self):
        return len(self.param_dicts)

    def __getitem__(self, idx):
        """Allow bracket access like loader[0]."""
        return self.param_dicts[idx]

    def get_params(self, idx):
        """Explicit method to get params by index."""
        if idx < 0 or idx >= len(self.param_dicts):
            raise IndexError(
                f"Index {idx} out of range (have {len(self.param_dicts)})."
            )
        return self.param_dicts[idx]
