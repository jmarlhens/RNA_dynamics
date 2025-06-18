import numpy as np
import pandas as pd
from typing import Any
from utils.process_experimental_data import prepare_experimental_data


def compute_condition_likelihood(
    simulation_results: Any,
    experimental_data: pd.DataFrame,
    tspan: np.ndarray,
    combined_params: pd.DataFrame,
    condition: str,
) -> pd.Series:
    """Compute log likelihood for a specific condition (legacy version)"""
    condition_mask = combined_params["condition"] == condition
    sim_indices = combined_params.index[condition_mask]
    param_set_indices = combined_params.loc[condition_mask, "param_set_idx"]

    # Calculate experimental data (this will be cached in the new version)
    _, exp_means, exp_vars = prepare_experimental_data(experimental_data, tspan)

    sim_values = np.array(
        [simulation_results.observables[i]["obs_Protein_GFP"] for i in sim_indices]
    )
    log_likelihoods = calculate_likelihoods(sim_values, exp_means, exp_vars)
    return pd.Series(log_likelihoods, index=param_set_indices)


def calculate_likelihoods(sim_values, exp_means, exp_vars):
    residuals = sim_values - exp_means
    # I try a new version normalised by the number of sampled time points/experiments, since some really have more
    n_points = len(exp_means[0])  # number of time points
    return -0.5 * np.sum((residuals**2) / exp_vars, axis=1) / n_points
