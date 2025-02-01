import numpy as np
import pandas as pd
from typing import Any, Dict
from .utils import prepare_experimental_data
from utils.GFP_calibration import convert_nM_to_AU


def compute_condition_likelihood(
        simulation_results: Any,
        experimental_data: pd.DataFrame,
        tspan: np.ndarray,
        combined_params: pd.DataFrame,
        condition: str,
        calibration_params: Dict
) -> pd.Series:
    """Compute log likelihood for a specific condition (legacy version)"""
    condition_mask = combined_params['condition'] == condition
    sim_indices = combined_params.index[condition_mask]
    param_set_indices = combined_params.loc[condition_mask, 'param_set_idx']

    # Calculate experimental data (this will be cached in the new version)
    exp_means, exp_vars = prepare_experimental_data(experimental_data, tspan)

    sim_values = np.array([
        simulation_results.observables[i]['obs_Protein_GFP']
        for i in sim_indices
    ])

    log_likelihoods = calculate_likelihoods(sim_values, exp_means, exp_vars)
    # import matplotlib.pyplot as plt
    # # color each simulation with respect to its ll stored in log_likelihoods
    # # Get 90th percentile of log likelihoods
    # min_ll = np.percentile(log_likelihoods, 70)
    # max_ll = np.percentile(log_likelihoods, 100)
    #
    # norm = plt.Normalize(vmin=min_ll, vmax=max_ll)
    # cmap = plt.cm.viridis
    #
    # plt.figure(figsize=(10, 6))
    # for i, sim in enumerate(sim_values_au):
    #     plt.plot(sim, color=cmap(norm(log_likelihoods[i])))
    # plt.plot(exp_means[0], 'k-', label='Experimental Data', color = 'red')
    # plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Log-likelihood')
    # plt.ylim(0, 1000000)
    # plt.legend()
    # plt.show()
    return pd.Series(log_likelihoods, index=param_set_indices)


def calculate_likelihoods(
        sim_values: np.ndarray,
        exp_means: np.ndarray,
        exp_vars: np.ndarray
) -> np.ndarray:
    """Calculate log likelihoods using vectorized operations"""
    residuals = sim_values - exp_means
    return -0.5 * np.sum((residuals ** 2) / exp_vars, axis=1)
