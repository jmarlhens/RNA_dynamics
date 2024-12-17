import numpy as np
import pandas as pd
from typing import Any
from .utils import prepare_experimental_data


def compute_condition_likelihood(
        simulation_results: Any,
        experimental_data: pd.DataFrame,
        tspan: np.ndarray,
        combined_params: pd.DataFrame,
        condition: str
) -> pd.Series:
    """
    Compute log likelihood for a specific condition
    """
    # Get indices for this condition
    condition_mask = combined_params['condition'] == condition
    sim_indices = combined_params.index[condition_mask]
    param_set_indices = combined_params.loc[condition_mask, 'param_set_idx']

    # Prepare experimental data
    exp_means, exp_vars = prepare_experimental_data(experimental_data, tspan)

    # Get simulation values using explicit indices
    # sim values is in nM and should be converted into AU
    sim_values = np.array([
        simulation_results.observables[i]['obs_Protein_GFP'] * 100
        for i in sim_indices
    ])

    # Calculate likelihoods
    log_likelihoods = calculate_likelihoods(sim_values, exp_means, exp_vars)

    # Return Series indexed by original parameter set indices
    return pd.Series(log_likelihoods, index=param_set_indices)


def calculate_likelihoods(
        sim_values: np.ndarray,
        exp_means: np.ndarray,
        exp_vars: np.ndarray
) -> np.ndarray:
    """Calculate log likelihoods using vectorized operations"""
    residuals = sim_values - exp_means
    return -0.5 * np.sum((residuals ** 2) / exp_vars, axis=1)
