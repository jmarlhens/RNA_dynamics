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
    """
    Compute log likelihood for a specific condition

    Parameters
    ----------
    simulation_results : Any
        Simulation results object
    experimental_data : pd.DataFrame
        Experimental data
    tspan : np.ndarray
        Time points array
    combined_params : pd.DataFrame
        Combined parameter sets
    condition : str
        Condition name
    calibration_params : Dict
        Dictionary containing:
            - slope: float
            - intercept: float
            - brightness_correction: float

    Returns
    -------
    pd.Series
        Log likelihood values indexed by parameter set indices
    """
    # Get indices for this condition
    condition_mask = combined_params['condition'] == condition
    sim_indices = combined_params.index[condition_mask]
    param_set_indices = combined_params.loc[condition_mask, 'param_set_idx']

    # Prepare experimental data
    exp_means, exp_vars = prepare_experimental_data(experimental_data, tspan)

    # Get simulation values and convert from nM to AU
    sim_values = np.array([
        simulation_results.observables[i]['obs_Protein_GFP']
        for i in sim_indices
    ])

    # Convert simulation values from nM to AU using calibration
    sim_values_au = convert_nM_to_AU(
        sim_values,
        calibration_params['slope'],
        calibration_params['intercept'],
        calibration_params['brightness_correction']
    )

    # Calculate likelihoods
    log_likelihoods = calculate_likelihoods(sim_values_au, exp_means, exp_vars)

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
