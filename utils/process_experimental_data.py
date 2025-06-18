import numpy as np
import pandas as pd
from typing import Tuple, Optional


def prepare_combined_params(
    param_sets: pd.DataFrame, condition_params: dict
) -> pd.DataFrame:
    """
    Create combined parameter sets with tracking columns

    Returns a DataFrame with:
    - All parameter values
    - param_set_idx: Index of original parameter set
    - condition: Name of the condition
    """
    # Create condition DataFrame with condition names
    conditions_list = []
    for condition_name, params in condition_params.items():
        condition_df = pd.DataFrame([params])
        condition_df["condition"] = condition_name
        conditions_list.append(condition_df)

    condition_df = pd.concat(conditions_list, ignore_index=True)

    # Add tracking index to param_sets
    param_sets_with_idx = param_sets.copy()
    param_sets_with_idx["param_set_idx"] = param_sets.index

    # Create cross product with conditions
    param_sets_with_idx["key"] = 1
    condition_df["key"] = 1

    combined_df = pd.merge(param_sets_with_idx, condition_df, on="key").drop(
        "key", axis=1
    )

    return combined_df


def prepare_experimental_data(
    experimental_data: pd.DataFrame, tspan: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare experimental data arrays for likelihood calculation"""
    condition_subset = experimental_data[experimental_data["time"].isin(tspan)]
    exp_means = np.array(
        [
            condition_subset[condition_subset["time"] == t]["fluorescence"].mean()
            for t in tspan
        ]
    )
    exp_vars = np.array(
        [
            max(
                condition_subset[condition_subset["time"] == t]["fluorescence"].var(), 1
            )
            for t in tspan
        ]
    )

    return condition_subset, exp_means.reshape(1, -1), exp_vars.reshape(1, -1)


def process_background_fluorescence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process experimental data by subtracting background fluorescence from its initial time point so that it starts at zero.

    Parameters
    ----------
    df : pd.DataFrame containing columns: 'time', 'fluorescence', 'condition', 'replicate'

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with time zero fluorescence subtracted from each replicate
    """
    # return process_negative_controls(df)
    processed_df = df.copy()
    # we need for each replicate for each condition, we need to create a column that contains those two information
    processed_df["condition_replicate"] = (
        processed_df["condition"] + "_" + processed_df["replicate"].astype(str)
    )
    initial_fluorescence = processed_df[processed_df["time"] == 0].set_index(
        "condition_replicate"
    )["fluorescence"]
    initial_fluorescence_lookup = pd.Series(
        initial_fluorescence, index=initial_fluorescence.index
    )
    mask = processed_df["condition"] != "Negative"
    processed_df.loc[mask, "fluorescence"] = processed_df[mask].apply(
        lambda row: row["fluorescence"]
        - initial_fluorescence_lookup[row["condition_replicate"]],
        axis=1,
    )
    return processed_df[processed_df["condition"] != "Negative"]


def process_negative_controls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process experimental data by subtracting averaged negative controls for each time point.

    Parameters
    ----------
    df : pd.DataFrame containing columns: 'time', 'fluorescence', 'condition', 'replicate'

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with negative controls subtracted from each replicate
    """
    processed_df = df.copy()

    neg_controls = processed_df[processed_df["condition"] == "Negative"].copy()
    avg_neg_controls = neg_controls.groupby("time")["fluorescence"].mean()

    neg_control_lookup = pd.Series(avg_neg_controls, index=avg_neg_controls.index)

    mask = processed_df["condition"] != "Negative"
    processed_df.loc[mask, "fluorescence"] = processed_df[mask].apply(
        lambda row: row["fluorescence"] - neg_control_lookup[row["time"]], axis=1
    )

    return processed_df[processed_df["condition"] != "Negative"]


def apply_time_constraints(
    experimental_data: pd.DataFrame,
    tspan: np.ndarray,
    min_time: Optional[float],
    max_time: Optional[float],
) -> Tuple[pd.DataFrame, np.ndarray, Optional[float]]:
    """
    Apply time constraints to experimental data and tspan, and adjust max_time if necessary.

    Parameters
    ----------
    experimental_data : pd.DataFrame
        The experimental data to process.
    tspan : np.ndarray
        The time span array.
    min_time : Optional[float]
        The minimum time constraint.
    max_time : Optional[float]
        The maximum time constraint.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, Optional[float]]
        Processed experimental data, updated tspan, and adjusted max_time.
    """
    if min_time is not None or max_time is not None:
        # Create a copy of the data to avoid modifying the original
        experimental_data = experimental_data.copy()

        # Apply min time filter if specified
        if min_time is not None:
            # Filter out data points before min_time
            experimental_data = experimental_data[experimental_data["time"] >= min_time]

            # Shift time values by subtracting min_time
            experimental_data["time"] = experimental_data["time"] - min_time

            # Adjust max_time if it's set
            if max_time is not None:
                max_time = max(0, max_time - min_time)

            # Update tspan to match shifted times
            tspan = tspan[tspan >= min_time] - min_time

        # Apply max time filter if specified (after min_time adjustments)
        if max_time is not None:
            # Truncate experimental data
            experimental_data = experimental_data[experimental_data["time"] <= max_time]

            # Truncate simulation timespan
            tspan = tspan[tspan <= max_time]

    return experimental_data, tspan, max_time


def organize_results(
    parameters_to_fit,
    log_params: np.ndarray,
    likelihood_data: dict,
    log_prior: np.ndarray,
) -> pd.DataFrame:
    """
    Organize parameter sets and their evaluations into a hierarchical DataFrame.

    Args:
        parameters_to_fit: List of parameter names
        log_params: Array of parameters in log space
        likelihood_data: Dictionary from calculate_likelihood_from_simulation
        log_prior: Array of log prior probabilities

    Returns:
        DataFrame with hierarchical columns
    """
    # Create tuples for MultiIndex
    column_tuples = []

    # Add parameter columns
    for name in parameters_to_fit:
        column_tuples.append(("parameters", name, ""))
        column_tuples.append(("log_parameters", name, ""))

    # Add likelihood columns
    for circuit_name, circuit_data in likelihood_data["circuits"].items():
        column_tuples.append(("likelihood", circuit_name, "total"))
        for cond_name in circuit_data["conditions"].keys():
            column_tuples.append(("likelihood", circuit_name, cond_name))

    # Add metrics columns
    column_tuples.append(("metrics", "log_prior", ""))
    column_tuples.append(("metrics", "log_likelihood", ""))
    column_tuples.append(("metrics", "log_posterior", ""))

    # Create MultiIndex
    columns = pd.MultiIndex.from_tuples(
        column_tuples, names=["category", "item", "subitem"]
    )

    # Initialize DataFrame with proper structure
    n_samples = len(log_params)
    results_df = pd.DataFrame(index=range(n_samples), columns=columns)

    # Fill parameters
    for i, name in enumerate(parameters_to_fit):
        results_df[("parameters", name, "")] = 10 ** log_params[:, i]
        results_df[("log_parameters", name, "")] = log_params[:, i]

    # Fill likelihoods
    for circuit_name, circuit_data in likelihood_data["circuits"].items():
        results_df[("likelihood", circuit_name, "total")] = circuit_data["total"]
        for cond_name, cond_ll in circuit_data["conditions"].items():
            results_df[("likelihood", circuit_name, cond_name)] = cond_ll

    # Fill metrics
    results_df[("metrics", "log_prior", "")] = log_prior
    results_df[("metrics", "log_likelihood", "")] = likelihood_data["total"]
    results_df[("metrics", "log_posterior", "")] = log_prior + likelihood_data["total"]

    return results_df
