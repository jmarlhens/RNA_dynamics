"""
Importance Resampling for Circuit Parameter Estimation

This module implements importance resampling to transfer parameter knowledge
from one circuit's posterior to another compatible circuit.

The idea:
- We have expensive MCMC samples from a source circuit's posterior: π_source(θ)
- We want samples from a target circuit's posterior: π_target(θ)
- We use importance weights: w_i ∝ π_target(θ_i) / π_source(θ_i)
- Then resample according to these weights

In log space:
log w_i = log_posterior_target(θ_i) - log_posterior_source(θ_i)
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import networkx as nx

from analysis_and_figures.mcmc_analysis_hierarchical import process_mcmc_data


def resample_by_target_only(
    log_posterior_target: np.ndarray,
    temperature: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Resample based only on target posterior.

    Ignores source posterior entirely - just ranks samples
    by how good they are for the target.
    """
    # Softmax with temperature
    log_weights = log_posterior_target / temperature
    log_weights_normalized = log_weights - logsumexp(log_weights)
    weights = np.exp(log_weights_normalized)

    # ESS measures how concentrated target posterior is over source samples
    ess = 1.0 / np.sum(weights**2)

    return weights, ess


def compute_importance_weights(
    log_posterior_target: np.ndarray,
    log_posterior_source: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute normalized importance weights and effective sample size.

    Parameters:
    -----------
    log_posterior_target : np.ndarray
        Log posterior values evaluated on target circuit
    log_posterior_source : np.ndarray
        Log posterior values from source circuit (from MCMC)

    Returns:
    --------
    weights : np.ndarray
        Normalized importance weights (sum to 1)
    ess : float
        Effective sample size
    """
    # Compute log importance weights
    log_weights = log_posterior_target - log_posterior_source

    # Normalize in log space for numerical stability
    log_weights_normalized = log_weights - logsumexp(log_weights)

    # Convert to regular weights
    weights = np.exp(log_weights_normalized)

    # Compute effective sample size: ESS = 1 / sum(w_i^2)
    # This is the standard ESS formula for normalized weights
    ess = 1.0 / np.sum(weights**2)

    return weights, ess


def importance_resample(
    samples: pd.DataFrame,
    weights: np.ndarray,
    n_resample: int = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Resample from samples according to importance weights.

    Parameters:
    -----------
    samples : pd.DataFrame
        Original samples (parameter values)
    weights : np.ndarray
        Normalized importance weights
    n_resample : int, optional
        Number of resampled particles. If None, use original sample size.
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    resampled : pd.DataFrame
        Resampled parameter values
    """
    np.random.seed(random_state)

    if n_resample is None:
        n_resample = len(samples)

    # Resample indices according to weights
    indices = np.random.choice(len(samples), size=n_resample, replace=True, p=weights)

    # Get resampled data
    resampled = samples.iloc[indices].reset_index(drop=True)

    return resampled


def build_parameter_dependency_graph(
    circuit_parameters: dict,
) -> tuple[nx.DiGraph, dict]:
    """
    Build a directed graph based on parameter subset relationships.

    If circuit A has all parameters of circuit B (params_A ⊇ params_B),
    then A can predict B, so we add edge A → B.

    Parameters:
    -----------
    circuit_parameters : dict
        Dictionary mapping circuit names to sets of parameter names

    Returns:
    --------
    G : networkx.DiGraph
        Directed graph of parameter dependencies
    edge_info : dict
        Information about each edge
    """
    G = nx.DiGraph()
    edge_info = {}

    circuit_names = list(circuit_parameters.keys())

    # Add all circuits as nodes
    for name in circuit_names:
        G.add_node(name, n_params=len(circuit_parameters[name]))

    # Check all pairs
    for source in circuit_names:
        source_params = circuit_parameters[source]

        for target in circuit_names:
            if source == target:
                continue

            target_params = circuit_parameters[target]

            # Check if source contains all target parameters
            if target_params.issubset(source_params):
                G.add_edge(source, target)

                extra_params = source_params - target_params
                edge_info[(source, target)] = {
                    "shared_params": target_params,
                    "extra_params_in_source": extra_params,
                    "n_shared": len(target_params),
                    "n_extra": len(extra_params),
                }

    return G, edge_info


def perform_importance_resampling(
    source_circuit: str,
    target_circuit: str,
    source_mcmc_samples: pd.DataFrame,
    source_circuit_fitter,
    target_circuit_fitter,
    n_initial_samples: int = 1000,
    n_resample: int = 200,
    burn_in: float = 0.4,
    random_state: int = 42,
) -> dict:
    """
    Perform importance resampling from source circuit posterior to target circuit.

    Parameters:
    -----------
    source_circuit : str
        Name of source circuit
    target_circuit : str
        Name of target circuit
    source_mcmc_samples : pd.DataFrame
        Raw MCMC samples from source circuit
    source_circuit_fitter : CircuitFitter
        Fitter for source circuit (to evaluate source posterior)
    target_circuit_fitter : CircuitFitter
        Fitter for target circuit (to evaluate target posterior)
    n_initial_samples : int
        Number of initial samples to use from source posterior
    n_resample : int
        Number of samples after resampling
    burn_in : float
        Fraction of burn-in to remove
    random_state : int
        Random seed

    Returns:
    --------
    results : dict
        Dictionary containing resampling results and diagnostics
    """
    # Process MCMC samples (apply burn-in and chain filtering)
    chain_idx = source_mcmc_samples["chain"].min()
    mcmc_processed = process_mcmc_data(
        source_mcmc_samples, burn_in=burn_in, chain_idx=chain_idx
    )
    mcmc_filtered = mcmc_processed["processed_data"]

    # Sample initial particles
    actual_n_samples = min(n_initial_samples, len(mcmc_filtered))
    np.random.seed(random_state)
    initial_samples = mcmc_filtered.sample(
        n=actual_n_samples, random_state=random_state
    )

    # Get source posterior values (these are already in the MCMC data)
    log_posterior_source = initial_samples["posterior"].values

    # Filter samples to target's parameters
    target_params = target_circuit_fitter.parameters_to_fit
    available_params = [p for p in target_params if p in initial_samples.columns]

    if len(available_params) != len(target_params):
        missing = set(target_params) - set(available_params)
        raise ValueError(f"Missing parameters for target circuit: {missing}")

    # Extract parameter values for target
    param_samples = initial_samples[available_params].copy()
    param_values = param_samples.values

    # Evaluate target posterior

    simulation_results = target_circuit_fitter.simulate_parameters(param_values)
    log_likelihood_target = (
        target_circuit_fitter.calculate_likelihood_from_simulation_with_breakdown(
            simulation_results
        )
    )["total"]
    # log_likelihood_target = target_circuit_fitter.calculate_log_likelihood(param_values)["total"]
    log_prior_target = target_circuit_fitter.calculate_log_prior(param_values)
    log_posterior_target = log_likelihood_target + log_prior_target

    # Compute importance weights
    # weights, ess = compute_importance_weights(log_posterior_target, log_posterior_source)
    temperature = 10.0
    weights, ess = resample_by_target_only(log_posterior_target, temperature)

    # Perform resampling
    resampled = importance_resample(
        param_samples, weights, n_resample=n_resample, random_state=random_state
    )

    # Evaluate resampled particles on target
    resampled_values = resampled.values
    log_likelihood_resampled = target_circuit_fitter.calculate_log_likelihood(
        resampled_values
    )["total"]
    log_prior_resampled = target_circuit_fitter.calculate_log_prior(resampled_values)
    log_posterior_resampled = log_likelihood_resampled + log_prior_resampled

    # add columns to resampled dataframe
    resampled["log_likelihood"] = log_likelihood_resampled
    resampled["log_prior"] = log_prior_resampled
    resampled["log_posterior"] = log_posterior_resampled

    # Compile results
    results = {
        "source_circuit": source_circuit,
        "target_circuit": target_circuit,
        "n_initial_samples": actual_n_samples,
        "n_resampled": n_resample,
        "effective_sample_size": ess,
        "ess_ratio": ess / actual_n_samples,
        "initial_samples": param_samples,
        "resampled_samples": resampled,
        "weights": weights,
        "log_weights": np.log(weights + 1e-300),
        "simulation_results_target": simulation_results,
        "log_posterior_source": log_posterior_source,
        "log_posterior_target": log_posterior_target,
        "log_posterior_resampled": log_posterior_resampled,
        "mean_log_posterior_before": np.mean(log_posterior_target),
        "mean_log_posterior_after": np.mean(log_posterior_resampled),
        "best_log_posterior_before": np.max(log_posterior_target),
        "best_log_posterior_after": np.max(log_posterior_resampled),
        "parameter_names": available_params,
    }

    return results
