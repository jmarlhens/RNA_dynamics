import numpy as np
import pandas as pd


def analyze_hierarchical_mcmc_results(
    parameters, priors, likelihoods, step_accepts, swap_accepts, hierarchical_fitter
):
    # Get parameter counts
    n_circuits = hierarchical_fitter.n_circuits
    n_params = hierarchical_fitter.n_parameters
    n_alpha_params = hierarchical_fitter.n_alpha_params

    # Convert samples to more usable format
    n_walkers, n_samples, n_chains, _ = parameters.shape
    flat_samples = parameters.reshape(n_walkers * n_samples * n_chains, -1)

    # Extract different parameter types
    theta_samples = flat_samples[:, : n_circuits * n_params].reshape(
        -1, n_circuits, n_params
    )
    alpha_samples = flat_samples[
        :, n_circuits * n_params : n_circuits * n_params + n_alpha_params
    ]
    sigma_flat_samples = flat_samples[:, n_circuits * n_params + n_alpha_params :]

    # Reconstruct sigma matrices
    sigma_matrices = np.zeros((len(flat_samples), n_params, n_params))
    for i in range(len(flat_samples)):
        sigma_matrices[i] = hierarchical_fitter._unflatten_covariance(
            sigma_flat_samples[i]
        )

    # Create parameter names
    param_names = hierarchical_fitter.parameters_to_fit
    circuit_names = [config.name for config in hierarchical_fitter.configs]

    # Create DataFrame with all parameters
    results_df = pd.DataFrame()

    # Add θ parameters (circuit-specific)
    for c, circuit in enumerate(circuit_names):
        for p, param in enumerate(param_names):
            col_name = f"theta_{circuit}_{param}"
            results_df[col_name] = theta_samples[:, c, p]

    # Add α parameters (global means)
    for p, param in enumerate(param_names):
        col_name = f"alpha_{param}"
        results_df[col_name] = alpha_samples[:, p]

    # Add diagonal elements of Σ (variances)
    for p, param in enumerate(param_names):
        col_name = f"sigma_{param}"
        results_df[col_name] = sigma_matrices[:, p, p]

    # Add correlations from Σ
    for p1 in range(n_params):
        for p2 in range(p1 + 1, n_params):
            param1 = param_names[p1]
            param2 = param_names[p2]
            col_name = f"corr_{param1}_{param2}"
            # Calculate correlation from covariance
            results_df[col_name] = sigma_matrices[:, p1, p2] / np.sqrt(
                sigma_matrices[:, p1, p1] * sigma_matrices[:, p2, p2]
            )

    # Create iteration, walker, chain columns to track where each sample came from
    iterations = np.repeat(np.arange(n_samples), n_walkers * n_chains)
    walkers = np.tile(np.repeat(np.arange(n_walkers), n_chains), n_samples)
    chains = np.tile(np.arange(n_chains), n_samples * n_walkers)

    # Add these columns to track the source of each sample
    results_df["iteration"] = iterations
    results_df["walker"] = walkers
    results_df["chain"] = chains

    # Add likelihoods and priors (flatten in the correct order)
    results_df["likelihood"] = likelihoods.flatten(order="C")
    results_df["prior"] = priors.flatten(order="C")
    results_df["posterior"] = results_df["likelihood"] + results_df["prior"]

    # Calculate acceptance rates properly (as additional analysis, not as columns)
    walker_accept_rates = np.mean(
        step_accepts, axis=(0, 2)
    )  # Average over samples and chains
    chain_accept_rates = np.mean(
        step_accepts, axis=(0, 1)
    )  # Average over samples and walkers

    # Create a separate DataFrame for acceptance statistics
    accept_stats = {
        "walker_index": np.arange(n_walkers),
        "accept_rate": walker_accept_rates,
    }
    walker_stats_df = pd.DataFrame(accept_stats)

    return {
        "dataframe": results_df,
        "theta_samples": theta_samples,
        "alpha_samples": alpha_samples,
        "sigma_matrices": sigma_matrices,
        "param_names": param_names,
        "circuit_names": circuit_names,
        "walker_stats": walker_stats_df,
        "walker_accept_rates": walker_accept_rates,
        "chain_accept_rates": chain_accept_rates,
    }
