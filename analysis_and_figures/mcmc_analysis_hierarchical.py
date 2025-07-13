import numpy as np
import pandas as pd


def analyze_hierarchical_mcmc_results(
    parameters, priors, likelihoods, step_accepts, swap_accepts, hierarchical_fitter
):
    # Convert samples to usable format
    n_walkers, n_samples, n_chains, _ = parameters.shape
    flat_samples = parameters.reshape(n_walkers * n_samples * n_chains, -1)

    # Parameter names for compatibility with downstream analysis functions
    param_names = hierarchical_fitter.parameters_to_fit  # Keep for return dictionary
    # hierarchical_param_names = hierarchical_fitter.hierarchical_parameter_names
    # shared_param_names = hierarchical_fitter.shared_parameter_names

    # Use hierarchical fitter's parameter splitting (handles both legacy and mixed effects)
    theta_samples, alpha_samples, sigma_matrices = (
        hierarchical_fitter.split_hierarchical_parameters(flat_samples)
    )

    # Get parameter names for mixed effects structure
    hierarchical_param_names = hierarchical_fitter.hierarchical_parameter_names
    # shared_param_names = hierarchical_fitter.shared_parameter_names
    circuit_names = [config.name for config in hierarchical_fitter.configs]

    # Create DataFrame with mixed effects parameters
    results_df = pd.DataFrame()

    # Add θ parameters (circuit-specific, hierarchical only)
    for c, circuit in enumerate(circuit_names):
        for p, param in enumerate(hierarchical_param_names):
            col_name = f"theta_{circuit}_{param}"
            results_df[col_name] = theta_samples[:, c, p]

    # Add α parameters (hierarchical means only)
    for p, param in enumerate(hierarchical_param_names):
        col_name = f"alpha_{param}"
        results_df[col_name] = alpha_samples[:, p]

    # Add Σ elements (hierarchical covariance only)
    for p, param in enumerate(hierarchical_param_names):
        col_name = f"sigma_{param}"
        results_df[col_name] = sigma_matrices[:, p, p]

    # Add correlations from Σ (hierarchical parameters only)
    for p1 in range(len(hierarchical_param_names)):
        for p2 in range(p1 + 1, len(hierarchical_param_names)):
            param1 = hierarchical_param_names[p1]
            param2 = hierarchical_param_names[p2]
            col_name = f"corr_{param1}_{param2}"
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

    # Calculate acceptance rates with proper shape handling
    print("DEBUG: Calculating acceptance rates...")

    # Handle step_accepts shape variations
    if step_accepts.shape == (n_samples, n_walkers, n_chains):
        # Expected shape: (n_samples, n_walkers, n_chains)
        walker_accept_rates = np.mean(
            step_accepts, axis=(0, 2)
        )  # Average over samples and chains
        chain_accept_rates = np.mean(
            step_accepts, axis=(0, 1)
        )  # Average over samples and walkers
    elif step_accepts.shape == (n_walkers, n_samples, n_chains):
        # Alternative shape: (n_walkers, n_samples, n_chains)
        walker_accept_rates = np.mean(
            step_accepts, axis=(1, 2)
        )  # Average over samples and chains
        chain_accept_rates = np.mean(
            step_accepts, axis=(0, 1)
        )  # Average over walkers and samples
    else:
        # Fallback: calculate mean across all but the first dimension
        print(f"WARNING: Unexpected step_accepts shape {step_accepts.shape}")
        print(
            f"Expected: ({n_samples}, {n_walkers}, {n_chains}) or ({n_walkers}, {n_samples}, {n_chains})"
        )

        # Try to calculate reasonable acceptance rates
        if len(step_accepts.shape) >= 2:
            walker_accept_rates = np.mean(
                step_accepts, axis=tuple(range(1, len(step_accepts.shape)))
            )
            chain_accept_rates = np.mean(
                step_accepts, axis=tuple(range(0, len(step_accepts.shape) - 1))
            )
        else:
            # If all else fails, create dummy rates
            walker_accept_rates = np.full(n_walkers, np.mean(step_accepts))
            chain_accept_rates = np.full(n_chains, np.mean(step_accepts))

    print(f"DEBUG: walker_accept_rates.shape = {walker_accept_rates.shape}")
    print(f"DEBUG: n_walkers = {n_walkers}")

    # Ensure walker_accept_rates has the right length
    if len(walker_accept_rates) != n_walkers:
        print(
            f"WARNING: walker_accept_rates length {len(walker_accept_rates)} != n_walkers {n_walkers}"
        )
        # Resize or pad as needed
        if len(walker_accept_rates) > n_walkers:
            walker_accept_rates = walker_accept_rates[:n_walkers]
        else:
            # Pad with the mean
            mean_rate = (
                np.mean(walker_accept_rates) if len(walker_accept_rates) > 0 else 0.3
            )
            walker_accept_rates = np.pad(
                walker_accept_rates,
                (0, n_walkers - len(walker_accept_rates)),
                mode="constant",
                constant_values=mean_rate,
            )

    # Create a separate DataFrame for acceptance statistics
    accept_stats = {
        "walker_index": np.arange(n_walkers),
        "accept_rate": walker_accept_rates,
    }

    # Verify lengths match before creating DataFrame
    print(f"DEBUG: walker_index length = {len(accept_stats['walker_index'])}")
    print(f"DEBUG: accept_rate length = {len(accept_stats['accept_rate'])}")

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


def process_mcmc_data(df, burn_in=0.3, chain_idx=0):
    """Filter dataframe and return metadata"""
    max_iteration = df["iteration"].max()
    burn_in_cutoff = int(max_iteration * burn_in)
    processed_df = df[
        (df["iteration"] > burn_in_cutoff) & (df["chain"] == chain_idx)
    ].copy()

    return {
        "processed_data": processed_df,
        "metadata": {
            "burn_in_fraction": burn_in,
            "n_samples_raw": len(df),
            "n_samples_processed": len(processed_df),
            "chain_used": chain_idx,
        },
    }


def calculate_mcmc_diagnostics_from_dataframe(df, param_names, circuit_names):
    """
    Calculate proper MCMC diagnostics from dataframe
    """
    print("Calculating MCMC convergence diagnostics...")

    # Get basic info
    n_chains = df["chain"].nunique() if "chain" in df.columns else 1
    n_iterations = df["iteration"].nunique() if "iteration" in df.columns else len(df)
    n_walkers = (
        len(df) // (n_chains * n_iterations) if n_chains > 0 and n_iterations > 0 else 1
    )

    print(f"Chains: {n_chains}, Iterations: {n_iterations}, Walkers: {n_walkers}")

    # Calculate diagnostics for each parameter type
    diagnostics = {}

    # Global parameters (α)
    print("\nGLOBAL PARAMETERS (α):")
    print("-" * 60)
    print(f"{'Parameter':<15} {'Mean':<10} {'Std':<10} {'R-hat*':<8} {'Status'}")
    print("-" * 60)

    for param in param_names:
        alpha_col = f"alpha_{param}"
        if alpha_col in df.columns:
            # Basic statistics
            mean_val = df[alpha_col].mean()
            std_val = df[alpha_col].std()

            # Simple R-hat approximation using chains
            r_hat = calculate_simple_rhat(df, alpha_col)
            status = "Good" if r_hat < 1.1 else "Fair" if r_hat < 1.2 else "Poor"

            print(
                f"{param:<15} {mean_val:<10.3f} {std_val:<10.3f} {r_hat:<8.3f} {status}"
            )

            diagnostics[f"alpha_{param}"] = {
                "mean": mean_val,
                "std": std_val,
                "r_hat": r_hat,
                "status": status,
            }

    # Circuit-specific parameters (θ) - just show summary
    print("\nCIRCUIT-SPECIFIC PARAMETERS (θ) - Summary:")
    print("-" * 60)

    for circuit in circuit_names:
        print(f"\n{circuit}:")
        for param in param_names:
            theta_col = f"theta_{circuit}_{param}"
            if theta_col in df.columns:
                mean_val = df[theta_col].mean()
                std_val = df[theta_col].std()
                r_hat = calculate_simple_rhat(df, theta_col)
                status = "Good" if r_hat < 1.1 else "Fair" if r_hat < 1.2 else "Poor"

                print(
                    f"  {param:<12} {mean_val:<10.3f} {std_val:<10.3f} {r_hat:<8.3f} {status}"
                )

                diagnostics[f"theta_{circuit}_{param}"] = {
                    "mean": mean_val,
                    "std": std_val,
                    "r_hat": r_hat,
                    "status": status,
                }

    print("\n* R-hat approximation using chain means (requires multiple chains)")
    print("R-hat: <1.1 (Good), 1.1-1.2 (Fair), >1.2 (Poor)")

    return diagnostics


def calculate_simple_rhat(df, param_col):
    """Simple R-hat calculation using chain means"""
    if "chain" not in df.columns or df["chain"].nunique() < 2:
        return 1.0  # Can't calculate without multiple chains

    # Calculate between-chain and within-chain variance
    chain_means = df.groupby("chain")[param_col].mean()
    chain_vars = df.groupby("chain")[param_col].var()

    # Between-chain variance
    B = chain_means.var() * (len(df) // df["chain"].nunique())

    # Within-chain variance
    W = chain_vars.mean()

    # R-hat estimate
    if W > 0:
        var_plus = ((len(df) // df["chain"].nunique() - 1) * W + B) / (
            len(df) // df["chain"].nunique()
        )
        r_hat = (var_plus / W) ** 0.5
        return r_hat
    else:
        return 1.0
