import matplotlib.pyplot as plt
import numpy as np


def simulate_hierarchical_posterior_theta_random(
    hierarchical_fitter, processed_df, param_names, circuit_names, n_samples=100
):
    """
    Sample randomly from already-processed posterior data and simulate
    Takes n random samples from the processed DataFrame
    """

    print(f"Starting with {len(processed_df)} processed samples")

    # Random sampling from MCMC posterior
    if len(processed_df) > n_samples:
        sampled_df = processed_df.sample(n=n_samples, random_state=42).reset_index(
            drop=True
        )
    else:
        sampled_df = processed_df.reset_index(drop=True)

    print(f"Using {len(sampled_df)} random samples for simulation")

    # Run simulations for each circuit
    circuit_results = {}

    for c, circuit_name in enumerate(circuit_names):
        print(f"Simulating circuit: {circuit_name}")

        # Extract circuit-specific parameters (theta) for all samples
        circuit_log_params = np.zeros((len(sampled_df), len(param_names)))
        for p, param in enumerate(param_names):
            col_name = f"theta_{circuit_name}_{param}"
            circuit_log_params[:, p] = sampled_df[col_name].values

        # Simulate only this specific circuit with its own parameters
        circuit_sim_data = hierarchical_fitter.simulate_single_circuit(
            c, circuit_log_params
        )

        # Use integer key for compatibility with calculate_likelihood_from_simulation
        circuit_results[c] = circuit_sim_data

    # Calculate likelihoods
    likelihood_results = hierarchical_fitter.calculate_likelihood_from_simulation(
        circuit_results
    )

    return {
        "simulation_data": circuit_results,
        "likelihood_results": likelihood_results,
        "sampled_parameters": sampled_df,
        "param_names": param_names,
        "circuit_names": circuit_names,
    }


def simulate_hierarchical_posterior_theta_best(
    hierarchical_fitter, processed_df, param_names, circuit_names, n_samples=100
):
    """
    Sample best-fitting samples from already-processed posterior data and simulate
    Takes the n best samples for each circuit based on circuit-specific likelihood
    """

    print(f"Starting with {len(processed_df)} processed samples")

    # Run simulations for each circuit
    circuit_results = {}

    for c, circuit_name in enumerate(circuit_names):
        print(f"Simulating circuit: {circuit_name}")

        # Find circuit-specific likelihood column
        likelihood_col = None
        for col in processed_df.columns:
            if (
                f"likelihood_{circuit_name}" in col
                or f"{circuit_name}_likelihood" in col
            ):
                likelihood_col = col
                break

        # If no circuit-specific likelihood found, use total likelihood
        if likelihood_col is None:
            likelihood_col = "likelihood"
            print("  Using total likelihood (no circuit-specific found)")
        else:
            print(f"  Using circuit-specific likelihood: {likelihood_col}")

        # Select top n samples for this circuit based on likelihood
        if len(processed_df) > n_samples:
            # Sort by circuit-specific likelihood (highest first) and take top n
            top_indices = processed_df.nlargest(n_samples, likelihood_col).index
            sampled_df = processed_df.loc[top_indices].reset_index(drop=True)
        else:
            sampled_df = processed_df.reset_index(drop=True)

        print(f"  Selected {len(sampled_df)} best samples for {circuit_name}")

        # Extract circuit-specific parameters (theta) for selected samples
        circuit_log_params = np.zeros((len(sampled_df), len(param_names)))
        for p, param in enumerate(param_names):
            col_name = f"theta_{circuit_name}_{param}"
            circuit_log_params[:, p] = sampled_df[col_name].values

        # Simulate only this specific circuit with its own parameters
        circuit_sim_data = hierarchical_fitter.simulate_single_circuit(
            c, circuit_log_params
        )

        # Use integer key for compatibility with calculate_likelihood_from_simulation
        circuit_results[c] = circuit_sim_data

    # Calculate likelihoods
    likelihood_results = hierarchical_fitter.calculate_likelihood_from_simulation(
        circuit_results
    )

    return {
        "simulation_data": circuit_results,
        "likelihood_results": likelihood_results,
        "sampled_parameters": sampled_df,  # Note: this will be the last circuit's samples
        "param_names": param_names,
        "circuit_names": circuit_names,
    }


def plot_hierarchical_posterior_theta_only(
    hierarchical_fitter, posterior_results, figsize=None, save_path=None, ll_quartile=20
):
    """
    Plot posterior samples using only circuit-specific (theta) parameters
    Uses the same approach as plot_all_simulation_results
    """

    simulation_data = posterior_results["simulation_data"]
    likelihood_results = posterior_results["likelihood_results"]

    n_circuits = len(simulation_data)
    max_conditions = max(
        len(data["config"].condition_params) for data in simulation_data.values()
    )

    if figsize is None:
        figsize = (4.8 * max_conditions, 4 * n_circuits)

    fig = plt.figure(figsize=figsize)

    # Calculate y limits per circuit (same as existing code)
    circuit_y_limits = {}
    for circuit_idx, data in simulation_data.items():
        config = data["config"]
        exp_data = config.experimental_data
        exp_max = np.percentile(exp_data["fluorescence"], 95)
        circuit_y_limits[config.name] = (0, exp_max * 1.2)

    # Plot each circuit (same structure as plot_all_simulation_results)
    for circuit_idx, data in simulation_data.items():
        config = data["config"]
        combined_params = data["combined_params"]
        simulation_results = data["simulation_results"]

        circuit_name = config.name

        for condition_idx, condition_name in enumerate(config.condition_params.keys()):
            ax = plt.subplot(
                n_circuits,
                max_conditions,
                circuit_idx * max_conditions + condition_idx + 1,
            )

            # Get condition-specific likelihoods for color coding
            condition_lls = likelihood_results["circuits"][circuit_name]["conditions"][
                condition_name
            ]
            min_ll = np.percentile(condition_lls, ll_quartile)
            max_ll = np.percentile(condition_lls, 100 - ll_quartile)
            norm = plt.Normalize(vmin=min_ll, vmax=max_ll)
            cmap = plt.cm.viridis  # Use viridis like existing plots

            # Plot simulations (exactly like existing code)
            condition_mask = combined_params["condition"] == condition_name
            sim_indices = combined_params.index[condition_mask]
            param_indices = combined_params.loc[condition_mask, "param_set_idx"]

            for param_idx, sim_idx in zip(param_indices, sim_indices):
                sim_values = simulation_results.observables[sim_idx]["obs_Protein_GFP"]
                ll = condition_lls[param_idx]
                ax.plot(
                    config.tspan,
                    sim_values,
                    color=cmap(norm(ll)),
                    alpha=0.4,
                    linewidth=1.5,
                    zorder=1,
                )

            # Plot experimental data
            exp_data = config.experimental_data[
                config.experimental_data["condition"] == condition_name
            ]
            ax.scatter(
                exp_data["time"],
                exp_data["fluorescence"],
                color="red",
                alpha=0.7,
                s=15,
                zorder=3,
                label="Experimental data"
                if circuit_idx == 0 and condition_idx == 0
                else "",
            )

            # Set limits and labels
            ax.set_ylim(*circuit_y_limits[circuit_name])
            ax.set_title(f"{circuit_name}\n{condition_name}")
            ax.set_xlabel("Time (min)")
            if condition_idx == 0:
                ax.set_ylabel("Fluorescence (nM)")

            # Add colorbar
            plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                label="Log-likelihood",
                shrink=0.6,
            )
            ax.grid(True, alpha=0.3)

            # Add legend to first subplot
            if circuit_idx == 0 and condition_idx == 0:
                ax.legend(loc="upper right", fontsize="small")

    plt.suptitle(
        "Hierarchical Model: Posterior Samples (Circuit-specific θ parameters)\n"
        "Color intensity: Circuit-specific likelihood",
        fontsize=14,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_hierarchical_results(results, hierarchical_fitter):
    """Generate plots for hierarchical model parameter distributions"""
    df = results["dataframe"]
    param_names = results["param_names"]
    circuit_names = results["circuit_names"]

    # 1. Plot global parameter distributions (α)
    plt.figure(figsize=(12, 8))
    for i, param in enumerate(param_names):
        plt.subplot(2, len(param_names) // 2 + len(param_names) % 2, i + 1)
        alpha_col = f"alpha_{param}"
        if alpha_col in df.columns:
            plt.hist(df[alpha_col], bins=30, alpha=0.7)
            plt.axvline(
                hierarchical_fitter.mu_alpha[i],
                color="r",
                linestyle="--",
                label="Prior mean",
            )
            plt.title(f"Global {param}")
            plt.grid(True)
    plt.tight_layout()
    plt.savefig("hierarchical_global_params.png")
    plt.close()

    # 2. Plot circuit-specific parameters (θ)
    for param in param_names:
        plt.figure(figsize=(12, 8))
        for i, circuit in enumerate(circuit_names):
            plt.subplot(2, len(circuit_names) // 2 + len(circuit_names) % 2, i + 1)
            theta_col = f"theta_{circuit}_{param}"
            alpha_col = f"alpha_{param}"
            if theta_col in df.columns and alpha_col in df.columns:
                plt.hist(df[theta_col], bins=30, alpha=0.7, label="Circuit-specific")
                plt.hist(df[alpha_col], bins=30, alpha=0.4, label="Global")
                plt.title(f"{circuit}: {param}")
                plt.legend()
                plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"hierarchical_{param}_by_circuit.png")
        plt.close()

    # 3. Plot posterior, likelihood, and prior
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(df["posterior"], bins=30, alpha=0.7)
    plt.title("Posterior Distribution")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.hist(df["likelihood"], bins=30, alpha=0.7)
    plt.title("Likelihood Distribution")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.hist(df["prior"], bins=30, alpha=0.7)
    plt.title("Prior Distribution")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("hierarchical_distributions.png")
    plt.close()
