import matplotlib.pyplot as plt
import numpy as np


def plot_hierarchical_results(results, hierarchical_fitter):
    """Generate plots for hierarchical model results"""
    df = results["dataframe"]
    param_names = results["param_names"]
    circuit_names = results["circuit_names"]

    # 1. Plot global parameter distributions (α)
    plt.figure(figsize=(12, 8))
    for i, param in enumerate(param_names):
        plt.subplot(2, len(param_names) // 2 + len(param_names) % 2, i + 1)
        alpha_col = f"alpha_{param}"
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

    # 2. Plot circuit-specific parameters (θ)
    for param in param_names:
        plt.figure(figsize=(12, 8))
        for i, circuit in enumerate(circuit_names):
            plt.subplot(2, len(circuit_names) // 2 + len(circuit_names) % 2, i + 1)
            theta_col = f"theta_{circuit}_{param}"
            alpha_col = f"alpha_{param}"
            plt.hist(df[theta_col], bins=30, alpha=0.7, label="Circuit-specific")
            plt.hist(df[alpha_col], bins=30, alpha=0.4, label="Global")
            plt.title(f"{circuit}: {param}")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"hierarchical_{param}_by_circuit.png")

    # 3. Plot covariance matrix elements (Σ)
    plt.figure(figsize=(10, 10))
    n_params = len(param_names)
    for i in range(n_params):
        for j in range(n_params):
            plt.subplot(n_params, n_params, i * n_params + j + 1)
            if i == j:
                # Variance
                sigma_col = f"sigma_{param_names[i]}"
                plt.hist(df[sigma_col], bins=30)
                plt.title(f"Var({param_names[i]})")
            else:
                # Correlation
                if i < j:
                    corr_col = f"corr_{param_names[i]}_{param_names[j]}"
                else:
                    corr_col = f"corr_{param_names[j]}_{param_names[i]}"
                plt.hist(df[corr_col], bins=30)
                plt.title(f"Corr({param_names[i]},{param_names[j]})")
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    plt.savefig("hierarchical_covariance.png")

    # 4. Plot posterior, likelihood, and prior
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


def simulate_hierarchical_comparison(hierarchical_fitter, results, n_best=5):
    """
    Simulate using both circuit-specific and global parameters

    Parameters
    ----------
    hierarchical_fitter : HierarchicalCircuitFitter
            The hierarchical fitter instance
    results : dict
            Results dictionary from analyze_hierarchical_mcmc_results
    n_best : int, optional
            Number of best parameter sets to use

    Returns
    -------
    dict
            Dictionary with simulation results for both parameter types
    """
    # Get the dataframe with all results
    df = results["dataframe"]

    # Get top n_best parameter sets by posterior probability
    best_indices = df["posterior"].argsort()[-n_best:][::-1]
    best_df = df.iloc[best_indices]

    # Extract parameter names and circuit names
    param_names = results["param_names"]
    circuit_names = results["circuit_names"]

    # Initialize results dictionaries
    theta_sim_results = {}
    alpha_sim_results = {}

    # For each top parameter set
    for i, idx in enumerate(best_indices):
        row = df.iloc[idx]

        # 1. Extract circuit-specific parameters (θ)
        theta_sim_results[i] = {}
        alpha_sim_results[i] = {}

        # Run simulations for each circuit using its specific parameters
        for c, circuit_name in enumerate(circuit_names):
            config = hierarchical_fitter.configs[c]

            # Extract circuit-specific parameters (θ) for this circuit
            circuit_log_params = np.zeros(len(param_names))
            for p, param in enumerate(param_names):
                col_name = f"theta_{circuit_name}_{param}"
                circuit_log_params[p] = row[col_name]

            # Extract global parameters (α)
            global_log_params = np.zeros(len(param_names))
            for p, param in enumerate(param_names):
                col_name = f"alpha_{param}"
                global_log_params[p] = row[col_name]

            # Convert to linear space
            circuit_linear_params = hierarchical_fitter.log_to_linear_params(
                circuit_log_params, param_names
            )

            global_linear_params = hierarchical_fitter.log_to_linear_params(
                global_log_params, param_names
            )

            # Prepare combined params for all conditions
            from likelihood_functions.utils import prepare_combined_params

            # Simulate with circuit-specific parameters
            theta_combined_params = prepare_combined_params(
                circuit_linear_params, config.condition_params
            )

            simulator = hierarchical_fitter.simulators[config.name]
            theta_sim_results_circuit = simulator.run(
                param_values=theta_combined_params.drop(
                    ["param_set_idx", "condition"], axis=1
                ),
            )

            theta_sim_results[i][c] = {
                "combined_params": theta_combined_params,
                "simulation_results": theta_sim_results_circuit,
                "config": config,
            }

            # Simulate with global parameters
            alpha_combined_params = prepare_combined_params(
                global_linear_params, config.condition_params
            )

            alpha_sim_results_circuit = simulator.run(
                param_values=alpha_combined_params.drop(
                    ["param_set_idx", "condition"], axis=1
                ),
            )

            alpha_sim_results[i][c] = {
                "combined_params": alpha_combined_params,
                "simulation_results": alpha_sim_results_circuit,
                "config": config,
            }

    # Calculate likelihoods for both sets
    theta_likelihoods = {}
    alpha_likelihoods = {}

    for i in range(n_best):
        # Calculate likelihoods for circuit-specific parameters
        theta_ll = {}
        for circuit_idx, circuit_results in theta_sim_results[i].items():
            config = hierarchical_fitter.configs[circuit_idx]
            circuit_name = config.name
            circuit_total = 0
            condition_likelihoods = {}

            for condition_name, _ in config.condition_params.items():
                condition_mask = (
                    circuit_results["combined_params"]["condition"] == condition_name
                )
                sim_indices = circuit_results["combined_params"].index[condition_mask]

                cached_data = hierarchical_fitter.experimental_data_cache[circuit_name][
                    condition_name
                ]
                exp_means, exp_vars = cached_data["means"], cached_data["vars"]

                sim_values = np.array(
                    [
                        circuit_results["simulation_results"].observables[i][
                            "obs_Protein_GFP"
                        ]
                        for i in sim_indices
                    ]
                )

                # Use the same likelihood calculation method as in the fitter
                if (
                    hasattr(hierarchical_fitter, "use_heteroscedastic")
                    and hierarchical_fitter.use_heteroscedastic
                ):
                    log_likelihood = (
                        hierarchical_fitter.calculate_heteroscedastic_likelihood(
                            sim_values, exp_means, exp_vars
                        )
                    )
                else:
                    from likelihood_functions.likelihood import calculate_likelihoods

                    log_likelihood = calculate_likelihoods(
                        sim_values, exp_means, exp_vars
                    )

                condition_likelihoods[condition_name] = log_likelihood[
                    0
                ]  # Get scalar value
                circuit_total += log_likelihood[0]

            theta_ll[circuit_name] = {
                "total": circuit_total,
                "conditions": condition_likelihoods,
            }
        theta_likelihoods[i] = theta_ll

        # Repeat for global parameters
        alpha_ll = {}
        for circuit_idx, circuit_results in alpha_sim_results[i].items():
            config = hierarchical_fitter.configs[circuit_idx]
            circuit_name = config.name
            circuit_total = 0
            condition_likelihoods = {}

            for condition_name, _ in config.condition_params.items():
                condition_mask = (
                    circuit_results["combined_params"]["condition"] == condition_name
                )
                sim_indices = circuit_results["combined_params"].index[condition_mask]

                cached_data = hierarchical_fitter.experimental_data_cache[circuit_name][
                    condition_name
                ]
                exp_means, exp_vars = cached_data["means"], cached_data["vars"]

                sim_values = np.array(
                    [
                        circuit_results["simulation_results"].observables[i][
                            "obs_Protein_GFP"
                        ]
                        for i in sim_indices
                    ]
                )

                if (
                    hasattr(hierarchical_fitter, "use_heteroscedastic")
                    and hierarchical_fitter.use_heteroscedastic
                ):
                    log_likelihood = (
                        hierarchical_fitter.calculate_heteroscedastic_likelihood(
                            sim_values, exp_means, exp_vars
                        )
                    )
                else:
                    from likelihood_functions.likelihood import calculate_likelihoods

                    log_likelihood = calculate_likelihoods(
                        sim_values, exp_means, exp_vars
                    )

                condition_likelihoods[condition_name] = log_likelihood[0]
                circuit_total += log_likelihood[0]

            alpha_ll[circuit_name] = {
                "total": circuit_total,
                "conditions": condition_likelihoods,
            }
        alpha_likelihoods[i] = alpha_ll

    return {
        "theta_simulations": theta_sim_results,
        "alpha_simulations": alpha_sim_results,
        "theta_likelihoods": theta_likelihoods,
        "alpha_likelihoods": alpha_likelihoods,
        "best_indices": best_indices,
        "best_parameters": best_df,
    }


def plot_hierarchical_comparison(
    hierarchical_fitter, comparison_results, param_set_idx=0
):
    """
    Plot comparison between circuit-specific and global parameter simulations

    Parameters
    ----------
    hierarchical_fitter : HierarchicalCircuitFitter
            The hierarchical fitter instance
    comparison_results : dict
            Results from simulate_hierarchical_comparison
    param_set_idx : int, optional
            Index of the parameter set to visualize

    Returns
    -------
    plt.Figure
            The comparison figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get simulation results for the specified parameter set
    theta_sim = comparison_results["theta_simulations"][param_set_idx]
    alpha_sim = comparison_results["alpha_simulations"][param_set_idx]
    theta_ll = comparison_results["theta_likelihoods"][param_set_idx]
    alpha_ll = comparison_results["alpha_likelihoods"][param_set_idx]

    # Setup the plot
    n_circuits = len(theta_sim)
    max_conditions = max(
        len(data["config"].condition_params) for data in theta_sim.values()
    )

    figsize = (5 * max_conditions, 4 * n_circuits)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.9, left=0.1, right=0.95, hspace=0.5, wspace=0.3)

    # Set colors
    colors = sns.color_palette("husl", 4)
    exp_color = colors[0]
    theta_color = colors[1]
    alpha_color = colors[2]

    # Calculate total likelihoods
    theta_total_ll = sum(circuit_data["total"] for circuit_data in theta_ll.values())
    alpha_total_ll = sum(circuit_data["total"] for circuit_data in alpha_ll.values())

    # Add title showing the parameter set and likelihoods
    plt.suptitle(
        f"Hierarchical Model Comparison - Parameter Set {param_set_idx + 1}\n"
        f"Circuit-specific LL: {theta_total_ll:.2f} | Global LL: {alpha_total_ll:.2f}",
        fontsize=16,
        y=0.98,
    )

    # Plot each circuit
    for circuit_idx, config in enumerate(hierarchical_fitter.configs):
        circuit_name = config.name

        # Theta (circuit-specific) simulation for this circuit
        theta_circuit = theta_sim[circuit_idx]
        # Alpha (global) simulation for this circuit
        alpha_circuit = alpha_sim[circuit_idx]

        # Get likelihoods
        theta_circuit_ll = theta_ll[circuit_name]["total"]
        alpha_circuit_ll = alpha_ll[circuit_name]["total"]

        # Add circuit title to the left side
        circuit_height = 1.0 / n_circuits
        circuit_center = 1.0 - (circuit_idx + 0.5) * circuit_height
        plt.figtext(
            0.01,
            circuit_center,
            f"{circuit_name}\nθ LL: {theta_circuit_ll:.2f}\nα LL: {alpha_circuit_ll:.2f}",
            fontsize=12,
            fontweight="bold",
            rotation=90,
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3.0),
        )

        # Plot each condition
        for condition_idx, condition_name in enumerate(config.condition_params.keys()):
            ax = plt.subplot(
                n_circuits,
                max_conditions,
                circuit_idx * max_conditions + condition_idx + 1,
            )

            # Get condition-specific likelihoods
            theta_condition_ll = theta_ll[circuit_name]["conditions"][condition_name]
            alpha_condition_ll = alpha_ll[circuit_name]["conditions"][condition_name]

            # Plot experimental data
            exp_data = config.experimental_data[
                config.experimental_data["condition"] == condition_name
            ]
            ax.scatter(
                exp_data["time"],
                exp_data["fluorescence"],
                color=exp_color,
                alpha=0.6,
                s=15,
                label="Experimental data",
            )

            # Plot theta (circuit-specific) simulation
            condition_mask = (
                theta_circuit["combined_params"]["condition"] == condition_name
            )
            sim_indices = theta_circuit["combined_params"].index[condition_mask]
            sim_values = theta_circuit["simulation_results"].observables[
                sim_indices[0]
            ]["obs_Protein_GFP"]

            ax.plot(
                config.tspan,
                sim_values,
                color=theta_color,
                label="Circuit-specific (θ)",
                linestyle="-",
                linewidth=2,
            )

            # Plot alpha (global) simulation
            condition_mask = (
                alpha_circuit["combined_params"]["condition"] == condition_name
            )
            sim_indices = alpha_circuit["combined_params"].index[condition_mask]
            sim_values = alpha_circuit["simulation_results"].observables[
                sim_indices[0]
            ]["obs_Protein_GFP"]

            ax.plot(
                config.tspan,
                sim_values,
                color=alpha_color,
                label="Global (α)",
                linestyle="--",
                linewidth=2,
            )

            # Set axis labels and title
            ax.set_title(
                f"{condition_name}\nθ LL: {theta_condition_ll:.2f} | α LL: {alpha_condition_ll:.2f}"
            )
            ax.set_xlabel("Time")
            if condition_idx == 0:
                ax.set_ylabel("Fluorescence")
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    return fig
