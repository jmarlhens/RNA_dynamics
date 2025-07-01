import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional
import pandas as pd


def plot_simulation_results(
    simulation_data: dict,
    results_df: pd.DataFrame,
    param_set_idx: int = 0,
    figsize: tuple = None,
    save_path: Optional[str] = None,
    share_y_by_circuit: bool = True,
) -> plt.Figure:
    """
    Plot simulation results vs experimental data with likelihood values from results DataFrame
    """
    n_circuits = len(simulation_data)
    max_conditions = max(
        len(data["config"].condition_params) for data in simulation_data.values()
    )

    if figsize is None:
        figsize = (4 * max_conditions, 4.5 * n_circuits)

    fig = plt.figure(figsize=figsize)
    # Increase spacing between subplots and edges
    plt.subplots_adjust(top=0.85, left=0.2, right=0.95, hspace=0.5, wspace=0.3)

    colors = sns.color_palette("husl", 3)
    circuit_y_limits = {}

    if share_y_by_circuit:
        for circuit_idx, data in simulation_data.items():
            y_min, y_max = float("inf"), float("-inf")
            for condition_name in data["config"].condition_params.keys():
                exp_data = data["config"].experimental_data[
                    data["config"].experimental_data["condition"] == condition_name
                ]
                y_min = min(y_min, exp_data["fluorescence"].min())
                y_max = max(y_max, exp_data["fluorescence"].max())

                condition_mask = data["combined_params"]["condition"] == condition_name
                sim_indices = data["combined_params"].index[condition_mask]
                param_indices = data["combined_params"].loc[
                    condition_mask, "param_set_idx"
                ]
                param_sim_idx = sim_indices[param_indices == param_set_idx][0]
                sim_values = data["simulation_results"].observables[param_sim_idx][
                    "obs_Protein_GFP"
                ]
                y_min = min(y_min, np.min(sim_values))
                y_max = max(y_max, np.max(sim_values))

            y_range = y_max - y_min
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
            circuit_y_limits[circuit_idx] = (y_min, y_max)

    total_ll = results_df.loc[param_set_idx, ("metrics", "log_likelihood", "")]
    plt.suptitle(
        f"Parameter Set {param_set_idx + 1}\nTotal Log Likelihood: {total_ll:.2f}",
        y=0.95,
        fontsize=14,
        fontweight="bold",
    )

    for circuit_idx, data in simulation_data.items():
        config = data["config"]
        combined_params = data["combined_params"]
        simulation_results = data["simulation_results"]

        circuit_ll = results_df.loc[param_set_idx, ("likelihood", config.name, "total")]

        # Vertical circuit title
        circuit_height = 1.0 / n_circuits
        circuit_center = 1.0 - (circuit_idx + 0.5) * circuit_height
        plt.figtext(
            0.05,
            circuit_center,
            f"{config.name}\nCircuit LL: {circuit_ll:.2f}",
            fontsize=12,
            fontweight="bold",
            rotation=90,
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3.0),
        )

        for condition_idx, condition_name in enumerate(config.condition_params.keys()):
            ax = plt.subplot(
                n_circuits,
                max_conditions,
                circuit_idx * max_conditions + condition_idx + 1,
            )

            condition_ll = results_df.loc[
                param_set_idx, ("likelihood", config.name, condition_name)
            ]
            exp_data = config.experimental_data[
                config.experimental_data["condition"] == condition_name
            ]

            ax.scatter(
                exp_data["time"],
                exp_data["fluorescence"],
                color=colors[0],
                alpha=0.6,
                s=15,
                label="Experimental data",
            )

            condition_mask = combined_params["condition"] == condition_name
            sim_indices = combined_params.index[condition_mask]
            param_indices = combined_params.loc[condition_mask, "param_set_idx"]
            param_sim_idx = sim_indices[param_indices == param_set_idx][0]
            sim_values = (
                simulation_results.observables[param_sim_idx]["obs_Protein_GFP"] * 100
            )

            ax.plot(
                config.tspan,
                sim_values,
                color=colors[1],
                label="Simulation",
                linestyle="-",
                linewidth=2,
            )

            if share_y_by_circuit:
                ax.set_ylim(circuit_y_limits[circuit_idx])

            ax.set_title(f"{condition_name}\nLL: {condition_ll:.2f}", pad=20)
            ax.set_xlabel("Time")
            if condition_idx == 0:
                ax.set_ylabel("Fluorescence")
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_all_simulation_results(
    simulation_data: dict,
    results_df: pd.DataFrame,
    figsize: tuple = None,
    save_path: Optional[str] = None,
    ll_quartile: int = 40,
    y_max_percentile: int = 100,
) -> plt.Figure:
    n_circuits = len(simulation_data)
    max_conditions = max(
        len(data["config"].condition_params) for data in simulation_data.values()
    )

    if figsize is None:
        figsize = (4.8 * max_conditions, 4 * n_circuits)

    fig = plt.figure(figsize=figsize)
    # plt.subplots_adjust(top=0.85, left=0.2, right=0.95, hspace=0.5, wspace=0.3)

    # Calculate y limits per circuit
    circuit_y_limits = {}
    for circuit_name, data in simulation_data.items():
        config = data["config"]
        # simulation_results = data['simulation_results']

        # Get all experimental data for this circuit
        exp_data = config.experimental_data
        exp_max = np.percentile(exp_data["fluorescence"], y_max_percentile)

        # Set y limit for this circuit
        circuit_y_limits[circuit_name] = (0, exp_max * 1.1)

    for circuit_idx, (circuit_name, data) in enumerate(simulation_data.items()):
        config = data["config"]
        combined_params = data["combined_params"]
        simulation_results = data["simulation_results"]

        # circuit_height = 1.0 / n_circuits
        # circuit_center = 1.0 - (circuit_idx + 0.5) * circuit_height
        # plt.figtext(0.02, circuit_center, f'{config.name}',
        #             fontsize=12, fontweight='bold', rotation=90,
        #             verticalalignment='center',
        #             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))

        for condition_idx, condition_name in enumerate(config.condition_params.keys()):
            ax = plt.subplot(
                n_circuits,
                max_conditions,
                circuit_idx * max_conditions + condition_idx + 1,
            )

            condition_lls = results_df[("likelihood", config.name, condition_name)]
            min_ll = np.percentile(condition_lls, ll_quartile)
            max_ll = np.percentile(condition_lls, 100 - ll_quartile)
            norm = plt.Normalize(vmin=min_ll, vmax=max_ll)
            cmap = plt.cm.viridis

            # Plot simulations
            condition_mask = combined_params["condition"] == condition_name
            sim_indices = combined_params.index[condition_mask]
            param_indices = combined_params.loc[condition_mask, "param_set_idx"]

            for param_idx, sim_idx in zip(param_indices, sim_indices):
                sim_values = simulation_results.observables[sim_idx]["obs_Protein_GFP"]
                ll = condition_lls[param_idx]
                ax.plot(
                    config.tspan, sim_values, color=cmap(norm(ll)), alpha=0.3, zorder=1
                )

            # Plot experimental data
            exp_data = config.experimental_data[
                config.experimental_data["condition"] == condition_name
            ]
            ax.scatter(
                exp_data["time"],
                exp_data["fluorescence"],
                color="red",
                alpha=0.6,
                s=5,
                zorder=3,
                label="Experimental data",
            )

            ax.set_ylim(*circuit_y_limits[circuit_name])
            ax.set_title(f"{condition_name}")
            ax.set_xlabel("Time")
            if condition_idx == 0:
                ax.set_ylabel("Fluorescence")

            plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                label="Log-likelihood",
            )
            ax.grid(True, alpha=0.3)

    # tight_layout() doesn't work well with colorbars
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_simulation_statistical_summaries(
    simulation_data_dict,
    likelihood_results_dataframe,
    save_path=None,
    max_time_cutoff=None,
    min_time_cutoff=None,
    summary_type="median_iqr",  # 'median_iqr' or 'mean_std'
    ribbon_alpha=0.25,
):
    """
    Plot statistical summaries of simulation trajectories instead of individual traces

    Parameters
    ----------
    simulation_data_dict : dict
        Dictionary mapping circuit indices to simulation data
    likelihood_results_dataframe : pd.DataFrame
        Results dataframe with likelihood information
    save_path : str, optional
        Path to save the figure
    max_time_cutoff : float, optional
        Maximum time to display
    min_time_cutoff : float, optional
        Minimum time to display
    summary_type : str
        'median_iqr' for median with 25th-75th percentiles
        'mean_std' for mean with standard deviation
    ribbon_alpha : float
        Transparency of error ribbons (0-1)
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Determine grid layout
    n_circuits = len(simulation_data_dict)
    n_cols = min(3, n_circuits)
    n_rows = (n_circuits + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_circuits == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # Extract and reshape all trajectory data
    trajectory_data_list = []

    for circuit_idx, circuit_data in simulation_data_dict.items():
        circuit_config = circuit_data["config"]
        combined_parameters = circuit_data["combined_params"]
        simulation_results = circuit_data["simulation_results"]
        time_points = circuit_config.tspan

        # Apply time filtering
        if min_time_cutoff is not None:
            time_mask = time_points >= min_time_cutoff
            time_points = time_points[time_mask]
        if max_time_cutoff is not None:
            time_mask = time_points <= max_time_cutoff
            time_points = time_points[time_mask]

        # Extract trajectories for each condition
        for condition_name in circuit_config.condition_params.keys():
            condition_mask = combined_parameters["condition"] == condition_name
            condition_indices = combined_parameters.index[condition_mask]

            for idx in condition_indices:
                trajectory = simulation_results.observables[idx]["obs_Protein_GFP"]

                # Apply time filtering to trajectory
                if min_time_cutoff is not None or max_time_cutoff is not None:
                    original_time = circuit_config.tspan
                    if min_time_cutoff is not None:
                        start_idx = np.where(original_time >= min_time_cutoff)[0][0]
                    else:
                        start_idx = 0
                    if max_time_cutoff is not None:
                        end_idx = np.where(original_time <= max_time_cutoff)[0][-1] + 1
                    else:
                        end_idx = len(trajectory)
                    trajectory = trajectory[start_idx:end_idx]

                # Create trajectory dataframe
                trajectory_df = pd.DataFrame(
                    {
                        "time": time_points,
                        "protein_concentration": trajectory,
                        "circuit": circuit_config.name,
                        "condition": condition_name,
                        "circuit_idx": circuit_idx,
                        "simulation_idx": idx,
                    }
                )
                trajectory_data_list.append(trajectory_df)

    # Combine all trajectory data
    all_trajectories_df = pd.concat(trajectory_data_list, ignore_index=True)

    # Calculate statistical summaries
    if summary_type == "median_iqr":
        summary_stats = (
            all_trajectories_df.groupby(["circuit", "condition", "time"])[
                "protein_concentration"
            ]
            .agg(
                [
                    ("median", "median"),
                    ("q25", lambda x: np.percentile(x, 25)),
                    ("q75", lambda x: np.percentile(x, 75)),
                ]
            )
            .reset_index()
        )
        summary_stats["lower"] = summary_stats["q25"]
        summary_stats["upper"] = summary_stats["q75"]
        summary_stats["central"] = summary_stats["median"]

    elif summary_type == "mean_std":
        summary_stats = (
            all_trajectories_df.groupby(["circuit", "condition", "time"])[
                "protein_concentration"
            ]
            .agg([("mean", "mean"), ("std", "std")])
            .reset_index()
        )
        summary_stats["lower"] = summary_stats["mean"] - summary_stats["std"]
        summary_stats["upper"] = summary_stats["mean"] + summary_stats["std"]
        summary_stats["central"] = summary_stats["mean"]

    else:
        raise ValueError(
            f"Unknown summary_type: {summary_type}. Use 'median_iqr' or 'mean_std'"
        )

    # Plot each circuit
    for circuit_idx, circuit_data in simulation_data_dict.items():
        ax = axes[circuit_idx]
        circuit_config = circuit_data["config"]
        circuit_name = circuit_config.name

        # Filter summary data for this circuit
        circuit_summary = summary_stats[summary_stats["circuit"] == circuit_name]

        # Get condition colors
        condition_names = list(circuit_config.condition_params.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(condition_names)))

        # Plot each condition
        for condition_idx, condition_name in enumerate(condition_names):
            condition_data = circuit_summary[
                circuit_summary["condition"] == condition_name
            ]
            color = colors[condition_idx]

            # Plot central line
            ax.plot(
                condition_data["time"],
                condition_data["central"],
                color=color,
                linewidth=2,
                label=condition_name,
            )

            # Plot ribbon
            ax.fill_between(
                condition_data["time"],
                condition_data["lower"],
                condition_data["upper"],
                alpha=ribbon_alpha,
                color=color,
            )

            # Add experimental data if available
            experimental_data = circuit_config.experimental_data
            condition_exp_data = experimental_data[
                experimental_data["condition"] == condition_name
            ]

            if len(condition_exp_data) > 0:
                ax.scatter(
                    condition_exp_data["time"],
                    condition_exp_data["fluorescence"],
                    color=color,
                    alpha=0.7,
                    s=15,
                    marker="o",
                    edgecolors="black",
                    linewidth=0.5,
                )

        # Format subplot
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Protein Concentration (nM)")
        ax.set_title(f"{circuit_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_circuits, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title with summary type
    summary_label = "Median ± IQR" if summary_type == "median_iqr" else "Mean ± Std"
    fig.suptitle(f"Circuit Simulation Summaries ({summary_label})", fontsize=16, y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Statistical summary plot saved: {save_path}")

    plt.show()
