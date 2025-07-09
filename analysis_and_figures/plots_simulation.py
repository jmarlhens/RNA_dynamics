import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Literal


def extract_circuit_experimental_ylimits(simulation_data_dict: dict) -> dict:
    """Extract y-axis limits from experimental fluorescence data per circuit."""
    experimental_ylimits = {}
    for circuit_key, circuit_data in simulation_data_dict.items():
        experimental_data = circuit_data["config"].experimental_data
        ylimit_max = experimental_data["fluorescence"].max() * 1.1
        experimental_ylimits[circuit_key] = (0, ylimit_max)
    return experimental_ylimits


def compute_trajectory_statistical_summaries(
    trajectory_dataframe: pd.DataFrame,
    simulation_mode: str,
    summary_type: str,
    percentile_bounds: tuple,
) -> Optional[pd.DataFrame]:
    """Compute statistical summaries for trajectory data. Returns None for individual mode."""
    if simulation_mode != "summary":
        return None

    if summary_type == "median_iqr":
        lower_percentile, upper_percentile = percentile_bounds
        summary_dataframe = (
            trajectory_dataframe.groupby(["circuit", "condition", "time"])[
                "protein_concentration"
            ]
            .agg(
                [
                    ("central", "median"),
                    ("lower", lambda x: np.percentile(x, lower_percentile)),
                    ("upper", lambda x: np.percentile(x, upper_percentile)),
                ]
            )
            .reset_index()
        )
    else:  # mean_std
        grouped_trajectory_stats = (
            trajectory_dataframe.groupby(["circuit", "condition", "time"])[
                "protein_concentration"
            ]
            .agg([("mean", "mean"), ("std", "std")])
            .reset_index()
        )
        summary_dataframe = grouped_trajectory_stats.copy()
        summary_dataframe["central"] = grouped_trajectory_stats["mean"]
        summary_dataframe["lower"] = (
            grouped_trajectory_stats["mean"] - grouped_trajectory_stats["std"]
        )
        summary_dataframe["upper"] = (
            grouped_trajectory_stats["mean"] + grouped_trajectory_stats["std"]
        )

    return summary_dataframe


def extract_trajectory_data(
    simulation_data_dict: dict, results_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """Extract trajectory data into unified DataFrame."""
    trajectory_records = []

    for circuit_key, circuit_data in simulation_data_dict.items():
        circuit_config = circuit_data["config"]
        combined_params = circuit_data["combined_params"]
        simulation_results = circuit_data["simulation_results"]
        time_points = circuit_config.tspan

        for condition_name in circuit_config.condition_params.keys():
            condition_mask = combined_params["condition"] == condition_name
            condition_indices = combined_params.index[condition_mask]
            param_set_indices = combined_params.loc[condition_mask, "param_set_idx"]

            for idx, param_set_idx in zip(condition_indices, param_set_indices):
                trajectory = simulation_results.observables[idx]["obs_Protein_GFP"]

                # Get likelihood
                likelihood_col = ("likelihood", circuit_config.name, condition_name)
                log_likelihood = (
                    results_dataframe.loc[param_set_idx, likelihood_col]
                    if likelihood_col in results_dataframe.columns
                    and param_set_idx < len(results_dataframe)
                    else 0.0
                )

                # Store all time points
                for time_val, protein_val in zip(time_points, trajectory):
                    trajectory_records.append(
                        {
                            "time": time_val,
                            "protein_concentration": protein_val,
                            "circuit": circuit_key,
                            "condition": condition_name,
                            "param_set_idx": param_set_idx,
                            "log_likelihood": log_likelihood,
                        }
                    )

    return pd.DataFrame(trajectory_records)


def plot_circuit_conditions_overlay(
    simulation_data_dict: dict,
    results_dataframe: pd.DataFrame,
    simulation_mode: Literal["individual", "summary"] = "individual",
    summary_type: Literal["median_iqr", "mean_std"] = "median_iqr",
    percentile_bounds: tuple = (25, 75),
    individual_alpha: float = 0.15,
    ribbon_alpha: float = 0.25,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot circuits with conditions overlaid: experimental scatter (left) | simulations (right).
    Uses consistent condition-based colors for both individual trajectories and statistical summaries.
    """

    trajectory_dataframe = extract_trajectory_data(
        simulation_data_dict, results_dataframe
    )
    circuit_count = len(simulation_data_dict)

    if figsize is None:
        figsize = (12, 4 * circuit_count)

    figure = plt.figure(figsize=figsize)

    # Extract experimental y-limits and statistical summaries using helpers
    experimental_y_limits = extract_circuit_experimental_ylimits(simulation_data_dict)
    summary_dataframe = compute_trajectory_statistical_summaries(
        trajectory_dataframe, simulation_mode, summary_type, percentile_bounds
    )

    # Plot each circuit as one row: experimental_axis | simulation_axis
    for circuit_idx, (circuit_key, circuit_data) in enumerate(
        simulation_data_dict.items()
    ):
        circuit_config = circuit_data["config"]
        condition_names = list(circuit_config.condition_params.keys())

        # Generate consistent colors for conditions across both subplots
        condition_colors = plt.cm.Set1(np.linspace(0, 1, len(condition_names)))
        condition_color_mapping = dict(zip(condition_names, condition_colors))

        # Create subplot pair for this circuit
        experimental_axis = plt.subplot(circuit_count, 2, circuit_idx * 2 + 1)
        simulation_axis = plt.subplot(circuit_count, 2, circuit_idx * 2 + 2)

        # Plot data for each condition (unified experimental + simulation loop)
        circuit_trajectories = trajectory_dataframe[
            trajectory_dataframe["circuit"] == circuit_key
        ]

        for condition_name in condition_names:
            condition_color = condition_color_mapping[condition_name]

            # Plot experimental data (left subplot)
            condition_experimental_data = circuit_config.experimental_data[
                circuit_config.experimental_data["condition"] == condition_name
            ]

            experimental_axis.scatter(
                condition_experimental_data["time"],
                condition_experimental_data["fluorescence"],
                color=condition_color,
                alpha=0.7,
                s=15,
                label=condition_name,
                marker="o",
                # edgecolors="black",
                # linewidth=0.5,
            )

            # Plot simulation data (right subplot) with unified color scheme
            if simulation_mode == "individual":
                condition_trajectories = circuit_trajectories[
                    circuit_trajectories["condition"] == condition_name
                ]

                for param_set_idx in condition_trajectories["param_set_idx"].unique():
                    param_trajectory = condition_trajectories[
                        condition_trajectories["param_set_idx"] == param_set_idx
                    ]

                    simulation_axis.plot(
                        param_trajectory["time"],
                        param_trajectory["protein_concentration"],
                        color=condition_color,
                        alpha=individual_alpha,
                        zorder=1,
                    )

                # Add condition label only once per condition
                if len(condition_trajectories) > 0:
                    simulation_axis.plot(
                        [], [], color=condition_color, label=condition_name, linewidth=2
                    )

            else:  # summary mode
                circuit_summaries = summary_dataframe[
                    summary_dataframe["circuit"] == circuit_key
                ]
                condition_summary = circuit_summaries[
                    circuit_summaries["condition"] == condition_name
                ]

                if len(condition_summary) > 0:
                    simulation_axis.plot(
                        condition_summary["time"],
                        condition_summary["central"],
                        color=condition_color,
                        linewidth=2,
                        label=condition_name,
                    )
                    simulation_axis.fill_between(
                        condition_summary["time"],
                        condition_summary["lower"],
                        condition_summary["upper"],
                        alpha=ribbon_alpha,
                        color=condition_color,
                    )

        # Set axis properties for both subplots
        for axis, title_suffix in [
            (experimental_axis, "Experimental"),
            (simulation_axis, "Simulation"),
        ]:
            axis.set_ylim(*experimental_y_limits[circuit_key])
            axis.set_xlabel("Time (min)")
            axis.set_ylabel("Protein Concentration (nM)")
            axis.set_title(f"{circuit_config.name} - {title_suffix}")
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right", fontsize="small")

    # Set figure title
    if simulation_mode == "individual":
        figure_title = (
            "Circuit Conditions Overlay: Experimental | Individual Trajectories"
        )
    else:
        summary_label = "Median ± IQR" if summary_type == "median_iqr" else "Mean ± Std"
        figure_title = f"Circuit Conditions Overlay: Experimental | {summary_label}"

    plt.suptitle(figure_title, fontsize=16, y=0.98)
    plt.tight_layout()
    return figure


def plot_circuit_simulations(
    simulation_data_dict: dict,
    results_dataframe: pd.DataFrame,
    plot_mode: Literal["individual", "summary"] = "individual",
    likelihood_percentile_range: int = 40,
    summary_type: Literal["median_iqr", "mean_std"] = "median_iqr",
    percentile_bounds: tuple = (25, 75),
    individual_alpha: float = 0.3,
    ribbon_alpha: float = 0.25,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Unified function to plot circuit simulation results.

    Parameters:
    -----------
    simulation_data_dict : dict
        Dictionary containing simulation data for each circuit
    results_dataframe : pd.DataFrame
        DataFrame containing likelihood results
    plot_mode : {"individual", "summary"}
        Whether to plot individual trajectories or statistical summaries
    likelihood_percentile_range : int
        Range for likelihood-based coloring (only used in individual mode)
    summary_type : {"median_iqr", "mean_std"}
        Type of statistical summary (only used in summary mode)
    percentile_bounds : tuple
        Lower and upper percentiles for IQR calculation
    individual_alpha : float
        Alpha transparency for individual trajectory lines
    ribbon_alpha : float
        Alpha transparency for summary ribbons
    figsize : tuple, optional
        Figure size tuple

    Returns:
    --------
    plt.Figure
        The generated figure
    """

    trajectory_df = extract_trajectory_data(simulation_data_dict, results_dataframe)

    # Setup subplot grid
    circuit_count = len(simulation_data_dict)
    max_conditions = max(
        len(data["config"].condition_params) for data in simulation_data_dict.values()
    )

    if figsize is None:
        figsize = (4.8 * max_conditions, 4 * circuit_count)

    fig = plt.figure(figsize=figsize)

    # Extract experimental y-limits and statistical summaries using helpers
    experimental_y_limits = extract_circuit_experimental_ylimits(simulation_data_dict)
    summary_df = compute_trajectory_statistical_summaries(
        trajectory_df, plot_mode, summary_type, percentile_bounds
    )

    # Plot each circuit
    for circuit_idx, (circuit_key, circuit_data) in enumerate(
        simulation_data_dict.items()
    ):
        circuit_config = circuit_data["config"]
        condition_names = list(circuit_config.condition_params.keys())

        # Generate colors for conditions (used in summary mode)
        condition_colors = plt.cm.Set1(np.linspace(0, 1, len(condition_names)))

        for condition_idx, condition_name in enumerate(condition_names):
            ax = plt.subplot(
                circuit_count,
                max_conditions,
                circuit_idx * max_conditions + condition_idx + 1,
            )

            # Plot simulation data based on mode
            if plot_mode == "individual":
                circuit_trajectories = trajectory_df[
                    trajectory_df["circuit"] == circuit_key
                ]
                condition_trajectories = circuit_trajectories[
                    circuit_trajectories["condition"] == condition_name
                ]

                if len(condition_trajectories) > 0:
                    # Setup likelihood colormap
                    likelihoods = condition_trajectories["log_likelihood"]
                    min_ll = np.percentile(likelihoods, likelihood_percentile_range)
                    max_ll = np.percentile(
                        likelihoods, 100 - likelihood_percentile_range
                    )
                    norm = plt.Normalize(vmin=min_ll, vmax=max_ll)
                    cmap = plt.cm.viridis

                    # Plot individual trajectories
                    for param_set_idx in condition_trajectories[
                        "param_set_idx"
                    ].unique():
                        param_data = condition_trajectories[
                            condition_trajectories["param_set_idx"] == param_set_idx
                        ]
                        likelihood = param_data["log_likelihood"].iloc[0]

                        ax.plot(
                            param_data["time"],
                            param_data["protein_concentration"],
                            color=cmap(norm(likelihood)),
                            alpha=individual_alpha,
                            zorder=1,
                        )

                    plt.colorbar(
                        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax,
                        label="Log-likelihood",
                    )

            else:  # summary mode
                circuit_summaries = summary_df[summary_df["circuit"] == circuit_key]
                condition_data = circuit_summaries[
                    circuit_summaries["condition"] == condition_name
                ]

                if len(condition_data) > 0:
                    color = condition_colors[condition_idx]

                    # Plot ribbon and central line
                    ax.plot(
                        condition_data["time"],
                        condition_data["central"],
                        color=color,
                        linewidth=2,
                    )
                    ax.fill_between(
                        condition_data["time"],
                        condition_data["lower"],
                        condition_data["upper"],
                        alpha=ribbon_alpha,
                        color=color,
                    )

            # Plot experimental data (same for both modes)
            experimental_data = circuit_config.experimental_data[
                circuit_config.experimental_data["condition"] == condition_name
            ]

            if plot_mode == "individual":
                ax.scatter(
                    experimental_data["time"],
                    experimental_data["fluorescence"],
                    color="red",
                    alpha=0.6,
                    s=5,
                    zorder=3,
                )
            else:  # summary mode
                color = condition_colors[condition_idx]
                ax.scatter(
                    experimental_data["time"],
                    experimental_data["fluorescence"],
                    color=color,
                    alpha=0.7,
                    s=15,
                    marker="o",
                    # edgecolors="black",
                    # linewidth=0.5,
                )

            # Set common axis properties
            ax.set_ylim(*experimental_y_limits[circuit_key])
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Protein Concentration (nM)" if condition_idx == 0 else "")

            if plot_mode == "individual":
                ax.set_title(condition_name)
            else:
                ax.set_title(f"{circuit_config.name} - {condition_name}")

            ax.grid(True, alpha=0.3)

    # Set figure title based on mode
    if plot_mode == "individual":
        plt.suptitle(
            "Individual Circuit Trajectories (colored by likelihood)",
            fontsize=16,
            y=0.98,
        )
    else:
        summary_label = "Median ± IQR" if summary_type == "median_iqr" else "Mean ± Std"
        plt.suptitle(
            f"Circuit Simulation Results ({summary_label})", fontsize=16, y=0.98
        )

    plt.tight_layout()
    return fig
