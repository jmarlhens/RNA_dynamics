import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def create_circuit_prior_comparison_pairplot(
    circuit_comparison_dataset,
    fitted_parameter_names,
    output_visualization_directory,
    diagonal_visualization_type="kde",
    offdiagonal_visualization_type="scatter",
):
    """
    diagonal_visualization_type: 'hist', 'kde', 'auto'
    offdiagonal_visualization_type: 'scatter', 'kde', 'hist', 'reg'
    """

    circuit_posterior_samples = circuit_comparison_dataset[
        circuit_comparison_dataset["type"] == "Circuit"
    ]
    prior_mean_coordinates = circuit_comparison_dataset[
        circuit_comparison_dataset["type"] == "Prior"
    ]

    # Adaptive plot_kws based on visualization type
    if offdiagonal_visualization_type == "scatter":
        offdiagonal_plot_parameters = {"alpha": 0.8, "s": 12, "edgecolor": "none"}
    elif offdiagonal_visualization_type == "kde":
        offdiagonal_plot_parameters = {"alpha": 0.6, "levels": 5, "bw_adjust": 3.0}
    else:
        offdiagonal_plot_parameters = {"alpha": 0.6}

    # Adaptive diag_kws based on visualization type
    if diagonal_visualization_type == "hist":
        diagonal_plot_parameters = {
            "alpha": 0.7,
            "bins": 25,
            "edgecolor": "black",
            "linewidth": 0.5,
        }
    elif diagonal_visualization_type == "kde":
        diagonal_plot_parameters = {"alpha": 0.7, "linewidth": 2, "bw_adjust": 3.0}
    else:
        diagonal_plot_parameters = {"alpha": 0.7}

    # Create base pairplot with circuit distributions
    plt.style.use("default")
    pairplot_figure = sns.pairplot(
        data=circuit_posterior_samples,
        vars=fitted_parameter_names,
        hue="Circuit",
        diag_kind=diagonal_visualization_type,
        kind=offdiagonal_visualization_type,
        plot_kws=offdiagonal_plot_parameters,
        diag_kws=diagonal_plot_parameters,
    )

    # Overlay prior mean crosses on off-diagonal scatter plots
    for row_param_index, row_parameter_name in enumerate(fitted_parameter_names):
        for col_param_index, col_parameter_name in enumerate(fitted_parameter_names):
            subplot_axis = pairplot_figure.axes[row_param_index, col_param_index]

            if row_param_index != col_param_index:  # Off-diagonal scatter plots
                prior_x_coordinate = prior_mean_coordinates[col_parameter_name].iloc[0]
                prior_y_coordinate = prior_mean_coordinates[row_parameter_name].iloc[0]

                # Add prior mean cross
                subplot_axis.scatter(
                    prior_x_coordinate,
                    prior_y_coordinate,
                    marker="x",
                    s=120,
                    c="red",
                    linewidths=4,
                    label="Prior Mean",
                    zorder=10,
                )

            else:  # Diagonal histograms
                prior_parameter_mean = prior_mean_coordinates[row_parameter_name].iloc[
                    0
                ]

                # Add vertical line at prior mean
                subplot_axis.axvline(
                    prior_parameter_mean,
                    color="red",
                    linestyle="--",
                    linewidth=3,
                    label="Prior Mean",
                    zorder=10,
                )

    pairplot_figure.fig.suptitle(
        "Circuit-Specific Parameters vs Prior Means\n"
        "Red crosses: Prior means, Distributions: Circuit posteriors",
        y=1.02,
        fontsize=14,
    )

    # Adjust legend
    pairplot_figure._legend.set_title("Parameter Groups")
    pairplot_figure._legend.set_bbox_to_anchor((1.05, 0.8))

    plt.tight_layout()
    pairplot_filepath = os.path.join(
        output_visualization_directory, "circuit_prior_comparison_pairplot.png"
    )
    plt.savefig(pairplot_filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Circuit-prior comparison pairplot saved: {pairplot_filepath}")


def prepare_hierarchical_pairplot_data_processed(
    processed_df, param_names, circuit_names, n_samples=100
):
    """
    Prepare pairplot data from already-processed dataframe (no burn-in filtering)
    """

    # Sample for performance if needed
    if len(processed_df) > n_samples:
        df_sample = processed_df.sample(n=n_samples, random_state=42).reset_index(
            drop=True
        )
    else:
        df_sample = processed_df.reset_index(drop=True)

    print(f"Using {len(df_sample)} processed samples for pairplot")

    # Add sample_id for tracking
    df_sample["sample_id"] = df_sample.index

    # --- Global parameters ---
    global_cols = [
        f"alpha_{p}" for p in param_names if f"alpha_{p}" in df_sample.columns
    ]
    global_df = df_sample[["sample_id"] + global_cols].copy()
    global_df = global_df.rename(
        columns={
            f"alpha_{p}": p for p in param_names if f"alpha_{p}" in df_sample.columns
        }
    )
    global_df["type"] = "Global"
    global_df["circuit"] = "Global"

    # --- Circuit-specific parameters ---
    circuit_dfs = []
    for circuit in circuit_names:
        theta_cols = [
            f"theta_{circuit}_{p}"
            for p in param_names
            if f"theta_{circuit}_{p}" in df_sample.columns
        ]
        if not theta_cols:
            continue
        circuit_df = df_sample[["sample_id"] + theta_cols].copy()
        circuit_df = circuit_df.rename(
            columns={
                f"theta_{circuit}_{p}": p
                for p in param_names
                if f"theta_{circuit}_{p}" in df_sample.columns
            }
        )
        circuit_df["type"] = "Circuit"
        circuit_df["circuit"] = circuit
        circuit_dfs.append(circuit_df)

    # Combine all data
    pairplot_df = pd.concat([global_df] + circuit_dfs, ignore_index=True)
    pairplot_df["Circuit"] = pairplot_df["type"] + " - " + pairplot_df["circuit"]

    print(
        f"Created pairplot data with {len(pairplot_df)} rows ({len(df_sample)} samples × {1 + len(circuit_names)} parameter sets)"
    )

    return pairplot_df


def create_hierarchical_pairplot(pairplot_df, param_names, output_folder):
    """Create seaborn pairplot and correlation matrix"""

    # Create pairplot
    plt.style.use("default")
    g = sns.pairplot(
        data=pairplot_df,
        vars=param_names,
        hue="Circuit",
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 15, "edgecolor": "none"},
        # diag_kws={'alpha': 0.7, 'bins': 25, 'edgecolor': 'black', 'linewidth': 0.5}
    )

    g.fig.suptitle(
        "Hierarchical Parameters: Correlations & Distributions\n"
        "Colors distinguish Global (α) vs Circuit-specific (θ) parameters",
        y=1.02,
        fontsize=14,
    )

    g._legend.set_title("Parameter Type")
    g._legend.set_bbox_to_anchor((1.05, 0.8))

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "hierarchical_parameter_pairplot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved hierarchical parameter pairplot to {output_folder}")


def create_hierarchical_histogram_grid(
    pairplot_dataset, parameter_names_list, output_visualization_directory
):
    """
    Create histogram-only comparison plot for hierarchical parameters
    Now supports Prior markers as vertical lines
    """

    # Separate data types
    circuit_posterior_data = pairplot_dataset[pairplot_dataset["type"] == "Circuit"]
    prior_mean_coordinates = pairplot_dataset[pairplot_dataset["type"] == "Prior"]
    global_alpha_data = pairplot_dataset[pairplot_dataset["type"] == "Global"]

    # Set up grid layout
    n_cols = 4
    n_rows = int(np.ceil(len(parameter_names_list) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Plot histograms for each parameter
    for parameter_index, parameter_name in enumerate(parameter_names_list):
        subplot_axis = axes[parameter_index]

        # Get all circuit data for this parameter to calculate shared bins
        all_circuit_parameter_values = circuit_posterior_data[parameter_name].dropna()

        # Only show legend on first parameter
        legend = False
        if parameter_name == parameter_names_list[0]:
            legend = True

        if len(all_circuit_parameter_values) > 0:
            # Plot KDE with hue for different circuits
            sns.kdeplot(
                data=circuit_posterior_data,
                x=parameter_name,
                hue="Circuit",
                fill=True,
                common_norm=False,
                alpha=0.3,
                linewidth=1.5,
                bw_adjust=3.0,
                ax=subplot_axis,
                legend=legend,
            )

            # Add prior mean as vertical line (only add label on second parameter)
            if len(prior_mean_coordinates) > 0:
                prior_parameter_mean = prior_mean_coordinates[parameter_name].iloc[0]
                if not np.isnan(prior_parameter_mean):
                    line_label = (
                        "Prior Mean"
                        if parameter_name == parameter_names_list[1]
                        else None
                    )
                    subplot_axis.axvline(
                        prior_parameter_mean,
                        color="red",
                        linestyle="--",
                        linewidth=3,
                        label=line_label,
                        zorder=10,
                    )

            # Add global alpha if exists (only add label on second parameter)
            if len(global_alpha_data) > 0:
                alpha_parameter_mean = global_alpha_data[parameter_name].iloc[0]
                if not np.isnan(alpha_parameter_mean):
                    line_label = (
                        "Global α"
                        if parameter_name == parameter_names_list[1]
                        else None
                    )
                    subplot_axis.axvline(
                        alpha_parameter_mean,
                        color="orange",
                        linestyle="-",
                        linewidth=3,
                        label=line_label,
                        zorder=10,
                    )

        if legend:
            # Add legend for KDEs
            plt.setp(
                subplot_axis.get_legend().get_texts(), fontsize="8"
            )  # for legend text

        # Handle legend for second parameter (prior/alpha lines)
        if parameter_name == parameter_names_list[1]:
            # Remove KDE legend and show only vertical line legends
            legend = subplot_axis.get_legend()
            if legend:
                legend.remove()
            # The vertical lines with labels will automatically create a legend
            subplot_axis.legend(loc="upper right", fontsize=8, framealpha=0.9)

        # Customize subplot
        subplot_axis.set_title(f"{parameter_name}", fontsize=12, fontweight="bold")
        subplot_axis.set_xlabel("Parameter Value (log10)")
        subplot_axis.set_ylabel("Frequency")
        subplot_axis.grid(True, alpha=0.3)

    # Hide empty subplots
    for subplot_index in range(len(parameter_names_list), len(axes)):
        axes[subplot_index].set_visible(False)

    # Set main title with proper spacing
    fig.suptitle(
        "Parameter Distributions: Circuits vs Priors",
        fontsize=24,
        y=0.98,  # Move title up slightly to avoid overlap
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for title

    # Save figure
    histogram_grid_filepath = os.path.join(
        output_visualization_directory, "circuit_prior_histogram_grid.png"
    )
    plt.savefig(histogram_grid_filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Circuit-prior histogram grid saved: {histogram_grid_filepath}")


def convert_individual_to_theta_format(
    individual_circuit_results, param_names_to_fit, circuit_names_list
):
    """Convert individual circuit DataFrames to hierarchical θ parameter format"""
    theta_formatted_data = []

    for circuit_name, circuit_dataframe in individual_circuit_results.items():
        if circuit_name not in circuit_names_list:
            continue

        # Sample for performance
        sampled_circuit_data = (
            circuit_dataframe.sample(n=1000, random_state=42)
            if len(circuit_dataframe) > 1000
            else circuit_dataframe.copy()
        )

        # Format as hierarchical θ parameters
        sampled_circuit_data["sample_id"] = range(len(sampled_circuit_data))

        # Rename parameters to theta format
        theta_columns = {}
        for param in param_names_to_fit:
            if param in sampled_circuit_data.columns:
                theta_columns[param] = param  # Keep same name for now

        formatted_circuit_data = sampled_circuit_data[
            ["sample_id"] + list(theta_columns.keys())
        ].copy()
        formatted_circuit_data.rename(columns=theta_columns, inplace=True)
        formatted_circuit_data["type"] = "Circuit"
        formatted_circuit_data["circuit"] = circuit_name
        formatted_circuit_data["Circuit"] = f"Circuit - {circuit_name}"

        theta_formatted_data.append(formatted_circuit_data)

    return pd.concat(theta_formatted_data, ignore_index=True)


def estimate_global_alpha_means(
    individual_circuit_results, param_names_to_fit, circuit_names_list
):
    """Estimate global α parameters as point estimates from individual circuits"""
    all_circuit_samples = []

    for circuit_name, circuit_dataframe in individual_circuit_results.items():
        if circuit_name not in circuit_names_list:
            continue

        # Extract parameter columns
        circuit_params = circuit_dataframe[param_names_to_fit].values
        all_circuit_samples.append(circuit_params)

    # Combine all samples and compute means
    combined_samples = np.vstack(all_circuit_samples)
    alpha_estimates = np.mean(combined_samples, axis=0)

    # Create single-row DataFrame for α
    alpha_dataframe = pd.DataFrame(
        {
            "sample_id": [0],
            **{
                param: [alpha_estimates[i]]
                for i, param in enumerate(param_names_to_fit)
            },
            "type": "Global",
            "circuit": "Global",
            "Circuit": "Global - Alpha",
        }
    )

    return alpha_dataframe


def add_prior_samples_to_comparison(
    existing_pairplot_data, prior_csv_path, param_names_to_fit, n_prior_samples=1000
):
    """Add prior distribution samples to comparison data"""
    prior_parameters = pd.read_csv(prior_csv_path)
    prior_samples_dict = {"sample_id": range(n_prior_samples)}

    for _, prior_row in prior_parameters.iterrows():
        if prior_row["Parameter"] in param_names_to_fit:
            param_name = prior_row["Parameter"]
            log_mean = prior_row["Mean"]
            log_std = prior_row["Std"]

            samples = np.random.normal(log_mean, log_std, n_prior_samples)
            prior_samples_dict[param_name] = samples

    # Fill missing parameters with NaN
    for param in param_names_to_fit:
        if param not in prior_samples_dict:
            prior_samples_dict[param] = [np.nan] * n_prior_samples

    prior_samples_dict.update(
        {
            "type": ["Prior"] * n_prior_samples,
            "circuit": ["Prior"] * n_prior_samples,
            "Circuit": ["Prior"] * n_prior_samples,
        }
    )

    prior_dataframe = pd.DataFrame(prior_samples_dict)
    return pd.concat([existing_pairplot_data, prior_dataframe], ignore_index=True)
