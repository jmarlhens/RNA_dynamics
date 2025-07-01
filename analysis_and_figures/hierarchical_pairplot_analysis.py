import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde


# ──────────────────────────────────────────────────────────────
def _ridgeline_panel(
    ax,
    data: pd.DataFrame,
    param: str,
    *,
    group: str = "Circuit",
    bw_adjust: float = 0.5,
    offset: float = 1.0,
    fill_alpha: float = 0.9,
    palette="cubehelix",
    label_pad_frac: float = 0.03,
    idx: int = 1,
    label_on_idx: int = 1,
    left_anchor: float = None,  # ▲ PATCH ▲
):
    """Draw one ridgeline panel.  `left_anchor` sets the furthest-left point used
    for positioning the circuit‐name labels."""
    data_min, data_max = data[param].min(), data[param].max()

    # if a left-anchor was provided, extend xmin further left
    xmin = min(data_min, left_anchor) if left_anchor is not None else data_min
    xmax = data_max
    xrng = xmax - xmin
    xs = np.linspace(xmin, xmax, 256)

    groups = pd.unique(data[group])
    colours = sns.color_palette(palette, len(groups))

    for j, g in enumerate(groups):
        subset = data.loc[data[group] == g, param].dropna()
        if subset.empty:
            continue
        kde = gaussian_kde(subset, bw_adjust / subset.std(ddof=1))
        ys = kde(xs)
        ys /= ys.max()
        base = j * offset

        ax.fill_between(xs, base, ys + base, color=colours[j], alpha=fill_alpha, lw=0)
        ax.plot(xs, ys + base, color="white", lw=1.2, zorder=3)

        # draw labels only on chosen subplot
        if idx == label_on_idx:
            ax.text(
                xmin - xrng * label_pad_frac,
                base + offset * 0.18,
                str(g),
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
                color=colours[j],
                clip_on=False,
            )

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.25)
    return ax


# ──────────────────────────────────────────────────────────────


def create_hierarchical_histogram_grid(
    pairplot_dataset: pd.DataFrame,
    parameter_names_list,
    output_filepath: str,
    *,
    plot_kind: str = "kde",
    share_y: bool = False,
    legend_on_idx: int = 1,
    ridge_offset: float = -1.0,
    ridge_bw_adjust: float = 0.5,
    ridge_label_pad: float = 1.1,
):
    """
    Grid of parameter distributions (Circuits vs Priors)

    New: overlays ±1 σ around the prior mean if a *log10stdv* value is available.
    """
    # sns.set_theme(style="white")

    circ = pairplot_dataset.query("type == 'Circuit'").copy()
    prior = pairplot_dataset.query("type == 'Prior'")
    alpha = pairplot_dataset.query("type == 'Global'")

    n_cols, n_rows = 4, int(np.ceil(len(parameter_names_list) / 4))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.8 * n_rows),
        sharey=(share_y if plot_kind == "kde" else False),
    )
    axes = axes.flatten()

    global_kde_max = 0.0  # for y-sharing

    for idx, param in enumerate(parameter_names_list):
        ax = axes[idx]

        # ▲ PATCH ▲ ——————————————————————————————————————————————
        # 1. Look up Prior μ and σ *first*
        prior_mean = np.nan
        prior_std = np.nan
        if not prior.empty and not np.isnan(prior[param].iloc[0]):
            prior_mean = prior[param].iloc[0]

            # try all reasonable std-column names
            std_candidates = [
                f"{param}_log10stdev",
                f"{param}_log10stdv",
                "log10stdev",
                "log10stdv",
                f"{param}_stdev",
                f"{param}_stdv",
                f"{param}_std",
            ]
            prior_std = next(
                (prior[c].iloc[0] for c in std_candidates if c in prior),
                np.nan,
            )

        # 2. Determine the left-most x we must show
        circuit_min = circ[param].min()
        prior_lower = prior_mean - prior_std if not np.isnan(prior_std) else prior_mean
        label_anchor = min(circuit_min, prior_lower)
        # ————————————————————————————————————————————————————————

        # --- KDE or ridgeline --------------------------------------------------
        if plot_kind == "kde":
            kde_ax = sns.kdeplot(
                data=circ,
                x=param,
                hue="Circuit",
                fill=True,
                alpha=0.3,
                linewidth=1.6,
                bw_adjust=3.0,
                common_norm=False,
                legend=(idx == legend_on_idx),
                ax=ax,
            )
            global_kde_max = max(global_kde_max, kde_ax.get_ylim()[1])

        else:  # ridgeline
            _ridgeline_panel(
                ax,
                circ[[param, "Circuit"]],
                param=param,
                bw_adjust=ridge_bw_adjust,
                offset=ridge_offset,
                label_pad_frac=ridge_label_pad,
                # idx=idx,
                # label_on_idx=legend_on_idx,
                left_anchor=label_anchor,  # ▲ PATCH ▲
            )

        # --- vertical reference(s) -------------------------------------------
        if not np.isnan(prior_mean):
            ax.axvline(
                prior_mean,
                ls="--",
                lw=1.5,
                color="red",
                label=("Prior μ" if idx == 1 else None),
                zorder=9,
            )

            if not np.isnan(prior_std):
                ax.axvline(
                    prior_mean - prior_std,
                    ls=":",
                    lw=1.0,
                    color="red",
                    label=("±1 σ" if idx == 1 else None),
                    zorder=9,
                )
                ax.axvline(
                    prior_mean + prior_std, ls=":", lw=1.0, color="red", zorder=9
                )

        if not alpha.empty and not np.isnan(alpha[param].iloc[0]):
            ax.axvline(
                alpha[param].iloc[0],
                lw=2.3,
                color="orange",
                label=("Global α" if idx == 1 else None),
                zorder=9,
            )

        # --- cosmetics / legends stay exactly as before -----------------------
        ax.set_xlim(left=min(label_anchor, ax.get_xlim()[0]))  # ▲ PATCH ▲

        # Title and labels
        ax.set_title(param, fontsize=10)

        # Show legend only on the first pane
        if legend_on_idx < len(axes):
            axes[legend_on_idx].legend(
                loc="upper left", fontsize=9, frameon=True, fancybox=True, shadow=True
            )

    # true y-sharing for KDE mode
    if plot_kind == "kde" and share_y:
        for a in axes[: len(parameter_names_list)]:
            a.set_ylim(0, global_kde_max * 1.05)

    # common x axis name only bottom
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Log10 Parameter Value", fontsize=11)

    # hide unused panes
    for j in range(len(parameter_names_list), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Parameter Distributions: Circuits vs Priors",
        fontsize=22,
        y=0.995,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    fig.savefig(output_filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved: {output_filepath}")


def create_circuit_prior_comparison_pairplot(
    circuit_comparison_dataset,
    fitted_parameter_names,
    pairplot_filepath,
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


def create_circuit_correlation_matrices(
    individual_circuit_fits, fitted_parameter_names, output_visualization_directory
):
    """
    Create correlation matrices for each circuit's parameters
    """
    correlation_matrices = {}
    n_circuits = len(individual_circuit_fits)

    # Calculate number of rows/cols for subplot layout
    n_cols = min(3, n_circuits)  # Max 3 columns
    n_rows = int(np.ceil(n_circuits / n_cols))

    # Create figure for all correlation matrices
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Handle single circuit case
    if n_circuits == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for idx, (circuit_name, circuit_data) in enumerate(individual_circuit_fits.items()):
        ax = axes[idx]

        # Get parameter columns that exist in this circuit's data
        param_cols = [p for p in fitted_parameter_names if p in circuit_data.columns]

        # Calculate correlation matrix
        if len(param_cols) > 1:
            log_params = circuit_data[param_cols]
            corr_matrix = log_params.corr()
            correlation_matrices[circuit_name] = corr_matrix

            # Create heatmap
            sns.heatmap(
                corr_matrix,
                # annot=True,
                # fmt=".2f",
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={"shrink": 0.8},
                ax=ax,
            )

            ax.set_title(
                f"{circuit_name}\nParameter Correlations",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Parameters")
            ax.set_ylabel("Parameters")

            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            # Not enough parameters for correlation
            ax.text(
                0.5,
                0.5,
                f"{circuit_name}\nInsufficient parameters\nfor correlation analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide empty subplots
    for idx in range(n_circuits, len(axes)):
        axes[idx].set_visible(False)

    # Set main title
    fig.suptitle(
        "Parameter Correlation Matrices by Circuit\n(Log10 space)", fontsize=16, y=0.98
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    correlation_filepath = os.path.join(
        output_visualization_directory, "circuit_parameter_correlations.png"
    )
    plt.savefig(correlation_filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Circuit correlation matrices saved: {correlation_filepath}")

    # Also save individual correlation matrices as CSV files
    for circuit_name, corr_matrix in correlation_matrices.items():
        csv_filepath = os.path.join(
            output_visualization_directory, f"correlation_matrix_{circuit_name}.csv"
        )
        corr_matrix.to_csv(csv_filepath)
        print(f"Saved correlation matrix for {circuit_name}: {csv_filepath}")

    return correlation_matrices


def create_circuit_correlation_summary(
    correlation_matrices, output_visualization_directory
):
    """
    Create a summary comparing correlation patterns across circuits
    """

    if not correlation_matrices:
        print("No correlation matrices to summarize")
        return

    # Get all parameter pairs
    all_params = set()
    for matrix in correlation_matrices.values():
        all_params.update(matrix.columns)
    all_params = sorted(list(all_params))

    # Create summary DataFrame
    summary_data = []
    for circuit_name, corr_matrix in correlation_matrices.items():
        for i, param1 in enumerate(all_params):
            for j, param2 in enumerate(all_params):
                if (
                    i < j
                    and param1 in corr_matrix.columns
                    and param2 in corr_matrix.columns
                ):
                    correlation = corr_matrix.loc[param1, param2]
                    summary_data.append(
                        {
                            "circuit": circuit_name,
                            "param1": param1,
                            "param2": param2,
                            "param_pair": f"{param1} vs {param2}",
                            "correlation": correlation,
                            "abs_correlation": abs(correlation),
                        }
                    )

    summary_df = pd.DataFrame(summary_data)

    if summary_df.empty:
        print("No correlation data to summarize")
        return summary_df

    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Correlation heatmap by parameter pair
    pivot_df = summary_df.pivot(
        index="param_pair", columns="circuit", values="correlation"
    )
    sns.heatmap(
        pivot_df,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax1,
        cbar_kws={"shrink": 0.8},
    )
    ax1.set_title("Parameter Correlations Across Circuits", fontweight="bold")
    ax1.set_xlabel("Circuit")
    ax1.set_ylabel("Parameter Pairs")
    ax1.tick_params(axis="x", rotation=45)

    # Plot 2: Distribution of correlation strengths
    sns.boxplot(data=summary_df, x="circuit", y="abs_correlation", ax=ax2)
    ax2.set_title("Distribution of Correlation Strengths", fontweight="bold")
    ax2.set_xlabel("Circuit")
    ax2.set_ylabel("Absolute Correlation")
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    # Save summary plot
    summary_filepath = os.path.join(
        output_visualization_directory, "correlation_summary_comparison.png"
    )
    plt.savefig(summary_filepath, dpi=300, bbox_inches="tight")
    plt.close()

    # Save summary data
    summary_csv_filepath = os.path.join(
        output_visualization_directory, "correlation_summary.csv"
    )
    summary_df.to_csv(summary_csv_filepath, index=False)

    print(f"Correlation summary saved: {summary_filepath}")
    print(f"Correlation summary data saved: {summary_csv_filepath}")

    return summary_df
