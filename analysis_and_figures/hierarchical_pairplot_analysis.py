import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde


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
    left_anchor: float = None,
    global_alpha_style=None,
):
    """Draw one ridgeline panel with special styling for global alpha parameters"""
    data_min, data_max = data[param].min(), data[param].max()

    # if a left-anchor was provided, extend min further left
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

        # Special styling for Global_Alpha
        if g == "Global_Alpha" and global_alpha_style:
            color = global_alpha_style.get("color", "darkred")
            alpha = global_alpha_style.get("alpha", 0.7)
            edge_color = "white"
            edge_width = 1.2
        else:
            color = colours[j]
            alpha = fill_alpha
            edge_color = "white"
            edge_width = 1.2

        ax.fill_between(xs, base, ys + base, color=color, alpha=alpha, lw=0)
        ax.plot(xs, ys + base, color=edge_color, lw=edge_width, zorder=3)

        # draw labels only on chosen subplot
        if idx == label_on_idx:
            label_color = color if g == "Global_Alpha" else colours[j]
            ax.text(
                xmin - xrng * label_pad_frac,
                base + offset * 0.18,
                str(g),
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
                color=label_color,
                clip_on=False,
            )

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.25)
    return ax


# Modified histogram grid function to use enhanced ridgeline panel
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
    Grid of parameter distributions (Circuits vs Priors) with enhanced global alpha visualization
    """
    # Include both Circuit and Global types to show circuit-specific AND global alpha parameters
    circuit_and_global_data = pairplot_dataset.query(
        "type in ['Circuit', 'Global']"
    ).copy()
    prior_data = pairplot_dataset.query("type == 'Prior'")

    n_cols, n_rows = 4, int(np.ceil(len(parameter_names_list) / 4))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.8 * n_rows),
        sharey=(share_y if plot_kind == "kde" else False),
    )
    axes = axes.flatten()

    # Define global alpha styling
    global_alpha_style = {
        "color": "darkred",
        "alpha": 0.8,
        "edge_color": "red",
        "edge_width": 2.5,
    }

    for idx, param in enumerate(parameter_names_list):
        ax = axes[idx]

        # Look up Prior μ and σ
        prior_mean = np.nan
        prior_std = np.nan
        if not prior_data.empty and not np.isnan(prior_data[param].iloc[0]):
            prior_mean = prior_data[param].iloc[0]

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
                (prior_data[c].iloc[0] for c in std_candidates if c in prior_data),
                np.nan,
            )

        # Determine the left-most x coordinate
        circuit_and_global_min = circuit_and_global_data[param].min()
        prior_lower = prior_mean - prior_std if not np.isnan(prior_std) else prior_mean
        label_anchor = min(circuit_and_global_min, prior_lower)

        if plot_kind == "kde":
            _ = sns.kdeplot(
                data=circuit_and_global_data,
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
        else:  # ridgeline
            _ridgeline_panel(
                ax,
                circuit_and_global_data[[param, "Circuit"]],
                param=param,
                bw_adjust=ridge_bw_adjust,
                offset=ridge_offset,
                label_pad_frac=ridge_label_pad,
                idx=idx,
                label_on_idx=legend_on_idx,
                left_anchor=label_anchor,
                global_alpha_style=global_alpha_style,
            )

        # Add prior reference lines
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

        ax.set_xlim(left=min(label_anchor, ax.get_xlim()[0]))
        ax.set_title(param, fontsize=10)

        # Show legend only on the first pane
        if legend_on_idx < len(axes):
            axes[legend_on_idx].legend(
                loc="upper left", fontsize=9, frameon=True, fancybox=True, shadow=True
            )

    # Common x-axis label only on bottom row
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Log10 Parameter Value", fontsize=11)

    # Hide unused panes
    for j in range(len(parameter_names_list), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Parameter Distributions: Circuits vs Global Alpha vs Priors",
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

    # plt.tight_layout()
    plt.savefig(pairplot_filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Circuit-prior comparison pairplot saved: {pairplot_filepath}")


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
