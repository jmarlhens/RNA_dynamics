import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


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
