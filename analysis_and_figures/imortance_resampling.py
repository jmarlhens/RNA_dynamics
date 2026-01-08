import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_resampling_diagnostics(results: dict, output_path: str = None):
    """
    Plot diagnostics for importance resampling results.

    Parameters:
    -----------
    results : dict
        Results from perform_importance_resampling
    output_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    source = results["source_circuit"]
    target = results["target_circuit"]

    # Plot 1: Weight distribution
    ax = axes[0, 0]
    ax.hist(results["weights"], bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(
        1.0 / len(results["weights"]), color="r", linestyle="--", label="Uniform weight"
    )
    ax.set_xlabel("Importance Weight")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Weight Distribution\nESS = {results['effective_sample_size']:.1f} "
        f"({results['ess_ratio'] * 100:.1f}%)"
    )
    ax.legend()

    # Plot 2: Log weights distribution
    ax = axes[0, 1]
    log_w = results["log_weights"]
    ax.hist(log_w[np.isfinite(log_w)], bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Log Importance Weight")
    ax.set_ylabel("Count")
    ax.set_title("Log Weight Distribution")

    # Plot 3: Posterior comparison (before vs after resampling)
    ax = axes[0, 2]
    ax.hist(
        results["log_posterior_target"],
        bins=30,
        alpha=0.5,
        label="Before resampling",
        edgecolor="black",
    )
    ax.hist(
        results["log_posterior_resampled"],
        bins=30,
        alpha=0.5,
        label="After resampling",
        edgecolor="black",
    )
    ax.axvline(results["mean_log_posterior_before"], color="blue", linestyle="--")
    ax.axvline(results["mean_log_posterior_after"], color="orange", linestyle="--")
    ax.set_xlabel("Log Posterior (Target)")
    ax.set_ylabel("Count")
    ax.set_title("Posterior Distribution\nBefore vs After Resampling")
    ax.legend()

    # Plot 4: Source vs Target posterior scatter
    ax = axes[1, 0]
    ax.scatter(
        results["log_posterior_source"],
        results["log_posterior_target"],
        alpha=0.3,
        s=10,
    )
    ax.set_xlabel("Log Posterior (Source)")
    ax.set_ylabel("Log Posterior (Target)")
    ax.set_title("Source vs Target Posterior")

    # Add diagonal line
    min_val = min(
        results["log_posterior_source"].min(), results["log_posterior_target"].min()
    )
    max_val = max(
        results["log_posterior_source"].max(), results["log_posterior_target"].max()
    )
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

    # Plot 5: Cumulative distribution of weights
    ax = axes[1, 1]
    sorted_weights = np.sort(results["weights"])[::-1]
    cumsum_weights = np.cumsum(sorted_weights)
    ax.plot(range(1, len(sorted_weights) + 1), cumsum_weights)
    ax.axhline(0.5, color="r", linestyle="--", alpha=0.5, label="50%")
    ax.axhline(0.9, color="g", linestyle="--", alpha=0.5, label="90%")
    ax.set_xlabel("Number of Particles (sorted by weight)")
    ax.set_ylabel("Cumulative Weight")
    ax.set_title("Weight Concentration")
    ax.legend()

    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis("off")

    stats_text = f"""
    Importance Resampling Summary
    ─────────────────────────────
    Source: {source}
    Target: {target}

    Initial samples: {results["n_initial_samples"]}
    Resampled: {results["n_resampled"]}

    Effective Sample Size: {results["effective_sample_size"]:.1f}
    ESS Ratio: {results["ess_ratio"] * 100:.1f}%

    Log Posterior (Target):
      Before: mean={results["mean_log_posterior_before"]:.2f}
              max={results["best_log_posterior_before"]:.2f}
      After:  mean={results["mean_log_posterior_after"]:.2f}
              max={results["best_log_posterior_after"]:.2f}

    Improvement: {results["mean_log_posterior_after"] - results["mean_log_posterior_before"]:.2f}
    """

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        fontfamily="monospace",
        fontsize=10,
        verticalalignment="top",
    )

    plt.suptitle(f"Importance Resampling: {source} → {target}", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def summary_visualizations_importance_resampling(
    ess_matrix: pd.DataFrame,
    improvement_matrix: pd.DataFrame,
    n_resample: int,
    all_results: dict,
    output_directory: str,
    n_initial_samples: int,
):
    """
    Create summary visualizations from importance resampling results.

    Parameters:
    -----------
    all_results : dict
        Dictionary of all resampling results
    output_directory : str
        Directory to save results
    n_initial_samples : int
        Number of initial samples used

    Returns:
    --------
    None
    """
    # This function is now integrated into run_importance_resampling_analysis
    # Step 4: Create summary visualizations
    print("\n" + "=" * 60)
    print("Step 4: Creating summary visualizations")
    print("=" * 60)

    # ESS matrix heatmap
    plt.figure(figsize=(12, 10))
    mask = ess_matrix.isna()
    sns.heatmap(
        ess_matrix.astype(float),
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        mask=mask,
        cbar_kws={"label": "Effective Sample Size"},
        linewidths=0.5,
    )
    plt.title(f"Effective Sample Size Matrix\n(Initial samples: {n_initial_samples})")
    plt.xlabel("Target Circuit")
    plt.ylabel("Source Circuit")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, "ess_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # ESS ratio matrix
    ess_ratio_matrix = ess_matrix / n_initial_samples * 100
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        ess_ratio_matrix.astype(float),
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        mask=mask,
        cbar_kws={"label": "ESS Ratio (%)"},
        linewidths=0.5,
    )
    plt.title("ESS Ratio Matrix (%)")
    plt.xlabel("Target Circuit")
    plt.ylabel("Source Circuit")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, "ess_ratio_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Improvement matrix heatmap
    plt.figure(figsize=(12, 10))
    mask = improvement_matrix.isna()
    sns.heatmap(
        improvement_matrix.astype(float),
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        mask=mask,
        cbar_kws={"label": "Mean Log Posterior Improvement"},
        linewidths=0.5,
    )
    plt.title("Log Posterior Improvement After Resampling")
    plt.xlabel("Target Circuit")
    plt.ylabel("Source Circuit")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, "improvement_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save matrices to CSV
    ess_matrix.to_csv(os.path.join(output_directory, "ess_matrix.csv"))
    improvement_matrix.to_csv(os.path.join(output_directory, "improvement_matrix.csv"))

    # Create summary report
    summary_path = os.path.join(output_directory, "importance_resampling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Importance Resampling Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Initial samples per source: {n_initial_samples}\n")
        f.write(f"Resampled particles: {n_resample}\n\n")

        f.write("Results by source-target pair:\n")
        f.write("-" * 50 + "\n")

        for key, results in sorted(all_results.items()):
            source, target = key
            f.write(f"\n{source} → {target}:\n")
            f.write(
                f"  ESS: {results['effective_sample_size']:.1f} "
                f"({results['ess_ratio'] * 100:.1f}%)\n"
            )
            f.write(
                f"  Mean log posterior before: {results['mean_log_posterior_before']:.2f}\n"
            )
            f.write(
                f"  Mean log posterior after: {results['mean_log_posterior_after']:.2f}\n"
            )
            f.write(
                f"  Improvement: {results['mean_log_posterior_after'] - results['mean_log_posterior_before']:.2f}\n"
            )

    print(f"\nResults saved to: {output_directory}")
