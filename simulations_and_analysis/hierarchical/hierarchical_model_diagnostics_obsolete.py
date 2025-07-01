from scipy.stats import invwishart
import seaborn as sns
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.circuit_utils import create_circuit_configs, setup_calibration
from likelihood_functions.hierarchical_likelihood.base_hierarchical import (
    HierarchicalCircuitFitter,
)


def load_individual_fits(data_dir="../../data/fit_data/individual_circuits/"):
    """Load all individual circuit fit results"""
    fit_files = glob.glob(f"{data_dir}/results_*.csv")

    all_fits = {}
    for file in fit_files:
        # Extract circuit name from filename
        filename = Path(file).stem
        parts = filename.split("_")
        # Handle circuit names with underscores
        circuit_name = "_".join(parts[1:-2])  # Skip 'results' and timestamp

        # Load data
        df = pd.read_csv(file)
        all_fits[circuit_name] = df
        print(f"Loaded {len(df)} samples for {circuit_name}")

    return all_fits


def get_best_fits(all_fits, n_best=100):
    """Extract n_best parameter sets from each circuit based on likelihood"""
    best_fits = {}

    for circuit, df in all_fits.items():
        # Sort by likelihood (or posterior if available)
        if "posterior" in df.columns:
            sorted_df = df.sort_values("posterior", ascending=False)
        else:
            sorted_df = df.sort_values("likelihood", ascending=False)

        best_fits[circuit] = sorted_df.head(n_best)

    return best_fits


def analyze_parameter_distributions(best_fits, parameters_to_fit):
    """Analyze how parameters vary across circuits"""
    # Collect parameter values across circuits
    param_data = {param: {} for param in parameters_to_fit}

    for circuit, df in best_fits.items():
        for param in parameters_to_fit:
            if param in df.columns:
                # Convert to log space for analysis
                param_data[param][circuit] = df[param].values

    # Calculate statistics
    stats = {}
    for param in parameters_to_fit:
        circuit_means = []
        circuit_stds = []

        for circuit, values in param_data[param].items():
            circuit_means.append(np.mean(values))
            circuit_stds.append(np.std(values))

        stats[param] = {
            "circuit_means": circuit_means,
            "circuit_stds": circuit_stds,
            "global_mean": np.mean(circuit_means),
            "between_circuit_std": np.std(circuit_means),
            "circuits": list(param_data[param].keys()),
        }

    return param_data, stats


def plot_parameter_distributions(param_data, stats, parameters_to_fit):
    """Visualize parameter distributions across circuits"""
    n_params = len(parameters_to_fit)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(40, 10))
    axes = axes.flatten()

    for idx, param in enumerate(parameters_to_fit):
        ax = axes[idx]

        # Plot distributions for each circuit
        for circuit, values in param_data[param].items():
            ax.hist(values, bins=30, alpha=0.5, label=circuit, density=True)

        # Add global mean
        ax.axvline(
            stats[param]["global_mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Global mean",
        )

        # Add ±2σ range
        global_mean = stats[param]["global_mean"]
        between_std = stats[param]["between_circuit_std"]
        ax.axvspan(
            global_mean - 2 * between_std,
            global_mean + 2 * between_std,
            alpha=0.2,
            color="red",
            label="±2σ between circuits",
        )

        ax.set_xlabel(f"log10({param})")
        ax.set_ylabel("Density")
        ax.set_title(f"{param}\nBetween-circuit σ = {between_std:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_correlation_matrix(best_fits, parameters_to_fit):
    """Create correlation matrices for each circuit and compare them"""
    correlation_matrices = {}

    for circuit, df in best_fits.items():
        # Get parameter columns that exist
        param_cols = [p for p in parameters_to_fit if p in df.columns]

        # Calculate correlation in log space
        log_params = df[param_cols]
        corr_matrix = log_params.corr()
        correlation_matrices[circuit] = corr_matrix

    # Plot correlation matrices
    n_circuits = len(correlation_matrices)
    fig, axes = plt.subplots(1, n_circuits, figsize=(9 * n_circuits, 8))

    if n_circuits == 1:
        axes = [axes]

    for idx, (circuit, corr_matrix) in enumerate(correlation_matrices.items()):
        ax = axes[idx]
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title(f"{circuit}")

    plt.tight_layout()
    return fig, correlation_matrices


def calculate_mahalanobis_distances(best_fits, stats, parameters_to_fit):
    """Calculate Mahalanobis distances between circuit parameters and global mean"""
    distances = {}

    # Get global mean and covariance
    all_params = []
    for circuit, df in best_fits.items():
        param_cols = [p for p in parameters_to_fit if p in df.columns]
        log_params = df[param_cols].values
        all_params.append(log_params)

    all_params = np.vstack(all_params)
    global_mean = np.mean(all_params, axis=0)
    global_cov = np.cov(all_params.T)

    # Calculate distances for each circuit
    for circuit, df in best_fits.items():
        param_cols = [p for p in parameters_to_fit if p in df.columns]
        circuit_mean = df[param_cols].mean()

        # Mahalanobis distance
        diff = circuit_mean - global_mean
        try:
            inv_cov = np.linalg.inv(global_cov)
            distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
            distances[circuit] = distance
        except np.linalg.LinAlgError:
            distances[circuit] = np.nan

    return distances, global_mean, global_cov


def analyze_individual_fits_main(parameters_to_fit, n_best=100):
    """Main function to analyze individual fits"""

    # Load all fits
    print("Loading individual fit results...")
    all_fits = load_individual_fits()

    if not all_fits:
        print("No individual fit files found!")
        return None

    # Get best fits
    print(f"\nExtracting top {n_best} fits from each circuit...")
    best_fits = get_best_fits(all_fits, n_best)

    # Analyze distributions
    print("\nAnalyzing parameter distributions...")
    param_data, stats = analyze_parameter_distributions(best_fits, parameters_to_fit)

    # Print summary statistics
    print("\n=== Parameter Distribution Summary ===")
    for param in parameters_to_fit:
        param_stats = stats[param]
        print(f"\n{param}:")
        print(f"  Global mean: {param_stats['global_mean']:.3f}")
        print(f"  Between-circuit std: {param_stats['between_circuit_std']:.3f}")
        print(
            f"  Coefficient of variation: {param_stats['between_circuit_std'] / abs(param_stats['global_mean']):.3f}"
        )

    # Calculate Mahalanobis distances
    distances, global_mean, global_cov = calculate_mahalanobis_distances(
        best_fits, stats, parameters_to_fit
    )

    print("\n=== Mahalanobis Distances from Global Mean ===")
    for circuit, distance in distances.items():
        print(f"{circuit}: {distance:.2f}")

    # Create visualizations
    print("\nCreating visualizations...")
    dist_fig = plot_parameter_distributions(param_data, stats, parameters_to_fit)
    corr_fig, corr_matrices = create_correlation_matrix(best_fits, parameters_to_fit)

    return {
        "all_fits": all_fits,
        "best_fits": best_fits,
        "param_data": param_data,
        "stats": stats,
        "distances": distances,
        "global_mean": global_mean,
        "global_cov": global_cov,
        "correlation_matrices": corr_matrices,
        "figures": {"distributions": dist_fig, "correlations": corr_fig},
    }


def extract_best_circuit_parameters(best_fits, parameters_to_fit):
    """Extract the single best parameter set from each circuit"""
    best_params = {}

    for circuit, df in best_fits.items():
        # Get the best fit (highest likelihood/posterior)
        if "posterior" in df.columns:
            best_idx = df["posterior"].idxmax()
        else:
            best_idx = df["likelihood"].idxmax()

        # Extract parameters in log space
        circuit_params = []
        for param in parameters_to_fit:
            if param in df.columns:
                value = df.loc[best_idx, param]
                circuit_params.append(value)

        best_params[circuit] = np.array(circuit_params)

    return best_params


def construct_hierarchical_parameters(
    best_circuit_params, hierarchical_fitter, alpha=None, sigma=None
):
    """
    Construct full hierarchical parameter vector from circuit parameters

    Parameters:
    - best_circuit_params: dict of circuit_name -> parameter array
    - hierarchical_fitter: HierarchicalCircuitFitter instance
    - alpha: global mean parameters (if None, computed from circuit params)
    - sigma: covariance matrix (if None, computed from circuit params)
    """
    # n_circuits = hierarchical_fitter.n_circuits
    n_params = hierarchical_fitter.n_parameters
    n_total = hierarchical_fitter.n_total_params

    # Initialize full parameter vector
    full_params = np.zeros(n_total)

    # 1. Set circuit-specific parameters (θ)
    circuit_names = [config.name for config in hierarchical_fitter.configs]
    for c, circuit_name in enumerate(circuit_names):
        if circuit_name in best_circuit_params:
            start_idx = c * n_params
            end_idx = (c + 1) * n_params
            full_params[start_idx:end_idx] = best_circuit_params[circuit_name]

    # 2. Compute/set global mean (α)
    if alpha is None:
        # Compute mean from circuit parameters
        all_circuit_params = np.array(list(best_circuit_params.values()))
        alpha = np.mean(all_circuit_params, axis=0)

    full_params[
        hierarchical_fitter.alpha_start_idx : hierarchical_fitter.alpha_start_idx
        + n_params
    ] = alpha

    # 3. Compute/set covariance (Σ)
    if sigma is None:
        # Compute covariance from circuit parameters
        all_circuit_params = np.array(list(best_circuit_params.values()))
        sigma = np.cov(all_circuit_params.T)
        # Add small diagonal to ensure positive definiteness
        sigma += 1e-4 * np.eye(n_params)

    # Flatten and store covariance
    flat_sigma = hierarchical_fitter._flatten_covariance(sigma)
    full_params[hierarchical_fitter.sigma_start_idx :] = flat_sigma

    return full_params, alpha, sigma


def diagnose_hierarchical_model(hierarchical_fitter, test_params):
    """
    Detailed diagnosis of hierarchical model with given parameters

    Returns dict with all likelihood/prior components
    """
    # Split parameters
    theta_params, alpha_params, sigma_matrices = (
        hierarchical_fitter.split_hierarchical_parameters(test_params.reshape(1, -1))
    )

    results = {
        "theta": theta_params[0],
        "alpha": alpha_params[0],
        "sigma": sigma_matrices[0],
        "components": {},
    }

    # 1. Calculate prior on α
    alpha_diff = alpha_params[0] - hierarchical_fitter.mu_alpha
    sigma_alpha_inv = np.linalg.inv(hierarchical_fitter.sigma_alpha)
    log_prior_alpha = -0.5 * np.dot(alpha_diff, np.dot(sigma_alpha_inv, alpha_diff))
    log_prior_alpha -= 0.5 * hierarchical_fitter.n_parameters * np.log(2 * np.pi)
    log_prior_alpha -= 0.5 * np.log(np.linalg.det(hierarchical_fitter.sigma_alpha))

    results["components"]["prior_alpha"] = {
        "value": log_prior_alpha,
        "alpha_diff_norm": np.linalg.norm(alpha_diff),
        "details": {
            "alpha": alpha_params[0],
            "mu_alpha": hierarchical_fitter.mu_alpha,
            "diff": alpha_diff,
        },
    }

    # 2. Calculate prior on Σ
    try:
        sigma_inv = np.linalg.inv(sigma_matrices[0])
        log_det_sigma = np.log(np.linalg.det(sigma_matrices[0]))

        log_prior_sigma = (
            -0.5
            * (hierarchical_fitter.nu + hierarchical_fitter.n_parameters + 1)
            * log_det_sigma
        )
        log_prior_sigma -= 0.5 * np.trace(np.dot(hierarchical_fitter.psi, sigma_inv))

        # Check eigenvalues
        eigvals = np.linalg.eigvalsh(sigma_matrices[0])

        results["components"]["prior_sigma"] = {
            "value": log_prior_sigma,
            "log_det": log_det_sigma,
            "min_eigenvalue": np.min(eigvals),
            "condition_number": np.max(eigvals) / np.min(eigvals),
            "details": {
                "eigenvalues": eigvals,
                "trace_term": np.trace(np.dot(hierarchical_fitter.psi, sigma_inv)),
            },
        }
    except np.linalg.LinAlgError as e:
        results["components"]["prior_sigma"] = {"value": -np.inf, "error": str(e)}

    # 3. Calculate prior on θ given α and Σ
    log_prior_theta_total = 0
    theta_components = []

    try:
        sigma_inv = np.linalg.inv(sigma_matrices[0])
        log_det_sigma = np.log(np.linalg.det(sigma_matrices[0]))

        for c in range(hierarchical_fitter.n_circuits):
            theta_diff = theta_params[0, c] - alpha_params[0]
            circuit_log_prior = -0.5 * np.dot(theta_diff, np.dot(sigma_inv, theta_diff))
            circuit_log_prior -= (
                0.5 * hierarchical_fitter.n_parameters * np.log(2 * np.pi)
            )
            circuit_log_prior -= 0.5 * log_det_sigma

            theta_components.append(
                {
                    "circuit_idx": c,
                    "circuit_name": hierarchical_fitter.configs[c].name,
                    "log_prior": circuit_log_prior,
                    "theta_diff_norm": np.linalg.norm(theta_diff),
                    "mahalanobis_distance": np.sqrt(
                        np.dot(theta_diff, np.dot(sigma_inv, theta_diff))
                    ),
                }
            )

            log_prior_theta_total += circuit_log_prior
    except np.linalg.LinAlgError as e:
        log_prior_theta_total = -np.inf
        theta_components = [{"error": str(e)}]

    results["components"]["prior_theta"] = {
        "value": log_prior_theta_total,
        "circuit_components": theta_components,
    }

    # 4. Calculate likelihood
    print("Calculating likelihood (this may take a moment)...")
    likelihood_results = hierarchical_fitter.calculate_hierarchical_likelihood(
        test_params.reshape(1, -1)
    )

    results["components"]["likelihood"] = {
        "value": likelihood_results["total"][0],
        "circuit_details": likelihood_results["circuits"][0],
    }

    # 5. Total prior and posterior
    total_log_prior = (
        results["components"]["prior_alpha"]["value"]
        + results["components"]["prior_sigma"]["value"]
        + results["components"]["prior_theta"]["value"]
    )

    total_log_posterior = total_log_prior + results["components"]["likelihood"]["value"]

    results["total_log_prior"] = total_log_prior
    results["total_log_likelihood"] = results["components"]["likelihood"]["value"]
    results["total_log_posterior"] = total_log_posterior

    return results


def compare_with_random_parameters(
    hierarchical_fitter, individual_results, n_random=10, noise_scale=0.1
):
    """
    Compare individual fits with random parameters from prior
    """
    comparison_results = []

    # 1. Test with individual fit parameters
    print("Testing with individual fit parameters...")
    best_circuit_params = extract_best_circuit_parameters(
        individual_results["best_fits"], hierarchical_fitter.parameters_to_fit
    )

    individual_params, alpha, sigma = construct_hierarchical_parameters(
        best_circuit_params, hierarchical_fitter
    )

    individual_diagnosis = diagnose_hierarchical_model(
        hierarchical_fitter, individual_params
    )
    individual_diagnosis["type"] = "individual_fits"
    comparison_results.append(individual_diagnosis)

    # 2. Test with prior mean
    print("Testing with prior mean parameters...")
    prior_circuit_params = {}
    for config in hierarchical_fitter.configs:
        prior_circuit_params[config.name] = hierarchical_fitter.alpha

    prior_params, _, _ = construct_hierarchical_parameters(
        prior_circuit_params,
        hierarchical_fitter,
        alpha=hierarchical_fitter.alpha,
        sigma=hierarchical_fitter.sigma,
    )

    prior_diagnosis = diagnose_hierarchical_model(hierarchical_fitter, prior_params)
    prior_diagnosis["type"] = "prior_mean"
    comparison_results.append(prior_diagnosis)

    # 3. Test with random samples from prior
    print(f"Testing with {n_random} random samples from prior...")
    for i in range(n_random):
        # Sample from prior
        random_alpha = np.random.multivariate_normal(
            hierarchical_fitter.mu_alpha, hierarchical_fitter.sigma_alpha
        )

        # Sample Sigma from inverse Wishart
        random_sigma = invwishart.rvs(
            df=hierarchical_fitter.nu, scale=hierarchical_fitter.psi
        )

        # Sample circuit parameters
        random_circuit_params = {}
        for config in hierarchical_fitter.configs:
            random_circuit_params[config.name] = np.random.multivariate_normal(
                random_alpha, random_sigma
            )

        random_params, _, _ = construct_hierarchical_parameters(
            random_circuit_params,
            hierarchical_fitter,
            alpha=random_alpha,
            sigma=random_sigma,
        )

        random_diagnosis = diagnose_hierarchical_model(
            hierarchical_fitter, random_params
        )
        random_diagnosis["type"] = f"random_{i}"
        comparison_results.append(random_diagnosis)

    return comparison_results


def visualize_diagnosis_results(comparison_results):
    """Create visualizations of diagnostic results"""
    # Extract data for plotting
    types = [r["type"] for r in comparison_results]
    log_priors = [r["total_log_prior"] for r in comparison_results]
    log_likelihoods = [r["total_log_likelihood"] for r in comparison_results]
    log_posteriors = [r["total_log_posterior"] for r in comparison_results]

    # Component breakdown
    prior_alpha = [r["components"]["prior_alpha"]["value"] for r in comparison_results]
    prior_sigma = [r["components"]["prior_sigma"]["value"] for r in comparison_results]
    prior_theta = [r["components"]["prior_theta"]["value"] for r in comparison_results]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Total values comparison
    ax = axes[0, 0]
    x = np.arange(len(types))
    width = 0.25
    ax.bar(x - width, log_priors, width, label="Prior", alpha=0.7)
    ax.bar(x, log_likelihoods, width, label="Likelihood", alpha=0.7)
    ax.bar(x + width, log_posteriors, width, label="Posterior", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45)
    ax.set_ylabel("Log probability")
    ax.set_title("Total Log Probabilities")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Prior components breakdown
    ax = axes[0, 1]
    ax.bar(x - width, prior_alpha, width, label="α prior", alpha=0.7)
    ax.bar(x, prior_sigma, width, label="Σ prior", alpha=0.7)
    ax.bar(x + width, prior_theta, width, label="θ prior", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45)
    ax.set_ylabel("Log probability")
    ax.set_title("Prior Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Circuit-specific θ priors
    ax = axes[1, 0]
    for r in comparison_results:
        if "circuit_components" in r["components"]["prior_theta"]:
            circuit_priors = [
                c["log_prior"]
                for c in r["components"]["prior_theta"]["circuit_components"]
                if "log_prior" in c
            ]
            if circuit_priors:
                ax.plot(circuit_priors, "o-", label=r["type"], alpha=0.7)
    ax.set_xlabel("Circuit index")
    ax.set_ylabel("Log prior θ_c")
    ax.set_title("Circuit-specific Parameter Priors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Mahalanobis distances
    ax = axes[1, 1]
    for r in comparison_results:
        if "circuit_components" in r["components"]["prior_theta"]:
            distances = [
                c["mahalanobis_distance"]
                for c in r["components"]["prior_theta"]["circuit_components"]
                if "mahalanobis_distance" in c
            ]
            if distances:
                ax.plot(distances, "o-", label=r["type"], alpha=0.7)
    ax.set_xlabel("Circuit index")
    ax.set_ylabel("Mahalanobis distance")
    ax.set_title("Circuit Parameters Distance from α")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_diagnosis_summary(diagnosis_result):
    """Print a detailed summary of diagnosis results"""
    print("\n" + "=" * 60)
    print(f"Diagnosis for: {diagnosis_result['type']}")
    print("=" * 60)

    print(f"\nTotal log posterior: {diagnosis_result['total_log_posterior']:.2f}")
    print(f"  - Log prior: {diagnosis_result['total_log_prior']:.2f}")
    print(f"  - Log likelihood: {diagnosis_result['total_log_likelihood']:.2f}")

    print("\nPrior components:")
    print(f"  - α prior: {diagnosis_result['components']['prior_alpha']['value']:.2f}")
    print(
        f"    - ||α - μ_α||: {diagnosis_result['components']['prior_alpha']['alpha_diff_norm']:.3f}"
    )

    print(f"  - Σ prior: {diagnosis_result['components']['prior_sigma']['value']:.2f}")
    if "min_eigenvalue" in diagnosis_result["components"]["prior_sigma"]:
        print(
            f"    - Min eigenvalue: {diagnosis_result['components']['prior_sigma']['min_eigenvalue']:.3e}"
        )
        print(
            f"    - Condition number: {diagnosis_result['components']['prior_sigma']['condition_number']:.2e}"
        )

    print(f"  - θ prior: {diagnosis_result['components']['prior_theta']['value']:.2f}")
    if "circuit_components" in diagnosis_result["components"]["prior_theta"]:
        for comp in diagnosis_result["components"]["prior_theta"]["circuit_components"]:
            if "circuit_name" in comp:
                print(
                    f"    - {comp['circuit_name']}: {comp['log_prior']:.2f} "
                    f"(Mahalanobis dist: {comp['mahalanobis_distance']:.2f})"
                )

    print("\nLikelihood by circuit:")
    if "circuit_details" in diagnosis_result["components"]["likelihood"]:
        for circuit, details in diagnosis_result["components"]["likelihood"][
            "circuit_details"
        ].items():
            print(f"  - {circuit}: {details['total'][0]:.2f}")


def run_hierarchical_diagnostics(hierarchical_fitter, individual_results):
    """Run complete diagnostic analysis"""
    print("Running hierarchical model diagnostics...")

    # Run comparison
    comparison_results = compare_with_random_parameters(
        hierarchical_fitter, individual_results, n_random=5
    )

    # Print summaries
    for result in comparison_results:
        print_diagnosis_summary(result)

    # Create visualizations
    fig = visualize_diagnosis_results(comparison_results)

    return comparison_results, fig


def debug_hierarchical_model():
    """Main debugging workflow"""

    # 1. Setup
    print("=== STEP 1: SETUP ===")

    # Initialize CircuitManager
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    # Load priors
    priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]
    parameters_to_fit = priors.Parameter.tolist()
    print(f"Parameters to fit: {parameters_to_fit}")

    # Setup calibration
    calibration_params = setup_calibration()

    # 2. Analyze individual fits
    print("\n=== STEP 2: ANALYZE INDIVIDUAL FITS ===")
    individual_results = analyze_individual_fits_main(parameters_to_fit, n_best=100)

    if not individual_results:
        print(
            "ERROR: No individual fits found. Please run individual circuit fitting first."
        )
        return

    # Save distribution plots
    individual_results["figures"]["distributions"].savefig(
        "debug_individual_distributions.png", dpi=300, bbox_inches="tight"
    )
    individual_results["figures"]["correlations"].savefig(
        "debug_individual_correlations.png", dpi=300, bbox_inches="tight"
    )

    # 3. Setup hierarchical model with same circuits
    print("\n=== STEP 3: SETUP HIERARCHICAL MODEL ===")

    # Get circuits that have individual fits
    fitted_circuits = list(individual_results["best_fits"].keys())
    print(f"Circuits with individual fits: {fitted_circuits}")

    # Create circuit configurations
    circuit_configs = create_circuit_configs(
        circuit_manager, fitted_circuits, min_time=30, max_time=210
    )

    # Create hierarchical fitter
    hierarchical_fitter = HierarchicalCircuitFitter(
        circuit_configs, parameters_to_fit, priors, calibration_params
    )

    # 4. Run diagnostics
    print("\n=== STEP 4: RUN HIERARCHICAL DIAGNOSTICS ===")
    comparison_results, diagnostic_fig = run_hierarchical_diagnostics(
        hierarchical_fitter, individual_results
    )

    # Save diagnostic plots
    diagnostic_fig.savefig(
        "debug_hierarchical_diagnostics.png", dpi=300, bbox_inches="tight"
    )

    # 5. Additional analysis: Parameter space exploration
    print("\n=== STEP 5: PARAMETER SPACE EXPLORATION ===")

    # Extract key findings
    individual_fit_result = comparison_results[0]  # Results using individual fits
    prior_mean_result = comparison_results[1]  # Results using prior means

    print("\nKey Findings:")
    print(
        f"Individual fits log posterior: {individual_fit_result['total_log_posterior']:.2f}"
    )
    print(f"Prior mean log posterior: {prior_mean_result['total_log_posterior']:.2f}")
    print(
        f"Difference: {individual_fit_result['total_log_posterior'] - prior_mean_result['total_log_posterior']:.2f}"
    )

    # Identify problematic components
    print("\nProblematic components (individual fits):")
    if individual_fit_result["components"]["prior_sigma"]["value"] < -1000:
        print("  - Covariance prior (Σ) is very negative!")
        print(
            f"    Check condition number: {individual_fit_result['components']['prior_sigma'].get('condition_number', 'N/A')}"
        )

    if individual_fit_result["components"]["prior_theta"]["value"] < -1000:
        print("  - Circuit parameter prior (θ) is very negative!")
        print("    This suggests circuit parameters are too far from global mean")

    # 6. Test with modified hyperparameters
    print("\n=== STEP 6: TEST MODIFIED HYPERPARAMETERS ===")

    # Modify hyperpriors to be less restrictive
    print("Testing with relaxed hyperpriors...")

    # Create a new hierarchical fitter with relaxed priors
    hierarchical_fitter_relaxed = HierarchicalCircuitFitter(
        circuit_configs, parameters_to_fit, priors, calibration_params
    )

    # Relax the prior on α (make Σ_α larger)
    hierarchical_fitter_relaxed.sigma_alpha = 2.0 * np.eye(
        hierarchical_fitter.n_parameters
    )

    # Relax the prior on Σ (use actual covariance from individual fits)
    best_circuit_params = extract_best_circuit_parameters(
        individual_results["best_fits"], parameters_to_fit
    )
    all_params = np.array(list(best_circuit_params.values()))
    empirical_cov = np.cov(all_params.T)
    hierarchical_fitter_relaxed.psi = empirical_cov * hierarchical_fitter.n_parameters

    # Construct parameters using individual fits
    test_params, _, _ = construct_hierarchical_parameters(
        best_circuit_params, hierarchical_fitter_relaxed
    )

    relaxed_diagnosis = diagnose_hierarchical_model(
        hierarchical_fitter_relaxed, test_params
    )
    relaxed_diagnosis["type"] = "individual_fits_relaxed"

    print("\nRelaxed model results:")
    print(f"Log posterior: {relaxed_diagnosis['total_log_posterior']:.2f}")
    print(f"  - Prior: {relaxed_diagnosis['total_log_prior']:.2f}")
    print(f"  - Likelihood: {relaxed_diagnosis['total_log_likelihood']:.2f}")

    # 7. Summary and recommendations
    print("\n=== SUMMARY AND RECOMMENDATIONS ===")

    # Calculate parameter spread
    param_spreads = {}
    for param in parameters_to_fit:
        if param in individual_results["stats"]:
            spread = individual_results["stats"][param]["between_circuit_std"]
            param_spreads[param] = spread

    print("\nParameter spread across circuits (log10 scale):")
    for param, spread in sorted(
        param_spreads.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {param}: {spread:.3f}")

    print("\nRecommendations:")
    if any(spread > 0.5 for spread in param_spreads.values()):
        print("- Large parameter variation across circuits detected!")
        print("- Consider using more flexible hyperpriors (larger Σ_α)")
        print("- May need to allow for more circuit-specific variation")

    if individual_fit_result["components"]["prior_sigma"]["value"] < -100:
        print("- Covariance matrix prior is too restrictive")
        print("- Consider using empirical covariance from individual fits as Ψ")

    if (
        individual_fit_result["total_log_likelihood"]
        < prior_mean_result["total_log_likelihood"]
    ):
        print("- Individual fits have WORSE likelihood than prior mean!")
        print("- Check for overfitting in individual fits")
        print("- May need more data or regularization")

    # Save all results
    print("\nSaving results...")
    import pickle

    results = {
        "individual_results": individual_results,
        "comparison_results": comparison_results,
        "relaxed_diagnosis": relaxed_diagnosis,
        "hierarchical_fitter": hierarchical_fitter,
        "parameter_spreads": param_spreads,
    }

    with open("hierarchical_debug_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nDebugging complete! Check the generated plots and summaries.")
    return results


def plot_parameter_comparison(results):
    """Create additional diagnostic plots"""
    individual_results = results["individual_results"]
    _ = results["comparison_results"]

    # Plot showing how individual circuit parameters compare to hierarchical estimates
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # 1. Parameter values by circuit
    best_params = extract_best_circuit_parameters(
        individual_results["best_fits"], list(individual_results["stats"].keys())
    )

    circuit_names = list(best_params.keys())
    n_params = len(list(individual_results["stats"].keys()))

    # Create heatmap of parameters
    param_matrix = np.array(list(best_params.values()))
    im = ax.imshow(param_matrix.T, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(n_params))
    ax.set_yticklabels(list(individual_results["stats"].keys()))
    ax.set_xticks(range(len(circuit_names)))
    ax.set_xticklabels(circuit_names, rotation=45)
    ax.set_title("Best-fit Parameters by Circuit (log10)")
    plt.colorbar(im, ax=ax)

    return fig


if __name__ == "__main__":
    # Run the debugging workflow
    results = debug_hierarchical_model()

    # Create additional diagnostic plots
    if results:
        additional_fig = plot_parameter_comparison(results)
        additional_fig.savefig(
            "debug_parameter_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close("all")
