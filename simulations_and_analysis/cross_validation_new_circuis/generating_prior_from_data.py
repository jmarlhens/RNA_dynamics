import pandas as pd
import os
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

from simulations_and_analysis.individual.individual_circuits_statistics import (
    generate_prior_mean_coordinates,
    convert_individual_to_theta_format,
)
from analysis_and_figures.hierarchical_pairplot_analysis import (
    create_circuit_prior_comparison_pairplot,
)
from circuits.circuit_generation.circuit_manager import CircuitManager
from simulations_and_analysis.individual.individual_circuits_simulations import (
    create_circuit_simulation_data,
    simulate_and_organize_parameter_sets,
)
from analysis_and_figures.plots_simulation import (
    extract_trajectory_data,
    plot_single_circuit_two_column,
)
from utils.GFP_calibration import setup_calibration


def load_circuit_posterior_data(results_filepath, prior_filepath):
    """Load and process circuit fitting results and prior parameters."""
    # prior_parameters = pd.read_csv(prior_filepath)
    circuit_results = pd.read_csv(results_filepath)

    excluded_columns = {
        "iteration",
        "walker",
        "chain",
        "likelihood",
        "prior",
        "posterior",
        "step_accepted",
    }
    parameter_names = circuit_results.columns.difference(excluded_columns).tolist()

    samples_posterior_processed = convert_individual_to_theta_format(
        {"trigger_antitrigger_and_star_antistar": circuit_results},
        parameter_names,
        ["trigger_antitrigger_and_star_antistar"],
        burn_in_fraction=0.5,
        post_burnin_samples_per_circuit=2000000,
    )

    prior_mean_coordinates = generate_prior_mean_coordinates(
        prior_filepath, parameter_names
    )

    return samples_posterior_processed, prior_mean_coordinates, parameter_names


def estimate_posterior_statistics_ledoitwolf(samples_df, parameter_names):
    """Compute posterior mean and regularized covariance using Ledoit-Wolf estimator."""
    posterior_mean_coordinates = samples_df[parameter_names].mean().to_dict()

    ledoit_wolf_estimator = LedoitWolf()
    ledoit_wolf_estimator.fit(samples_df[parameter_names])

    posterior_covariance = pd.DataFrame(
        ledoit_wolf_estimator.covariance_,
        index=parameter_names,
        columns=parameter_names,
    ).to_dict()

    return posterior_mean_coordinates, posterior_covariance


def create_posterior_visualization_with_contours(
    prior_posterior_combined,
    visualization_parameters,
    posterior_statistics,
    output_directory,
):
    """Generate pairplot with posterior distribution contours overlaid."""
    posterior_mean, posterior_covariance = posterior_statistics
    samples_data = prior_posterior_combined[
        prior_posterior_combined["type"] == "Circuit"
    ]

    plot_columns = visualization_parameters + ["type", "Circuit"]
    plot_columns += [
        f"{param}_log10stdev"
        for param in visualization_parameters
        if f"{param}_log10stdev" in prior_posterior_combined.columns
    ]

    pairplot_figure = create_circuit_prior_comparison_pairplot(
        prior_posterior_combined[plot_columns],
        visualization_parameters,
        output_directory,
        diagonal_visualization_type="hist",
    )

    # Add multivariate normal contours to off-diagonal plots
    for row_idx, row_param in enumerate(visualization_parameters):
        for col_idx, col_param in enumerate(visualization_parameters):
            if row_idx != col_idx:
                axis = pairplot_figure.axes[row_idx, col_idx]

                x_range = np.linspace(
                    samples_data[col_param].min(), samples_data[col_param].max(), 100
                )
                y_range = np.linspace(
                    samples_data[row_param].min(), samples_data[row_param].max(), 100
                )

                X_grid, Y_grid = np.meshgrid(x_range, y_range)
                grid_positions = np.dstack((X_grid, Y_grid))

                bivariate_mean = [
                    posterior_mean[col_param],
                    posterior_mean[row_param],
                ]
                bivariate_covariance = [
                    [
                        posterior_covariance[col_param][col_param],
                        posterior_covariance[col_param][row_param],
                    ],
                    [
                        posterior_covariance[row_param][col_param],
                        posterior_covariance[row_param][row_param],
                    ],
                ]

                multivariate_distribution = multivariate_normal(
                    bivariate_mean, bivariate_covariance
                )
                density_values = multivariate_distribution.pdf(grid_positions)

                axis.contour(
                    X_grid,
                    Y_grid,
                    density_values,
                    colors="yellow",
                    alpha=0.4,
                    levels=20,
                    linewidths=1,
                )

    return pairplot_figure


def generate_posterior_samples_multivariate(
    posterior_mean,
    posterior_covariance,
    parameter_names,
    final_sample_count=50,
    oversample_factor=10,
):
    """Generate samples from fitted multivariate normal posterior distribution."""
    total_samples = final_sample_count * oversample_factor

    multivariate_distribution = multivariate_normal(
        mean=[posterior_mean[param] for param in parameter_names],
        cov=[
            [posterior_covariance[i][j] for j in parameter_names]
            for i in parameter_names
        ],
    )

    raw_samples = multivariate_distribution.rvs(size=total_samples)
    samples_dataframe = pd.DataFrame(raw_samples, columns=parameter_names)

    return samples_dataframe.sample(n=final_sample_count, random_state=42)


def generate_truncated_multivariate_posterior_samples(
    posterior_mean,
    posterior_covariance,
    parameter_names,
    confidence_level=0.95,
    final_sample_count=50,
    max_rejection_ratio=20,
):
    """Generate samples from multivariate normal truncated at confidence ellipsoid boundary."""
    from scipy.stats import chi2

    # Convert to numpy arrays for efficient computation
    mean_vector = np.array([posterior_mean[param] for param in parameter_names])
    covariance_matrix = np.array(
        [
            [posterior_covariance[param_i][param_j] for param_j in parameter_names]
            for param_i in parameter_names
        ]
    )

    # Mahalanobis distance threshold for confidence ellipsoid
    num_parameters = len(parameter_names)
    mahalanobis_threshold = chi2.ppf(confidence_level, df=num_parameters)

    # Precompute inverse covariance for efficiency
    inverse_covariance = np.linalg.inv(covariance_matrix)

    multivariate_distribution = multivariate_normal(mean_vector, covariance_matrix)

    max_attempts = final_sample_count * max_rejection_ratio
    accepted_samples = []

    for _ in range(max_attempts):
        candidate_sample = multivariate_distribution.rvs()

        # Compute Mahalanobis distance squared
        deviation = candidate_sample - mean_vector
        mahalanobis_squared = deviation.T @ inverse_covariance @ deviation

        if mahalanobis_squared <= mahalanobis_threshold:
            accepted_samples.append(candidate_sample)

            if len(accepted_samples) == final_sample_count:
                break

    if len(accepted_samples) < final_sample_count:
        raise ValueError(
            f"Generated only {len(accepted_samples)}/{final_sample_count} samples "
            f"within {confidence_level * 100}% confidence ellipsoid after {max_attempts} attempts"
        )

    return pd.DataFrame(accepted_samples, columns=parameter_names)


def simulate_circuits_with_posterior_samples(
    samples_dataframe,
    circuit_names_list,
    circuit_manager,
    calibration_parameters,
    time_boundaries,
    output_directory,
):
    """Run circuit simulations with posterior samples and generate trajectory plots."""
    time_max, time_min = time_boundaries
    fitted_parameters = samples_dataframe.columns.tolist()

    for circuit_name in circuit_names_list:
        circuit_configuration, circuit_fitter = create_circuit_simulation_data(
            circuit_name,
            fitted_parameters,
            circuit_manager,
            calibration_parameters,
            time_max,
            time_min,
        )

        simulation_data, results_dataframe = simulate_and_organize_parameter_sets(
            samples_dataframe, circuit_fitter, fitted_parameters
        )

        single_circuit_data = {
            circuit_name: {
                "config": circuit_configuration,
                "combined_params": simulation_data["combined_params"],
                "simulation_results": simulation_data["simulation_results"],
            }
        }

        trajectory_data = extract_trajectory_data(
            single_circuit_data, results_dataframe
        )
        circuit_trajectory_subset = trajectory_data[
            trajectory_data["circuit"] == circuit_name
        ]

        # Generate individual and summary trajectory plots
        for simulation_mode in ["individual", "summary"]:
            plot_single_circuit_two_column(
                circuit_name,
                single_circuit_data[circuit_name],
                circuit_trajectory_subset,
                results_dataframe,
                simulation_mode=simulation_mode,
                summary_type="median_iqr",
                percentile_bounds=(10, 90),
            )

            mode_suffix = "_summary" if simulation_mode == "summary" else "_individual"
            output_filename = f"individual_{circuit_name}_two_column{mode_suffix}.png"

            plt.show()

            plt.savefig(
                os.path.join(output_directory, output_filename),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()


def simulations_from_posterior(
    results_filepath,
    prior_filepath,
    output_directory,
    circuit_names_list,
    visualization_parameters,
):
    """Main workflow for posterior analysis, visualization, and circuit simulation."""
    os.makedirs(output_directory, exist_ok=True)

    # Load and process data
    samples_posterior, prior_coordinates, parameter_names = load_circuit_posterior_data(
        results_filepath, prior_filepath
    )

    # Estimate posterior statistics
    posterior_mean, posterior_covariance = estimate_posterior_statistics_ledoitwolf(
        samples_posterior, parameter_names
    )

    # Create combined dataset for visualization
    prior_posterior_combined = pd.concat(
        [samples_posterior, prior_coordinates], ignore_index=True
    )

    # Generate visualization with contours
    pairplot_figure = create_posterior_visualization_with_contours(
        prior_posterior_combined,
        visualization_parameters,
        (posterior_mean, posterior_covariance),
        output_directory,
    )

    pairplot_figure.savefig(
        f"{output_directory}/posterior_sampled_and_approximated_pairplot.png", dpi=300
    )
    plt.show()

    # Generate posterior samples and run circuit simulations
    posterior_samples = generate_posterior_samples_multivariate(
        posterior_mean, posterior_covariance, parameter_names
    )

    # instead we try with truncated samples
    posterior_samples = generate_truncated_multivariate_posterior_samples(
        posterior_mean,
        posterior_covariance,
        parameter_names,
        final_sample_count=100,
        confidence_level=0.1,
    )

    calibration_parameters = setup_calibration()
    circuit_manager = CircuitManager(
        parameters_file=prior_filepath,
        json_file="../../data/circuits/circuits.json",
    )

    # samples from the mcmc posterior
    simulate_circuits_with_posterior_samples(
        samples_posterior[parameter_names].sample(n=100, random_state=42),
        circuit_names_list,
        circuit_manager,
        calibration_parameters,
        (130, 30),
        output_directory,
    )

    # samples from the ledoit wolf posterior
    simulate_circuits_with_posterior_samples(
        posterior_samples,
        circuit_names_list,
        circuit_manager,
        calibration_parameters,
        (130, 30),
        output_directory,
    )


if __name__ == "__main__":
    # Configuration parameters
    subfolder = "/shared_parameters/star_antistar_trigger_antitrigger"
    individual_results_directory = "../../data/fit_data" + subfolder
    results_filename = (
        "results_star_antistar_1_and_trigger_antitrigger_20250902_215615.csv"
    )

    results_filepath = f"{individual_results_directory}/{results_filename}"
    prior_parameters_filepath = (
        "../../data/prior/model_parameters_priors_updated_tighter.csv"
    )
    output_visualization_directory = "../../figures/calibrated_prior" + subfolder

    circuit_names_to_simulate = ["trigger_antitrigger", "star_antistar_1"]
    parameters_for_visualization = ["K_tx", "k_tx", "K_tl", "k_tl"]

    simulations_from_posterior(
        results_filepath,
        prior_parameters_filepath,
        output_visualization_directory,
        circuit_names_to_simulate,
        parameters_for_visualization,
    )
