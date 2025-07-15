import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from circuits.circuit_generation.circuit_manager import CircuitManager
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor
from analysis_and_figures.mcmc_analysis_hierarchical import process_mcmc_data
from analysis_and_figures.plots_simulation import (
    plot_circuit_simulations,
    plot_circuit_conditions_overlay,
)
from simulations_and_analysis.individual.individual_circuits_simulations import (
    create_circuit_simulation_data,
    simulate_and_organize_parameter_sets,
)


def setup_gfp_calibration_parameters():
    """Configure GFP calibration parameters for circuit simulations"""
    gfp_calibration_data = pd.read_csv(
        "../../utils/calibration_gfp/gfp_Calibration.csv"
    )
    gfp_calibration_results = fit_gfp_calibration(
        gfp_calibration_data,
        concentration_col="GFP Concentration (nM)",
        fluorescence_pattern="F.I. (a.u)",
    )
    brightness_correction_factor, _ = get_brightness_correction_factor("avGFP", "sfGFP")

    return {
        "slope": gfp_calibration_results["slope"],
        "intercept": gfp_calibration_results["intercept"],
        "brightness_correction": brightness_correction_factor,
    }


def sample_circuit_parameters_from_hierarchical_posterior(
    hierarchical_mcmc_csv_filepath: str,
    fitted_parameter_names: list,
    n_circuit_parameter_samples: int,
    burn_in_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Sample circuit parameters θ ~ N(α, Σ) from hierarchical posterior for test circuit evaluation.

    Args:
            hierarchical_mcmc_csv_filepath: Path to hierarchical MCMC results CSV
            fitted_parameter_names: Parameter names for matrix reconstruction
            n_circuit_parameter_samples: Number of circuit parameter samples to generate
            burn_in_fraction: Burn-in fraction for posterior filtering

    Returns:
            DataFrame with sampled circuit parameters formatted for simulation pipeline
    """
    # Load and filter hierarchical MCMC results
    hierarchical_mcmc_raw_results = pd.read_csv(hierarchical_mcmc_csv_filepath)

    # Handle potential column naming inconsistency
    if (
        "walker" in hierarchical_mcmc_raw_results.columns
        and "iteration" in hierarchical_mcmc_raw_results.columns
    ):
        hierarchical_mcmc_raw_results = hierarchical_mcmc_raw_results.rename(
            columns={"walker": "iteration", "iteration": "walker"}
        )

    # Apply burn-in filtering
    mcmc_processing_result = process_mcmc_data(
        hierarchical_mcmc_raw_results, burn_in=burn_in_fraction, chain_idx=0
    )
    hierarchical_posterior_samples = mcmc_processing_result["processed_data"]

    print(
        f"Hierarchical posterior samples after burn-in: {len(hierarchical_posterior_samples)}"
    )

    # Extract alpha parameter vectors
    alpha_column_identifiers = [
        f"alpha_{parameter_name}" for parameter_name in fitted_parameter_names
    ]
    missing_alpha_columns = [
        col
        for col in alpha_column_identifiers
        if col not in hierarchical_posterior_samples.columns
    ]
    if missing_alpha_columns:
        raise ValueError(
            f"Missing alpha columns in hierarchical results: {missing_alpha_columns}"
        )

    alpha_posterior_matrix = hierarchical_posterior_samples[
        alpha_column_identifiers
    ].values
    n_posterior_samples, n_fitted_parameters = alpha_posterior_matrix.shape

    # Reconstruct sigma covariance matrices from diagonal variances and correlations
    sigma_covariance_matrices = np.zeros(
        (n_posterior_samples, n_fitted_parameters, n_fitted_parameters)
    )

    # Verify sigma diagonal columns exist
    sigma_diagonal_column_identifiers = [
        f"sigma_{parameter_name}" for parameter_name in fitted_parameter_names
    ]
    missing_sigma_diagonal_columns = [
        col
        for col in sigma_diagonal_column_identifiers
        if col not in hierarchical_posterior_samples.columns
    ]
    if missing_sigma_diagonal_columns:
        raise ValueError(
            f"Missing sigma diagonal columns: {missing_sigma_diagonal_columns}"
        )

    # Verify correlation columns exist
    correlation_column_identifiers = []
    for param1_index in range(n_fitted_parameters):
        for param2_index in range(param1_index + 1, n_fitted_parameters):
            correlation_column_name = f"corr_{fitted_parameter_names[param1_index]}_{fitted_parameter_names[param2_index]}"
            correlation_column_identifiers.append(correlation_column_name)

    missing_correlation_columns = [
        col
        for col in correlation_column_identifiers
        if col not in hierarchical_posterior_samples.columns
    ]
    if missing_correlation_columns:
        raise ValueError(f"Missing correlation columns: {missing_correlation_columns}")

    # Reconstruct covariance matrices for each posterior sample
    for posterior_sample_index in range(n_posterior_samples):
        # Fill diagonal variance elements
        for parameter_index, parameter_name in enumerate(fitted_parameter_names):
            diagonal_variance = hierarchical_posterior_samples[
                f"sigma_{parameter_name}"
            ].iloc[posterior_sample_index]
            sigma_covariance_matrices[
                posterior_sample_index, parameter_index, parameter_index
            ] = diagonal_variance

        # Fill off-diagonal covariance elements using correlation reconstruction
        for param1_index in range(n_fitted_parameters):
            for param2_index in range(param1_index + 1, n_fitted_parameters):
                param1_name = fitted_parameter_names[param1_index]
                param2_name = fitted_parameter_names[param2_index]

                correlation_coefficient = hierarchical_posterior_samples[
                    f"corr_{param1_name}_{param2_name}"
                ].iloc[posterior_sample_index]
                variance1 = sigma_covariance_matrices[
                    posterior_sample_index, param1_index, param1_index
                ]
                variance2 = sigma_covariance_matrices[
                    posterior_sample_index, param2_index, param2_index
                ]

                covariance_element = correlation_coefficient * np.sqrt(
                    variance1 * variance2
                )
                sigma_covariance_matrices[
                    posterior_sample_index, param1_index, param2_index
                ] = covariance_element
                sigma_covariance_matrices[
                    posterior_sample_index, param2_index, param1_index
                ] = covariance_element

    # Calculate sampling allocation across posterior samples
    samples_per_posterior_draw = max(
        1, n_circuit_parameter_samples // n_posterior_samples
    )

    # Generate circuit parameter samples using multivariate normal sampling
    circuit_parameter_sample_collection = []
    for posterior_sample_index in range(n_posterior_samples):
        multivariate_normal_samples = np.random.multivariate_normal(
            mean=alpha_posterior_matrix[posterior_sample_index],
            cov=sigma_covariance_matrices[posterior_sample_index],
            size=samples_per_posterior_draw,
        )
        circuit_parameter_sample_collection.extend(multivariate_normal_samples)

    # Format output as DataFrame matching simulation pipeline expectations
    final_circuit_parameter_samples = np.array(
        circuit_parameter_sample_collection[:n_circuit_parameter_samples]
    )
    circuit_parameter_dataframe = pd.DataFrame(
        final_circuit_parameter_samples, columns=fitted_parameter_names
    )
    circuit_parameter_dataframe["likelihood"] = np.arange(
        len(circuit_parameter_dataframe)
    )

    print(f"Generated {len(circuit_parameter_dataframe)} circuit parameter samples")

    return circuit_parameter_dataframe


def generate_test_circuit_simulation_plots(
    test_circuit_name: str,
    sampled_circuit_parameters: pd.DataFrame,
    circuit_fitter_instance,
    fitted_parameter_names: list,
    output_directory: str,
    simulation_sample_count: int,
):
    """Generate simulation plots for individual test circuit using sampled parameters"""

    # Select subset of samples for simulation performance
    final_simulation_sample_count = min(
        simulation_sample_count, len(sampled_circuit_parameters)
    )
    selected_circuit_parameters = (
        sampled_circuit_parameters.sample(
            n=final_simulation_sample_count, random_state=42
        )
        if len(sampled_circuit_parameters) > final_simulation_sample_count
        else sampled_circuit_parameters.copy()
    )

    # Sort by likelihood for best samples visualization
    best_likelihood_samples = selected_circuit_parameters.sort_values(
        by="likelihood", ascending=False
    )
    random_likelihood_samples = selected_circuit_parameters.sample(
        n=final_simulation_sample_count, random_state=42
    )

    # Generate plots for both best and random sample sets
    for sample_selection_type, parameter_samples in [
        ("best_posterior_samples", best_likelihood_samples),
        ("random_posterior_samples", random_likelihood_samples),
    ]:
        simulation_data, simulation_results_dataframe = (
            simulate_and_organize_parameter_sets(
                parameter_samples, circuit_fitter_instance, fitted_parameter_names
            )
        )

        test_circuit_simulation_dict = {test_circuit_name: simulation_data}
        plot_circuit_simulations(
            test_circuit_simulation_dict,
            simulation_results_dataframe,
            plot_mode="individual",
            likelihood_percentile_range=20,
        )

        actual_sample_count = len(parameter_samples)
        plot_title_prefix = (
            "Top Likelihood"
            if sample_selection_type == "best_posterior_samples"
            else "Random"
        )
        plot_title = f"{plot_title_prefix} {actual_sample_count} Hierarchical Posterior Samples for {test_circuit_name}"

        plt.suptitle(plot_title)
        plt.savefig(
            os.path.join(
                output_directory,
                f"hierarchical_posterior_{sample_selection_type}_{test_circuit_name}.png",
            )
        )
        plt.close()

    return best_likelihood_samples, random_likelihood_samples


def execute_hierarchical_test_circuit_simulations(
    test_circuit_identifiers: list,
    hierarchical_mcmc_results_filepath: str,
    output_directory: str = ".",
    simulation_sample_count: int = 60,
    circuit_parameter_samples_per_test_circuit: int = 1000,
    simulation_time_bounds_max: int = None,
    simulation_time_bounds_min: int = None,
):
    """Execute complete hierarchical test circuit simulation workflow"""

    # Initialize circuit management and calibration
    circuit_manager_instance = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    model_parameter_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    fitted_parameter_names = model_parameter_priors[
        model_parameter_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()

    gfp_calibration_parameters = setup_gfp_calibration_parameters()

    # Storage for combined plotting across test circuits
    combined_best_simulation_data_dict = {}
    combined_random_simulation_data_dict = {}
    combined_best_results_dataframes = []
    combined_random_results_dataframes = []

    # Process each test circuit individually
    for test_circuit_name in test_circuit_identifiers:
        print(f"Processing hierarchical test circuit: {test_circuit_name}")

        # Sample circuit parameters from hierarchical posterior
        sampled_circuit_parameters = (
            sample_circuit_parameters_from_hierarchical_posterior(
                hierarchical_mcmc_results_filepath,
                fitted_parameter_names,
                circuit_parameter_samples_per_test_circuit,
                burn_in_fraction=0.5,
            )
        )

        # Create circuit configuration and fitter for test circuit
        test_circuit_configuration, test_circuit_fitter = (
            create_circuit_simulation_data(
                test_circuit_name,
                sampled_circuit_parameters,  # Dummy parameter for function signature compatibility
                fitted_parameter_names,
                circuit_manager_instance,
                gfp_calibration_parameters,
                simulation_time_bounds_max,
                simulation_time_bounds_min,
            )
        )

        # Generate individual circuit simulation plots
        best_parameter_samples, random_parameter_samples = (
            generate_test_circuit_simulation_plots(
                test_circuit_name,
                sampled_circuit_parameters,
                test_circuit_fitter,
                fitted_parameter_names,
                output_directory,
                simulation_sample_count,
            )
        )

        # Prepare simulation data for combined plotting
        best_simulation_data, best_simulation_results = (
            simulate_and_organize_parameter_sets(
                best_parameter_samples, test_circuit_fitter, fitted_parameter_names
            )
        )
        random_simulation_data, random_simulation_results = (
            simulate_and_organize_parameter_sets(
                random_parameter_samples, test_circuit_fitter, fitted_parameter_names
            )
        )

        # Store for combined visualization
        combined_best_simulation_data_dict[test_circuit_name] = {
            "config": test_circuit_configuration,
            "combined_params": best_simulation_data["combined_params"],
            "simulation_results": best_simulation_data["simulation_results"],
        }
        combined_random_simulation_data_dict[test_circuit_name] = {
            "config": test_circuit_configuration,
            "combined_params": random_simulation_data["combined_params"],
            "simulation_results": random_simulation_data["simulation_results"],
        }

        combined_best_results_dataframes.append(best_simulation_results)
        combined_random_results_dataframes.append(random_simulation_results)

    # Generate combined test circuit visualizations
    combined_best_results_dataframe = pd.concat(
        combined_best_results_dataframes, ignore_index=True
    )
    combined_random_results_dataframe = pd.concat(
        combined_random_results_dataframes, ignore_index=True
    )

    print("Generating combined hierarchical test circuit visualizations...")

    # Combined individual simulation plots
    plot_circuit_simulations(
        combined_best_simulation_data_dict,
        combined_best_results_dataframe,
        plot_mode="individual",
        likelihood_percentile_range=20,
    )
    plt.savefig(
        os.path.join(
            output_directory,
            "all_test_circuits_hierarchical_best_posterior_samples.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Combined summary simulation plots
    plot_circuit_simulations(
        combined_best_simulation_data_dict,
        combined_best_results_dataframe,
        plot_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
    )
    plt.savefig(
        os.path.join(
            output_directory,
            "all_test_circuits_hierarchical_best_posterior_summary.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Combined random simulation plots
    plot_circuit_simulations(
        combined_random_simulation_data_dict,
        combined_random_results_dataframe,
        plot_mode="individual",
        likelihood_percentile_range=20,
    )
    plt.savefig(
        os.path.join(
            output_directory,
            "all_test_circuits_hierarchical_random_posterior_samples.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_simulations(
        combined_random_simulation_data_dict,
        combined_random_results_dataframe,
        plot_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
    )
    plt.savefig(
        os.path.join(
            output_directory,
            "all_test_circuits_hierarchical_random_posterior_summary.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Generate experimental vs simulation overlay plots
    print("Generating hierarchical test circuit overlay visualizations...")

    plot_circuit_conditions_overlay(
        combined_best_simulation_data_dict,
        combined_best_results_dataframe,
        simulation_mode="individual",
    )
    plt.savefig(
        os.path.join(
            output_directory,
            "all_test_circuits_hierarchical_best_posterior_overlay_individual.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_conditions_overlay(
        combined_best_simulation_data_dict,
        combined_best_results_dataframe,
        simulation_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
    )
    plt.savefig(
        os.path.join(
            output_directory,
            "all_test_circuits_hierarchical_best_posterior_overlay_summary.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_conditions_overlay(
        combined_random_simulation_data_dict,
        combined_random_results_dataframe,
        simulation_mode="individual",
    )
    plt.savefig(
        os.path.join(
            output_directory,
            "all_test_circuits_hierarchical_random_posterior_overlay_individual.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_conditions_overlay(
        combined_random_simulation_data_dict,
        combined_random_results_dataframe,
        simulation_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
    )
    plt.savefig(
        os.path.join(
            output_directory,
            "all_test_circuits_hierarchical_random_posterior_overlay_summary.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def main():
    """Execute hierarchical test circuit simulation workflow"""

    # Configuration parameters
    hierarchical_mcmc_project_identifier = "simple_hierarchical_results_20250714_142603"
    hierarchical_mcmc_results_filepath = (
        f"../../data/fit_data/hierarchical/{hierarchical_mcmc_project_identifier}.csv"
    )
    test_circuit_output_directory = f"../../figures/hierarchical_test_circuits/{hierarchical_mcmc_project_identifier}"

    # Test circuit identifiers for generalization evaluation
    test_circuit_identifiers = [
        # "or_gate_c1ffl",
        # "iffl_1",
        # "cffl_12",
        "inhibited_incoherent_cascade",
        "inhibited_cascade",
    ]

    # Create output directory
    os.makedirs(test_circuit_output_directory, exist_ok=True)

    # Execute complete simulation workflow
    execute_hierarchical_test_circuit_simulations(
        test_circuit_identifiers=test_circuit_identifiers,
        hierarchical_mcmc_results_filepath=hierarchical_mcmc_results_filepath,
        output_directory=test_circuit_output_directory,
        simulation_sample_count=30,
        circuit_parameter_samples_per_test_circuit=1000,
        simulation_time_bounds_max=130,
        simulation_time_bounds_min=30,
    )

    print(
        f"Hierarchical test circuit simulations completed. Results saved to: {test_circuit_output_directory}"
    )


if __name__ == "__main__":
    main()
