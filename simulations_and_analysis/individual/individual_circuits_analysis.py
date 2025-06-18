"""
Individual Circuits Analysis: Convert individual circuit fits to hierarchical format for comparison
"""

import os
import glob
import pandas as pd
import numpy as np
from analysis_and_figures.hierarchical_pairplot_analysis import (
    create_hierarchical_histogram_grid,
)
from analysis_and_figures.hierarchical_pairplot_analysis import (
    create_circuit_prior_comparison_pairplot,
)
from analysis_and_figures.mcmc_analysis_hierarchical import (
    process_mcmc_data,
)


def load_individual_circuit_results(
    individual_results_directory="../../data/fit_data/individual_circuits",
):
    """Load individual circuit MCMC results from CSV files"""
    individual_circuit_fits = {}
    results_pattern = os.path.join(individual_results_directory, "results_*.csv")
    result_filepaths = glob.glob(results_pattern)

    for filepath in result_filepaths:
        filename = os.path.basename(filepath)
        # Extract: results_circuit_name_timestamp.csv -> circuit_name
        circuit_name_parts = filename.split("_")[1:-2]
        circuit_name = "_".join(circuit_name_parts)

        circuit_results = pd.read_csv(filepath)
        individual_circuit_fits[circuit_name] = circuit_results

        print(
            f"Loaded {circuit_name}: {len(circuit_results)} samples, "
            f"best likelihood: {circuit_results['likelihood'].max():.2f}"
        )

    return individual_circuit_fits


def convert_individual_to_theta_format(
    individual_circuit_mcmc_results,
    fitted_parameter_names,
    target_circuit_names,
    burn_in_fraction=0.8,
    post_burnin_samples_per_circuit=3000,
):
    """Convert individual circuit DataFrames to hierarchical θ format with chain 0 + burn-in filtering"""
    theta_formatted_samples = []

    for (
        circuit_name,
        circuit_mcmc_raw_samples,
    ) in individual_circuit_mcmc_results.items():
        if circuit_name not in target_circuit_names:
            continue

        # Apply burn-in filtering and chain 0 selection using existing function
        processed_data_result = process_mcmc_data(
            circuit_mcmc_raw_samples, burn_in=burn_in_fraction, chain_idx=0
        )
        circuit_filtered_samples = processed_data_result["processed_data"]

        print(
            f"{circuit_name}: {processed_data_result['metadata']['n_samples_raw']} → "
            f"{processed_data_result['metadata']['n_samples_processed']} samples after burn-in"
        )

        # Sample from filtered data
        circuit_final_samples = (
            circuit_filtered_samples.sample(
                n=post_burnin_samples_per_circuit, random_state=42
            )
            if len(circuit_filtered_samples) > post_burnin_samples_per_circuit
            else circuit_filtered_samples.copy()
        )

        # Extract parameter columns and format
        circuit_theta_parameters = circuit_final_samples[fitted_parameter_names].copy()
        circuit_theta_parameters["sample_id"] = range(len(circuit_theta_parameters))
        circuit_theta_parameters["type"] = "Circuit"
        circuit_theta_parameters["circuit"] = circuit_name
        circuit_theta_parameters["Circuit"] = circuit_name

        theta_formatted_samples.append(circuit_theta_parameters)

    return pd.concat(theta_formatted_samples, ignore_index=True)


def generate_prior_mean_markers(prior_parameters_filepath, fitted_parameter_names):
    """Generate single prior mean markers instead of distributions"""
    prior_parameter_specifications = pd.read_csv(prior_parameters_filepath)
    prior_mean_markers = {"sample_id": [0]}

    for _, prior_specification_row in prior_parameter_specifications.iterrows():
        if prior_specification_row["Parameter"] in fitted_parameter_names:
            parameter_name = prior_specification_row["Parameter"]
            log10_mean = np.log10(prior_specification_row["Mean"])  # Fixed conversion
            prior_mean_markers[parameter_name] = [log10_mean]

    # Fill missing parameters with NaN
    for parameter_name in fitted_parameter_names:
        if parameter_name not in prior_mean_markers:
            prior_mean_markers[parameter_name] = [np.nan]

    prior_mean_markers.update(
        {"type": ["Prior"], "circuit": ["Prior"], "Circuit": ["Prior"]}
    )

    return pd.DataFrame(prior_mean_markers)


def generate_prior_mean_coordinates(prior_parameters_filepath, fitted_parameter_names):
    """Generate single prior mean coordinates (no distributions)"""
    prior_parameter_specifications = pd.read_csv(prior_parameters_filepath)
    prior_mean_coordinates = {"sample_id": [0]}

    for _, prior_specification_row in prior_parameter_specifications.iterrows():
        if prior_specification_row["Parameter"] in fitted_parameter_names:
            parameter_name = prior_specification_row["Parameter"]
            log10_mean = np.log10(
                prior_specification_row["Mean"]
            )  # FIXED: Convert to log10
            prior_mean_coordinates[parameter_name] = [log10_mean]

    # Fill missing parameters with NaN
    for parameter_name in fitted_parameter_names:
        if parameter_name not in prior_mean_coordinates:
            prior_mean_coordinates[parameter_name] = [np.nan]

    prior_mean_coordinates.update(
        {"type": ["Prior"], "circuit": ["Prior"], "Circuit": ["Prior"]}
    )

    return pd.DataFrame(prior_mean_coordinates)


def execute_individual_to_hierarchical_comparison(
    individual_results_directory,
    prior_parameters_filepath,
    fitted_parameter_names,
    output_visualization_directory,
    target_circuit_names=None,
    weight_alpha_by_likelihood=False,
):
    """Execute complete individual-to-hierarchical comparison pipeline"""

    os.makedirs(output_visualization_directory, exist_ok=True)

    # Load individual circuit MCMC results
    individual_circuit_fits = load_individual_circuit_results(
        individual_results_directory
    )
    available_circuit_names = list(individual_circuit_fits.keys())

    if target_circuit_names is None:
        target_circuit_names = available_circuit_names

    print(f"Target circuits for analysis: {target_circuit_names}")

    # Convert to θ format (circuit-specific parameters only)
    theta_formatted_data = convert_individual_to_theta_format(
        individual_circuit_fits, fitted_parameter_names, target_circuit_names
    )

    # Generate prior mean coordinates (single points)
    prior_mean_coordinates = generate_prior_mean_coordinates(
        prior_parameters_filepath, fitted_parameter_names
    )

    # Combine: circuits + prior means (NO ALPHA)
    circuit_prior_comparison_dataset = pd.concat(
        [theta_formatted_data, prior_mean_coordinates], ignore_index=True
    )

    print(f"Comparison dataset: {len(circuit_prior_comparison_dataset)} samples")
    print(f"Data groups: {circuit_prior_comparison_dataset['Circuit'].unique()}")

    # Generate visualizations using NEW functi

    create_hierarchical_histogram_grid(
        circuit_prior_comparison_dataset,
        fitted_parameter_names,
        output_visualization_directory,
    )
    create_circuit_prior_comparison_pairplot(
        circuit_prior_comparison_dataset,
        fitted_parameter_names,
        output_visualization_directory,
        diagonal_visualization_type="kde",
        offdiagonal_visualization_type="kde",
    )

    return circuit_prior_comparison_dataset


def main():
    """Execute individual circuits hierarchical comparison analysis"""

    # Configuration
    individual_results_directory = "../../data/fit_data/individual_circuits"
    prior_parameters_filepath = "../../data/prior/model_parameters_priors.csv"
    output_visualization_directory = "../../figures/individual_hierarchical_comparison"

    # Load parameter specifications
    prior_parameters = pd.read_csv(prior_parameters_filepath)
    fitted_parameter_names = prior_parameters[
        prior_parameters["Parameter"] != "k_prot_deg"
    ]["Parameter"].tolist()

    print(f"Fitted parameters: {fitted_parameter_names}")

    # Execute comparison pipeline
    _ = execute_individual_to_hierarchical_comparison(
        individual_results_directory=individual_results_directory,
        prior_parameters_filepath=prior_parameters_filepath,
        fitted_parameter_names=fitted_parameter_names,
        output_visualization_directory=output_visualization_directory,
        weight_alpha_by_likelihood=False,  # Set to True for likelihood-weighted α
    )

    print(f"Analysis complete. Results saved to: {output_visualization_directory}")


if __name__ == "__main__":
    main()
