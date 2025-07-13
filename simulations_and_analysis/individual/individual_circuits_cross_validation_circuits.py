import os
import pandas as pd
import matplotlib.pyplot as plt
from data.circuits.circuit_configs import get_circuit_conditions, get_data_file
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from utils.import_and_visualise_data import load_and_process_csv
from analysis_and_figures.mcmc_analysis_hierarchical import process_mcmc_data
from analysis_and_figures.plots_simulation import (
    plot_circuit_simulations,
    plot_circuit_conditions_overlay,
)
from simulations_and_analysis.individual.individual_circuits_statistics import (
    load_individual_circuit_results,
)
from individual_circuits_simulations import (
    setup_calibration,
    simulate_and_organize_parameter_sets,
)


def identify_transferable_parameters(
    source_mcmc_samples, target_circuit_manager, target_circuit_name
):
    """
    Identify parameters that can be transferred from source circuit to target circuit.

    Parameters:
    -----------
    source_mcmc_samples : pd.DataFrame
            MCMC samples from source circuit containing kinetic parameters
    target_circuit_manager : CircuitManager
            Circuit manager to get target circuit parameter structure
    target_circuit_name : str
            Name of target circuit

    Returns:
    --------
    list : Compatible parameter names that exist in both source and target
    """
    # Get target circuit's default parameters to understand its parameter structure
    # target_circuit_config = target_circuit_manager.get_circuit_config(
    #     target_circuit_name
    # )
    # target_default_parameters = target_circuit_config["default_parameters"]

    # Get kinetic parameters from model priors (these are typically shared across circuits)
    model_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    kinetic_parameter_names = model_priors[
        model_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()

    # Find parameters that exist in both source MCMC and target circuit
    source_parameter_names = source_mcmc_samples.columns.tolist()

    # Filter to kinetic parameters that exist in source MCMC samples
    transferable_kinetic_parameters = [
        param for param in kinetic_parameter_names if param in source_parameter_names
    ]

    print(f"Source circuit parameters: {len(source_parameter_names)}")
    print(f"Target circuit kinetic parameters: {len(kinetic_parameter_names)}")
    print(f"Transferable parameters: {len(transferable_kinetic_parameters)}")
    print(f"Transferable parameter list: {transferable_kinetic_parameters}")

    return transferable_kinetic_parameters


def extract_compatible_parameter_samples(
    source_mcmc_samples, transferable_parameter_names, sample_count=60
):
    """
    Extract random samples of transferable parameters from source circuit MCMC results.

    Parameters:
    -----------
    source_mcmc_samples : pd.DataFrame
            Processed MCMC samples from source circuit
    transferable_parameter_names : list
            Names of parameters that can be transferred
    sample_count : int
            Number of parameter sets to sample

    Returns:
    --------
    pd.DataFrame : Sampled parameter sets with only transferable parameters
    """
    # Sample random parameter sets
    available_sample_count = min(sample_count, len(source_mcmc_samples))
    random_parameter_samples = source_mcmc_samples.sample(
        n=available_sample_count, random_state=42
    )

    # Extract only transferable parameters
    transferable_parameter_samples = random_parameter_samples[
        transferable_parameter_names
    ]

    # Add likelihood column if available for visualization
    if "likelihood" in source_mcmc_samples.columns:
        transferable_parameter_samples = transferable_parameter_samples.copy()
        transferable_parameter_samples["likelihood"] = random_parameter_samples[
            "likelihood"
        ]

    print(
        f"Extracted {len(transferable_parameter_samples)} parameter sets with {len(transferable_parameter_names)} parameters each"
    )

    return transferable_parameter_samples


def create_target_circuit_configuration(
    target_circuit_name,
    transferable_parameter_names,
    circuit_manager,
    calibration_parameters,
    time_bounds_max,
    time_bounds_min,
):
    """Create circuit configuration for cross-validation without MCMC processing."""
    print(f"Loading experimental data for {target_circuit_name}...")

    target_circuit_conditions = get_circuit_conditions(target_circuit_name)
    target_experimental_data_file = get_data_file(target_circuit_name)

    print(f"Data file path: {target_experimental_data_file}")
    print(f"Circuit conditions: {list(target_circuit_conditions.keys())}")

    if target_experimental_data_file is None:
        raise ValueError(f"No data file found for circuit '{target_circuit_name}'")

    # Load experimental data
    target_experimental_data, target_time_span = load_and_process_csv(
        target_experimental_data_file
    )
    print(f"Successfully loaded data. Time span: {len(target_time_span)} points")
    print(f"Experimental data shape: {target_experimental_data.shape}")
    print(f"Experimental data columns: {target_experimental_data.columns.tolist()}")

    # Debug condition name mismatch
    csv_condition_names = target_experimental_data["condition"].unique().tolist()
    config_condition_names = list(target_circuit_conditions.keys())
    print(f"CSV condition names: {csv_condition_names}")
    print(f"Config condition names: {config_condition_names}")

    # Check if condition names match
    if not set(csv_condition_names).issubset(set(config_condition_names)):
        print(f"WARNING: Condition name mismatch detected for {target_circuit_name}")
        print(
            f"CSV conditions not in config: {set(csv_condition_names) - set(config_condition_names)}"
        )
        print(
            "This may cause integration failures due to uninitialized concentration parameters"
        )

    first_target_condition = list(target_circuit_conditions.keys())[0]
    print(f"Creating circuit instance with condition: {first_target_condition}")
    print(f"Condition parameters: {target_circuit_conditions[first_target_condition]}")

    target_circuit_instance = circuit_manager.create_circuit(
        target_circuit_name,
        parameters=target_circuit_conditions[first_target_condition],
    )
    print("Circuit instance created successfully")

    target_circuit_configuration = CircuitConfig(
        model=target_circuit_instance.model,
        name=target_circuit_name,
        condition_params=target_circuit_conditions,
        experimental_data=target_experimental_data,
        tspan=target_time_span,
        max_time=time_bounds_max,
        min_time=time_bounds_min,
    )

    model_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    target_circuit_fitter = CircuitFitter(
        [target_circuit_configuration],
        transferable_parameter_names,
        model_priors,
        calibration_parameters,
    )

    print(
        f"Circuit configuration and fitter created successfully for {target_circuit_name}"
    )
    return target_circuit_configuration, target_circuit_fitter


def simulate_cross_validation_circuit(
    target_circuit_name,
    source_parameter_samples,
    transferable_parameter_names,
    circuit_manager,
    calibration_parameters,
    time_bounds_max=None,
    time_bounds_min=None,
):
    """Simulate target circuit using parameters from source circuit."""
    print(f"Cross-validating {target_circuit_name} using transferred parameters...")

    target_circuit_configuration, target_circuit_fitter = (
        create_target_circuit_configuration(
            target_circuit_name,
            transferable_parameter_names,
            circuit_manager,
            calibration_parameters,
            time_bounds_max,
            time_bounds_min,
        )
    )

    print(
        f"Source parameter sample columns: {source_parameter_samples.columns.tolist()}"
    )
    print(f"Transferable parameter names: {transferable_parameter_names}")
    print(f"Parameter sample shape: {source_parameter_samples.shape}")

    # Direct simulation - process_mcmc_data() already returns log10 space parameters
    target_simulation_results, target_organized_results = (
        simulate_and_organize_parameter_sets(
            source_parameter_samples,
            target_circuit_fitter,
            transferable_parameter_names,
        )
    )

    return (
        target_circuit_configuration,
        target_simulation_results,
        target_organized_results,
    )


def plot_cross_validation_comparison(
    target_circuit_name,
    source_circuit_name,
    target_simulation_results,
    target_organized_results,
    target_circuit_configuration,
    output_directory,
    sample_count,
):
    """
    Generate cross-validation plots comparing experimental data vs transferred parameter predictions.

    Parameters:
    -----------
    target_circuit_name : str
            Name of target circuit being validated
    source_circuit_name : str
            Name of source circuit providing parameters
    target_simulation_results : dict
            Simulation results for target circuit
    target_organized_results : pd.DataFrame
            Organized results dataframe
    target_circuit_configuration : CircuitConfig
            Circuit configuration for target
    output_directory : str
            Directory to save plots
    sample_count : int
            Number of samples used
    """
    # Create individual cross-validation plot
    plt.figure(figsize=(12, 8))
    target_simulation_data_dict = {target_circuit_name: target_simulation_results}

    plot_circuit_simulations(
        target_simulation_data_dict,
        target_organized_results,
        plot_mode="individual",
        likelihood_percentile_range=20,
    )

    cross_validation_title = f"Cross-Validation: {target_circuit_name} using {source_circuit_name} parameters ({sample_count} samples)"
    plt.suptitle(cross_validation_title)

    cross_validation_filename = f"cross_validation_{source_circuit_name}_to_{target_circuit_name}_individual.png"
    plt.savefig(
        os.path.join(output_directory, cross_validation_filename),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Create summary cross-validation plot
    plt.figure(figsize=(12, 8))
    plot_circuit_simulations(
        target_simulation_data_dict,
        target_organized_results,
        plot_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
    )

    summary_title = f"Cross-Validation Summary: {target_circuit_name} using {source_circuit_name} parameters"
    plt.suptitle(summary_title)

    summary_filename = (
        f"cross_validation_{source_circuit_name}_to_{target_circuit_name}_summary.png"
    )
    plt.savefig(
        os.path.join(output_directory, summary_filename), bbox_inches="tight", dpi=300
    )
    plt.close()

    # Create overlay comparison plot
    plot_circuit_conditions_overlay(
        target_simulation_data_dict,
        target_organized_results,
        simulation_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
    )

    overlay_title = f"Cross-Validation Overlay: {target_circuit_name} | {source_circuit_name} parameters"
    plt.suptitle(overlay_title)

    overlay_filename = (
        f"cross_validation_{source_circuit_name}_to_{target_circuit_name}_overlay.png"
    )
    plt.savefig(
        os.path.join(output_directory, overlay_filename), bbox_inches="tight", dpi=300
    )
    plt.close()

    print(f"✓ Cross-validation plots saved for {target_circuit_name}")


def run_cross_validation_analysis(
    source_circuit_name,
    target_circuit_names,
    mcmc_results_by_circuit,
    output_directory=".",
    sample_count=60,
    time_bounds_max=None,
    time_bounds_min=None,
):
    """
    Perform cross-validation analysis using one circuit's parameters to simulate others.

    Parameters:
    -----------
    source_circuit_name : str
            Name of circuit providing MCMC parameters (e.g., 'cascade')
    target_circuit_names : list
            Names of circuits to validate (e.g., ['iffl_1', 'cffl_12'])
    mcmc_results_by_circuit : dict
            Dictionary containing MCMC results for all circuits
    output_directory : str
            Directory to save cross-validation plots
    sample_count : int
            Number of parameter samples to use for validation
    time_bounds_max : float, optional
            Maximum time bound for simulations
    time_bounds_min : float, optional
            Minimum time bound for simulations
    """

    print("=== Cross-Validation Analysis ===")
    print(f"Source circuit: {source_circuit_name}")
    print(f"Target circuits: {target_circuit_names}")
    print(f"Sample count: {sample_count}")

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Initialize circuit manager and calibration
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )
    calibration_parameters = setup_calibration()

    # Load and process source circuit MCMC results
    if source_circuit_name not in mcmc_results_by_circuit:
        raise ValueError(
            f"Source circuit '{source_circuit_name}' not found in MCMC results"
        )

    source_mcmc_raw_samples = mcmc_results_by_circuit[source_circuit_name]
    source_mcmc_processed = process_mcmc_data(
        source_mcmc_raw_samples, burn_in=0.4, chain_idx=0
    )
    source_mcmc_filtered_samples = source_mcmc_processed["processed_data"]

    print(
        f"Source circuit {source_circuit_name}: {len(source_mcmc_raw_samples)} → {len(source_mcmc_filtered_samples)} samples after burn-in"
    )

    # Perform cross-validation for each target circuit
    for target_circuit_name in target_circuit_names:
        print(f"\n--- Cross-validating {target_circuit_name} ---")

        # Identify transferable parameters
        transferable_parameter_names = identify_transferable_parameters(
            source_mcmc_filtered_samples, circuit_manager, target_circuit_name
        )

        if not transferable_parameter_names:
            print(
                f"⚠️  No transferable parameters found for {target_circuit_name}, skipping..."
            )
            continue

        # Extract compatible parameter samples
        transferable_parameter_samples = extract_compatible_parameter_samples(
            source_mcmc_filtered_samples, transferable_parameter_names, sample_count
        )

        # Simulate target circuit with transferred parameters
        try:
            (
                target_circuit_configuration,
                target_simulation_results,
                target_organized_results,
            ) = simulate_cross_validation_circuit(
                target_circuit_name,
                transferable_parameter_samples,
                transferable_parameter_names,
                circuit_manager,
                calibration_parameters,
                time_bounds_max,
                time_bounds_min,
            )

            # Generate cross-validation plots
            plot_cross_validation_comparison(
                target_circuit_name,
                source_circuit_name,
                target_simulation_results,
                target_organized_results,
                target_circuit_configuration,
                output_directory,
                len(transferable_parameter_samples),
            )

        except Exception as simulation_error:
            print(f"❌ Error simulating {target_circuit_name}: {simulation_error}")
            continue

    print("\n=== Cross-validation completed ===")
    print(f"Results saved to: {output_directory}")


def main_cross_validation():
    """Main function to run cross-validation analysis"""

    # Configuration
    subfolder = "/updated_constrained_prior_2_heteroscedastic_model"
    input_directory = "../../data/fit_data/individual_circuits" + subfolder
    output_directory = "../../figures/cross_validation" + subfolder

    # Source circuit (providing parameters)
    source_circuit_name = "star_antistar_1"

    # Target circuits (to validate)
    target_circuit_names = ["iffl_1", "cffl_12"]

    # Load MCMC results
    mcmc_results = load_individual_circuit_results(input_directory)

    # Check if source circuit exists in results
    if source_circuit_name not in mcmc_results:
        available_circuits = list(mcmc_results.keys())
        print(f"Source circuit '{source_circuit_name}' not found.")
        print(f"Available circuits: {available_circuits}")
        return

    # Run cross-validation analysis
    run_cross_validation_analysis(
        source_circuit_name=source_circuit_name,
        target_circuit_names=target_circuit_names,
        mcmc_results_by_circuit=mcmc_results,
        output_directory=output_directory,
        sample_count=60,
        time_bounds_max=130,
        time_bounds_min=30,
    )


if __name__ == "__main__":
    main_cross_validation()
