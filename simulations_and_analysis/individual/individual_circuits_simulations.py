import os
import pandas as pd
import matplotlib.pyplot as plt
from data.circuits.circuit_configs import get_circuit_conditions, get_data_file
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from utils.process_experimental_data import organize_results
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import setup_calibration, convert_nm_to_au
from analysis_and_figures.mcmc_analysis_hierarchical import process_mcmc_data
from analysis_and_figures.plots_simulation import (
    plot_circuit_simulations,
    plot_circuit_conditions_overlay,
    extract_trajectory_data,
    plot_single_circuit_two_column,
    plot_single_circuit_overlay,
)
from simulations_and_analysis.individual.individual_circuits_statistics import (
    load_individual_circuit_results,
)


def create_circuit_simulation_data(
    circuit_name,
    circuit_manager,
    calibration_parameters,
    time_bounds_max,
    time_bounds_min,
    priors_csv_path="../../data/prior/model_parameters_priors_updated_tighter.csv",
):
    """Create circuit configuration and simulate parameter sets"""
    circuit_conditions = get_circuit_conditions(circuit_name)
    experimental_data_file = get_data_file(circuit_name)
    experimental_data, time_span = load_and_process_csv(experimental_data_file)
    prior_kinetic_rates = pd.read_csv(priors_csv_path)

    first_condition = list(circuit_conditions.keys())[0]
    circuit_instance = circuit_manager.create_circuit(
        circuit_name, parameters=circuit_conditions[first_condition]
    )

    circuit_configuration = CircuitConfig(
        model=circuit_instance.model,
        name=circuit_name,
        condition_params=circuit_conditions,
        experimental_data=experimental_data,
        tspan=time_span,
        max_time=time_bounds_max,
        min_time=time_bounds_min,
        calibration_params=calibration_parameters,
    )

    kinetic_parameters = [
        param
        for param in prior_kinetic_rates.Parameter.to_list()
        if param in circuit_configuration.model_parameters
    ]

    circuit_fitter = CircuitFitter(
        [circuit_configuration],
        kinetic_parameters,
        prior_kinetic_rates,
        calibration_parameters,
    )

    return circuit_configuration, circuit_fitter


def simulate_and_organize_parameter_sets(
    parameter_samples, circuit_fitter, parameters_to_fit
):
    """Simulate parameters and organize results"""
    log_parameters = parameter_samples[parameters_to_fit].values
    simulation_results = circuit_fitter.simulate_parameters(log_parameters)
    log_likelihoods = (
        circuit_fitter.calculate_likelihood_from_simulation_with_breakdown(
            simulation_results
        )
    )
    log_priors = circuit_fitter.calculate_log_prior(log_parameters)

    organized_results = organize_results(
        parameters_to_fit,
        log_parameters,
        log_likelihoods,
        log_priors,  # Pass log space parameters
    )

    return simulation_results[0], organized_results


def plot_individual_circuit(
    parameter_samples,
    sample_type,
    circuit_name,
    circuit_fitter,
    parameters_to_fit,
    output_directory,
):
    """Plot individual circuit parameter fits"""
    simulation_data, results_dataframe = simulate_and_organize_parameter_sets(
        parameter_samples, circuit_fitter, parameters_to_fit
    )

    plt.figure(figsize=(12, 8))
    circuit_name_keyed_data = {circuit_name: simulation_data}
    plot_circuit_simulations(
        circuit_name_keyed_data,
        results_dataframe,
        plot_mode="individual",
        likelihood_percentile_range=20,
    )

    sample_count = len(parameter_samples)
    plot_title = f"{'Top' if sample_type == 'best' else 'Random'} {sample_count} Fits for {circuit_name}"

    plt.suptitle(plot_title)
    plt.savefig(
        os.path.join(output_directory, f"{sample_type}_fits_{circuit_name}.png")
    )
    plt.close()


def generate_per_circuit_individual_plots(
    mcmc_results_by_circuit,
    output_directory,
    sample_count,
    time_bounds_max,
    time_bounds_min,
    priors_csv_path,
):
    """Generate separate two-column and overlay plots for each individual circuit"""

    circuit_manager = CircuitManager(
        parameters_file=priors_csv_path,
        json_file="../../data/circuits/circuits.json",
    )

    # model_priors = pd.read_csv(priors_csv_path)
    calibration_parameters = setup_calibration()

    for circuit_name, mcmc_raw_samples in mcmc_results_by_circuit.items():
        print(f"Generating per-circuit plots for individual circuit {circuit_name}")

        # Process MCMC samples with burn-in filtering
        mcmc_processed_result = process_mcmc_data(
            mcmc_raw_samples, burn_in=0.4, chain_idx=0
        )
        mcmc_filtered_samples = mcmc_processed_result["processed_data"]

        print(
            f"{circuit_name}: {len(mcmc_raw_samples)} → {len(mcmc_filtered_samples)} samples after burn-in"
        )

        final_sample_size = min(sample_count, len(mcmc_filtered_samples))
        mcmc_final_samples = (
            mcmc_filtered_samples.sample(n=final_sample_size, random_state=42)
            if len(mcmc_filtered_samples) > final_sample_size
            else mcmc_filtered_samples.copy()
        )

        # Create circuit configuration and fitter
        circuit_configuration, circuit_fitter = create_circuit_simulation_data(
            circuit_name,
            circuit_manager,
            calibration_parameters,
            time_bounds_max,
            time_bounds_min,
        )

        random_samples = mcmc_final_samples.sample(n=final_sample_size, random_state=42)

        # Generate plots for both sample types
        for sample_type, samples in [
            ("random", random_samples),
        ]:
            simulation_data, results_dataframe = simulate_and_organize_parameter_sets(
                samples,
                circuit_fitter,
                circuit_fitter.parameters_to_fit,
            )

            # Prepare single-circuit data structure
            single_circuit_simulation_dict = {
                circuit_name: {
                    "config": circuit_configuration,
                    "combined_params": simulation_data["combined_params"],
                    "simulation_results": simulation_data["simulation_results"],
                }
            }

            trajectory_data = extract_trajectory_data(
                single_circuit_simulation_dict, results_dataframe
            )
            circuit_trajectory_data = trajectory_data[
                trajectory_data["circuit"] == circuit_name
            ]
            circuit_data = single_circuit_simulation_dict[circuit_name]

            circuit_trajectory_data["protein_concentration"] = convert_nm_to_au(
                circuit_trajectory_data["protein_concentration"],
                circuit_fitter.calibration_params["slope"],
                circuit_fitter.calibration_params["intercept"],
                circuit_fitter.calibration_params["brightness_correction"],
            )

            # circuit_trajectory_data.to_csv(
            #     "../../data/data_parameter_estimation/constitutive_sfGFP_simulated_data_au.csv",
            #     index=False,
            # )

            # Generate two-column plots (experimental | simulation)
            for simulation_mode in ["individual", "summary"]:
                _ = plot_single_circuit_two_column(
                    circuit_name,
                    circuit_data,
                    circuit_trajectory_data,
                    results_dataframe,
                    simulation_mode=simulation_mode,
                    summary_type="median_iqr",
                    percentile_bounds=(10, 90),
                )

                mode_suffix = (
                    "_summary" if simulation_mode == "summary" else "_individual"
                )
                two_column_filename = f"individual_{sample_type}_{circuit_name}_two_column{mode_suffix}.png"
                plt.savefig(
                    os.path.join(output_directory, two_column_filename),
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()

            # Generate overlay plots (experimental + simulation superposed)
            for simulation_mode in ["individual", "summary"]:
                _ = plot_single_circuit_overlay(
                    circuit_name,
                    circuit_data,
                    circuit_trajectory_data,
                    results_dataframe,
                    simulation_mode=simulation_mode,
                    summary_type="median_iqr",
                    percentile_bounds=(10, 90),
                    figsize=(6, 4),
                    title_true=False,
                )

                mode_suffix = (
                    "_summary" if simulation_mode == "summary" else "_individual"
                )
                overlay_filename = (
                    f"individual_{sample_type}_{circuit_name}_overlay{mode_suffix}.png"
                )
                plt.savefig(
                    os.path.join(output_directory, overlay_filename),
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()


def plot_fits(
    mcmc_results_by_circuit,
    output_directory=".",
    sample_count=60,
    time_bounds_max=None,
    time_bounds_min=None,
    priors_csv_path="../../data/prior/model_parameters_priors_updated_tighter.csv",
):
    """Plot fits for each circuit using both best and random samples"""

    circuit_manager = CircuitManager(
        parameters_file=priors_csv_path,
        json_file="../../data/circuits/circuits.json",
    )

    # model_priors = pd.read_csv(priors_csv_path)

    calibration_parameters = setup_calibration()

    combined_random_simulation_data = {}
    combined_random_results = []

    for circuit_name, mcmc_raw_samples in mcmc_results_by_circuit.items():
        # skip constitutive sfGFP
        if circuit_name == "constitutive sfGFP":
            continue

        print(f"Processing circuit {circuit_name}")

        # Filter and sample MCMC data
        mcmc_processed = process_mcmc_data(mcmc_raw_samples, burn_in=0.4, chain_idx=0)
        mcmc_filtered_samples = mcmc_processed["processed_data"]

        print(
            f"{circuit_name}: {len(mcmc_raw_samples)} → {len(mcmc_filtered_samples)} samples after burn-in"
        )

        final_sample_size = min(sample_count, len(mcmc_filtered_samples))
        mcmc_final_samples = (
            mcmc_filtered_samples.sample(n=final_sample_size, random_state=42)
            if len(mcmc_filtered_samples) > final_sample_size
            else mcmc_filtered_samples.copy()
        )

        random_samples = mcmc_final_samples.sample(n=final_sample_size, random_state=42)

        # Create circuit configuration
        # Create circuit configuration
        circuit_configuration, circuit_fitter = create_circuit_simulation_data(
            circuit_name,
            circuit_manager,
            calibration_parameters,
            time_bounds_max,
            time_bounds_min,
        )

        plot_individual_circuit(
            random_samples,
            "random",
            circuit_name,
            circuit_fitter,
            circuit_fitter.parameters_to_fit,
            output_directory,
        )

        random_simulation_data, random_results_dataframe = (
            simulate_and_organize_parameter_sets(
                random_samples,
                circuit_fitter,
                circuit_fitter.parameters_to_fit,
            )
        )

        combined_random_simulation_data[circuit_name] = {
            "config": circuit_configuration,
            "combined_params": random_simulation_data["combined_params"],
            "simulation_results": random_simulation_data["simulation_results"],
        }

        combined_random_results.append(random_results_dataframe)

    # Generate combined plots
    combined_random_dataframe = pd.concat(combined_random_results, ignore_index=True)

    print("Generating combined random fits figure...")
    plot_circuit_simulations(
        combined_random_simulation_data,
        combined_random_dataframe,
        plot_mode="individual",
        likelihood_percentile_range=20,
        # show_title=False,
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_random_fits.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_simulations(
        combined_random_simulation_data,
        combined_random_dataframe,
        plot_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
        show_title=False,
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_random_fits_summary.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_conditions_overlay(
        combined_random_simulation_data,
        combined_random_dataframe,
        simulation_mode="individual",
        show_title=False,
    )
    plt.savefig(
        os.path.join(
            output_directory, "all_circuits_random_fits_overlay_individual.png"
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_conditions_overlay(
        combined_random_simulation_data,
        combined_random_dataframe,
        simulation_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
        show_title=False,
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_random_fits_overlay_summary.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def main():
    subfolder = "/50000_steps"
    # subfolder = "/conv_AU_corr"
    # subfolder = "/cross_val_circuits"
    subfolder = "/transfer_learning"
    subfolder = "/fit_data_2025-08-28_50000_steps_Generalized_Adaptive_Metropolis_with_Global_Scaling/individual_circuits"

    input_directory = "../../data/fit_data/individual_circuits" + subfolder
    output_visualization_directory = "../../figures/individual_circuits" + subfolder
    priors_csv_path = "../../data/prior/model_parameters_priors_updated_tighter.csv"

    # creatre output directory if it does not exist
    os.makedirs(output_visualization_directory, exist_ok=True)

    mcmc_results = load_individual_circuit_results(input_directory)

    # Define processing order and inclusion
    circuit_processing_sequence = [
        # "constitutive sfGFP",
        "sense_star_6",
        "toehold_trigger",
        "cascade",
        "cffl_type_1",
        "or_gate_c1ffl",
        "star_antistar_1",
        "trigger_antitrigger",
        "inhibited_incoherent_cascade",
        "inhibited_cascade",
        "cffl_12",
    ]

    filtered_mcmc_results = {
        circuit_name: mcmc_results[circuit_name]
        for circuit_name in circuit_processing_sequence
        if circuit_name in mcmc_results
    }

    # Generate combined plots (existing functionality)
    plot_fits(
        filtered_mcmc_results,
        output_visualization_directory,
        sample_count=10,
        time_bounds_max=130,
        time_bounds_min=30,
        priors_csv_path=priors_csv_path,
    )

    # Generate per-circuit plots (new functionality)
    generate_per_circuit_individual_plots(
        filtered_mcmc_results,
        output_visualization_directory,
        sample_count=1,
        time_bounds_max=130,
        time_bounds_min=30,
        priors_csv_path=priors_csv_path,
    )


if __name__ == "__main__":
    main()
