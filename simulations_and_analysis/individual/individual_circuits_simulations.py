import os
import pandas as pd
import matplotlib.pyplot as plt
from data.circuits.circuit_configs import get_circuit_conditions, get_data_file
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from utils.process_experimental_data import organize_results
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor
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


def setup_calibration():
    """Set up GFP calibration parameters"""
    calibration_data = pd.read_csv("../../utils/calibration_gfp/gfp_Calibration.csv")
    calibration_results = fit_gfp_calibration(
        calibration_data,
        concentration_col="GFP Concentration (nM)",
        fluorescence_pattern="F.I. (a.u)",
    )
    brightness_correction, _ = get_brightness_correction_factor("avGFP", "sfGFP")

    return {
        "slope": calibration_results["slope"],
        "intercept": calibration_results["intercept"],
        "brightness_correction": brightness_correction,
    }


def create_circuit_simulation_data(
    circuit_name,
    mcmc_samples,
    parameters_to_fit,
    circuit_manager,
    calibration_parameters,
    time_bounds_max,
    time_bounds_min,
):
    """Create circuit configuration and simulate parameter sets"""
    circuit_conditions = get_circuit_conditions(circuit_name)
    experimental_data_file = get_data_file(circuit_name)
    experimental_data, time_span = load_and_process_csv(experimental_data_file)

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
    )

    circuit_fitter = CircuitFitter(
        [circuit_configuration],
        parameters_to_fit,
        pd.read_csv("../../data/prior/model_parameters_priors.csv"),
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

    linear_parameters = 10**log_parameters
    organized_results = organize_results(
        parameters_to_fit, linear_parameters, log_likelihoods, log_priors
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
):
    """Generate separate two-column and overlay plots for each individual circuit"""

    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    model_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    parameters_to_fit = model_priors[
        model_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()
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
            mcmc_raw_samples,
            parameters_to_fit,
            circuit_manager,
            calibration_parameters,
            time_bounds_max,
            time_bounds_min,
        )

        best_likelihood_samples = mcmc_final_samples.sort_values(
            by="likelihood", ascending=False
        )
        random_samples = mcmc_final_samples.sample(n=final_sample_size, random_state=42)

        # Generate plots for both sample types
        for sample_type, samples in [
            ("best", best_likelihood_samples),
            ("random", random_samples),
        ]:
            simulation_data, results_dataframe = simulate_and_organize_parameter_sets(
                samples, circuit_fitter, parameters_to_fit
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
):
    """Plot fits for each circuit using both best and random samples"""

    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    model_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    parameters_to_fit = model_priors[
        model_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()
    calibration_parameters = setup_calibration()

    combined_best_simulation_data = {}
    combined_random_simulation_data = {}
    combined_best_results = []
    combined_random_results = []

    for circuit_name, mcmc_raw_samples in mcmc_results_by_circuit.items():
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

        best_likelihood_samples = mcmc_final_samples.sort_values(
            by="likelihood", ascending=False
        )
        random_samples = mcmc_final_samples.sample(n=final_sample_size, random_state=42)

        # Create circuit configuration
        circuit_configuration, circuit_fitter = create_circuit_simulation_data(
            circuit_name,
            mcmc_raw_samples,
            parameters_to_fit,
            circuit_manager,
            calibration_parameters,
            time_bounds_max,
            time_bounds_min,
        )

        # Generate individual plots
        plot_individual_circuit(
            best_likelihood_samples,
            "best",
            circuit_name,
            circuit_fitter,
            parameters_to_fit,
            output_directory,
        )
        plot_individual_circuit(
            random_samples,
            "random",
            circuit_name,
            circuit_fitter,
            parameters_to_fit,
            output_directory,
        )

        # Prepare combined plotting data
        best_simulation_data, best_results_dataframe = (
            simulate_and_organize_parameter_sets(
                best_likelihood_samples, circuit_fitter, parameters_to_fit
            )
        )
        random_simulation_data, random_results_dataframe = (
            simulate_and_organize_parameter_sets(
                random_samples, circuit_fitter, parameters_to_fit
            )
        )

        combined_best_simulation_data[circuit_name] = {
            "config": circuit_configuration,
            "combined_params": best_simulation_data["combined_params"],
            "simulation_results": best_simulation_data["simulation_results"],
        }
        combined_random_simulation_data[circuit_name] = {
            "config": circuit_configuration,
            "combined_params": random_simulation_data["combined_params"],
            "simulation_results": random_simulation_data["simulation_results"],
        }

        combined_best_results.append(best_results_dataframe)
        combined_random_results.append(random_results_dataframe)

    # Generate combined plots
    combined_best_dataframe = pd.concat(combined_best_results, ignore_index=True)
    combined_random_dataframe = pd.concat(combined_random_results, ignore_index=True)

    print("Generating combined best fits figure...")
    plot_circuit_simulations(
        combined_best_simulation_data,
        combined_best_dataframe,
        plot_mode="individual",
        likelihood_percentile_range=20,
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_best_fits.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_simulations(
        combined_best_simulation_data,
        combined_best_dataframe,
        plot_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_best_fits_summary.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    print("Generating combined random fits figure...")
    plot_circuit_simulations(
        combined_random_simulation_data,
        combined_random_dataframe,
        plot_mode="individual",
        likelihood_percentile_range=20,
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
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_random_fits_summary.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Generate overlay plots: experimental | simulation
    print("Generating overlay plots...")

    plot_circuit_conditions_overlay(
        combined_best_simulation_data,
        combined_best_dataframe,
        simulation_mode="individual",
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_best_fits_overlay_individual.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_conditions_overlay(
        combined_best_simulation_data,
        combined_best_dataframe,
        simulation_mode="summary",
        summary_type="median_iqr",
        percentile_bounds=(10, 90),
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_best_fits_overlay_summary.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_conditions_overlay(
        combined_random_simulation_data,
        combined_random_dataframe,
        simulation_mode="individual",
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
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_random_fits_overlay_summary.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def main():
    subfolder = "/10000_steps_updated"
    subfolder = "/constrained_prior_3_tighter"
    subfolder = "/cross_val_circuits"

    input_directory = "../../data/fit_data/individual_circuits" + subfolder
    output_visualization_directory = "../../figures/individual_circuits" + subfolder

    mcmc_results = load_individual_circuit_results(input_directory)

    # Generate combined plots (existing functionality)
    plot_fits(
        mcmc_results,
        output_visualization_directory,
        sample_count=30,
        time_bounds_max=130,
        time_bounds_min=30,
    )

    # Generate per-circuit plots (new functionality)
    generate_per_circuit_individual_plots(
        mcmc_results,
        output_visualization_directory,
        sample_count=30,
        time_bounds_max=130,
        time_bounds_min=30,
    )


if __name__ == "__main__":
    main()
