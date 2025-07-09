import os
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


def extract_circuit_theta_parameters(
    hierarchical_results, circuit_name, parameters_to_fit
):
    """Extract circuit-specific theta parameters from hierarchical MCMC results"""
    theta_columns = [f"theta_{circuit_name}_{param}" for param in parameters_to_fit]

    # Check which theta columns exist in the data
    available_theta_columns = [
        col for col in theta_columns if col in hierarchical_results.columns
    ]

    if not available_theta_columns:
        print(f"Warning: No theta parameters found for circuit {circuit_name}")
        return None

    # Extract theta parameters and rename to match individual circuit format
    circuit_theta_parameters = hierarchical_results[available_theta_columns].copy()

    # Rename columns from theta_circuit_parameter to parameter
    column_mapping = {
        f"theta_{circuit_name}_{param}": param
        for param in parameters_to_fit
        if f"theta_{circuit_name}_{param}" in available_theta_columns
    }
    circuit_theta_parameters = circuit_theta_parameters.rename(columns=column_mapping)

    # Add likelihood column if available (use sum of all circuit likelihoods or global likelihood)
    if "likelihood" in hierarchical_results.columns:
        circuit_theta_parameters["likelihood"] = hierarchical_results["likelihood"]
    else:
        # Set dummy likelihood for sorting
        circuit_theta_parameters["likelihood"] = range(len(circuit_theta_parameters))

    print(f"Extracted {len(circuit_theta_parameters)} theta samples for {circuit_name}")
    print(f"Available parameters: {list(circuit_theta_parameters.columns)}")

    return circuit_theta_parameters


def load_hierarchical_circuit_results(
    hierarchical_results_filepath, parameters_to_fit, burn_in_fraction=0.4
):
    """Load hierarchical results and extract circuit-specific parameters"""
    # Load hierarchical results
    hierarchical_raw_results = pd.read_csv(hierarchical_results_filepath)

    # Handle column naming issue if present (walker/iteration swap)
    if (
        "walker" in hierarchical_raw_results.columns
        and "iteration" in hierarchical_raw_results.columns
    ):
        hierarchical_raw_results = hierarchical_raw_results.rename(
            columns={"walker": "iteration", "iteration": "walker"}
        )

    print(f"Loaded hierarchical results: {len(hierarchical_raw_results)} raw samples")

    # Apply burn-in filtering
    hierarchical_processed = process_mcmc_data(
        hierarchical_raw_results, burn_in=burn_in_fraction, chain_idx=0
    )
    hierarchical_filtered_results = hierarchical_processed["processed_data"]

    print(
        f"After burn-in ({burn_in_fraction * 100:.0f}%): {len(hierarchical_filtered_results)} samples"
    )

    # Extract circuit names from theta columns
    theta_column_patterns = [
        col for col in hierarchical_filtered_results.columns if col.startswith("theta_")
    ]
    print(f"DEBUG: Sample theta columns: {theta_column_patterns[:10]}")

    # Parse circuit names by finding the parameter part and working backwards
    circuit_names_with_params = []
    known_parameters = set(parameters_to_fit)

    for theta_column in theta_column_patterns:
        column_parts = theta_column.split("_")
        # Find where the parameter name starts by checking against known parameters
        for param_start_index in range(2, len(column_parts)):
            potential_param = "_".join(column_parts[param_start_index:])
            if potential_param in known_parameters:
                circuit_name = "_".join(column_parts[1:param_start_index])
                circuit_names_with_params.append((circuit_name, potential_param))
                break

    circuit_names = list(set([name for name, _ in circuit_names_with_params]))
    print(f"Found circuits in hierarchical results: {circuit_names}")
    print(f"DEBUG: Circuit-parameter pairs: {circuit_names_with_params[:5]}")

    # Extract circuit-specific parameters
    circuit_theta_results = {}
    for circuit_name in circuit_names:
        circuit_parameters = extract_circuit_theta_parameters(
            hierarchical_filtered_results, circuit_name, parameters_to_fit
        )
        if circuit_parameters is not None:
            circuit_theta_results[circuit_name] = circuit_parameters

    return circuit_theta_results


def generate_hierarchical_individual_plots(
    circuit_name,
    theta_samples,
    circuit_fitter,
    parameters_to_fit,
    output_directory,
    sample_count,
):
    """Generate individual circuit plots using existing plot_circuit_simulations() directly"""
    final_sample_size = min(sample_count, len(theta_samples))
    theta_final_samples = (
        theta_samples.sample(n=final_sample_size, random_state=42)
        if len(theta_samples) > final_sample_size
        else theta_samples.copy()
    )

    best_theta_samples = theta_final_samples.sort_values(
        by="likelihood", ascending=False
    )
    random_theta_samples = theta_final_samples.sample(
        n=final_sample_size, random_state=42
    )

    # Generate plots for both best and random samples
    for sample_type, samples in [
        ("best", best_theta_samples),
        ("random", random_theta_samples),
    ]:
        simulation_data, results_dataframe = simulate_and_organize_parameter_sets(
            samples, circuit_fitter, parameters_to_fit
        )

        circuit_simulation_dict = {circuit_name: simulation_data}
        plot_circuit_simulations(
            circuit_simulation_dict,
            results_dataframe,
            plot_mode="individual",
            likelihood_percentile_range=20,
        )

        sample_count_actual = len(samples)
        plot_title = f"{'Top' if sample_type == 'best' else 'Random'} {sample_count_actual} Hierarchical Theta Fits for {circuit_name}"
        plt.suptitle(plot_title)
        plt.savefig(
            os.path.join(
                output_directory, f"hierarchical_{sample_type}_fits_{circuit_name}.png"
            )
        )
        plt.close()

    return best_theta_samples, random_theta_samples


def plot_hierarchical_fits(
    circuit_theta_results,
    output_directory=".",
    sample_count=60,
    time_bounds_max=None,
    time_bounds_min=None,
):
    """Plot hierarchical fits by reusing existing plotting functions from plots_simulation.py"""

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

    for circuit_name, theta_samples in circuit_theta_results.items():
        print(f"Processing hierarchical circuit {circuit_name}")

        # Create circuit configuration
        circuit_configuration, circuit_fitter = create_circuit_simulation_data(
            circuit_name,
            theta_samples,
            parameters_to_fit,
            circuit_manager,
            calibration_parameters,
            time_bounds_max,
            time_bounds_min,
        )

        # Generate individual plots and get samples
        best_theta_samples, random_theta_samples = (
            generate_hierarchical_individual_plots(
                circuit_name,
                theta_samples,
                circuit_fitter,
                parameters_to_fit,
                output_directory,
                sample_count,
            )
        )

        # Prepare data for combined plotting
        best_simulation_data, best_results_dataframe = (
            simulate_and_organize_parameter_sets(
                best_theta_samples, circuit_fitter, parameters_to_fit
            )
        )
        random_simulation_data, random_results_dataframe = (
            simulate_and_organize_parameter_sets(
                random_theta_samples, circuit_fitter, parameters_to_fit
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

    # Direct reuse of existing plotting functions with proper data structure
    combined_best_dataframe = pd.concat(combined_best_results, ignore_index=True)
    combined_random_dataframe = pd.concat(combined_random_results, ignore_index=True)

    # All plots below use existing functions from plots_simulation.py unchanged
    print(
        "Generating combined hierarchical plots using existing plot_circuit_simulations()..."
    )

    plot_circuit_simulations(
        combined_best_simulation_data,
        combined_best_dataframe,
        plot_mode="individual",
        likelihood_percentile_range=20,
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_hierarchical_best_fits.png"),
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
        os.path.join(
            output_directory, "all_circuits_hierarchical_best_fits_summary.png"
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    plot_circuit_simulations(
        combined_random_simulation_data,
        combined_random_dataframe,
        plot_mode="individual",
        likelihood_percentile_range=20,
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_hierarchical_random_fits.png"),
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
        os.path.join(
            output_directory, "all_circuits_hierarchical_random_fits_summary.png"
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    print(
        "Generating overlay plots using existing plot_circuit_conditions_overlay()..."
    )

    plot_circuit_conditions_overlay(
        combined_best_simulation_data,
        combined_best_dataframe,
        simulation_mode="individual",
    )
    plt.savefig(
        os.path.join(
            output_directory,
            "all_circuits_hierarchical_best_fits_overlay_individual.png",
        ),
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
        os.path.join(
            output_directory, "all_circuits_hierarchical_best_fits_overlay_summary.png"
        ),
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
            output_directory,
            "all_circuits_hierarchical_random_fits_overlay_individual.png",
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
        os.path.join(
            output_directory,
            "all_circuits_hierarchical_random_fits_overlay_summary.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def main():
    # Specify hierarchical results file
    project = "hierarchical_results_20250628_234739"
    hierarchical_results_file = "../../data/fit_data/hierarchical/" + project + ".csv"
    output_directory = "../../figures/hierarchical_simulations/" + project

    os.makedirs(output_directory, exist_ok=True)

    # Load model parameters to fit
    model_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    parameters_to_fit = model_priors[
        model_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()

    # Load hierarchical circuit results
    circuit_theta_results = load_hierarchical_circuit_results(
        hierarchical_results_file, parameters_to_fit, burn_in_fraction=0.4
    )

    # Plot hierarchical fits
    plot_hierarchical_fits(
        circuit_theta_results,
        output_directory,
        sample_count=30,
        time_bounds_max=130,
        time_bounds_min=30,
    )


if __name__ == "__main__":
    main()
