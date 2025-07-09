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


def extract_mixed_effects_circuit_parameters(
    mixed_effects_results,
    circuit_name,
    fitted_parameter_names,
    shared_parameter_names,
    hierarchical_parameter_names,
):
    """Extract circuit-specific mixed effects parameters (β + θ) from MCMC results"""

    reconstructed_circuit_parameters = pd.DataFrame()

    for param_name in fitted_parameter_names:
        if param_name in shared_parameter_names:
            beta_column = f"beta_{param_name}"
            if beta_column in mixed_effects_results.columns:
                reconstructed_circuit_parameters[param_name] = mixed_effects_results[
                    beta_column
                ]
        elif param_name in hierarchical_parameter_names:
            theta_column = f"theta_{circuit_name}_{param_name}"
            if theta_column in mixed_effects_results.columns:
                reconstructed_circuit_parameters[param_name] = mixed_effects_results[
                    theta_column
                ]

    # Add likelihood column for sorting compatibility
    if "likelihood" in mixed_effects_results.columns:
        reconstructed_circuit_parameters["likelihood"] = mixed_effects_results[
            "likelihood"
        ]
    else:
        reconstructed_circuit_parameters["likelihood"] = range(
            len(reconstructed_circuit_parameters)
        )

    print(
        f"Extracted {len(reconstructed_circuit_parameters)} mixed effects samples for {circuit_name}"
    )
    print(f"Available parameters: {list(reconstructed_circuit_parameters.columns)}")

    return reconstructed_circuit_parameters


def load_mixed_effects_circuit_results(
    mixed_effects_results_filepath, fitted_parameter_names, burn_in_fraction=0.4
):
    """Load mixed effects results and extract circuit-specific parameters"""
    # Load mixed effects results
    mixed_effects_raw_results = pd.read_csv(mixed_effects_results_filepath)

    # Handle column naming issue if present (walker/iteration swap)
    if (
        "walker" in mixed_effects_raw_results.columns
        and "iteration" in mixed_effects_raw_results.columns
    ):
        mixed_effects_raw_results = mixed_effects_raw_results.rename(
            columns={"walker": "iteration", "iteration": "walker"}
        )

    print(f"Loaded mixed effects results: {len(mixed_effects_raw_results)} raw samples")

    # Apply burn-in filtering
    mixed_effects_processed = process_mcmc_data(
        mixed_effects_raw_results, burn_in=burn_in_fraction, chain_idx=0
    )
    mixed_effects_filtered_results = mixed_effects_processed["processed_data"]

    print(
        f"After burn-in ({burn_in_fraction * 100:.0f}%): {len(mixed_effects_filtered_results)} samples"
    )

    # Infer mixed effects parameter structure from column names
    beta_parameter_columns = [
        col for col in mixed_effects_filtered_results.columns if col.startswith("beta_")
    ]
    actual_shared_parameter_names = [
        col.replace("beta_", "") for col in beta_parameter_columns
    ]

    alpha_parameter_columns = [
        col
        for col in mixed_effects_filtered_results.columns
        if col.startswith("alpha_")
    ]
    actual_hierarchical_parameter_names = [
        col.replace("alpha_", "") for col in alpha_parameter_columns
    ]

    print(f"Detected shared parameters (β): {actual_shared_parameter_names}")
    print(
        f"Detected hierarchical parameters (θ): {actual_hierarchical_parameter_names}"
    )

    # Extract circuit names from theta columns
    theta_column_patterns = [
        col
        for col in mixed_effects_filtered_results.columns
        if col.startswith("theta_")
    ]
    print(f"DEBUG: Sample theta columns: {theta_column_patterns[:10]}")

    # Parse circuit names by finding the parameter part and working backwards
    circuit_names_with_params = []
    known_parameters = set(fitted_parameter_names)

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
    print(f"Found circuits in mixed effects results: {circuit_names}")
    print(f"DEBUG: Circuit-parameter pairs: {circuit_names_with_params[:5]}")

    # Extract circuit-specific mixed effects parameters
    circuit_mixed_effects_results = {}
    for circuit_name in circuit_names:
        circuit_parameters = extract_mixed_effects_circuit_parameters(
            mixed_effects_filtered_results,
            circuit_name,
            fitted_parameter_names,
            actual_shared_parameter_names,
            actual_hierarchical_parameter_names,
        )
        if not circuit_parameters.empty:
            circuit_mixed_effects_results[circuit_name] = circuit_parameters

    return circuit_mixed_effects_results


def generate_mixed_effects_individual_plots(
    circuit_name,
    mixed_effects_samples,
    circuit_fitter,
    fitted_parameter_names,
    output_directory,
    sample_count,
):
    """Generate individual circuit plots using existing plot_circuit_simulations() directly"""
    final_sample_size = min(sample_count, len(mixed_effects_samples))
    mixed_effects_final_samples = (
        mixed_effects_samples.sample(n=final_sample_size, random_state=42)
        if len(mixed_effects_samples) > final_sample_size
        else mixed_effects_samples.copy()
    )

    best_mixed_effects_samples = mixed_effects_final_samples.sort_values(
        by="likelihood", ascending=False
    )
    random_mixed_effects_samples = mixed_effects_final_samples.sample(
        n=final_sample_size, random_state=42
    )

    # Generate plots for both best and random samples
    for sample_type, samples in [
        ("best", best_mixed_effects_samples),
        ("random", random_mixed_effects_samples),
    ]:
        simulation_data, results_dataframe = simulate_and_organize_parameter_sets(
            samples, circuit_fitter, fitted_parameter_names
        )

        circuit_simulation_dict = {circuit_name: simulation_data}
        plot_circuit_simulations(
            circuit_simulation_dict,
            results_dataframe,
            plot_mode="individual",
            likelihood_percentile_range=20,
        )

        sample_count_actual = len(samples)
        plot_title = f"{'Top' if sample_type == 'best' else 'Random'} {sample_count_actual} Mixed Effects Fits for {circuit_name}"
        plt.suptitle(plot_title)
        plt.savefig(
            os.path.join(
                output_directory, f"mixed_effects_{sample_type}_fits_{circuit_name}.png"
            )
        )
        plt.close()

    return best_mixed_effects_samples, random_mixed_effects_samples


def plot_mixed_effects_fits(
    circuit_mixed_effects_results,
    output_directory=".",
    sample_count=60,
    fitted_parameter_names=None,
    time_bounds_max=None,
    time_bounds_min=None,
):
    """Plot mixed effects fits by reusing existing plotting functions from plots_simulation.py"""

    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    _ = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    calibration_parameters = setup_calibration()

    combined_best_simulation_data = {}
    combined_random_simulation_data = {}
    combined_best_results = []
    combined_random_results = []

    for circuit_name, mixed_effects_samples in circuit_mixed_effects_results.items():
        print(f"Processing mixed effects circuit {circuit_name}")

        # Create circuit configuration
        circuit_configuration, circuit_fitter = create_circuit_simulation_data(
            circuit_name,
            mixed_effects_samples,
            fitted_parameter_names,
            circuit_manager,
            calibration_parameters,
            time_bounds_max,
            time_bounds_min,
        )

        # Generate individual plots and get samples
        best_mixed_effects_samples, random_mixed_effects_samples = (
            generate_mixed_effects_individual_plots(
                circuit_name,
                mixed_effects_samples,
                circuit_fitter,
                fitted_parameter_names,
                output_directory,
                sample_count,
            )
        )

        # Prepare data for combined plotting
        best_simulation_data, best_results_dataframe = (
            simulate_and_organize_parameter_sets(
                best_mixed_effects_samples, circuit_fitter, fitted_parameter_names
            )
        )
        random_simulation_data, random_results_dataframe = (
            simulate_and_organize_parameter_sets(
                random_mixed_effects_samples, circuit_fitter, fitted_parameter_names
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
        "Generating combined mixed effects plots using existing plot_circuit_simulations()..."
    )

    plot_circuit_simulations(
        combined_best_simulation_data,
        combined_best_dataframe,
        plot_mode="individual",
        likelihood_percentile_range=20,
    )
    plt.savefig(
        os.path.join(output_directory, "all_circuits_mixed_effects_best_fits.png"),
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
            output_directory, "all_circuits_mixed_effects_best_fits_summary.png"
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
        os.path.join(output_directory, "all_circuits_mixed_effects_random_fits.png"),
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
            output_directory, "all_circuits_mixed_effects_random_fits_summary.png"
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
            "all_circuits_mixed_effects_best_fits_overlay_individual.png",
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
            output_directory, "all_circuits_mixed_effects_best_fits_overlay_summary.png"
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
            "all_circuits_mixed_effects_random_fits_overlay_individual.png",
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
            "all_circuits_mixed_effects_random_fits_overlay_summary.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def main():
    """Execute mixed effects best parameter simulation"""
    mixed_effects_data_id = "hierarchical_results_20250703_063716"

    # Configuration
    mixed_effects_results_file = (
        "../../data/fit_data/mixed_effect/" + mixed_effects_data_id + ".csv"
    )
    output_directory = "../../figures/mixed_effects/" + mixed_effects_data_id

    os.makedirs(output_directory, exist_ok=True)

    # Load model parameters to fit
    model_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    fitted_parameter_names = model_priors[
        model_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()
    # remove k_rna_deg and k_rna_km
    fitted_parameter_names = [
        p for p in fitted_parameter_names if p not in ["k_rna_km", "k_rna_deg"]
    ]

    # Load mixed effects circuit results
    circuit_mixed_effects_results = load_mixed_effects_circuit_results(
        mixed_effects_results_file, fitted_parameter_names, burn_in_fraction=0.4
    )

    # Plot mixed effects fits
    plot_mixed_effects_fits(
        circuit_mixed_effects_results,
        output_directory,
        fitted_parameter_names=fitted_parameter_names,
        sample_count=100,
        time_bounds_max=130,
        time_bounds_min=30,
    )


if __name__ == "__main__":
    main()
