import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from circuits.circuit_generation.circuit_manager import CircuitManager
from utils.GFP_calibration import setup_calibration
from simulations_and_analysis.individual.individual_circuits_statistics import (
    load_individual_circuit_results,
)
from simulations_and_analysis.individual.individual_circuits_simulations import (
    create_circuit_simulation_data,
)
from importance_resampling.importance_resampling import (
    perform_importance_resampling,
    build_parameter_dependency_graph,
)
from analysis_and_figures.imortance_resampling import (
    plot_resampling_diagnostics,
    summary_visualizations_importance_resampling,
)
from analysis_and_figures.plots_simulation import (
    plot_circuit_simulations,
    plot_circuit_conditions_overlay,
)


def run_importance_resampling_analysis(
    mcmc_results_by_circuit: dict,
    output_directory: str = ".",
    n_initial_samples: int = 1000,
    n_resample: int = 200,
    time_bounds_max: float = None,
    time_bounds_min: float = None,
    priors_csv_path: str = "../../data/prior/model_parameters_priors.csv",
):
    """
    Run complete importance resampling analysis across all compatible circuit pairs.

    Parameters:
    -----------
    mcmc_results_by_circuit : dict
            Dictionary mapping circuit names to MCMC DataFrames
    output_directory : str
            Directory to save results
    n_initial_samples : int
            Number of initial samples to use
    n_resample : int
            Number of samples after resampling
    time_bounds_max : float
            Maximum time for simulations
    time_bounds_min : float
            Minimum time for simulations
    priors_csv_path : str
            Path to priors CSV file

    Returns:
    --------
    all_results : dict
            Dictionary of all resampling results
    ess_matrix : pd.DataFrame
            Matrix of ESS values for each source-target pair
    """
    os.makedirs(output_directory, exist_ok=True)

    circuit_manager = CircuitManager(
        parameters_file=priors_csv_path,
        json_file="../../data/circuits/circuits.json",
    )
    calibration_parameters = setup_calibration()

    # Step 1: Create circuit fitters and extract parameters
    print("=" * 60)
    print("Step 1: Creating circuit fitters")
    print("=" * 60)

    circuit_fitters = {}
    circuit_configurations = {}
    circuit_parameters = {}

    for circuit_name in mcmc_results_by_circuit.keys():
        print(f"Creating fitter for: {circuit_name}")

        circuit_configuration, circuit_fitter = create_circuit_simulation_data(
            circuit_name,
            circuit_manager,
            calibration_parameters,
            time_bounds_max,
            time_bounds_min,
            priors_csv_path,
        )

        circuit_fitters[circuit_name] = circuit_fitter
        circuit_configurations[circuit_name] = circuit_configuration
        circuit_parameters[circuit_name] = set(circuit_fitter.parameters_to_fit)

    # Step 2: Build dependency graph
    print("\n" + "=" * 60)
    print("Step 2: Building parameter dependency graph")
    print("=" * 60)

    G, edge_info = build_parameter_dependency_graph(circuit_parameters)

    # Step 3: Perform importance resampling for all valid pairs
    print("\n" + "=" * 60)
    print("Step 3: Performing importance resampling")
    print("=" * 60)

    circuit_names = list(mcmc_results_by_circuit.keys())

    # Initialize results storage
    all_results = {}
    ess_matrix = pd.DataFrame(
        index=circuit_names,
        columns=circuit_names,
        dtype=float,
    )
    ess_matrix[:] = np.nan

    improvement_matrix = pd.DataFrame(
        index=circuit_names,
        columns=circuit_names,
        dtype=float,
    )
    improvement_matrix[:] = np.nan

    for source_circuit in circuit_names:
        # Get valid targets based on DAG
        valid_targets = [source_circuit]  # Always test self
        valid_targets.extend([t for t in G.successors(source_circuit)])

        print(f"\nSource: {source_circuit}")
        print(f"  Valid targets: {valid_targets}")

        for target_circuit in valid_targets:
            if target_circuit not in circuit_fitters:
                continue

            print(f"  → Target: {target_circuit}")

            # Perform importance resampling
            results = perform_importance_resampling(
                source_circuit=source_circuit,
                target_circuit=target_circuit,
                source_mcmc_samples=mcmc_results_by_circuit[source_circuit],
                source_circuit_fitter=circuit_fitters[source_circuit],
                target_circuit_fitter=circuit_fitters[target_circuit],
                n_initial_samples=n_initial_samples,
                n_resample=n_resample,
            )

            # Store results
            key = (source_circuit, target_circuit)
            all_results[key] = results

            # Update matrices
            ess_matrix.at[source_circuit, target_circuit] = results[
                "effective_sample_size"
            ]
            improvement = (
                results["mean_log_posterior_after"]
                - results["mean_log_posterior_before"]
            )
            improvement_matrix.at[source_circuit, target_circuit] = improvement

            print(
                f"    ESS: {results['effective_sample_size']:.1f} "
                f"({results['ess_ratio'] * 100:.1f}%)"
            )
            print(f"    Improvement: {improvement:.2f}")

            # Plot diagnostics
            plot_path = os.path.join(
                output_directory, f"resampling_{source_circuit}_to_{target_circuit}.png"
            )
            plot_resampling_diagnostics(results, plot_path)

            # plot_circuit_simulations()

            simulation_data_dict = results["simulation_results_target"]
            plot_circuit_simulations(
                # target_circuit,
                simulation_data_dict,
                results["resampled_samples"],
                plot_mode="individual",
                likelihood_percentile_range=20,
            )
            # plt.show()

            plot_circuit_conditions_overlay(
                simulation_data_dict,
                results["resampled_samples"],
                simulation_mode="summary",
                show_title=False,
            )
            plt.savefig(
                os.path.join(
                    output_directory,
                    f"conditions_overlay_{source_circuit}_to_{target_circuit}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )

    # Step 4: Create summary visualizations
    summary_visualizations_importance_resampling(
        ess_matrix=ess_matrix,
        improvement_matrix=improvement_matrix,
        n_resample=n_resample,
        all_results=all_results,
        output_directory=output_directory,
        n_initial_samples=n_initial_samples,
    )

    return all_results, ess_matrix, improvement_matrix


def main():
    """Main function to run importance resampling analysis."""

    # Configuration
    subfolder = "/2025-11-11_100000_steps_adaptive_new_priors_part_1"
    # subfolder = "/202511_new_prior"

    input_directory = "../../data/fit_data/individual_circuits" + subfolder
    output_directory = "../../figures/importance_resampling" + subfolder
    priors_csv_path = "../../data/prior/model_parameters_priors_092025_correction.csv"

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Load MCMC results
    print("Loading MCMC results...")
    mcmc_results = load_individual_circuit_results(input_directory, prefix="results_")

    # Define circuits to analyze
    circuit_processing_sequence = [
        "constitutive sfGFP",
        "constitutive sfGFP sim",
        "sense_star_6",
        "toehold_trigger",
        "cascade",
        "to1_cascadecffl_type_1",
        "or_gate_c1ffl",
        "star_antistar_1",
        "trigger_antitrigger",
        "inhibited_incoherent_cascade",
        "inhibited_cascade",
        "cffl_12",
        "iffl_1",
    ]

    # Filter to available circuits
    filtered_mcmc_results = {
        circuit_name: mcmc_results[circuit_name]
        for circuit_name in circuit_processing_sequence
        if circuit_name in mcmc_results
    }

    print(f"Circuits to analyze: {list(filtered_mcmc_results.keys())}")

    # Run importance resampling analysis
    all_results, ess_matrix, improvement_matrix = run_importance_resampling_analysis(
        filtered_mcmc_results,
        output_directory,
        n_initial_samples=1000,
        n_resample=200,
        time_bounds_max=210,
        time_bounds_min=30,
        priors_csv_path=priors_csv_path,
    )

    print("\nAnalysis complete!")
    print(f"Total pairs analyzed: {len(all_results)}")

    # Print summary
    print("\nTop 5 pairs by ESS ratio:")
    ess_ratios = []
    for key, results in all_results.items():
        ess_ratios.append((key, results["ess_ratio"]))

    for key, ratio in sorted(ess_ratios, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {key[0]} → {key[1]}: {ratio * 100:.1f}%")


if __name__ == "__main__":
    main()
