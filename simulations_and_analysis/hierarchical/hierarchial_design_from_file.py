import pandas as pd
import os
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.circuit_utils import create_circuit_configs, setup_calibration
from likelihood_functions.hierarchical_likelihood.base_hierarchical import (
    HierarchicalCircuitFitter,
)


def setup_hierarchical_model(circuits_to_fit):
    """Setup the hierarchical model components"""
    model_parameters_priors_file = "../../data/prior/model_parameters_priors.csv"

    circuit_manager = CircuitManager(
        parameters_file=model_parameters_priors_file,
        json_file="../../data/circuits/circuits.json",
    )

    available_circuits = circuit_manager.list_circuits()
    circuits_to_fit = [c for c in circuits_to_fit if c in available_circuits]

    circuit_configs = create_circuit_configs(
        circuit_manager, circuits_to_fit, min_time=30, max_time=210
    )

    priors = pd.read_csv(model_parameters_priors_file)
    priors = priors[priors["Parameter"] != "k_prot_deg"]
    parameters_to_fit = priors.Parameter.tolist()

    calibration_params = setup_calibration()

    hierarchical_fitter = HierarchicalCircuitFitter(
        circuit_configs, parameters_to_fit, priors, calibration_params
    )

    return hierarchical_fitter, parameters_to_fit


def analyze_hierarchical_results(
    results_file, output_folder, burn_in=0.3, n_simulation_samples=100
):
    """Complete analysis workflow"""

    os.makedirs(output_folder, exist_ok=True)

    # Load and setup
    print("Loading results and setting up model...")
    df = pd.read_csv(results_file)

    # misnaming, walker is the iteration column and iteration is the walker column. swap the col names
    df = df.rename(columns={"walker": "iteration", "iteration": "walker"})

    circuits_to_fit = [
        "trigger_antitrigger",
        "toehold_trigger",
        "sense_star_6",
        "cascade",
        "cffl_type_1",
        "star_antistar_1",
    ]
    hierarchical_fitter, parameters_to_fit = setup_hierarchical_model(circuits_to_fit)

    # STEP 1: MCMC Processing & Diagnostics
    print("=" * 50)
    print("STEP 1: MCMC PROCESSING & DIAGNOSTICS")
    print("=" * 50)

    from analysis_and_figures.mcmc_analysis_hierarchical import (
        process_mcmc_data,
        calculate_mcmc_diagnostics_from_dataframe,
    )

    # Process the data (filter + metadata) - DO THIS ONCE
    processed = process_mcmc_data(df, burn_in=burn_in, chain_idx=0)
    processed_df = processed["processed_data"]  # This is the filtered data

    print(f"Raw samples: {processed['metadata']['n_samples_raw']}")
    print(f"Processed samples: {processed['metadata']['n_samples_processed']}")
    print(f"Processed samples: {processed['metadata']['n_samples_processed']}")
    print(f"Burn-in: {burn_in * 100:.0f}%")

    # Calculate convergence diagnostics
    _ = calculate_mcmc_diagnostics_from_dataframe(
        df, parameters_to_fit, [config.name for config in hierarchical_fitter.configs]
    )

    # STEP 2: Parameter Distributions
    print("=" * 50)
    print("STEP 2: PARAMETER DISTRIBUTIONS")
    print("=" * 50)

    # FIX: Create results dict with PROCESSED data instead of raw data
    results = {
        "dataframe": processed_df,
        "param_names": parameters_to_fit,
        "circuit_names": [config.name for config in hierarchical_fitter.configs],
    }

    # plot_hierarchical_results(results, hierarchical_fitter)

    # Move plots
    import glob

    for png_file in glob.glob("hierarchical_*.png"):
        new_path = os.path.join(output_folder, png_file)
        os.rename(png_file, new_path)

    # STEP 3: Parameter Correlations (use processed data)
    print("=" * 50)
    print("STEP 3: PARAMETER CORRELATIONS")
    print("=" * 50)

    from analysis_and_figures.hierarchical_pairplot_analysis import (
        prepare_hierarchical_pairplot_data_processed,
        create_hierarchical_pairplot,
        create_hierarchical_histogram_grid,
    )

    # In your analyze_hierarchical_results function:
    pairplot_df = prepare_hierarchical_pairplot_data_processed(
        processed_df, parameters_to_fit, results["circuit_names"], n_samples=1000
    )

    # Create both figures
    create_hierarchical_histogram_grid(pairplot_df, parameters_to_fit, output_folder)
    create_hierarchical_pairplot(pairplot_df, parameters_to_fit, output_folder)

    # STEP 4: Simulations (use processed data)
    print("=" * 50)
    print("STEP 4: POSTERIOR SIMULATIONS")
    print("=" * 50)

    from analysis_and_figures.visualization_hierarchical import (
        simulate_hierarchical_posterior_theta_random,
        simulate_hierarchical_posterior_theta_best,
        plot_hierarchical_posterior_theta_only,
    )

    # Pass the already processed data - no more burn-in filtering
    posterior_results = simulate_hierarchical_posterior_theta_random(
        hierarchical_fitter,
        processed_df,
        parameters_to_fit,
        results["circuit_names"],
        n_samples=n_simulation_samples,
    )

    fig = plot_hierarchical_posterior_theta_only(hierarchical_fitter, posterior_results)
    fig.savefig(os.path.join(output_folder, "hierarchical_posterior_simulations.png"))
    import matplotlib.pyplot as plt

    plt.close()

    # best
    best_results = simulate_hierarchical_posterior_theta_best(
        hierarchical_fitter,
        processed_df,
        parameters_to_fit,
        results["circuit_names"],
        n_samples=n_simulation_samples,
    )

    fig = plot_hierarchical_posterior_theta_only(hierarchical_fitter, best_results)
    fig.savefig(os.path.join(output_folder, "hierarchical_n_best_simulations.png"))
    import matplotlib.pyplot as plt

    plt.close()

    print(f"Analysis complete! Results saved to: {output_folder}")
    return results, hierarchical_fitter


def main():
    """Main function - set your parameters here"""

    # SET YOUR PARAMETERS HERE
    results_file = (
        "../../data/fit_data/hierarchical/hierarchical_results_20250521_082008.csv"
    )
    results_file = (
        "../../data/fit_data/hierarchical/hierarchical_results_20250606_214415.csv"
    )
    results_file = (
        "../../data/fit_data/hierarchical/hierarchical_results_20250608_110254.csv"
    )
    results_file = (
        "../../data/fit_data/hierarchical/hierarchical_results_20250628_234739.csv"
    )
    output_folder = None  # Will auto-generate based on timestamp
    burn_in = 0.5
    n_simulation_samples = 60

    # Auto-generate output folder if not provided
    if output_folder is None:
        timestamp = (
            results_file.split("_")[-2]
            + "_"
            + results_file.split("_")[-1].replace(".csv", "")
        )
        output_folder = f"../../figures/hierarchical_results_{timestamp}"

    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found!")
        return

    # Run analysis
    analyze_hierarchical_results(
        results_file=results_file,
        output_folder=output_folder,
        burn_in=burn_in,
        n_simulation_samples=n_simulation_samples,
    )


if __name__ == "__main__":
    main()
