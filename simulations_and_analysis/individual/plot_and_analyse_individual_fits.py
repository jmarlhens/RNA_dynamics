import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from utils.process_experimental_data import organize_results
from analysis_and_figures.simulation import plot_all_simulation_results
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor
from circuits.circuit_generation.circuit_manager import CircuitManager
from data.circuits.circuit_configs import get_circuit_conditions, get_data_file
from analysis_and_figures.mcmc_analysis_hierarchical import process_mcmc_data
from analysis_and_figures.simulation import plot_simulation_statistical_summaries


def setup_calibration():
    """Set up GFP calibration parameters"""
    data = pd.read_csv("../../utils/calibration_gfp/gfp_Calibration.csv")
    calibration_results = fit_gfp_calibration(
        data,
        concentration_col="GFP Concentration (nM)",
        fluorescence_pattern="F.I. (a.u)",
    )
    correction_factor, _ = get_brightness_correction_factor("avGFP", "sfGFP")

    return {
        "slope": calibration_results["slope"],
        "intercept": calibration_results["intercept"],
        "brightness_correction": correction_factor,
    }


def load_circuit_results(results_dir="../data/fit_data/individual_circuits"):
    """Load all circuit results from CSV files"""
    results = {}
    # Find all results CSV files
    pattern = os.path.join(results_dir, "results_*.csv")
    result_files = glob.glob(pattern)

    for file_path in result_files:
        # Extract circuit name from filename
        filename = os.path.basename(file_path)

        circuit_name = filename.split("_")[1:-2]
        circuit_name = "_".join(circuit_name)

        # Load results
        df = pd.read_csv(file_path)
        results[circuit_name] = df

        print(f"Loaded {circuit_name} results from {filename}")
        print(f"Number of samples: {len(df)}")
        print(f"Best likelihood: {df['likelihood'].max():.2f}\n")

    return results


def plot_parameter_distributions(results, save_dir="."):
    """Plot parameter distributions for all circuits"""
    for circuit_name, df in results.items():
        # Get parameter columns (exclude likelihood and other non-parameter columns)
        param_cols = [col for col in df.columns if col != "likelihood"]

        # Create parameter distribution plots
        n_params = len(param_cols)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        plt.figure(figsize=(15, 5 * n_rows))

        for i, param in enumerate(param_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data=df, x=param, bins=30)
            plt.title(f"{circuit_name} - {param}")
            plt.xlabel("Parameter Value")
            plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"parameter_distributions_{circuit_name}.png")
        )
        plt.close()


def plot_likelihood_distributions(results, save_dir="."):
    """Plot likelihood distributions for all circuits"""
    plt.figure(figsize=(12, 6))

    for circuit_name, df in results.items():
        sns.kdeplot(data=df["likelihood"], label=circuit_name)

    plt.title("Likelihood Distributions Across Circuits")
    plt.xlabel("Log Likelihood")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "likelihood_distributions.png"))
    plt.close()


def process_and_plot_params(
    df_params,
    params_type,
    circuit_name,
    circuit_fitter,
    parameters_to_fit,
    save_dir,
    max_time=None,
    min_time=None,
):
    """
    Process a set of parameters, simulate circuit behavior, and plot results

    Args:
        df_params: DataFrame containing parameter samples
        params_type: String describing parameter type ("best" or "random")
        circuit_name: Name of the circuit
        circuit_fitter: CircuitFitter instance
        parameters_to_fit: List of parameter names
        save_dir: Directory to save output plots
        max_time: Optional maximum time for simulation
        min_time: Optional minimum time for simulation
    """
    # Extract parameter values
    log_params = df_params[parameters_to_fit].values

    # Simulate with params
    sim_data = circuit_fitter.simulate_parameters(log_params)
    log_likelihood = circuit_fitter.calculate_likelihood_from_simulation(sim_data)
    log_prior = circuit_fitter.calculate_log_prior(log_params)

    # Convert to linear for plotting
    linear_params = 10**log_params
    results_df = organize_results(
        parameters_to_fit, linear_params, log_likelihood, log_prior
    )

    # Plot fits
    plt.figure(figsize=(12, 8))
    plot_all_simulation_results(sim_data, results_df, ll_quartile=20)

    n_samples = len(df_params)
    if params_type == "best":
        title = f"Top {n_samples} Best Fits for {circuit_name}"
    else:
        title = f"{n_samples} Random Samples Fits for {circuit_name}"

    if min_time:
        title += f" (t₀={min_time}min)"

    if max_time:
        title += f" (t₁={max_time}min)"

    plt.suptitle(title)
    plt.savefig(os.path.join(save_dir, f"{params_type}_fits_{circuit_name}.png"))
    plt.close()


def plot_fits(
    mcmc_results_by_circuit, save_dir=".", n_samples=400, max_time=None, min_time=None
):
    """Plot fits for each circuit using both best and random samples

    Generates:
    - Individual plots per circuit (existing functionality)
    - Combined multi-circuit figures (new functionality)
    """
    # Initialize CircuitManager and load requirements
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )
    priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]
    parameters_to_fit = priors.Parameter.tolist()

    # Storage for combined plotting
    combined_simulation_data_best = {}
    combined_simulation_data_random = {}
    combined_results_dataframes_best = []
    combined_results_dataframes_random = []

    for circuit_idx, (circuit_name, mcmc_raw_samples) in enumerate(
        mcmc_results_by_circuit.items()
    ):
        print(f"Processing circuit {circuit_name}")

        # Get circuit data and setup
        circuit_conditions = get_circuit_conditions(circuit_name)
        experimental_data_file = get_data_file(circuit_name)
        experimental_data, time_span = load_and_process_csv(experimental_data_file)

        # Create circuit and configuration
        first_condition = list(circuit_conditions.keys())[0]
        circuit_instance = circuit_manager.create_circuit(
            circuit_name, parameters=circuit_conditions[first_condition]
        )
        circuit_config = CircuitConfig(
            model=circuit_instance.model,
            name=circuit_name,
            condition_params=circuit_conditions,
            experimental_data=experimental_data,
            tspan=time_span,
            max_time=max_time,
            min_time=min_time,
        )

        # Create fitter
        calibration_parameters = setup_calibration()
        circuit_fitter = CircuitFitter(
            [circuit_config], parameters_to_fit, priors, calibration_parameters
        )

        # Apply burn-in filtering (same as individual_circuits_analysis.py)
        mcmc_processed_result = process_mcmc_data(
            mcmc_raw_samples, burn_in=0.4, chain_idx=0
        )
        mcmc_filtered_samples = mcmc_processed_result["processed_data"]

        print(
            f"{circuit_name}: {len(mcmc_raw_samples)} → {len(mcmc_filtered_samples)} samples after burn-in"
        )

        # Sample from filtered data
        mcmc_final_samples = (
            mcmc_filtered_samples.sample(n=n_samples, random_state=42)
            if len(mcmc_filtered_samples) > n_samples
            else mcmc_filtered_samples.copy()
        )

        # Best likelihood from filtered samples
        best_likelihood_samples = mcmc_final_samples.sort_values(
            by="likelihood", ascending=False
        )

        # Random samples from filtered pool
        random_filtered_samples = mcmc_final_samples.sample(
            n=min(n_samples, len(mcmc_final_samples)), random_state=42
        )

        # INDIVIDUAL PLOTS
        process_and_plot_params(
            best_likelihood_samples,
            "best",
            circuit_name,
            circuit_fitter,
            parameters_to_fit,
            save_dir,
            max_time=max_time,
            min_time=min_time,
        )

        process_and_plot_params(
            random_filtered_samples,
            "random",
            circuit_name,
            circuit_fitter,
            parameters_to_fit,
            save_dir,
            max_time=max_time,
            min_time=min_time,
        )

        # COMBINED PLOTS DATA COLLECTION (new functionality)
        # Extract parameter values for best/random samples
        best_log_parameters = best_likelihood_samples[parameters_to_fit].values
        random_log_parameters = random_filtered_samples[parameters_to_fit].values

        # Simulate both parameter sets
        best_simulation_results = circuit_fitter.simulate_parameters(
            best_log_parameters
        )
        random_simulation_results = circuit_fitter.simulate_parameters(
            random_log_parameters
        )

        # DEBUG: Complete structure inspection
        # Extract single circuit data (CircuitFitter with [single_config] returns {0: circuit_data})
        best_single_circuit_data = best_simulation_results[0]
        # random_single_circuit_data = random_simulation_results[0]

        # DEBUG: Verify extracted data structure
        print(
            f"DEBUG {circuit_name} - best_single_circuit_data keys: {list(best_single_circuit_data.keys())}"
        )
        print(
            f"DEBUG {circuit_name} - best_single_circuit_data['combined_params'] type: {type(best_single_circuit_data['combined_params'])}"
        )

        # Store for combined plotting with proper multi-circuit indexing
        combined_simulation_data_best[circuit_idx] = {
            "config": circuit_config,
            "combined_params": best_single_circuit_data["combined_params"],
            "simulation_results": best_single_circuit_data["simulation_results"],
        }

        # DEBUG: Verify stored structure
        print(
            f"DEBUG {circuit_name} - stored structure keys: {list(combined_simulation_data_best[circuit_idx].keys())}"
        )

        # Calculate likelihoods for combined plotting
        best_log_likelihoods = circuit_fitter.calculate_likelihood_from_simulation(
            best_simulation_results
        )
        random_log_likelihoods = circuit_fitter.calculate_likelihood_from_simulation(
            random_simulation_results
        )

        best_log_priors = circuit_fitter.calculate_log_prior(best_log_parameters)
        random_log_priors = circuit_fitter.calculate_log_prior(random_log_parameters)

        # Organize results for combined plotting
        best_results_dataframe = organize_results(
            parameters_to_fit,
            10**best_log_parameters,
            best_log_likelihoods,
            best_log_priors,
        )
        random_results_dataframe = organize_results(
            parameters_to_fit,
            10**random_log_parameters,
            random_log_likelihoods,
            random_log_priors,
        )

        # FIXED: Extract and store with explicit variable isolation
        best_circuit_data = best_simulation_results[0]
        random_circuit_data = random_simulation_results[0]

        # Store with explicit dictionary construction
        combined_simulation_data_best[circuit_idx] = {
            "config": circuit_config,
            "combined_params": best_circuit_data["combined_params"].copy(),
            "simulation_results": best_circuit_data["simulation_results"],
        }
        combined_simulation_data_random[circuit_idx] = {
            "config": circuit_config,
            "combined_params": random_circuit_data["combined_params"].copy(),
            "simulation_results": random_circuit_data["simulation_results"],
        }

        combined_results_dataframes_best.append(best_results_dataframe)
        combined_results_dataframes_random.append(random_results_dataframe)

    # GENERATE COMBINED FIGURES (new functionality)
    combined_best_results = pd.concat(
        combined_results_dataframes_best, ignore_index=True
    )
    combined_random_results = pd.concat(
        combined_results_dataframes_random, ignore_index=True
    )

    # DEBUG: Verify final combined structure
    print(
        f"DEBUG - combined_simulation_data_best keys: {list(combined_simulation_data_best.keys())}"
    )
    for idx in combined_simulation_data_best.keys():
        print(
            f"DEBUG - circuit {idx} keys: {list(combined_simulation_data_best[idx].keys())}"
        )
        if "config" in combined_simulation_data_best[idx]:
            print(
                f"DEBUG - circuit {idx} config.name: {combined_simulation_data_best[idx]['config'].name}"
            )

    print("Generating combined best fits figure...")
    plot_all_simulation_results(
        combined_simulation_data_best, combined_best_results, ll_quartile=20
    )

    # save the figure here
    plt.savefig(
        os.path.join(save_dir, "all_circuits_best_fits.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Replace individual trajectory plots with statistical summaries
    plot_simulation_statistical_summaries(
        combined_simulation_data_best,
        combined_best_results,
        save_path=os.path.join(save_dir, "all_circuits_summary_ribbons.png"),
        max_time_cutoff=130,
        min_time_cutoff=30,
        summary_type="median_iqr",  # or 'mean_std'
    )

    print("Generating combined random fits figure...")
    plot_all_simulation_results(
        combined_simulation_data_random, combined_random_results, ll_quartile=20
    )

    # save the figure here
    plt.savefig(
        os.path.join(save_dir, "all_circuits_random_fits.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def main():
    """Main function to load and plot all results"""
    """Execute individual circuits hierarchical comparison analysis"""
    subfolder = "/10000_steps_updated"
    # Configuration
    individual_results_directory = "../../data/fit_data/individual_circuits" + subfolder
    # prior_parameters_filepath = "../../data/prior/model_parameters_priors_updated.csv"
    output_visualization_directory = (
        "../../figures/individual_hierarchical_comparison" + subfolder
    )

    # Load all results
    results = load_circuit_results(individual_results_directory)

    # Generate plots
    # plot_parameter_distributions(results, output_visualization_directory)
    # plot_likelihood_distributions(results, output_visualization_directory)
    plot_fits(
        results, output_visualization_directory, n_samples=30, max_time=130, min_time=30
    )


if __name__ == "__main__":
    main()
