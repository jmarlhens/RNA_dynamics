from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from likelihood_functions.base import CircuitFitter
from utils.process_experimental_data import organize_results
from analysis_and_figures.plots_simulation import plot_all_simulation_results
from likelihood_functions.base import MCMCAdapter
from analysis_and_figures.mcmc_analysis import analyze_mcmc_results
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.circuit_utils import create_circuit_configs, setup_calibration


def fit_multiple_circuits(
    circuit_configs,
    parameters_to_fit,
    priors,
    calibration_params,
    n_samples=2000,
    n_walkers=10,
    n_chains=6,
    n_sets=60,
):
    """Fit multiple circuits simultaneously with shared parameters"""
    # Create circuit fitter with all configs
    circuit_fitter = CircuitFitter(
        circuit_configs, parameters_to_fit, priors, calibration_params
    )

    # Generate test parameters (in log space)
    log_params = circuit_fitter.generate_test_parameters(n_sets=n_sets)

    # Run simulations (takes log params)
    import time

    tic = time.time()
    sim_data = circuit_fitter.simulate_parameters(log_params)
    toc = time.time()
    simulation_time_process = toc - tic
    print(f"Simulation time: {simulation_time_process:.2f} seconds")

    # Calculate likelihood from simulation data
    log_likelihood = circuit_fitter.calculate_likelihood_from_simulation(sim_data)

    # Calculate prior (takes log params)
    log_prior = circuit_fitter.calculate_log_prior(log_params)

    # # Calculate posterior (takes log params)
    # log_posterior = log_prior + log_likelihood["total"]

    # Organize results
    results_df = organize_results(
        parameters_to_fit, log_params, log_likelihood, log_prior
    )

    # Plot initial simulation results
    plot_all_simulation_results(sim_data, results_df, ll_quartile=20)
    plt.show()

    # Create MCMC adapter
    adapter = MCMCAdapter(circuit_fitter)
    initial_parameters = adapter.get_initial_parameters()

    # Setup and run parallel tempering
    pt = adapter.setup_parallel_tempering(n_walkers=n_walkers, n_chains=n_chains)
    parameters, priors_out, likelihoods, step_accepts, swap_accepts = pt.run(
        initial_parameters=initial_parameters,
        n_samples=n_samples,
        target_acceptance_ratio=0.4,
        adaptive_temperature=True,
    )

    # Analyze results
    results = analyze_mcmc_results(
        parameters=parameters,
        priors=priors_out,
        likelihoods=likelihoods,
        step_accepts=step_accepts,
        swap_accepts=swap_accepts,
        parameter_names=circuit_fitter.parameters_to_fit,
        circuit_fitter=circuit_fitter,
    )

    # Save results
    df = results["analyzer"].to_dataframe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multi_circuit_results_{timestamp}.csv"
    df.to_csv(filename, index=False)

    # Plot distribution of likelihoods
    plt.figure(figsize=(10, 6))
    plt.hist(likelihoods.flatten(), color="blue", alpha=0.7)
    plt.xlabel("Log Likelihood")
    plt.ylabel("Frequency")
    plt.title("Distribution of Log Likelihoods")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Pick best parameters and simulate again
    best_params = df.sort_values(by="likelihood", ascending=False).head(1000)
    best_params_values = best_params[results["analyzer"].parameter_names].values

    # Simulate with best parameters
    sim_data = circuit_fitter.simulate_parameters(best_params_values)
    log_likelihood = circuit_fitter.calculate_likelihood_from_simulation(sim_data)
    log_prior = circuit_fitter.calculate_log_prior(best_params_values)
    results_df = organize_results(
        parameters_to_fit, best_params_values, log_likelihood, log_prior
    )

    # Plot final results
    plt.figure(figsize=(12, 8))
    plot_all_simulation_results(sim_data, results_df, ll_quartile=20)
    plt.show()

    return results, df


def main():
    # Initialize CircuitManager with existing circuits file
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    # List available circuits
    available_circuits = circuit_manager.list_circuits()
    print(f"Available circuits: {available_circuits}")

    # Define which circuits to fit together
    circuits_to_fit = [
        "trigger_antitrigger",
        "toehold_trigger",
        "sense_star_6",
        "cascade",
        "cffl_type_1",
        "star_antistar_1",
    ]

    # Filter to only include available circuits
    circuits_to_fit = [c for c in circuits_to_fit if c in available_circuits]

    if not circuits_to_fit:
        print("Error: None of the specified circuits are available.")
        return

    # Create circuit configurations
    circuit_configs = create_circuit_configs(
        circuit_manager, circuits_to_fit, min_time=30, max_time=210
    )

    # Load priors
    priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]
    parameters_to_fit = priors.Parameter.tolist()

    # Setup calibration
    calibration_params = setup_calibration()

    # Fit all circuits together
    results, df = fit_multiple_circuits(
        circuit_configs=circuit_configs,
        parameters_to_fit=parameters_to_fit,
        priors=priors,
        calibration_params=calibration_params,
        n_samples=1000,
        n_walkers=10,
        n_chains=6,
        n_sets=60,
    )
    print("Completed multi-circuit parameter estimation")


if __name__ == "__main__":
    main()
