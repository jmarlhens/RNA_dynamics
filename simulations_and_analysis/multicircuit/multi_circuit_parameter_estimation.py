from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from utils.process_experimental_data import organize_results
from analysis_and_figures.plots_simulation import plot_circuit_simulations
from likelihood_functions.base import MCMCAdapter
from analysis_and_figures.mcmc_analysis import analyze_mcmc_results
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import setup_calibration
from circuits.circuit_generation.circuit_manager import CircuitManager
from data.circuits.circuit_configs import DATA_FILES, get_circuit_conditions


def load_best_parameters_from_csv(csv_path, parameter_names):
    """Load best parameter set from previous MCMC run"""
    previous_results = pd.read_csv(csv_path)
    best_row = previous_results.loc[previous_results["likelihood"].idxmax()]

    # add little nnoise (stdv 0.05)
    import numpy as np

    best_row[parameter_names] += np.random.normal(0, 0.1, size=len(parameter_names))
    return best_row[parameter_names].values


def fit_multiple_circuits(
    circuit_manager,
    circuit_names,  # List of circuit names
    all_condition_params,  # Dict of {circuit_name: condition_params}
    all_experimental_data,  # Dict of {circuit_name: experimental_data}
    all_tspan,  # Dict of {circuit_name: tspan}
    priors,
    min_time=30,
    max_time=210,
    n_samples=500,
    n_walkers=10,
    n_chains=10,
):
    """
    Fit multiple circuits simultaneously with shared parameters
    """
    # Create circuit instances and configs for all circuits
    circuit_configs = []

    calibration_params = setup_calibration()

    for circuit_name in circuit_names:
        # Create circuit instance
        first_condition = list(all_condition_params[circuit_name].keys())[0]
        circuit = circuit_manager.create_circuit(
            circuit_name, parameters=all_condition_params[circuit_name][first_condition]
        )

        # Create circuit configuration
        circuit_config = CircuitConfig(
            model=circuit.model,
            name=circuit_name,
            condition_params=all_condition_params[circuit_name],
            experimental_data=all_experimental_data[circuit_name],
            tspan=all_tspan[circuit_name],
            min_time=min_time,
            max_time=max_time,
            calibration_params=calibration_params,
        )
        circuit_configs.append(circuit_config)

    # Create circuit fitter with MULTIPLE configs (shared parameters)
    parameters_to_fit = priors.Parameter.tolist()
    circuit_fitter = CircuitFitter(
        circuit_configs, parameters_to_fit, priors, calibration_params
    )

    print(f"Fitting circuits together: {circuit_names}")
    print("Total experimental data points across all circuits:")
    for config in circuit_configs:
        print(f"  {config.name}: {len(config.experimental_data)} data points")

    # Create MCMC adapter
    adapter = MCMCAdapter(circuit_fitter)
    # initial_parameters = adapter.get_initial_parameters()

    # Load initial parameters from previous run
    previous_csv = "../../data/fit_data/shared_parameters/cross_val/results_star_antistar_1_and_trigger_antitrigger_20250716_224438.csv"
    initial_parameters = load_best_parameters_from_csv(previous_csv, parameters_to_fit)

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

    # Save results with both circuit names
    df = results["analyzer"].to_dataframe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    circuit_names_str = "_and_".join([name.replace("/", "_") for name in circuit_names])
    filename = f"../../data/fit_data/shared_parameters/results_{circuit_names_str}_{timestamp}.csv"
    df.to_csv(filename, index=False)

    # Plot and save best fit results for ALL circuits
    best_params = df.sort_values(by="likelihood", ascending=False).head(50)
    best_params_values = best_params[results["analyzer"].parameter_names].values
    param_df = pd.DataFrame(best_params_values, columns=parameters_to_fit)

    # Simulate with multiple parameter sets (will simulate ALL circuits)
    sim_data = circuit_fitter.simulate_parameters(param_df.values)
    likelihood_breakdown = (
        circuit_fitter.calculate_likelihood_from_simulation_with_breakdown(sim_data)
    )
    log_prior = circuit_fitter.calculate_log_prior(param_df.values)
    results_df = organize_results(
        parameters_to_fit, param_df.values, likelihood_breakdown, log_prior
    )

    # Plot results for all circuits
    plt.figure(figsize=(15, 8))
    plot_circuit_simulations(sim_data, results_df)
    plt.suptitle(f"Shared Parameter Fit: {' & '.join(circuit_names)}")
    plt.savefig(f"fit_shared_{circuit_names_str}_{timestamp}.png")
    plt.close()

    return results, df


def main_shared_fit():
    """Modified main function to fit star_antistar_1 and inhibited_cascade together"""

    # Initialize CircuitManager
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors_updated_tighter.csv",
        json_file="../../data/circuits/circuits.json",
    )

    # Define circuits to fit together
    circuits_to_fit = [
        "star_antistar_1",
        # "toehold_trigger",
        # "inhibited_cascade",
        "trigger_antitrigger",
    ]
    # inhibited cascade
    # inhibited_incoherent_cascade

    # Define maximum simulation time
    min_time = 30
    max_time = 210

    # Load data for the specific circuits
    all_experimental_data = {}
    all_tspan = {}
    all_condition_params = {}

    for circuit_name in circuits_to_fit:
        if circuit_name in DATA_FILES:
            # Load experimental data
            data, tspan = load_and_process_csv(DATA_FILES[circuit_name])
            all_experimental_data[circuit_name] = data
            all_tspan[circuit_name] = tspan

            # Get condition parameters
            condition_params = get_circuit_conditions(circuit_name)
            all_condition_params[circuit_name] = condition_params

            print(f"Loaded data for {circuit_name}")
            print(f"  Conditions: {list(condition_params.keys())}")
            print(f"  Data points: {len(data)}")
        else:
            print(f"ERROR: No data file found for {circuit_name}")
            return

    # Load priors
    priors = pd.read_csv("../../data/prior/model_parameters_priors_updated_tighter.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]

    # Fit both circuits together with shared parameters
    print("\n=== FITTING CIRCUITS TOGETHER WITH SHARED PARAMETERS ===")
    print(f"Circuits: {circuits_to_fit}")
    print(f"Parameters to fit: {priors.Parameter.tolist()}")

    results, df = fit_multiple_circuits(
        circuit_manager=circuit_manager,
        circuit_names=circuits_to_fit,
        all_condition_params=all_condition_params,
        all_experimental_data=all_experimental_data,
        all_tspan=all_tspan,
        priors=priors,
        min_time=min_time,
        max_time=max_time,
        n_samples=2000,
        n_walkers=4,
        n_chains=12,
    )

    print("Completed shared parameter fitting!")
    print("Results saved to CSV and plot generated")

    # Print some summary statistics
    best_likelihood = df["likelihood"].max()
    print(f"Best likelihood achieved: {best_likelihood:.2f}")

    # Show parameter estimates (posterior means)
    param_names = results["analyzer"].parameter_names
    param_means = df[param_names].mean()
    print("\nPosterior parameter means (log10):")
    for param, mean_val in param_means.items():
        print(f"  {param}: {mean_val:.3f} (linear: {10**mean_val:.2e})")


if __name__ == "__main__":
    main_shared_fit()
