from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from likelihood_functions.utils import organize_results
from likelihood_functions.visualization import plot_all_simulation_results
from likelihood_functions.base import MCMCAdapter
from likelihood_functions.mcmc_analysis import analyze_mcmc_results
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor
from circuits.circuit_generation.circuit_manager import CircuitManager
from data.circuits.circuit_configs import DATA_FILES, get_circuit_conditions


def setup_calibration():
    # Load calibration data
    data = pd.read_csv("../../utils/calibration_gfp/gfp_Calibration.csv")

    # Fit the calibration curve
    calibration_results = fit_gfp_calibration(
        data,
        concentration_col="GFP Concentration (nM)",
        fluorescence_pattern="F.I. (a.u)",
    )

    # Get correction factor
    correction_factor, _ = get_brightness_correction_factor("avGFP", "sfGFP")

    return {
        "slope": calibration_results["slope"],
        "intercept": calibration_results["intercept"],
        "brightness_correction": correction_factor,
    }


def fit_single_circuit(
    circuit_manager,
    circuit_name,
    condition_params,
    experimental_data,
    tspan,
    priors,
    min_time=30,
    max_time=210,
    n_samples=1000,
    n_walkers=12,
    n_chains=8,
):
    """
    Fit a single circuit and save its results using the new CircuitManager system
    """
    # Create a single circuit instance
    # We'll use the first condition's parameters to initialize the circuit
    first_condition = list(condition_params.keys())[0]
    circuit = circuit_manager.create_circuit(
        circuit_name, parameters=condition_params[first_condition]
    )

    # Create circuit configuration with single model
    circuit_config = CircuitConfig(
        model=circuit.model,  # Now passing a single PySB model
        name=circuit_name,
        condition_params=condition_params,
        experimental_data=experimental_data,
        tspan=tspan,
        min_time=min_time,
        max_time=max_time,
    )

    # Create circuit fitter with single config
    parameters_to_fit = priors.Parameter.tolist()
    calibration_params = setup_calibration()
    circuit_fitter = CircuitFitter(
        [circuit_config], parameters_to_fit, priors, calibration_params
    )

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
    # Only replace invalid filename characters but preserve case
    safe_circuit_name = circuit_name.replace("/", "_")
    filename = f"../data/fit_data/individual_circuits/results_prior_updated_{safe_circuit_name}_{timestamp}.csv"
    df.to_csv(filename, index=False)

    # Plot and save best fit results
    best_params = df.sort_values(by="likelihood", ascending=False).head(100)
    best_params_values = best_params[results["analyzer"].parameter_names].values

    # Using param_values for multiple simulations
    # Convert best_params_values to a DataFrame for easier handling
    param_df = pd.DataFrame(best_params_values, columns=parameters_to_fit)

    # Simulate with multiple parameter sets
    # sim_data = circuit_fitter.simulate_parameters(param_df)
    # Expected type 'ndarray | ndarray', got 'DataFrame' instead
    sim_data = circuit_fitter.simulate_parameters(param_df.values)
    log_likelihood = circuit_fitter.calculate_likelihood_from_simulation(sim_data)
    log_prior = circuit_fitter.calculate_log_prior(param_df.values)
    results_df = organize_results(
        parameters_to_fit, param_df.values, log_likelihood, log_prior
    )

    plt.figure(figsize=(12, 8))
    plot_all_simulation_results(sim_data, results_df, ll_quartile=20)
    plt.savefig(f"fit_{safe_circuit_name}_{timestamp}.png")
    plt.close()

    return results, df


def main():
    # Initialize CircuitManager with existing circuits file
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors_updated.csv",
        json_file="../../data/circuits/circuits.json",
    )

    # List available circuits to verify
    available_circuits = circuit_manager.list_circuits()
    print(f"Available circuits: {available_circuits}")

    # Define maximum simulation time
    min_time = 30
    max_time = 210

    # Load data for all circuits in central configuration
    circuit_data = {}
    for circuit_name, data_file in DATA_FILES.items():
        try:
            data, tspan = load_and_process_csv(data_file)
            circuit_data[circuit_name] = {"experimental_data": data, "tspan": tspan}
            print(f"Loaded data for {circuit_name}")
        except Exception as e:
            print(f"Error loading data for {circuit_name}: {e}")

    # Load priors
    priors = pd.read_csv("../../data/prior/model_parameters_priors_updated.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]

    # Fit each circuit individually
    # You can specify which circuits to fit
    # circuits_to_fit = ["star", "cascade", "cffl_type_1", "toehold"]
    # circuits_to_fit = ["toehold_trigger", "cffl_type_1", "star_antistar_1"]
    circuits_to_fit = [
        "cffl_type_1",
        "sense_star_6",
        "cascade",
    ]
    circuits_to_fit = [
        "toehold_trigger",
        "star_antistar_1",
        "trigger_antitrigger",
        "sense_star_6",
        "cascade",
        "cffl_type_1",
    ]
    # circuits_to_fit = ["sense_star_6"]

    for circuit_name in circuits_to_fit:
        if circuit_name in available_circuits:
            print(f"\nFitting {circuit_name}...")

            # Get condition parameters from centralized configuration
            condition_params = get_circuit_conditions(circuit_name)
            if not condition_params:
                print(
                    f"Warning: No conditions defined for circuit '{circuit_name}' in configuration, skipping."
                )
                continue

            # Get the experimental data for this circuit
            if circuit_name not in circuit_data:
                print(
                    f"Warning: No data loaded for circuit '{circuit_name}', skipping."
                )
                continue

            data_info = circuit_data[circuit_name]

            # Fit the circuit
            try:
                _, _ = fit_single_circuit(
                    circuit_manager=circuit_manager,
                    circuit_name=circuit_name,
                    condition_params=condition_params,
                    experimental_data=data_info["experimental_data"],
                    tspan=data_info["tspan"],
                    priors=priors,
                    min_time=min_time,
                    max_time=max_time,
                )
                print(f"Completed fitting {circuit_name}")
            except Exception as e:
                print(f"Error fitting circuit {circuit_name}: {e}")
        else:
            print(f"Warning: Circuit '{circuit_name}' not found in available circuits.")


if __name__ == "__main__":
    main()
