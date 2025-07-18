import pandas as pd
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import setup_calibration
from circuits.circuit_generation.circuit_manager import CircuitManager
from data.circuits.circuit_configs import DATA_FILES, get_circuit_conditions


def recalculate_priors_for_previous_mcmc(csv_path, circuit_fitter, parameter_names):
    """Recalculate priors for previous MCMC samples using current prior calculation"""
    mcmc_results = pd.read_csv(csv_path)

    # Extract parameter values (already in log space from MCMC)
    log_parameter_matrix = mcmc_results[parameter_names].values

    # Recalculate priors using current CircuitFitter
    recalculated_log_priors = circuit_fitter.calculate_log_prior(log_parameter_matrix)

    # Update DataFrame
    mcmc_results["prior_new"] = recalculated_log_priors
    mcmc_results["posterior_new"] = mcmc_results["likelihood"] + recalculated_log_priors

    # Save updated results
    return mcmc_results


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

    # After creating circuit_fitter in fit_multiple_circuits()
    previous_csv = "../../data/fit_data/shared_parameters/cross_val/results_star_antistar_1_and_trigger_antitrigger_20250716_224438.csv"
    _ = recalculate_priors_for_previous_mcmc(
        previous_csv, circuit_fitter, parameters_to_fit
    )

    # Then use updated results for initial parameters
    # initial_parameters = load_best_parameters_from_csv(previous_csv, parameters_to_fit)


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

    fit_multiple_circuits(
        circuit_manager,
        circuits_to_fit,
        all_condition_params,
        all_experimental_data,
        all_tspan,
        priors,
        min_time=min_time,
        max_time=max_time,
    )


if __name__ == "__main__":
    main_shared_fit()
