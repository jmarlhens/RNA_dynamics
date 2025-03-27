from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from likelihood_functions import CircuitConfig, CircuitFitter
from likelihood_functions.utils import organize_results
from likelihood_functions.visualization import plot_all_simulation_results
from likelihood_functions.base import MCMCAdapter
from likelihood_functions.mcmc_analysis import analyze_mcmc_results
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor
from circuits.circuit_manager import CircuitManager


def setup_calibration():
    # Load calibration data
    data = pd.read_csv('../utils/calibration_gfp/gfp_Calibration.csv')

    # Fit the calibration curve
    calibration_results = fit_gfp_calibration(
        data,
        concentration_col='GFP Concentration (nM)',
        fluorescence_pattern='F.I. (a.u)'
    )

    # Get correction factor
    correction_factor, _ = get_brightness_correction_factor('avGFP', 'sfGFP')

    return {
        'slope': calibration_results['slope'],
        'intercept': calibration_results['intercept'],
        'brightness_correction': correction_factor
    }


def fit_single_circuit(circuit_manager, circuit_name, condition_params,
                       experimental_data, tspan, priors,
                       max_time=360, n_samples=10, n_walkers=10, n_chains=6):
    """
    Fit a single circuit and save its results using the new CircuitManager system
    """
    # Create a single circuit instance
    # We'll use the first condition's parameters to initialize the circuit
    first_condition = list(condition_params.keys())[0]
    circuit = circuit_manager.create_circuit(circuit_name, parameters=condition_params[first_condition])

    # Create circuit configuration with single model
    circuit_config = CircuitConfig(
        model=circuit.model,  # Now passing a single PySB model
        name=circuit_name,
        condition_params=condition_params,
        experimental_data=experimental_data,
        tspan=tspan,
        max_time=max_time
    )

    # Create circuit fitter with single config
    parameters_to_fit = priors.Parameter.tolist()
    calibration_params = setup_calibration()
    circuit_fitter = CircuitFitter([circuit_config], parameters_to_fit, priors, calibration_params)

    # Create MCMC adapter
    adapter = MCMCAdapter(circuit_fitter)
    initial_parameters = adapter.get_initial_parameters()

    # Setup and run parallel tempering
    pt = adapter.setup_parallel_tempering(n_walkers=n_walkers, n_chains=n_chains)
    parameters, priors_out, likelihoods, step_accepts, swap_accepts = pt.run(
        initial_parameters=initial_parameters,
        n_samples=n_samples,
        target_acceptance_ratio=0.4,
        adaptive_temperature=True
    )

    # Analyze results
    results = analyze_mcmc_results(
        parameters=parameters,
        priors=priors_out,
        likelihoods=likelihoods,
        step_accepts=step_accepts,
        swap_accepts=swap_accepts,
        parameter_names=circuit_fitter.parameters_to_fit,
        circuit_fitter=circuit_fitter
    )

    # Save results
    df = results['analyzer'].to_dataframe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{circuit_name.lower().replace('/', '_')}_{timestamp}.csv"
    df.to_csv(filename, index=False)

    # Plot and save best fit results
    best_params = df.sort_values(by='likelihood', ascending=False).head(100)
    best_params_values = best_params[results['analyzer'].parameter_names].values

    # Using param_values for multiple simulations
    # Convert best_params_values to a DataFrame for easier handling
    param_df = pd.DataFrame(best_params_values, columns=parameters_to_fit)

    # Simulate with multiple parameter sets
    sim_data = circuit_fitter.simulate_parameters(param_df)
    log_likelihood = circuit_fitter.calculate_likelihood_from_simulation(sim_data)
    log_prior = circuit_fitter.calculate_log_prior(param_df.values)
    results_df = organize_results(parameters_to_fit, param_df.values, log_likelihood, log_prior)

    plt.figure(figsize=(12, 8))
    plot_all_simulation_results(sim_data, results_df, ll_quartile=20)
    plt.savefig(f"fit_{circuit_name.lower().replace('/', '_')}_{timestamp}.png")
    plt.close()

    return results, df


def main():
    # Initialize CircuitManager with existing circuits file
    circuit_manager = CircuitManager(
        parameters_file="../data/model_parameters_priors.csv",
        json_file="../data/circuits/circuits.json"
    )
    # No need to register circuits as they're already in the JSON file
    # But uncomment this if you have additional circuits to register
    # register_all_circuits(circuit_manager)

    # List available circuits to verify
    available_circuits = circuit_manager.list_circuits()
    print(f"Available circuits: {available_circuits}")

    # Load data
    max_time = 360
    toehold_trigger_data, tspan_toehold = load_and_process_csv('../data/data_parameter_estimation/toehold_trigger.csv')
    sense_star_data, tspan_star = load_and_process_csv('../data/data_parameter_estimation/sense_star.csv')
    cascade_data, tspan_cascade = load_and_process_csv('../data/data_parameter_estimation/cascade.csv')
    cffl_type_1_data, tspan_cffl_type_1 = load_and_process_csv('../data/data_parameter_estimation/c1_ffl_and.csv')

    # Define condition parameters for each circuit
    condition_configs = {
        "toehold": {
            "condition_params": {
                "To3 5 + Tr3 5": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 5},
                "To3 5 + Tr3 4": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 4},
                "To3 5 + Tr3 3": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 3},
                "To3 5 + Tr3 2": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 2},
                "To3 5 + Tr3 1": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 1},
                "To3 5": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 0}
            },
            "experimental_data": toehold_trigger_data,
            "tspan": tspan_toehold
        },
        "star": {
            "condition_params": {
                "Se6 5 nM + St6 15 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 15},
                "Se6 5 nM + St6 10 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 10},
                "Se6 5 nM + St6 5 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 5},
                "Se6 5 nM + St6 3 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 3},
                "Se6 5 nM + St6 0 nM": {"k_Sense6_GFP_concentration": 0, "k_Star6_concentration": 0}
            },
            "experimental_data": sense_star_data,
            "tspan": tspan_star
        },
        "cascade": {
            "condition_params": {
                "To3 3 nM + Se6Tr3P 5 nM + St6 15 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 15
                },
                "To3 3 nM + Se6Tr3P 5 nM + St6 10 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 10
                },
                "To3 3 nM + Se6Tr3P 5 nM + St6 5 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 5
                },
                "To3 3 nM + Se6Tr3P 5 nM + St6 3 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 3
                },
                "To3 3 nM + Se6Tr3P 5 nM + St6 0 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 0
                }
            },
            "experimental_data": cascade_data,
            "tspan": tspan_cascade
        },
        "cffl_type_1": {
            "condition_params": {
                "15 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 15,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "12 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 12,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "10 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 10,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "7 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 7,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "5 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 5,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "3 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 3,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "0 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 0,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                }
            },
            "experimental_data": cffl_type_1_data,
            "tspan": tspan_cffl_type_1
        }
    }

    # Load priors
    priors = pd.read_csv('../data/model_parameters_priors.csv')
    priors = priors[priors['Parameter'] != 'k_prot_deg']

    # Fit each circuit individually
    circuits_to_fit = ["star", "cascade", "cffl_type_1", "toehold"]  # Using exact names from the JSON file
    for circuit_name in circuits_to_fit:
        if circuit_name in available_circuits:
            print(f"\nFitting {circuit_name}...")
            # Get the configuration, handling the case where the name might differ
            config_key = circuit_name
            if circuit_name not in condition_configs and circuit_name == "cffl_type_1":
                # Handle the special case for cffl_type_1
                config_key = "cffl_type_1"

            if config_key in condition_configs:
                config = condition_configs[config_key]
                results, df = fit_single_circuit(
                    circuit_manager=circuit_manager,
                    circuit_name=circuit_name,
                    condition_params=config["condition_params"],
                    experimental_data=config["experimental_data"],
                    tspan=config["tspan"],
                    priors=priors,
                    max_time=max_time
                )
                print(f"Completed fitting {circuit_name}")
            else:
                print(f"Warning: Configuration for circuit '{circuit_name}' not found.")
        else:
            print(f"Warning: Circuit '{circuit_name}' not found in available circuits.")


if __name__ == '__main__':
    main()
