from datetime import datetime
import pandas as pd
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from likelihood_functions.base import MCMCAdapter
from analysis_and_figures.mcmc_analysis import analyze_mcmc_results
from utils.import_and_visualise_data import load_and_process_csv
from circuits.circuit_generation.circuit_manager import CircuitManager
from data.circuits.circuit_configs import DATA_FILES, get_circuit_conditions
from utils.GFP_calibration import setup_calibration


def fit_single_circuit(
    circuit_manager,
    circuit_name,
    condition_params,
    experimental_data,
    tspan,
    priors,
    min_time=30,
    max_time=210,
    n_samples=20000,
    n_walkers=5,
    n_chains=10,
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

    calibration_params = setup_calibration()

    # Create circuit configuration with single model
    circuit_config = CircuitConfig(
        model=circuit.model,
        name=circuit_name,
        condition_params=condition_params,
        experimental_data=experimental_data,
        tspan=tspan,
        min_time=min_time,
        max_time=max_time,
        calibration_params=calibration_params,
    )

    # Create circuit fitter with single config
    parameters_to_fit = priors.Parameter.tolist()

    # Only keep the parameters that are actually part of the model ([parameter_name.name for parameter_name in circuit_config.model.parameters])
    parameter_in_the_model = [
        parameter_name.name for parameter_name in circuit_config.model.parameters
    ]
    parameters_to_fit = [
        param for param in parameters_to_fit if param in parameter_in_the_model
    ]

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
    filename = f"../../data/fit_data/individual_circuits/results_{safe_circuit_name}_{timestamp}.csv"
    df.to_csv(filename, index=False)

    return results, df


def main():
    # Initialize CircuitManager with existing circuits file
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors_updated_tighter.csv",
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
        data, tspan = load_and_process_csv(data_file)
        circuit_data[circuit_name] = {"experimental_data": data, "tspan": tspan}
        print(f"Loaded data for {circuit_name}")

    # Load priors
    priors = pd.read_csv("../../data/prior/model_parameters_priors_updated_tighter.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]

    # Fit each circuit individually
    circuits_to_fit = [
        # "constitutive sfGFP",
        # "sense_star_6",
        # "toehold_trigger",
        # "star_antistar_1",
        # "trigger_antitrigger",
        # "cascade",
        # "cffl_type_1",
        "inhibited_incoherent_cascade",
        "inhibited_cascade",
        "or_gate_c1ffl",
        "iffl_1",
        "cffl_12",
    ]

    # for circuit_name in ["cffl_12", "iffl_1", "inhibited_cascade"]:
    #     conditions = get_circuit_conditions(circuit_name)
    #     print(f"\n{circuit_name} conditions:")
    #     for cond_name, params in conditions.items():
    #         print(f"  {cond_name}: {params}")

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

        else:
            print(f"Warning: Circuit '{circuit_name}' not found in available circuits.")


if __name__ == "__main__":
    main()
