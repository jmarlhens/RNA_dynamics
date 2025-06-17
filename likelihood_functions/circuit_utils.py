from data.circuits.circuit_configs import DATA_FILES, get_circuit_conditions
from utils.import_and_visualise_data import load_and_process_csv
from likelihood_functions.config import CircuitConfig
import pandas as pd
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor


def create_circuit_configs(
    circuit_manager, circuit_names, min_time=30, max_time=210, calibration_params=None
):
    """
    Create circuit configurations for multiple circuits

    Parameters
    ----------
    circuit_manager : CircuitManager
            Instance of circuit manager to create circuit models
    circuit_names : list
            List of circuit names to configure
    min_time : float, optional
            Minimum time to include in analysis
    max_time : float, optional
            Maximum time to include in analysis
    calibration_params : dict, optional
            Calibration parameters for fluorescence conversion

    Returns
    -------
    list
            List of CircuitConfig objects
    """

    circuit_configs = []

    for circuit_name in circuit_names:
        print(f"Creating configuration for {circuit_name}")
        # Get condition parameters from centralized configuration
        condition_params = get_circuit_conditions(circuit_name)

        # Load experimental data
        if circuit_name in DATA_FILES:
            data_file = DATA_FILES[circuit_name]
            experimental_data, tspan = load_and_process_csv(data_file)
        else:
            print(
                f"Warning: No data file defined for circuit '{circuit_name}', skipping."
            )
            continue

        # Create a circuit instance with the first condition's parameters
        first_condition = list(condition_params.keys())[0]
        circuit = circuit_manager.create_circuit(
            circuit_name, parameters=condition_params[first_condition]
        )

        # Create the circuit configuration
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

        circuit_configs.append(circuit_config)
        print(f"Created configuration for {circuit_name}")

    return circuit_configs


def setup_calibration():
    """Setup GFP calibration parameters"""
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
