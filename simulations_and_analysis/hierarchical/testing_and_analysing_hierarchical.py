from likelihood_functions.hierarchical_likelihood.base_hierarchical import (
    HierarchicalCircuitFitter,
)
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.circuit_utils import create_circuit_configs, setup_calibration
from simulations_and_analysis.individual.individual_circuits_statistics import (
    load_individual_circuit_results,
)
from likelihood_functions.hierarchical_likelihood.mcmc_adaption import (
    HierarchicalMCMCAdapter,
)
import numpy as np
import pandas as pd
import cProfile
import time


def fit_hierarchical_multiple_circuits(
    circuit_configs,
    parameters_to_fit,
    priors,
    calibration_params,
):
    print("Loading individual circuit MCMC results")
    individual_circuit_posterior_results = load_individual_circuit_results(
        individual_results_directory="../../../data/fit_data/individual_circuits/"
    )
    print(
        f"Loaded {len(individual_circuit_posterior_results)} individual circuit results for hyperparameter estimation"
    )

    print("Setting up hierarchical model...")
    # Create hierarchical circuit fitter
    hierarchical_fitter = HierarchicalCircuitFitter(
        circuit_configs,
        parameters_to_fit,
        priors,
        calibration_params,
    )

    # Generate test parameters
    np.random.seed(42)
    params = hierarchical_fitter.generate_hierarchical_parameters(96)

    # =========================================================================
    # Prior Calculation Performance Test
    # =========================================================================

    # Test 1: Direct calculation
    profiler = cProfile.Profile()
    profiler.enable()
    direct_log_prior = hierarchical_fitter.calculate_hyperparameter_prior(params)
    profiler.disable()
    profiler.dump_stats("prior_calculation_profile.prof")

    print("=== DIRECT CALCULATION ===")
    print(f"Shape: {params.shape}")
    print(f"Results: {direct_log_prior}")

    # Test 2: MCMC adapter calculation (with reshaping)
    mcmc_adapter = HierarchicalMCMCAdapter(hierarchical_fitter)
    log_prior_function = mcmc_adapter.get_log_prior_function()

    # Test the exact same parameters through MCMC adapter
    start_time = time.time()
    adapter_log_prior_total = log_prior_function(params)
    end_time = time.time()
    print("=== MCMC ADAPTER CALCULATION ===")
    print(f"Shape: {params.shape}")
    print(f"Results: {adapter_log_prior_total}")
    print(
        "Direct log prior matches MCMC adapter log prior:",
        np.allclose(direct_log_prior["total"], adapter_log_prior_total),
    )
    print(
        f"Time taken for MCMC adapter prior calculation: {end_time - start_time:.4f} seconds"
    )

    # =========================================================================
    # Likelihood Calculation Performance Test
    # =========================================================================

    # Test 1: Direct calculation
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()
    direct_log_likelihood = hierarchical_fitter.calculate_data_likelihood(params)
    end_time = time.time()
    profiler.disable()
    profiler.dump_stats("likelihood_calculation_profile.prof")
    print(
        f"Time taken for direct likelihood calculation: {end_time - start_time:.4f} seconds"
    )
    print("=== DIRECT CALCULATION ===")
    print(f"Shape: {params.shape}")
    print(f"Results: {direct_log_likelihood}")

    # Test 2: MCMC adapter calculation (with reshaping)
    log_likelihood_function = mcmc_adapter.get_log_likelihood_function()
    start_time = time.time()
    adapter_log_likelihood_total = log_likelihood_function(params)
    end_time = time.time()
    print(
        f"Time taken for MCMC adapter likelihood calculation: {end_time - start_time:.4f} seconds"
    )
    print("=== MCMC ADAPTER CALCULATION ===")
    print(f"Shape: {params.shape}")
    print(f"Results: {adapter_log_likelihood_total}")
    print(
        "Direct log prior matches MCMC adapter log prior:",
        np.allclose(direct_log_prior["total"], adapter_log_prior_total),
    )
    print(
        "Direct log likelihood matches MCMC adapter log likelihood:",
        np.allclose(direct_log_likelihood["total"], adapter_log_likelihood_total),
    )


if __name__ == "__main__":
    # Initialize CircuitManager with existing circuits file
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    # Define circuits to fit
    circuits_to_fit = [
        "trigger_antitrigger",
        "toehold_trigger",
        "sense_star_6",
        "cascade",
        "cffl_type_1",
        "star_antistar_1",
    ]

    # List available circuits
    available_circuits = circuit_manager.list_circuits()
    print(f"Available circuits: {available_circuits}")

    # Filter to only include available circuits
    circuits_to_fit = [c for c in circuits_to_fit if c in available_circuits]

    if not circuits_to_fit:
        print("Error: None of the specified circuits are available.")
        exit()

    # Create circuit configurations
    circuit_configs = create_circuit_configs(
        circuit_manager, circuits_to_fit, min_time=30, max_time=210
    )

    # Load priors
    priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]  # Exclude protein degradation
    parameters_to_fit = priors.Parameter.tolist()

    # Setup calibration
    calibration_params = setup_calibration()

    # Run hierarchical parameter estimation
    results = fit_hierarchical_multiple_circuits(
        circuit_configs,
        parameters_to_fit,
        priors,
        calibration_params,
    )

    print("Hierarchical parameter estimation completed successfully!")
