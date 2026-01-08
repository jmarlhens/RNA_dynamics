import numpy as np
from datetime import datetime
import os
import pandas as pd
from simulations_and_analysis.cross_validation_new_circuis.generating_prior_from_data import (
    estimate_posterior_statistics_ledoitwolf,
    load_circuit_posterior_data,
)
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import setup_calibration
from data.circuits.circuit_configs import DATA_FILES, get_circuit_conditions
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import (
    CircuitFitter,
    MCMCAdapter,
)
from optimization.mcmc_utils import MCMCResultsWriter


def create_multivariate_log_prior(
    log_mean_vector: np.ndarray, log_covariance_matrix: np.ndarray
):
    """
    Create a multivariate normal log prior function with precomputed constants.

    Args:
        log_mean_vector: Mean vector for parameters in log space
        log_covariance_matrix: Covariance matrix for parameters in log space

    Returns:
        Function that calculates log prior probabilities for MCMC walker arrays
    """
    # Precompute expensive operations once
    covariance_inv = np.linalg.inv(log_covariance_matrix)
    # n_params = len(log_mean_vector)
    # log_det_cov = np.linalg.slogdet(log_covariance_matrix)[1]
    # log_normalization_constant = -0.5 * (log_det_cov + n_params * np.log(2 * np.pi))

    def calculate_log_prior(walker_params: np.ndarray) -> np.ndarray:
        """
        Calculate log prior probability for MCMC walker parameters.

        Args:
            walker_params: Array of shape (n_walkers, n_chains, n_params) or (n_samples, n_params)

        Returns:
            Array of total log prior probabilities, flattened to match reshaped input n_samples
        """
        # Handle both MCMC walker format and standard 2D format
        original_shape = walker_params.shape
        if walker_params.ndim == 3:
            # Reshape from (n_walkers, n_chains, n_params) to (n_samples, n_params)
            flattened_params = walker_params.reshape(-1, original_shape[-1])
        elif walker_params.ndim == 1:
            flattened_params = walker_params.reshape(1, -1)
        else:
            flattened_params = walker_params

        # Center the parameters
        centered_params = flattened_params - log_mean_vector

        # Compute quadratic form: (x - μ)ᵀ Σ⁻¹ (x - μ) for all samples
        quadratic_form = np.einsum(
            "ij,jk,ik->i", centered_params, covariance_inv, centered_params
        )

        log_prior = -0.5 * quadratic_form
        # + log_normalization_constant)

        # Reshape back to (n_samples,)
        return log_prior.reshape(original_shape[:-1])

    return calculate_log_prior


subfolder = "/shared_parameters/results_star_antistar_1_and_trigger_antitrigger"
individual_results_directory = "../../data/fit_data" + subfolder
results_filename = "results_star_antistar_1_and_trigger_antitrigger_20251116_023618.csv"
results_filepath = f"{individual_results_directory}/{results_filename}"
prior_parameters_filepath = (
    "../../data/prior/model_parameters_priors_092025_correction.csv"
)

# Load and process data
samples_posterior, prior_coordinates, _ = load_circuit_posterior_data(
    results_filepath, prior_parameters_filepath
)


# Initialize CircuitManager with existing circuits file
circuit_manager = CircuitManager(
    parameters_file="../../data/prior/model_parameters_priors_092025_correction.csv",
    json_file="../../data/circuits/circuits.json",
)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "outputs/"
os.makedirs(output_dir, exist_ok=True)

# Define maximum simulation time
min_time = 30
max_time = 210
n_samples = 60000
n_walkers = 5
n_chains = 12


circuit_data = {}
for circuit_name, data_file in DATA_FILES.items():
    data, tspan = load_and_process_csv(data_file)
    circuit_data[circuit_name] = {"experimental_data": data, "tspan": tspan}

circuit_name = "cffl_12"
selected_conditions = ["STAR1 5 nM", "STAR6 15 nM"]

circuit_name = "iffl_1"
selected_conditions = [
    "Sense-aTrigger 0 nM",
    "STAR-Trigger 0 nM",
    "Sense-aTrigger 5 nM",
]

# Load priors
priors = pd.read_csv("../../data/prior/model_parameters_priors_092025_correction.csv")

# Get condition parameters from centralized configuration
condition_params = get_circuit_conditions(circuit_name)
data_info = circuit_data[circuit_name]

# Fit the circuit
first_condition = list(condition_params.keys())[0]
circuit = circuit_manager.create_circuit(
    circuit_name, parameters=condition_params[first_condition]
)

calibration_params = setup_calibration()


# only keep  a single condition_params ('STAR1 5 nM')
condition_params_single_condition = {
    condition: condition_params[condition] for condition in selected_conditions
}


# Create circuit configuration with single model
circuit_config = CircuitConfig(
    model=circuit.model,
    name=circuit_name,
    condition_params=condition_params_single_condition,
    experimental_data=circuit_data[circuit_name]["experimental_data"],
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
parameter_names = [
    param for param in parameters_to_fit if param in parameter_in_the_model
]

circuit_fitter = CircuitFitter(
    [circuit_config], parameter_names, priors, calibration_params
)

# Estimate posterior statistics
posterior_mean, posterior_covariance = estimate_posterior_statistics_ledoitwolf(
    samples_posterior, parameter_names
)
mean = np.array([posterior_mean[param] for param in parameter_names])
cov = np.array(
    [[posterior_covariance[i][j] for j in parameter_names] for i in parameter_names]
)

calculate_log_prior = create_multivariate_log_prior(
    log_mean_vector=mean, log_covariance_matrix=cov
)


# Create MCMC adapter
adapter = MCMCAdapter(circuit_fitter, log_prior=calculate_log_prior)
initial_parameters = adapter.get_initial_parameters()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Only replace invalid filename characters but preserve case
safe_circuit_name = circuit_name.replace("/", "_")
os.makedirs("../../data/fit_data/individual_circuits_buffer/", exist_ok=True)
os.makedirs("../../data/fit_data/individual_circuits/", exist_ok=True)
os.makedirs(
    "../../data/fit_data/individual_circuits/obsolete/trajectories/", exist_ok=True
)
os.makedirs(
    "../../data/fit_data/individual_circuits/obsolete/analysis_trajectories/",
    exist_ok=True,
)

buffer_writer = MCMCResultsWriter(
    path=f"../../data/fit_data/individual_circuits_buffer/buffer_{safe_circuit_name}_{timestamp}.csv",
    param_names=parameter_names,
)

# Setup and run parallel tempering
pt = adapter.setup_parallel_tempering(n_walkers=n_walkers, n_chains=n_chains)
parameters, priors_out, likelihoods, step_accepts, swap_accepts = pt.run(
    initial_parameters=initial_parameters,
    n_samples=n_samples,
    target_acceptance_ratio=0.4,
    adaptive_temperature=True,
    mcmc_writer=buffer_writer,
)
buffer_writer.close()

print("Completed Model Calibration", flush=True)

results_path = f"../../data/fit_data/individual_circuits/transfer_learning/data_informed_prior/results_{safe_circuit_name}_{timestamp}_informed_prior.csv"
# create the folder if it doesn't exist
os.makedirs(os.path.dirname(results_path), exist_ok=True)
results_writer = MCMCResultsWriter(path=results_path, param_names=parameter_names)
results_writer.save_state_in_file(
    parameters, priors_out, likelihoods, step_accepts, swap_accepts
)
results_writer.close()
