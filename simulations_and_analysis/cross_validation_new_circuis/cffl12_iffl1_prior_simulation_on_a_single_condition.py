from datetime import datetime
import os
import pandas as pd

from utils.import_and_visualise_data import load_and_process_csv
from data.circuits.circuit_configs import DATA_FILES, get_circuit_conditions
from utils.GFP_calibration import setup_calibration
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from likelihood_functions.base import MCMCAdapter

from optimization.mcmc_utils import MCMCResultsWriter


# Initialize CircuitManager with existing circuits file
circuit_manager = CircuitManager(
    parameters_file="../../data/prior/model_parameters_priors_updated_tighter.csv",
    json_file="../../data/circuits/circuits.json",
)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "outputs/"
os.makedirs(output_dir, exist_ok=True)

# Define maximum simulation time
min_time = 30
max_time = 210
n_samples = 10000
n_walkers = 5
n_chains = 12


circuit_data = {}
for circuit_name, data_file in DATA_FILES.items():
    data, tspan = load_and_process_csv(data_file)
    circuit_data[circuit_name] = {"experimental_data": data, "tspan": tspan}

circuit_name = "cffl_12"

# Load priors
priors = pd.read_csv("../../data/prior/model_parameters_priors_updated_tighter.csv")

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
single_condition = "STAR1 5 nM"
condition_params_single_condition = {
    single_condition: condition_params[single_condition]
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
parameters_to_fit = [
    param for param in parameters_to_fit if param in parameter_in_the_model
]

circuit_fitter = CircuitFitter(
    [circuit_config], parameters_to_fit, priors, calibration_params
)

# Create MCMC adapter
adapter = MCMCAdapter(circuit_fitter)
initial_parameters = adapter.get_initial_parameters()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Only replace invalid filename characters but preserve case
safe_circuit_name = circuit_name.replace("/", "_")
os.makedirs("../../data/fit_data/individual_circuits_buffer/", exist_ok=True)
os.makedirs("../../data/fit_data/individual_circuits/", exist_ok=True)
os.makedirs("../../data/fit_data/individual_circuits/trajectories/", exist_ok=True)
os.makedirs(
    "../../data/fit_data/individual_circuits/analysis_trajectories/", exist_ok=True
)

buffer_writer = MCMCResultsWriter(
    path=f"../../data/fit_data/individual_circuits_buffer/buffer_{safe_circuit_name}_{timestamp}.csv",
    param_names=parameters_to_fit,
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

results_path = f"../../data/fit_data/individual_circuits/results_{safe_circuit_name}_{timestamp}_literature_prior.csv"
results_writer = MCMCResultsWriter(path=results_path, param_names=parameters_to_fit)
results_writer.save_state_in_file(
    parameters, priors_out, likelihoods, step_accepts, swap_accepts
)
results_writer.close()
