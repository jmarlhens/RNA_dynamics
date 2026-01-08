import os
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from utils.GFP_calibration import setup_calibration
from circuits.circuit_generation.circuit_manager import CircuitManager
from simulations_and_analysis.individual.individual_circuits_simulations import (
    create_circuit_simulation_data,
    plot_individual_circuit,
)

priors_csv_path = "../../data/prior/model_parameters_priors_092025_correction.csv"
circuit_manager = CircuitManager(
    parameters_file=priors_csv_path,
    json_file="../../data/circuits/circuits.json",
)
output_directory = "../../figures/individual_circuits/prior_simulations"

# make output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)
# model_priors = pd.read_csv(priors_csv_path)

calibration_parameters = setup_calibration()

time_bounds_max = 210
time_bounds_min = 30

circuit_name = "cffl_12"
circuit_name = "iffl_1"

model_priors = pd.read_csv(priors_csv_path)

# create a multivariate gaussian prior based on the mean and covariance of the priors
# mean
mean = np.log10(model_priors["Mean"].values)
# Civ is a diagonal matrix with the variance of each parameter
cov = np.diag(model_priors["log10stddev"].values)

multivariate_distribution = multivariate_normal(mean=mean, cov=cov)

#

multivariate_gaussian_prior_samples = multivariate_distribution.rvs(size=100)

# transform it in a df whose columns are the parameter names
multivariate_gaussian_prior_samples = pd.DataFrame(
    multivariate_gaussian_prior_samples, columns=model_priors["Parameter"].values
)

# Create circuit configuration
circuit_configuration, circuit_fitter = create_circuit_simulation_data(
    circuit_name,
    circuit_manager,
    calibration_parameters,
    time_bounds_max,
    time_bounds_min,
)

plot_individual_circuit(
    multivariate_gaussian_prior_samples,
    "prior",
    circuit_name,
    circuit_fitter,
    circuit_fitter.parameters_to_fit,
    output_directory,
)

# random_simulation_data, random_results_dataframe = (
# 	simulate_and_organize_parameter_sets(
# 		random_samples,
# 		circuit_fitter,
# 		circuit_fitter.parameters_to_fit,
# 	)
# )
#
# combined_random_simulation_data[circuit_name] = {
# 	"config": circuit_configuration,
# 	"combined_params": random_simulation_data["combined_params"],
# 	"simulation_results": random_simulation_data["simulation_results"],
# }
