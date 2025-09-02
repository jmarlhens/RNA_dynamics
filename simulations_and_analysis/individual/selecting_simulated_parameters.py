import pandas as pd
import matplotlib.pyplot as plt
from circuits.circuit_generation.circuit_manager import CircuitManager
from utils.GFP_calibration import setup_calibration, convert_nm_to_au
from simulations_and_analysis.individual.individual_circuits_simulations import (
    create_circuit_simulation_data,
)
from utils.process_experimental_data import organize_results
from analysis_and_figures.plots_simulation import extract_trajectory_data


subfolder = "/50000_steps"
file = "results_constitutive sfGFP_20250721_122844.csv"
parameter_sim_folder = "../../data/data_parameter_estimation"
input_directory = "../../data/fit_data/individual_circuits" + subfolder
filepath = f"{input_directory}/{file}"

mcmc_results = pd.read_csv(filepath)

# scatter likelihood vs prior
mcmc_results.plot.scatter(
    x="prior", y="likelihood", title="Likelihood vs Prior (Log Scale)", figsize=(10, 6)
)
plt.show()
#
# # SELECT HIGHEST POSTERIOR
# highest_posterior = mcmc_results.loc[mcmc_results['posterior'].idxmax()]
# print("Highest Posterior Parameters:")
# for param, value in highest_posterior.items():
# 	if param in ['prior', 'likelihood', 'posterior']:
# 		print(f"{param}: {value}")
#
# print("kinetic parameters:")
# for param in mcmc_results.columns:
# 	if 'k_' in param or 'conc_' in param or "K_" in param:
# 		print(f"{param}: {highest_posterior[param]}")

# new criteria is 2 * prior + likelihood
mcmc_results["new_criteria"] = 6 * mcmc_results["prior"] + mcmc_results["likelihood"]

# best parameters based on new criteria
best_parameters = mcmc_results.loc[mcmc_results["new_criteria"].idxmax()]
print("Best Parameters based on New Criteria:")
for param, value in best_parameters.items():
    if param in ["prior", "likelihood", "posterior", "new_criteria"]:
        print(f"{param}: {value}")
# print kinetic parameters
print("Kinetic Parameters based on New Criteria:")
for param in mcmc_results.columns:
    if "k_" in param or "conc_" in param or "K_" in param:
        print(f"{param}: {best_parameters[param]}")


# best_parameters = mcmc_results[(mcmc_results["iteration"]==31954) & (mcmc_results["walker"]==0) & (mcmc_results["chain"]==0)].iloc[0]
#
# # print("Best Parameters based on Iteration 31954, Walker 0, Chain 0:")
# print("Best Parameters based on Iteration 31954, Walker 0, Chain 0:")
# for param, value in best_parameters.items():
# 	if param in ['prior', 'likelihood', 'posterior']:
# 		print(f"{param}: {value}")

circuit_manager = CircuitManager(
    parameters_file="../../data/prior/model_parameters_priors_updated_tighter.csv",
    json_file="../../data/circuits/circuits.json",
)

model_priors = pd.read_csv(
    "../../data/prior/model_parameters_priors_updated_tighter.csv"
)
parameters_to_fit = model_priors[
    model_priors["Parameter"] != "k_prot_deg"
].Parameter.tolist()

calibration_parameters = setup_calibration()

parameters_to_fit = [p for p in parameters_to_fit if p in best_parameters.index]
circuit_name = "constitutive sfGFP"
time_bounds_max = 210
time_bounds_min = 30

# Create circuit configuration
circuit_configuration, circuit_fitter = create_circuit_simulation_data(
    circuit_name,
    parameters_to_fit,
    circuit_manager,
    calibration_parameters,
    time_bounds_max,
    time_bounds_min,
)
log_parameters = best_parameters[parameters_to_fit].values
log_parameters = log_parameters.reshape(1, -1)

simulation_results = circuit_fitter.simulate_parameters(log_parameters)
log_likelihoods = circuit_fitter.calculate_likelihood_from_simulation_with_breakdown(
    simulation_results
)
log_priors = circuit_fitter.calculate_log_prior(log_parameters)
organized_results = organize_results(
    parameters_to_fit,
    log_parameters,
    log_likelihoods,
    log_priors,  # Pass log space parameters
)


trajectory_records = extract_trajectory_data(simulation_results, organized_results)

trajectory_records["protein_au"] = convert_nm_to_au(
    trajectory_records["protein_concentration"],
    circuit_configuration.calibration_params["slope"],
    circuit_configuration.calibration_params["intercept"],
    circuit_configuration.calibration_params["brightness_correction"],
)

# add zeros for 30 first minutes. how much is dt, then how many steps for 30 minutes?
dt = circuit_configuration.tspan[1] - circuit_configuration.tspan[0]
num_steps = int(time_bounds_min / dt) + 1

# let's create dataframe with zeros for first 30 minutes and then append the trajectory records
# results should look like
#          time    sfGFP 3 nM  sfGFP 3 nM  sfGFP 3 nM
# 0         0  0.000000e+00  0.000000e+00  0.000000e+00
# 1    2  0.000000e+00  0.000000e+00  0.000000e+00
# 2    4  0.000000e+00  0.000000e+00  0.000000e+00
# 3    6  0.000000e+00  0.000000e+00  0.000000e+00
# 4    8  0.000000e+00  0.000000e+00  0.000000e+00

zeros_df = pd.DataFrame(
    {
        "time": [i * dt for i in range(num_steps)],
        "protein_au": [0.0] * num_steps,
    }
)
trajectory_records["time"] = (
    trajectory_records["time"] + num_steps * dt
)  # Shift time to start after zeros
trajectory_records = trajectory_records[trajectory_records["condition"] == "sfGFP 3 nM"]
trajectory_records = trajectory_records[["time", "protein_au"]]


# concatenate zeros_df and trajectory_records
trajectory_records = pd.concat([zeros_df, trajectory_records], ignore_index=True)


# convert time writing from integrers 0, 2, .. .to "0 h", "0h 2 min", "0h 4 min", ...
def format_time(t):
    hours = int(t // 60)
    minutes = int(t % 60)
    return f"{hours} h {minutes} min"


trajectory_records["time"] = trajectory_records["time"].apply(format_time)

# triple the column "protein_au" to match the original format
# rename the column to "sfGFP 3 nM", "sfGFP 3 nM", "sfGFP 3 nM"
trajectory_records = trajectory_records.rename(columns={"protein_au": "sfGFP 3 nM"})
# repeat the column three times
trajectory_records = pd.concat([trajectory_records] * 3, axis=1)

# save
trajectory_records.to_csv(
    f"{parameter_sim_folder}/constitutive_sfGFP_simulated_data_au_2.csv", index=False
)
# save parameters
best_parameters.to_csv(
    f"{parameter_sim_folder}/constitutive_sfGFP_simulated_parameters_2.csv", index=True
)
