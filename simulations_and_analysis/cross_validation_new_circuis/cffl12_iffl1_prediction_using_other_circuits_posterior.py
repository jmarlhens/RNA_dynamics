import os
import pandas as pd
import matplotlib.pyplot as plt
from circuits.circuit_generation.circuit_manager import CircuitManager
from utils.GFP_calibration import setup_calibration
from analysis_and_figures.mcmc_analysis_hierarchical import process_mcmc_data
from analysis_and_figures.plots_simulation import (
    plot_circuit_simulations,
    plot_circuit_conditions_overlay,
)

# from simulations_and_analysis.individual.individual_circuits_statistics import (
#     load_individual_circuit_results,
# )
from simulations_and_analysis.individual.individual_circuits_simulations import (
    create_circuit_simulation_data,
    plot_individual_circuit,
    simulate_and_organize_parameter_sets,
)

# priors
priors_csv_path = "../../data/prior/model_parameters_priors_092025_correction.csv"

# prediction based on another circuits' posterior
subfolder_data = "/shared_parameters/results_star_antistar_1_and_trigger_antitrigger"
results_filename = "results_star_antistar_1_and_trigger_antitrigger_20251116_023618.csv"
subfolder_vis_output = "/transfer_learning/data_informed_prior"

# transferred parameters from informed prior
# subfolder_data = "/individual_circuits/transfer_learning/data_informed_prior"
# results_filename = (
# 	# "results_cffl_12_20251117_102026_informed_prior.csv"
# 	# "results_iffl_1_20251117_161335_informed_prior.csv"
# 	# "results_iffl_1_20251117_213326_informed_prior.csv"
# 	"results_iffl_1_20251117_234507_informed_prior.csv"
# )
# subfolder_vis_output = "/transfer_learning/data_informed_prior"
#


individual_results_directory = "../../data/fit_data" + subfolder_data
results_filepath = f"{individual_results_directory}/{results_filename}"
prior_parameters_filepath = priors_csv_path
output_visualization_directory = "../../figures/individual_circuits"
output_directory = output_visualization_directory + subfolder_vis_output

print(f"loading individual circuit results from {results_filepath}...")
mcmc_results = {}
mcmc_raw_samples = pd.read_csv(results_filepath)
# mcmc_results = load_individual_circuit_results(individual_results_directory)

sample_count = 10
time_bounds_max = 210
time_bounds_min = 30

circuit_manager = CircuitManager(
    parameters_file=priors_csv_path,
    json_file="../../data/circuits/circuits.json",
)

# model_priors = pd.read_csv(priors_csv_path)

calibration_parameters = setup_calibration()

combined_random_simulation_data = {}
combined_random_results = []

for circuit_name in [
    # "trigger_antitrigger",
    # "star_antistar_1",
    "cffl_12",
    "iffl_1",
]:
    print(f"Processing circuit {circuit_name}")

    # Filter and sample MCMC data
    mcmc_processed = process_mcmc_data(mcmc_raw_samples, burn_in=0.4, chain_idx=0)
    mcmc_filtered_samples = mcmc_processed["processed_data"]

    print(
        f"{circuit_name}: {len(mcmc_raw_samples)} → {len(mcmc_filtered_samples)} samples after burn-in"
    )

    final_sample_size = min(sample_count, len(mcmc_filtered_samples))
    mcmc_final_samples = (
        mcmc_filtered_samples.sample(n=final_sample_size, random_state=42)
        if len(mcmc_filtered_samples) > final_sample_size
        else mcmc_filtered_samples.copy()
    )

    random_samples = mcmc_final_samples.sample(n=final_sample_size, random_state=42)

    # Create circuit configuration
    # Create circuit configuration
    circuit_configuration, circuit_fitter = create_circuit_simulation_data(
        circuit_name,
        circuit_manager,
        calibration_parameters,
        time_bounds_max,
        time_bounds_min,
    )

    plot_individual_circuit(
        random_samples,
        "random",
        circuit_name,
        circuit_fitter,
        circuit_fitter.parameters_to_fit,
        output_directory,
    )

    random_simulation_data, random_results_dataframe = (
        simulate_and_organize_parameter_sets(
            random_samples,
            circuit_fitter,
            circuit_fitter.parameters_to_fit,
        )
    )

    combined_random_simulation_data[circuit_name] = {
        "config": circuit_configuration,
        "combined_params": random_simulation_data["combined_params"],
        "simulation_results": random_simulation_data["simulation_results"],
    }

    combined_random_results.append(random_results_dataframe)

# Generate combined plots
combined_random_dataframe = pd.concat(combined_random_results, ignore_index=True)

print("Generating combined random fits figure...")
plot_circuit_simulations(
    combined_random_simulation_data,
    combined_random_dataframe,
    plot_mode="individual",
    likelihood_percentile_range=20,
    # show_title=False,
)
plt.savefig(
    os.path.join(output_directory, "all_circuits_random_fits.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.close()

plot_circuit_simulations(
    combined_random_simulation_data,
    combined_random_dataframe,
    plot_mode="summary",
    summary_type="median_iqr",
    percentile_bounds=(10, 90),
    show_title=False,
)
plt.savefig(
    os.path.join(output_directory, "all_circuits_random_fits_summary.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.close()

plot_circuit_conditions_overlay(
    combined_random_simulation_data,
    combined_random_dataframe,
    simulation_mode="individual",
    show_title=False,
)
plt.savefig(
    os.path.join(output_directory, "all_circuits_random_fits_overlay_individual.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.close()

plot_circuit_conditions_overlay(
    combined_random_simulation_data,
    combined_random_dataframe,
    simulation_mode="summary",
    summary_type="median_iqr",
    percentile_bounds=(10, 90),
    show_title=False,
)
plt.savefig(
    os.path.join(output_directory, "all_circuits_random_fits_overlay_summary.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.close()
