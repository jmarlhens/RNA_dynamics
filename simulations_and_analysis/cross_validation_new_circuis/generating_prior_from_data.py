import pandas as pd
import os
import numpy as np
from scipy.stats import multivariate_normal

from simulations_and_analysis.individual.individual_circuits_statistics import (
    generate_prior_mean_coordinates,
    convert_individual_to_theta_format,
)
import matplotlib.pyplot as plt
from analysis_and_figures.hierarchical_pairplot_analysis import (
    create_circuit_prior_comparison_pairplot,
)
from circuits.circuit_generation.circuit_manager import CircuitManager
from simulations_and_analysis.individual.individual_circuits_simulations import (
    create_circuit_simulation_data,
    simulate_and_organize_parameter_sets,
)
from analysis_and_figures.plots_simulation import (
    extract_trajectory_data,
    plot_single_circuit_two_column,
)
from utils.GFP_calibration import setup_calibration


# Configuration
subfolder = "/shared_parameters/star_antistar_trigger_antitrigger"
individual_results_directory = "../../data/fit_data" + subfolder
file = "results_star_antistar_1_and_trigger_antitrigger_20250829_143823.csv"
filepath = f"{individual_results_directory}/{file}"
# individual_results_directory = "../../data/fit_data/individual_circuits" + subfolder
prior_parameters_filepath = (
    "../../data/prior/model_parameters_priors_updated_tighter.csv"
)
burn_in_fraction = 0.5
output_visualization_directory = "../../figures/calibrated_prior" + subfolder

os.makedirs(output_visualization_directory, exist_ok=True)


# Compute posterior samples after burn-in
prior_parameters = pd.read_csv(prior_parameters_filepath)
circuit_results = pd.read_csv(filepath)
excluded = {
    "iteration",
    "walker",
    "chain",
    "likelihood",
    "prior",
    "posterior",
    "step_accepted",
}
parameter_names = circuit_results.columns.difference(excluded).tolist()
samples_posterior_processed = convert_individual_to_theta_format(
    {"trigger_antitrigger_and_star_antistar": circuit_results},
    parameter_names,
    ["trigger_antitrigger_and_star_antistar"],
)
prior_mean_coordinates = generate_prior_mean_coordinates(
    prior_parameters_filepath, parameter_names
)
# estimate mean and covarianceof posterior samples
posterior_mean_coordinates = (
    samples_posterior_processed[parameter_names].mean().to_dict()
)
posterior_covariance = samples_posterior_processed[parameter_names].cov().to_dict()
prior_and_posterior = pd.concat(
    [samples_posterior_processed, prior_mean_coordinates], ignore_index=True
)


# Visualization
parameter_names_for_plot = [
    "K_tx",
    "k_tx",
    "K_tl",
    "k_tl",
]
columns_for_plot = parameter_names_for_plot + [
    "type",
    "Circuit",
]
columns_for_plot += [
    f"{param}_log10stdev"
    for param in parameter_names_for_plot
    if f"{param}_log10stdev" in prior_and_posterior.columns
]

pairplot_figure = create_circuit_prior_comparison_pairplot(
    prior_and_posterior[columns_for_plot],
    parameter_names_for_plot,
    output_visualization_directory,
    diagonal_visualization_type="hist",
)
# add on top the estimated gaussian using estimated mean and covariance
for row_param_index, row_parameter_name in enumerate(parameter_names_for_plot):
    for col_param_index, col_parameter_name in enumerate(parameter_names_for_plot):
        if row_param_index != col_param_index:
            ax = pairplot_figure.axes[row_param_index, col_param_index]
            # create grid
            x = np.linspace(
                samples_posterior_processed[col_parameter_name].min(),
                samples_posterior_processed[col_parameter_name].max(),
                100,
            )
            y = np.linspace(
                samples_posterior_processed[row_parameter_name].min(),
                samples_posterior_processed[row_parameter_name].max(),
                100,
            )
            X, Y = np.meshgrid(x, y)
            pos = np.dstack((X, Y))
            mean = [
                posterior_mean_coordinates[col_parameter_name],
                posterior_mean_coordinates[row_parameter_name],
            ]
            cov = [
                [
                    posterior_covariance[col_parameter_name][col_parameter_name],
                    posterior_covariance[col_parameter_name][row_parameter_name],
                ],
                [
                    posterior_covariance[row_parameter_name][col_parameter_name],
                    posterior_covariance[row_parameter_name][row_parameter_name],
                ],
            ]

            rv = multivariate_normal(mean, cov)
            Z = rv.pdf(pos)
            ax.contour(X, Y, Z, colors="yellow", alpha=0.4, levels=20, linewidths=1)

plt.show()

pairplot_figure.savefig(
    f"{output_visualization_directory}/posterior_sampled_and_approximated_pairplot.png",
    dpi=300,
)

# create a random multivariate normal sample from the estimated posterior
# first create the multivariate gaussian with the computed mean and covariance
mv_normal = multivariate_normal(
    mean=[posterior_mean_coordinates[param] for param in parameter_names],
    cov=[
        [posterior_covariance[i][j] for j in parameter_names] for i in parameter_names
    ],
)
# sample from the multivariate normal
num_samples = 500
final_sample_size = 50

mv_samples = mv_normal.rvs(size=num_samples)
# transform into a dataframe
mv_samples_df = pd.DataFrame(mv_samples, columns=parameter_names)
# sample from there
random_samples = mv_samples_df.sample(n=final_sample_size, random_state=42)

fitted_parameter_names = parameter_names
time_bounds_max = 130
time_bounds_min = 30
random_samples = samples_posterior_processed.sample(
    n=final_sample_size, random_state=42
)
calibration_parameters = setup_calibration()

circuit_manager = CircuitManager(
    parameters_file=prior_parameters_filepath,
    json_file="../../data/circuits/circuits.json",
)

circuit_names = ["trigger_antitrigger", "star_antistar_1"]

for circuit_name in circuit_names:
    circuit_configuration, circuit_fitter = create_circuit_simulation_data(
        circuit_name,
        fitted_parameter_names,
        circuit_manager,
        calibration_parameters,
        time_bounds_max,
        time_bounds_min,
    )

    simulation_data, results_dataframe = simulate_and_organize_parameter_sets(
        random_samples, circuit_fitter, parameter_names
    )

    # Prepare single-circuit data structure
    single_circuit_simulation_dict = {
        circuit_name: {
            "config": circuit_configuration,
            "combined_params": simulation_data["combined_params"],
            "simulation_results": simulation_data["simulation_results"],
        }
    }

    trajectory_data = extract_trajectory_data(
        single_circuit_simulation_dict, results_dataframe
    )
    circuit_trajectory_data = trajectory_data[
        trajectory_data["circuit"] == circuit_name
    ]
    circuit_data = single_circuit_simulation_dict[circuit_name]

    # Generate two-column plots (experimental | simulation)
    for simulation_mode in ["individual", "summary"]:
        _ = plot_single_circuit_two_column(
            circuit_name,
            circuit_data,
            circuit_trajectory_data,
            results_dataframe,
            simulation_mode=simulation_mode,
            summary_type="median_iqr",
            percentile_bounds=(10, 90),
        )

        mode_suffix = "_summary" if simulation_mode == "summary" else "_individual"
        two_column_filename = f"individual_{circuit_name}_two_column{mode_suffix}.png"
        plt.savefig(
            os.path.join(output_visualization_directory, two_column_filename),
            bbox_inches="tight",
            dpi=300,
        )
        plt.show()
        plt.close()
