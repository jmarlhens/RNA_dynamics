import os
from simulations_and_analysis.individual.individual_circuits_simulations import (
    load_individual_circuit_results,
    plot_fits,
)

subfolder = "/conv_AU_corr"


input_directory = "../../data/fit_data/individual_circuits" + subfolder
output_visualization_directory = "../../figures_paper/figure_individual_circuits"
priors_csv_path = "../../data/prior/model_parameters_priors_updated_tighter.csv"

# creatre output directory if it does not exist
os.makedirs(output_visualization_directory, exist_ok=True)

mcmc_results = load_individual_circuit_results(input_directory)

# Define processing order and inclusion
circuit_processing_sequence = [
    # "constitutive sfGFP",
    "sense_star_6",
    "toehold_trigger",
    "cascade",
    "cffl_type_1",
    "or_gate_c1ffl",
    "star_antistar_1",
    "trigger_antitrigger",
    "inhibited_incoherent_cascade",
    "inhibited_cascade",
    "cffl_12",
    "iffl_1",
]

filtered_mcmc_results = {
    circuit_name: mcmc_results[circuit_name]
    for circuit_name in circuit_processing_sequence
    if circuit_name in mcmc_results
}

# Generate combined plots (existing functionality)
plot_fits(
    filtered_mcmc_results,
    output_visualization_directory,
    sample_count=10,
    time_bounds_max=130,
    time_bounds_min=30,
    priors_csv_path=priors_csv_path,
)
