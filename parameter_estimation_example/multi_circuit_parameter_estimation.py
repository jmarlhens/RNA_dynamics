from likelihood_functions import (
    CircuitConfig,
    CircuitFitter,
)
from likelihood_functions.visualization import plot_simulation_results
from likelihood_functions.utils import organize_results
from circuits.toehold import test_toehold
from circuits.star import test_star
from circuits.GFP_positive_control import test_pos_control_constant
from circuits.cascade import test_cascade
from circuits.cffl_type_1 import test_coherent_feed_forward_loop
from utils.import_and_visualise_data import load_and_process_csv, plot_replicates
import pandas as pd
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor

# Parameters
max_time = 360

# Load data
toehold_trigger_data, tspan_toehold = load_and_process_csv('../data/data_parameter_estimation/toehold_trigger.csv')
sense_star_data, tspan_star = load_and_process_csv('../data/data_parameter_estimation/sense_star.csv')
positive_control_data, tspan_positive_control = load_and_process_csv('../data/data_parameter_estimation/positive_control_sfGFP.csv')
cascade_data, tspan_cascade = load_and_process_csv('../data/data_parameter_estimation/cascade.csv')
cffl_type_1_data, tspan_cffl_type_1 = load_and_process_csv('../data/data_parameter_estimation/c1_ffl_and.csv')
# plot_replicates(toehold_trigger_data, "Toehold Trigger")
# plot_replicates(sense_star_data, "Sense Star")
# plot_replicates(positive_control_data, "Positive Control")
# plot_replicates(cascade_data, "Cascade")
# plot_replicates(cffl_type_1_data, "CFFL Type 1")

# Load models
toehold_model = test_toehold()
sense_model = test_star()
gfp_pos_control_model = test_pos_control_constant()
cascade_model = test_cascade()
cffl_type_1_model = test_coherent_feed_forward_loop()

# Create configs
circuit_configs = [
    CircuitConfig(
        model=gfp_pos_control_model,
        name="Positive Control (sfGFP)",
        condition_params={"sfGFP 3 nM + Se6Tr3 5 nM + St6 15 nM": {"k_GFP_concentration": 3}},
        experimental_data=positive_control_data,
        tspan=tspan_positive_control,
        max_time=max_time
    ),
    CircuitConfig(
        model=toehold_model,
        name="Toehold/Trigger",
        condition_params={"To3 5 + Tr3 5": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 5},
                        "To3 5 + Tr3 4": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 4},
                        "To3 5 + Tr3 3": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 3},
                        "To3 5 + Tr3 2": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 2},
                        "To3 5 + Tr3 1": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 1},
                        "To3 5": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 0}},
        experimental_data=toehold_trigger_data,
        tspan=tspan_toehold,
        max_time=max_time
    ),
    CircuitConfig(
        model=sense_model,
        name="Sense/Star",
        condition_params={"Se6 5 nM + St6 15 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 15},
                        "Se6 5 nM + St6 10 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 10},
                        "Se6 5 nM + St6 5 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 5},
                        "Se6 5 nM + St6 3 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 3},
                        "Se6 5 nM + St6 0 nM": {"k_Sense6_GFP_concentration": 0, "k_Star6_concentration": 0}},
        experimental_data=sense_star_data,
        tspan=tspan_star,
        max_time=max_time
    ),
    CircuitConfig(
        model=cascade_model,
        name="Cascade",
        condition_params={"To3 3 nM + Se6Tr3P 5 nM + St6 15 nM": {"k_Toehold3_GFP_concentration": 3, "k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 15},
        "To3 3 nM + Se6Tr3P 5 nM + St6 10 nM": {"k_Toehold3_GFP_concentration": 3, "k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 10},
        "To3 3 nM + Se6Tr3P 5 nM + St6 5 nM": {"k_Toehold3_GFP_concentration": 3, "k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 5},
        "To3 3 nM + Se6Tr3P 5 nM + St6 3 nM": {"k_Toehold3_GFP_concentration": 3, "k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 3},
        "To3 3 nM + Se6Tr3P 5 nM + St6 0 nM": {"k_Toehold3_GFP_concentration": 3, "k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 0}},
        experimental_data=cascade_data,
        tspan=tspan_cascade,
        max_time=max_time
    ),
    CircuitConfig(
        model=cffl_type_1_model,
        name="CFFL Type 1",
        condition_params={"15 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 15, "k_Sense6_Toehold3_GFP_concentration": 3},
        "12 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 12, "k_Sense6_Toehold3_GFP_concentration": 3},
        "10 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 10, "k_Sense6_Toehold3_GFP_concentration": 3},
        "7 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 7, "k_Sense6_Toehold3_GFP_concentration": 3},
        "5 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 5, "k_Sense6_Toehold3_GFP_concentration": 3},
        "3 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 3, "k_Sense6_Toehold3_GFP_concentration": 3},
        "0 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 0, "k_Sense6_Toehold3_GFP_concentration": 3}},
        experimental_data=cffl_type_1_data,
        tspan=tspan_cffl_type_1,
        max_time=max_time
    )
]

# Load your calibration data
data = pd.read_csv('../calibration_gfp/gfp_Calibration.csv')

# Fit the calibration curve
calibration_results = fit_gfp_calibration(
    data,
    concentration_col='GFP Concentration (nM)',
    fluorescence_pattern='F.I. (a.u)'
)

# Get the correction factor for sfGFP
correction_factor, protein_info = get_brightness_correction_factor('avGFP', 'sfGFP')

# Create calibration parameters dictionary
calibration_params = {
    'slope': calibration_results['slope'],
    'intercept': calibration_results['intercept'],
    'brightness_correction': correction_factor
}

# Load priors
priors = pd.read_csv('../data/model_parameters_priors.csv')
priors = priors[priors['Parameter'] != 'k_prot_deg']
parameters_to_fit = priors.Parameter.tolist()
n_sets=10

# Create fitter
circuit_fitter = CircuitFitter(circuit_configs, parameters_to_fit, priors, calibration_params)

# Generate test parameters (in log space)
log_params = circuit_fitter.generate_test_parameters(n_sets=n_sets)

# Run simulations (takes log params)
sim_data = circuit_fitter.simulate_parameters(log_params)

# Plot results
for i in range(n_sets):
    fig = plot_simulation_results(sim_data, param_set_idx=i)

# Calculate likelihood from simulation data
log_likelihood = circuit_fitter.calculate_likelihood_from_simulation(sim_data)

# Calculate prior (takes log params)
log_prior = circuit_fitter.calculate_log_prior(log_params)

# Calculate posterior (takes log params)
log_posterior = log_prior + log_likelihood['total']

results_df = organize_results(parameters_to_fit, log_params, log_likelihood, log_prior)

results_df.head()