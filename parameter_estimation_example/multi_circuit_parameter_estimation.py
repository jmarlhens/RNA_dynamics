from datetime import datetime

from likelihood_functions import (
    CircuitConfig,
    CircuitFitter,
)
from likelihood_functions.visualization import plot_simulation_results, plot_all_simulation_results
from likelihood_functions.utils import organize_results
from likelihood_functions.base import MCMCAdapter
from likelihood_functions.mcmc_analysis import analyze_mcmc_results
from circuits.toehold import test_toehold
from circuits.star import test_star
from circuits.GFP_positive_control import test_pos_control_constant
from circuits.cascade import test_cascade
from circuits.cffl_type_1 import test_coherent_feed_forward_loop
from utils.import_and_visualise_data import load_and_process_csv, plot_replicates
import pandas as pd
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor
import matplotlib.pyplot as plt

import cProfile

# Parameters
max_time = 360

# Load data
toehold_trigger_data, tspan_toehold = load_and_process_csv('../data/data_parameter_estimation/toehold_trigger.csv')
sense_star_data, tspan_star = load_and_process_csv('../data/data_parameter_estimation/sense_star.csv')
positive_control_data, tspan_positive_control = load_and_process_csv(
    '../data/data_parameter_estimation/positive_control_sfGFP.csv')
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
        condition_params={"To3 3 nM + Se6Tr3P 5 nM + St6 15 nM": {"k_Toehold3_GFP_concentration": 3,
                                                                  "k_Sense6_Trigger3_concentration": 5,
                                                                  "k_Star6_concentration": 15},
                          "To3 3 nM + Se6Tr3P 5 nM + St6 10 nM": {"k_Toehold3_GFP_concentration": 3,
                                                                  "k_Sense6_Trigger3_concentration": 5,
                                                                  "k_Star6_concentration": 10},
                          "To3 3 nM + Se6Tr3P 5 nM + St6 5 nM": {"k_Toehold3_GFP_concentration": 3,
                                                                 "k_Sense6_Trigger3_concentration": 5,
                                                                 "k_Star6_concentration": 5},
                          "To3 3 nM + Se6Tr3P 5 nM + St6 3 nM": {"k_Toehold3_GFP_concentration": 3,
                                                                 "k_Sense6_Trigger3_concentration": 5,
                                                                 "k_Star6_concentration": 3},
                          "To3 3 nM + Se6Tr3P 5 nM + St6 0 nM": {"k_Toehold3_GFP_concentration": 3,
                                                                 "k_Sense6_Trigger3_concentration": 5,
                                                                 "k_Star6_concentration": 0}},
        experimental_data=cascade_data,
        tspan=tspan_cascade,
        max_time=max_time
    ),
    CircuitConfig(
        model=cffl_type_1_model,
        name="CFFL Type 1",
        condition_params={"15 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 15,
                                    "k_Sense6_Toehold3_GFP_concentration": 3},
                          "12 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 12,
                                    "k_Sense6_Toehold3_GFP_concentration": 3},
                          "10 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 10,
                                    "k_Sense6_Toehold3_GFP_concentration": 3},
                          "7 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 7,
                                   "k_Sense6_Toehold3_GFP_concentration": 3},
                          "5 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 5,
                                   "k_Sense6_Toehold3_GFP_concentration": 3},
                          "3 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 3,
                                   "k_Sense6_Toehold3_GFP_concentration": 3},
                          "0 nM": {"k_Sense6_Trigger3_concentration": 5, "k_Star6_concentration": 0,
                                   "k_Sense6_Toehold3_GFP_concentration": 3}},
        experimental_data=cffl_type_1_data,
        tspan=tspan_cffl_type_1,
        max_time=max_time
    )
]

if __name__ == '__main__':
    profiler = cProfile.Profile()

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
    n_sets = 60

    # Create fitter
    circuit_fitter = CircuitFitter(circuit_configs, parameters_to_fit, priors, calibration_params)

    # # Generate test parameters (in log space)
    # log_params = circuit_fitter.generate_test_parameters(n_sets=n_sets)
    #
    # # Run simulations (takes log params)
    # import time
    #
    # tic = time.time()
    # sim_data = circuit_fitter.simulate_parameters(log_params)
    # toc = time.time()
    # simulation_time_process = toc - tic
    # print(f"Simulation time: {simulation_time_process:.2f} seconds")
    #
    # # Calculate likelihood from simulation data
    # log_likelihood = circuit_fitter.calculate_likelihood_from_simulation(sim_data)
    #
    # # Calculate prior (takes log params)
    # log_prior = circuit_fitter.calculate_log_prior(log_params)
    #
    # # Calculate posterior (takes log params)
    # log_posterior = log_prior + log_likelihood['total']
    #
    # # Organize results
    # results_df = organize_results(parameters_to_fit, log_params, log_likelihood, log_prior)
    #
    # plot_all_simulation_results(sim_data, results_df, ll_quartile=10)
    # plt.show()
    #
    # # Plot results
    # for i in range(min(n_sets, 6)):
    #     fig = plot_simulation_results(sim_data, results_df, param_set_idx=i)

    # Create the adapter
    adapter = MCMCAdapter(circuit_fitter)

    # Get initial parameters from prior means
    initial_parameters = adapter.get_initial_parameters()

    # Setup parallel tempering
    pt = adapter.setup_parallel_tempering(n_walkers=10, n_chains=6)


    profiler.enable()
    # Run sampling with initial parameters from priors
    parameters, priors, likelihoods, step_accepts, swap_accepts = pt.run(
        initial_parameters=initial_parameters,
        n_samples=2, #10 ** 2,
        target_acceptance_ratio=0.4,
        adaptive_temperature=True
    )

    profiler.disable()

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    profiler.dump_stats(f"profiling_{timestamp}.prof")
    profiler.print_stats()

    # parameters.shape is (10, 10, 6, 20)
    # likelihoods.shape is (10, 10, 6)
    # priors.shape is (10, 10, 6) and so on

    # Analyze results
    results = analyze_mcmc_results(
        parameters=parameters,
        priors=priors,
        likelihoods=likelihoods,
        step_accepts=step_accepts,
        swap_accepts=swap_accepts,
        parameter_names=circuit_fitter.parameters_to_fit,
        circuit_fitter=circuit_fitter  # Pass the circuit_fitter instance
    )

    # Access individual components
    best_params = results['best_parameters']
    stats = results['statistics']
    figures = results['figures']

    # Display figures
    plt.show()

    # simulate the best parameters
    best_params = best_params['parameters']

    # Summarize results in a DataFrame
    df = results['analyzer'].to_dataframe()

    # Save results
    df.to_csv('results.csv', index=False)

    # Distribution of likelihoods
    plt.figure(figsize=(10, 6))
    plt.hist(likelihoods.flatten(), color='blue', alpha=0.7)
    plt.xlabel('Log Likelihood')
    plt.ylabel('Frequency')
    plt.title('Distribution of Log Likelihoods')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
