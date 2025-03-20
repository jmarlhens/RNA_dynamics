from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from likelihood_functions import CircuitConfig, CircuitFitter
from likelihood_functions.utils import organize_results
from likelihood_functions.visualization import plot_all_simulation_results
from likelihood_functions.base import MCMCAdapter
from likelihood_functions.mcmc_analysis import analyze_mcmc_results
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor


def setup_calibration():
    # Load calibration data
    data = pd.read_csv('../calibration_gfp/gfp_Calibration.csv')

    # Fit the calibration curve
    calibration_results = fit_gfp_calibration(
        data,
        concentration_col='GFP Concentration (nM)',
        fluorescence_pattern='F.I. (a.u)'
    )

    # Get correction factor
    correction_factor, _ = get_brightness_correction_factor('avGFP', 'sfGFP')

    return {
        'slope': calibration_results['slope'],
        'intercept': calibration_results['intercept'],
        'brightness_correction': correction_factor
    }


def fit_single_circuit(circuit_config, priors, max_time=360, n_samples=1000, n_walkers=10, n_chains=6):
    """
    Fit a single circuit and save its results
    """
    # Create circuit fitter with single config
    parameters_to_fit = priors.Parameter.tolist()
    calibration_params = setup_calibration()
    circuit_fitter = CircuitFitter([circuit_config], parameters_to_fit, priors, calibration_params)

    # Create MCMC adapter
    adapter = MCMCAdapter(circuit_fitter)
    initial_parameters = adapter.get_initial_parameters()

    # Setup and run parallel tempering
    pt = adapter.setup_parallel_tempering(n_walkers=n_walkers, n_chains=n_chains)
    parameters, priors_out, likelihoods, step_accepts, swap_accepts = pt.run(
        initial_parameters=initial_parameters,
        n_samples=n_samples,
        target_acceptance_ratio=0.4,
        adaptive_temperature=True
    )

    # Analyze results
    results = analyze_mcmc_results(
        parameters=parameters,
        priors=priors_out,
        likelihoods=likelihoods,
        step_accepts=step_accepts,
        swap_accepts=swap_accepts,
        parameter_names=circuit_fitter.parameters_to_fit,
        circuit_fitter=circuit_fitter
    )

    # Save results
    df = results['analyzer'].to_dataframe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{circuit_config.name.lower().replace('/', '_')}_{timestamp}.csv"
    df.to_csv(filename, index=False)

    # Plot and save best fit results
    best_params = df.sort_values(by='likelihood', ascending=False).head(100)
    best_params_values = best_params[results['analyzer'].parameter_names].values
    sim_data = circuit_fitter.simulate_parameters(best_params_values)
    log_likelihood = circuit_fitter.calculate_likelihood_from_simulation(sim_data)
    log_prior = circuit_fitter.calculate_log_prior(best_params_values)
    results_df = organize_results(parameters_to_fit, best_params_values, log_likelihood, log_prior)

    plt.figure(figsize=(12, 8))
    plot_all_simulation_results(sim_data, results_df, ll_quartile=20)
    plt.savefig(f"fit_{circuit_config.name.lower().replace('/', '_')}_{timestamp}.png")
    plt.close()

    return results, df


def main():
    # Load all required models
    from obsolete.star import test_star
    from obsolete.cascade import test_cascade
    from obsolete.cffl_type_1 import test_coherent_feed_forward_loop

    # Load data
    max_time = 360
    toehold_trigger_data, tspan_toehold = load_and_process_csv('../data/data_parameter_estimation/toehold_trigger.csv')
    sense_star_data, tspan_star = load_and_process_csv('../data/data_parameter_estimation/sense_star.csv')
    cascade_data, tspan_cascade = load_and_process_csv('../data/data_parameter_estimation/cascade.csv')
    cffl_type_1_data, tspan_cffl_type_1 = load_and_process_csv('../data/data_parameter_estimation/c1_ffl_and.csv')

    # Create individual configs
    circuit_configs = {
        # "toehold": CircuitConfig(
        #     model=test_toehold(),
        #     name="Toehold/Trigger",
        #     condition_params={
        #         "To3 5 + Tr3 5": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 5},
        #         "To3 5 + Tr3 4": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 4},
        #         "To3 5 + Tr3 3": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 3},
        #         "To3 5 + Tr3 2": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 2},
        #         "To3 5 + Tr3 1": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 1},
        #         "To3 5": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 0}
        #     },
        #     experimental_data=toehold_trigger_data,
        #     tspan=tspan_toehold,
        #     max_time=max_time
        # ),
        "star": CircuitConfig(
            model=test_star(),
            name="Sense/Star",
            condition_params={
                "Se6 5 nM + St6 15 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 15},
                "Se6 5 nM + St6 10 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 10},
                "Se6 5 nM + St6 5 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 5},
                "Se6 5 nM + St6 3 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 3},
                "Se6 5 nM + St6 0 nM": {"k_Sense6_GFP_concentration": 0, "k_Star6_concentration": 0}
            },
            experimental_data=sense_star_data,
            tspan=tspan_star,
            max_time=max_time
        ),
        "cascade": CircuitConfig(
            model=test_cascade(),
            name="Cascade",
            condition_params={
                "To3 3 nM + Se6Tr3P 5 nM + St6 15 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 15
                },
                "To3 3 nM + Se6Tr3P 5 nM + St6 10 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 10
                },
                "To3 3 nM + Se6Tr3P 5 nM + St6 5 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 5
                },
                "To3 3 nM + Se6Tr3P 5 nM + St6 3 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 3
                },
                "To3 3 nM + Se6Tr3P 5 nM + St6 0 nM": {
                    "k_Toehold3_GFP_concentration": 3,
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 0
                }
            },
            experimental_data=cascade_data,
            tspan=tspan_cascade,
            max_time=max_time
        ),
        "cffl": CircuitConfig(
            model=test_coherent_feed_forward_loop(),
            name="CFFL Type 1",
            condition_params={
                "15 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 15,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "12 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 12,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "10 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 10,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "7 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 7,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "5 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 5,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "3 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 3,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                },
                "0 nM": {
                    "k_Sense6_Trigger3_concentration": 5,
                    "k_Star6_concentration": 0,
                    "k_Sense6_Toehold3_GFP_concentration": 3
                }
            },
            experimental_data=cffl_type_1_data,
            tspan=tspan_cffl_type_1,
            max_time=max_time
        )
    }

    # Load priors
    priors = pd.read_csv('../data/model_parameters_priors.csv')
    priors = priors[priors['Parameter'] != 'k_prot_deg']

    # Fit each circuit individually
    for circuit_name, config in circuit_configs.items():
        print(f"\nFitting {circuit_name}...")
        results, df = fit_single_circuit(config, priors)
        print(f"Completed fitting {circuit_name}")


if __name__ == '__main__':
    main()