import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from likelihood_functions import CircuitConfig, CircuitFitter
from likelihood_functions.utils import organize_results
from likelihood_functions.visualization import plot_all_simulation_results
from utils.import_and_visualise_data import load_and_process_csv
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor


def setup_calibration():
    """Set up GFP calibration parameters"""
    data = pd.read_csv('../calibration_gfp/gfp_Calibration.csv')
    calibration_results = fit_gfp_calibration(
        data,
        concentration_col='GFP Concentration (nM)',
        fluorescence_pattern='F.I. (a.u)'
    )
    correction_factor, _ = get_brightness_correction_factor('avGFP', 'sfGFP')

    return {
        'slope': calibration_results['slope'],
        'intercept': calibration_results['intercept'],
        'brightness_correction': correction_factor
    }


def load_circuit_results(results_dir='../fit_data/individual_circuits'):
    """Load all circuit results from CSV files"""
    results = {}
    # Find all results CSV files
    pattern = os.path.join(results_dir, 'results_cffl*.csv')
    result_files = glob.glob(pattern)

    for file_path in result_files:
        # Extract circuit name from filename
        filename = os.path.basename(file_path)
        circuit_name = filename.split('_')[1]  # Assumes format: results_circuitname_timestamp.csv

        # Load results
        df = pd.read_csv(file_path)
        results[circuit_name] = df

        print(f"Loaded {circuit_name} results from {filename}")
        print(f"Number of samples: {len(df)}")
        print(f"Best likelihood: {df['likelihood'].max():.2f}\n")

    return results


def plot_parameter_distributions(results, save_dir='.'):
    """Plot parameter distributions for all circuits"""
    for circuit_name, df in results.items():
        # Get parameter columns (exclude likelihood and other non-parameter columns)
        param_cols = [col for col in df.columns if col != 'likelihood']

        # Create parameter distribution plots
        n_params = len(param_cols)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        plt.figure(figsize=(15, 5 * n_rows))

        for i, param in enumerate(param_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data=df, x=param, bins=30)
            plt.title(f'{circuit_name} - {param}')
            plt.xlabel('Parameter Value')
            plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'parameter_distributions_{circuit_name}.png'))
        plt.close()


def plot_likelihood_distributions(results, save_dir='.'):
    """Plot likelihood distributions for all circuits"""
    plt.figure(figsize=(12, 6))

    for circuit_name, df in results.items():
        sns.kdeplot(data=df['likelihood'], label=circuit_name)

    plt.title('Likelihood Distributions Across Circuits')
    plt.xlabel('Log Likelihood')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'likelihood_distributions.png'))
    plt.close()


def plot_fits(results, save_dir='.', n_samples=400):
    """Plot fits for each circuit using both best and random samples"""
    # First, we need to recreate the circuit configurations and load the data
    from obsolete.toehold import test_toehold
    from obsolete.star import test_star
    from obsolete.cascade import test_cascade
    from obsolete.cffl_type_1 import test_coherent_feed_forward_loop

    # Load data
    max_time = 360
    data_files = {
        'toehold': '../data/data_parameter_estimation/toehold_trigger.csv',
        'sense': '../data/data_parameter_estimation/sense_star.csv',
        'cascade': '../data/data_parameter_estimation/cascade.csv',
        'cffl type 1': '../data/data_parameter_estimation/c1_ffl_and.csv'
    }

    # Load priors for parameter names
    priors = pd.read_csv('../data/model_parameters_priors.csv')
    priors = priors[priors['Parameter'] != 'k_prot_deg']
    parameters_to_fit = priors.Parameter.tolist()

    for circuit_name, df in results.items():
        # Load experimental data
        exp_data, tspan = load_and_process_csv(data_files[circuit_name])

        # Get the circuit configuration
        if circuit_name == 'toehold':
            model = test_toehold()
            conditions = {
                "To3 5 + Tr3 5": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 5},
                "To3 5 + Tr3 4": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 4},
                "To3 5 + Tr3 3": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 3},
                "To3 5 + Tr3 2": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 2},
                "To3 5 + Tr3 1": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 1},
                "To3 5": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 0}
            }
        elif circuit_name == 'sense':
            model = test_star()
            conditions = {
                "Se6 5 nM + St6 15 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 15},
                "Se6 5 nM + St6 10 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 10},
                "Se6 5 nM + St6 5 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 5},
                "Se6 5 nM + St6 3 nM": {"k_Sense6_GFP_concentration": 5, "k_Star6_concentration": 3},
                "Se6 5 nM + St6 0 nM": {"k_Sense6_GFP_concentration": 0, "k_Star6_concentration": 0}
            }
        elif circuit_name == 'cascade':
            model = test_cascade()
            conditions = {
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
            }
        else:  # cffl
            model = test_coherent_feed_forward_loop()
            conditions = {
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
            }

        config = CircuitConfig(
            model=model,
            name=circuit_name,
            condition_params=conditions,
            experimental_data=exp_data,
            tspan=tspan,
            max_time=max_time
        )

        # Create circuit fitter
        calibration_params = setup_calibration()
        circuit_fitter = CircuitFitter([config], parameters_to_fit, priors, calibration_params)

        # Get best parameters and simulate
        best_params = df.sort_values(by='likelihood', ascending=False).head(n_samples)
        best_params_values = best_params[parameters_to_fit].values
        sim_data_best = circuit_fitter.simulate_parameters(best_params_values)

        # Calculate likelihood and prior for best parameters
        log_likelihood_best = circuit_fitter.calculate_likelihood_from_simulation(sim_data_best)
        log_prior_best = circuit_fitter.calculate_log_prior(best_params_values)

        # Organize results and plot best fits
        results_df_best = organize_results(parameters_to_fit, best_params_values, log_likelihood_best, log_prior_best)

        plt.figure(figsize=(12, 8))
        plot_all_simulation_results(sim_data_best, results_df_best, ll_quartile=20)
        plt.suptitle(f'Top {n_samples} Best Fits for {circuit_name}')
        plt.savefig(os.path.join(save_dir, f'best_fits_{circuit_name}.png'))
        plt.close()

        # Get random parameters and simulate
        random_params = df.sample(n=n_samples)
        random_params_values = random_params[parameters_to_fit].values
        sim_data_random = circuit_fitter.simulate_parameters(random_params_values)

        # Calculate likelihood and prior for random parameters
        log_likelihood_random = circuit_fitter.calculate_likelihood_from_simulation(sim_data_random)
        log_prior_random = circuit_fitter.calculate_log_prior(random_params_values)

        # Organize results and plot random fits
        results_df_random = organize_results(parameters_to_fit, random_params_values, log_likelihood_random,
                                             log_prior_random)

        plt.figure(figsize=(12, 8))
        plot_all_simulation_results(sim_data_random, results_df_random, ll_quartile=20)
        plt.suptitle(f'{n_samples} Random Samples Fits for {circuit_name}')
        plt.savefig(os.path.join(save_dir, f'random_fits_{circuit_name}.png'))
        plt.close()


def main():
    """Main function to load and plot all results"""
    # Create output directory for plots
    output_dir = '../figures/analysis_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Load all results
    results = load_circuit_results()

    # Generate plots
    plot_parameter_distributions(results, output_dir)
    plot_likelihood_distributions(results, output_dir)
    plot_fits(results, output_dir, n_samples=200)


if __name__ == '__main__':
    main()