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
from circuits.circuit_manager import CircuitManager
from data.circuit_configs import get_circuit_conditions, get_data_file


def setup_calibration():
    """Set up GFP calibration parameters"""
    data = pd.read_csv('../utils/calibration_gfp/gfp_Calibration.csv')
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


def load_circuit_results(results_dir='../data/fit_data/individual_circuits'):
    """Load all circuit results from CSV files"""
    results = {}
    # Find all results CSV files
    pattern = os.path.join(results_dir, 'results_*.csv')
    result_files = glob.glob(pattern)

    for file_path in result_files:
        # Extract circuit name from filename
        filename = os.path.basename(file_path)

        circuit_name = filename.split('_')[1:-2]
        circuit_name = '_'.join(circuit_name)

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


def plot_fits(results, save_dir='.', n_samples=400, min_time=None):
    """Plot fits for each circuit using both best and random samples"""
    # Initialize CircuitManager and load requirements
    circuit_manager = CircuitManager(
        parameters_file="../data/model_parameters_priors.csv",
        json_file="../data/circuits/circuits.json"
    )
    priors = pd.read_csv('../data/model_parameters_priors.csv')
    priors = priors[priors['Parameter'] != 'k_prot_deg']
    parameters_to_fit = priors.Parameter.tolist()

    for circuit_name, df in results.items():
        print(f"Processing circuit {circuit_name}")

        # Get circuit data and setup
        conditions = get_circuit_conditions(circuit_name)
        data_file = get_data_file(circuit_name)
        exp_data, tspan = load_and_process_csv(data_file)

        # Create circuit and configuration
        first_condition = list(conditions.keys())[0]
        circuit = circuit_manager.create_circuit(circuit_name, parameters=conditions[first_condition])
        config = CircuitConfig(
            model=circuit.model,
            name=circuit_name,
            condition_params=conditions,
            experimental_data=exp_data,
            tspan=tspan,
            max_time=360,
            min_time=min_time  # Add min_time parameter
        )

        # Create fitter
        calibration_params = setup_calibration()
        circuit_fitter = CircuitFitter([config], parameters_to_fit, priors, calibration_params)

        # BEST PARAMETERS: Select, simulate and plot
        best_params = df.sort_values(by='likelihood', ascending=False).head(n_samples)
        log_params_best = best_params[parameters_to_fit].values

        # Simulate with params
        sim_data_best = circuit_fitter.simulate_parameters(log_params_best)
        log_likelihood_best = circuit_fitter.calculate_likelihood_from_simulation(sim_data_best)
        log_prior_best = circuit_fitter.calculate_log_prior(log_params_best)

        # Convert to linear for plotting
        linear_params_best = 10 ** log_params_best
        results_df_best = organize_results(
            parameters_to_fit,
            linear_params_best,
            log_likelihood_best,
            log_prior_best
        )

        # Plot best fits
        plt.figure(figsize=(12, 8))
        plot_all_simulation_results(sim_data_best, results_df_best, ll_quartile=20)
        plt.suptitle(f'Top {n_samples} Best Fits for {circuit_name}')
        if min_time:
            plt.suptitle(f'Top {n_samples} Best Fits for {circuit_name} (t₀={min_time}min)')
        plt.savefig(os.path.join(save_dir, f'best_fits_{circuit_name}.png'))
        plt.close()

        # RANDOM PARAMETERS: Select, simulate and plot
        random_params = df.sample(n=n_samples)
        log_params_random = random_params[parameters_to_fit].values

        # Simulate with params
        sim_data_random = circuit_fitter.simulate_parameters(log_params_random)
        log_likelihood_random = circuit_fitter.calculate_likelihood_from_simulation(sim_data_random)
        log_prior_random = circuit_fitter.calculate_log_prior(log_params_random)

        # Convert to linear for plotting
        linear_params_random = 10 ** log_params_random
        results_df_random = organize_results(
            parameters_to_fit,
            linear_params_random,
            log_likelihood_random,
            log_prior_random
        )

        # Plot random fits
        plt.figure(figsize=(12, 8))
        plot_all_simulation_results(sim_data_random, results_df_random, ll_quartile=20)
        plt.suptitle(f'{n_samples} Random Samples Fits for {circuit_name}')
        if min_time:
            plt.suptitle(f'{n_samples} Random Samples Fits for {circuit_name} (t₀={min_time}min)')
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
    # plot_parameter_distributions(results, output_dir)
    plot_likelihood_distributions(results, output_dir)
    plot_fits(results, output_dir, n_samples=210, min_time=30)


if __name__ == '__main__':
    main()
