from utils.import_and_visualise_data import load_and_process_csv
from tests.test_GFP_positive_control import test_pos_control
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysb.simulator import ScipyOdeSimulator
import scipy.stats
from optimization.parallel_tempering import ParallelTempering
import seaborn as sns


def plot_simulations(tspan, sim_values, experimental_data, likelihoods=None, color_by_likelihood=False):
    """Plot simulation results with experimental data."""
    plt.figure(figsize=(10, 6))

    if color_by_likelihood and likelihoods is not None:
        norm = plt.Normalize(likelihoods.min(), likelihoods.max())
        cmap = plt.cm.viridis
        for i, sim in enumerate(sim_values):
            plt.plot(tspan, sim,
                    color=cmap(norm(likelihoods[i])),
                    alpha=0.5, linewidth=1, linestyle='--')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, label='Log Likelihood')
    else:
        plt.plot(tspan, sim_values.T,
                color='grey', alpha=0.5, linewidth=1, linestyle='--')

    pos_control_subset = experimental_data[experimental_data['time'].isin(tspan)]
    for replicate in pos_control_subset['replicate'].unique():
        replicate_data = pos_control_subset[pos_control_subset['replicate'] == replicate]
        plt.plot(replicate_data['time'], replicate_data['fluorescence'],
                marker='x', color='red', linestyle='None', markersize=5)

    plt.xlabel('Time')
    plt.ylabel('Fluorescence')
    plt.title('Comparison of Experimental and Simulated GFP')
    plt.grid(True)
    plt.show()


def likelihood_pos_ctrl(model, param_sets, plot_comparison=True, color_by_likelihood=False):
    """Vectorized computation of log likelihood for multiple parameter sets."""
    # Process experimental data once
    file_path = '../data/pos_ctrl.csv'
    experimental_data = load_and_process_csv(file_path)

    neg_control = experimental_data[experimental_data['condition'] == 'negative control']
    neg_control_mean = neg_control.groupby('time')['fluorescence'].mean()

    pos_control = experimental_data[experimental_data['condition'] == '3 nM sfGFP'].copy()
    pos_control['fluorescence'] = pos_control.apply(
        lambda row: row['fluorescence'] - neg_control_mean.get(row['time'], 0), axis=1
    )

    # Time points for simulation
    tmax = 300
    tspan = pos_control[pos_control['time'] <= tmax]['time'].unique()

    # Run simulations for all parameter sets at once
    simulator = ScipyOdeSimulator(model, tspan)
    simulation_results = simulator.run(param_values=param_sets)

    # Prepare experimental data arrays
    pos_control_subset = pos_control[pos_control['time'].isin(tspan)]
    exp_means = np.array([
        pos_control_subset[pos_control_subset['time'] == t]['fluorescence'].mean()
        for t in tspan
    ])
    exp_vars = np.array([
        max(pos_control_subset[pos_control_subset['time'] == t]['fluorescence'].var(), 1)
        for t in tspan
    ])

    # Vectorized likelihood calculation
    n_sets = param_sets.shape[0]
    sim_values = np.array([
        simulation_results.observables[i]['obs_Protein_GFP'] * 100
        for i in range(n_sets)
    ])

    # Reshape arrays for broadcasting
    exp_means = exp_means.reshape(1, -1)
    exp_vars = exp_vars.reshape(1, -1)

    # Compute residuals and likelihoods using broadcasting
    residuals = sim_values - exp_means
    log_likelihoods = -0.5 * np.sum((residuals ** 2) / exp_vars, axis=1)

    if plot_comparison:
        plot_simulations(tspan, sim_values, pos_control_subset,
                        likelihoods=log_likelihoods if color_by_likelihood else None,
                        color_by_likelihood=color_by_likelihood)

    return pd.Series(log_likelihoods, index=param_sets.index)


def prior(param_sets, model_parameters_priors):
    """Vectorized computation of log prior probability for multiple parameter sets."""
    means = np.array([
        model_parameters_priors[model_parameters_priors['Parameter'] == param]['Value'].iloc[0]
        for param in param_sets.columns
    ])
    stddevs = np.array([
        10 ** model_parameters_priors[model_parameters_priors['Parameter'] == param]['log10stddev'].iloc[0]
        for param in param_sets.columns
    ])

    param_values = param_sets.values
    log_priors = -0.5 * np.sum(
        ((param_values - means) / stddevs) ** 2,
        axis=1
    )

    return pd.Series(log_priors, index=param_sets.index)




def create_likelihood_for_pt(model, parameters_to_be_fitted):
    """Create likelihood function for parallel tempering"""

    def log_likelihood(parameters):
        """
        Adapter function for parallel tempering
        parameters: dict with parameter names and values
        """
        # Convert dictionary to DataFrame (what your likelihood expects)
        param_df = pd.DataFrame([parameters])

        # Use your existing likelihood function
        log_like = likelihood_pos_ctrl(
            model=model,
            param_sets=param_df,
            plot_comparison=False  # Don't plot during MCMC
        ).iloc[0]  # Get scalar value

        return log_like

    return log_likelihood
def create_priors_for_pt(parameters_to_be_fitted, model_parameters_priors):
    """Create prior distributions for parallel tempering"""
    priors = {}
    for param in parameters_to_be_fitted:
        # Get prior information
        mean = model_parameters_priors[model_parameters_priors['Parameter'] == param]['Value'].iloc[0]
        stddev = 10 ** model_parameters_priors[model_parameters_priors['Parameter'] == param]['log10stddev'].iloc[0]

        # Create log-normal distribution
        priors[param] = scipy.stats.lognorm(s=np.log(stddev), scale=mean)

    return priors


def create_pt_likelihood(model, parameters_to_be_fitted):
    """Create likelihood function for parallel tempering"""

    def pt_likelihood(log_params_array):
        """
        Adapter function for parallel tempering
        log_params_array: numpy array of log parameters [n_walkers, n_chains, n_dim] or [n_dim]
        """
        # Handle different input shapes
        if len(log_params_array.shape) == 3:  # [n_walkers, n_chains, n_dim]
            n_walkers, n_chains, n_dim = log_params_array.shape
            params_reshaped = log_params_array.reshape(-1, n_dim)
        elif len(log_params_array.shape) == 2:  # [n_walkers, n_dim]
            n_walkers, n_dim = log_params_array.shape
            params_reshaped = log_params_array
        else:  # single parameter set [n_dim]
            params_reshaped = log_params_array.reshape(1, -1)

        # Convert to list of parameter dictionaries with exponentiated values
        param_dicts = []
        for log_params in params_reshaped:
            param_dict = {param: float(np.exp(val)) for param, val in zip(parameters_to_be_fitted, log_params)}
            param_dicts.append(param_dict)

        # Convert to DataFrame
        param_df = pd.DataFrame(param_dicts)

        # Calculate likelihood
        log_likes = likelihood_pos_ctrl(
            model=model,
            param_sets=param_df,
            plot_comparison=False
        )

        # Reshape result if needed
        if len(log_params_array.shape) == 3:
            return log_likes.values.reshape(n_walkers, n_chains)
        elif len(log_params_array.shape) == 2:
            return log_likes.values
        else:
            return log_likes.values[0]

    return pt_likelihood


def create_pt_prior(model_parameters_priors, parameters_to_be_fitted):
    """Create prior function for parallel tempering"""

    def pt_prior(log_params_array):
        """
        Adapter function for parallel tempering
        log_params_array: numpy array of log parameters [n_walkers, n_chains, n_dim] or [n_dim]
        """
        # Handle different input shapes
        if len(log_params_array.shape) == 3:  # [n_walkers, n_chains, n_dim]
            n_walkers, n_chains, n_dim = log_params_array.shape
            params_reshaped = log_params_array.reshape(-1, n_dim)
        elif len(log_params_array.shape) == 2:  # [n_walkers, n_dim]
            n_walkers, n_dim = log_params_array.shape
            params_reshaped = log_params_array
        else:  # single parameter set [n_dim]
            params_reshaped = log_params_array.reshape(1, -1)

        # Convert to DataFrame with exponentiated values
        param_dicts = []
        for log_params in params_reshaped:
            param_dict = {param: float(np.exp(val)) for param, val in zip(parameters_to_be_fitted, log_params)}
            param_dicts.append(param_dict)

        param_df = pd.DataFrame(param_dicts)

        # Calculate prior
        log_priors = prior(param_df, model_parameters_priors)

        # Reshape result if needed
        if len(log_params_array.shape) == 3:
            return log_priors.values.reshape(n_walkers, n_chains)
        elif len(log_params_array.shape) == 2:
            return log_priors.values
        else:
            return log_priors.values[0]

    return pt_prior


def create_mcmc_dataframe(parameters, priors, likelihoods, parameters_to_be_fitted):
    """
    Create a DataFrame from MCMC parameter samples, priors, and likelihoods to facilitate plotting posterior distributions and correlations.
    Includes both log-transformed and exponentiated parameters.

    :param parameters: numpy array of MCMC samples with shape (n_samples, n_walkers, n_chains, n_params)
    :param priors: numpy array of prior values in log space with shape (n_samples, n_walkers, n_chains)
    :param likelihoods: numpy array of likelihood values in log space with shape (n_samples, n_walkers, n_chains)
    :param parameters_to_be_fitted: list of parameter names corresponding to the columns in parameters
    :return: pandas DataFrame with columns for log and exponentiated parameter values, priors, likelihoods, chain, walker, and sample number
    """
    n_samples, n_walkers, n_chains, n_params = parameters.shape

    # Flatten parameters and build a DataFrame with sample, walker, and chain information
    data = {
        'sample': np.repeat(np.arange(n_samples), n_walkers * n_chains),
        'walker': np.tile(np.repeat(np.arange(n_walkers), n_chains), n_samples),
        'chain': np.tile(np.arange(n_chains), n_samples * n_walkers),
        'log_prior': priors.reshape(-1),  # Flattened prior values in log space
        'log_likelihood': likelihoods.reshape(-1),  # Flattened likelihood values in log space
    }

    # Flatten each parameter and create columns for log-transformed and real-space values
    log_params = parameters.reshape(n_samples * n_walkers * n_chains, n_params)  # Flattened parameters in log space
    real_params = np.exp(log_params)  # Convert to real space

    # Add log and real-space parameter columns
    for i, param in enumerate(parameters_to_be_fitted):
        data[f"log_{param}"] = log_params[:, i]  # Log-scale parameter
        data[param] = real_params[:, i]  # Exponentiated parameter

    # Create and return the DataFrame
    mcmc_df = pd.DataFrame(data)

    return mcmc_df


def plot_mcmc_trajectory(mcmc_df, parameters_to_be_fitted):
    """
    Plot MCMC chain trajectories for each parameter using Seaborn with colorful lines.

    Parameters:
    -----------
    mcmc_df : pandas.DataFrame
        DataFrame containing MCMC samples
    parameters_to_be_fitted : list
        List of parameter names to plot
    """
    fig, axes = plt.subplots(len(parameters_to_be_fitted), 1, figsize=(12, 3 * len(parameters_to_be_fitted)))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, param in zip(axes, parameters_to_be_fitted):
        sns.lineplot(
            data=mcmc_df,
            x='sample',
            y=param,
            hue='chain',
            style='walker',
            alpha=0.3,
            legend=False,
            ax=ax,
            palette='Paired'
        )
        ax.set_yscale('log')
        ax.set_ylabel(param)

    plt.tight_layout()
    plt.show()

def plot_posterior_distribution(mcmc_df, parameters_to_be_fitted):
    # Plot posterior distributions
    sns.pairplot(
        mcmc_df,
        vars=parameters_to_be_fitted,
        # hue='likelihood',
        # plot_kws={'transform': 'log'},
        # palette="viridis",
        diag_kind='kde',
    )
    sns.pairplot(
        mcmc_df,
        vars=['log_' + param for param in parameters_to_be_fitted],
        # hue='likelihood',
        # plot_kws={'transform': 'log'},
        # palette="viridis",
        diag_kind='kde',
    )
    plt.show()

def plot_all_figures(mcmc_df, parameters_to_be_fitted):
    plot_mcmc_trajectory(mcmc_df, parameters_to_be_fitted)
    plot_posterior_distribution(mcmc_df, parameters_to_be_fitted)


if __name__ == "__main__":
    # Setup
    parameters_to_be_fitted = ["k_tx", "K_tx", "k_rna_deg", "k_tl", "K_tl", "k_prot_deg", "k_mat", "k_csy4"]
    model_parameters_priors = pd.read_csv('../data/model_parameters_priors.csv')
    model = test_pos_control(plot=False, print_rules=False, print_odes=False,
                             parameters_plasmids={"k_GFP_concentration": 3})

    # Get initial parameters
    initial_params = model_parameters_priors[
        model_parameters_priors['Parameter'].isin(parameters_to_be_fitted)
    ].set_index('Parameter')['Value']

    # Convert initial parameters to log space
    log_initial_params = np.log(initial_params)

    # Test likelihood function first
    test_params_array = np.array([log_initial_params[param] for param in parameters_to_be_fitted])
    pt_likelihood = create_pt_likelihood(model, parameters_to_be_fitted)
    pt_prior = create_pt_prior(model_parameters_priors, parameters_to_be_fitted)

    print("Testing single parameter set...")
    test_like = pt_likelihood(test_params_array)
    test_prior = pt_prior(test_params_array)
    print(f"Log likelihood: {test_like}")
    print(f"Log prior: {test_prior}")

    # Initialize sampler
    print("\nInitializing sampler...")
    sampler = ParallelTempering(
        log_likelihood=pt_likelihood,
        log_prior=pt_prior,
        n_dim=len(parameters_to_be_fitted),
        n_walkers=10,
        n_chains=5
    )

    # Create initial parameters array in log space [n_walkers, n_chains, n_dim]
    initial_params_array = np.array([log_initial_params[param] for param in parameters_to_be_fitted])
    initial_params_array = np.tile(initial_params_array, (10, 5, 1))  # Replicate for all walkers and chains

    # Add normal noise in log space (equivalent to log-normal in real space)
    noise = np.random.normal(0, 0.1, initial_params_array.shape)
    initial_params_array += noise

    # Run sampling
    print("Running parallel tempering...")
    parameters, priors, likelihoods, step_accepts, swap_accepts = sampler.run(
        initial_parameters=initial_params_array,
        n_samples=200,
        n_swaps=2
    )

    # Print acceptance rates
    print("\nStep acceptance rates:", np.mean(step_accepts, axis=0))
    print("Swap acceptance rates:", np.mean(swap_accepts, axis=0))



    mcmc_df = create_mcmc_dataframe(parameters, priors, likelihoods, parameters_to_be_fitted)
    # keep he samples that are over 100 (burnin phase)
    mcmc_df = mcmc_df[mcmc_df['sample'] > 10]

    plot_all_figures(mcmc_df, parameters_to_be_fitted)




