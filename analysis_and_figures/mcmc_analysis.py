import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from analysis_and_figures.plots_simulation import plot_circuit_simulations
from optimization.mcmc_utils import convergence_test
from utils.process_experimental_data import organize_results


class MCMCAnalysis:
    def __init__(
        self,
        parameters,
        priors,
        likelihoods,
        step_accepts,
        swap_accepts,
        parameter_names,
        circuit_fitter,
    ):
        """
        Analyze MCMC results from parallel tempering

        Parameters
        ----------
        parameters : np.ndarray
            Shape (n_samples, n_walkers, n_chains, n_params)
        priors : np.ndarray
            Shape (n_samples, n_walkers, n_chains)
        likelihoods : np.ndarray
            Shape (n_samples, n_walkers, n_chains)
        step_accepts : np.ndarray
            Shape (n_samples, n_walkers, n_chains)
        swap_accepts : np.ndarray
            Shape (n_swap_samples, n_walkers, n_chains-1)
        parameter_names : list
            List of parameter names
        circuit_fitter : CircuitFitter
            Instance of CircuitFitter for simulations
        """
        self.parameters = parameters
        self.priors = priors
        self.likelihoods = likelihoods
        self.step_accepts = step_accepts
        self.swap_accepts = swap_accepts
        self.parameter_names = parameter_names
        self.circuit_fitter = circuit_fitter

        # Cache some useful quantities
        self.n_samples, self.n_walkers, self.n_chains, self.n_params = parameters.shape

    def to_dataframe(self):
        """
        Create a pandas DataFrame containing all MCMC results with their corresponding
        chain, walker, and iteration information.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing all MCMC results with columns for:
            - iteration, chain, walker
            - parameters (one column per parameter)
            - likelihood, prior, posterior
            - step_accepted
        """
        # Calculate total size
        total_size = self.n_samples * self.n_walkers * self.n_chains

        data = {
            "iteration": np.repeat(
                np.arange(self.n_samples), self.n_walkers * self.n_chains
            ),
            "walker": np.tile(
                np.repeat(np.arange(self.n_walkers), self.n_chains), self.n_samples
            ),
            "chain": np.tile(np.arange(self.n_chains), self.n_samples * self.n_walkers),
        }

        # Reshape and flatten parameters more efficiently
        # Reshape to (total_size, n_params) without intermediate copies
        params_reshaped = self.parameters.reshape(total_size, self.n_params)
        for i, param_name in enumerate(self.parameter_names):
            data[param_name] = params_reshaped[:, i]

        # Flatten other arrays directly (ravel is faster than flatten for contiguous arrays)
        data["likelihood"] = self.likelihoods.ravel()
        data["prior"] = self.priors.ravel()
        data["posterior"] = data["likelihood"] + data["prior"]
        data["step_accepted"] = self.step_accepts.ravel()

        # Create DataFrame
        return pd.DataFrame(data)

    def get_best_parameters(self):
        """Find parameter set with highest posterior probability"""
        posterior = self.likelihoods + self.priors
        # Find the maximum across all dimensions
        max_idx = np.unravel_index(np.argmax(posterior), posterior.shape)
        best_params = self.parameters[max_idx[0], max_idx[1], max_idx[2], :]

        return {
            "parameters": dict(zip(self.parameter_names, best_params)),
            "likelihood": self.likelihoods[max_idx],
            "prior": self.priors[max_idx],
            "posterior": posterior[max_idx],
        }

    def plot_traces(self, chain_idx=0):
        """Plot parameter traces for a specific chain"""
        n_params = len(self.parameter_names)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3 * n_params))

        for i, (param_name, ax) in enumerate(zip(self.parameter_names, axes)):
            for walker in range(self.n_walkers):
                ax.plot(self.parameters[:, walker, chain_idx, i], alpha=0.5)
            ax.set_ylabel(param_name)
            ax.set_xlabel("Sample")

        plt.tight_layout()
        return fig

    def plot_distributions(self, burn_in=0.2):
        """Plot parameter distributions after burn-in"""
        # Remove burn-in period
        burn_samples = int(self.n_samples * burn_in)
        params = self.parameters[burn_samples:, :, :, :]

        # Create subplots for each parameter
        fig, axes = plt.subplots(
            len(self.parameter_names), 1, figsize=(12, 3 * len(self.parameter_names))
        )

        for i, (param_name, ax) in enumerate(zip(self.parameter_names, axes)):
            # Plot distribution for each chain
            for chain in range(self.n_chains):
                chain_data = params[:, :, chain, i].flatten()
                sns.kdeplot(chain_data, ax=ax, label=f"Chain {chain}")
            ax.set_xlabel(param_name)
            ax.legend()

        plt.tight_layout()
        return fig

    def plot_acceptance_rates(self):
        """Plot step and swap acceptance rates"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot step acceptance rates
        step_rates = np.mean(self.step_accepts, axis=0)  # Average over samples
        sns.heatmap(step_rates, ax=ax1, cmap="viridis")
        ax1.set_title("Step Acceptance Rates")
        ax1.set_xlabel("Chain")
        ax1.set_ylabel("Walker")

        # Plot swap acceptance rates
        swap_rates = np.mean(self.swap_accepts, axis=0)  # Average over samples
        sns.heatmap(swap_rates, ax=ax2, cmap="viridis")
        ax2.set_title("Swap Acceptance Rates")
        ax2.set_xlabel("Chain Pair")
        ax2.set_ylabel("Walker")

        plt.tight_layout()
        return fig

    def compute_statistics(self, burn_in=0.2):
        """Compute summary statistics for each parameter"""
        burn_samples = int(self.n_samples * burn_in)
        params = self.parameters[burn_samples:, :, :, :]

        stats_dict = {}
        for i, param_name in enumerate(self.parameter_names):
            # Combine all chains and walkers after burn-in
            param_samples = params[:, :, :, i].flatten()

            stats_dict[param_name] = {
                "mean": np.mean(param_samples),
                "median": np.median(param_samples),
                "std": np.std(param_samples),
                "percentile_5": np.percentile(param_samples, 5),
                "percentile_95": np.percentile(param_samples, 95),
                "effective_sample_size": self._compute_ess(param_samples),
            }

        return stats_dict

    def _compute_ess(self, samples):
        """Compute effective sample size using autocorrelation"""
        n = len(samples)
        if n <= 1:
            return n

        acf = np.correlate(
            samples - np.mean(samples), samples - np.mean(samples), mode="full"
        )[n - 1 :]
        acf = acf / acf[0]

        # Find where autocorrelation drops below 0.05
        cutoff = np.where(acf < 0.05)[0]
        if len(cutoff) > 0:
            cutoff = cutoff[0]
        else:
            cutoff = len(acf)

        tau = 1 + 2 * np.sum(acf[:cutoff])
        return n / tau

    def plot_simulations(self, n_best=5, plot_all_chains=False, temperature_idx=0):
        """
        Simulate and plot results for the best parameter sets

        Parameters
        ----------
        n_best : int
            Number of best parameter sets to simulate
        plot_all_chains : bool
            If True, get best parameters from all chains
            If False, only use the lowest temperature chain
        temperature_idx : int
            Which temperature chain to use if plot_all_chains=False
        """
        # Calculate posterior
        posterior = self.likelihoods + self.priors

        if plot_all_chains:
            # Flatten across all chains and walkers
            flat_posterior = posterior.reshape(-1)
            flat_params = self.parameters.reshape(-1, self.n_params)
        else:
            # Only use specified temperature chain
            flat_posterior = posterior[:, :, temperature_idx].reshape(-1)
            flat_params = self.parameters[:, :, temperature_idx].reshape(
                -1, self.n_params
            )

        # Get indices of best parameter sets
        best_indices = np.argsort(flat_posterior)[-n_best:]

        # Extract best parameter sets
        best_params = flat_params[best_indices]

        # Simulate these parameters
        sim_data = self.circuit_fitter.simulate_parameters(best_params)

        # Calculate likelihoods and priors for these parameters
        likelihood_data = (
            self.circuit_fitter.calculate_likelihood_from_simulation_with_breakdown(
                sim_data
            )
        )
        prior_data = self.circuit_fitter.calculate_log_prior(best_params)

        # Organize results into DataFrame
        results_df = organize_results(
            self.parameter_names, best_params, likelihood_data, prior_data
        )

        # Plot results for each parameter set
        figs = []
        for i in range(n_best):
            fig = plot_circuit_simulations(sim_data, results_df)
            fig.suptitle(
                f"Simulation for Parameter Set {i + 1}\nPosterior: {flat_posterior[best_indices[i]]:.2f}"
            )
            figs.append(fig)

        return figs, best_params


def analyze_mcmc_results(
    parameters,
    priors,
    likelihoods,
    step_accepts,
    swap_accepts,
    parameter_names,
    circuit_fitter,
):
    """
    Convenience function to create analyzer and generate all plots
    """
    analyzer = MCMCAnalysis(
        parameters,
        priors,
        likelihoods,
        step_accepts,
        swap_accepts,
        parameter_names,
        circuit_fitter,
    )

    R_hat = convergence_test(
        parameters[int(len(parameters) / 2) :], per_parameter_test=True
    )

    # Compute and print statistics
    stats = analyzer.compute_statistics()
    print("\nParameter Statistics:")
    iP = 0
    for param, stat_dict in stats.items():
        print(f"\n{param}:")
        for stat_name, value in stat_dict.items():
            print(f"  {stat_name}: {value:.3e}")

        print(f"  R_hat: {R_hat[iP]:.3e}")
        iP += 1

    # Generate plots
    trace_fig = analyzer.plot_traces()
    dist_fig = analyzer.plot_distributions()
    # accept_fig = analyzer.plot_acceptance_rates()

    # Generate simulation plots for top 5 parameter sets
    # sim_figs, best_params_array = analyzer.plot_simulations(n_best=5)

    return {
        "analyzer": analyzer,
        "statistics": stats,
        "figures": {
            "traces": trace_fig,
            "distributions": dist_fig,
            # 'acceptance_rates': accept_fig,
            # "simulations": sim_figs,
        },
        # "best_parameters_array": best_params_array,
    }


def dataframe_to_mcmc_arrays(mcmc_dataframe, parameter_names):
    """Convert MCMC DataFrame to numpy arrays for mcmc_utils functions"""

    # only keep steps > 500
    # mcmc_dataframe = mcmc_dataframe[mcmc_dataframe['iteration'] > 500]

    unique_iterations = sorted(mcmc_dataframe["iteration"].unique())
    unique_walkers = sorted(mcmc_dataframe["walker"].unique())
    unique_chains = sorted(mcmc_dataframe["chain"].unique())

    # n_samples = len(unique_iterations)
    # n_walkers = len(unique_walkers)
    # n_chains = len(unique_chains)
    # n_params = len(parameter_names)

    # convergence_test format: (n_samples, n_walkers, n_chains, n_params)
    # convergence_array = np.zeros((n_samples, n_walkers, n_chains, n_params))
    #
    # for _, row in mcmc_dataframe.iterrows():
    #     sample_idx = unique_iterations.index(row['iteration'])
    #     walker_idx = unique_walkers.index(row['walker'])
    #     chain_idx = unique_chains.index(row['chain'])
    #
    #     for param_idx, param_name in enumerate(parameter_names):
    #         convergence_array[sample_idx, walker_idx, chain_idx, param_idx] = row[param_name]

    valid_parameter_names = list(
        set(parameter_names).intersection(mcmc_dataframe.columns)
    )

    data = mcmc_dataframe[valid_parameter_names].values
    # data = data[:, 3:]
    convergence_array = data.reshape(
        len(unique_iterations), len(unique_walkers), len(unique_chains), -1
    )

    pass

    return {"convergence": convergence_array, "parameter_names": valid_parameter_names}
