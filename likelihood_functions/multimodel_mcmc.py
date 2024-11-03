import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pysb.simulator import ScipyOdeSimulator
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple
from optimization.parallel_tempering import ParallelTempering


@dataclass
class ModelConfig:
    """Configuration for a single model and its conditions"""
    model_func: Callable
    data_file: str
    conditions: Dict[str, Dict]  # condition_name -> parameters
    observable: str = 'obs_Protein_GFP'
    scaling: float = 100.0
    data_condition_column: str = 'condition'
    data_time_column: str = 'time'
    data_value_column: str = 'fluorescence'
    data_replicate_column: str = 'replicate'


class MultiModelMCMC:
    def __init__(
            self,
            models: List[ModelConfig],
            parameters_to_fit: List[str],
            parameter_priors: pd.DataFrame,
            n_walkers: int = 10,
            n_chains: int = 4,
            n_swaps: int = 2
    ):
        """
        Initialize MultiModelMCMC system

        Parameters:
        -----------
        models : List[ModelConfig]
            List of model configurations
        parameters_to_fit : List[str]
            Names of parameters to be fitted
        parameter_priors : pd.DataFrame
            DataFrame with prior information for parameters
        n_walkers : int
            Number of walkers per chain
        n_chains : int
            Number of temperature chains
        n_swaps : int
            Number of chain swaps per step
        """
        self.models = models
        self.parameters_to_fit = parameters_to_fit
        self.parameter_priors = parameter_priors
        self.n_walkers = n_walkers
        self.n_chains = n_chains
        self.n_swaps = n_swaps

        # Initialize MCMC sampler
        self.sampler = ParallelTempering(
            log_likelihood=self._create_likelihood(),
            log_prior=self._create_prior(),
            n_dim=len(parameters_to_fit),
            n_walkers=n_walkers,
            n_chains=n_chains
        )

        # Storage for results
        self.mcmc_results = None
        self.mcmc_df = None
        self.best_params = None

    def simulate(self,
                 log_params_array: np.ndarray,
                 model_indices: Optional[List[int]] = None
                 ) -> List[Tuple[np.ndarray, pd.DataFrame, ModelConfig]]:
        """
        Simulate models for given parameter sets

        Parameters:
        -----------
        log_params_array : np.ndarray
            Parameters in log space with shape (n_walkers, n_chains, n_dim) or (n_sets, n_dim)
        model_indices : list[int] | None
            Indices of models to simulate. If None, simulates all models

        Returns:
        --------
        list[tuple] : List of (simulation_results, experimental_data, model_config) for each condition
        """
        # Handle different input shapes
        if len(log_params_array.shape) == 3:
            n_walkers, n_chains, n_dim = log_params_array.shape
            params_reshaped = log_params_array.reshape(-1, n_dim)
        else:
            params_reshaped = log_params_array.reshape(-1, log_params_array.shape[-1])

        # Convert to real-space parameters
        real_params = np.exp(params_reshaped)

        # Select models to simulate
        models_to_simulate = self.models
        if model_indices is not None:
            models_to_simulate = [self.models[i] for i in model_indices]

        all_results = []
        for model_config in models_to_simulate:
            # Load experimental data
            experimental_data = pd.read_csv(model_config.data_file)

            for condition_name, condition_params in model_config.conditions.items():
                # Get condition data
                condition_data = experimental_data[
                    experimental_data[model_config.data_condition_column] == condition_name
                    ]

                if len(condition_data) == 0:
                    print(f"No data found for condition: {condition_name}")
                    continue

                # Get simulation timepoints
                tspan = condition_data[model_config.data_time_column].unique()

                # Create model instance
                model = model_config.model_func(
                    plot=False,
                    print_rules=False,
                    print_odes=False,
                    parameters_plasmids=condition_params
                )

                # Run simulation for all parameter sets at once
                simulator = ScipyOdeSimulator(model, tspan)
                simulation_results = simulator.run(param_values=real_params)

                all_results.append((simulation_results, condition_data, model_config))

        return all_results

    def _calculate_likelihood(
            self,
            simulation_results: np.ndarray,
            experimental_data: pd.DataFrame,
            config: ModelConfig
    ) -> np.ndarray:
        """Calculate log likelihood between simulation and experimental data"""
        # Group experimental data by time point
        exp_stats = experimental_data.groupby(config.data_time_column).agg({
            config.data_value_column: ['mean', 'var']
        }).reset_index()

        # Use maximum of variance and 1 to avoid division by zero
        exp_vars = np.maximum(exp_stats[config.data_value_column]['var'].values, 1)
        exp_means = exp_stats[config.data_value_column]['mean'].values

        # Scale simulation results
        sim_values = simulation_results.observables[config.observable] * config.scaling

        # Compute residuals and log likelihood
        residuals = sim_values - exp_means.reshape(1, -1)
        log_likelihoods = -0.5 * np.sum((residuals ** 2) / exp_vars, axis=1)

        return log_likelihoods

    def _create_likelihood(self) -> Callable:
        """Create likelihood function for all models"""

        def multi_model_likelihood(log_params_array: np.ndarray) -> np.ndarray:
            # Get simulation results
            sim_results = self.simulate(log_params_array)

            # Calculate total log likelihood
            total_log_likes = np.zeros(len(log_params_array.reshape(-1, log_params_array.shape[-1])))

            for simulation_results, condition_data, model_config in sim_results:
                log_likes = self._calculate_likelihood(
                    simulation_results,
                    condition_data,
                    model_config
                )
                total_log_likes += log_likes

            # Reshape if needed
            if len(log_params_array.shape) == 3:
                n_walkers, n_chains, _ = log_params_array.shape
                return total_log_likes.reshape(n_walkers, n_chains)
            return total_log_likes

        return multi_model_likelihood

    def _create_prior(self) -> Callable:
        """Create prior function"""

        def multi_model_prior(log_params_array: np.ndarray) -> np.ndarray:
            # Get prior means and standard deviations
            means = np.array([
                self.parameter_priors[
                    self.parameter_priors['Parameter'] == param
                    ]['Value'].iloc[0]
                for param in self.parameters_to_fit
            ])

            stddevs = np.array([
                10 ** self.parameter_priors[
                    self.parameter_priors['Parameter'] == param
                    ]['log10stddev'].iloc[0]
                for param in self.parameters_to_fit
            ])

            # Handle different input shapes
            if len(log_params_array.shape) == 3:
                n_walkers, n_chains, n_dim = log_params_array.shape
                params_reshaped = log_params_array.reshape(-1, n_dim)
            else:
                params_reshaped = log_params_array.reshape(-1, log_params_array.shape[-1])

            # Convert to real space
            real_params = np.exp(params_reshaped)

            # Calculate log prior
            log_priors = -0.5 * np.sum(
                ((real_params - means) / stddevs) ** 2,
                axis=1
            )

            # Reshape if needed
            if len(log_params_array.shape) == 3:
                return log_priors.reshape(n_walkers, n_chains)
            return log_priors

        return multi_model_prior

    def run_mcmc(
            self,
            n_samples: int,
            initial_params: Optional[np.ndarray] = None,
            noise_scale: float = 0.1
    ) -> None:
        """
        Run MCMC sampling

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        initial_params : np.ndarray, optional
            Initial parameter values (will use prior means if not provided)
        noise_scale : float
            Scale of normal noise to add to initial parameters
        """
        if initial_params is None:
            # Use prior means as initial parameters
            initial_params = np.array([
                self.parameter_priors[
                    self.parameter_priors['Parameter'] == param
                    ]['Value'].iloc[0]
                for param in self.parameters_to_fit
            ])
            initial_params = np.log(initial_params)  # Convert to log space

        # Create initial parameter array
        initial_params_array = np.tile(
            initial_params,
            (self.n_walkers, self.n_chains, 1)
        )

        # Add noise
        noise = np.random.normal(0, noise_scale, initial_params_array.shape)
        initial_params_array += noise

        # Run sampling
        print("Running parallel tempering...")
        parameters, priors, likelihoods, step_accepts, swap_accepts = self.sampler.run(
            initial_parameters=initial_params_array,
            n_samples=n_samples,
            n_swaps=self.n_swaps
        )

        # Store results
        self.mcmc_results = {
            'parameters': parameters,
            'priors': priors,
            'likelihoods': likelihoods,
            'step_accepts': step_accepts,
            'swap_accepts': swap_accepts
        }

        # Create DataFrame
        self.mcmc_df = self._create_mcmc_dataframe()

        # Find best parameters
        best_idx = np.argmax(likelihoods.reshape(-1))
        self.best_params = dict(zip(
            self.parameters_to_fit,
            np.exp(parameters.reshape(-1, len(self.parameters_to_fit))[best_idx])
        ))

        print("\nStep acceptance rates:", np.mean(step_accepts, axis=0))
        print("Swap acceptance rates:", np.mean(swap_accepts, axis=0))

    def _create_mcmc_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from MCMC results"""
        parameters = self.mcmc_results['parameters']
        priors = self.mcmc_results['priors']
        likelihoods = self.mcmc_results['likelihoods']

        n_samples, n_walkers, n_chains, n_params = parameters.shape

        data = {
            'sample': np.repeat(np.arange(n_samples), n_walkers * n_chains),
            'walker': np.tile(np.repeat(np.arange(n_walkers), n_chains), n_samples),
            'chain': np.tile(np.arange(n_chains), n_samples * n_walkers),
            'log_prior': priors.reshape(-1),
            'log_likelihood': likelihoods.reshape(-1),
        }

        log_params = parameters.reshape(-1, n_params)
        real_params = np.exp(log_params)

        for i, param in enumerate(self.parameters_to_fit):
            data[f"log_{param}"] = log_params[:, i]
            data[param] = real_params[:, i]

        return pd.DataFrame(data)

    def plot_simulations(self,
                         log_params_array: np.ndarray,
                         model_indices: Optional[List[int]] = None,
                         show_replicates: bool = True):
        """
        Plot simulation results against experimental data

        Parameters:
        -----------
        log_params_array : np.ndarray
            Parameters in log space with shape (n_walkers, n_chains, n_dim) or (n_sets, n_dim)
        model_indices : list[int] | None
            Indices of models to plot. If None, plots all models
        show_replicates : bool
            Whether to show individual replicates
        """
        # Get simulation results
        sim_results = self.simulate(log_params_array, model_indices)

        # Organize results by model
        model_results = {}
        for sim_result, condition_data, model_config in sim_results:
            if model_config not in model_results:
                model_results[model_config] = []
            model_results[model_config].append((sim_result, condition_data))

        # Plot results
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(len(model_results), 1,
                                 figsize=(12, 5 * len(model_results)))
        if len(model_results) == 1:
            axes = [axes]

        for ax, (model_config, results) in zip(axes, model_results.items()):
            plot_data = []

            # Process each condition
            for sim_result, condition_data in results:
                condition_name = condition_data[model_config.data_condition_column].iloc[0]

                # Add simulation data
                sim_values = sim_result.observables[model_config.observable] * model_config.scaling
                for i in range(len(sim_values)):
                    plot_data.append(pd.DataFrame({
                        'Time': condition_data[model_config.data_time_column].unique(),
                        'Value': sim_values[i],
                        'Type': 'Simulation',
                        'Condition': condition_name,
                        'Parameter Set': f'Set {i}'
                    }))

                # Add experimental data
                plot_data.append(pd.DataFrame({
                    'Time': condition_data[model_config.data_time_column],
                    'Value': condition_data[model_config.data_value_column],
                    'Type': 'Experimental',
                    'Condition': condition_name,
                    'Replicate': condition_data[model_config.data_replicate_column]
                }))

            # Create plot
            plot_df = pd.concat(plot_data, ignore_index=True)

            # Plot simulations
            sns.lineplot(
                data=plot_df[plot_df['Type'] == 'Simulation'],
                x='Time',
                y='Value',
                hue='Condition',
                style='Parameter Set',
                dashes=True,
                ax=ax
            )

            # Plot experimental data
            if show_replicates:
                sns.scatterplot(
                    data=plot_df[plot_df['Type'] == 'Experimental'],
                    x='Time',
                    y='Value',
                    hue='Condition',
                    style='Replicate',
                    ax=ax,
                    alpha=0.5,
                    legend='brief'
                )
            else:
                sns.lineplot(
                    data=plot_df[plot_df['Type'] == 'Experimental'],
                    x='Time',
                    y='Value',
                    hue='Condition',
                    ax=ax,
                    err_style='band',
                    marker='o'
                )

            ax.set_xlabel('Time')
            ax.set_ylabel('Signal')
            ax.set_title(f'Model Fits')

        plt.tight_layout()
        plt.show()
        sns.reset_defaults()

    def plot_fits(self):
        """Plot model fits using best parameters"""
        if self.best_params is None:
            raise ValueError("No MCMC results available. Run MCMC first.")

        # Convert best parameters to log space and correct shape
        log_params = np.log(np.array([list(self.best_params.values())]))
        self.plot_simulations(log_params)

    def plot_trajectories(self, burnin: int = 0):
        """
        Plot MCMC chain trajectories for each parameter

        Parameters:
        -----------
        burnin : int
            Number of initial samples to discard
        """
        if self.mcmc_df is None:
            raise ValueError("No MCMC results available. Run MCMC first.")

        plot_data = self.mcmc_df[self.mcmc_df['sample'] > burnin]

        fig, axes = plt.subplots(
            len(self.parameters_to_fit), 1,
            figsize=(12, 3 * len(self.parameters_to_fit))
        )
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for ax, param in zip(axes, self.parameters_to_fit):
            sns.lineplot(
                data=plot_data,
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

    def plot_posteriors(self, burnin: int = 0):
        """
        Plot posterior distributions

        Parameters:
        -----------
        burnin : int
            Number of initial samples to discard
        """
        if self.mcmc_df is None:
            raise ValueError("No MCMC results available. Run MCMC first.")

        plot_data = self.mcmc_df[self.mcmc_df['sample'] > burnin]

        # Plot real-space parameters
        sns.pairplot(
            plot_data,
            vars=self.parameters_to_fit,
            diag_kind='kde',
        )

        # Plot log-space parameters
        sns.pairplot(
            plot_data,
            vars=[f'log_{param}' for param in self.parameters_to_fit],
            diag_kind='kde',
        )
        plt.show()

    def plot_chain_temperatures(self):
        """Plot acceptance rates vs chain temperatures"""
        if self.mcmc_results is None:
            raise ValueError("No MCMC results available. Run MCMC first.")

        step_accepts = np.mean(self.mcmc_results['step_accepts'], axis=0)
        swap_accepts = np.mean(self.mcmc_results['swap_accepts'], axis=0)

        temps = np.exp(np.linspace(0, np.log(20), self.n_chains))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(temps, step_accepts, '-o')
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Step Acceptance Rate')
        ax1.set_xscale('log')
        ax1.grid(True)

        ax2.plot(temps[:-1], swap_accepts, '-o')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Swap Acceptance Rate')
        ax2.set_xscale('log')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def get_parameter_statistics(self, burnin: int = 0) -> pd.DataFrame:
        """
        Calculate statistics for fitted parameters

        Parameters:
        -----------
        burnin : int
            Number of initial samples to discard

        Returns:
        --------
        pd.DataFrame with parameter statistics
        """
        if self.mcmc_df is None:
            raise ValueError("No MCMC results available. Run MCMC first.")

        stats_df = self.mcmc_df[self.mcmc_df['sample'] > burnin][self.parameters_to_fit].agg([
            'mean', 'std', 'median',
            lambda x: np.percentile(x, 2.5),
            lambda x: np.percentile(x, 97.5)
        ])

        stats_df.rename(index={
            '<lambda_0>': '2.5%',
            '<lambda_1>': '97.5%'
        }, inplace=True)

        return stats_df

    def check_data_compatibility(self):
        """Check if the data files are compatible with the configuration"""
        for model_config in self.models:
            try:
                data = pd.read_csv(model_config.data_file)
                print(f"\nChecking {model_config.data_file}:")
                print("Available columns:", data.columns.tolist())

                # Check required columns exist
                required_columns = [
                    model_config.data_condition_column,
                    model_config.data_time_column,
                    model_config.data_value_column,
                    model_config.data_replicate_column
                ]

                for col in required_columns:
                    if col in data.columns:
                        print(f"✓ Found column: {col}")
                    else:
                        print(f"✗ Missing column: {col}")

                # Check conditions exist
                conditions = data[model_config.data_condition_column].unique()
                print("\nAvailable conditions:", conditions.tolist())

                for condition in model_config.conditions:
                    if condition in conditions:
                        print(f"✓ Found condition: {condition}")
                    else:
                        print(f"✗ Missing condition: {condition}")

            except Exception as e:
                print(f"Error checking data file: {str(e)}")


if __name__ == "__main__":
    from tests.test_GFP_positive_control import test_pos_control

    # Define model configurations
    models = [
        ModelConfig(
            model_func=test_pos_control,
            data_file='../data/pos_ctrl.csv',
            conditions={
                '3 nM sfGFP': {'k_GFP_concentration': 3},
            }
        ),
    ]

    # Parameters to fit
    parameters_to_fit = ["k_tx", "K_tx", "k_rna_deg", "k_tl", "K_tl",
                         "k_prot_deg", "k_mat", "k_csy4"]

    # Load parameter priors
    parameter_priors = pd.read_csv('../data/model_parameters_priors.csv')

    # Initialize MCMC
    mcmc = MultiModelMCMC(
        models=models,
        parameters_to_fit=parameters_to_fit,
        parameter_priors=parameter_priors,
        n_walkers=10,
        n_chains=4
    )

    # Check data compatibility
    mcmc.check_data_compatibility()

    # Run MCMC
    mcmc.run_mcmc(n_samples=200)

    # Analyze results
    print("\nParameter Statistics:")
    print(mcmc.get_parameter_statistics(burnin=50))

    # Plot results
    mcmc.plot_trajectories(burnin=50)
    mcmc.plot_posteriors(burnin=50)
    mcmc.plot_fits()
    mcmc.plot_chain_temperatures()