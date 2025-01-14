import numpy as np
import pandas as pd
from pysb.simulator import ScipyOdeSimulator

from optimization.adaptive_parallel_tempering import ParallelTempering
from .likelihood import compute_condition_likelihood
from .utils import prepare_combined_params


class CircuitFitter:
    def __init__(self, configs, parameters_to_fit, model_parameters_priors, calibration_data):
        """
        Initialize CircuitFitter

        Parameters
        ----------
        configs : list
            List of configuration objects
        parameters_to_fit : list
            List of parameter names to fit
        model_parameters_priors : pd.DataFrame
            Prior distributions for model parameters
        calibration_data : dict
            Dictionary containing:
                - slope: float
                - intercept: float
                - calibration_protein_slug: str (e.g., 'avGFP')
                - target_protein_slug: str (e.g., 'sfGFP')
        """
        self.configs = configs
        self.parameters_to_fit = parameters_to_fit
        self.model_parameters_priors = model_parameters_priors
        self.calibration_params = calibration_data
        self.simulators = {}

        # Process calibration data and set up GFP variants
        self._setup_priors()
        self._validate_configs()
        self._setup_simulators()

    def _validate_configs(self):
        """Check that all configs have the same model"""
        # TODO: Check that all models are the same
        pass

    def _setup_simulators(self):
        for config in self.configs:
            simulator = ScipyOdeSimulator(config.model, config.tspan, compiler='cython', cleanup=True)
            self.simulators[config.name] = simulator

    def _setup_priors(self):
        """Setup prior parameters in log space for easy access"""
        self.log_means = {}
        self.log_stds = {}
        for param_name in self.parameters_to_fit:
            prior_row = self.model_parameters_priors[
                self.model_parameters_priors['Parameter'] == param_name
                ].iloc[0]
            self.log_means[param_name] = np.log10(prior_row['Value'])
            self.log_stds[param_name] = prior_row['log10stddev']

    @staticmethod
    def log_to_linear_params(log_params: np.ndarray, param_names: list) -> pd.DataFrame:
        """Convert parameters from log space to linear space."""
        linear_params = 10 ** log_params
        if log_params.ndim == 1:
            return pd.DataFrame([linear_params], columns=param_names)
        return pd.DataFrame(linear_params, columns=param_names, index=range(len(log_params)))

    @staticmethod
    def linear_to_log_params(linear_params: pd.DataFrame) -> np.ndarray:
        """Convert parameters from linear space to log space."""
        return np.log10(linear_params.values)

    def generate_test_parameters(self, n_sets: int = 20) -> np.ndarray:
        """
        Generate test parameters in log space based on Gaussian priors.

        Args:
            n_sets: Number of parameter sets to generate

        Returns:
            Array of shape (n_sets, n_params) in log space
        """
        n_params = len(self.parameters_to_fit)
        log_params = np.zeros((n_sets, n_params))

        for i, param_name in enumerate(self.parameters_to_fit):
            mean = self.log_means[param_name]
            std = self.log_stds[param_name]
            log_params[:, i] = np.random.normal(mean, std, n_sets)

        return log_params

    def simulate_parameters(self, log_params: np.ndarray) -> dict:
        """
        Run simulations for given parameter sets (in log space) across all conditions

        Args:
            log_params: Array of shape (n_samples, n_params) or (n_params,) in log space

        Returns:
            Dictionary with simulation results
        """
        # Convert to linear space for simulation
        linear_params = self.log_to_linear_params(log_params, self.parameters_to_fit)
        results = {}

        for i, config in enumerate(self.configs):
            combined_params_df = prepare_combined_params(
                linear_params,
                config.condition_params
            )

            simulator = ScipyOdeSimulator(config.model, config.tspan, compiler='cython', cleanup=True)
            simulation_results = simulator.run(
                param_values=combined_params_df.drop(['param_set_idx', 'condition'], axis=1),
            )

            results[i] = {
                'combined_params': combined_params_df,
                'simulation_results': simulation_results,
                'config': config
            }

        return results

    # def simulate_parameters(self, log_params: np.ndarray) -> dict:
    #     """
    #     Run simulations for given parameter sets (in log space) across all conditions
    #
    #     Args:
    #         log_params: Array of shape (n_samples, n_params) or (n_params,) in log space
    #
    #     Returns:
    #         Dictionary with simulation results
    #     """
    #     # Convert to linear space for simulation
    #     linear_params = self.log_to_linear_params(log_params, self.parameters_to_fit)
    #
    #
    #     arguments = [[linear_params, config] for config in self.configs]
    #
    #     c_count = multiprocessing.cpu_count()
    #     # with multiprocessing.Pool(c_count - 2) as p:
    #     #     cur_results = p.map(self.simulate_parameters_for_config, iterable=arguments)
    #
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=c_count - 2) as executor:
    #         cur_results = executor.map(self.simulate_parameters_for_config, arguments)
    #
    #     results = {}
    #     for i, config, cur_result in zip(range(len(self.configs)), self.configs, cur_results):
    #         results[i] = {
    #             'combined_params': cur_result[1],
    #             'simulation_results': cur_result[0],
    #             'config': config
    #         }
    #
    #     return results
    #
    # def simulate_parameters_for_config(self, args):
    #     linear_params, config = args
    #     combined_params_df = prepare_combined_params(
    #         linear_params,
    #         config.condition_params
    #     )
    #
    #     simulator = self.simulators[config.name]
    #     simulation_results = simulator.run(
    #         param_values=combined_params_df.drop(['param_set_idx', 'condition'], axis=1),
    #     )
    #
    #     return simulation_results, combined_params_df


    def calculate_likelihood_from_simulation(self, simulation_data: dict) -> dict:
        """Calculate log likelihood from pre-computed simulation data."""
        first_config_data = simulation_data[0]
        n_param_sets = len(first_config_data['combined_params']['param_set_idx'].unique())
        total_log_likelihood = np.zeros(n_param_sets)
        circuit_likelihoods = {}

        for config_idx, data in simulation_data.items():
            config = self.configs[config_idx]
            circuit_name = config.name
            circuit_total = np.zeros(n_param_sets)
            condition_likelihoods = {}

            for condition_name, _ in config.condition_params.items():
                condition_data = config.experimental_data[
                    config.experimental_data['condition'] == condition_name
                ]

                if len(condition_data) == 0:
                    raise ValueError(f"No experimental data found for condition: {condition_name}")

                condition_likelihood = compute_condition_likelihood(
                    data['simulation_results'],
                    condition_data,
                    config.tspan,
                    data['combined_params'],
                    condition_name,
                    self.calibration_params
                )

                condition_likelihoods[condition_name] = condition_likelihood.values
                circuit_total += condition_likelihood.values

            circuit_likelihoods[circuit_name] = {
                'total': circuit_total,
                'conditions': condition_likelihoods
            }
            total_log_likelihood += circuit_total

        return {
            'total': total_log_likelihood,
            'circuits': circuit_likelihoods
        }

    def calculate_log_likelihood(self, log_params: np.ndarray) -> dict:
        """
        Calculate log likelihood for parameters in log space.

        Args:
            log_params: Array of shape (n_samples, n_params) or (n_params,) in log space

        Returns:
            Dictionary containing:
                'total': total log likelihood array
                'circuits': dict of circuit likelihoods
                    each circuit contains:
                        'total': total circuit likelihood
                        'conditions': dict of condition likelihoods
        """
        simulation_data = self.simulate_parameters(log_params)
        return self.calculate_likelihood_from_simulation(simulation_data)

    def calculate_log_prior(self, log_params: np.ndarray) -> np.ndarray:
        """
        Calculate log prior probability for parameters in log space.

        Args:
            log_params: Array of shape (n_samples, n_params) or (n_params,) in log space

        Returns:
            Array of total log prior probabilities
        """
        if log_params.ndim == 1:
            log_params = log_params.reshape(1, -1)

        n_samples = log_params.shape[0]
        total_log_prior = np.zeros(n_samples)
        param_priors = {}

        for i, param_name in enumerate(self.parameters_to_fit):
            mean = self.log_means[param_name]
            std = self.log_stds[param_name]
            param_log_prior = -0.5 * ((log_params[:, i] - mean) / std) ** 2
            param_log_prior += -np.log(std) - 0.5 * np.log(2 * np.pi)
            param_priors[param_name] = param_log_prior
            total_log_prior += param_log_prior

        return total_log_prior

    def calculate_log_posterior(self, log_params: np.ndarray) -> np.ndarray:
        """
        Calculate (non normalized) log posterior (log likelihood + log prior)

        Args:
            log_params: Array of shape (n_samples, n_params) or (n_params,) in log space

        Returns:
            Array of total log posterior probabilities
        """
        prior_data = self.calculate_log_prior(log_params)
        likelihood_data = self.calculate_log_likelihood(log_params)

        return prior_data['total'] + likelihood_data['total']


class MCMCAdapter:
    def __init__(self, circuit_fitter):
        """
        Adapter to make CircuitFitter compatible with ParallelTempering

        Parameters
        ----------
        circuit_fitter : CircuitFitter
            Instance of your CircuitFitter class
        """
        self.circuit_fitter = circuit_fitter

    def get_initial_parameters(self):
        """Get initial parameters from prior means in log space"""
        initial_params = []
        for param_name in self.circuit_fitter.parameters_to_fit:
            initial_params.append(self.circuit_fitter.log_means[param_name])
        return np.array(initial_params)

    def get_log_likelihood_function(self):
        """Returns a likelihood function compatible with ParallelTempering"""

        def log_likelihood(params):
            # Reshape from (n_walkers, n_chains, n_params) to (n_samples, n_params)
            original_shape = params.shape
            params_2d = params.reshape(-1, original_shape[-1])

            # Calculate likelihood and reshape back
            likelihood_dict = self.circuit_fitter.calculate_log_likelihood(params_2d)
            return likelihood_dict['total'].reshape(original_shape[:-1])

        return log_likelihood

    def get_log_prior_function(self):
        """Returns a prior function compatible with ParallelTempering"""

        def log_prior(params):
            # Reshape from (n_walkers, n_chains, n_params) to (n_samples, n_params)
            original_shape = params.shape
            params_2d = params.reshape(-1, original_shape[-1])

            # Calculate prior and reshape back
            prior_values = self.circuit_fitter.calculate_log_prior(params_2d)
            return prior_values.reshape(original_shape[:-1])

        return log_prior

    def setup_parallel_tempering(self, n_walkers=1, n_chains=10):
        """
        Creates and configures a ParallelTempering instance

        Parameters
        ----------
        n_walkers : int
            Number of walkers per chain
        n_chains : int
            Number of temperature chains

        Returns
        -------
        ParallelTempering
            Configured instance ready for sampling
        """
        return ParallelTempering(
            log_likelihood=self.get_log_likelihood_function(),
            log_prior=self.get_log_prior_function(),
            n_dim=len(self.circuit_fitter.parameters_to_fit),
            n_walkers=n_walkers,
            n_chains=n_chains
        )
