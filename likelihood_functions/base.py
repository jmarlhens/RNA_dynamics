import numpy as np
import pandas as pd
from optimization.adaptive_parallel_tempering import ParallelTempering
from utils.process_experimental_data import (
    prepare_experimental_data,
    prepare_combined_params,
)


class CircuitFitter:
    def __init__(
        self,
        configs,
        parameters_to_fit,
        model_parameters_priors,
        calibration_data,
        sigma_0_squared=1e2,
    ):
        """
        Initialize CircuitFitter with caching of experimental data

        Parameters
        ----------
        configs : List of configuration objects
        parameters_to_fit : List of parameter names to fit
        model_parameters_priors : pd.DataFrame of prior distributions for model parameters
        calibration_data : Dictionary containing calibration parameters
        sigma_0_squared : float, optional
        """
        self.configs = configs
        self.parameters_to_fit = parameters_to_fit
        self.model_parameters_priors = model_parameters_priors
        self.calibration_params = calibration_data
        self.simulators = {}
        self.experimental_data_cache = {}

        self._setup_priors()
        self._validate_configs()
        self._setup_simulators()
        self._cache_experimental_data()

        self.sigma_0_squared = sigma_0_squared

    def _cache_experimental_data(self):
        """Pre-calculate and cache experimental data means and variances"""
        for config in self.configs:
            config_cache = {}
            for condition_name, _ in config.condition_params.items():
                condition_data = config.experimental_data[
                    config.experimental_data["condition"] == condition_name
                ]

                if len(condition_data) == 0:
                    raise ValueError(
                        f"No experimental data found for condition: {condition_name}"
                    )

                _, exp_means, exp_vars = prepare_experimental_data(
                    condition_data, config.tspan
                )
                config_cache[condition_name] = {"means": exp_means, "vars": exp_vars}
            self.experimental_data_cache[config.name] = config_cache

    def _validate_configs(self):
        """Check that all configs have the same model"""
        if not self.configs:
            raise ValueError("No configurations provided")

    def _setup_simulators(self):
        """Setup simulatora for simulation"""
        # from pysb.simulator import CupSodaSimulator
        # for config in self.configs:
        #     simulator = CupSodaSimulator(
        #         config.model,
        #         tspan=config.tspan,
        #         verbose=True,
        #     )
        #     self.simulators[config.name] = simulator

        from pysb.simulator import ScipyOdeSimulator

        for config in self.configs:
            simulator = ScipyOdeSimulator(
                config.model, config.tspan, compiler="cython", cleanup=True
            )
            self.simulators[config.name] = simulator

    def _setup_priors(self):
        """Setup prior parameters in log space for easy access"""
        self.log_means = {}
        self.log_stds = {}
        for param_name in self.parameters_to_fit:
            prior_row = self.model_parameters_priors[
                self.model_parameters_priors["Parameter"] == param_name
            ].iloc[0]
            self.log_means[param_name] = np.log10(prior_row["Mean"])
            self.log_stds[param_name] = prior_row["log10stddev"]

    @staticmethod
    def log_to_linear_params(log_params: np.ndarray, param_names: list) -> pd.DataFrame:
        """Convert parameters from log space to linear space."""
        linear_params = 10**log_params
        if log_params.ndim == 1:
            return pd.DataFrame([linear_params], columns=param_names)
        return pd.DataFrame(
            linear_params, columns=param_names, index=range(len(log_params))
        )

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
                linear_params, config.condition_params
            )

            # simulator = ScipyOdeSimulator(config.model, config.tspan, compiler='cython', cleanup=True)
            simulator = self.simulators[config.name]
            simulation_results = simulator.run(
                param_values=combined_params_df.drop(
                    ["param_set_idx", "condition"], axis=1
                ),
            )

            results[i] = {
                "combined_params": combined_params_df,
                "simulation_results": simulation_results,
                "config": config,
            }

        return results

    def _compute_likelihood_vectorized_unified(
        self, circuit_simulation_dict, n_parameter_sets
    ):
        """Unified vectorized computation with breakdown reconstruction capability"""
        total_likelihood_vector = np.zeros(n_parameter_sets)
        circuit_likelihood_breakdown = {}

        for circuit_idx, simulation_results_dict in circuit_simulation_dict.items():
            circuit_configuration = simulation_results_dict["config"]
            circuit_name = circuit_configuration.name

            # Vectorized simulation extraction
            observables_list = simulation_results_dict["simulation_results"].observables
            simulation_matrix = np.array(
                [obs["obs_Protein_GFP"] for obs in observables_list]
            )

            combined_params_df = simulation_results_dict["combined_params"]
            condition_labels = combined_params_df["condition"].values
            parameter_set_indices = combined_params_df["param_set_idx"].values

            # Align experimental means with simulation order
            experimental_means_matrix = np.zeros_like(simulation_matrix)
            for condition_name in circuit_configuration.condition_params.keys():
                condition_mask = condition_labels == condition_name
                cached_experimental_means = self.experimental_data_cache[circuit_name][
                    condition_name
                ]["means"]
                experimental_means_matrix[condition_mask] = cached_experimental_means

            # Vectorized likelihood computation
            likelihood_per_simulation = (
                self._calculate_heteroscedastic_likelihood_vectorized(
                    simulation_matrix, experimental_means_matrix
                )
            )

            # Aggregate by parameter set
            circuit_total_likelihoods = np.bincount(
                parameter_set_indices,
                weights=likelihood_per_simulation,
                minlength=n_parameter_sets,
            )

            # Store breakdown data for reconstruction
            circuit_likelihood_breakdown[circuit_idx] = {
                "circuit_totals": circuit_total_likelihoods,
                "condition_labels": condition_labels,
                "parameter_set_indices": parameter_set_indices,
                "simulation_likelihoods": likelihood_per_simulation,
                "condition_names": list(circuit_configuration.condition_params.keys()),
                "circuit_name": circuit_name,
            }

            total_likelihood_vector += circuit_total_likelihoods

        return total_likelihood_vector, circuit_likelihood_breakdown

    def _reconstruct_breakdown_matrices(
        self, circuit_likelihood_breakdown, n_parameter_sets
    ):
        """Reconstruct legacy matrix format from vectorized results"""
        n_circuits = len(circuit_likelihood_breakdown)
        circuit_likelihood_matrix = np.zeros((n_parameter_sets, n_circuits))
        condition_likelihood_arrays = []

        for circuit_idx, breakdown_data in circuit_likelihood_breakdown.items():
            circuit_likelihood_matrix[:, circuit_idx] = breakdown_data["circuit_totals"]

            # Reconstruct per-condition likelihoods
            condition_names = breakdown_data["condition_names"]
            condition_matrix = np.zeros((n_parameter_sets, len(condition_names)))

            for condition_idx, condition_name in enumerate(condition_names):
                condition_mask = breakdown_data["condition_labels"] == condition_name
                condition_parameter_indices = breakdown_data["parameter_set_indices"][
                    condition_mask
                ]
                condition_likelihoods = breakdown_data["simulation_likelihoods"][
                    condition_mask
                ]

                # Aggregate condition likelihoods by parameter set
                condition_totals = np.bincount(
                    condition_parameter_indices,
                    weights=condition_likelihoods,
                    minlength=n_parameter_sets,
                )
                condition_matrix[:, condition_idx] = condition_totals

            condition_likelihood_arrays.append(condition_matrix)

        return circuit_likelihood_matrix, condition_likelihood_arrays

    def _calculate_heteroscedastic_likelihood_vectorized(
        self, simulation_values, experimental_means_matrix
    ):
        """Vectorized likelihood calculation with aligned experimental means"""
        minimum_signal_threshold = 1e-6
        heteroscedastic_variances = self.sigma_0_squared * np.maximum(
            simulation_values, minimum_signal_threshold
        )

        residuals = simulation_values - experimental_means_matrix
        time_points_count = simulation_values.shape[1]

        return (
            -0.5
            * np.sum((residuals**2) / heteroscedastic_variances, axis=1)
            / time_points_count
        )

    def calculate_likelihood_from_simulation(self, simulation_data: dict) -> dict:
        """Unified MCMC-optimized computation"""
        first_circuit_data = next(iter(simulation_data.values()))
        n_parameter_sets = len(
            first_circuit_data["combined_params"]["param_set_idx"].unique()
        )

        total_likelihoods, _ = self._compute_likelihood_vectorized_unified(
            simulation_data, n_parameter_sets
        )
        return {"total": total_likelihoods}

    def calculate_likelihood_from_simulation_with_breakdown(
        self, simulation_data: dict
    ) -> dict:
        """Unified computation with breakdown reconstruction"""
        first_circuit_data = next(iter(simulation_data.values()))
        n_parameter_sets = len(
            first_circuit_data["combined_params"]["param_set_idx"].unique()
        )

        total_likelihoods, circuit_breakdown_data = (
            self._compute_likelihood_vectorized_unified(
                simulation_data, n_parameter_sets
            )
        )

        circuit_breakdown = {}
        for circuit_idx, breakdown_data in circuit_breakdown_data.items():
            circuit_name = breakdown_data["circuit_name"]
            condition_names = breakdown_data["condition_names"]

            condition_breakdown = {}
            for condition_idx, condition_name in enumerate(condition_names):
                condition_mask = breakdown_data["condition_labels"] == condition_name
                condition_parameter_indices = breakdown_data["parameter_set_indices"][
                    condition_mask
                ]
                condition_likelihoods = breakdown_data["simulation_likelihoods"][
                    condition_mask
                ]

                condition_totals = np.bincount(
                    condition_parameter_indices,
                    weights=condition_likelihoods,
                    minlength=n_parameter_sets,
                )
                condition_breakdown[condition_name] = condition_totals

            circuit_breakdown[circuit_name] = {
                "total": breakdown_data["circuit_totals"],
                "conditions": condition_breakdown,
            }

        return {"total": total_likelihoods, "circuits": circuit_breakdown}

    def _compute_likelihood_matrix_core(
        self, circuit_simulation_dict, n_parameter_sets
    ):
        """Legacy interface: reconstruct matrix format"""
        total_likelihoods, circuit_breakdown_data = (
            self._compute_likelihood_vectorized_unified(
                circuit_simulation_dict, n_parameter_sets
            )
        )
        circuit_likelihood_matrix, condition_likelihood_arrays = (
            self._reconstruct_breakdown_matrices(
                circuit_breakdown_data, n_parameter_sets
            )
        )
        return total_likelihoods, circuit_likelihood_matrix, condition_likelihood_arrays

    def calculate_log_likelihood(self, log_params: np.ndarray) -> dict:
        """MCMC-optimized likelihood calculation"""
        simulation_data = self.simulate_parameters(log_params)
        return self.calculate_likelihood_from_simulation(simulation_data)

    def calculate_comprehensive_circuit_likelihood_analysis(
        self, circuit_simulation_results: dict
    ) -> dict:
        """Legacy method - redirects to breakdown calculation"""
        comprehensive_results = (
            self.calculate_likelihood_from_simulation_with_breakdown(
                circuit_simulation_results
            )
        )
        return comprehensive_results["circuits"]

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

        return prior_data["total"] + likelihood_data["total"]


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
            return likelihood_dict["total"].reshape(original_shape[:-1])

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
            n_chains=n_chains,
        )

    @staticmethod
    def _reshape_parameters_for_parallel_tempering(parameter_array):
        """Standard parameter reshaping for parallel tempering compatibility"""
        original_shape = parameter_array.shape
        parameters_flattened = parameter_array.reshape(-1, original_shape[-1])
        return parameters_flattened, original_shape

    @staticmethod
    def _reshape_likelihood_results_from_parallel_tempering(
        likelihood_results, original_shape
    ):
        """Reshape likelihood results back to parallel tempering structure"""
        return likelihood_results.reshape(original_shape[:-1])
