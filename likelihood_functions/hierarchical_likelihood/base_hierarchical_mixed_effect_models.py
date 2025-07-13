import numpy as np
from ..base import CircuitFitter
from utils.process_experimental_data import prepare_combined_params
from ..likelihood import calculate_likelihoods


class HierarchicalCircuitFitter(CircuitFitter):
    """
    Circuit fitter with hierarchical Bayesian model capabilities.

    The hierarchical model structures parameters as:
    - α: Global mean parameters
    - Σ: Covariance matrix for parameter variation
    - θ_c: Circuit-specific parameters drawn from N(α, Σ)

    FORMULATION:
    Prior     = p(α) · p(Σ)
    Likelihood = ∏_c p(θ_c | α, Σ) · ∏_{i,j,k} p(Y_{i,j,k} | θ_i)
    """

    def __init__(
        self,
        configs,
        parameters_to_fit,
        model_parameters_priors,
        calibration_data,
        sigma_0_squared=0.01,
        shared_parameter_names=None,
        # individual_circuit_posterior_results=None,
    ):
        """Initialize with support for hierarchical structure"""
        super().__init__(
            configs, parameters_to_fit, model_parameters_priors, calibration_data
        )

        # Setup hierarchical model components
        self.n_parameters = len(parameters_to_fit)
        self.n_circuits = len(configs)

        # Add after super().__init__():
        self.shared_parameter_names = shared_parameter_names or []
        self.hierarchical_parameter_names = [
            param
            for param in parameters_to_fit
            if param not in self.shared_parameter_names
        ]
        self.n_shared_params = len(self.shared_parameter_names)
        self.n_hierarchical_params = len(self.hierarchical_parameter_names)

        # Track parameter indices for better organization
        self._setup_parameter_indices()

        # Setup hyperparameters α and Σ
        self._setup_hyperparameters()

        self._setup_parameter_mapping()

        # Verify mapping completeness
        expected_total_mappings = len(self.parameters_to_fit)
        actual_total_mappings = len(self.shared_param_indices) + len(
            self.hierarchical_param_indices
        )
        if actual_total_mappings != expected_total_mappings:
            print(
                f"ERROR: Parameter mapping mismatch. Expected {expected_total_mappings}, got {actual_total_mappings}"
            )
            print(
                f"Unmapped parameters: {set(range(expected_total_mappings)) - {idx[0] for idx in self.shared_param_indices + self.hierarchical_param_indices}}"
            )

        # Setup homoscedastic noise model with configurable parameter
        self.sigma_0_squared = sigma_0_squared

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================

    def _setup_parameter_indices(self):
        # β: shared parameters identical across circuits
        self.beta_start_idx = 0
        self.n_beta_params = self.n_shared_params

        # θ: circuit-specific hierarchical parameters
        self.theta_start_idx = self.n_beta_params
        self.n_theta_params = self.n_circuits * self.n_hierarchical_params

        # α: population means for hierarchical parameters
        self.alpha_start_idx = self.theta_start_idx + self.n_theta_params
        self.n_alpha_params = self.n_hierarchical_params

        # Σ: covariance for hierarchical parameters
        self.sigma_start_idx = self.alpha_start_idx + self.n_alpha_params
        self.n_sigma_params = (
            self.n_hierarchical_params * (self.n_hierarchical_params + 1)
        ) // 2

        self.n_total_params = (
            self.n_beta_params
            + self.n_theta_params
            + self.n_alpha_params
            + self.n_sigma_params
        )

    def _setup_parameter_mapping(self):
        """Precompute parameter index mappings for fastest simulation reconstruction"""
        self.shared_param_indices = []
        self.hierarchical_param_indices = []

        for param_idx, param_name in enumerate(self.parameters_to_fit):
            if param_name in self.shared_parameter_names:
                shared_idx = self.shared_parameter_names.index(param_name)
                self.shared_param_indices.append((param_idx, shared_idx))
            else:
                hierarchical_idx = self.hierarchical_parameter_names.index(param_name)
                self.hierarchical_param_indices.append((param_idx, hierarchical_idx))

    def _setup_hyperparameters(self, individual_circuit_posterior_results=None):
        """Initialize hyperparameters for mixed effects model"""
        # α (global mean vector for hierarchical parameters only)
        self.alpha = np.array(
            [self.log_means[param] for param in self.hierarchical_parameter_names]
        )
        self.mu_alpha = self.alpha.copy()  # Prior mean for alpha parameters

        # Σ (covariance matrix for hierarchical parameters only)
        self.sigma = np.diag(
            [
                5 * self.log_stds[param] ** 2
                for param in self.hierarchical_parameter_names
            ]
        )

        # Beta prior setup (shared parameters)
        if self.n_shared_params > 0:
            self.beta_prior_means = np.array(
                [self.log_means[param] for param in self.shared_parameter_names]
            )
            self.beta_prior_variances = np.array(
                [self.log_stds[param] ** 2 for param in self.shared_parameter_names]
            )
            self.beta_normalization = -0.5 * self.n_shared_params * np.log(
                2 * np.pi
            ) - 0.5 * np.sum(np.log(self.beta_prior_variances))

        # Alpha hyperprior setup (hierarchical dimensions only)
        self.hierarchical_alpha_covariance = self.sigma.copy()
        self.hierarchical_alpha_covariance_inv = np.linalg.inv(
            self.hierarchical_alpha_covariance
        )
        self.hierarchical_alpha_log_det = np.linalg.slogdet(
            self.hierarchical_alpha_covariance
        )[1]
        self.hierarchical_alpha_normalization = (
            -0.5 * self.n_hierarchical_params * np.log(2 * np.pi)
            - 0.5 * self.hierarchical_alpha_log_det
        )

        # Wishart hyperprior setup
        if individual_circuit_posterior_results is not None:
            estimated_wishart_hyperparameters = (
                self._estimate_wishart_hyperparameters_from_individual_posteriors(
                    individual_circuit_posterior_results
                )
            )
            self.nu = estimated_wishart_hyperparameters["degrees_of_freedom_nu"]
            self.psi = estimated_wishart_hyperparameters["scale_matrix_psi"]
            print(
                f"Estimated ν={self.nu:.1f}, Ψ condition number={np.linalg.cond(self.psi):.2f}"
            )
        else:
            self.nu = self.n_hierarchical_params + 4
            self.psi = 8 * np.eye(self.n_hierarchical_params)

        self._sigma_degree_coefficient = -0.5 * (
            self.nu + self.n_hierarchical_params + 1
        )
        self._sigma_trace_coefficient = -0.5

    def _flatten_covariance(self, cov_matrix):
        """Convert covariance matrix to flattened lower triangle"""
        flat_values = cov_matrix[np.tril_indices(cov_matrix.shape[0])]
        return np.array(flat_values)

    def _unflatten_covariance(self, flat_values):
        """Reconstruct symmetric covariance matrix from flattened values"""
        n = self.n_hierarchical_params
        cov_matrix = np.zeros((n, n))
        indices = np.tril_indices(n)
        cov_matrix[indices] = flat_values
        cov_matrix[(indices[1], indices[0])] = flat_values  # Symmetry
        return cov_matrix

    # reject if matrix.
    # mcmc for inverse wishart
    def _ensure_positive_definite(self, matrix):
        """Ensure matrix is positive definite"""
        # Get eigenvalues
        eigvals = np.linalg.eigvalsh(matrix)

        # If already positive definite, return original
        if np.all(eigvals > 1e-8):
            return matrix

        # Otherwise, adjust eigenvalues to make positive definite
        min_eig = np.min(eigvals)
        if min_eig <= 0:
            matrix += (-min_eig + 1e-8) * np.eye(matrix.shape[0])

        return matrix

    def _prepare_simulation_parameters_vectorized(
        self, beta_parameters, theta_parameters
    ):
        """Zero-loop vectorized parameter reconstruction - maximum performance"""
        n_samples, n_circuits = theta_parameters.shape[:2]
        total_simulation_rows = n_samples * n_circuits

        simulation_parameter_matrix = np.zeros(
            (total_simulation_rows, len(self.parameters_to_fit))
        )

        # Single vectorized shared parameter broadcast - no loops
        if self.shared_param_indices:
            simulation_positions = np.array(
                [mapping[0] for mapping in self.shared_param_indices]
            )
            beta_source_positions = np.array(
                [mapping[1] for mapping in self.shared_param_indices]
            )

            shared_broadcast_matrix = np.repeat(
                beta_parameters[:, beta_source_positions], n_circuits, axis=0
            )
            simulation_parameter_matrix[:, simulation_positions] = (
                shared_broadcast_matrix
            )

        # Single vectorized hierarchical parameter assignment - no loops
        if self.hierarchical_param_indices:
            simulation_positions = np.array(
                [mapping[0] for mapping in self.hierarchical_param_indices]
            )
            theta_source_positions = np.array(
                [mapping[1] for mapping in self.hierarchical_param_indices]
            )

            theta_circuit_flattened = theta_parameters.transpose(1, 0, 2).reshape(
                total_simulation_rows, -1
            )
            simulation_parameter_matrix[:, simulation_positions] = (
                theta_circuit_flattened[:, theta_source_positions]
            )

        return simulation_parameter_matrix

    def _reconstruct_simulation_parameters(
        self, beta_circuit_params, theta_circuit_params
    ):
        """Reconstruct parameters in parameters_to_fit order for simulation"""
        reconstructed_parameters = np.zeros(len(self.parameters_to_fit))

        for simulation_param_idx, parameter_name in enumerate(self.parameters_to_fit):
            if parameter_name in self.shared_parameter_names:
                source_idx = self.shared_parameter_names.index(parameter_name)
                reconstructed_parameters[simulation_param_idx] = beta_circuit_params[
                    source_idx
                ]
            else:
                source_idx = self.hierarchical_parameter_names.index(parameter_name)
                reconstructed_parameters[simulation_param_idx] = theta_circuit_params[
                    source_idx
                ]

        return reconstructed_parameters

    def generate_hierarchical_parameters(self, n_sets=20):
        parameters_array = np.zeros((n_sets, self.n_total_params))

        # Beta: shared parameters (only if they exist)
        if self.n_shared_params > 0:
            parameters_array[:, : self.n_beta_params] = np.random.normal(
                self.beta_prior_means,
                np.sqrt(self.beta_prior_variances),
                (n_sets, self.n_shared_params),
            )

        # Theta: hierarchical parameters per circuit
        hierarchical_parameters_per_circuit = np.tile(
            self.alpha, (n_sets, self.n_circuits, 1)
        )
        parameters_array[:, self.theta_start_idx : self.alpha_start_idx] = (
            hierarchical_parameters_per_circuit.reshape(n_sets, -1)
        )

        # Alpha: global hierarchical means
        parameters_array[:, self.alpha_start_idx : self.sigma_start_idx] = self.alpha

        # Sigma: flattened hierarchical covariance
        flattened_sigma = self._flatten_covariance(self.sigma)
        parameters_array[:, self.sigma_start_idx :] = flattened_sigma

        return parameters_array

    def split_hierarchical_parameters(self, parameters_array):
        if parameters_array.ndim == 1:
            parameters_array = parameters_array.reshape(1, -1)

        beta_parameters = parameters_array[:, : self.n_beta_params]

        theta_flattened = parameters_array[
            :, self.theta_start_idx : self.alpha_start_idx
        ]
        theta_parameters = theta_flattened.reshape(
            parameters_array.shape[0], self.n_circuits, self.n_hierarchical_params
        )

        alpha_parameters = parameters_array[
            :, self.alpha_start_idx : self.sigma_start_idx
        ]

        sigma_flattened = parameters_array[:, self.sigma_start_idx :]
        sigma_matrices = np.zeros(
            (
                parameters_array.shape[0],
                self.n_hierarchical_params,
                self.n_hierarchical_params,
            )
        )
        for sample_index in range(parameters_array.shape[0]):
            sigma_matrices[sample_index] = self._unflatten_covariance(
                sigma_flattened[sample_index]
            )
            sigma_matrices[sample_index] = self._ensure_positive_definite(
                sigma_matrices[sample_index]
            )

        return beta_parameters, theta_parameters, alpha_parameters, sigma_matrices

    # =========================================================================
    # PRIOR CALCULATION (HYPERPARAMETERS ONLY)
    # =========================================================================

    # def _calculate_wishart_prior(self, sigma_matrices, log_det_sigmas, valid_indices, log_prior_sigma):
    #     for idx in valid_indices:
    #         try:
    #             psi_sigma_inv = np.linalg.solve(sigma_matrices[idx], self.psi.T).T
    #             trace_term = self._sigma_trace_coefficient * np.trace(psi_sigma_inv)
    #             log_prior_sigma[idx] = self._sigma_degree_coefficient * log_det_sigmas[idx] + trace_term
    #         except np.linalg.LinAlgError:
    #             log_prior_sigma[idx] = -np.inf
    #
    #     return log_prior_sigma

    def calculate_hyperparameter_prior(self, parameters_array):
        beta_parameters, theta_parameters, alpha_parameters, sigma_matrices = (
            self.split_hierarchical_parameters(parameters_array)
        )
        n_samples = alpha_parameters.shape[0]

        # Beta prior: independent shared parameters
        if self.n_shared_params > 0:
            beta_deviations = beta_parameters - self.beta_prior_means[None, :]
            log_prior_beta = (
                -0.5
                * np.sum(
                    (beta_deviations**2) / self.beta_prior_variances[None, :], axis=1
                )
                + self.beta_normalization
            )
        else:
            log_prior_beta = np.zeros(n_samples)

        # Alpha prior: use hierarchical dimensions and precomputed values
        alpha_deviations = (
            alpha_parameters - self.alpha[None, :]
        )  # self.alpha already hierarchical-only
        quadratic_forms = np.einsum(
            "ni,ij,nj->n",
            alpha_deviations,
            self.hierarchical_alpha_covariance_inv,
            alpha_deviations,
        )
        log_prior_alpha = -0.5 * quadratic_forms + self.hierarchical_alpha_normalization

        # --------------------------------------------------------------------
        # ---------------------- Vectorised sigma prior ----------------------
        # --------------------------------------------------------------------
        signs, log_det_sigmas = np.linalg.slogdet(sigma_matrices)
        positive_definite_mask = signs > 0

        # Initialize all to -inf, then compute for valid matrices
        log_prior_sigma = np.full(n_samples, -np.inf)

        if np.any(positive_definite_mask):
            valid_indices = np.where(positive_definite_mask)[0]
            valid_sigma_matrices = sigma_matrices[valid_indices]

            # Compute determinant terms for valid matrices
            valid_log_dets = log_det_sigmas[valid_indices]
            determinant_terms = self._sigma_degree_coefficient * valid_log_dets
            # Compute trace terms: tr(Ψ Σ⁻¹) for each valid matrix
            psi_sigma_inv_batch = np.linalg.solve(
                valid_sigma_matrices, np.tile(self.psi.T, (len(valid_indices), 1, 1))
            )
            trace_terms = self._sigma_trace_coefficient * np.trace(
                psi_sigma_inv_batch, axis1=1, axis2=2
            )
            log_prior_sigma[valid_indices] = determinant_terms + trace_terms

        return {
            "log_prior_beta": log_prior_beta,
            "log_prior_alpha": log_prior_alpha,
            "log_prior_sigma": log_prior_sigma,
            "total": log_prior_beta + log_prior_alpha + log_prior_sigma,
        }

    # =========================================================================
    # NEW: CIRCUIT PARAMETER LIKELIHOOD
    # =========================================================================

    def calculate_circuit_parameter_likelihood(self, params):
        """
        Calculate likelihood of circuit parameters given hyperparameters:
        ∏_{C_i} P(θ_{C_i} | α, Σ)

        Returns:
            dict with:
            - 'log_likelihood_per_circuit': array of shape (n_samples, n_circuits)
            - 'total': sum across all circuits for each sample
        """
        # Split parameters
        beta_params, theta_params, alpha_params, sigma_matrices = (
            self.split_hierarchical_parameters(params)
        )

        n_samples = params.shape[0]
        log_likelihood_per_circuit = np.zeros((n_samples, self.n_circuits))

        for sample_idx in range(n_samples):
            try:
                sigma_inv = np.linalg.inv(sigma_matrices[sample_idx])
                log_det_sigma = np.log(np.linalg.det(sigma_matrices[sample_idx]))

                for circuit_idx in range(self.n_circuits):
                    theta_diff = (
                        theta_params[sample_idx, circuit_idx] - alpha_params[sample_idx]
                    )

                    circuit_log_likelihood = -0.5 * np.dot(
                        theta_diff, np.dot(sigma_inv, theta_diff)
                    )
                    circuit_log_likelihood -= (
                        0.5 * self.n_hierarchical_params * np.log(2 * np.pi)
                    )
                    circuit_log_likelihood -= 0.5 * log_det_sigma

                    log_likelihood_per_circuit[sample_idx, circuit_idx] = (
                        circuit_log_likelihood
                    )

            except np.linalg.LinAlgError:
                log_likelihood_per_circuit[sample_idx, :] = -np.inf

        return {
            "log_likelihood_per_circuit": log_likelihood_per_circuit,
            "total": np.sum(log_likelihood_per_circuit, axis=1),
        }

    # =========================================================================
    # DATA LIKELIHOOD
    # =========================================================================

    def simulate_hierarchical_circuit_grouped_parameters(
        self, circuit_grouped_log_params: np.ndarray, n_samples: int
    ) -> dict:
        """
        Simulate circuit-grouped parameters where each circuit gets its own parameter subset.

        Args:
            circuit_grouped_log_params: Array of shape (n_samples * n_circuits, n_params)
                                       grouped as [s0c0, s1c0, ..., s0c1, s1c1, ...]
            n_samples: Number of samples per circuit

        Returns:
            Dictionary with simulation results maintaining circuit grouping
        """
        # Convert to linear space
        linear_params = self.log_to_linear_params(
            circuit_grouped_log_params, self.parameters_to_fit
        )
        results = {}

        # Simulate each circuit with its specific parameter subset
        for circuit_idx, circuit_config in enumerate(self.configs):
            # Extract parameter rows for this circuit
            start_row = circuit_idx * n_samples
            end_row = (circuit_idx + 1) * n_samples
            circuit_specific_linear_params = linear_params.iloc[start_row:end_row]

            # Prepare combined params for this circuit only
            circuit_combined_params_df = prepare_combined_params(
                circuit_specific_linear_params, circuit_config.condition_params
            )

            # Simulate this circuit with its parameters
            circuit_simulator = self.simulators[circuit_config.name]
            circuit_simulation_results = circuit_simulator.run(
                param_values=circuit_combined_params_df.drop(
                    ["param_set_idx", "condition"], axis=1
                )
            )

            # Store with circuit-aware indexing
            results[circuit_idx] = {
                "combined_params": circuit_combined_params_df,
                "simulation_results": circuit_simulation_results,
                "config": circuit_config,
            }

        return results

    def calculate_data_likelihood(self, params):
        beta_params, theta_params, alpha_params, sigma_matrices = (
            self.split_hierarchical_parameters(params)
        )
        n_samples, n_circuits, n_hierarchical_parameters = theta_params.shape

        theta_circuit_grouped = self._prepare_simulation_parameters_vectorized(
            beta_params, theta_params
        )

        hierarchical_simulation_results = (
            self.simulate_hierarchical_circuit_grouped_parameters(
                theta_circuit_grouped, n_samples
            )
        )

        numpy_likelihood_results = self._calculate_hierarchical_likelihoods(
            hierarchical_simulation_results, n_samples, n_circuits
        )

        return {
            "total": numpy_likelihood_results["total"],
            "circuit_matrix": numpy_likelihood_results["circuit_matrix"],
            "condition_arrays": numpy_likelihood_results["condition_arrays"],
        }

    def _calculate_hierarchical_likelihoods(
        self, hierarchical_simulation_results, n_samples, n_circuits
    ):
        """
        Pure numpy likelihood calculation. No dictionaries, maximum vectorization.

        Returns:
            total_log_likelihood: (n_samples,) - primary MCMC output
            circuit_likelihood_matrix: (n_samples, n_circuits) - per-circuit breakdowns
            condition_likelihood_arrays: List[array(n_samples, n_conditions_circuit_i)] - detailed breakdowns
        """
        # Primary MCMC output
        total_log_likelihood = np.zeros(n_samples)

        # Structured outputs for analysis
        circuit_likelihood_matrix = np.zeros((n_samples, n_circuits))
        condition_likelihood_arrays = []

        # Vectorized circuit processing
        for circuit_idx in range(n_circuits):
            circuit_data = hierarchical_simulation_results[circuit_idx]
            circuit_config = self.configs[circuit_idx]
            circuit_name = circuit_config.name

            # Collect condition likelihoods: (n_conditions_this_circuit, n_samples)
            circuit_condition_stack = []

            for condition_name in circuit_config.condition_params.keys():
                condition_mask = (
                    circuit_data["combined_params"]["condition"] == condition_name
                )
                sim_indices = circuit_data["combined_params"].index[condition_mask]

                sim_values = np.array(
                    [
                        circuit_data["simulation_results"].observables[i][
                            "obs_Protein_GFP"
                        ]
                        for i in sim_indices
                    ]
                )

                cached_experimental_data = self.experimental_data_cache[circuit_name][
                    condition_name
                ]
                exp_means, exp_vars = (
                    cached_experimental_data["means"],
                    cached_experimental_data["vars"],
                )

                condition_likelihoods = calculate_likelihoods(
                    sim_values, exp_means, exp_vars
                )
                circuit_condition_stack.append(condition_likelihoods)

            # Stack and sum: (n_conditions, n_samples) -> (n_samples,)
            circuit_condition_matrix = np.array(circuit_condition_stack)
            circuit_total_likelihoods = np.sum(circuit_condition_matrix, axis=0)

            # Store in structured arrays
            circuit_likelihood_matrix[:, circuit_idx] = circuit_total_likelihoods
            condition_likelihood_arrays.append(
                circuit_condition_matrix.T
            )  # (n_samples, n_conditions)

            # Vectorized accumulation
            total_log_likelihood += circuit_total_likelihoods

        return {
            "total": total_log_likelihood,
            "circuit_matrix": circuit_likelihood_matrix,
            "condition_arrays": condition_likelihood_arrays,
        }

    # =========================================================================
    # POSTERIOR CALCULATION
    # =========================================================================

    def calculate_hierarchical_posterior(self, params):
        """
        Calculate joint posterior: p(θ, α, Σ | Y)
        = p(α, Σ) · ∏_{C_i} p(θ_{C_i} | α, Σ) · ∏_{C_i, c_j, r_k} p(Y_{C_i,c_j,r_k} | θ_{C_i})

        Returns:
            dict with all components separated and combined
        """
        # Calculate hyperparameter prior
        prior_results = self.calculate_hyperparameter_prior(params)

        # Calculate full likelihood
        likelihood_results = self.calculate_data_likelihood(params)

        # Calculate posterior
        log_posterior = prior_results["total"] + likelihood_results["total"]

        return {
            "log_posterior": log_posterior,
            "prior_components": prior_results,
            "likelihood_components": likelihood_results,
            "log_prior_total": prior_results["total"],
            "log_likelihood_total": likelihood_results["total"],
        }

    # =========================================================================
    # CONVENIENCE FUNCTIONS FOR ACCESSING SPECIFIC COMPONENTS
    # =========================================================================
    # =========================================================================
    # SIMULATION FUNCTIONS
    # =========================================================================
    #
    # def simulate_hierarchical_parameters(self, params):
    #     """
    #     Simulate parameters in hierarchical model structure
    #     Each circuit uses its own specific parameters from θ
    #     """
    #     # Split parameters
    #     theta_params, _, _ = self.split_hierarchical_parameters(params)
    #
    #     # Prepare simulation results
    #     n_samples = params.shape[0]
    #     results = {}
    #
    #     # For each sample
    #     for sample_idx in range(n_samples):
    #         sample_results = {}
    #
    #         # For each circuit
    #         for circuit_idx, config in enumerate(self.configs):
    #             # Extract parameters for this circuit
    #             circuit_log_params = theta_params[sample_idx, circuit_idx]
    #
    #             # Convert to linear space
    #             circuit_linear_params = self.log_to_linear_params(
    #                 circuit_log_params, self.parameters_to_fit
    #             )
    #
    #             # Prepare combined params for all conditions
    #             combined_params_df = (
    #                 prepare_combined_params(  # Use the imported function
    #                     circuit_linear_params, config.condition_params
    #                 )
    #             )
    #
    #             # Simulate
    #             simulator = self.simulators[config.name]
    #             simulation_results = simulator.run(
    #                 param_values=combined_params_df.drop(
    #                     ["param_set_idx", "condition"], axis=1
    #                 ),
    #             )
    #
    #             # Store results
    #             sample_results[circuit_idx] = {
    #                 "combined_params": combined_params_df,
    #                 "simulation_results": simulation_results,
    #                 "config": config,
    #             }
    #
    #         results[sample_idx] = sample_results
    #
    #     return results
    #
    # def simulate_single_circuit(self, circuit_idx: int, log_params: np.ndarray) -> dict:
    #     """
    #     Simulate a single circuit with given parameters
    #
    #     Args:
    #         circuit_idx: Index of the circuit to simulate
    #         log_params: Array of shape (n_samples, n_params) in log space
    #
    #     Returns:
    #         Dictionary with simulation results for the single circuit
    #     """
    #     # Get the specific config
    #     config = self.configs[circuit_idx]
    #
    #     # Convert to linear space for simulation
    #     linear_params = self.log_to_linear_params(log_params, self.parameters_to_fit)
    #
    #     # Prepare combined params for all conditions of this circuit
    #     combined_params_df = prepare_combined_params(
    #         linear_params, config.condition_params
    #     )
    #
    #     # Simulate only this circuit
    #     simulator = self.simulators[config.name]
    #     simulation_results = simulator.run(
    #         param_values=combined_params_df.drop(
    #             ["param_set_idx", "condition"], axis=1
    #         ),
    #     )
    #
    #     return {
    #         "combined_params": combined_params_df,
    #         "simulation_results": simulation_results,
    #         "config": config,
    #     }

    def _estimate_wishart_hyperparameters_from_individual_posteriors(
        self, individual_circuit_posterior_results
    ):
        """Estimate inverse-Wishart hyperparameters from individual circuit posterior means"""
        circuit_posterior_parameter_means = []

        for (
            circuit_name,
            circuit_mcmc_dataframe,
        ) in individual_circuit_posterior_results.items():
            filtered_circuit_posterior = self._apply_mcmc_filtering(
                circuit_mcmc_dataframe
            )
            hierarchical_circuit_mean_parameters = (
                filtered_circuit_posterior[self.hierarchical_parameter_names]
                .mean()
                .values
            )

            circuit_posterior_parameter_means.append(
                hierarchical_circuit_mean_parameters
            )

        empirical_between_circuit_covariance = np.cov(
            np.array(circuit_posterior_parameter_means).T
        )

        degrees_of_freedom_nu = self.n_hierarchical_params + 5
        scale_matrix_psi = empirical_between_circuit_covariance * (
            degrees_of_freedom_nu - self.n_hierarchical_params - 1
        )

        return {
            "degrees_of_freedom_nu": degrees_of_freedom_nu,
            "scale_matrix_psi": scale_matrix_psi,
        }

    def _apply_mcmc_filtering(
        self, circuit_mcmc_dataframe, burn_in_fraction=0.5, target_chain_index=0
    ):
        """Apply burn-in and chain filtering to individual circuit MCMC results"""
        from analysis_and_figures.mcmc_analysis_hierarchical import (
            process_mcmc_data,
        )

        processed_mcmc_data = process_mcmc_data(
            circuit_mcmc_dataframe,
            burn_in=burn_in_fraction,
            chain_idx=target_chain_index,
        )
        return processed_mcmc_data["processed_data"]
