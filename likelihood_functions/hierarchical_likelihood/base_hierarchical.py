import numpy as np
from ..base import CircuitFitter
from utils.process_experimental_data import prepare_combined_params


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
        # individual_circuit_posterior_results=None,
    ):
        """Initialize with support for hierarchical structure"""
        super().__init__(
            configs, parameters_to_fit, model_parameters_priors, calibration_data
        )

        # Setup hierarchical model components
        self.n_parameters = len(parameters_to_fit)
        self.n_circuits = len(configs)

        # Track parameter indices for better organization
        self._setup_parameter_indices()

        # Setup hyperparameters α and Σ
        self._setup_hyperparameters()

        # Setup homoscedastic noise model with configurable parameter
        self.sigma_0_squared = sigma_0_squared

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================

    def _setup_parameter_indices(self):
        """Setup indices for different parameter groups"""
        # θ parameters (circuit-specific)
        self.n_theta_params = self.n_circuits * self.n_parameters

        # α parameters (global means)
        self.alpha_start_idx = self.n_theta_params
        self.n_alpha_params = self.n_parameters

        # Σ parameters (covariance matrix)
        self.sigma_start_idx = self.alpha_start_idx + self.n_alpha_params
        self.n_sigma_params = (
            self.n_parameters * (self.n_parameters + 1) // 2
        )  # Lower triangle

        # Total parameters
        self.n_total_params = (
            self.n_theta_params + self.n_alpha_params + self.n_sigma_params
        )

    def _setup_hyperparameters(self, individual_circuit_posterior_results=None):
        """Initialize hyperparameters for hierarchical model"""
        # α (global mean vector for parameters)
        self.alpha = np.array(
            [self.log_means[param] for param in self.parameters_to_fit]
        )

        # Σ (covariance matrix for parameters)
        self.sigma = np.diag(
            [self.log_stds[param] ** 2 for param in self.parameters_to_fit]
        )

        # Hyperpriors
        # For α: N(μ_α, Σ_α)
        self.mu_alpha = self.alpha.copy()
        self.sigma_alpha = np.diag(
            [self.log_stds[param] ** 2 for param in self.parameters_to_fit]
        )
        self.sigma_alpha_inv = np.linalg.inv(self.sigma_alpha)
        self._alpha_log_det_sigma = np.linalg.slogdet(self.sigma_alpha)[1]
        self._alpha_normalization_constant = (
            -0.5 * self.n_parameters * np.log(2 * np.pi)
            - 0.5 * self._alpha_log_det_sigma
        )

        # # For Σ: Inverse-Wishart(ν, Ψ) - MODIFIED SECTION
        # if individual_circuit_posterior_results is not None:
        #     estimated_wishart_hyperparameters = (
        #         self._estimate_wishart_hyperparameters_from_individual_posteriors(
        #             individual_circuit_posterior_results
        #         )
        #     )
        #     self.nu = estimated_wishart_hyperparameters["degrees_of_freedom_nu"]
        #     self.psi = estimated_wishart_hyperparameters["scale_matrix_psi"]
        #
        #     print(
        #         f"Estimated ν={self.nu:.1f}, Ψ condition number={np.linalg.cond(self.psi):.2f}"
        #     )
        # else:
        # Original hardcoded values
        self.nu = self.n_parameters - 6
        self.psi = 4 * np.eye(self.n_parameters)

        self._sigma_degree_coefficient = -0.5 * (self.nu + self.n_parameters + 1)
        self._sigma_trace_coefficient = -0.5

    def _matrix_to_cholesky_unconstrained_params(self, covariance_matrix):
        """Transform covariance matrix to unconstrained Cholesky parameters"""
        cholesky_lower_triangular = np.linalg.cholesky(covariance_matrix)

        unconstrained_params = []
        for row_idx in range(covariance_matrix.shape[0]):
            for col_idx in range(row_idx + 1):
                if row_idx == col_idx:
                    # Log-transform diagonal elements for positivity
                    unconstrained_params.append(
                        np.log(cholesky_lower_triangular[row_idx, col_idx])
                    )
                else:
                    # Off-diagonal elements remain unconstrained
                    unconstrained_params.append(
                        cholesky_lower_triangular[row_idx, col_idx]
                    )

        return np.array(unconstrained_params)

    def _cholesky_unconstrained_params_to_matrix(self, unconstrained_cholesky_params):
        """Reconstruct covariance matrix from unconstrained Cholesky parameters"""
        matrix_dimension = self.n_parameters

        # matrix_dimension = self.n_parameters  # Version 2

        cholesky_lower_triangular = np.zeros((matrix_dimension, matrix_dimension))

        param_index = 0
        for row_idx in range(matrix_dimension):
            for col_idx in range(row_idx + 1):
                if row_idx == col_idx:
                    # Exp-transform for positive diagonal elements
                    cholesky_lower_triangular[row_idx, col_idx] = np.exp(
                        unconstrained_cholesky_params[param_index]
                    )
                else:
                    # Direct assignment for off-diagonal elements
                    cholesky_lower_triangular[row_idx, col_idx] = (
                        unconstrained_cholesky_params[param_index]
                    )
                param_index += 1

        return cholesky_lower_triangular @ cholesky_lower_triangular.T

    def generate_hierarchical_parameters(self, n_sets=20):
        """
        Generate parameter sets for the hierarchical model
        Returns array of shape (n_sets, n_total_params)
        """
        # Initialize parameters array
        params = np.zeros((n_sets, self.n_total_params))

        # # Generate all circuit-specific θ parameters at once
        # circuit_params = np.random.multivariate_normal(
        #     mean=self.alpha, cov=self.sigma, size=(n_sets, self.n_circuits)
        # )

        # instead, do n_sets copies of the global mean α
        circuit_params = np.tile(self.alpha, (n_sets, self.n_circuits, 1))

        # Reshape and store in parameters array
        params[:, : self.n_theta_params] = circuit_params.reshape(n_sets, -1)

        # Set α parameters (global means)
        params[:, self.alpha_start_idx : self.alpha_start_idx + self.n_alpha_params] = (
            self.alpha
        )

        # Set Σ parameters (flattened covariance)
        flat_sigma = self._matrix_to_cholesky_unconstrained_params(self.sigma)
        params[:, self.sigma_start_idx :] = flat_sigma

        return params

    def split_hierarchical_parameters(self, params):
        """Split hierarchical parameters into θ, α, and Σ components"""
        if params.ndim == 1:
            params = params.reshape(1, -1)

        # Extract θ parameters (circuit-specific)
        theta_params = params[:, : self.n_theta_params].reshape(
            params.shape[0], self.n_circuits, self.n_parameters
        )

        # Extract α parameters (global means)
        alpha_params = params[
            :, self.alpha_start_idx : self.alpha_start_idx + self.n_alpha_params
        ]

        # Extract Σ parameters (flattened covariance)
        sigma_flat = params[:, self.sigma_start_idx :]

        # Reconstruct Σ matrices
        sigma_matrices = np.zeros(
            (params.shape[0], self.n_parameters, self.n_parameters)
        )

        for sample_index in range(params.shape[0]):
            sigma_matrices[sample_index] = (
                self._cholesky_unconstrained_params_to_matrix(sigma_flat[sample_index])
            )

        return theta_params, alpha_params, sigma_matrices

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

    def calculate_hyperparameter_prior(self, params):
        """
        Calculate prior ONLY on hyperparameters: p(α, Σ) = p(α) · p(Σ)

        Returns:
            dict with separate components:
            - 'log_prior_alpha': log p(α) for each sample
            - 'log_prior_sigma': log p(Σ) for each sample
            - 'total': sum of both components
        """
        # Split parameters
        theta_params, alpha_params, sigma_matrices = self.split_hierarchical_parameters(
            params
        )
        # number of parameters (chains *
        n_samples = alpha_params.shape[0]

        # --------------------------------------------------------------------
        # ---------------------- Vectorised alpha prior ----------------------
        # --------------------------------------------------------------------

        alpha_diff_all = (
            alpha_params - self.mu_alpha[None, :]
        )  # Broadcast to (n_samples, p)
        quadratic_forms = np.einsum(
            "ni,ij,nj->n", alpha_diff_all, self.sigma_alpha_inv, alpha_diff_all
        )
        # alpha_normalization = -0.5 * self.n_parameters * np.log(2 * np.pi) - 0.5 * np.log(
        #     np.linalg.det(self.sigma_alpha))
        log_prior_alpha = -0.5 * quadratic_forms + self._alpha_normalization_constant

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
            "log_prior_alpha": log_prior_alpha,
            "log_prior_sigma": log_prior_sigma,
            "total": log_prior_alpha + log_prior_sigma,
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
        theta_params, alpha_params, sigma_matrices = self.split_hierarchical_parameters(
            params
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
                        0.5 * self.n_parameters * np.log(2 * np.pi)
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
        """
        Calculate likelihood of experimental data given circuit parameters.
        Uses efficient batched simulation - no sample loops.

        Returns:
            dict with:
            - 'log_likelihood_per_circuit': dict[sample_idx][circuit_name] = circuit likelihood
            - 'log_likelihood_per_condition': detailed breakdown by condition
            - 'total': total data likelihood for each sample
        """
        # Extract circuit-specific parameters: (n_samples, n_circuits, n_parameters)
        theta_params, _, _ = self.split_hierarchical_parameters(params)
        n_samples, n_circuits, n_parameters = theta_params.shape

        # Correct flattening: transpose then reshape to group by circuit
        theta_transposed = theta_params.transpose(
            1, 0, 2
        )  # (n_circuits, n_samples, n_parameters)
        theta_circuit_grouped = theta_transposed.reshape(-1, n_parameters)
        # Results in: [s0c0, s1c0, s2c0, ..., s0c1, s1c1, s2c1, ...]

        # Use hierarchical simulation (each circuit gets its own parameter subset)
        hierarchical_simulation_results = (
            self.simulate_hierarchical_circuit_grouped_parameters(
                theta_circuit_grouped, n_samples
            )
        )

        # Calculate likelihoods using pure numpy (no dictionaries)
        numpy_likelihood_results = self._calculate_hierarchical_likelihoods(
            hierarchical_simulation_results, n_samples, n_circuits
        )

        # Return primary result for MCMC performance
        return {
            "total": numpy_likelihood_results["total"],
            "circuit_matrix": numpy_likelihood_results["circuit_matrix"],
            "condition_arrays": numpy_likelihood_results["condition_arrays"],
        }

    def _compute_likelihood_matrix_core(
        self, hierarchical_simulation_results, n_samples
    ):
        """Override: hierarchical parameter structure requires circuit-specific indexing"""
        total_likelihood_vector = np.zeros(n_samples)
        circuit_likelihood_matrix = np.zeros((n_samples, self.n_circuits))
        condition_likelihood_arrays = []

        for circuit_idx in range(self.n_circuits):
            circuit_simulation_data = hierarchical_simulation_results[circuit_idx]
            circuit_configuration = self.configs[circuit_idx]
            circuit_name = circuit_configuration.name
            condition_likelihood_stack = []

            for condition_name in circuit_configuration.condition_params.keys():
                condition_mask = (
                    circuit_simulation_data["combined_params"]["condition"]
                    == condition_name
                )
                simulation_indices = circuit_simulation_data["combined_params"].index[
                    condition_mask
                ]

                simulation_values = np.array(
                    [
                        circuit_simulation_data["simulation_results"].observables[i][
                            "obs_Protein_GFP"
                        ]
                        for i in simulation_indices
                    ]
                )

                experimental_means = self.experimental_data_cache[circuit_name][
                    condition_name
                ]["means"]
                condition_likelihoods = (
                    self._calculate_heteroscedastic_likelihood_vectorized(
                        simulation_values, experimental_means
                    )
                )
                condition_likelihood_stack.append(condition_likelihoods)

            circuit_condition_matrix = np.array(condition_likelihood_stack)
            circuit_total_likelihoods = np.sum(circuit_condition_matrix, axis=0)

            circuit_likelihood_matrix[:, circuit_idx] = circuit_total_likelihoods
            condition_likelihood_arrays.append(circuit_condition_matrix.T)
            total_likelihood_vector += circuit_total_likelihoods

        return (
            total_likelihood_vector,
            circuit_likelihood_matrix,
            condition_likelihood_arrays,
        )

    def _calculate_hierarchical_likelihoods(
        self, hierarchical_simulation_results, n_samples, n_circuits
    ):
        """Simplified delegation to overridden core method"""
        total_likelihoods, circuit_matrix, condition_arrays = (
            self._compute_likelihood_matrix_core(
                hierarchical_simulation_results, n_samples
            )
        )

        return {
            "total": total_likelihoods,
            "circuit_matrix": circuit_matrix,
            "condition_arrays": condition_arrays,
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
        circuit_parameter_likelihood = self.calculate_circuit_parameter_likelihood(
            params
        )
        log_posterior = (
            prior_results["total"]
            + circuit_parameter_likelihood["total"]
            + likelihood_results["total"]
        )

        return {
            "log_posterior": log_posterior,
            "prior_components": prior_results,
            "likelihood_components": likelihood_results,
            "log_prior_total": prior_results["total"],
            "log_likelihood_total": likelihood_results["total"],
        }

    @property
    def hierarchical_parameter_names(self):
        """Parameters with hierarchical structure (θ and α)"""
        return self.parameters_to_fit

    @property
    def shared_parameter_names(self):
        """Parameters shared across circuits (empty in pure hierarchical model)"""
        return []
