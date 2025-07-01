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

        # For Σ: Inverse-Wishart(ν, Ψ) - MODIFIED SECTION
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
            # Original hardcoded values
            self.nu = self.n_parameters - 6
            self.psi = 4 * np.eye(self.n_parameters)

        self._sigma_degree_coefficient = -0.5 * (self.nu + self.n_parameters + 1)
        self._sigma_trace_coefficient = -0.5

    def _flatten_covariance(self, cov_matrix):
        """Convert covariance matrix to flattened lower triangle"""
        flat_values = cov_matrix[np.tril_indices(cov_matrix.shape[0])]
        return np.array(flat_values)

    def _unflatten_covariance(self, flat_values):
        """Reconstruct symmetric covariance matrix from flattened values"""
        n = self.n_parameters
        cov_matrix = np.zeros((n, n))
        indices = np.tril_indices(n)
        cov_matrix[indices] = flat_values
        cov_matrix[(indices[1], indices[0])] = flat_values  # Symmetry
        return cov_matrix

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
        flat_sigma = self._flatten_covariance(self.sigma)
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
        for i in range(params.shape[0]):
            sigma_matrices[i] = self._unflatten_covariance(sigma_flat[i])
            # Ensure positive definiteness
            sigma_matrices[i] = self._ensure_positive_definite(sigma_matrices[i])

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
            circuit_mean_parameters = (
                filtered_circuit_posterior[self.parameters_to_fit].mean().values
            )
            circuit_posterior_parameter_means.append(circuit_mean_parameters)

        empirical_between_circuit_covariance = np.cov(
            np.array(circuit_posterior_parameter_means).T
        )

        degrees_of_freedom_nu = self.n_parameters + 5  # Moderate prior strength
        scale_matrix_psi = empirical_between_circuit_covariance * (
            degrees_of_freedom_nu - self.n_parameters - 1
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
