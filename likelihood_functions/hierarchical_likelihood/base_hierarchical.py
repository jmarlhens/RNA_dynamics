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
        individual_circuit_posterior_results=None,
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
    # UNCHANGED UTILITY FUNCTIONS
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

        # Generate all circuit-specific θ parameters at once
        circuit_params = np.random.multivariate_normal(
            mean=self.alpha, cov=self.sigma, size=(n_sets, self.n_circuits)
        )

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
    # CORRECTED PRIOR CALCULATION (HYPERPARAMETERS ONLY)
    # =========================================================================

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

        n_samples = params.shape[0]
        log_prior_alpha = np.zeros(n_samples)
        log_prior_sigma = np.zeros(n_samples)

        for i in range(n_samples):
            # 1. Prior on α: p(α) = N(μ_α, Σ_α)
            alpha_diff = alpha_params[i] - self.mu_alpha
            try:
                log_prior_alpha[i] = -0.5 * np.dot(
                    alpha_diff, np.dot(self.sigma_alpha_inv, alpha_diff)
                )
                log_prior_alpha[i] -= 0.5 * self.n_parameters * np.log(2 * np.pi)
                log_prior_alpha[i] -= 0.5 * np.log(np.linalg.det(self.sigma_alpha))
            except np.linalg.LinAlgError:
                log_prior_alpha[i] = -np.inf

            # 2. Prior on Σ: p(Σ) = InvWishart(ν, Ψ)
            try:
                sigma_inv = np.linalg.inv(sigma_matrices[i])
                log_det_sigma = np.log(np.linalg.det(sigma_matrices[i]))

                log_prior_sigma[i] = (
                    -0.5 * (self.nu + self.n_parameters + 1) * log_det_sigma
                )
                log_prior_sigma[i] -= 0.5 * np.trace(np.dot(self.psi, sigma_inv))
            except np.linalg.LinAlgError:
                log_prior_sigma[i] = -np.inf

        return {
            "log_prior_alpha": log_prior_alpha,
            "log_prior_sigma": log_prior_sigma,
            "total": log_prior_alpha + log_prior_sigma,
        }

    def calculate_hierarchical_prior(self, params):
        return self.calculate_hyperparameter_prior(params)["total"]

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
    # NEW: DATA LIKELIHOOD
    # =========================================================================

    def calculate_data_likelihood(self, params):
        """
        Calculate likelihood of experimental data given circuit parameters:
        ∏_{C_i, c_j, r_k} P(Y_{C_i,c_j,r_k} | θ_{C_i})

        Returns:
            dict with:
            - 'log_likelihood_per_circuit': dict[sample_idx][circuit_name] = circuit likelihood
            - 'log_likelihood_per_condition': detailed breakdown by condition
            - 'total': total data likelihood for each sample
        """
        # Simulate with hierarchical parameters
        simulation_results = self.simulate_hierarchical_parameters(params)

        n_samples = params.shape[0]
        total_log_likelihood = np.zeros(n_samples)

        # Store detailed results
        sample_circuit_likelihoods = {}
        sample_condition_likelihoods = {}

        for sample_idx in range(n_samples):
            sample_results = simulation_results[sample_idx]
            circuit_likelihoods = {}
            condition_likelihoods = {}

            sample_total = 0

            # Process each circuit
            for circuit_idx, circuit_results in sample_results.items():
                config = self.configs[circuit_idx]
                circuit_name = config.name

                circuit_total = 0
                circuit_conditions = {}

                # Process each condition
                for condition_name, _ in config.condition_params.items():
                    # Get simulation results for this condition
                    condition_mask = (
                        circuit_results["combined_params"]["condition"]
                        == condition_name
                    )
                    sim_indices = circuit_results["combined_params"].index[
                        condition_mask
                    ]

                    # Get cached experimental data
                    cached_data = self.experimental_data_cache[circuit_name][
                        condition_name
                    ]
                    exp_means, exp_vars = cached_data["means"], cached_data["vars"]

                    # Get simulation values
                    sim_values = np.array(
                        [
                            circuit_results["simulation_results"].observables[i][
                                "obs_Protein_GFP"
                            ]
                            for i in sim_indices
                        ]
                    )

                    # Calculate likelihood
                    condition_log_likelihood = calculate_likelihoods(
                        sim_values, exp_means, exp_vars
                    )

                    circuit_conditions[condition_name] = condition_log_likelihood
                    circuit_total += condition_log_likelihood

                circuit_likelihoods[circuit_name] = circuit_total
                condition_likelihoods[circuit_name] = circuit_conditions
                sample_total += circuit_total

            sample_circuit_likelihoods[sample_idx] = circuit_likelihoods
            sample_condition_likelihoods[sample_idx] = condition_likelihoods
            total_log_likelihood[sample_idx] = sample_total

        return {
            "log_likelihood_per_circuit": sample_circuit_likelihoods,
            "log_likelihood_per_condition": sample_condition_likelihoods,
            "total": total_log_likelihood,
        }

    # =========================================================================
    # CORRECTED: COMBINED LIKELIHOOD
    # =========================================================================

    def calculate_hierarchical_likelihood(self, params):
        """
        Calculate complete likelihood:
        ∏_{C_i} P(θ_{C_i} | α, Σ) · ∏_{C_i, c_j, r_k} P(Y_{C_i,c_j,r_k} | θ_{C_i})

        Returns:
            dict with separate components and total
        """
        # Calculate circuit parameter likelihood
        circuit_param_results = self.calculate_circuit_parameter_likelihood(params)

        # Calculate data likelihood
        data_results = self.calculate_data_likelihood(params)

        # Combine
        total_likelihood = circuit_param_results["total"] + data_results["total"]

        return {
            "circuit_parameter_likelihood": circuit_param_results,
            "data_likelihood": data_results,
            "total": total_likelihood,
        }

    # =========================================================================
    # CORRECTED: POSTERIOR CALCULATION
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
        likelihood_results = self.calculate_hierarchical_likelihood(params)

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

    def get_alpha_prior(self, params):
        """Get just the α prior component"""
        results = self.calculate_hyperparameter_prior(params)
        return results["log_prior_alpha"]

    def get_sigma_prior(self, params):
        """Get just the Σ prior component"""
        results = self.calculate_hyperparameter_prior(params)
        return results["log_prior_sigma"]

    def get_circuit_parameter_likelihood(self, params):
        """Get just the circuit parameter likelihood component"""
        results = self.calculate_circuit_parameter_likelihood(params)
        return results["total"]

    def get_data_likelihood(self, params):
        """Get just the data likelihood component"""
        results = self.calculate_data_likelihood(params)
        return results["total"]

    def get_total_prior(self, params):
        """Get total prior: p(α) + p(Σ)"""
        results = self.calculate_hyperparameter_prior(params)
        return results["total"]

    def get_total_likelihood(self, params):
        """Get total likelihood: circuit params + data"""
        results = self.calculate_hierarchical_likelihood(params)
        return results["total"]

    def debug_hierarchical_components(self, params):
        """
        Debug function to see all components separately
        """
        results = self.calculate_hierarchical_posterior(params)

        print("=== HIERARCHICAL MODEL COMPONENTS ===")
        print(f"α Prior: {results['prior_components']['log_prior_alpha']}")
        print(f"Σ Prior: {results['prior_components']['log_prior_sigma']}")
        print(f"Total Prior: {results['log_prior_total']}")
        print()
        print(
            f"Circuit Parameter Likelihood: {results['likelihood_components']['circuit_parameter_likelihood']['total']}"
        )
        print(
            f"Data Likelihood: {results['likelihood_components']['data_likelihood']['total']}"
        )
        print(f"Total Likelihood: {results['log_likelihood_total']}")
        print()
        print(f"Posterior: {results['log_posterior']}")

        return results

    # =========================================================================
    # UNCHANGED SIMULATION FUNCTIONS
    # =========================================================================

    def simulate_hierarchical_parameters(self, params):
        """
        Simulate parameters in hierarchical model structure
        Each circuit uses its own specific parameters from θ
        """
        # Split parameters
        theta_params, _, _ = self.split_hierarchical_parameters(params)

        # Prepare simulation results
        n_samples = params.shape[0]
        results = {}

        # For each sample
        for sample_idx in range(n_samples):
            sample_results = {}

            # For each circuit
            for circuit_idx, config in enumerate(self.configs):
                # Extract parameters for this circuit
                circuit_log_params = theta_params[sample_idx, circuit_idx]

                # Convert to linear space
                circuit_linear_params = self.log_to_linear_params(
                    circuit_log_params, self.parameters_to_fit
                )

                # Prepare combined params for all conditions
                combined_params_df = (
                    prepare_combined_params(  # Use the imported function
                        circuit_linear_params, config.condition_params
                    )
                )

                # Simulate
                simulator = self.simulators[config.name]
                simulation_results = simulator.run(
                    param_values=combined_params_df.drop(
                        ["param_set_idx", "condition"], axis=1
                    ),
                )

                # Store results
                sample_results[circuit_idx] = {
                    "combined_params": combined_params_df,
                    "simulation_results": simulation_results,
                    "config": config,
                }

            results[sample_idx] = sample_results

        return results

    def simulate_single_circuit(self, circuit_idx: int, log_params: np.ndarray) -> dict:
        """
        Simulate a single circuit with given parameters

        Args:
            circuit_idx: Index of the circuit to simulate
            log_params: Array of shape (n_samples, n_params) in log space

        Returns:
            Dictionary with simulation results for the single circuit
        """
        # Get the specific config
        config = self.configs[circuit_idx]

        # Convert to linear space for simulation
        linear_params = self.log_to_linear_params(log_params, self.parameters_to_fit)

        # Prepare combined params for all conditions of this circuit
        combined_params_df = prepare_combined_params(
            linear_params, config.condition_params
        )

        # Simulate only this circuit
        simulator = self.simulators[config.name]
        simulation_results = simulator.run(
            param_values=combined_params_df.drop(
                ["param_set_idx", "condition"], axis=1
            ),
        )

        return {
            "combined_params": combined_params_df,
            "simulation_results": simulation_results,
            "config": config,
        }

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
