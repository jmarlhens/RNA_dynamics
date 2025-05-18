import numpy as np
from ..base import CircuitFitter
from ..utils import prepare_combined_params


class HierarchicalCircuitFitter(CircuitFitter):
    """
    Circuit fitter with hierarchical Bayesian model capabilities.

    The hierarchical model structures parameters as:
    - α: Global mean parameters
    - Σ: Covariance matrix for parameter variation
    - θ_c: Circuit-specific parameters drawn from N(α, Σ)
    """

    def __init__(
        self,
        configs,
        parameters_to_fit,
        model_parameters_priors,
        calibration_data,
        sigma_0_squared=0.01,
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

        # Setup heteroscedastic noise model with configurable parameter
        self.sigma_0_squared = sigma_0_squared

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

    def _setup_hyperparameters(self):
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
        self.sigma_alpha = 2.0 * np.eye(self.n_parameters)

        # For Σ: Inverse-Wishart(ν, Ψ)
        self.nu = self.n_parameters + 1
        self.psi = np.eye(self.n_parameters)

    def _setup_noise_model(self):
        """Setup heteroscedastic noise model"""
        # Base variance parameter (could be fitted)
        self.sigma_0_squared = 0.01

    # Matrix utility methods
    def _flatten_covariance(self, cov_matrix):
        """Convert covariance matrix to flattened lower triangle"""
        n = cov_matrix.shape[0]
        flat_values = []
        for i in range(n):
            for j in range(i + 1):  # Lower triangle including diagonal
                flat_values.append(cov_matrix[i, j])
        return np.array(flat_values)

    def _unflatten_covariance(self, flat_values):
        """Reconstruct symmetric covariance matrix from flattened values"""
        n = self.n_parameters
        cov_matrix = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                cov_matrix[i, j] = flat_values[idx]
                if i != j:
                    cov_matrix[j, i] = flat_values[idx]  # Symmetry
                idx += 1
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

    # Parameter generation and manipulation
    def generate_hierarchical_parameters(self, n_sets=20):
        """
        Generate parameter sets for the hierarchical model
        Returns array of shape (n_sets, n_total_params)
        """
        # Initialize parameters array
        params = np.zeros((n_sets, self.n_total_params))

        # Generate circuit-specific θ parameters
        for c in range(self.n_circuits):
            # For each circuit, sample from N(α, Σ)
            circuit_params = np.random.multivariate_normal(
                mean=self.alpha, cov=self.sigma, size=n_sets
            )

            # Store in parameters array
            start_idx = c * self.n_parameters
            end_idx = (c + 1) * self.n_parameters
            params[:, start_idx:end_idx] = circuit_params

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

    # Prior, likelihood, and posterior calculation
    def calculate_hierarchical_prior(self, params):
        """
        Calculate hierarchical prior:
        p(θ, α, Σ) = p(α) · p(Σ) · ∏_c p(θ_c | α, Σ)
        """
        # Split parameters
        theta_params, alpha_params, sigma_matrices = self.split_hierarchical_parameters(
            params
        )

        n_samples = params.shape[0]
        log_prior = np.zeros(n_samples)

        for i in range(n_samples):
            # 1. Prior on α: p(α) = N(μ_α, Σ_α)
            alpha_diff = alpha_params[i] - self.mu_alpha
            try:
                sigma_alpha_inv = np.linalg.inv(self.sigma_alpha)
                log_prior_alpha = -0.5 * np.dot(
                    alpha_diff, np.dot(sigma_alpha_inv, alpha_diff)
                )
                log_prior_alpha -= 0.5 * self.n_parameters * np.log(2 * np.pi)
                log_prior_alpha -= 0.5 * np.log(np.linalg.det(self.sigma_alpha))
            except np.linalg.LinAlgError:
                log_prior_alpha = -np.inf

            # 2. Prior on Σ: p(Σ) = InvWishart(ν, Ψ)
            try:
                sigma_inv = np.linalg.inv(sigma_matrices[i])
                log_det_sigma = np.log(np.linalg.det(sigma_matrices[i]))

                log_prior_sigma = (
                    -0.5 * (self.nu + self.n_parameters + 1) * log_det_sigma
                )
                log_prior_sigma -= 0.5 * np.trace(np.dot(self.psi, sigma_inv))
            # Normalization constant omitted for simplicity
            except np.linalg.LinAlgError:
                log_prior_sigma = -np.inf

            # 3. Prior on θ (circuit-specific parameters): p(θ_c | α, Σ)
            log_prior_theta = 0
            try:
                sigma_inv = np.linalg.inv(sigma_matrices[i])
                log_det_sigma = np.log(np.linalg.det(sigma_matrices[i]))

                for c in range(self.n_circuits):
                    theta_diff = theta_params[i, c] - alpha_params[i]
                    circuit_log_prior = -0.5 * np.dot(
                        theta_diff, np.dot(sigma_inv, theta_diff)
                    )
                    circuit_log_prior -= 0.5 * self.n_parameters * np.log(2 * np.pi)
                    circuit_log_prior -= 0.5 * log_det_sigma
                    log_prior_theta += circuit_log_prior
            except np.linalg.LinAlgError:
                log_prior_theta = -np.inf

            # Combine all priors
            log_prior[i] = log_prior_alpha + log_prior_sigma + log_prior_theta

        return log_prior

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

    def calculate_hierarchical_likelihood(self, params):
        """
        Calculate likelihood using circuit-specific parameters
        Each circuit gets its own θ parameters
        """
        # Simulate with hierarchical parameters
        simulation_results = self.simulate_hierarchical_parameters(params)

        # Initialize likelihood arrays
        n_samples = params.shape[0]
        total_log_likelihood = np.zeros(n_samples)
        circuit_likelihoods = {}

        # Process each sample
        for sample_idx in range(n_samples):
            sample_results = simulation_results[sample_idx]
            sample_circuit_likelihoods = {}

            # Process each circuit
            for circuit_idx, circuit_results in sample_results.items():
                config = self.configs[circuit_idx]
                circuit_name = config.name

                # Initialize likelihoods for this circuit
                circuit_total = 0
                condition_likelihoods = {}

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

                    # Calculate likelihood with heteroscedastic noise
                    log_likelihood = self.calculate_heteroscedastic_likelihood(
                        sim_values, exp_means, exp_vars
                    )

                    condition_likelihoods[condition_name] = log_likelihood
                    circuit_total += log_likelihood

                sample_circuit_likelihoods[circuit_name] = {
                    "total": circuit_total,
                    "conditions": condition_likelihoods,
                }
                total_log_likelihood[sample_idx] += circuit_total

            circuit_likelihoods[sample_idx] = sample_circuit_likelihoods

        return {"total": total_log_likelihood, "circuits": circuit_likelihoods}

    def calculate_heteroscedastic_likelihood(self, sim_values, exp_means, exp_vars):
        """
        Calculate likelihood with heteroscedastic noise model:
        σ²(y) = σ₀² · y
        """
        # Calculate heteroscedastic variance
        hetero_vars = self.sigma_0_squared * np.maximum(sim_values, 1e-6)

        # Calculate residuals
        residuals = sim_values - exp_means

        # Calculate log likelihood
        n_points = exp_means.shape[1]  # Number of time points
        return -0.5 * np.sum((residuals**2) / hetero_vars, axis=1) / n_points

    def calculate_hierarchical_posterior(self, params):
        """Calculate joint posterior: p(θ, α, Σ | Y)"""
        # Calculate prior
        log_prior = self.calculate_hierarchical_prior(params)

        # Calculate likelihood
        likelihood_results = self.calculate_hierarchical_likelihood(params)
        log_likelihood = likelihood_results["total"]

        # Calculate posterior
        log_posterior = log_prior + log_likelihood

        return {
            "log_posterior": log_posterior,
            "log_prior": log_prior,
            "log_likelihood": log_likelihood,
            "likelihood_details": likelihood_results,
        }
