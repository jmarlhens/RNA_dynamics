import numpy as np


class HierarchicalAdaptiveProposal:
    def __init__(self, hierarchical_circuit_fitter):
        self.hierarchical_fitter = hierarchical_circuit_fitter
        self.n_alpha_params = hierarchical_circuit_fitter.n_alpha_params
        self.n_sigma_params = hierarchical_circuit_fitter.n_sigma_params
        self.theta_start_idx = 0
        self.alpha_start_idx = hierarchical_circuit_fitter.alpha_start_idx
        self.sigma_start_idx = hierarchical_circuit_fitter.sigma_start_idx

        # Print parameter structure for debugging
        print("=== HIERARCHICAL PARAMETER STRUCTURE ===")
        print(f"n_circuits: {hierarchical_circuit_fitter.n_circuits}")
        print(f"n_parameters: {hierarchical_circuit_fitter.n_parameters}")
        print(f"n_theta_params: {hierarchical_circuit_fitter.n_theta_params}")
        print(f"n_alpha_params: {self.n_alpha_params}")
        print(f"n_sigma_params: {self.n_sigma_params}")
        print(f"n_total_params: {hierarchical_circuit_fitter.n_total_params}")
        print(f"theta_range: [{self.theta_start_idx}:{self.alpha_start_idx}]")
        print(f"alpha_range: [{self.alpha_start_idx}:{self.sigma_start_idx}]")
        print(
            f"sigma_range: [{self.sigma_start_idx}:{hierarchical_circuit_fitter.n_total_params}]"
        )
        print("==========================================")

    def __call__(self, prev_state=None, radius=None):
        modified_radius = radius.copy() * 0.1
        modified_radius[..., : self.alpha_start_idx] = (
            0  # Works for any number of dimensions
        )

        # Generate random walk with same shape as radius
        random_walk = np.random.normal(0, modified_radius, radius.shape)

        # broadcasting
        # 1sr step: (n_params,) + (n_walkers, n_chains, n_params) -> (n_walkers, n_chains, n_params)
        # 2nd steps: (n_walkers, n_chains, n_params) + (n_walkers, n_chains, n_params) -> (n_walkers, n_chains, n_params)
        proposed = prev_state + random_walk

        # Store original shape and flatten for resampling loop
        original_shape = proposed.shape
        flattened_proposed = proposed.reshape(-1, proposed.shape[-1])

        # Resample theta for each parameter vector
        resampled_vectors = []
        for param_vector in flattened_proposed:
            resampled_vector = (
                self._resample_circuit_parameters_from_updated_hyperparameters(
                    param_vector
                )
            )
            resampled_vectors.append(resampled_vector)

        # Reshape back to original shape
        result = np.array(resampled_vectors).reshape(original_shape)
        return result

    def _resample_circuit_parameters_from_updated_hyperparameters(
        self, hierarchical_parameters_with_updated_hyperparameters
    ):
        """Sample θ ~ N(α_new, Σ_new) using updated hyperparameters"""
        # Extract α and Σ from parameter vector
        _, updated_alpha_means, updated_sigma_covariances = (
            self.hierarchical_fitter.split_hierarchical_parameters(
                hierarchical_parameters_with_updated_hyperparameters.reshape(1, -1)
            )
        )

        # # Validate covariance matrix before sampling
        # sigma_eigenvalues = np.linalg.eigvals(updated_sigma_covariances[0])
        # min_eigenvalue = np.min(sigma_eigenvalues)

        # if min_eigenvalue <= 1e-10:
        #     # Regularize near-singular covariance matrix
        #     regularization_strength = 1e-6
        #     updated_sigma_covariances[0] += regularization_strength * np.eye(
        #         len(updated_alpha_means[0])
        #     )

        # Sample fresh circuit parameters: θ ~ N(α, Σ)
        fresh_circuit_parameters = np.random.multivariate_normal(
            mean=updated_alpha_means[0],
            cov=updated_sigma_covariances[0],
            size=self.hierarchical_fitter.n_circuits,
        )

        # Update θ section in parameter vector
        updated_hierarchical_parameters = (
            hierarchical_parameters_with_updated_hyperparameters.copy()
        )
        updated_hierarchical_parameters[self.theta_start_idx : self.alpha_start_idx] = (
            fresh_circuit_parameters.flatten()
        )

        return updated_hierarchical_parameters
