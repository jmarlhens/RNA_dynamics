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
        if prev_state.ndim == 1:
            # Extract single radius vector from (n_walkers, n_chains, n_params) structure
            extracted_radius = radius[0, 0, :] if radius is not None else None
            return self._propose_hierarchical_parameters(prev_state, extracted_radius)
        elif prev_state.ndim == 3:
            # Process (n_walkers, n_chains, n_parameters) structure
            # input from ParallelTempering: (n_walkers, n_chains, n_params)
            walker_chain_shape = prev_state.shape[:-1]  # (n_walkers, n_chains)
            n_walkers, n_chains = walker_chain_shape
            n_parameters_per_vector = prev_state.shape[-1]

            flattened_parameter_vectors = prev_state.reshape(
                -1, n_parameters_per_vector
            )
            flattened_radius_vectors = (
                radius.reshape(-1, n_parameters_per_vector)
                if radius is not None
                else None
            )

            # Process each parameter vector independently
            proposed_parameter_vectors = []
            for vector_idx, current_parameters in enumerate(
                flattened_parameter_vectors
            ):
                current_radius = (
                    flattened_radius_vectors[vector_idx]
                    if flattened_radius_vectors is not None
                    else None
                )
                proposed_parameters = self._propose_hierarchical_parameters(
                    current_parameters, current_radius
                )
                proposed_parameter_vectors.append(proposed_parameters)

            # Reshape back to (n_walkers, n_chains, n_params)
            proposed_parameters_array = np.array(proposed_parameter_vectors)
            return proposed_parameters_array.reshape(
                (*walker_chain_shape, n_parameters_per_vector)
            )

    def _propose_hierarchical_parameters(
        self, current_hierarchical_parameters, current_proposal_radius
    ):
        """Apply hierarchical-aware proposal to single parameter vector"""
        current_hierarchical_parameters = np.array(current_hierarchical_parameters)

        if current_proposal_radius is None:
            current_proposal_radius = 0.1 * np.ones(
                current_hierarchical_parameters.shape
            )

        # Random walk on hyperparameters (α, Σ) only
        hyperparameter_random_walk = np.random.normal(
            loc=0, scale=current_proposal_radius
        )
        proposed_hierarchical_parameters = (
            current_hierarchical_parameters + hyperparameter_random_walk
        )

        # Preserve θ parameters (will be resampled from updated hyperparameters)
        proposed_hierarchical_parameters[
            self.theta_start_idx : self.alpha_start_idx
        ] = current_hierarchical_parameters[self.theta_start_idx : self.alpha_start_idx]

        # Resample circuit parameters from updated hyperparameter distributions
        proposed_hierarchical_parameters = (
            self._resample_circuit_parameters_from_updated_hyperparameters(
                proposed_hierarchical_parameters
            )
        )

        return proposed_hierarchical_parameters

    def _resample_circuit_parameters_from_updated_hyperparameters(
        self, hierarchical_parameters_with_updated_hyperparameters
    ):
        """Sample fresh θ ~ N(α_new, Σ_new) using updated hyperparameters"""
        # Extract α and Σ from parameter vector
        _, updated_alpha_means, updated_sigma_covariances = (
            self.hierarchical_fitter.split_hierarchical_parameters(
                hierarchical_parameters_with_updated_hyperparameters.reshape(1, -1)
            )
        )

        # Validate covariance matrix before sampling
        sigma_eigenvalues = np.linalg.eigvals(updated_sigma_covariances[0])
        min_eigenvalue = np.min(sigma_eigenvalues)

        if min_eigenvalue <= 1e-10:
            # Regularize near-singular covariance matrix
            regularization_strength = 1e-6
            updated_sigma_covariances[0] += regularization_strength * np.eye(
                len(updated_alpha_means[0])
            )

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
