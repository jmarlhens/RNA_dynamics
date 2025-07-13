import numpy as np


class HierarchicalAdaptiveProposal:
    def __init__(self, hierarchical_circuit_fitter):
        self.hierarchical_fitter = hierarchical_circuit_fitter
        self.n_alpha_params = hierarchical_circuit_fitter.n_alpha_params
        self.n_sigma_params = hierarchical_circuit_fitter.n_sigma_params
        self.theta_start_idx = 0
        self.alpha_start_idx = hierarchical_circuit_fitter.alpha_start_idx
        self.sigma_start_idx = hierarchical_circuit_fitter.sigma_start_idx

    def __call__(self, previous_parameter_state=None, proposal_radius=None):
        if previous_parameter_state is None:
            return self.hierarchical_fitter.generate_hierarchical_parameters(n_sets=1)[
                0
            ]

        current_parameter_state = np.array(previous_parameter_state)

        if proposal_radius is None:
            proposal_radius = 0.1 * np.ones(current_parameter_state.shape)

        # Standard random walk for α and Σ hyperparameters only
        hyperparameter_move = np.random.normal(loc=0, scale=proposal_radius)
        proposed_parameter_state = current_parameter_state + hyperparameter_move

        # Zero out θ move (will be resampled)
        proposed_parameter_state[self.theta_start_idx : self.alpha_start_idx] = (
            current_parameter_state[self.theta_start_idx : self.alpha_start_idx]
        )

        # Extract updated α and Σ, then sample fresh θ values
        proposed_parameter_state = (
            self._resample_circuit_parameters_from_hyperparameters(
                proposed_parameter_state
            )
        )

        return proposed_parameter_state

    def _resample_circuit_parameters_from_hyperparameters(
        self, parameter_vector_with_updated_hyperparameters
    ):
        # Extract α and Σ from updated parameter vector
        _, alpha_values, sigma_matrices = (
            self.hierarchical_fitter.split_hierarchical_parameters(
                parameter_vector_with_updated_hyperparameters.reshape(1, -1)
            )
        )

        # Sample fresh θ ~ N(α, Σ) for all circuits
        fresh_circuit_parameters = np.random.multivariate_normal(
            mean=alpha_values[0],
            cov=sigma_matrices[0],
            size=self.hierarchical_fitter.n_circuits,
        )

        # Replace θ section in parameter vector
        updated_parameter_vector = parameter_vector_with_updated_hyperparameters.copy()
        updated_parameter_vector[self.theta_start_idx : self.alpha_start_idx] = (
            fresh_circuit_parameters.flatten()
        )

        return updated_parameter_vector
