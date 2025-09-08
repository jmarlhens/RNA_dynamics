import numpy as np


def create_multivariate_log_prior(
    log_mean_vector: np.ndarray, log_covariance_matrix: np.ndarray
):
    """
    Create a multivariate normal log prior function with precomputed constants.

    Args:
        log_mean_vector: Mean vector for parameters in log space
        log_covariance_matrix: Covariance matrix for parameters in log space

    Returns:
        Function that calculates log prior probabilities for MCMC walker arrays
    """
    # Precompute expensive operations once
    covariance_inv = np.linalg.inv(log_covariance_matrix)
    n_params = len(log_mean_vector)
    log_det_cov = np.linalg.slogdet(log_covariance_matrix)[1]
    log_normalization_constant = -0.5 * (log_det_cov + n_params * np.log(2 * np.pi))

    def calculate_log_prior(walker_params: np.ndarray) -> np.ndarray:
        """
        Calculate log prior probability for MCMC walker parameters.

        Args:
            walker_params: Array of shape (n_walkers, n_chains, n_params) or (n_samples, n_params)

        Returns:
            Array of total log prior probabilities, flattened to match reshaped input
        """
        # Handle both MCMC walker format and standard 2D format
        original_shape = walker_params.shape
        if walker_params.ndim == 3:
            # Reshape from (n_walkers, n_chains, n_params) to (n_samples, n_params)
            flattened_params = walker_params.reshape(-1, original_shape[-1])
        elif walker_params.ndim == 1:
            flattened_params = walker_params.reshape(1, -1)
        else:
            flattened_params = walker_params

        # Center the parameters
        centered_params = flattened_params - log_mean_vector

        # Compute quadratic form: (x - μ)ᵀ Σ⁻¹ (x - μ) for all samples
        quadratic_form = np.einsum(
            "ij,jk,ik->i", centered_params, covariance_inv, centered_params
        )

        # Calculate log prior: -0.5 * quadratic_form + log_normalization_constant
        return -0.5 * quadratic_form + log_normalization_constant

    return calculate_log_prior
