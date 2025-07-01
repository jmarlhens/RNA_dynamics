from likelihood_functions.base import MCMCAdapter
from optimization.adaptive_parallel_tempering import ParallelTempering


class HierarchicalMCMCAdapter(MCMCAdapter):
    """Adapter for hierarchical model to use with MCMC"""

    def __init__(self, hierarchical_fitter):
        """Initialize with hierarchical circuit fitter"""
        super().__init__(hierarchical_fitter)
        self.hierarchical_fitter = hierarchical_fitter

    def get_initial_parameters(self):
        """Get initial parameters for hierarchical model"""
        # Generate one set of hierarchical parameters
        return self.hierarchical_fitter.generate_hierarchical_parameters(n_sets=1)[0]

    def get_log_likelihood_function(self):
        """Return likelihood function for hierarchical model"""

        def log_likelihood(params):
            # Reshape to handle parallel tempering structure
            original_shape = params.shape
            params_2d = params.reshape(-1, original_shape[-1])

            # Calculate likelihood using correct method name
            likelihood_results = self.hierarchical_fitter.calculate_data_likelihood(
                params_2d
            )

            # Extract numpy array directly (no dictionary overhead)
            return likelihood_results["total"].reshape(original_shape[:-1])

        return log_likelihood

    def get_log_prior_function(self):
        """Return prior function for hierarchical model"""

        def log_prior(params):
            # Reshape to handle parallel tempering structure
            original_shape = params.shape
            params_2d = params.reshape(-1, original_shape[-1])

            # Calculate prior
            prior_values = self.hierarchical_fitter.calculate_hyperparameter_prior(
                params_2d
            )["total"]

            # Reshape back to original shape
            return prior_values.reshape(original_shape[:-1])

        return log_prior

    def setup_hierarchical_parallel_tempering(self, n_walkers=5, n_chains=12):
        """Setup parallel tempering for hierarchical model"""
        return ParallelTempering(  # Use the imported class, not self.ParallelTempering
            log_likelihood=self.get_log_likelihood_function(),
            log_prior=self.get_log_prior_function(),
            n_dim=self.hierarchical_fitter.n_total_params,
            n_walkers=n_walkers,
            n_chains=n_chains,
        )
