from likelihood_functions.base import MCMCAdapter
from optimization.adaptive_parallel_tempering import ParallelTempering
from likelihood_functions.hierarchical_likelihood.base_hierarchical_adaptive_proposal import (
    HierarchicalAdaptiveProposal,
)


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
        def log_likelihood(hierarchical_parameter_array):
            parameters_flattened, original_shape = (
                self._reshape_parameters_for_parallel_tempering(
                    hierarchical_parameter_array
                )
            )
            likelihood_results = self.hierarchical_fitter.calculate_data_likelihood(
                parameters_flattened
            )
            return self._reshape_likelihood_results_from_parallel_tempering(
                likelihood_results["total"], original_shape
            )

        return log_likelihood

    def get_log_prior_function(self):
        def log_prior(hierarchical_parameter_array):
            parameters_flattened, original_shape = (
                self._reshape_parameters_for_parallel_tempering(
                    hierarchical_parameter_array
                )
            )
            prior_values = self.hierarchical_fitter.calculate_hyperparameter_prior(
                parameters_flattened
            )["total"]
            return self._reshape_likelihood_results_from_parallel_tempering(
                prior_values, original_shape
            )

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

    def setup_adaptive_hierarchical_parallel_tempering(self, n_walkers=5, n_chains=12):
        hierarchical_proposal_function = HierarchicalAdaptiveProposal(
            self.hierarchical_fitter
        )

        return ParallelTempering(
            log_likelihood=self.get_log_likelihood_function(),
            log_prior=self.get_log_prior_function(),
            n_dim=self.hierarchical_fitter.n_total_params,
            n_walkers=n_walkers,
            n_chains=n_chains,
            proposal_function=hierarchical_proposal_function,
        )
