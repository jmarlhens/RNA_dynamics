import os.path
import time

import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from optimization.mcmc_utils import convergence_test, plot_traces
from optimization.optimization_algorithm import OptimizationAlgorithm
from optimization.proposal_methods import RAMProposal, GeneralizedAdaptiveProposal


class ParallelTempering(OptimizationAlgorithm):

    def __init__(self, log_likelihood, log_prior, n_dim, n_walkers=1, n_chains=10, swap_round_period=10,
                 proposal_function=None):
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.n_dim = n_dim

        self.n_walkers = n_walkers
        self.n_chains = n_chains
        self.swap_round_period = swap_round_period

        swap_mask = np.zeros(shape=(n_walkers, int(np.ceil(n_chains / 2) * 2)), dtype=bool)
        swap_mask[:, ::2] = 1
        self.swap_mask = swap_mask

        self.temperatures = np.power(4, np.arange(self.n_chains), dtype=float)
        self.temperatures[-1] = np.inf
        # Value choice follows Vousden et al. 2016

        # Diffs of T_2 - T_1, ..., T_(N-1) - T_(N-2). The diff T_N - T_(N-1) is excluded by purpose following 1 < i < N for the S_i

        if proposal_function is None:
            # proposal_function = RAMProposal(n_walkers, n_chains, n_dim)
            proposal_function = GeneralizedAdaptiveProposal(n_walkers, n_chains, n_dim)

        self.proposal_function = proposal_function
        # self.file = None

    def run(self, initial_parameters=None, n_samples=10 ** 3,
            target_acceptance_ratio=None,
            adaptive_temperature=True,
            mcmc_writer=None):
        # Variance -> Will be adapted per chain, how to adapt per parameter (e.g. one could use gradient evaluation once in a while to choose variance in dependence to current gradient)
        # ? How to adapt number of chains dynamically so that also there a desired acceptance rate is achieved?

        n_walkers = self.n_walkers
        n_chains = self.n_chains

        save_to_file = mcmc_writer is not None

        if initial_parameters is None:
            initial_parameters = self.proposal_function()
        else:
            initial_parameters = np.array(initial_parameters)

        v_factor = 10 ** 2
        v = int(np.ceil(v_factor / n_walkers))
        t0 = 10 * v
        S = np.log(np.diff(self.temperatures, axis=-1))
        S = S[:-1]

        if adaptive_temperature and n_chains <= 2:
            print(
                f"PT: Disabling adaptive temperature for n_chains={n_chains}. Minimal number of chains for adaptive temperature is 3, but more chains are recommended.")
            adaptive_temperature = False

        adaptive_proposal_distribution = target_acceptance_ratio is not None and target_acceptance_ratio > 0 and target_acceptance_ratio < 1.0

        adaptive_temperature_stop_iteration = int(n_samples / 2)

        parameters = np.zeros(shape=(n_samples, n_walkers, n_chains, self.n_dim))
        priors = np.zeros(shape=(n_samples, n_walkers, n_chains))
        likelihoods = np.zeros(shape=(n_samples, n_walkers, n_chains))
        step_accepts = np.zeros(shape=(n_samples, n_walkers, n_chains))
        # swap_accepts = [None] * n_samples
        swap_accepts = []

        params = np.array(initial_parameters)
        likelihood = self.log_likelihood(params)
        prior = self.log_prior(params)
        # max_iN = 0
        start = time.time()
        for iN in tqdm(range(n_samples)):
            self.beta = 1 / np.expand_dims(self.temperatures, axis=0)

            params, prior, likelihood, step_accept, alpha = self.step(params, prior, likelihood, index=iN)
            # swap_accept = np.nan * np.ones(shape=(self.n_walkers, self.n_chains - 1))
            swap_round = iN % self.swap_round_period == 0 and iN > 0
            if swap_round:
                params, prior, likelihood, swap_accept = self.swap(params, prior, likelihood, index=iN)
                swap_accepts.append(swap_accept)

            parameters[iN] = params
            priors[iN] = prior
            likelihoods[iN] = likelihood
            step_accepts[iN] = step_accept

            ##################################
            # Adaptive Proposal Distribution #
            ##################################
            if False and adaptive_proposal_distribution and iN >= 100 and iN % 10 == 0:
                # Considers Windowed average of the last 100 steps
                acc_rate_deviation = np.mean(step_accepts[max(iN - 100 + 1, 0):iN + 1],
                                             axis=0) - target_acceptance_ratio
                scaling_params = np.exp(0.5 * acc_rate_deviation)
                self.variance = self.variance * np.expand_dims(scaling_params, axis=-1)
            if iN % 50 == 0:
                print(f"PT: Iteration {iN}:\n", np.mean(step_accepts[max(iN - 200 + 1, 0):iN + 1], axis=0))

            if adaptive_proposal_distribution:
                """
                Adaptive proposal mechanism following Robust adaptive Metropolis (RAM) by Matti Vihola
                As the computation involves derivation of the cholesky factors has complexity O(d^3), this clearly decreases performance for high dimensional spaces (i.e. d=n_dim)
                """
                self.proposal_function.update_proposal(parameters, priors, likelihoods, step_accepts, alpha, iN)

            ###############################
            # Adaptive Temperature Ladder #
            ###############################

            if adaptive_temperature and swap_round and iN > 20 and iN < adaptive_temperature_stop_iteration:
                kappa = 1 / v * t0 / (iN + t0)
                # Be aware that only every 10th iteration is a swap iteration
                rel_accepts = swap_accepts[max(len(swap_accepts) - 100, 0):]  # Select relevant data
                swap_acceptance_rate = np.mean(rel_accepts, axis=0)  # Average over multiple samples
                swap_acceptance_rate = np.mean(swap_acceptance_rate, axis=0)  # Average over multiple walkers
                swap_rate_diff = -np.diff(swap_acceptance_rate, axis=0)  # Compute the diff over the chains
                S = S + kappa * swap_rate_diff
                temp_diffs = self.temperatures
                temp_diffs[1:-1] = np.exp(S)
                self.temperatures = np.cumsum(temp_diffs)
                # print(f"Swap Acceptance Rate: {swap_acceptance_rate}")
                # print(f"Temperatures: {self.temperatures}")
                # max_iN = max([iN, max_iN])
                pass

            # print(iN)
            if save_to_file and iN % 1000 == 0:
                mcmc_writer.save_state_in_file(parameters, priors, likelihoods, step_accepts, swap_accepts, index=iN)

            pass
        end = time.time()
        print(f"PT: Sampling completed (Duration {end - start})")
        if save_to_file:
            mcmc_writer.save_state_in_file(parameters, priors, likelihoods, step_accepts, swap_accepts, index=iN)
            mcmc_writer.close()

        parameters = np.array(parameters)
        priors = np.array(priors)
        likelihoods = np.array(likelihoods)
        step_accepts = np.array(step_accepts)
        swap_accepts = np.array(swap_accepts)

        print("PT: Wrapping up completed.", flush=True)
        # print(f"max_iN: {max_iN}")
        return parameters, priors, likelihoods, step_accepts, swap_accepts

    def step(self, params, prior, likelihood, index):
        proposal = self.proposal_function(prev_state=params)

        proposal_likelihood = self.log_likelihood(proposal)
        proposal_prior = self.log_prior(proposal)

        proposal_tempered_likelihood = self.beta * proposal_likelihood
        # proposal_tempered_likelihood[np.isnan(proposal_tempered_likelihood)] = -np.inf
        proposal_tempered_likelihood[np.tile(self.beta, (proposal_tempered_likelihood.shape[0], 1)) == 0] = 0
        proposal_prob = proposal_tempered_likelihood + proposal_prior

        tempered_likelihood = self.beta * likelihood
        # tempered_likelihood[np.isnan(tempered_likelihood)] = -np.inf
        tempered_likelihood[np.tile(self.beta, (tempered_likelihood.shape[0], 1)) == 0] = 0
        prob = tempered_likelihood + prior

        log_diff = proposal_prob - prob
        log_diff[proposal_prob == prob] = 0
        alpha = np.exp(log_diff)
        alpha[alpha > 1] = 1
        u = np.random.uniform(size=(self.n_walkers, self.n_chains))
        accept = u < alpha

        new_prior = np.where(accept, proposal_prior, prior)
        new_likelihood = np.where(accept, proposal_likelihood, likelihood)

        params_accepts = np.expand_dims(accept, -1)
        new_params = np.where(params_accepts, proposal, params)
        return new_params, new_prior, new_likelihood, accept, alpha

    def swap(self, params, prior, likelihood, index):
        log_diff = np.diff(likelihood, axis=-1)
        beta_diff = -np.diff(self.beta, axis=-1)

        log_criterion = beta_diff * log_diff
        criterion = np.exp(log_criterion)
        u = np.random.uniform(size=(self.n_walkers, self.n_chains - 1))
        # Ensure in the accepts step that a single chain does not swap to both adjacent chains (it should be possible to check this by using np.diff(accept) which should not yield 0 at a position including a 1 in accept
        proposed_accept = u < criterion
        self.swap_mask = np.roll(self.swap_mask, 1)
        swap_mask = self.swap_mask[:, :self.n_chains - 1]
        accept = np.logical_and(proposed_accept, swap_mask)
        # accept[:, i] defines whether to swap between chain i and i+1.

        # swap_matrice_1 is accept matrice with an additional all zeros entry
        swap_matrice_1 = np.concatenate((accept, np.zeros((self.n_walkers, 1))), axis=1)
        swap_matrice_2 = np.roll(swap_matrice_1, 1, axis=1)

        left_rolled_prior = np.roll(prior, -1, axis=1)
        right_rolled_prior = np.roll(prior, 1, axis=1)
        left_rolled_likelihood = np.roll(likelihood, -1, axis=1)
        right_rolled_likelihood = np.roll(likelihood, 1, axis=1)
        left_rolled_params = np.roll(params, -1, axis=1)
        right_rolled_params = np.roll(params, 1, axis=1)

        new_prior = np.where(swap_matrice_1, left_rolled_prior, prior)
        new_prior = np.where(swap_matrice_2, right_rolled_prior, new_prior)
        new_likelihood = np.where(swap_matrice_1, left_rolled_likelihood, likelihood)
        new_likelihood = np.where(swap_matrice_2, right_rolled_likelihood, new_likelihood)
        new_params = np.where(np.expand_dims(swap_matrice_1, -1), left_rolled_params, params)
        new_params = np.where(np.expand_dims(swap_matrice_2, -1), right_rolled_params, new_params)

        return new_params, new_prior, new_likelihood, accept


def log_smile_adapt(params):
    val = np.exp(-0.5 * (np.sum(np.power(params, 2), axis=-1) - 1) ** 2 / (0.01))
    val *= (np.sum(params * np.array([0, 1]), axis=-1) < -0.2) * 1
    val += np.exp(-0.5 * np.sum(np.power(params + np.array([-0.6, -1]), 2), axis=-1) / 0.01)
    val += np.exp(-0.5 * np.sum(np.power(params + np.array([+0.6, -1]), 2), axis=-1) / 0.01)
    return np.log(val)


def test_smile():
    n_dim = 2
    n_walkers = 4
    n_chains = 10
    n_samples = 10 ** 4
    target_acceptance_ratio = 0.4
    log_likelihood = log_smile_adapt

    adaptive_temperature = True

    # log_prior = lambda params: np.log(np.all(np.logical_and(params <= 2, params >= -2), axis=-1) * 1)
    def log_prior(params):
        return np.log(np.all(np.logical_and(params <= 2, params >= -2), axis=-1) * 1)

    # class AdaptiveProposal:
    #
    #     def __init__(self, n_walkers, n_chains, n_dim, target_acceptance_rate = 0.234, func=None, inital_variance = 0.1):
    #         if func is None:
    #             func = lambda x: np.random.normal(loc=x ** 2)
    #         self.func = func
    #
    #         self.target_acceptance_rate = target_acceptance_rate
    #
    #         variance = np.ones(shape=(n_walkers, n_chains, n_dim, n_dim))
    #         variance = variance * np.expand_dims(np.arange(1, n_chains + 1), axis=(0, -2, -1))
    #         variance = variance * np.eye(n_dim, n_dim)  # Make variables independent initially
    #         variance *= inital_variance
    #         self.L_variance = np.linalg.cholesky(variance)
    #
    #         self.params_shape = (n_walkers, n_chains, n_dim)
    #
    #         c = 0.5 # In the range (0, 1]
    #         e = 0.5 # In the range (0.5, 1)
    #         self.nu = lambda n: c * (n + 1)**(-e)
    #
    #     def __call__(self, prev_state=None):
    #         shape = self.params_shape
    #         # Perform multivariate batch sampling
    #
    #         samples = np.random.normal(size=np.prod(shape))
    #         samples = samples.reshape(shape)
    #         L = self.L_variance
    #         move = np.squeeze(L @ np.expand_dims(samples, axis=-1))
    #         if prev_state is None:
    #             state = move
    #         else:
    #             state = np.array(prev_state)
    #
    #             state = state + move
    #
    #         return state
    #
    #     def update_proposal(self, parameters, priors, likelihoods, step_accepts, alpha, iN):
    #         if iN <= 0:
    #             return
    #         L = self.L_variance
    #
    #         U = parameters[iN] - parameters[iN - 1]
    #         M = np.expand_dims(U, 3) @ np.expand_dims(U, 2)
    #         m = np.power(np.linalg.norm(U), 2)
    #         M = M / m
    #         I = np.expand_dims(np.eye(L.shape[3]), (0, 1))
    #         COV = L @ (I + self.nu(iN) * np.expand_dims(alpha - self.target_acceptance_rate, (2, 3)) * M) @ np.transpose(L, axes=(0, 1, 3, 2))
    #         self.L_variance = np.linalg.cholesky(COV)

    # proposal_function = AdaptiveProposal(func=lambda x: np.random.normal(loc=x ** 2))
    # proposal_function = AdaptiveProposal(n_walkers, n_chains, n_dim, target_acceptance_rate = 0.234, func=None, inital_variance = 0.1)

    storage_path = "data.csv"
    if os.path.exists(storage_path):
        os.remove(storage_path)

    pt = ParallelTempering(log_likelihood=log_likelihood, log_prior=log_prior,
                           n_dim=n_dim, n_walkers=n_walkers, n_chains=n_chains,
                           proposal_function=None)
    prev_parameters, priors, likelihoods, step_accepts, swap_accepts = pt.run(initial_parameters=[0, 0],
                                                                              n_samples=n_samples,
                                                                              target_acceptance_ratio=target_acceptance_ratio,
                                                                              adaptive_temperature=adaptive_temperature)

    # prev_parameters_2, priors_2, likelihoods_2, step_accepts_2, swap_accepts, index = ParallelTempering.load_state_from_file(path=storage_path)

    # if not np.all(np.abs(prev_parameters - prev_parameters_2) < 10**(-12)):
    #     print("Parameters are different")
    # if not np.all(np.abs(priors - priors_2) < 10**(-12)):
    #     print("Priors are different")
    # if not np.all(np.abs(likelihoods - likelihoods_2) < 10**(-12)):
    #     print("Likelihoods are different")
    # if not np.all(np.abs(step_accepts - step_accepts_2) < 10**(-12)):
    #     print("Step Accepts are different")

    parameters = prev_parameters

    # By not reinitializing the parallel tempering object, the previous state will persist
    # One thing to note is, that the adaptive temperature schedule will be activated again.
    # To circumvent this, either set adaptive temperature to False or drop half of the samples generated.
    # parameters, priors, likelihoods, step_accepts, swap_accepts = pt.run(initial_parameters=prev_parameters[-1],
    #                                                                      n_samples=n_samples,
    #                                                                      target_acceptance_ratio=target_acceptance_ratio,
    #                                                                      adaptive_temperature=False,
    #                                                                      path="data.csv",
    #                                                                      param_names=["x1", "x2"])

    print(f"Completed Sampling ({len(parameters)})")

    R_hat = convergence_test(parameters[int(len(parameters) / 2):], per_parameter_test=True)

    print(f"Potential Scale Reduction: {R_hat}")

    # animate_parameter_trace_2D(parameters[:, :, 0])
    # for iW in range(n_walkers):
    plot_traces(data=parameters, file_path=f"traces_walker.pdf", param_names=["x1", "x2"])

    # R_hat value below 1.2 are favorable
    # tau = integrated_autocorrelation_time(parameters)
    # print("Average Integrated Correlation Times")
    # print(np.mean(tau, axis=0))

    step_acceptance_rates = np.mean(step_accepts, axis=0)
    swap_acceptance_rates = np.mean(swap_accepts, axis=0)
    for parameters in [parameters, prev_parameters]:
        print("Creating Figures")
        fig, ax = plt.subplots()
        for iW in range(n_walkers):
            ax.scatter(parameters[:, iW, 0, 0].reshape(-1), parameters[:, iW, 0, 1].reshape(-1), alpha=0.1)
            ax.scatter(parameters[:, iW, 1:, 0].reshape(-1), parameters[:, iW, 1:, 1].reshape(-1), marker=".",
                       alpha=0.1)
        plt.show()

        fig, axes = plt.subplots(ncols=n_chains, sharex=True, sharey=True)
        for iC in range(n_chains):
            ax = axes
            if hasattr(axes, "shape"):
                ax = axes[iC]

            # ax.scatter(parameters[:, :, iC, 0].reshape(-1), parameters[:, :, iC, 1].reshape(-1), alpha=0.1)
            sns.kdeplot(x=parameters[::10, :, iC, 0].reshape(-1), y=parameters[::10, :, iC, 1].reshape(-1), ax=ax,
                        cmap="Reds")
        plt.show()

    fig, axes = plt.subplots(ncols=n_chains, sharex=True, sharey=True)
    for iC in range(n_chains):
        ax = axes[iC]
        for iW in range(n_walkers):
            ax.plot(parameters[:, iW, iC, 0].reshape(-1), parameters[:, iW, iC, 1].reshape(-1), alpha=0.1)
            ax.scatter(parameters[:, iW, iC, 0].reshape(-1), parameters[:, iW, iC, 1].reshape(-1), alpha=0.1)
    plt.show()


def test_multivariate_normal():
    mean = [0.5, 0.5]
    cov = [[0.5, -0.24],
           [-0.24, 0.25]]

    dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)

    ref_samples = dist.rvs(size=10 ** 5)

    def log_multivariate_normal(params):
        val = dist.pdf(params)
        return np.log(val)

    n_dim = 2
    n_walkers = 4
    n_chains = 10
    n_samples = 10 ** 3
    target_acceptance_ratio = 0.234
    log_likelihood = log_multivariate_normal

    adaptive_temperature = True

    def log_prior(params):
        return np.log(np.all(np.logical_and(params <= 2, params >= -2), axis=-1) * 1)

    storage_path = "data.csv"
    if os.path.exists(storage_path):
        os.remove(storage_path)

    pt = ParallelTempering(log_likelihood=log_likelihood, log_prior=log_prior,
                           n_dim=n_dim, n_walkers=n_walkers, n_chains=n_chains,
                           proposal_function=None)
    parameters, priors, likelihoods, step_accepts, swap_accepts = pt.run(initial_parameters=[0, 0],
                                                                         n_samples=n_samples,
                                                                         target_acceptance_ratio=target_acceptance_ratio,
                                                                         adaptive_temperature=adaptive_temperature)

    print(f"Completed Sampling ({len(parameters)})")

    R_hat = convergence_test(parameters[int(len(parameters) / 2):], per_parameter_test=True)

    print(f"Potential Scale Reduction: {R_hat}")

    # animate_parameter_trace_2D(parameters[:, :, 0])
    # for iW in range(n_walkers):
    plot_traces(data=parameters, file_path=f"traces_walker.pdf", param_names=["x1", "x2"])

    # R_hat value below 1.2 are favorable
    # tau = integrated_autocorrelation_time(parameters)
    # print("Average Integrated Correlation Times")
    # print(np.mean(tau, axis=0))

    samples = parameters[n_samples // 2:, :, 0]

    fig, ax = plt.subplots()
    sns.kdeplot(x=samples[..., 0].reshape(-1), y=samples[..., 1].reshape(-1), ax=ax, cmap="Reds")
    sns.kdeplot(x=ref_samples[..., 0].reshape(-1), y=ref_samples[..., 1].reshape(-1), ax=ax, cmap="Blues")

    plt.show()

    covariances = np.array(pt.proposal_function.covariances)[:, :, 0]
    visualize_covariance_evolution(means=parameters[:, 0, 0], covariances=covariances[:, 0])

    step_acceptance_rates = np.mean(step_accepts, axis=0)
    swap_acceptance_rates = np.mean(swap_accepts, axis=0)
    for parameters in [parameters]:
        print("Creating Figures")
        fig, ax = plt.subplots()
        for iW in range(n_walkers):
            ax.scatter(parameters[:, iW, 0, 0].reshape(-1), parameters[:, iW, 0, 1].reshape(-1), alpha=0.1)
            ax.scatter(parameters[:, iW, 1:, 0].reshape(-1), parameters[:, iW, 1:, 1].reshape(-1), marker=".",
                       alpha=0.1)
        plt.show()

        fig, axes = plt.subplots(ncols=n_chains, sharex=True, sharey=True)
        for iC in range(n_chains):
            ax = axes
            if hasattr(axes, "shape"):
                ax = axes[iC]

            # ax.scatter(parameters[:, :, iC, 0].reshape(-1), parameters[:, :, iC, 1].reshape(-1), alpha=0.1)
            sns.kdeplot(x=parameters[::10, :, iC, 0].reshape(-1), y=parameters[::10, :, iC, 1].reshape(-1), ax=ax,
                        cmap="Reds")
        plt.show()

    fig, axes = plt.subplots(ncols=n_chains, sharex=True, sharey=True)
    for iC in range(n_chains):
        ax = axes[iC]
        for iW in range(n_walkers):
            ax.plot(parameters[:, iW, iC, 0].reshape(-1), parameters[:, iW, iC, 1].reshape(-1), alpha=0.1)
            ax.scatter(parameters[:, iW, iC, 0].reshape(-1), parameters[:, iW, iC, 1].reshape(-1), alpha=0.1)
    plt.show()


def sampling_test():
    from scipy.stats import beta
    from numpy.lib.stride_tricks import sliding_window_view

    target_acceptance_ratio = 0.4

    def log_prior(params):
        prior = np.all(np.logical_and(params > -1, params < 1), axis=-1)
        return np.log(prior * 1)

    beta_a, beta_b = 2.3 / 2, 0.6

    def log_likelihood(params):
        # likelihood = np.ones(shape=params.shape[:-1])
        # likelihood = np.sum(np.exp(- np.power(params, 2) / 0.1), axis=-1)
        likelihood = beta.pdf(params, beta_a, beta_b)
        # likelihood = 0
        # for x in np.linspace(-0.5, 0.5, 4):
        #     likelihood += np.abs(x) * np.exp(- np.power(params - x, 2) / 0.001)

        likelihood = np.sum(likelihood, axis=-1)
        return np.log(likelihood)

    init_params = np.array([0])

    n_walkers = 5
    n_chains = 5
    n_samples = 10000

    pt = ParallelTempering(log_likelihood=log_likelihood, log_prior=log_prior, n_dim=len(init_params),
                           n_walkers=n_walkers,
                           n_chains=n_chains)
    parameters, priors, likelihoods, step_accepts, swap_accepts = pt.run(initial_parameters=init_params,
                                                                         n_samples=n_samples,
                                                                         target_acceptance_ratio=target_acceptance_ratio,
                                                                         adaptive_temperature=True)
    best_index = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)
    params = np.exp(parameters[*best_index])
    posterior_samples = parameters[len(parameters) // 2:, :, 0]
    posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[-1])

    R_hat = convergence_test(parameters[int(len(parameters) / 2):])

    bins = np.linspace(-2, 2, 100)
    fig, ax = plt.subplots()

    ax.hist(posterior_samples[:, 0], bins=bins, alpha=0.5)
    samples = beta.rvs(beta_a, beta_b, size=(n_samples * n_walkers) // 2)
    ax.hist(samples, bins=bins, alpha=0.5)

    plt.show()

    step_accepts_sliding_window = np.transpose(sliding_window_view(step_accepts[:, :, :], 100, axis=0),
                                               axes=(0, 3, 1, 2))
    step_accepts_avg = np.mean(step_accepts_sliding_window, axis=1)

    swap_accepts_sliding_window = np.transpose(sliding_window_view(swap_accepts, 50, axis=0), axes=(0, 3, 1, 2))
    swap_accepts_avg = np.mean(swap_accepts_sliding_window, axis=1)

    for iWalker in range(n_walkers):
        fig, axes = plt.subplots(ncols=2)

        for iChain in range(step_accepts_avg.shape[-1]):
            axes[0].plot(np.arange(2) * (len(step_accepts_avg) - 1),
                         np.ones(2) * target_acceptance_ratio + (n_chains - iChain - 1), "k--", alpha=0.5)
            axes[0].plot(np.arange(len(step_accepts_avg)),
                         step_accepts_avg[:, iWalker, iChain] + (n_chains - iChain - 1), label=iChain, alpha=1)

        for iChain in range(swap_accepts_avg.shape[-1]):
            axes[1].plot(np.arange(2) * (len(swap_accepts_avg) - 1), np.ones(2) * 0 + (n_chains - iChain - 1), "r--",
                         alpha=0.5)
            axes[1].plot(np.arange(len(swap_accepts_avg)),
                         swap_accepts_avg[:, iWalker, iChain] + (n_chains - iChain - 1), label=iChain, alpha=1)

        ylim = axes[0].get_ylim()
        axes[1].set_ylim(ylim)
        axes[0].legend()
        axes[1].legend()
        plt.show()

    for samps in [posterior_samples, samples]:
        print(f"Mean {np.mean(samps)}, Variance {np.var(samps)}")


def visualize_covariance_evolution(means, covariances):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.animation as animation

    xlim = np.min(means[:, 0] - 4 * np.sqrt(covariances[:, 0, 0])), np.max(
        means[:, 0] + 4 * np.sqrt(covariances[:, 0, 0]))
    ylim = np.min(means[:, 1] - 4 * np.sqrt(covariances[:, 1, 1])), np.max(
        means[:, 1] + 4 * np.sqrt(covariances[:, 1, 1]))

    def confidence_ellipse(ax, mean, cov, n_std=1.0, **kwargs):
        from matplotlib.transforms import Affine2D
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        ellipse = patches.Ellipse(mean, width, height, angle=angle, fill=False, **kwargs)
        ax.add_patch(ellipse)

    def plot_cov_matrix(axs, data_mean, cov_matrix, sigmas=[1, 2, 3]):
        d = len(data_mean)
        for i in range(d):
            for j in range(d):
                ax = axs[i, j]
                ax.clear()
                if i == j:
                    mu = data_mean[i]
                    std = np.sqrt(cov_matrix[i, i])
                    ax.plot([mu] * 2, [mu - 3 * std, mu + 3 * std], color="black")
                    for s in sigmas:
                        ax.axhline(mu + s * std, color="red", linestyle='--')
                        ax.axhline(mu - s * std, color="red", linestyle='--')
                    ax.set_xlim(mu - 4 * std, mu + 4 * std)
                    ax.set_ylim(mu - 4 * std, mu + 4 * std)
                else:
                    mean = [data_mean[j], data_mean[i]]
                    subcov = cov_matrix[np.ix_([j, i], [j, i])]
                    ax.scatter(*mean, color="black")
                    for s in sigmas:
                        confidence_ellipse(ax, mean, subcov, n_std=s, edgecolor="blue", alpha=0.3)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                ax.set_xticks([])
                ax.set_yticks([])

    # Dummy covariance evolution data (replace with your own)
    # timesteps = 30
    # dims = 4
    # np.random.seed(42)
    # means = np.cumsum(np.random.randn(timesteps, dims), axis=0)
    # covs = np.array([np.eye(dims) + 0.4 * np.random.randn(dims, dims) for _ in range(timesteps)])
    # for i in range(timesteps):
    #     covs[i] = (covs[i] + covs[i].T) / 2 + dims * np.eye(dims)  # Symmetrize and ensure positive-definite
    dims = covariances.shape[-1]
    timesteps = len(covariances)

    fig, axs = plt.subplots(dims, dims, figsize=(2.5 * dims, 2.5 * dims))

    plt.tight_layout()

    def update(frame):
        plot_cov_matrix(axs, means[frame], covariances[frame])
        fig.suptitle(f"Covariance Evolution: timestep {frame}")

    ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=0.1)
    ani.save("covariance_evolution.mp4", writer="ffmpeg")  # Remove/save as needed
    plt.show()


if __name__ == '__main__':
    # test_multivariate_normal()
    test_smile()
    # sampling_test()
    pass
