import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from optimization.mcmc_utils import convergence_test
from optimization.optimization_algorithm import OptimizationAlgorithm


class ParallelTempering(OptimizationAlgorithm):

    def __init__(self, log_likelihood, log_prior, n_dim, n_walkers=1, n_chains=10):
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.n_dim = n_dim

        self.n_walkers = n_walkers
        self.n_chains = n_chains

        swap_mask = np.zeros(shape=(n_walkers, int(np.ceil(n_chains / 2) * 2)), dtype=bool)
        swap_mask[:, ::2] = 1
        self.swap_mask = swap_mask
        pass

    def run(self, initial_parameters, n_samples=10 ** 3, target_acceptance_ratio=None,
            adaptive_temperature=True):
        # Variance -> Will be adapted per chain, how to adapt per parameter (e.g. one could use gradient evaluation once in a while to choose variance in dependence to current gradient
        # ? How to adapt number of chains dynamically so that also there a desired acceptance rate is achieved?

        n_walkers = self.n_walkers
        n_chains = self.n_chains

        initial_parameters = np.array(initial_parameters)
        self.temperatures = np.power(2, np.arange(self.n_chains), dtype=float)

        if adaptive_temperature and n_chains <= 2:
            print(
                f"Disabling adaptive temperature for n_chains={n_chains}. Minimal number of chains for adaptive temperature is 3, but more chains are recommended.")
            adaptive_temperature = False

        if adaptive_temperature:
            self.temperatures[-1] = np.inf
            # Value choice follows Vousden et al. 2016
            v_factor = 10 ** 2
            v = int(np.ceil(v_factor / n_walkers))
            t0 = 10 * v  # ToDo Choose in dependence to number of samples to generate
            S = np.log(np.diff(self.temperatures, axis=-1))
            S = S[:-1]
            # Diffs of T_2 - T_1, ..., T_(N-1) - T_(N-2). The diff T_N - T_(N-1) is excluded by purpose following 1 < i < N for the S_i

        variance = 0.1
        self.variance = np.ones(shape=(self.n_walkers, self.n_chains, self.n_dim))
        self.variance = self.variance * np.expand_dims(np.expand_dims(np.arange(1, self.n_chains + 1), axis=0),
                                                       axis=-1)
        self.variance *= variance

        adaptive_proposal_distribution = target_acceptance_ratio is not None and target_acceptance_ratio > 0 and target_acceptance_ratio < 1.0

        adaptive_temperature_stop_iteration = int(n_samples / 2)

        parameters = np.zeros(shape=(n_samples, n_walkers, n_chains, *initial_parameters.shape))
        priors = np.zeros(shape=(n_samples, n_walkers, n_chains))
        likelihoods = np.zeros(shape=(n_samples, n_walkers, n_chains))
        step_accepts = np.zeros(shape=(n_samples, n_walkers, n_chains))
        # swap_accepts = [None] * n_samples
        swap_accepts = []

        params = np.array(initial_parameters)
        likelihood = self.log_likelihood(params)
        prior = self.log_prior(params)
        # max_iN = 0
        for iN in tqdm(range(n_samples)):
            self.beta = 1 / np.expand_dims(self.temperatures, axis=0)

            params, prior, likelihood, step_accept = self.step(params, prior, likelihood, index=iN)
            # swap_accept = np.nan * np.ones(shape=(self.n_walkers, self.n_chains - 1))
            swap_round = iN % 10 == 9
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
            if adaptive_proposal_distribution and iN > 100 and iN % 10 == 0:
                # Considers Windowed average of the last 100 steps
                acc_rate_deviation = step_accepts[max(iN - 1000 + 1, 0):iN + 1] - target_acceptance_ratio
                scaling_params = np.exp((np.mean(acc_rate_deviation, axis=0)))
                self.variance = self.variance * np.expand_dims(scaling_params, axis=-1)

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
            pass
        parameters = np.array(parameters)
        priors = np.array(priors)
        likelihoods = np.array(likelihoods)
        step_accepts = np.array(step_accepts)
        swap_accepts = np.array(swap_accepts)
        # print(f"max_iN: {max_iN}")
        return parameters, priors, likelihoods, step_accepts, swap_accepts

    def step(self, params, prior, likelihood, index):
        move = np.random.normal(loc=0, scale=self.variance)
        proposal = params + move

        proposal_likelihood = self.log_likelihood(proposal)
        proposal_prior = self.log_prior(proposal)
        proposal_prob = self.beta * proposal_likelihood + proposal_prior

        # likelihood = self.log_likelihood(params)
        # prior = self.log_prior(params)
        prob = self.beta * likelihood + prior

        log_diff = proposal_prob - prob
        diff = np.exp(log_diff)
        u = np.random.uniform(size=(self.n_walkers, self.n_chains))
        accept = u < diff

        new_prior = np.where(accept, proposal_prior, prior)
        new_likelihood = np.where(accept, proposal_likelihood, likelihood)

        params_accepts = np.expand_dims(accept, -1)
        new_params = np.where(params_accepts, proposal, params)
        return new_params, new_prior, new_likelihood, accept

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


if __name__ == '__main__':
    n_walkers = 10
    n_chains = 6
    n_samples = 10 ** 5
    target_acceptance_ratio = 0.4
    log_likelihood = log_smile_adapt

    adaptive_temperature = True

    # log_prior = lambda params: np.log(np.all(np.logical_and(params <= 2, params >= -2), axis=-1) * 1)
    def log_prior(params):
        return np.log(np.all(np.logical_and(params <= 2, params >= -2), axis=-1) * 1)

    pt = ParallelTempering(log_likelihood=log_likelihood, log_prior=log_prior, n_dim=2, n_walkers=n_walkers,
                           n_chains=n_chains)
    parameters, priors, likelihoods, step_accepts, swap_accepts = pt.run(initial_parameters=[0, 0], n_samples=n_samples,
                                                                         target_acceptance_ratio=target_acceptance_ratio,
                                                                         adaptive_temperature=adaptive_temperature)
    print("Completed Sampling")

    R_hat = convergence_test(parameters[int(len(parameters) / 2):])
    print(f"Potential Scale Reduction: {R_hat}")

    # R_hat value below 1.2 are favorable
    # tau = integrated_autocorrelation_time(parameters)
    # print("Average Integrated Correlation Times")
    # print(np.mean(tau, axis=0))

    step_acceptance_rates = np.mean(step_accepts, axis=0)
    swap_acceptance_rates = np.mean(swap_accepts, axis=0)

    print("Creating Figures")
    fig, ax = plt.subplots()
    ax.scatter(parameters[:, :, :, 0].reshape(-1), parameters[:, :, :, 1].reshape(-1), alpha=0.1)
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

    # fig, axes = plt.subplots(ncols=n_chains, sharex=True, sharey=True)
    # for iC in range(n_chains):
    #     ax = axes[iC]
    #     ax.plot(parameters[:, :, iC, 0].reshape(-1), parameters[:, :, iC, 1].reshape(-1), alpha=0.1)
    #     ax.scatter(parameters[:, :, iC, 0].reshape(-1), parameters[:, :, iC, 1].reshape(-1), alpha=0.1)
    # plt.show()
