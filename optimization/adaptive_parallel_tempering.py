import os.path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from optimization.mcmc_utils import convergence_test, animate_parameter_trace_2D, plot_traces
from optimization.optimization_algorithm import OptimizationAlgorithm


class ParallelTempering(OptimizationAlgorithm):

    def __init__(self, log_likelihood, log_prior, n_dim, n_walkers=1, n_chains=10, proposal_function=None):
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.n_dim = n_dim

        self.n_walkers = n_walkers
        self.n_chains = n_chains

        swap_mask = np.zeros(shape=(n_walkers, int(np.ceil(n_chains / 2) * 2)), dtype=bool)
        swap_mask[:, ::2] = 1
        self.swap_mask = swap_mask

        self.temperatures = np.power(2, np.arange(self.n_chains), dtype=float)
        self.temperatures[-1] = np.inf
        # Value choice follows Vousden et al. 2016

        # Diffs of T_2 - T_1, ..., T_(N-1) - T_(N-2). The diff T_N - T_(N-1) is excluded by purpose following 1 < i < N for the S_i
        variance = 0.1
        self.variance = np.ones(shape=(self.n_walkers, self.n_chains, self.n_dim))
        self.variance = self.variance * np.expand_dims(np.expand_dims(np.arange(1, self.n_chains + 1), axis=0), axis=-1)
        self.variance *= variance

        if proposal_function is None:
            def adaptive_proposal(prev_state=None, radius=None):

                if prev_state is None:
                    if radius is None:
                        radius = 0.1
                    state = radius * np.random.randn(n_dim)

                else:
                    state = np.array(prev_state)
                    if radius is None:
                        radius = 0.1 * np.ones(state.shape)

                    move = np.random.normal(loc=0,
                                            scale=radius)  # The size is implicitly defined via the shape of radius
                    state = state + move

                return state

            proposal_function = adaptive_proposal

        self.proposal_function = proposal_function
        self.file = None

    def run(self, initial_parameters=None, n_samples=10 ** 3,
            target_acceptance_ratio=None,
            adaptive_temperature=True,
            path=None,
            param_names=None):
        # Variance -> Will be adapted per chain, how to adapt per parameter (e.g. one could use gradient evaluation once in a while to choose variance in dependence to current gradient)
        # ? How to adapt number of chains dynamically so that also there a desired acceptance rate is achieved?

        n_walkers = self.n_walkers
        n_chains = self.n_chains

        save_to_file = path is not None
        self.save_to_file = save_to_file
        self.param_names = param_names
        if save_to_file:
            self.init_file(path)

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
                f"Disabling adaptive temperature for n_chains={n_chains}. Minimal number of chains for adaptive temperature is 3, but more chains are recommended.")
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
            if adaptive_proposal_distribution and iN >= 100 and iN % 10 == 0:
                # Considers Windowed average of the last 100 steps
                acc_rate_deviation = np.mean(step_accepts[max(iN - 100 + 1, 0):iN + 1],
                                             axis=0) - target_acceptance_ratio
                scaling_params = np.exp(0.5 * acc_rate_deviation)
                self.variance = self.variance * np.expand_dims(scaling_params, axis=-1)
                if iN % 50 == 0:
                    print(f"Iteration {iN}:\n", np.mean(step_accepts[max(iN - 100 + 1, 0):iN + 1], axis=0))

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
                self.save_state_in_file(parameters, priors, likelihoods, step_accepts, swap_accepts, index=iN)

            pass
        if save_to_file:
            self.save_state_in_file(parameters, priors, likelihoods, step_accepts, swap_accepts, index=iN)
            self.close_file()

        parameters = np.array(parameters)
        priors = np.array(priors)
        likelihoods = np.array(likelihoods)
        step_accepts = np.array(step_accepts)
        swap_accepts = np.array(swap_accepts)

        # print(f"max_iN: {max_iN}")
        return parameters, priors, likelihoods, step_accepts, swap_accepts

    def step(self, params, prior, likelihood, index):
        # move = np.random.normal(loc=0, scale=np.sqrt(self.variance))
        # proposal = params + move

        proposal = self.proposal_function(prev_state=params, radius=np.sqrt(self.variance))

        proposal_likelihood = self.log_likelihood(proposal)
        proposal_prior = self.log_prior(proposal)
        proposal_tempered_likelihood = self.beta * proposal_likelihood
        # proposal_tempered_likelihood[np.isnan(proposal_tempered_likelihood)] = -np.inf
        proposal_tempered_likelihood[np.tile(self.beta, (proposal_tempered_likelihood.shape[0], 1)) == 0] = 0
        proposal_prob = proposal_tempered_likelihood + proposal_prior

        # likelihood = self.log_likelihood(params)
        # prior = self.log_prior(params)
        tempered_likelihood = self.beta * likelihood
        # tempered_likelihood[np.isnan(tempered_likelihood)] = -np.inf
        tempered_likelihood[np.tile(self.beta, (tempered_likelihood.shape[0], 1)) == 0] = 0
        prob = tempered_likelihood + prior

        log_diff = proposal_prob - prob
        log_diff[proposal_prob == prob] = 0
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

    def init_file(self, path):

        abspath = os.path.abspath(path)
        if self.file is not None and os.path.abspath(self.file.name) != abspath:
            self.close_file()

        self.file = open(abspath, "a")
        print(f"Opened file {self.file.name}")

        param_names = self.param_names
        if param_names is None:
            param_names = [f"Parameter {iX}" for iX in range(self.n_dim)]
        self.write_to_file(
            lines=["iteration,walker,chain," + ",".join(param_names) + ",likelihood,prior,posterior,step_accepted\n"])
        self.start_index = 0

    def write_to_file(self, lines):
        self.file.writelines(lines)
        self.file.flush()
        print(f"Updated file {self.file.name}")

    def save_state_in_file(self, parameters, priors, likelihoods, step_accepts, swap_accepts, index):
        start_index = self.start_index
        end_index = index + 1

        cur_parameters = parameters[start_index:end_index]
        cur_likelihoods = likelihoods[start_index:end_index]
        cur_priors = priors[start_index:end_index]
        cur_step_accepts = step_accepts[start_index:end_index]

        iterations = np.arange(start_index, end_index)
        walkers = np.arange(self.n_walkers)
        chains = np.arange(self.n_chains)

        # Create meshgrid for all combinations
        iter_grid, walker_grid, chain_grid = np.meshgrid(
            iterations, walkers, chains, indexing="ij"
        )

        cols = []
        cols += [iter_grid.flatten(),
                 walker_grid.flatten(),
                 chain_grid.flatten()]
        cols += [cur_parameters[..., iP].flatten() for iP in range(self.n_dim)]
        cols += [cur_likelihoods.flatten(),
                 cur_priors.flatten(),
                 (cur_priors.flatten() + cur_likelihoods.flatten()),
                 cur_step_accepts.flatten()]

        data = np.concatenate([np.expand_dims(col, axis=1) for col in cols], axis=1)

        encoded_data = list(map(lambda row: ",".join(row.astype(str)) + "\n", data))

        self.write_to_file(lines=encoded_data)
        # if end_index - start_index >= 1:
        self.start_index = end_index

        pass

        # #######
        # # Create dictionary to store data
        # data = {
        #     "iteration": iter_grid.flatten(),
        #     "walker": walker_grid.flatten(),
        #     "chain": chain_grid.flatten(),
        # }
        #
        # # Add parameters
        # for i, param_name in enumerate(self.parameter_names):
        #     data[param_name] = self.parameters[..., i].flatten()
        #
        # # Add likelihood, prior, posterior
        # data["likelihood"] = self.likelihoods.flatten()
        # data["prior"] = self.priors.flatten()
        # data["posterior"] = self.likelihoods.flatten() + self.priors.flatten()
        #
        # # Add step acceptance
        # data["step_accepted"] = self.step_accepts.flatten()

    def close_file(self):
        if self.file is not None:
            self.file.close()
            print(f"Closed file {self.file.name}")
            self.file = None

    @staticmethod
    def load_state_from_file(path):
        abspath = os.path.abspath(path)

        swap_accepts = None

        df = pd.read_csv(abspath)

        data = df.values
        n_samples = int(np.max(data[:,0])) + 1
        n_walkers = int(np.max(data[:, 1])) + 1
        n_chains = int(np.max(data[:, 2])) + 1

        data = data.reshape((n_samples, n_walkers, n_chains, -1))


        parameters= data[..., 3:data.shape[-1] - 4]
        likelihoods = data[..., -4]
        priors = data[..., -3]
        posterior = data[..., -2]
        step_accepts = data[..., -1]
        index = n_samples - 1

        return parameters, priors, likelihoods, step_accepts, swap_accepts, index


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

    class AdaptiveProposal:
        def __init__(self, func=None):
            if func is None:
                func = lambda x: np.random.normal(loc=x ** 2)
            self.func = func

        def __call__(self, prev_state=None, radius=None):
            if prev_state is None:
                state = radius * np.random.randn(n_dim)
            else:
                state = np.array(prev_state)
                if radius is None:
                    radius = 0.1 * np.ones(state.shape)

                move = np.random.normal(loc=0, scale=radius)  # The size is implicitly defined via the shape of radius
                state = state + move

                # if len(state.shape) > 1:
                #     state[..., 1:] = self.func(state[..., 0:1])
                # else:
                #     state[1] = self.func(state[0])

            return state

    proposal_function = AdaptiveProposal(func=lambda x: np.random.normal(loc=x ** 2))

    storage_path = "data.csv"
    if os.path.exists(storage_path):
        os.remove(storage_path)

    pt = ParallelTempering(log_likelihood=log_likelihood, log_prior=log_prior,
                           n_dim=n_dim, n_walkers=n_walkers, n_chains=n_chains,
                           proposal_function=proposal_function)
    prev_parameters, priors, likelihoods, step_accepts, swap_accepts = pt.run(initial_parameters=[0, 0],
                                                                              n_samples=n_samples,
                                                                              target_acceptance_ratio=target_acceptance_ratio,
                                                                              adaptive_temperature=adaptive_temperature,
                                                                              path=storage_path,
                                                                              param_names=["x1", "x2"])

    prev_parameters_2, priors_2, likelihoods_2, step_accepts_2, swap_accepts, index = ParallelTempering.load_state_from_file(path=storage_path)

    if not np.all(np.abs(prev_parameters - prev_parameters_2) < 10**(-12)):
        print("Parameters are different")
    if not np.all(np.abs(priors - priors_2) < 10**(-12)):
        print("Priors are different")
    if not np.all(np.abs(likelihoods - likelihoods_2) < 10**(-12)):
        print("Likelihoods are different")
    if not np.all(np.abs(step_accepts - step_accepts_2) < 10**(-12)):
        print("Step Accepts are different")

    parameters = prev_parameters

    # By not reinitializing the parallel tempering object, the previous state will persist
    # One thing to note is, that the adaptive temperature schedule will be activated again.
    # To circumvent this, either set adaptive temperature to False or drop half of the samples generated.
    parameters, priors, likelihoods, step_accepts, swap_accepts = pt.run(initial_parameters=prev_parameters[-1],
                                                                         n_samples=n_samples,
                                                                         target_acceptance_ratio=target_acceptance_ratio,
                                                                         adaptive_temperature=False,
                                                                         path="data.csv",
                                                                         param_names=["x1", "x2"])

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


if __name__ == '__main__':
    test_smile()
    # sampling_test()
    pass
