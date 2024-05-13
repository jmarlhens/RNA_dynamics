import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from optimization.optimization_algorithm import OptimizationAlgorithm


class ParallelTempering(OptimizationAlgorithm):
    def __init__(self):
        self.sample_history = None
        self.posterior_history = None
        self.tempered_posterior_history = None
        self.map_params = None
        pass

    def run(self, log_likelihood, priors: dict, n_chains: int = 10, n_swaps: int = 2, n_samples: int = 1000,
            minimal_temp: float = 10 ** (-3), var_ref: float = 0.01):
        """

        :param log_likelihood:
        :param priors: a dict containing the prior distribution of each parameter in the form of scipy.stats distribution
        :param n_chains:
        :param n_swaps:
        :param n_samples:
        :param minimal_temp:
        :param var_ref:
        :return:
        """
        param_ids = list(priors.keys())
        q = np.power(minimal_temp, 1 / (n_chains - 1))
        temperatures = np.power(q, np.arange(n_chains))

        """
        Sample Initial Parameters from Priors
        """

        cur_samples = []
        cur_tempered_posterior = []
        cur_posterior = []
        for iC in range(n_chains):
            sample_history = {}
            for id in param_ids:
                sample_history[id] = priors[id].rvs()

            log_likelihood_val = log_likelihood(params=sample_history)
            log_prob = np.sum([priors[id].logpdf(sample_history[id]) for id in param_ids])
            # ToDo Crosscheck whether log_prob is the right quantitiy here
            # As we consider logarithmic quantities, the multiplication is an addition
            tempered_posterior = temperatures[iC] * log_likelihood_val + log_prob
            posterior = log_likelihood_val + log_prob

            cur_samples.append(sample_history)
            cur_tempered_posterior.append(tempered_posterior)
            cur_posterior.append(posterior)

        """
        Perform Parallel Tempering
        """

        def swap_elems(arr, i, j):
            elem_i = arr[i]
            elem_j = arr[j]
            arr[i] = elem_j
            arr[j] = elem_i
            return arr

        tempered_posterior_history = [cur_tempered_posterior]
        posterior_history = [cur_posterior]
        sample_history = [cur_samples]

        for iS in range(1, n_samples + 1):
            cur_samples = []
            cur_tempered_posterior = []
            cur_posterior = []
            for iC in range(n_chains):
                cur_temp = temperatures[iC]
                cur_var = var_ref * np.exp(2 * cur_temp)
                prev_samples = sample_history[-1][iC]
                proposal_means = [prev_samples[id] for id in param_ids]
                samples = np.random.normal(proposal_means, cur_var)
                samples = {id: samples[iI] for iI, id in enumerate(param_ids)}

                prev_tempered_posterior = tempered_posterior_history[-1][iC]
                log_likelihood_val = log_likelihood(params=samples)
                log_prob = np.sum([priors[id].logpdf(samples[id]) for id in param_ids])
                # ToDo Crosscheck whether log_prob is the right quantitiy here
                # As we consider logarithmic quantities, the multiplication is an addition
                tempered_posterior = cur_temp * log_likelihood_val + log_prob
                posterior = log_likelihood_val + log_prob

                # Due to logarithms the reference is zero (e.g. np.log(1)=0 and the fraction is a difference
                a = min(0, tempered_posterior - prev_tempered_posterior)
                a = np.exp(a)
                samp = np.random.rand()
                if samp < a:
                    # Accept
                    cur_samples.append(samples)
                    cur_tempered_posterior.append(tempered_posterior)
                    cur_posterior.append(posterior)
                else:
                    # Reject
                    prev_posterior = posterior_history[-1][iC]
                    cur_samples.append(prev_samples)
                    cur_tempered_posterior.append(prev_tempered_posterior)
                    cur_posterior.append(prev_posterior)
                pass

            for iS in range(n_swaps):
                iC = np.random.randint(0, n_chains - 1)
                temp_diff = temperatures[iC] - temperatures[iC + 1]
                rel_log_likelihood_val = log_likelihood(params=cur_samples[iC + 1])
                # post_diff = cur_posterior[iC + 1] - cur_posterior[iC]
                likelihood_diff = rel_log_likelihood_val - log_likelihood_val
                a = min(0, temp_diff * likelihood_diff)
                a = np.exp(a)
                samp = np.random.rand()
                if samp < a:
                    # Accept Swap
                    swap_elems(cur_samples, iC, iC + 1)
                    swap_elems(cur_tempered_posterior, iC, iC + 1)
                    swap_elems(cur_posterior, iC, iC + 1)

            tempered_posterior_history.append(cur_tempered_posterior)
            sample_history.append(cur_samples)
            posterior_history.append(cur_posterior)
            pass
        pass

        posterior_history = np.array(posterior_history)
        tempered_posterior_history = np.array(tempered_posterior_history)

        best_config_id = np.argmax(posterior_history)
        best_config_id = [int(best_config_id / n_chains), int(best_config_id % n_chains)]
        map_params = sample_history[best_config_id[0]][best_config_id[1]]
        self.sample_history = sample_history
        self.posterior_history = posterior_history
        self.tempered_posterior_history = tempered_posterior_history

        self.map_params = map_params
        return sample_history, posterior_history, tempered_posterior_history, map_params


if __name__ == '__main__':
    n_chains = 20
    minimal_temp = 10 ** (-6)
    n_samples = 1000
    n_swaps = 1
    var_ref = 0.1

    n_measurement_samples = 100

    p1 = 5
    p2 = 1

    measurement_data = np.random.randn(n_measurement_samples) * p2 + p1


    def log_likelihood(params):
        p1 = 10 ** params["p1"]
        p2 = 10 ** params["p2"]
        dist = scipy.stats.norm(loc=p1, scale=p2)
        log_like = dist.logpdf(measurement_data)
        log_like = np.sum(log_like)
        return log_like


    print(log_likelihood({"p1": 5, "p2": 1}))
    print(log_likelihood({"p1": 2, "p2": 1}))
    print(log_likelihood({"p1": 10, "p2": 1}))
    print(log_likelihood({"p1": 5, "p2": 5}))
    print(log_likelihood({"p1": 5, "p2": 0.5}))

    opt = ParallelTempering()
    # # a = 10 ** (-5)
    # # b = 10 ** 5
    # # priors = {"p1": scipy.stats.loguniform(a=a, b=b),  # (loc=a, scale=b - a),
    # #           "p2": scipy.stats.loguniform(a=a, b=b)}
    a = -5
    b = 5
    priors = {"p1": scipy.stats.uniform(loc=a, scale=b - a),
              "p2": scipy.stats.uniform(loc=a, scale=b - a)}
    sample_history, posterior_history, tempered_posterior_history, map_params = opt.run(log_likelihood=log_likelihood,
                                                                                        priors=priors,
                                                                                        n_chains=n_chains,
                                                                                        minimal_temp=minimal_temp,
                                                                                        var_ref=var_ref,
                                                                                        n_samples=n_samples,
                                                                                        n_swaps=n_swaps)

    print("Inferred Params are:")
    print({key: 10**(map_params[key]) for key in map_params})

    rel_samples = sample_history[-int(len(sample_history) / 2):]  # Drop others to prevent artefacts from burn in phase
    rel_posterior = posterior_history[-int(len(sample_history) / 2):]
    rel_posterior = np.array(rel_posterior)
    param_ids = list(priors.keys())
    samples = np.array([[[chain[id] for id in param_ids] for chain in step] for step in rel_samples])
    samples = samples.reshape(-1, len(param_ids))

    posterior = rel_posterior.reshape(-1, 1)
    #
    # samples = []
    # posterior = []
    # for pp1 in np.linspace(-1, 1, 50):
    #     for pp2 in np.linspace(-1, 1, 50):
    #         params = {"p1": pp1, "p2": pp2}
    #         samples.append((pp1, pp2))
    #         posterior.append(log_likelihood(params))
    # samples = np.array(samples)
    # posterior = np.array(posterior)
    #
    # # color = np.sign(posterior) * np.log(np.abs(posterior))
    color = np.exp(posterior)

    fig, ax = plt.subplots()
    scatter = ax.scatter(samples[:, 0], samples[:, 1], c=color, alpha=1, cmap="Blues")
    ax.scatter(np.log10(p1), np.log10(p2), marker="x", color="red")
    ax.scatter(map_params["p1"], map_params["p2"], marker="x", color="magenta")
    ax.set_xlabel(param_ids[0])
    ax.set_ylabel(param_ids[1])
    ax.set_xlim((a, b))
    ax.set_ylim((a, b))
    plt.colorbar(scatter)
    plt.show()

    fig, ax = plt.subplots()
    ax.hist2d(samples[:, 0], samples[:, 1], bins=np.linspace(a, b, 100), alpha=1)  # , cmap="Blues")
    ax.scatter(np.log10(p1), np.log10(p2), marker="x", color="red")
    ax.scatter(map_params["p1"], map_params["p2"], marker="x", color="magenta")
    ax.set_xlabel(param_ids[0])
    ax.set_ylabel(param_ids[1])
    ax.set_xlim((a, b))
    ax.set_ylim((a, b))
    # plt.colorbar(scatter)
    plt.show()

    # minimal_temp = 10**(-3)
    # n_chains = 16
    # q = np.power(minimal_temp, 1 / (n_chains - 1))
    # temperatures = np.power(q, np.arange(n_chains))
    # # temperatures = np.logspace(-3, 0, n_chains)
    #
    # fig, axes = plt.subplots(4, 4)
    # for iX, row in enumerate(axes):
    #     for iY, ax in enumerate(row):
    #         iI = len(row) * iX + iY
    #         temp = temperatures[iI]
    #         scatter = ax.scatter(samples[:, 0], samples[:, 1], c=color**temp, alpha=1, cmap="Blues")
    #         ax.scatter(np.log10(p1), np.log10(p2), marker="x", color="red")
    #         ax.set_xlabel(param_ids[0])
    #         ax.set_ylabel(param_ids[1])
    #         # plt.colorbar(scatter)
    #         ax.set_title(f"Temp {temp}")
    # plt.show()

    for iC in range(n_chains):
        trajectory = np.array([[step[iC][id] for id in param_ids] for step in sample_history])
        trajectory_posterior = np.array([step[iC] for step in posterior_history])
        color = np.exp(trajectory_posterior)
        fig, ax = plt.subplots()
        ax.plot(trajectory[:, 0], trajectory[:, 1], c="k", alpha=1, zorder=1)
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], c=color, alpha=1, cmap="Blues", zorder=2)
        ax.scatter(np.log10(p1), np.log10(p2), marker="x", color="red", zorder=3)
        ax.scatter(map_params["p1"], map_params["p2"], marker="x", color="magenta")
        ax.set_xlabel(param_ids[0])
        ax.set_ylabel(param_ids[1])
        ax.set_xlim((a, b))
        ax.set_ylim((a, b))
        ax.set_title(f"Chain {iC}")
        plt.show()
    pass
