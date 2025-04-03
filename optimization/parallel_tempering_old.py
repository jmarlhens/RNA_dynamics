import multiprocessing

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from optimization.optimization_algorithm import OptimizationAlgorithm

# from minimal_test import memory_location


import cProfile

class ParallelTempering(OptimizationAlgorithm):
    def __init__(self):
        self.sample_history = None
        self.posterior_history = None
        self.tempered_posterior_history = None
        self.acceptance_rates = None
        self.swap_acceptance_rates = None

        self.map_params = None
        pass

    @staticmethod
    def evaluate_posterior(priors, log_likelihood, samples):
        log_prob = np.sum([priors[id].logpdf(samples[id]) for id in samples.keys()])
        if np.isinf(
                log_prob):  # In case the parameters do not satisfy the priors, we can ommit evaluating the corresponding likelihood
            log_likelihood_val = - np.inf
        else:
            log_likelihood_val = log_likelihood(parameters=samples)

        return log_likelihood_val, log_prob

    @staticmethod
    def __step__(args):
        chain_id, priors, log_likelihood, cur_inv_temp, cur_var, prev_samples, prev_posterior, prev_log_likelihood, prev_log_prob = args
        param_ids = list(prev_samples.keys())
        proposal_means = [prev_samples[id] for id in param_ids]

        samples = np.random.normal(proposal_means, cur_var)
        samples = {id: samples[iI] for iI, id in enumerate(param_ids)}

        log_likelihood_val, log_prob = ParallelTempering.evaluate_posterior(priors, log_likelihood, samples)

        # ToDo Crosscheck whether log_prob is the right quantitiy here
        # As we consider logarithmic quantities, the multiplication is an addition
        tempered_posterior = cur_inv_temp * log_likelihood_val + log_prob
        posterior = log_likelihood_val + log_prob

        # print(f"Step Chain {chain_id}: {posterior}")
        prev_tempered_posterior = cur_inv_temp * prev_log_likelihood + prev_log_prob

        # Due to logarithms the reference is zero (e.g. np.log(1)=0 and the fraction is a difference)
        a = min(0, tempered_posterior - prev_tempered_posterior)
        a = np.exp(a)
        samp = np.random.rand()
        accept = samp < a

        _ = sum(ParallelTempering.evaluate_posterior(priors, log_likelihood, prev_samples))
        # if prev_posterior_test != prev_posterior:
        #     raise Exception("The history is not correct in case this exception is thrown.")

        if accept:
            # Accept
            samples_to_append = samples
            tempered_posterior_to_append = tempered_posterior
            posterior_to_append = posterior
            log_likelihood_val_to_append = log_likelihood_val
            log_probability_to_append = log_prob

        else:
            # Reject
            samples_to_append = prev_samples
            tempered_posterior_to_append = prev_tempered_posterior
            posterior_to_append = prev_posterior
            log_likelihood_val_to_append = prev_log_likelihood
            log_probability_to_append = prev_log_prob

        return accept, samples_to_append, tempered_posterior_to_append, posterior_to_append, log_likelihood_val_to_append, log_probability_to_append

    def run(self, log_likelihood, priors: dict, n_chains: int = 10, n_swaps: int = 2, n_samples: int = 1000,
            minimal_inverse_temp: float = 10 ** (-3), var_ref: float = 0.01, use_multiprocessing=False):
        """

        :param log_likelihood:
        :param priors: a dict containing the prior distribution of each parameter in the form of scipy.stats distribution
        :param n_chains:
        :param n_swaps:
        :param n_samples:
        :param minimal_inverse_temp:
        :param var_ref:
        :return:
        """
        target_acceptance_rate = 0.44

        param_ids = list(priors.keys())
        q = 1
        if n_chains > 1:
            q = np.power(minimal_inverse_temp, 1 / (n_chains - 1))
        inverse_temperatures = np.power(q, np.arange(n_chains))

        map_func = map

        if use_multiprocessing:
            num_processes = os.cpu_count()
            pool_size = num_processes - 1
            print(f"Creating Multiprocess Pool with {pool_size} processes")
            pool = multiprocessing.Pool(pool_size)
            # pool = concurrent.futures.ThreadPoolExecutor(num_processes - 1)
            map_func = pool.map

        """
        Sample Initial Parameters from Priors
        """
        print("Sample Initial Parameters from Priors")
        cur_samples = []
        cur_tempered_posterior = []
        cur_posterior = []
        cur_log_likelihood = []
        cur_log_prob = []
        cur_accepts = []
        for iC in range(n_chains):
            samples = {}
            for id in param_ids:
                samples[id] = priors[id].rvs()

            log_likelihood_val, log_prob = ParallelTempering.evaluate_posterior(priors, log_likelihood, samples)
            # ToDo Crosscheck whether log_prob is the right quantitiy here
            # As we consider logarithmic quantities, the multiplication is an addition
            tempered_posterior = inverse_temperatures[iC] * log_likelihood_val + log_prob
            posterior = log_likelihood_val + log_prob
            print(f"Chain {iC}: {posterior} ({samples})")
            cur_samples.append(samples)
            cur_tempered_posterior.append(tempered_posterior)
            cur_posterior.append(posterior)
            cur_log_likelihood.append(log_likelihood_val)
            cur_log_prob.append(log_prob)
            cur_accepts.append(0)

        """
        Perform Parallel Tempering
        """
        print("Start Parallel Tempering")

        def swap_elems(arr, i, j):
            elem_i = arr[i]
            elem_j = arr[j]
            arr[i] = elem_j
            arr[j] = elem_i
            return arr

        radii = np.ones(n_chains)
        acceptance_rates = np.zeros(n_chains)
        acceptance_history = [cur_accepts]
        swap_acceptance_rates = np.zeros(n_chains)
        tempered_posterior_history = [cur_tempered_posterior]
        posterior_history = [cur_posterior]
        sample_history = [cur_samples]
        log_likelihood_history = [cur_log_likelihood]
        log_probability_history = [cur_log_prob]
        swap_acceptance_history = [[0 for _ in range(n_chains - 1)]]


        chain_ids = np.arange(n_chains)
        prior_references = [priors for _ in range(n_chains)]
        log_likelihood_references = [log_likelihood for _ in range(n_chains)]

        cur_swap_count = 0
        cur_swap_acceptance_rate = 0

        start = time.time()
        for iS in range(1, n_samples + 1):
            # cur_samples = []
            # cur_tempered_posterior = []
            # cur_posterior = []
            # cur_log_likelihood = []
            # cur_accepts = []
            swap_count = 0

            # for iC in range(n_chains):
            #     cur_temp = temperatures[iC]  # [len(temperatures) - iC]
            #     cur_radius = radii[iC]
            #
            #     # cur_var = var_ref * np.exp(2 * cur_temp)
            #     # cur_var = var_ref * 10**(cur_temp)
            #     prev_samples = sample_history[-1][iC]
            #     prev_tempered_posterior = tempered_posterior_history[-1][iC]
            #
            #     prev_posterior = posterior_history[-1][iC]
            #     prev_log_likelihood = log_likelihood_history[-1][iC]
            #
            #     accept, samples, tempered_posterior, posterior, log_likelihood_val = ParallelTempering.__step__(
            #         param_ids, cur_temp, cur_radius, prev_samples, prev_tempered_posterior, prev_posterior,
            #         prev_log_likelihood)
            #
            #     cur_samples.append(samples)
            #     cur_tempered_posterior.append(tempered_posterior)
            #     cur_posterior.append(posterior)@
            #     cur_log_likelihood.append(log_likelihood_val)
            #     acceptance_rates[iC] += accept
            #     cur_accepts.append(accept)
            # variances = var_ref * radii * 1/inverse_temperatures#np.exp(2 * 1/inverse_temperatures)
            variances = var_ref * radii * (1 + np.log2(1 / inverse_temperatures))
            arguments = zip(chain_ids,
                            prior_references,
                            log_likelihood_references,
                            inverse_temperatures,
                            variances,
                            sample_history[-1],
                            posterior_history[-1],
                            log_likelihood_history[-1],
                            log_probability_history[-1])
            arguments = list(arguments)
            # Alternative is to implement vectorized versions, as ScipyOde employed allows for multiprocessed parallelism in case of multiple parameter sets.
            results = map_func(ParallelTempering.__step__, arguments)
            results = list(results)
            # print(results)
            results = list(map(list, zip(*results)))
            cur_accepts, cur_samples, cur_tempered_posterior, cur_posterior, cur_log_likelihood, cur_log_prob = results

            cur_swap_accepts = np.zeros(n_chains - 1)
            accepted_swaps = []
            for _ in range(n_swaps):
                iC = np.random.randint(0, n_chains - 1)
                temp_diff = inverse_temperatures[iC] - inverse_temperatures[iC + 1]
                # rel_log_likelihood_val = log_likelihood(parameters=cur_samples[iC + 1])
                log_likelihood_val = cur_log_likelihood[iC]
                rel_log_likelihood_val = cur_log_likelihood[iC + 1]
                # post_diff = cur_posterior[iC + 1] - cur_posterior[iC]
                likelihood_diff = rel_log_likelihood_val - log_likelihood_val
                a = min(0, temp_diff * likelihood_diff)
                a = np.exp(a)
                samp = np.random.rand()
                accept = samp < a
                if accept:
                    # Accept Swap
                    swap_elems(cur_samples, iC, iC + 1)
                    # swap_elems(cur_tempered_posterior, iC, iC + 1) # Swapping Tempered Posteriors is useless
                    swap_elems(cur_posterior, iC, iC + 1)
                    swap_elems(cur_log_likelihood, iC, iC + 1)
                    swap_elems(cur_log_prob, iC, iC + 1)
                    cur_swap_accepts[iC] = 1
                    swap_count += 1

                    accepted_swaps.append(f"{iC} <-> {iC + 1}")
                    # radii[iC] = radii[iC] / 10
                    # radii[iC + 1]


            tempered_posterior_history.append(cur_tempered_posterior)
            sample_history.append(cur_samples)
            posterior_history.append(cur_posterior)
            log_likelihood_history.append(cur_log_likelihood)
            log_probability_history.append(cur_log_prob)
            acceptance_history.append(cur_accepts)
            swap_acceptance_history.append(cur_swap_accepts)

            # Adaptive Radius selection to adapt variance to yield beneficial acceptance rates
            cur_acceptance_rates = np.mean(acceptance_history[-100:], axis=0)
            if iS % 2 == 0 and True:
                radii[cur_acceptance_rates > target_acceptance_rate * 1.1] *= 1.01
                radii[cur_acceptance_rates < target_acceptance_rate * 0.9] *= 0.99
            if iS % 10 == 0 and True:
                radii[cur_acceptance_rates > target_acceptance_rate * 1.3] *= 1.2
                radii[cur_acceptance_rates < target_acceptance_rate * 0.7] *= 0.9
            radii[radii < 0.01] = 0.01
            radii[radii > 100] = 100
            if n_swaps > 0:
                cur_swap_count = len(accepted_swaps)
                cur_swap_acceptance_rate = cur_swap_count / n_swaps * 100
            duration = time.time() - start
            print(f"Iteration {iS}:")
            print(
                f"    cur acceptance rates [{', '.join(map(lambda val: f'{val:0.3}', cur_acceptance_rates))}] ({duration / iS} s per iteration)")
            print(f"    Accepted {cur_swap_count} ({cur_swap_acceptance_rate}%): {', '.join(accepted_swaps)}")
            print(f"    Radii: [{', '.join(map(lambda val: f'{val:0.2}', radii))}]")
            print(f"    Posteriors: [{', '.join(map(lambda val: f'{val:0.2}', cur_posterior))}]")
            # print(f"Iteration {iS}, accepts {acceptance_rates}, swap count {swap_count} ({duration / iS} s per iteration) \r")

            # sys.stdout.write("\r")
            # sys.stdout.write(f"Iteration {iS}, acceptance rates {acceptance_rates / iS}\r")
            sys.stdout.flush()
            pass

        pass

        if use_multiprocessing:
            pool.close()
            # pool.terminate()

        posterior_history = np.array(posterior_history)
        tempered_posterior_history = np.array(tempered_posterior_history)
        acceptance_history = np.array(acceptance_history)
        swap_acceptance_history = np.array(swap_acceptance_history)

        best_config_id = np.argmax(posterior_history)
        best_config_id = [int(best_config_id / n_chains), int(best_config_id % n_chains)]
        map_params = sample_history[best_config_id[0]][best_config_id[1]]
        self.sample_history = sample_history
        self.posterior_history = posterior_history
        self.tempered_posterior_history = tempered_posterior_history
        self.acceptance_history = acceptance_history
        self.swap_acceptance_rates = swap_acceptance_rates

        divisor = np.expand_dims(np.arange(n_samples + 1) + 1, 1)
        acceptance_rates = np.cumsum(acceptance_history, axis=0) / divisor
        swap_acceptance_rates = np.cumsum(swap_acceptance_history, axis=0) / divisor
        self.acceptance_rates = acceptance_rates
        self.swap_acceptance_rates = swap_acceptance_rates

        self.map_params = map_params
        return sample_history, posterior_history, tempered_posterior_history, map_params


if __name__ == '__main__':
    n_chains = 10
    minimal_temp = 10 ** (-6)
    n_samples = 1000
    n_swaps = 2
    var_ref = 0.1

    n_measurement_samples = 10

    p1 = 5
    p2 = 1

    measurement_data = np.random.randn(n_measurement_samples) * p2 + p1


    def log_likelihood(parameters):
        params = parameters
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

    profiler = cProfile.Profile()
    profiler.enable()
    sample_history, posterior_history, tempered_posterior_history, map_params = opt.run(log_likelihood=log_likelihood,
                                                                                        priors=priors,
                                                                                        n_chains=n_chains,
                                                                                        minimal_inverse_temp=minimal_temp,
                                                                                        var_ref=var_ref,
                                                                                        n_samples=n_samples,
                                                                                        n_swaps=n_swaps)

    profiler.disable()
    profiler.dump_stats("Simulation_Profile.prof")
    profiler.print_stats()

    print("Inferred Params are:")
    print({key: 10 ** (map_params[key]) for key in map_params})
    print("Acceptance Rates:", opt.acceptance_rates)
    print("Swap Acceptance Rates:", opt.swap_acceptance_rates)

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
