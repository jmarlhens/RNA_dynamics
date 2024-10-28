"""
Adapted from PyDream Example authored by Erin
"""
import pandas as pd
import scipy
from pydream.core import run_dream
from pysb import Model, Parameter, Observable
from pysb.integrate import Solver
import numpy as np
from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
import os
import inspect
from pydream.convergence import Gelman_Rubin

from modules.molecules import RNA
from optimization.parallel_tempering_old import ParallelTempering
from rna_dynamics_main import process_plasmid

import matplotlib.pyplot as plt
import seaborn as sns
import cython
from minimal_test import memory_location

print("RUN py_dream_model_test.py")

"""
Setup Parameters
"""
# fixed_parameters = {
#     "k_tx": 2,
#     "k_tl": 1,
#     "k_rna_deg": 0.5,
#     "k_prot_deg": 0.0,  # 0.25,
#     "k_csy4": 1,
#     "k_mat": 10 ** 2,
# }
# parameters = {
#     # "k_mat": 10 ** 10,
#     "k_tl_bound_toehold": 0.1,
#     "k_trigger_binding": 5,
#     "k_trigger_unbinding": 0.5,
#     "k_tx_init": 1,
#     "k_star_bind": 5,
#     "k_star_unbind": 0.1,
#     "k_star_act": 2,
#     "k_star_act_reg": 0.01,
#     "k_star_stop": 1,
#     "k_star_stop_reg": 0.01
# }

fixed_parameters = {
    "k_csy4": 1,
    "k_mat": 10 ** 2,
    "k_prot_deg": 0.0,  # 0.25,
}

parameters = {
    "k_tx": 2,
    "k_tl": 1,
    "k_rna_deg": 0.5,
    # "k_tx_init": 1.5,
    # "k_star_bind": 5,
    # "k_star_unbind": 0.1,
    # "k_star_act": 2,
    # "k_star_act_reg": 0.01,
    # "k_star_stop": 1,
    # "k_star_stop_reg": 0.01
}

# parameters = {"k_tx": 2,
#                   "k_tl": 2,
#                   # "k_mat": 10 ** 10,
#                   "k_rna_deg": 0.5,
#                   "k_prot_deg": 0.5,
#                   # "k_csy4": 1,
#                   # "k_tl_bound_toehold": 0.1,
#                   # "k_trigger_binding": 5,
#                   # "k_trigger_unbinding": 0.5,
#                   # "k_tx_init": 1,
#                   # "k_star_bind": 5,
#                   # "k_star_unbind": 0.1,
#                   # "k_star_act": 2,
#                   # "k_star_act_reg": 0.01,
#                   # "k_star_stop": 1,
#                   # "k_star_stop_reg": 0.01
#                   }

log_parameters = {p_name: np.log10(parameters[p_name]) for p_name in parameters}
log_fixed_parameters = {p_name: np.log10(fixed_parameters[p_name]) for p_name in fixed_parameters}
"""
Setup Model
"""

# plasmids = [
#     (("Sense1", "Star1"), None, [(True, "GFP")]),
#     (None, None, [(True, "RFP")]),
#     # (None, None, [(False, "Trigger1")]),
#     (None, None, [(False, "Star1")]),
# ]
plasmids = [
    (None, None, [(True, "RFP")]),
    # (None, None, [(False, "Trigger1")]),
    # (None, None, [(False, "Star1")]),
]

# plasmids = [(("Sense_6", "STAR_6"), ("Toehold_3", "Trigger_3"), [(True, "GFP")]),
#             (None, None, [(False, "STAR_6")]),
#             (None, None, [(False, "Trigger_3")]),
#             ]

omega_val = 1000000
model = Model()
Parameter('omega', omega_val)  # in L

target_parameters = dict()
target_parameters.update(parameters)
target_parameters.update(fixed_parameters)

for param in target_parameters:
    Parameter(param, target_parameters[param])

for plasmid in plasmids:
    process_plasmid(plasmid=plasmid, model=model)

observable_names = []
observables = []
for monomer in model.monomers:
    desired_state = "full" if isinstance(monomer, RNA) else "mature"
    obs_name = "obs_" + monomer.name
    observable = Observable(obs_name, monomer(state=desired_state))
    observable_names.append(obs_name)
    observables.append(observable)

# measured_observables_names = observable_names

measured_observables_names = [
    # "obs_Protein_GFP",
    "obs_Protein_RFP",
    # "obs_RNA_Star1",
    # "obs_RNA_RFP"
]

# Create lists of sampled pysb parameter names to use for subbing in parameter values in likelihood function.
pysb_sampled_parameter_names = [param for param in parameters]

"""
Setup Data
"""
n_replicates = 3
# Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.
tspan = np.linspace(0, 10, 50)
solver = Solver(model, tspan)
solver.run(target_parameters)

measurements = solver.result.dataframe[measured_observables_names]
cols = list(measurements.columns)
for col in cols:
    for iR in range(n_replicates):
        measurements[col + f" Replicate {iR}"] = measurements[col].map(
            lambda val: (val + val * 0.1 * np.random.randn()) if val > 0 else val)

experimental_data = measurements[[col for col in measurements.columns if "Replicate" in col]]

pass

#    return log_likelihood

# likelihood = get_log_likelihood_dream(solver=solver,
#                                       parameter_names=list(parameters.keys()),
#                                       fixed_parameters=fixed_parameters,
#                                       observable_names=observable_names,
#                                       experimental_data=experimental_data)

# Add vector of PySB rate parameters to be sampled as unobserved random variables to DREAM with uniform priors.
"""
Define Priors
"""
original_params = np.log10([param.value for param in model.parameters_rules()])
# Set upper and lower limits for uniform prior to be 3 orders of magnitude above and below original parameter values.
lower_limits = original_params - 3

# parameters_to_sample = SampledParam(uniform, loc=lower_limits, scale=6)

prior_range = 6
prior_offset = prior_range / 2
log_priors = {}
for p_id in log_parameters:
    act_val = log_parameters[p_id]
    a = act_val - prior_offset
    b = act_val + prior_offset
    log_priors[p_id] = scipy.stats.uniform(loc=a, scale=b - a)

# sampled_parameter_names = [parameters_to_sample]

niterations = 10000
converged = False
total_iterations = niterations
nchains = 5

"""
Define Log Likelihood
"""

# Define likelihood function to generate simulated data that corresponds to experimental time points.
# This function should take as input a parameter vector (parameter values are in the order dictated by first argument to run_dream function below).
# The function returns a log probability value for the parameter vector given the experimental data.

# def get_log_likelihood_dream(solver, parameter_names: list, fixed_parameters: dict, observable_names: list,
#                              experimental_data: pd.DataFrame):
# dist_props = {}
likelihoods = {}
for observable in measured_observables_names:
    y_meas = experimental_data[[col for col in experimental_data.columns if observable in col]].values
    mean = np.mean(y_meas, axis=1)
    var = np.var(y_meas, axis=1)
    var[var < 0.01] = 0.01
    likelihoods[observable] = norm(loc=mean, scale=np.sqrt(var))
    # print(f"{observable}: [{', '.join(map(lambda val: f'{val:0.2}', mean))}] ({hex(id(likelihoods[observable]))})")
    # my_likelihood = norm(loc=mean, scale=np.sqrt(var))
    # dist_props[observable] = {"loc": mean,
    #                            "scale": np.sqrt(var)}
    # print(f"{observable} dist_props location {memory_location(dist_props[observable])} ({memory_location(dist_props)}, loc[-1]={dist_props[observable]['loc'][-1]})")
parameter_names = list(parameters.keys())


def log_likelihood(parameters):
    global likelihoods, dist_props
    cur_parameters = {p_name: 10 ** parameters[p_name] for p_name in parameters}

    solver.run(cur_parameters)
    # print(f"Parameters: {parameters}")
    # ToDo For adequate likelihood treatment see:
    # https://github.com/LoLab-MSM/PyDREAM/blob/master/pydream/examples/robertson/example_sample_robertson_with_dream.py#L38
    ll = np.zeros(len(measured_observables_names))
    for iL, observable in enumerate(measured_observables_names):
        cur_vals = solver.yobs[observable]
        cur_ll = likelihoods[observable].logpdf(cur_vals)
        # cur_ll = my_likelihood.logpdf(cur_vals)
        ll[iL] = np.sum(cur_ll)
        # print(f"Loc={likelihoods[observable].kwds['loc'][-1]}, Scale={likelihoods[observable].kwds['scale'][-1]} ({hex(id(likelihoods[observable]))})")
        # print(f"{observable} dist_props location {memory_location(dist_props[observable])} ({memory_location(dist_props)}, loc[-1]={dist_props[observable]['loc'][-1]})")

    output_ll = np.sum(ll)
    # print(f"Log Likelihood: {parameters} -> {output_ll}")
    return output_ll


def log_likelihood_vec(parameter_vector):
    parameters = {p_name: p_val for p_name, p_val in zip(parameter_names, parameter_vector)}
    parameters.update(fixed_parameters)

    ll = log_likelihood(parameters)
    return ll


if __name__ == '__main__':
    print("RUN Main of py_dream_model_test.py")
    """
    Model Calibration
    """

    log_likelihood_val = log_likelihood(parameters=log_parameters)
    log_prob = np.sum([log_priors[id].logpdf(log_parameters[id]) for id in log_parameters])
    target_posterior = log_likelihood_val + log_prob

    print("Original Posterior:", target_posterior)

    opt = ParallelTempering()

    n_chains = 6
    minimal_inverse_temp = 10 ** (-5)  # 10 ** (-10)
    var_ref = 0.01
    n_samples = 10 * 10 ** 3
    n_swaps = 2

    sample_history, posterior_history, tempered_posterior_history, map_params = opt.run(log_likelihood=log_likelihood,
                                                                                        priors=log_priors,
                                                                                        n_chains=n_chains,
                                                                                        minimal_inverse_temp=minimal_inverse_temp,
                                                                                        var_ref=var_ref,
                                                                                        n_samples=n_samples,
                                                                                        n_swaps=n_swaps,
                                                                                        use_multiprocessing=False)

    print(f"Minimum Posterior: {np.min(posterior_history)}")

    COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
    labels = ["Ref.", "Pred."]
    linestyles = ["-", "--"]
    best_fit = dict(map_params)
    print("Best Fit:", best_fit)
    log_best_fit = dict(best_fit)
    best_fit = {p_name: 10 ** best_fit[p_name] for p_name in best_fit}
    # best_fit.update(fixed_parameters)

    sample_cutoff = int(len(sample_history) / 2)
    posterior_samples = [elem[0] for elem in sample_history[sample_cutoff:]]
    posterior_samples = {p_name: [elem[p_name] for elem in posterior_samples] for p_name in log_priors}

    print("Original Params:", parameters)
    print("Best Fit Params:", best_fit)
    print("Original Posterior:", target_posterior)
    print("Best Fit Posterior:", np.max(posterior_history))

    fig, ax = plt.subplots()
    T = tspan

    for label, linestyle, param_dict in zip(labels, linestyles, [target_parameters, best_fit]):
        solver.run(param_dict)
        for iC, obs_name in enumerate(measured_observables_names):
            color = COLORS[iC]
            Y = solver.yobs[obs_name]
            ax.plot(T, Y, linestyle, color=color, label=f"{label} {obs_name}")

        for iC, obs_name in enumerate(observable_names):
            color = COLORS[iC]
            if obs_name in measured_observables_names:
                continue
            Y = solver.yobs[obs_name]
            ax.plot(T, Y, ":", color=color, label=f"{label} {obs_name}")

    ax.legend()
    ax.set_title("Prediction vs. Measurement")
    plt.show()

    color_map = "red"
    color_ref = "black"

    param_names = list(log_priors.keys())
    n_params = len(log_priors)
    fig, axes = plt.subplots(ncols=n_params, nrows=n_params, sharex=False, sharey=False)
    for iR in range(n_params):
        id_r = param_names[iR]
        data_Y = posterior_samples[id_r]
        map_r = log_best_fit[id_r]
        ref_r = log_parameters[id_r]
        for iC in range(n_params):
            ax = axes[iR, iC]
            id_c = param_names[iC]
            map_c = log_best_fit[id_c]
            ref_c = log_parameters[id_c]

            if iR == iC:
                data = posterior_samples[id_r]
                hist, bins, patches = ax.hist(data, density=True)
                cur_max = np.max(hist)
                ax.plot(map_r * np.ones(2), np.arange(2) * cur_max, "--", color=color_map)
                ax.plot(ref_r * np.ones(2), np.arange(2) * cur_max, color=color_ref)
            else:
                data_X = posterior_samples[id_c]

                ax.scatter(data_X, data_Y, alpha=0.1)
                interval = np.min(data_X), np.max(data_X)
                ax.plot(interval, map_r * np.ones(2), "--", color=color_map)
                ax.plot(interval, ref_r * np.ones(2), color=color_ref)
                interval = np.min(data_Y), np.max(data_Y)
                ax.plot(map_c * np.ones(2), interval, "--", color=color_map)
                ax.plot(ref_c * np.ones(2), interval, color=color_ref)
                # data = list(zip(data_X, data_Y))

            if iR == n_params - 1:
                ax.set_xlabel(id_c)
            if iC == 0:
                ax.set_ylabel(id_r)

    plt.tight_layout()
    plt.show()

    posterior_history = opt.posterior_history
    fig, axes = plt.subplots(nrows=2, ncols=2)
    X = np.arange(1 + n_samples)

    cutoff = 200
    ax = axes[0, 0]
    for iC in range(n_chains):
        Y = posterior_history[:, iC]
        Y_sign = np.sign(Y)
        Y_abs = np.abs(Y)
        Y = Y_sign * np.log(Y_abs)
        ax.plot(X[cutoff:], Y[cutoff:], label=f"{iC}")
    ax.set_title("Log Log Posterior History")
    ax.legend()

    ax = axes[0, 1]
    for iC in range(n_chains):
        ax.plot(X, opt.acceptance_rates[:, iC], label=f"{iC}")
    ax.set_title("Sample Acceptance Rates")
    ax.legend()

    ax = axes[1, 0]
    for iC in range(n_chains - 1):
        ax.plot(X, opt.swap_acceptance_rates[:, iC], label=f"{iC} <-> {iC + 1}")
    ax.set_title("Swap Acceptance Rates")
    ax.legend()

    plt.tight_layout()
    plt.show()
    exit()
#     # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
#     sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood=log_likelihood, niterations=niterations,
#                                        nchains=nchains,
#                                        multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1,
#                                        model_name='robertson_dreamzs_5chain', verbose=True)
#
#     print("First Run of DREAM completed")
#     # Save sampling output (sampled parameter values and their corresponding logps).
#     for chain in range(len(sampled_params)):
#         np.save('robertson_dreamzs_5chain_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
#                 sampled_params[chain])
#         np.save('robertson_dreamzs_5chain_logps_chain_' + str(chain) + '_' + str(total_iterations), log_ps[chain])
#
#     # Check convergence and continue sampling if not converged
#
#     GR = Gelman_Rubin(sampled_params)
#     print('At iteration: ', total_iterations, ' GR = ', GR)
#     np.savetxt('robertson_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)
#
#     old_samples = sampled_params
#     if np.any(GR > 1.2):
#         starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
#         while not converged:
#             total_iterations += niterations
#
#             sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood=log_likelihood, start=starts,
#                                                niterations=niterations,
#                                                nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
#                                                history_thin=1, model_name='robertson_dreamzs_5chain', verbose=True,
#                                                restart=True)
#
#             for chain in range(len(sampled_params)):
#                 np.save('robertson_dreamzs_5chain_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
#                         sampled_params[chain])
#                 np.save('robertson_dreamzs_5chain_logps_chain_' + str(chain) + '_' + str(total_iterations),
#                         log_ps[chain])
#
#             old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
#             GR = Gelman_Rubin(old_samples)
#             print('At iteration: ', total_iterations, ' GR = ', GR)
#             np.savetxt('robertson_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)
#
#             if np.all(GR < 1.2):
#                 converged = True
#
#     try:
#         # Plot output
#         import seaborn as sns
#         from matplotlib import pyplot as plt
#
#         total_iterations = len(old_samples[0])
#         burnin = total_iterations / 2
#         samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],
#                                   old_samples[3][burnin:, :], old_samples[4][burnin:, :]))
#
#         ndims = len(old_samples[0][0])
#         colors = sns.color_palette(n_colors=ndims)
#         for dim in range(ndims):
#             fig = plt.figure()
#             sns.distplot(samples[:, dim], color=colors[dim])
#             fig.savefig('PyDREAM_example_Robertson_dimension_' + str(dim))
#
#     except ImportError:
#         pass
#
# else:
#     run_kwargs = {'parameters': sampled_parameter_names, 'likelihood': log_likelihood_vec, 'niterations': 10000,
#                   'nchains': nchains, 'multitry': False, 'gamma_levels': 4, 'adapt_gamma': True, 'history_thin': 1,
#                   'model_name': 'robertson_dreamzs_5chain', 'verbose': True}
