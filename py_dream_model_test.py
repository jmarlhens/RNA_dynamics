"""
Adapted from PyDream Example authored by Erin
"""
import pandas as pd
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
from rna_dynamics_main import process_plasmid

"""
Setup Parameters
"""
fixed_parameters = {
    "k_mat": 10 ** 10,
}
parameters = {
    "k_tx": 2,
    "k_tl": 2,
    # "k_mat": 10 ** 10,
    "k_rna_deg": 0.5,
    "k_prot_deg": 0.5,
}

"""
Setup Model
"""

plasmids = [
    # (("Sense1", "Star1"), None, [(True, "GFP")]),
    (None, None, [(True, "RFP")]),
    # (None, None, [(False, "Trigger1")]),
    # (None, None, [(False, "Star1")]),
]

omega_val = 1000000
model = Model()
Parameter('omega', omega_val)  # in L

cur_parameters = dict()
cur_parameters.update(parameters)
cur_parameters.update(fixed_parameters)

for param in cur_parameters:
    Parameter(param, cur_parameters[param])

for plasmid in plasmids:
    process_plasmid(plasmid=plasmid, model=model)

measured_observables_names = [
    # "obs_Protein_GFP",
    "obs_Protein_RFP",
]

observable_names = []
observables = []
for monomer in model.monomers:
    desired_state = "full" if isinstance(monomer, RNA) else "mature"
    obs_name = "obs_" + monomer.name
    observable = Observable(obs_name, monomer(state=desired_state))
    observable_names.append(obs_name)
    observables.append(observable)

# Create lists of sampled pysb parameter names to use for subbing in parameter values in likelihood function.
pysb_sampled_parameter_names = [param for param in parameters]

"""
Setup Data
"""
n_replicates = 3
# Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.
tspan = np.linspace(0, 30, 10)
solver = Solver(model, tspan)
solver.run()

measurements = solver.result.dataframe[measured_observables_names]
cols = list(measurements.columns)
for col in cols:
    for iR in range(n_replicates):
        measurements[col + f" Replicate {iR}"] = measurements[col].map(
            lambda val: (val + val * 0.1 * np.random.randn()) if val > 0 else val)
experimental_data = measurements[[col for col in measurements.columns if "Replicate" in col]]

pass

"""
Define Log Likelihood
"""


# Define likelihood function to generate simulated data that corresponds to experimental time points.
# This function should take as input a parameter vector (parameter values are in the order dictated by first argument to run_dream function below).
# The function returns a log probability value for the parameter vector given the experimental data.

# def get_log_likelihood_dream(solver, parameter_names: list, fixed_parameters: dict, observable_names: list,
#                              experimental_data: pd.DataFrame):
likelihoods = {}
for observable in observable_names:
    y_meas = experimental_data[[col for col in experimental_data.columns if observable in col]].values
    mean = np.mean(y_meas, axis=1)
    var = np.var(y_meas, axis=1)
    likelihoods[observable] = norm(loc=mean, scale=np.sqrt(var))

parameter_names = list(parameters.keys())

def log_likelihood(parameter_vector):
    parameters = {p_name: p_val for p_name, p_val in zip(parameter_names, parameter_vector)}
    parameters.update(fixed_parameters)

    parameters = {p_name: 10**parameters[p_name] for p_name in parameters}

    sim_result = solver.run(parameters)

    # ToDo For adequate likelihood treatment see:
    # https://github.com/LoLab-MSM/PyDREAM/blob/master/pydream/examples/robertson/example_sample_robertson_with_dream.py#L38
    ll = []
    for observable in observable_names:
        cur_vals = solver.yobs[observable]
        cur_ll = likelihoods[observable].logpdf(cur_vals)
        ll.append(np.sum(cur_ll))
        # ll_mRNA = -np.sum((sim_data_mRNA - experimental_data_mRNA) ** 2)

    return np.sum(ll)

#    return log_likelihood

# likelihood = get_log_likelihood_dream(solver=solver,
#                                       parameter_names=list(parameters.keys()),
#                                       fixed_parameters=fixed_parameters,
#                                       observable_names=observable_names,
#                                       experimental_data=experimental_data)

# Add vector of PySB rate parameters to be sampled as unobserved random variables to DREAM with uniform priors.

original_params = np.log10([param.value for param in model.parameters_rules()])
# Set upper and lower limits for uniform prior to be 3 orders of magnitude above and below original parameter values.
lower_limits = original_params - 3

parameters_to_sample = SampledParam(uniform, loc=lower_limits, scale=6)

sampled_parameter_names = [parameters_to_sample]

niterations = 10000
converged = False
total_iterations = niterations
nchains = 5



if __name__ == '__main__':
    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood=log_likelihood, niterations=niterations, nchains=nchains,
                                       multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1,
                                       model_name='robertson_dreamzs_5chain', verbose=True)

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save('robertson_dreamzs_5chain_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
                sampled_params[chain])
        np.save('robertson_dreamzs_5chain_logps_chain_' + str(chain) + '_' + str(total_iterations), log_ps[chain])

    # Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    np.savetxt('robertson_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)

    old_samples = sampled_params
    if np.any(GR > 1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations

            sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood=log_likelihood, start=starts,
                                               niterations=niterations,
                                               nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
                                               history_thin=1, model_name='robertson_dreamzs_5chain', verbose=True,
                                               restart=True)

            for chain in range(len(sampled_params)):
                np.save('robertson_dreamzs_5chain_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
                        sampled_params[chain])
                np.save('robertson_dreamzs_5chain_logps_chain_' + str(chain) + '_' + str(total_iterations),
                        log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ', total_iterations, ' GR = ', GR)
            np.savetxt('robertson_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)

            if np.all(GR < 1.2):
                converged = True

    try:
        # Plot output
        import seaborn as sns
        from matplotlib import pyplot as plt

        total_iterations = len(old_samples[0])
        burnin = total_iterations / 2
        samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],
                                  old_samples[3][burnin:, :], old_samples[4][burnin:, :]))

        ndims = len(old_samples[0][0])
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim])
            fig.savefig('PyDREAM_example_Robertson_dimension_' + str(dim))

    except ImportError:
        pass

else:
    run_kwargs = {'parameters': sampled_parameter_names, 'likelihood': log_likelihood, 'niterations': 10000,
                  'nchains': nchains, 'multitry': False, 'gamma_levels': 4, 'adapt_gamma': True, 'history_thin': 1,
                  'model_name': 'robertson_dreamzs_5chain', 'verbose': True}
