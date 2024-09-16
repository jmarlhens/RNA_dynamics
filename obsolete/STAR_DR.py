from obsolete.ffl.STAR_rules import RNA_regulatory_base_module, RNA_regulatory_base_parameters
from pysb import Model, Monomer, Parameter
from pysb import Observable
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from itertools import product


def generate_two_node_RNA_TX_circuit():
    # 'omega' is the Avoqadro constant times the volume of the system
    # Cf Darren Wilkinson's book: https://www.taylorfrancis.com/books/mono/10.1201/b11812/stochastic-modelling-systems-biology-darren-wilkinson
    omega_val = 6 * 1e23 * np.pi / 2 * 1e-15
    omega_val = 1000000
    model = Model()
    Parameter('omega', omega_val)  # in L

    # Define the parameters for the kinetic rates
    parameters = {'k_init': 1e-5,  # in s^-1
                  'k_bind': 1e-0,  # in M^-1 s^-1
                  'k_unbind': 1e-5,  # in s^-1
                  'k_act': 1e-4,  # in s^-1
                  'k_act_reg': 1e-2,
                  'k_stop': 1e-1,
                  'k_stop_reg': 1e-3,
                  'k_deg': 1e-2}

    # Define the Monomer for the RNA
    Monomer("RNA_1", ['terminator', 'r', 'state'], {'state': ['init', 'full', 'partial']})
    Monomer("RNA_2", ['terminator', 'r', 'state'], {'state': ['init', 'full', 'partial']})

    # Define the initial conditions
    Parameter('RNA_1_0', 1e-2 * model.parameters['omega'].value)  # Initial concentration of RNA_1
    Parameter('RNA_2_0', 0 * model.parameters['omega'].value)  # Initial concentration of RNA_2
    RNA_regulatory_base_parameters(model, RNA_2, RNA_1, parameters)
    RNA_regulatory_base_module(RNA_2, RNA_1, k_init_RNA_2_molnum, k_bind_RNA_2_RNA_1_molnum,
                               k_unbind_RNA_2_RNA_1_molnum, k_act_RNA_2_molnum, k_act_reg_RNA_2_RNA_1_molnum,
                               k_stop_RNA_2_molnum, k_stop_reg_RNA_2_RNA_1_molnum, k_deg_RNA_2_molnum)

    model.initial(RNA_1(state='init', r=None, terminator=None), RNA_1_0)
    model.initial(RNA_1(state='full', r=None, terminator=None), RNA_1_0)
    model.initial(RNA_1(state='partial', r=None, terminator=None), RNA_1_0)
    model.initial(RNA_2(state='init', r=None, terminator=None), RNA_2_0)
    model.initial(RNA_2(state='full', r=None, terminator=None), RNA_2_0)
    model.initial(RNA_2(state='partial', r=None, terminator=None), RNA_2_0)

    # Define the observables
    Observable('obs_RNA_1', RNA_1(state='full'))
    Observable('obs_RNA_2', RNA_2(state='full'))
    Observable('obs_RNA_2_bound', RNA_2(state='full', r=None))
    Observable('obs_RNA_2_free', RNA_2(state='full', r=1) % RNA_1(state='full', r=1))
    Observable('obs_RNA_2_partial', RNA_2(state='partial'))


if __name__ == '__main__':
    generate_two_node_RNA_TX_circuit()

    t = np.linspace(0, 1000, 1000)
    y = ScipyOdeSimulator(model).run(tspan=t).all

    # Plot the results
    import matplotlib.pyplot as plt

    plt.figure()
    # plt.plot(t, y['obs_RNA_1'], label='RNA_1')
    plt.plot(t, y['obs_RNA_2'], label='RNA_2')
    plt.plot(t, y['obs_RNA_2_bound'], label='RNA_2_bound')
    plt.plot(t, y['obs_RNA_2_free'], label='RNA_2_free')
    plt.plot(t, y['obs_RNA_2_bound'] + y['obs_RNA_2_free'], label='RNA_2_bound + RNA_2_free')
    # plt.plot(t, y['obs_RNA_2_partial'], label='RNA_2_partial')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (M)')
    plt.legend(loc=0)
    plt.show()

    # Stochastic build_simulate_analyse
    from pysb.simulator import BngSimulator

    t = np.linspace(0, 100, 1000)
    y = BngSimulator(model, tspan=t, verbose=False).run(method="nf").all

    # Plot the results
    import matplotlib.pyplot as plt

    plt.figure()
    # plt.plot(t, y['obs_RNA_1'], label='RNA_1')
    plt.plot(t, y['obs_RNA_2'], label='RNA_2')
    plt.plot(t, y['obs_RNA_2_bound'], label='RNA_2_bound')
    plt.plot(t, y['obs_RNA_2_free'], label='RNA_2_free')
    plt.plot(t, y['obs_RNA_2_bound'] + y['obs_RNA_2_free'], label='RNA_2_bound + RNA_2_free')
    # plt.plot(t, y['obs_RNA_2_partial'], label='RNA_2_partial')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (M)')
    plt.legend(loc=0)
    plt.show()

    # Dose response
    RNA_1_0_list = np.logspace(-3, 3, 60).tolist()
    y = ScipyOdeSimulator(model, tspan=t).run(param_values={'RNA_1_0': RNA_1_0_list}).all

    RNA_2_endpoints = []
    for k, RNA_1_0_tmp in enumerate(RNA_1_0_list):
        RNA_2_endpoints.append(y[k]['obs_RNA_2'][-1])

    plt.figure()
    plt.semilogx(RNA_1_0_list, RNA_2_endpoints)
    plt.xlabel('RNA_1_0 (M)')
    plt.ylabel('RNA_2_endpoints (M)')
    plt.ylim(0)
    plt.grid(True, which="both")
    plt.show()

    # R_1_0 and k_act_RNA_2 combinations

    # Get all combinations of two elements
    k_act_RNA_2_list = np.logspace(-3, 3, 10).tolist()
    RNA_1_0_list = np.logspace(-3, 3, 40).tolist()
    combinations = list(product(RNA_1_0_list, k_act_RNA_2_list))
    RNA_1_0_combinations = [comb[0] for comb in combinations]
    k_act_RNA_2_combinations = [comb[1] for comb in combinations]
    positions = [(RNA_1_0_list.index(comb[0]), k_act_RNA_2_list.index(comb[1])) for comb in combinations]

    y = ScipyOdeSimulator(model, tspan=t).run(
        param_values={'RNA_1_0': RNA_1_0_combinations, 'k_act_RNA_2': k_act_RNA_2_combinations}).all

    RNA_2_endpoints = np.zeros((len(RNA_1_0_list), len(k_act_RNA_2_list)))
    for k, param_tmp in enumerate(zip(RNA_1_0_combinations, k_act_RNA_2_combinations, positions)):
        RNA_1_0_tmp, k_act_RNA_2_tmp, position = param_tmp
        RNA_2_endpoints[position] = y[k]['obs_RNA_2'][-1]

    plt.figure()
    for k, k_act_RNA_2_tmp in enumerate(k_act_RNA_2_list):
        plt.semilogx(RNA_1_0_list, RNA_2_endpoints[:, k], label='k_act_RNA_2 = ' + str(k_act_RNA_2_tmp))
    plt.xlabel('RNA_1_0 (M)')
    plt.ylabel('RNA_2_endpoints (M)')
    plt.ylim(0)
    plt.grid(True, which="both")
    # plt.legend(loc=0)
    # save as .svg file in current location/figures
    plt.savefig('figures/RNA_2_endpoints.svg', bbox_inches='tight')
    plt.show()

# Repression model
model.parameters['k_act_' + RNA_2.name].value = 1e-2
model.parameters['k_act_reg_' + RNA_2.name + "_" + RNA_1.name].value = 1e-3
model.parameters['k_stop_' + RNA_2.name].value = 1e-3
model.parameters['k_stop_reg_' + RNA_2.name + "_" + RNA_1.name].value = 1e-1

# Dose response
RNA_1_0_list = np.logspace(-3, 3, 60).tolist()
y = ScipyOdeSimulator(model, tspan=t).run(param_values={'RNA_1_0': RNA_1_0_list}).all

RNA_2_endpoints = []
for k, RNA_1_0_tmp in enumerate(RNA_1_0_list):
    RNA_2_endpoints.append(y[k]['obs_RNA_2'][-1])

plt.figure()
plt.semilogx(RNA_1_0_list, RNA_2_endpoints)
plt.xlabel('RNA_1_0 (M)')
plt.ylabel('RNA_2_endpoints (M)')
plt.ylim(0)
plt.grid(True, which="both")
plt.show()
