from obsolete.ffl.STAR_rules import RNA_regulatory_base_module, RNA_regulatory_base_parameters
from pysb import Model, Monomer, Parameter
from pysb import Observable
from pysb.simulator import BngSimulator
import numpy as np

#Oscillator, with RNA 1 inhibiting RNA 2, and RNA 2 inhibiting RNA 1:
# 'omega' is the Avoqadro constant times the volume of the system
# Cf Darren Wilkinson's book: https://www.taylorfrancis.com/books/mono/10.1201/b11812/stochastic-modelling-systems-biology-darren-wilkinson
omega_val = 6 * 1e23 * np.pi/2 * 1e-15
omega_val = 3000
k_deg_val = 1e-3
model = Model()
Parameter('omega', omega_val)  # in L

# Define the Monomer for the RNA
Monomer("RNA_1", ['terminator', 'r', 'state'], {'state': ['init', 'full', 'partial']})
Monomer("RNA_2", ['terminator', 'r', 'state'], {'state': ['init', 'full', 'partial']})
Monomer("RNA_3", ['terminator', 'r', 'state'], {'state': ['init', 'full', 'partial']})

# Define the parameters for the kinetic rates}
# RNA 1 activating RNA 2
parameters_RNA_2_RNA_1 = {'k_init': 20*1e-1,
                'k_bind': 1e+6,
                'k_unbind': 1e-8,
                'k_act': 2*1e-1,
                'k_act_reg': 1e+8,
                'k_stop': 1e+2,
                'k_stop_reg': 1*1e-6,
                'k_deg': k_deg_val}

# RNA 3 inhibiting RNA 1
parameters_RNA_1_RNA_3 = {'k_init': 10*1e-1,
                'k_bind': 1e+6,
                'k_unbind': 1e-8,
                'k_act': 1e+0,
                'k_act_reg': 1*1e-8,
                'k_stop': 2*1e-4,
                'k_stop_reg': 1e+1,
                'k_deg': k_deg_val}


# RNA 2 inhibiting RNA 3
parameters_RNA_3_RNA_2 = {'k_init': 5*1e-1,
                'k_bind': 1e+8,
                'k_unbind': 1e-8,
                'k_act': 1e+4,
                'k_act_reg': 1e-4,
                'k_stop': 1e-6,
                'k_stop_reg': 1e+2,
                'k_deg': k_deg_val}

# Define the parameters for the kinetic rates
RNA_regulatory_base_parameters(model, RNA_2, RNA_1, parameters_RNA_2_RNA_1)
RNA_regulatory_base_parameters(model, RNA_3, RNA_2, parameters_RNA_3_RNA_2)
RNA_regulatory_base_parameters(model, RNA_1, RNA_3, parameters_RNA_1_RNA_3)

# Define the rules for the RNA
RNA_regulatory_base_module(RNA_2, RNA_1, k_init_RNA_2, k_bind_RNA_2_RNA_1, k_unbind_RNA_2_RNA_1, k_act_RNA_2, k_act_reg_RNA_2_RNA_1, k_stop_RNA_2, k_stop_reg_RNA_2_RNA_1, k_deg_RNA_2)
RNA_regulatory_base_module(RNA_3, RNA_2, k_init_RNA_3, k_bind_RNA_3_RNA_2, k_unbind_RNA_3_RNA_2, k_act_RNA_3, k_act_reg_RNA_3_RNA_2, k_stop_RNA_3, k_stop_reg_RNA_3_RNA_2, k_deg_RNA_3)
RNA_regulatory_base_module(RNA_1, RNA_3, k_init_RNA_1, k_bind_RNA_1_RNA_3, k_unbind_RNA_1_RNA_3, k_act_RNA_1, k_act_reg_RNA_1_RNA_3, k_stop_RNA_1, k_stop_reg_RNA_1_RNA_3, k_deg_RNA_1)

# Define the observables
Observable("obs_RNA_1_init", RNA_1(state='init'))
Observable("obs_RNA_2_init", RNA_2(state='init'))
Observable("obs_RNA_3_init", RNA_3(state='init'))
Observable("obs_RNA_1_partial", RNA_1(state='partial'))
Observable("obs_RNA_2_partial", RNA_2(state='partial'))
Observable("obs_RNA_3_partial", RNA_3(state='partial'))
Observable("obs_RNA_1_full", RNA_1(state='full'))
Observable("obs_RNA_2_full", RNA_2(state='full'))
Observable("obs_RNA_3_full", RNA_3(state='full'))

# Define the initial conditions
Parameter('RNA_1_0', 1e-1 * model.parameters['omega'].value)  # Initial concentration of RNA_1
Parameter('RNA_2_0', 1e-4 * model.parameters['omega'].value) # Initial concentration of RNA_2
Parameter('RNA_3_0', 1e-4 * model.parameters['omega'].value) # Initial concentration of RNA_3
model.initial(RNA_1(state='init', r=None, terminator=None), RNA_1_0)
model.initial(RNA_2(state='init', r=None, terminator=None), RNA_2_0)
model.initial(RNA_3(state='init', r=None, terminator=None), RNA_3_0)

# Define the build_simulate_analyse conditions
t = np.linspace(0, 10, 10)
# y = ScipyOdeSimulator(model, verbose=True).run(tspan=t).all

# stochastic build_simulate_analyse
t = np.linspace(0, 10000, 10000)
y = BngSimulator(model, tspan=t, verbose=False).run(method = "nf").all

# Plot the results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(t, y['obs_RNA_1_full'], label='RNA_1_full')
plt.plot(t, y['obs_RNA_2_full'], label='RNA_2_full')
plt.plot(t, y['obs_RNA_3_full'], label='RNA_3_full')
# plt.plot(t, y['obs_RNA_1_partial'], label='RNA_1_partial')
# plt.plot(t, y['obs_RNA_2_partial'], label='RNA_2_partial')
# plt.plot(t, y['obs_RNA_3_partial'], label='RNA_3_partial')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (nM)')
plt.legend(loc=0)
plt.show()
