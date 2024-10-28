import numpy as np
import sympy as sp
from pysb import Model, Monomer, Parameter, Rule, Observable, Expression, Initial
from pysb.simulator import ScipyOdeSimulator
import matplotlib.pyplot as plt
from sympy import Piecewise

# Step 1: Define the PySB model
model = Model()

# Define the basic components (Monomers)
Monomer('Protein')
Monomer('Time')

# Define parameters
Parameter('Protein_0', 0)
Parameter('Time_0', 0)
Parameter('k_clock', 1)

# Define initial conditions
Initial(Protein(), Protein_0)
Initial(Time(), Time_0)

# Observables
Observable('obs_Time', Time())
Observable('obs_Protein', Protein())

# Define the piecewise expression for k_syn
Expression('k_syn', Piecewise((0, sp.Lt(obs_Time, 4)), (0, sp.Gt(obs_Time, 15)), (5, True)))

# Define reactions
Rule('Protein_synthesis', None >> Protein(), model.expressions['k_syn'])
Rule('Clock', None >> Time(), k_clock)
# add degradation rule
Parameter('k_deg', 0.4)
Rule('Protein_degradation', Protein() >> None, k_deg)


# Step 2: Simulate the model
t_span = np.linspace(0, 30, 3001)
sim = ScipyOdeSimulator(model, tspan=t_span)
result = sim.run()

# Calculate k_syn values
k_syn_values = [float(model.expressions['k_syn'].expr.subs(obs_Time, t)) for t in t_span]

# Step 3: Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot Protein concentration
ax1.plot(t_span, result.observables['obs_Protein'], label='Protein')
ax1.set_ylabel('Protein concentration')
ax1.set_title('Protein Concentration Over Time')
ax1.legend()
ax1.grid(True)

# Plot k_syn values
ax2.plot(t_span, k_syn_values, label='k_syn', color='red')
ax2.set_xlabel('Time')
ax2.set_ylabel('k_syn value')
ax2.set_title('Synthesis Rate (k_syn) Over Time')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
