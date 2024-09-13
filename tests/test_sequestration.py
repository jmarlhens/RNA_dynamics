import numpy as np
from build_and_simulate.build_model import setup_model, simulate_model, visualize_simulation
from pysb import Observable
import pandas as pd

def test_sequestration():
    # Plasmid design for sequestration test:
    # 1. First plasmid expresses Star6
    # 2. Second plasmid expresses aStar6 (anti-Star6)
    plasmids = [
        (None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (None, None, [(False, "aStar6")]),  # Step 2: Express aStar6
    ]

    # Define model parameters
    parameters_plasmids = {
        'k_Star6_concentration': 1,  # Initial concentration of Star6
        'k_aStar6_concentration': 4,  # Initial concentration of aStar6
    }
    # load and add parameters_plasmids
    parameters_df = pd.read_csv('../data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Define sequestration reactions: [(species1, species2)]
    bindings = [
        ("Star6", "aStar6"),  # Sequestration between Star6 and aStar6
    ]

    # Setup the model with the plasmids, parameters, and bindings
    model = setup_model(plasmids, parameters, bindings=bindings)

    # Define observables
    Observable("sequestered", model.monomers['RNA_Star6'](state="full", binding=1) % model.monomers['RNA_aStar6'](state="full", binding=1))
    Observable("free_star", model.monomers['RNA_Star6'](state="full", binding=None))
    Observable("free_antistar", model.monomers['RNA_aStar6'](state="full", binding=None))

    # Time span for build_and_simulate
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the build_and_simulate
    y_res = simulate_model(model, t)

    # Define species to plot including the new observables
    species_to_plot = list(model.observables.keys())

    # Visualize results
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

    # Print the ODEs for debugging
    for ode in model.odes:
        print(ode)

if __name__ == "__main__":
    test_sequestration()
