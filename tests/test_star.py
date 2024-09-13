import numpy as np
from build_and_simulate.build_model import setup_model, simulate_model, visualize_simulation
import pandas as pd

def test_star():
    # Plasmid design
    plasmids = [
        (("Sense1", "Star1"), None, [(True, "GFP")]),
        (None, None, [(False, "Star1")]),
    ]

    # Define model parameters
    parameters_plasmids = {
        "k_Sense1_GFP_concentration": 1,
        "k_Star1_concentration": 0.4,
    }

    # load and add parameters_plasmids
    parameters_df = pd.read_csv('../data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Setup the model
    model = setup_model(plasmids, parameters)

    # Time span for build_and_simulate
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the build_and_simulate
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

if __name__ == "__main__":
    test_star()
