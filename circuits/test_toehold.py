import numpy as np
from build_simulate_analyse.build_model import setup_model, simulate_model, visualize_simulation
from pysb import Observable
import pandas as pd

def test_toehold(plot=False, parameters_plasmids = {"k_Toehold3_GFP_concentration": 1, "k_Trigger3_concentration": 1}):
    # Plasmid design
    plasmids = [
        (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),
        (None, None, [(False, "Trigger3")]),
    ]

    # load and add parameters_plasmids
    parameters_df = pd.read_csv('../data/model_parameters_priors.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Setup the model
    model = setup_model(plasmids, parameters)

    # Time span for build_simulate_analyse
    n_steps = 1000
    t = np.linspace(0, 20, n_steps)

    # Run the build_simulate_analyse
    y_res = simulate_model(model, t)

    if plot:
        # Visualize results
        species_to_plot = list(model.observables.keys())
        visualize_simulation(t, y_res, species_to_plot=species_to_plot)

    return model

if __name__ == "__main__":
    model = test_toehold()
