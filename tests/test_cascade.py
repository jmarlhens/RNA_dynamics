import numpy as np
from build_simulate_analyse.build_model import setup_model, simulate_model, visualize_simulation
import pandas as pd

def test_cascade():
    # Plasmid design for the cascade:
    # 1. First plasmid expresses Star6
    # 2. Second plasmid uses Sense6 to control Trigger1 transcription
    # 3. Third plasmid uses Trigger1 to control Toehold1, which translates GFP
    plasmids = [
        (None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (("Sense6", "Star6"), None, [(False, "Trigger1")]),  # Step 2: Sense6 (controlled by Star6) controls Trigger1
        (None, ("Toehold1", "Trigger1"), [(True, "GFP")]),  # Step 3: Trigger1 controls Toehold1 and GFP translation
    ]


    # Define model parameters
    parameters_plasmids = {
        "k_Star6_concentration": 1,
        "k_Sense6_Trigger1_concentration": 1,
        "k_Toehold1_GFP_concentration": 1,
    }

    # load and add parameters_plasmids
    parameters_df = pd.read_csv('../data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Setup the model with the plasmids and parameters
    model = setup_model(plasmids, parameters)

    # Time span for build_simulate_analyse
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the build_simulate_analyse
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

if __name__ == "__main__":
    test_cascade()
