import numpy as np
from circuits.build_model import setup_model, simulate_model, visualize_simulation
import pandas as pd

def test_cascade(plot=False, parameters_plasmids=None, print_rules=False, print_odes=False):
    """
    Test the cascade circuit dynamics with STAR, Sense, Trigger, and Toehold control elements.

    :param plot: Boolean to indicate if the results should be plotted.
    :param parameters_plasmids: Optional dictionary of specific parameters for the plasmids.
    :param print_rules: Boolean to indicate if model rules should be printed.
    :param print_odes: Boolean to indicate if model ODEs should be printed.
    :return: PySB model object.
    """
    # Plasmid design for the cascade:
    # 1. First plasmid expresses Star6
    # 2. Second plasmid uses Sense6 to control Trigger3 transcription
    # 3. Third plasmid uses Trigger3 to control Toehold3, which translates GFP
    plasmids = [
        (None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (("Sense6", "Star6"), None, [(False, "Trigger3")]),  # Step 2: Sense6 (controlled by Star6) controls Trigger3
        (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),  # Step 3: Trigger3 controls Toehold3 and GFP translation
    ]

    # Default parameters for the plasmids if none are provided
    if parameters_plasmids is None:
        parameters_plasmids = {
            "k_Star6_concentration": 1,
            "k_Sense6_Trigger3_concentration": 1,
            "k_Toehold3_GFP_concentration": 1,
        }

    # Load base parameters from the CSV and update with specific plasmid parameters
    try:
        parameters_df = pd.read_csv('../data/model_parameters_priors.csv')
        parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
        parameters.update(parameters_plasmids)
    except FileNotFoundError:
        raise FileNotFoundError("The model_parameters.csv file could not be found. Please check the path.")

    # Setup the model with the plasmids and parameters
    model = setup_model(plasmids, parameters)

    # Time span for simulation
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the simulation
    y_res = simulate_model(model, t)

    if plot:
        # Visualize results
        species_to_plot = list(model.observables.keys())
        visualize_simulation(t, y_res, species_to_plot=species_to_plot)

    # Optional: Print the rules and ODEs for debugging
    if print_rules:
        print("Model Rules:")
        for rule in model.rules:
            print(rule)

    if print_odes:
        print("Model ODEs:")
        for ode in model.odes:
            print(ode)

    return model

if __name__ == "__main__":
    test_cascade(print_rules=True, print_odes=True)
