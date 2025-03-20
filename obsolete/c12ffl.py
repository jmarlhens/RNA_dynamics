import numpy as np
from circuits.build_model import setup_model, simulate_model, visualize_simulation
import pandas as pd

def test_cffl_12(plot=True, parameters_plasmids=None, print_rules=False, print_odes=False):
    """
    Test the CFFL-1,2 circuit dynamics involving Star, Sense, Toehold, and Trigger elements.

    :param plot: Boolean to indicate if the results should be plotted.
    :param parameters_plasmids: Optional dictionary of specific parameters for the plasmids.
    :param print_rules: Boolean to indicate if model rules should be printed.
    :param print_odes: Boolean to indicate if model ODEs should be printed.
    :return: PySB model object.
    """

    plasmids = [
        (None, None, [(False, "Star1")]),  # Express Star1
        (None, None, [(False, "Star6")]),  # Express Star6
        (("Sense1", "Star1"), None, [(False, "aSTAR6")]),  # Express aSTAR6 under control of Sense1
        (("Sense6", "Star6"), None, [(False, "Trigger3")]),  # Express Trigger3 under control of Sense6
        (("Sense6", "Star6"), ("Toehold3", "Trigger3"), [(True, "GFP")]),  # Control GFP with Toehold3 and Trigger3
    ]

    # Default parameters if none provided
    if parameters_plasmids is None:
        parameters_plasmids = {
            "k_Star1_concentration": 1,
            "k_Star6_concentration": 1,
            "k_Sense1_aSTAR6_concentration": 1000,
            "k_Sense6_Trigger3_concentration": 1,
            "k_Sense6_Toehold3_GFP_concentration": 1,
        }

    # Load base parameters from CSV and update with specific plasmid parameters
    try:
        parameters_df = pd.read_csv('../data/model_parameters.csv')
        parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
        parameters.update(parameters_plasmids)
    except FileNotFoundError:
        raise FileNotFoundError("The model_parameters.csv file could not be found. Please check the path.")

    # Define sequestration reactions
    bindings = [
        ("Star6", "Sense1_aSTAR6"),  # Sequestration between Star6 and aSTAR6
    ]

    # Setup the model with the plasmids, parameters, and bindings
    model = setup_model(plasmids, parameters, bindings=bindings)

    # Observables for the output specie


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
    test_cffl_12(plot=True, print_rules=True, print_odes=True)
