import numpy as np
from circuits.build_model import setup_model, simulate_model, visualize_simulation
import pandas as pd


def test_AND_gate(plot=True, parameters_plasmids=None, print_rules=False, print_odes=False):
    """
    Test the AND gate circuit dynamics involving STAR, Sense, Toehold, and Trigger elements.

    :param plot: Boolean to indicate if the results should be plotted.
    :param parameters_plasmids: Optional dictionary of specific parameters for the plasmids.
    :param print_rules: Boolean to indicate if model rules should be printed.
    :param print_odes: Boolean to indicate if model ODEs should be printed.
    :return: PySB model object.
    """
    # Plasmid design for the AND gate:
    # 1. First plasmid uses Sense6 and STAR6 for transcriptional and translational control.
    # 2. Second plasmid expresses STAR6.
    # 3. Third plasmid expresses Trigger3.
    plasmids = [
        (("Sense_6", "STAR_6"), ("Toehold_3", "Trigger_3"), [(True, "GFP")]),  # AND gate: Sense6/STAR6 and Toehold/Trigger3
        (None, None, [(False, "STAR_6")]),  # Express STAR6
        (None, None, [(False, "Trigger_3")]),  # Express Trigger3
    ]

    # Default parameters if none provided
    if parameters_plasmids is None:
        parameters_plasmids = {
            "k_Sense_6_Toehold_3_GFP_concentration": 1,
            "k_STAR_6_concentration": 1,
            "k_Trigger_3_concentration": 1,
        }

    # Load base parameters from CSV and update with specific plasmid parameters
    try:
        parameters_df = pd.read_csv('../data/model_parameters.csv')
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
    test_AND_gate(plot=True, print_rules=True, print_odes=True)
