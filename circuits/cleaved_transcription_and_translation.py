import numpy as np
from build_simulate_analyse.build_model import setup_model, simulate_model, visualize_simulation
from pysb import Observable
import pandas as pd


def test_cleaved_transcription_and_translation(plot=True, parameters_plasmids=None):
    """
    Test cleaved transcription and translation dynamics with STAR and GFP.

    :param plot: Boolean to indicate if the results should be plotted.
    :param parameters_plasmids: Optional dictionary of specific parameters for the plasmids.
    :return: PySB model object.
    """
    # Plasmid design
    plasmid = (None, None, [(False, "STAR"), (True, "GFP")])

    # Default parameters for cleaved transcription and translation if none are provided
    if parameters_plasmids is None:
        parameters_plasmids = {
            "k_STAR_GFP_concentration": 1,  # Concentration factor for STAR and GFP
        }

    # Load base parameters from the CSV and update with specific plasmid parameters
    parameters_df = pd.read_csv('../data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Setup the model with the plasmid and parameters
    model = setup_model([plasmid], parameters)

    # Time span for simulation
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the simulation
    y_res = simulate_model(model, t)

    if plot:
        # Visualize results
        species_to_plot = list(model.observables.keys())
        visualize_simulation(t, y_res, species_to_plot=species_to_plot)

    # Optional: Print the ODEs for debugging
    print("Model ODEs:")
    for ode in model.odes:
        print(ode)

    return model


if __name__ == "__main__":
    test_cleaved_transcription_and_translation()
