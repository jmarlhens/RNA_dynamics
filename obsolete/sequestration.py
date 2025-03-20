import numpy as np
from circuits.build_model import setup_model, simulate_model, visualize_simulation
from pysb import Observable
import pandas as pd

def test_sequestration(plot=True, parameters_plasmids=None):
    """
    Test sequestration dynamics between Star6 and aStar6.

    :param plot: Boolean to indicate if the results should be plotted.
    :param parameters_plasmids: Optional dictionary of specific parameters for the plasmids.
    :return: PySB model object.
    """
    # Plasmid design for sequestration test:
    # 1. First plasmid expresses Star6
    # 2. Second plasmid expresses aStar6 (anti-Star6)
    plasmids = [
        (None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (None, None, [(False, "aStar6")]),  # Step 2: Express aStar6
    ]

    # Default parameters for the sequestration test if none are provided
    if parameters_plasmids is None:
        parameters_plasmids = {
            'k_Star6_concentration': 1,  # Initial concentration of Star6
            'k_aStar6_concentration': 1,  # Initial concentration of aStar6
        }

    # Load base parameters from the CSV and update with specific plasmid parameters
    parameters_df = pd.read_csv('../data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Define sequestration reactions: [(species1, species2)]
    bindings = [
        ("Star6", "aStar6"),  # Sequestration between Star6 and aStar6
    ]

    # Setup the model with the plasmids, parameters, and bindings
    model = setup_model(plasmids, parameters, bindings=bindings)

    # Add observables for the species of interest (complexed vs free)
    Observable("Star6_free", model.monomers["RNA_Star6"](state="full", sense=None))
    Observable("aStar6_free", model.monomers["RNA_aStar6"](state="full", sense=None))
    Observable("Star6_aStar6_complex", model.monomers["RNA_Star6"](state="full", sense=1) % model.monomers["RNA_aStar6"](state="full", sense=1))

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
    for ode in model.odes:
        print(ode)

    return model

if __name__ == "__main__":
    model = test_sequestration()
