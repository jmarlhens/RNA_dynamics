import numpy as np
from build_simulate_analyse.build_model import setup_model, simulate_model, visualize_simulation
from pysb import Observable
import pandas as pd


def test_iffl_1(plot=True, parameters_plasmids=None, print_rules=False, print_odes=False):
    """
    Test the Incoherent Feed-Forward Loop (IFFL-1) dynamics with sequestration.

    :param plot: Boolean to indicate if the results should be plotted.
    :param parameters_plasmids: Optional dictionary of specific parameters for the plasmids.
    :return: PySB model object.
    """
    # Plasmid design for IFFL-1:
    # 1. First plasmid produces Star6 and Trigger3 as the same transcript, cleaved by Csy4.
    # 2. Second plasmid produces aTrigger3 controlled by Sense6.
    # 3. Third plasmid uses Toehold3 to control GFP translation.
    plasmids = [
        (None, None, [(False, "Star6"), (False, "Trigger3")]),  # Step 1: Express Star6 and Trigger3, cleaved by Csy4
        (("Sense6", "Star6"), None, [(False, "aTrigger3")]),  # Step 2: Express aTrigger3 under control of Sense6
        (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),  # Step 3: Use Toehold3 to control GFP translation
    ]

    # Default parameters for IFFL-1 if none are provided
    if parameters_plasmids is None:
        parameters_plasmids = {
            "k_Star6_Trigger3_concentration": 1,  # Concentration factor for Trigger3 production
            "k_Sense6_aTrigger3_concentration": 1,  # Concentration factor for aTrigger3 production
            "k_Toehold3_GFP_concentration": 1,  # Concentration factor for GFP translation
        }

    # Load base parameters from the CSV and update with specific plasmid parameters
    parameters_df = pd.read_csv('../data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Define sequestration reactions: [(species1, species2)]
    bindings = [
        ("Trigger3", "aTrigger3"),  # Sequestration between Trigger3 and aTrigger3
    ]

    # Setup the model with the plasmids, parameters, and bindings
    model = setup_model(plasmids, parameters, bindings=bindings)

    # more observalbes: complex trigger3/antitrigger3
    Observable('Obs_free_RNA_Trigger3', model.monomers['RNA_Trigger3'](state='full', toehold=None))
    Observable('Obs_RNA_aTrigger3', model.monomers['RNA_aTrigger3'](state='full'))
    Observable('Obs_RNA_Trigger3_aTrigger3', model.monomers['RNA_Trigger3'](state='full', toehold=1) % model.monomers['RNA_aTrigger3'](state='full', toehold=1))

    # Time span for simulation
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the simulation
    y_res = simulate_model(model, t)

    if plot:
        # Visualize results
        species_to_plot = list(model.observables.keys())
        visualize_simulation(t, y_res, species_to_plot=species_to_plot)

    if print_rules:
        print("Model Rules:")
        for rule in model.rules:
            print(rule)

    if print_odes:
        print("\nModel ODEs:")
        for ode in model.odes:
            print(ode)

    return model


if __name__ == "__main__":
    model = test_iffl_1(print_rules=True, print_odes=True)
