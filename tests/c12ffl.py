import numpy as np
from build_simulate_analyse.build_model import setup_model, simulate_model, visualize_simulation
from pysb import Observable
import pandas as pd

def test_iffl_1():
    # Plasmid design for IFFL-1:
    # 1. First plasmid produces Star6 and Trigger3 as the same transcript, cleaved by Csy4.
    # 2. Second plasmid produces aTrigger3 controlled by Sense6.
    # 3. Third plasmid uses Toehold3 to control GFP translation.
    plasmids = [
        (None, None, [(False, "Star1")]),
        (None, None, [(False, "Star6")]),
        (("Sense1", "Star1"), None, [(False, "aSTAR6")]),
        (("Sense6", "Star6"), None, [(False, "Trigger3")]),
        (("Sense6", "Star6"), ("Toehold3", "Trigger3"), [(True, "GFP")]),
    ]

    # Define model parameters
    parameters_plasmids = {
        "k_Star1_concentration": 1,
        "k_Star6_concentration": 1,
        "k_Sense1_aSTAR6_concentration": 1,
        "k_Sense6_Trigger3_concentration": 1,
        "k_Sense6_Toehold3_GFP_concentration": 1,
    }

    # load and add parameters_plasmids
    parameters_df = pd.read_csv('../data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    bindings = [
        ("Star6", "Sense1_aSTAR6"),
    ]

    # Setup the model with the plasmids, parameters, and bindings
    model = setup_model(plasmids, parameters, bindings=bindings)

    # Time span for build_simulate_analyse
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the build_simulate_analyse
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())

    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

    model._odes

    print(model.rules)

    for ode in model._odes:
        print(ode)


if __name__ == "__main__":
    test_iffl_1()


