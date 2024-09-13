import numpy as np
from build_and_simulate.build_model import setup_model, simulate_model, visualize_simulation
from pysb import Observable
import pandas as pd

def test_iffl_1():
    # Plasmid design for IFFL-1:
    # 1. First plasmid produces Star6 and Trigger3 as the same transcript, cleaved by Csy4.
    # 2. Second plasmid produces aTrigger3 controlled by Sense6.
    # 3. Third plasmid uses Toehold3 to control GFP translation.
    plasmids = [
        (None, None, [(False, "Star6"), (False, "Trigger3")]),  # Step 1: Express Star6 and Trigger3, cleaved by Csy4
        (("Sense6", "Star6"), None, [(False, "aTrigger3")]),  # Step 2: Express aTrigger3 under control of Sense6
        (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),  # Step 3: Use Toehold3 to control GFP translation
    ]

    # Define model parameters
    parameters_plasmids = {
        "k_Star6_Trigger3_concentration": 1,  # Concentration factor for Trigger3 production
        "k_Sense6_aTrigger3_concentration": 1,  # Concentration factor for aTrigger3 production
        "k_Toehold3_GFP_concentration": 1,  # Concentration factor for GFP translation
    }

    # load and add parameters_plasmids
    parameters_df = pd.read_csv('../data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Define sequestration reactions: [(species1, species2)]
    bindings = [
        ("Trigger3", "Sense6_aTrigger3"),  # Sequestration between Trigger3 and aTrigger3
    ]

    # Setup the model with the plasmids, parameters, and bindings
    model = setup_model(plasmids, parameters, bindings=bindings)
    # add observables for visualization
    # Observable("sequestered", model.monomers['RNA_Trigger3'](state="full", binding = 1) %
    #            model.monomers['RNA_Sense6_aTrigger3'](state="full", binding = 1))
    # Observable("free_trigger", model.monomers['RNA_Trigger3'](state="full", binding = None))
    # Observable("free_antitrigger", model.monomers['RNA_Sense6_aTrigger3'](state="full", binding = None))

    # Time span for build_and_simulate
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the build_and_simulate
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())

    # species_to_plot.append("sequestered")

    # y_res.dataframe["sequestered"]
    # y_res.dataframe["free_trigger"]
    # y_res.dataframe["free_antitrigger"]

    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

    model._odes

    print(model.rules)

    for ode in model._odes:
        print(ode)


if __name__ == "__main__":
    test_iffl_1()


