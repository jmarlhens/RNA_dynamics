import numpy as np
from build_simulate_analyse.build_model import setup_model, simulate_model, visualize_simulation
import pandas as pd
from utils.print_odes import find_ODEs_from_Pysb_model, convert_to_latex, write_to_file

def test_pos_control(plot=True, print_rules=True, print_odes=True, parameters_plasmids={"k_GFP_concentration": 3}):
    # Plasmid design
    plasmids = [
        (None, None, [(True, "GFP")]),
    ]


    # load and add parameters_plasmids
    parameters_df = pd.read_csv('data/model_parameters_priors.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Setup the model
    model = setup_model(plasmids, parameters)

    # Time span for build_simulate_analyse
    T = 600
    n_steps = T * 10
    t = np.linspace(0, T, n_steps)

    # Run the build_simulate_analyse
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

        equations = find_ODEs_from_Pysb_model(model)
        print(equations)

    return model




if __name__ == "__main__":
    model = test_pos_control()




