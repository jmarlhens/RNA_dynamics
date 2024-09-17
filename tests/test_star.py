import numpy as np
from build_simulate_analyse.build_model import setup_model, simulate_model, visualize_simulation
import pandas as pd
from utils.print_odes import find_ODEs_from_Pysb_model, convert_to_latex, write_to_file

def test_star(plot=True, parameters_plasmids={"k_Sense1_GFP_concentration": 1, "k_Star1_concentration": 1}):
    # Plasmid design
    plasmids = [
        (("Sense1", "Star1"), None, [(True, "GFP")]),
        (None, None, [(False, "Star1")]),
    ]


    # load and add parameters_plasmids
    parameters_df = pd.read_csv('../data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Setup the model
    model = setup_model(plasmids, parameters)

    # Time span for build_simulate_analyse
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the build_simulate_analyse
    y_res = simulate_model(model, t)

    if plot:
        # Visualize results
        species_to_plot = list(model.observables.keys())
        visualize_simulation(t, y_res, species_to_plot=species_to_plot)
    return model




if __name__ == "__main__":
    model = test_star()

    equations = find_ODEs_from_Pysb_model(model)
    print(equations)

    # Convert and write to file
    latex_equations = convert_to_latex(equations)
    write_to_file(latex_equations)
