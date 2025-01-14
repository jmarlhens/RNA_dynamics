import numpy as np
from build_simulate_analyse.build_model import setup_model, simulate_model, visualize_simulation
import pandas as pd


def main():
    # Define plasmid design for the build_simulate_analyse
    plasmids = [(None, None, [(True, "GFP")])]  # Example plasmid configuration

    # Time span for build_simulate_analyse
    t = np.linspace(0, 20, 100)

    # Define model parameters
    parameters_plasmids = {
        "k_GFP_concentration": 1
    }

    # load and add parameters_plasmids
    parameters_df = pd.read_csv('data/model_parameters.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Initialize model and add plasmids
    model = setup_model(plasmids, parameters)

    # Run the build_simulate_analyse
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)




if __name__ == "__main__":
    main()
