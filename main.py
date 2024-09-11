import numpy as np
from simulation.simulate import setup_model, simulate_model, visualize_simulation


def main():
    # Define plasmid design for the simulation
    plasmids = [(None, None, [(True, "GFP")])]  # Example plasmid configuration

    # Time span for simulation
    t = np.linspace(0, 20, 100)

    # Define model parameters
    parameters = {
        "k_tx": 2,
        "k_rna_deg": 0.5,
        "k_tl": 2,
        "k_prot_deg": 0.5,
        "k_mat": 1,
        "k_csy4": 1,
        "k_GFP_concentration": 1
    }

    # Initialize model and add plasmids
    model = setup_model(plasmids, parameters)

    # Run the simulation
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)




if __name__ == "__main__":
    main()
