import numpy as np
from simulation.simulate import setup_model, simulate_model, visualize_simulation

def test_cleaved_transcription_and_translation():
    # Plasmid design
    plasmid = (None, None, [(False, "STAR"), (True, "GFP")])

    # Define model parameters
    parameters = {
        "k_tx": 2,
        "k_rna_deg": 0.5,
        "k_tl": 2,
        "k_prot_deg": 0.5,
        "k_mat": 1,
        "k_csy4": 1,
        "k_STAR_GFP_concentration": 1,
    }

    # Setup the model with the plasmid and parameters
    model = setup_model([plasmid], parameters)

    # Time span for simulation
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the simulation
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

if __name__ == "__main__":
    test_cleaved_transcription_and_translation()
