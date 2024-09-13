import numpy as np
from simulation.simulate import setup_model, simulate_model, visualize_simulation

def test_star():
    # Plasmid design
    plasmids = [
        (("Sense1", "Star1"), None, [(True, "GFP")]),
        (None, None, [(False, "Star1")]),
    ]

    # Define model parameters
    parameters = {
        "k_tx": 2,
        "k_rna_deg": 0.5,
        "k_tl": 2,
        "k_prot_deg": 0.5,
        "k_mat": 1,
        "k_csy4": 1,
        "k_tl_bound_toehold": 0.1,
        "k_trigger_binding": 5,
        "k_trigger_unbinding": 0.5,
        "k_tx_init": 1,
        "k_star_bind": 5,
        "k_star_unbind": 0.1,
        "k_star_act": 2,
        "k_star_act_reg": 0.01,
        "k_star_stop": 1,
        "k_star_stop_reg": 0.01,
        "k_Sense1_GFP_concentration": 1,
        "k_Star1_concentration": 1,
    }

    # Setup the model
    model = setup_model(plasmids, parameters)

    # Time span for simulation
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the simulation
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

if __name__ == "__main__":
    test_star()
