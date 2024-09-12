import numpy as np
from simulation.simulate import setup_model, simulate_model, visualize_simulation

def test_coherent_feed_forward_loop():
    # Plasmid design for the coherent feed-forward loop:
    # 1. First plasmid expresses Star6
    # 2. Second plasmid uses Sense6 to control Trigger1 transcription
    # 3. Third plasmid uses Sense6 and Toehold1 to control GFP transcription and translation
    plasmids = [
        (None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (("Sense6", "Star6"), None, [(False, "Trigger1")]),  # Step 2: Sense6 (controlled by Star6) controls Trigger1
        (("Sense6", "Star6"), ("Toehold1", "Trigger1"), [(True, "GFP")]),  # Step 3: Sense6 and Toehold1 control GFP
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
        "k_Sense6_Trigger1_concentration": 1,
        "k_Star6_concentration": 1,
        "k_Sense6_Toehold1_GFP_concentration": 1,
    }

    # Setup the model with the plasmids and parameters
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
    test_coherent_feed_forward_loop()
