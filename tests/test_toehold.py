import numpy as np
from simulation.simulate import setup_model, simulate_model, visualize_simulation
from pysb import Observable

def test_toehold():
    # Plasmid design
    plasmids = [
        (None, ("Toehold1", "Trigger1"), [(True, "GFP")]),
        (None, None, [(False, "Trigger1")]),
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
        "k_Toehold1_GFP_concentration": 1,
        "k_Trigger1_concentration": 1,
    }

    # Setup the model
    model = setup_model(plasmids, parameters)

    # Add observables for visualization
    Observable("Free_Trigger", model.monomers['RNA_Trigger1'](state="full", toehold=None))
    Observable("Bound_Toehold", model.monomers['RNA_Trigger1'](state="full", toehold=1) %
               model.monomers['RNA_Toehold1_GFP'](state="full", toehold=1))

    # Time span for simulation
    n_steps = 1000
    t = np.linspace(0, 20, n_steps)

    # Run the simulation
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

if __name__ == "__main__":
    test_toehold()
