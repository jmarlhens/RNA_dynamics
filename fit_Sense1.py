
import numpy as np
from pysb import Model, Parameter, Rule, Observable

# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# import process_plasmid
from rna_dynamics_main import process_plasmid

# import parallel tempering
from parallel_tempering import simulate_model, visualize_simulation

def test_star():
    # Plasmid design:   First position is transcriptional control (not added to the compartment)
    #                   Second position is translational control (added to the compartment as complex with the following)
    #                   Third position is a list of tuples containing a boolean (True for translation) and a sequence to express (added to the compartment and can be RNA, cleaved RNA (indicated by "cleavage-") and protein)
    # None encodes no control and no sequence

    # The plasmid defines the design of the system
    plasmids = [(("Sense1", "Star1"), None, [(True, "GFP")]),
                (None, None, [(True, "RFP")]),
                # (None, None, [(False, "Trigger1")]),
                (None, None, [(False, "Star1")]),
                ]

    parameters = {"k_tx": 2,
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
                  "k_star_stop_reg": 0.01}

    """
    Model Setup
    """
    omega_val = 6 * 1e23 * np.pi / 2 * 1e-15
    omega_val = 1000000
    model = Model()
    Parameter('omega', omega_val)  # in L

    for param in parameters:
        Parameter(param, parameters[param])

    for plasmid in plasmids:
        process_plasmid(plasmid=plasmid, model=model)

    # Observable("Free_Trigger", RNA_Trigger1(state="full", toehold=None))
    # Observable("Bound_Toehold",
    #            RNA_Trigger1(state="full", toehold=1) % RNA_Toehold1_GFP(state="full", toehold=1))

    n_steps = 100
    t = np.linspace(0, 20, n_steps)
    y_res = simulate_model(model, t)
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)
