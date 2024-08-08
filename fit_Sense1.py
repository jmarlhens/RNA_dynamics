
import numpy as np
from pysb import Model, Parameter, Observable
import pandas as pd
from modules.molecules import RNA


# import process_plasmid
from rna_dynamics_main import process_plasmid, simulate_model, visualize_simulation

# import parallel tempering
from optimization.parallel_tempering import ParallelTempering


def STAR_model(parameters):
    # The plasmid defines the design of the system
    plasmids = [(("Sense1", "Star1"), None, [(True, "GFP")]),
                # (None, None, [(False, "Trigger1")]),
                (None, None, [(False, "Star1")]),
                ]


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



    return model

def test_star(model):
    # Plasmid design:   First position is transcriptional control (not added to the compartment)
    #                   Second position is translational control (added to the compartment as complex with the following)
    #                   Third position is a list of tuples containing a boolean (True for translation) and a sequence to express (added to the compartment and can be RNA, cleaved RNA (indicated by "cleavage-") and protein)
    # None encodes no control and no sequence



    # Observable("Free_Trigger", RNA_Trigger1(state="full", toehold=None))
    # Observable("Bound_Toehold",
    #            RNA_Trigger1(state="full", toehold=1) % RNA_Toehold1_GFP(state="full", toehold=1))

    n_steps = 100
    t = np.linspace(0, 20, n_steps)
    y_res = simulate_model(model, t)
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)


if __name__ == "__main__":
    parameters = {"k_tx": 2,
                  "k_Sense1_GFP_concentration": 1,
                  "k_Star1_concentration": 1,
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
                  "k_star_act": 0.01,
                  "k_star_act_reg": 3,
                  "k_star_stop": 3,
                  "k_star_stop_reg": 0.01}

    # initialize the model
    model = STAR_model(parameters)

    # Experimental Design
    parameters_to_be_fitted = ["k_tx", "k_rna_deg", "k_tl", "k_prot_deg", "k_mat", "k_csy4", "k_tl_bound_toehold",
                                 "k_trigger_binding", "k_trigger_unbinding", "k_tx_init", "k_star_bind", "k_star_unbind",
                                    "k_star_act", "k_star_act_reg", "k_star_stop", "k_star_stop_reg"]

    # concentrations (nM) of each plasmid for each condition
    experimental_design = pd.DataFrame({"Se1 3 nM": {"k_Star1_concentration": 0, "k_Sense1_GFP_concentration": 3},
                           "Se1 3 nM + St1 15 nM": {"k_Star1_concentration": 15, "k_Sense1_GFP_concentration": 3}})
    conditions = experimental_design.columns
    plasmid_concentrations = experimental_design.index

    for condition in conditions:
        for parameter in plasmid_concentrations:
            model.parameters[parameter].value = experimental_design.loc[parameter, condition]

        test_star(model)

    pass

