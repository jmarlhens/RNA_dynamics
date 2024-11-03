
import numpy as np
from pysb import Model, Parameter, Observable
import pandas as pd
from build_simulate_analyse.build_model import setup_model

# import process_plasmid
from rna_dynamics_main import process_plasmid, simulate_model, visualize_simulation

# import parallel tempering
from optimization.parallel_tempering_old import ParallelTempering

def test_star():
    # Plasmid design
    plasmids = [
        (("Sense1", "Star1"), None, [(True, "GFP")]),
        (None, None, [(False, "Star1")]),
    ]

    # Define model parameters
    parameters_plasmids = {
        "k_Sense1_GFP_concentration": 1,
        "k_Star1_concentration": 0.4,
    }

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

    # Visualize results
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

    return model






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

    model = test_star()

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

