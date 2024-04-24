import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pydream.convergence import Gelman_Rubin
from pysb import Model, Parameter, Rule, Observable
from pysb.simulator import ScipyOdeSimulator

from modules.Csy4_activity import Csy4Activity
from modules.base_modules import Transcription, Translation
from modules.molecules import RNA
from modules.star_ek import STAR
from modules.toehold import Toehold
from modules.star_ek import STAR
from utils import print_odes

from pydream.core import run_dream
from pydream.parameters import SampledParam
from scipy.stats import norm


#
# def transcription(RNA, model):
#     parameters = model.parameters
#     k_tx = parameters["k_tx"]
#     k_deg = parameters["k_deg"]
#     Rule(f'transcription_{RNA.name}', None >> RNA(state="full"), k_tx)
#     Rule(f'degradation_{RNA.name}', RNA(state="full") >> None, k_deg)
#     pass
#
#
# def translation(RNA, prot, model):
#     parameters = model.parameters
#     k_tx = parameters["k_tx"]
#     k_deg = parameters["k_deg"]
#     Rule(f'translation_{prot.name}', RNA(state="full") >> prot(state="unmature"), k_tx)
#     Rule(f'degradation_{prot.name}', Protein(state="full") >> None, k_deg)
#     pass


# def process_plasmid(plasmid, part_library):
#     sense_name, toehold_name, cds_name = plasmid
#
#     # ToDo Add the respective rules and params and not only the species
#     if sense_name is None and toehold_name is None:
#         Monomer("RNA", ["state"], {"state": {"full", "partial", "init"}})  # Raw
#     elif sense_name is not None and toehold_name is None:
#         Monomer("RNA", ["sense", "state"], {"state": {"full", "partial", "init"}})  # Sense
#     elif sense_name is None and toehold_name is not None:
#         Monomer("RNA", ["toehold", "state"], {"state": {"full", "partial", "init"}})  # Toehold
#     else:
#         Monomer("RNA", ["sense", "toehold", "state"], {"state": {"full", "partial", "init"}})  # Sense and Toehold
#
#     if cds_name is not None:
#         # ToDo Add the respective rules for protein production and maturation
#         Monomer("Protein", ["state"], {"state": {"mature", "unmature"}})

def process_plasmid(plasmid, model):
    transcriptional_control = plasmid[0]
    translational_control = plasmid[1]
    cds = plasmid[2]
    if len(cds) == 0:
        return

    cleave_rna = len(cds) > 1

    sequence_names = [elem[1] for elem in cds]

    rna_name = transcriptional_control[0] if transcriptional_control else ""
    rna_name += "_" if len(rna_name) > 0 else ""
    rna_name += translational_control[0] if translational_control else ""
    rna_name += "_" if len(rna_name) > 0 else ""
    rna_name += "_".join([elem[1] for elem in cds])

    # All sequences encoded are transcribed jointly
    # After their transcription, they can be cleaved
    if transcriptional_control:
        star = STAR(sequence_name=rna_name, transcriptional_control=transcriptional_control, model=model)
        rna = star.product
        products = [rna]
        # Add STAR
        # (self, rna: RNA = None, rna_regulator: RNA = None, model: Model = None)
        # star_tmp = STAR(rna=rna_name, rna_regulator=transcriptional_control, model=model)
        # products = star_tmp.product
    else:
        transcription = Transcription(sequence_name=rna_name, model=model)
        rna = transcription.product
        products = [rna]

    if cleave_rna:
        csy4 = Csy4Activity(rna=rna, product_rna_names=sequence_names, model=model)
        products = csy4.product

    # After cleavage (or single sequences), the sequences are translated independently.
    for (translate, seq), rna in zip(cds, products):
        if not translate:
            continue
        if translational_control:
            Toehold(rna=rna, translational_control=translational_control, prot_name=seq, model=model)
        else:
            Translation(rna=rna, model=model)


def simulate_model(model, t):
    observable_names = []
    observables = []
    for monomer in model.monomers:
        desired_state = "full" if isinstance(monomer, RNA) else "mature"
        obs_name = "obs_" + monomer.name
        observable = Observable(obs_name, monomer(state=desired_state))
        observable_names.append(obs_name)
        observables.append(observable)

    # Observable("Protein_immature", Protein_GFP(state="immature"))

    # List of parameters yield that the same model is evaluated for different parameter instantiations but not for changing parameters.
    simulator = ScipyOdeSimulator(model)
    y_res = simulator.run(tspan=t)
    # odes = print_odes.find_ODEs_from_Pysb_model(model)

    return y_res


def visualize_simulation(t, y_res, species_to_plot):
    fig, ax = plt.subplots()
    markers = []

    species = y_res.dataframe.columns
    for iS, spec in enumerate(species):
        if spec not in species_to_plot:
            continue

        ax.plot(t, y_res.dataframe[spec], linestyle=(iS, [2, 6]), lw=3, label=spec.replace("obs_", ""))

    ax.legend()
    plt.show()


def test_cleaved_transcription_and_translation():
    # Plasmid design:   First position is transcriptional control (not added to the compartment)
    #                   Second position is translational control (added to the compartment as complex with the following)
    #                   Third position is a list of tuples containing a boolean (True for translation) and a sequence to express (added to the compartment and can be RNA, cleaved RNA (indicated by "cleavage-") and protein)
    # None encodes no control and no sequence

    # The plasmid defines the design of the system
    plasmid = (None, None, [(False, "STAR"), (True, "GFP")])

    parameters = {"k_tx": 2,
                  "k_rna_deg": 0.5,
                  "k_tl": 2,
                  "k_prot_deg": 0.5,
                  "k_mat": 1,
                  "k_csy4": 1}

    """
    Model Setup
    """
    # omega_val = 6 * 1e23 * np.pi / 2 * 1e-15
    omega_val = 1000000
    model = Model()
    Parameter('omega', omega_val)  # in L

    for param in parameters:
        Parameter(param, parameters[param])

    process_plasmid(plasmid=plasmid, model=model)

    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    y_res = simulate_model(model, t)

    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)


def test_toehold():
    # Plasmid design:   First position is transcriptional control (not added to the compartment)
    #                   Second position is translational control (added to the compartment as complex with the following)
    #                   Third position is a list of tuples containing a boolean (True for translation) and a sequence to express (added to the compartment and can be RNA, cleaved RNA (indicated by "cleavage-") and protein)
    # None encodes no control and no sequence

    # The plasmid defines the design of the system
    plasmids = [(None, ("Toehold1", "Trigger1"), [(True, "GFP")]),
                # (None, None, [(False, "Trigger1")]),
                (None, None, [(False, "Trigger2")]),
                ]

    parameters = {"k_tx": 2,
                  "k_rna_deg": 0.5,
                  "k_tl": 2,
                  "k_prot_deg": 0.5,
                  "k_mat": 1,
                  "k_csy4": 1,
                  "k_tl_bound_toehold": 0.1,
                  "k_trigger_binding": 5,
                  "k_trigger_unbinding": 0.5}

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

    Observable("Free_Trigger", RNA_Trigger1(state="full", toehold=None))
    Observable("Bound_Toehold",
               RNA_Trigger1(state="full", toehold=1) % RNA_Toehold1_GFP(state="full", toehold=1))

    n_steps = 100
    t = np.linspace(0, 20, n_steps)
    y_res = simulate_model(model, t)
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)


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


def test_AND_gate():
    plasmids = [(("Sense_6", "STAR_6"), ("Toehold_3", "Trigger_3"), [(True, "GFP")]),
                (None, None, [(False, "STAR_6")]),
                (None, None, [(False, "Trigger_3")]),
                ]

    parameters = {"k_tx": 2,
                  "k_rna_deg": 0.5,
                  "k_tl": 2,
                  "k_prot_deg": 0.5,
                  "k_mat": 1,
                  "k_csy4": 1,
                  "k_tl_bound_toehold": 0.1,
                  "k_trigger_binding": 5,
                  "k_trigger_unbinding": 0.5}

    omega_val = 1000000
    model = Model()
    Parameter('omega', omega_val)  # in L

    for param in parameters:
        Parameter(param, parameters[param])

    for plasmid in plasmids:
        process_plasmid(plasmid=plasmid, model=model)

    # Observe the gfp protein
    Observable("Protein_GFP", Protein_GFP(state="mature"))

    n_steps = 100
    t = np.linspace(0, 20, n_steps)
    y_res = simulate_model(model, t)
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)



simulator = None
experimental_data_protein = None
param_ids = None


def likelihood(params):
    if any(p <= 0 for p in params):  # Parameters must be positive
        return -np.inf
    param_dict = {param_id: val for param_id, val in zip(param_ids, params)}
    sim_result = simulator.run(param_values=param_dict)
    # sim_data_mRNA = sim_result.observables['mRNA_obs']
    sim_data_protein = sim_result.observables['obs_Protein_GFP']
    # ToDo For adequate likelihood treatment see:
    # https://github.com/LoLab-MSM/PyDREAM/blob/master/pydream/examples/robertson/example_sample_robertson_with_dream.py#L38

    # ll_mRNA = -np.sum((sim_data_mRNA - experimental_data_mRNA) ** 2)
    ll_protein = -np.sum((sim_data_protein - experimental_data_protein) ** 2)
    # ll = ll_mRNA + ll_protein
    ll = ll_protein

    return ll


if __name__ == '__main__':
    """
    AND gate
    """

    test_AND_gate()


    # test_cleaved_transcription_and_translation()
    test_toehold()
    # test_star()
    exit(0)
    # Plasmid design:   First position is transcriptional control (not added to the compartment)
    #                   Second position is translational control (added to the compartment as complex with the following)
    #                   Third position is a list of tuples containing a boolean (True for translation) and a sequence to express (added to the compartment and can be RNA, cleaved RNA (indicated by "cleavage-") and protein)
    # None encodes no control and no sequence

    # The plasmid defines the design of the system
    plasmids = [  # (None, None, None),
        (None, None, [(True, "GFP")]),
        (None, None, [(False, "STAR"), (True, "GFP")]),
        ("Sense1", "Toehold1", [(True, "GFP")]),
        ("Sense1", "Toehold1", [(False, "STAR"), (True, "GFP")])]

    # The list of induced elements
    induced_components = []

    parameters = {"k_tx": 2,
                  "k_rna_deg": 0.5,
                  "k_tl": 2,
                  "k_prot_deg": 0.5,
                  "k_mat": 1,
                  "k_csy4": 1}

    param_ids = list(parameters.keys())
    """
    Model Setup
    """
    omega_val = 6 * 1e23 * np.pi / 2 * 1e-15
    omega_val = 1000000
    model = Model()
    Parameter('omega', omega_val)  # in L

    for param in parameters:
        Parameter(param, parameters[param])

    # process_plasmid(plasmid=plasmids[0], model=model)
    process_plasmid(plasmid=plasmids[1], model=model)
    # STAR_D binds to Sense_D and Trigger_D binds to Toehold_D with binding and unbinding rates s_D and t_D respectively

    # States of any RNA for the STAR:
    #               free: the RNA is complete and isn't bound to anything
    #               bound: the RNA is complete and bound to a cognate RNA sequence
    #               init: the RNA is incomplete and bound to RNAP at the gene
    #               init-bound: the RNA is incomplete and bound to another RNA (and the RNAP)
    #               partial: the RNA is incomplete and unbound
    #               partial-bound: the RNA is incomplete and bound to another RNA
    # In pysb these states can be subsumed to full, init, and partial with a binding site for RNA
    # To also allow for the Toehold at the same RNA, another binding site for the Trigger needs to be added

    """
    Simulation
    """

    observable_names = []
    observables = []
    for monomer in model.monomers:
        desired_state = "full" if isinstance(monomer, RNA) else "mature"
        obs_name = "obs_" + monomer.name
        observable = Observable(obs_name, monomer(state=desired_state))
        observable_names.append(obs_name)
        observables.append(observable)

    # Observable("Protein_immature", Protein_GFP(state="immature"))

    n_steps = 100
    t = np.linspace(0, 20, n_steps)
    # List of parameters yield that the same model is evaluated for different parameter instantiations but not for changing parameters.
    parameters = {param_id: parameters[param_id] * np.ones(1) for param_id in parameters}
    parameters["k_tx"][50:] = 0
    parameters = {param_id: parameters[param_id].tolist() for param_id in parameters}
    simulator = ScipyOdeSimulator(model)
    y_res = simulator.run(tspan=t, param_values=parameters)
    # odes = print_odes.find_ODEs_from_Pysb_model(model)

    y = y_res.all
    fig, ax = plt.subplots()

    species = y_res.dataframe.columns
    for spec in species:
        if spec not in model.observables.keys():
            continue
        ax.plot(t, y_res.dataframe[spec], ":", label=spec.replace("obs_", ""))

    ax.legend()
    plt.show()

    # experimental_data_protein = y_res.observables['obs_Protein_GFP'] + np.random.normal(scale=5, size=len(t))
    #
    # priors = [
    #     SampledParam(norm, loc=1, scale=1),  # Prior for k_tx
    #     SampledParam(norm, loc=1, scale=1),  # Prior for k_rna_deg
    #     SampledParam(norm, loc=1, scale=1),  # Prior for k_tl
    #     SampledParam(norm, loc=1, scale=1),  # Prior for k_prot_deg
    #     SampledParam(norm, loc=1, scale=1),  # Prior for k_mat
    #     SampledParam(norm, loc=1, scale=1)  # Prior for k_csy4
    # ]
    #
    # sampled_params, log_ps = run_dream(priors, likelihood, niterations=5000, nchains=4, multitry=False, gamma_levels=4,
    #                                    adapt_gamma=True, history_thin=1)
    #
    # sampled_params = np.array(sampled_params)
    # np.save('dream_samples.npy', sampled_params)
    # converged = Gelman_Rubin(sampled_params)
    # print('Convergence statistic:', converged)

    pass
