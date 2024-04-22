import matplotlib.pyplot as plt
import numpy as np
from pysb import Model, Parameter, Rule, Observable
from pysb.simulator import ScipyOdeSimulator

from modules.Csy4_activity import Csy4Activity
from modules.base_modules import Transcription, Translation
from modules.molecules import RNA


def transcription(RNA, model):
    parameters = model.parameters
    k_tx = parameters["k_tx"]
    k_deg = parameters["k_deg"]
    Rule(f'transcription_{RNA.name}', None >> RNA(state="full"), k_tx)
    Rule(f'degradation_{RNA.name}', RNA(state="full") >> None, k_deg)
    pass


def translation(RNA, prot, model):
    parameters = model.parameters
    k_tx = parameters["k_tx"]
    k_deg = parameters["k_deg"]
    Rule(f'translation_{prot.name}', RNA(state="full") >> prot(state="unmature"), k_tx)
    Rule(f'degradation_{prot.name}', Protein(state="full") >> None, k_deg)
    pass


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

    rna_name = transcriptional_control if transcriptional_control else ""
    rna_name += ("_" + translational_control) if translational_control else ""
    rna_name += "_" if len(rna_name) > 0 else ""
    rna_name += "_".join([elem[1] for elem in cds])

    # All sequences encoded are transcribed jointly
    # After their transcription, they can be cleaved
    if transcriptional_control:
        pass
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
            pass
        else:
            Translation(rna=rna, model=model)


if __name__ == '__main__':
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

    # Four types of RNA included in the system
    #    Monomer("RNA", ["state"], {"state": {"full", "partial", "init"}})  # Raw
    #    Monomer("RNA", ["sense", "state"], {"state": {"full", "partial", "init"}})  # Sense
    #    Monomer("RNA", ["toehold", "state"], {"state": {"full", "partial", "init"}})  # Toehold

    # Monomer("RNA", ["sense", "toehold", "state"], {"state": {"full", "partial", "init"}})  # Sense and Toehold
    # Monomer("RNA_1", ["sense", "toehold", "state"], {"state": {"full", "partial", "init"}})  # Sense and Toehold
    # Monomer("RNA_2", ["sense", "toehold", "state"], {"state": {"full", "partial", "init"}})  # Sense and Toehold
    # rule = Rule(f'RNA_degradation_{RNA.name}', RNA() >> None, k_rna_deg)
    # rule = Rule(f'RNA_degradation_{RNA_1.name}', RNA_1() >> None, k_rna_deg)
    # rule = Rule(f'RNA_degradation_{RNA_2.name}', RNA_2() >> None, k_rna_deg)
    #
    # Monomer("Protein", ["state"], {"state": {"mature", "immature"}})
    # rule = Rule(f'degradation_{Protein.name}', Protein() >> None, k_prot_deg)
    #
    # Transcription(sequence=RNA, model=model)
    # Csy4Activity(rna=RNA, product_rnas=[RNA_1, RNA_2], model=model)
    # Translation(rna=RNA_1, protein=Protein, model=model)
    #
    # Observable("obs_RNA", RNA(state="full"))
    # Observable("obs_RNA_1", RNA_1(state="full"))
    # Observable("obs_RNA_Gfp", RNA_2(state="full"))
    # Observable("obs_Protein_immature", Protein(state="immature"))
    # Observable("obs_Protein_mature", Protein(state="mature"))

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
    model_simulator = ScipyOdeSimulator(model)
    y_res = model_simulator.run(tspan=t, param_values=parameters)

    y = y_res.all
    fig, ax = plt.subplots()
    species = y_res.dataframe.columns
    for spec in species:
        if spec not in model.observables.keys():
            continue
        ax.plot(t, y_res.dataframe[spec], ":", label=spec.replace("obs_", ""))

    ax.legend()
    plt.show()
    pass
