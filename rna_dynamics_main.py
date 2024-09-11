import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pydream.convergence import Gelman_Rubin
from pysb import Model, Parameter, Rule, Observable
from pysb.simulator import ScipyOdeSimulator, BngSimulator

from modules.Csy4_activity import Csy4Activity
from modules.base_modules import Transcription, Translation
from modules.molecules import RNA
from modules.star import STAR
from modules.toehold import Toehold
from optimization.parallel_tempering import ParallelTempering
from utils import print_odes

from pydream.core import run_dream
from pydream.parameters import SampledParam
from scipy.stats import norm
import scipy.stats


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

    # rna_name = transcriptional_control[0] if transcriptional_control else ""
    # rna_name += "_" if len(rna_name) > 0 else ""
    # rna_name += translational_control[0] if translational_control else ""
    # rna_name += "_" if len(rna_name) > 0 and rna_name[-1] != "_" else ""
    # rna_name += "_".join([elem[1] for elem in cds])

    rna_name = []
    if transcriptional_control:
        rna_name.append(transcriptional_control[0])
    if translational_control:
        rna_name.append(translational_control[0])
    rna_name += [elem[1] for elem in cds]
    rna_name = "_".join(rna_name)

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
            Translation(rna=rna, prot_name=seq, model=model)

def generate_observables(model):
    observable_names = []
    observables = []
    for monomer in model.monomers:
        desired_state = "full" if isinstance(monomer, RNA) else "mature"
        obs_name = "obs_" + monomer.name
        observable = Observable(obs_name, monomer(state=desired_state))
        observable_names.append(obs_name)
        observables.append(observable)


def simulate_model(model, t):
    # List of parameters yield that the same model is evaluated for different parameter instantiations but not for changing parameters.
    # cur_simulator = BngSimulator(model)
    cur_simulator = ScipyOdeSimulator(model)
    y_res = cur_simulator.run(tspan=t)
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
    generate_observables(model)

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
                (None, None, [(False, "Trigger1")]),
                # (None, None, [(False, "Trigger2")]),
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
                  "k_Toehold1_GFP_concentration": 1,
                  "k_Trigger1_concentration": 1,
                  }

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

    n_steps = 1000
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
    generate_observables(model)
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
                  "k_trigger_unbinding": 0.5,
                  "k_tx_init": 1,
                  "k_star_bind": 5,
                  "k_star_unbind": 0.1,
                  "k_star_act": 2,
                  "k_star_act_reg": 0.01,
                  "k_star_stop": 1,
                  "k_star_stop_reg": 0.01,
                  "k_Sense_6_Toehold_3_GFP_concentration": 1,
                  "k_STAR_6_concentration": 1,
                  "k_Trigger_3_concentration": 1,
                  }

    omega_val = 1000000
    model = Model()
    Parameter('omega', omega_val)  # in L

    for param in parameters:
        Parameter(param, parameters[param])

    for plasmid in plasmids:
        process_plasmid(plasmid=plasmid, model=model)
    generate_observables(model)
    # Observe the gfp protein
    # Observable("Protein_GFP", Protein_GFP(state="mature"))

    n_steps = 100
    t = np.linspace(0, 20, n_steps)
    y_res = simulate_model(model, t)
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)


def get_wrapped_model(t, plasmids, observable_names, fixed_parameters=None):
    # Plasmid design:   First position is transcriptional control (not added to the compartment)
    #                   Second position is translational control (added to the compartment as complex with the following)
    #                   Third position is a list of tuples containing a boolean (True for translation) and a sequence to express (added to the compartment and can be RNA, cleaved RNA (indicated by "cleavage-") and protein)
    # None encodes no control and no sequence
    def my_model(parameters: dict):
        cur_params = {id: 10 ** (parameters[id]) for id in parameters}
        if fixed_parameters:
            cur_params.update({id: 10 ** (fixed_parameters[id]) for id in fixed_parameters})

        """
        Model Setup
        """
        omega_val = 6 * 1e23 * np.pi / 2 * 1e-15
        omega_val = 1000000
        model = Model()
        Parameter('omega', omega_val)  # in L

        for id in cur_params:
            Parameter(id, cur_params[id])

        for plasmid in plasmids:
            process_plasmid(plasmid=plasmid, model=model)

        generate_observables(model)

        y_res = simulate_model(model, t)
        species_to_plot = list(model.observables.keys())
        # visualize_simulation(t, y_res, species_to_plot=species_to_plot)
        measurement_data = y_res.dataframe[observable_names]
        return measurement_data

    return my_model


def get_model_dummy(t, plasmids, observable_names, fixed_parameters=None):
    # Plasmid design:   First position is transcriptional control (not added to the compartment)
    #                   Second position is translational control (added to the compartment as complex with the following)
    #                   Third position is a list of tuples containing a boolean (True for translation) and a sequence to express (added to the compartment and can be RNA, cleaved RNA (indicated by "cleavage-") and protein)
    # None encodes no control and no sequence
    def my_model(parameters: dict):
        cur_params = {id: 10 ** (parameters[id]) for id in parameters}
        if fixed_parameters:
            cur_params.update({id: 10 ** (fixed_parameters[id]) for id in fixed_parameters})

        """
        Model Setup
        """
        l = cur_params["k_tx"] * cur_params["k_tl"] / (cur_params["k_rna_deg"] * cur_params["k_prot_deg"])
        vals = l * (t ** 3 / (np.sqrt(l) ** 3 + t ** 3))
        data = {"t": t, "obs_Protein_RFP": vals}
        measurement_data = pd.DataFrame(data=data)
        measurement_data.set_index("t", inplace=True)
        return measurement_data

    return my_model


def get_log_likelihood(wrapped_model, observable_names: list, n_replicates: int, experimental_data: pd.DataFrame):
    likelihoods = {}
    for observable in observable_names:
        y_meas = experimental_data[[col for col in experimental_data.columns if observable in col]].values
        mean = np.mean(y_meas, axis=1)
        var = np.var(y_meas, axis=1)
        likelihoods[observable] = norm(loc=mean, scale=np.sqrt(var))

    def log_likelihood(parameters):
        sim_result = wrapped_model(parameters)

        sim_data_protein = sim_result[observable_names]
        # ToDo For adequate likelihood treatment see:
        # https://github.com/LoLab-MSM/PyDREAM/blob/master/pydream/examples/robertson/example_sample_robertson_with_dream.py#L38
        ll = 0
        for observable in observable_names:
            y_pred = sim_data_protein[observable].values.reshape(-1, 1)
            y_meas = experimental_data[[col for col in experimental_data.columns if observable in col]].values
            cur_var = np.var(y_meas, axis=1).reshape(-1, 1)
            exp_part = np.power(y_meas - y_pred, 2) / (2 * cur_var)
            non_exp_part = 0.5 * np.log(2 * np.pi * cur_var)
            cur_ll = non_exp_part + exp_part
            cur_ll[np.isnan(cur_ll)] = 0
            ll += - np.sum(cur_ll)
            # ll_mRNA = -np.sum((sim_data_mRNA - experimental_data_mRNA) ** 2)

        return ll

    return log_likelihood


def get_log_likelihood_dream(solver,
                             parameter_names: list,
                             observable_names: list,
                             experimental_data: pd.DataFrame,
                             fixed_parameters: dict = None):
    likelihoods = {}
    for observable in observable_names:
        y_meas = experimental_data[[col for col in experimental_data.columns if observable in col]].values
        mean = np.mean(y_meas, axis=1)
        var = np.var(y_meas, axis=1)
        likelihoods[observable] = norm(loc=mean, scale=np.sqrt(var))

    def log_likelihood(parameter_vector):
        parameters = {p_name: p_val for p_name, p_val in zip(parameter_names, parameter_vector)}
        if fixed_parameters:
            parameters.update(fixed_parameters)

        sim_result = solver.run(parameters)

        # ToDo For adequate likelihood treatment see:
        # https://github.com/LoLab-MSM/PyDREAM/blob/master/pydream/examples/robertson/example_sample_robertson_with_dream.py#L38
        ll = []
        for observable in observable_names:
            cur_vals = solver.yobs[observable]
            cur_ll = likelihoods[observable].logpdf(cur_vals)
            ll.append(np.sum(cur_ll))

        return np.sum(ll)

    return log_likelihood


def fit_model_to_data(log_likelihood, priors):
    opt = ParallelTempering()

    n_chains = 20
    minimal_temp = 10 ** (-14)
    var_ref = 1
    n_samples = 10 ** 3
    n_swaps = 2

    sample_history, posterior_history, tempered_posterior_history, map_params = opt.run(log_likelihood=log_likelihood,
                                                                                        priors=priors,
                                                                                        n_chains=n_chains,
                                                                                        minimal_inverse_temp=minimal_temp,
                                                                                        var_ref=var_ref,
                                                                                        n_samples=n_samples,
                                                                                        n_swaps=n_swaps)

    return map_params


def exemplary_model_fitting():
    observable_names = [
        # "obs_Protein_GFP",
        "obs_Protein_RFP",
    ]
    n_steps = 10
    t = np.linspace(0, 10, n_steps)

    n_replicates = 3
    fixed_parameters = {
        "k_mat": 10 ** 10,
    }
    parameters = {"k_tx": 2,
                  "k_tl": 2,
                  # "k_mat": 10 ** 10,
                  "k_rna_deg": 0.5,
                  "k_prot_deg": 0.5,
                  # "k_csy4": 1,
                  # "k_tl_bound_toehold": 0.1,
                  # "k_trigger_binding": 5,
                  # "k_trigger_unbinding": 0.5,
                  # "k_tx_init": 1,
                  # "k_star_bind": 5,
                  # "k_star_unbind": 0.1,
                  # "k_star_act": 2,
                  # "k_star_act_reg": 0.01,
                  # "k_star_stop": 1,
                  # "k_star_stop_reg": 0.01
                  }
    parameters = {id: np.log10(parameters[id]) for id in parameters}
    fixed_parameters = {id: np.log10(fixed_parameters[id]) for id in fixed_parameters}
    priors = {}
    for id in parameters:
        act_val = parameters[id]
        a = act_val - 5
        b = act_val + 5
        priors[id] = scipy.stats.uniform(loc=a, scale=b - a)

    plasmids = [
        # (("Sense1", "Star1"), None, [(True, "GFP")]),
        (None, None, [(True, "RFP")]),
        # (None, None, [(False, "Trigger1")]),
        # (None, None, [(False, "Star1")]),
    ]

    my_model = get_wrapped_model(t=t, plasmids=plasmids, observable_names=observable_names,
                                 fixed_parameters=fixed_parameters)
    # my_model = get_model_dummy(t=t, plasmids=plasmids, observable_names=observable_names, fixed_parameters=fixed_parameters)

    measurements = my_model(parameters)

    fig, ax = plt.subplots()
    T = t
    Y_true = measurements
    ax.plot(T, Y_true, label="Reference")
    ax.plot(T, 16 * (T ** 3 / (4 ** 3 + T ** 3)), "--", label="Approximation")
    ax.legend()
    ax.set_title("Measurement")
    plt.show()

    cols = list(measurements.columns)
    for col in cols:
        for iR in range(n_replicates):
            measurements[col + f" Replicate {iR}"] = measurements[col].map(
                lambda val: np.random.randn() * 0.1 * val + val if val > 0 else val)
    experimental_data = measurements[[col for col in measurements.columns if "Replicate" in col]]

    l_likelihood = get_log_likelihood(wrapped_model=my_model, observable_names=observable_names,
                                      n_replicates=n_replicates,
                                      experimental_data=experimental_data)

    best_fit = fit_model_to_data(l_likelihood, priors=priors)

    fig, ax = plt.subplots()
    T = t
    Y_true = my_model(parameters)
    Y_pred = my_model(best_fit)
    ax.plot(T, Y_true, label="Reference")
    ax.plot(T, Y_pred, "--", label="Prediction")
    ax.legend()
    ax.set_title("Prediction vs. Measurement")
    plt.show()

    parameters = {id: 10 ** (parameters[id]) for id in parameters}
    best_fit = {id: 10 ** (best_fit[id]) for id in best_fit}
    print("Original Params:", parameters)
    print("Best Fit Params:", best_fit)
    pass


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


"""
One can include time dependency into the simulation by separating the intervals of different inducer concentrations and apply multiple simulation runs.
Results need to be joined afterwards
"""

if __name__ == '__main__':
    """
    AND gate
    """

    # exemplary_model_fitting()
    # exit(0)
    # test_star()

    test_AND_gate()

    # test_cleaved_transcription_and_translation()
    test_toehold()

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
