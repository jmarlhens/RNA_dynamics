import numpy as np
import matplotlib.pyplot as plt
from pysb import Model, Parameter, Observable
from pysb.simulator import ScipyOdeSimulator
from modules.star import STAR
from modules.base_modules import Transcription, Translation
from modules.molecules import RNA
from modules.Csy4_activity import Csy4Activity
from modules.toehold import Toehold
from modules.sequestration import Sequestration


def setup_model(plasmids, parameters, bindings=None):
    """
    Set up the PySB model with the given plasmids, parameters, and optional sequestration reactions.

    :param plasmids: List of plasmids to be processed.
    :param parameters: Dictionary of model parameters.
    :param bindings: Optional list of tuples specifying sequestration reactions between species.
    :return: PySB Model object.
    """
    model = Model()

    # Add parameters to the model
    for param_name, param_value in parameters.items():
        Parameter(param_name, param_value)

    # Process each plasmid
    for plasmid in plasmids:
        process_plasmid(plasmid, model)

    # Generate observables for the model
    generate_observables(model)

    # Process sequestration (binding and unbinding) reactions if specified
    if bindings:
        for species1_name, species2_name in bindings:
            # Retrieve or create instances of the species (RNA or Protein based on prefix)
            species1 = RNA.get_instance(sequence_name=species1_name, model=model)
            species2 = RNA.get_instance(sequence_name=species2_name, model=model)
            # Create the sequestration reaction
            Sequestration(species1, species2, model)

    return model


def process_plasmid(plasmid, model):
    """
    Processes the given plasmid by adding it to the model with appropriate controls.

    :param plasmid: Tuple containing transcriptional control, translational control, and CDS list.
    :param model: The PySB model to which components are added.
    """
    transcriptional_control = plasmid[0]
    translational_control = plasmid[1]
    cds = plasmid[2]

    if len(cds) == 0:
        return

    cleave_rna = len(cds) > 1
    sequence_names = [elem[1] for elem in cds]

    # Construct the RNA name by only including non-None control elements
    rna_name_parts = []
    if transcriptional_control:
        rna_name_parts.append(transcriptional_control[0])
    if translational_control:
        rna_name_parts.append(translational_control[0])
    rna_name_parts.extend(sequence_names)

    # Join the parts with an underscore, but only if there are parts to join
    rna_name = "_".join(rna_name_parts)

    if transcriptional_control:
        star = STAR(sequence_name=rna_name, transcriptional_control=transcriptional_control, model=model)
        rna = star.product
        products = [rna]
    else:
        transcription = Transcription(sequence_name=rna_name, model=model)
        rna = transcription.product
        products = [rna]

    if cleave_rna:
        csy4 = Csy4Activity(rna=rna, product_rna_names=sequence_names, model=model)
        products = csy4.product

    for (translate, seq), rna in zip(cds, products):
        if not translate:
            continue
        if translational_control:
            Toehold(rna=rna, translational_control=translational_control, prot_name=seq, model=model)
        else:
            Translation(rna=rna, prot_name=seq, model=model)



def generate_observables(model):
    """
    Generate observables for the model based on monomers.
    """
    for monomer in model.monomers:
        desired_state = "full" if "RNA" in monomer.name else "mature"
        obs_name = "obs_" + monomer.name
        Observable(obs_name, monomer(state=desired_state))


def simulate_model(model, t):
    """
    Simulates the model using the specified time points.
    """
    simulator = ScipyOdeSimulator(model)
    y_res = simulator.run(tspan=t)
    return y_res


def visualize_simulation(t, y_res, species_to_plot):
    """
    Visualizes the simulation results.
    """
    fig, ax = plt.subplots()
    species = y_res.dataframe.columns

    for spec in species:
        if spec in species_to_plot:
            ax.plot(t, y_res.dataframe[spec], label=spec.replace("obs_", ""))

    ax.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Simulation Results')
    plt.show()
