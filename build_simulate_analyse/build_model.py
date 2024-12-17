import matplotlib.pyplot as plt
from pysb import Model, Parameter, Observable
from pysb.simulator import ScipyOdeSimulator
from modules.star import STAR
from modules.base_modules import Transcription, Translation, PulsedTranscription
from modules.Csy4_activity import Csy4Activity
from modules.toehold import Toehold
from modules.sequestration import Sequestration


def setup_model(plasmids, parameters, bindings=None, use_pulses=False, pulse_config=None):
    """
    Set up the PySB model with the given plasmids, parameters, and optional sequestration reactions.

    :param plasmids: List of plasmids to be processed.
    :param parameters: Dictionary of model parameters.
    :param bindings: Optional list of tuples specifying sequestration reactions between species.
    :param use_pulses: Boolean indicating whether to use pulsed transcription.
    :param pulse_config: Dictionary containing pulse configuration if use_pulses is True.
    :return: PySB Model object.
    """
    model = Model()

    # Add parameters to the model
    for param_name, param_value in parameters.items():
        Parameter(param_name, param_value)

    # Process each plasmid
    for plasmid in plasmids:
        process_plasmid(plasmid, model, use_pulses, pulse_config)

    # Process sequestration (binding and unbinding) reactions if specified
    if bindings:
        for species1_name, species2_name in bindings:
            Sequestration(species1_name, species2_name, model)

    # Generate observables for the model
    generate_observables(model)

    return model


def process_plasmid(plasmid, model, use_pulses=False, pulse_config=None):
    """
    Processes the given plasmid by adding it to the model with appropriate controls.

    :param plasmid: Tuple containing transcriptional control, translational control, and CDS list.
    :param model: The PySB model to which components are added.
    :param use_pulses: Boolean indicating whether to use pulsed transcription.
    :param pulse_config: Dictionary containing pulse configuration if use_pulses is True.
    """
    transcriptional_control = plasmid[0]
    translational_control = plasmid[1]
    cds = plasmid[2]

    if len(cds) == 0:
        return

    sequence_names = [elem[1] for elem in cds]

    # Construct the RNA name by including transcriptional and translational controls
    rna_name_parts = []
    if transcriptional_control:
        rna_name_parts.append(transcriptional_control[0])
    if translational_control:
        rna_name_parts.append(translational_control[0])
    rna_name_parts.extend(sequence_names)

    rna_name = "_".join(rna_name_parts)

    # Step 1: Handle transcription with or without STAR regulation
    if transcriptional_control:
        # Create STAR-regulated transcription
        star = STAR(sequence_name=rna_name, transcriptional_control=transcriptional_control, model=model)
        rna = star.product
        products = [rna]
    else:
        # Choose between regular or pulsed transcription
        if use_pulses:
            transcription = PulsedTranscription(sequence_name=rna_name, model=model, pulse_config=pulse_config)
        else:
            transcription = Transcription(sequence_name=rna_name, model=model)
        rna = transcription.product
        products = [rna]

    # Rest of the function remains the same...
    # Step 2: Determine if Csy4 cleavage is needed
    cleavage_points = []

    if transcriptional_control and len(cds) > 0:
        cleavage_set = [transcriptional_control[0]] + sequence_names
        cleavage_points.append(cleavage_set)

    if len(cds) > 1:
        cleavage_points.append(sequence_names)

    for cleavage_set in cleavage_points:
        csy4 = Csy4Activity(rna=rna, product_rna_names=cleavage_set, model=model)
        products = csy4.product

    # Step 3: Handle translation of each CDS product
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
        if "RNA" in monomer.name:
            desired_state = "full"
            obs_name = "obs_" + monomer.name
            # Observable(obs_name, monomer(state=desired_state))
        elif "Protein" in monomer.name:
            desired_state = "mature"
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
    Visualizes the build_simulate_analyse results.
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
