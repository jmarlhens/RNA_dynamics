from pysb import Model, Parameter, Observable
from circuits.modules.star import STAR
from circuits.modules.base_modules import (
    TranscriptionFactory,
    TranscriptionType,
    KineticsType,
    TranslationFactory,
)
from circuits.modules.Csy4_activity import Csy4Activity
from circuits.modules.toehold import Toehold
from circuits.modules.sequestration import Sequestration
from circuits.modules.molecules import RNA


def setup_model(
    plasmids,
    parameters,
    bindings=None,
    use_pulses=False,
    pulse_config=None,
    pulse_indices=None,
    pulse_plasmids=None,
    kinetics_type=KineticsType.MICHAELIS_MENTEN,
):
    """
    Set up the PySB model with the given plasmids, parameters, and optional sequestration reactions.

    :param plasmids: List of plasmids to be processed in format [(name, tx_control, tl_control, cds), ...]
    :param parameters: Dictionary of model parameters.
    :param bindings: Optional list of tuples specifying sequestration reactions between species.
    :param use_pulses: Boolean indicating whether to use pulsed transcription.
    :param pulse_indices: List of indices indicating which plasmids should be pulsed.
    :param pulse_plasmids: List of plasmid names to pulse (new approach).
    :param pulse_config: Dictionary containing pulse configuration if use_pulses is True.
    :param kinetics_type: Type of kinetics to use (Michaelis-Menten or mass action).
    :return: PySB Model object.
    """
    model = Model()

    # Filter out parameters that will be replaced by expressions for pulsed plasmids
    filtered_parameters = parameters.copy()

    if use_pulses:
        # Create mapping from plasmid name to index for easier lookup
        plasmid_name_to_index = {plasmid[0]: i for i, plasmid in enumerate(plasmids)}
        # to change to only take the plasmid name into account

        # Determine which plasmids will be pulsed
        pulsed_indices = set()

        # Add indices from pulse_indices parameter (backward compatibility)
        if pulse_indices is not None:
            pulsed_indices.update(pulse_indices)

        # Add indices from pulse_plasmids parameter (new approach)
        if pulse_plasmids is not None:
            for plasmid_name in pulse_plasmids:
                if plasmid_name in plasmid_name_to_index:
                    pulsed_indices.add(plasmid_name_to_index[plasmid_name])

        # If no indices specified but use_pulses is True, pulse all plasmids
        if not pulsed_indices and use_pulses:
            pulsed_indices = set(range(len(plasmids)))

        # Identify concentration parameters to remove for pulsed plasmids
        parameters_to_remove = []
        for idx in pulsed_indices:
            if idx < len(plasmids):
                plasmid = plasmids[idx]

                # Extract sequence names from the plasmid (new format)
                _, transcriptional_control, translational_control, cds = plasmid

                if len(cds) == 0:
                    continue

                # Construct RNA name as in process_plasmid function
                rna_name_parts = []
                if transcriptional_control:
                    rna_name_parts.append(transcriptional_control[0])
                if translational_control:
                    rna_name_parts.append(translational_control[0])
                rna_name_parts.extend([elem[1] for elem in cds])

                rna_name = "_".join(rna_name_parts)
                parameters_to_remove.append(f"k_{rna_name}_concentration")

        # Remove identified parameters
        for param_name in parameters_to_remove:
            if param_name in filtered_parameters:
                print(
                    f"Removing parameter {param_name} as it will be replaced by a pulsed expression"
                )
                filtered_parameters.pop(param_name)

    # Add filtered parameters to the model
    for param_name, param_value in filtered_parameters.items():
        Parameter(param_name, param_value)

    # Process each plasmid
    for idx, plasmid in enumerate(plasmids):
        # Check if this plasmid should be pulsed
        apply_pulse = False

        if use_pulses:
            # Check against pulse_indices (backward compatibility)
            if pulse_indices is not None and idx in pulse_indices:
                apply_pulse = True

            # Check against pulse_plasmids (new approach)
            if pulse_plasmids is not None:
                plasmid_name = plasmid[0]
                if plasmid_name in pulse_plasmids:
                    apply_pulse = True

            # If neither is specified, pulse all plasmids
            if pulse_indices is None and pulse_plasmids is None:
                apply_pulse = True

        # Extract the actual plasmid components, ignoring the name
        process_plasmid(
            plasmid[1:],  # Skip the name, passing only (tx_control, tl_control, cds)
            model,
            use_pulses=apply_pulse,
            pulse_config=pulse_config,
            kinetics_type=kinetics_type,
        )

    # Process sequestration (binding and unbinding) reactions if specified
    if bindings:
        for species1_name, species2_name in bindings:
            Sequestration(species1_name, species2_name, model)

    # Generate observables for the model
    generate_observables(model)

    return model


def process_plasmid(
    plasmid,
    model,
    use_pulses=False,
    pulse_config=None,
    kinetics_type=KineticsType.MICHAELIS_MENTEN,
):
    """
    Processes the given plasmid by adding it to the model with appropriate controls.

    :param plasmid: Tuple containing transcriptional control, translational control, and CDS list.
    :param model: The PySB model to which components are added.
    :param use_pulses: Boolean indicating whether to use pulsed transcription.
    :param pulse_config: Dictionary containing pulse configuration if use_pulses is True.
    :param kinetics_type: Type of kinetics to use (Michaelis-Menten or mass action).
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
        # Create STAR-regulated transcription with specified kinetics type
        star = STAR(
            sequence_name=rna_name,
            transcriptional_control=transcriptional_control,
            model=model,
            kinetics_type=kinetics_type,
        )
        rna = star.product
        products = [rna]
    else:
        # Choose between regular or pulsed transcription using the factory pattern
        if use_pulses:
            # Use the TranscriptionFactory to create appropriate transcription instance with kinetics_type
            transcription = TranscriptionFactory.create_transcription(
                transcription_type=TranscriptionType.PULSED,
                sequence_name=rna_name,
                model=model,
                pulse_config=pulse_config,
                kinetics_type=kinetics_type,
            )
        else:
            # Use the TranscriptionFactory to create regular transcription with kinetics_type
            transcription = TranscriptionFactory.create_transcription(
                transcription_type=TranscriptionType.CONSTANT,
                sequence_name=rna_name,
                model=model,
                kinetics_type=kinetics_type,
            )
        rna = transcription.product
        products = [rna]

    # Step 2: Determine if Csy4 cleavage is needed
    cleavage_points = []

    if transcriptional_control and len(cds) > 0:
        cleavage_set = [transcriptional_control[0]] + sequence_names
        cleavage_points.append(cleavage_set)

    if len(cds) > 1:
        cleavage_points.append(sequence_names)

    for cleavage_set in cleavage_points:
        csy4 = Csy4Activity(
            rna=rna,
            product_rna_names=cleavage_set,
            model=model,
            kinetics_type=kinetics_type,
        )
        products = csy4.product

    # Step 3: Handle translation of each CDS product with specified kinetics type
    for translate, seq in cds:
        if not translate:
            continue

        # Find the appropriate RNA to translate for this CDS
        # First check if there's an RNA product with a matching name
        translated_rna = None

        # Option 1: Look for direct match in RNA names
        for rna_product in products:
            if seq in rna_product.sequence_name:
                # Use the existing RNA instance instead of creating a new one
                translated_rna = rna_product
                break

        # Option 2: If no match found, use the raw sequence name
        if translated_rna is None:
            translated_rna = RNA.get_instance(sequence_name=seq, model=model)

        # Now translate this RNA
        if translated_rna:
            if translational_control:
                Toehold(
                    rna=translated_rna,
                    translational_control=translational_control,
                    prot_name=seq,
                    model=model,
                    kinetics_type=kinetics_type,
                )
            else:
                TranslationFactory.create_translation(
                    rna=translated_rna,
                    prot_name=seq,
                    model=model,
                    kinetics_type=kinetics_type,
                )


def generate_observables(model):
    """
    Generate observables for the model based on monomers.
    """
    for monomer in model.monomers:
        if "RNA" in monomer.name:
            desired_state = "full"
            obs_name = "obs_" + monomer.name
            # Observable(obs_name, monomer(state=desired_state))
            # to solve
        elif "Protein" in monomer.name:
            desired_state = "mature"
            obs_name = "obs_" + monomer.name
            Observable(obs_name, monomer(state=desired_state))
