from circuits.modules.molecules import RNA, Protein
from pysb import Rule, Model, Expression, Observable, Monomer, Parameter, Initial
from sympy import Piecewise
import sympy as sp
from circuits.modules.reactioncomplex import ReactionComplex
from enum import Enum
from typing import Optional, Dict, Union


class KineticsType(Enum):
    """Enum representing the type of kinetics to use in the model."""

    MICHAELIS_MENTEN = "michaelis_menten"
    MASS_ACTION = "mass_action"


class TranscriptionType(Enum):
    CONSTANT = "constant"
    PULSED = "pulsed"


class Transcription(ReactionComplex):
    def __init__(
        self,
        sequence_name: str = None,
        model: Model = None,
        kinetic_parameters: dict = None,
    ):
        """
        Initialize a Transcription reaction complex.

        :param sequence_name:
        :param model:
        :param kinetic_parameters:
        """
        rna = RNA.get_instance(
            sequence_name=sequence_name,
            model=model,
            kinetic_parameters=kinetic_parameters,
        )

        super().__init__(substrate=None, product=rna, model=model)

        trancription_parameters = ["k_tx", "K_tx", "k_rna_deg"]
        existing_parameters = set(model.parameters.keys())
        for param_name in trancription_parameters:
            if param_name not in existing_parameters:
                Parameter(param_name, kinetic_parameters[param_name])

        self.k_concentration = model.parameters["k_" + sequence_name + "_concentration"]
        Expression(
            "k_tx_plasmid_" + sequence_name,
            (
                model.parameters["k_" + sequence_name + "_concentration"]
                * model.parameters["k_tx"]
            )
            / (
                model.parameters["K_tx"]
                + model.parameters["k_" + sequence_name + "_concentration"]
            ),
        )

        rules = []
        # Transcription rule: RNA is produced in the unbound state
        rule = Rule(
            f"transcription_{rna.name}",
            None >> rna(state="full", sense=None, toehold=None, b=None),
            model.expressions["k_tx_plasmid_" + sequence_name],
        )
        rules.append(rule)

        self.rules = rules


class PulsedTranscription(ReactionComplex):
    def __init__(
        self, sequence_name: str = None, model: Model = None, pulse_config: dict = None
    ):
        """
        Enhanced Transcription class that supports both constant and pulsing plasmid concentrations.

        Args:
            sequence_name (str): Name of the sequence being transcribed
            model (Model): PySB model object
            pulse_config (dict, optional): Configuration for pulse behavior with keys:
                - use_pulse (bool): Whether to use pulsing behavior
                - pulse_start (float): Time when pulse starts
                - pulse_end (float): Time when pulse ends
                - pulse_concentration (float): Concentration during pulse
                - base_concentration (float): Concentration outside pulse
        """
        rna = RNA.get_instance(sequence_name=sequence_name, model=model)
        super().__init__(substrate=None, product=rna, model=model)

        # Set up basic parameters
        self.k_tx = self.parameters["k_tx"]
        self.K_tx = self.parameters["K_tx"]
        self.k_deg = self.parameters["k_rna_deg"]

        # Set up time tracking if using pulses
        if pulse_config and pulse_config.get("use_pulse", False):
            # Check if Time monomer exists using proper PySB method
            time_monomer = next((m for m in model.monomers if m.name == "Time"), None)

            if time_monomer is None:
                # Add time monomer and parameters if they don't exist
                Monomer("Time")
                Parameter("Time_0", 0)
                Parameter("k_clock", 1)
                Initial(model.monomers["Time"], model.parameters["Time_0"])
                Rule(
                    "Clock",
                    None >> model.monomers["Time"](),
                    model.parameters["k_clock"],
                )

            Observable("obs_Time", model.monomers["Time"])

            # Create pulsed concentration expression
            Expression(
                "k_" + sequence_name + "_concentration",
                Piecewise(
                    (
                        pulse_config["base_concentration"],
                        sp.Lt(
                            model.observables["obs_Time"], pulse_config["pulse_start"]
                        ),
                    ),
                    (
                        pulse_config["base_concentration"],
                        sp.Gt(model.observables["obs_Time"], pulse_config["pulse_end"]),
                    ),
                    (pulse_config["pulse_concentration"], True),
                ),
            )
        else:
            # Use constant concentration as in original implementation
            self.k_concentration = self.parameters[
                "k_" + sequence_name + "_concentration"
            ]

        # Create transcription rate expression
        Expression(
            "k_tx_plasmid_" + sequence_name,
            (model.expressions["k_" + sequence_name + "_concentration"] * self.k_tx)
            / (self.K_tx + model.expressions["k_" + sequence_name + "_concentration"]),
        )

        # Create transcription rule
        rules = []
        rule = Rule(
            f"transcription_{rna.name}",
            None >> rna(state="full", sense=None, toehold=None, b=None),
            model.expressions["k_tx_plasmid_" + sequence_name],
        )
        rules.append(rule)
        self.rules = rules


class Translation(ReactionComplex):
    def __init__(
        self,
        rna: RNA = None,
        prot_name: str = None,
        model: Model = None,
        kinetic_parameters: dict = None,
    ):
        """
        Initialize a Translation reaction complex.
        Args:
            rna (RNA): RNA object to be translated
            prot_name (str): Name of the protein to be produced
            model (Model): PySB model object
            kinetic_parameters (dict): Kinetic parameters for translation
        """

        sequence_name = rna.sequence_name if prot_name is None else prot_name
        protein = Protein.get_instance(
            sequence_name=sequence_name,
            model=model,
            kinetic_parameters=kinetic_parameters,
        )

        super().__init__(substrate=rna, product=protein, model=model)

        translation_parameters = ["k_tl", "K_tl", "k_mat", "k_prot_deg"]
        existing_parameters = set(model.parameters.keys())

        for param_name in translation_parameters:
            if param_name not in existing_parameters:
                Parameter(param_name, kinetic_parameters[param_name])
        # self.k_tl = self.parameters["k_tl"]
        # self.K_tl = self.parameters["K_tl"]
        # self.k_mat = self.parameters["k_mat"]
        # self.k_deg = self.parameters["k_prot_deg"]
        Observable(
            "obs_RNA_" + sequence_name,
            model.monomers["RNA_" + sequence_name](state="full"),
        )
        Expression(
            "k_tl_eff_" + sequence_name,
            model.parameters["k_tl"]
            / (
                model.parameters["K_tl"] + model.observables["obs_RNA_" + sequence_name]
            ),
        )

        rules = []

        # Translation rule: Translation occurs only if RNA is not bound at any site
        rule = Rule(
            f"translation_of_{rna.name}_to_{protein.name}",
            rna(state="full") >> rna(state="full") + protein(state="immature"),
            model.expressions["k_tl_eff_" + sequence_name],
        )
        rules.append(rule)

        self.rules = rules


class MassActionTranscription(ReactionComplex):
    def __init__(self, sequence_name: str = None, model: Model = None):
        rna = RNA.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=None, product=rna, model=model)

        polymerase_name = "RNAPolymerase"
        polymerase = next(
            (m for m in model.monomers if m.name == polymerase_name), None
        )
        if polymerase is None:
            Monomer(polymerase_name, ["b"])
            Initial(model.monomers[polymerase_name](b=None), model.parameters["rnap_0"])

        dna_name = "DNA_" + sequence_name
        dna = next((m for m in model.monomers if m.name == dna_name), None)
        if dna is None:
            Monomer(dna_name, ["b"])
            self.k_concentration = self.parameters[f"k_{sequence_name}_concentration"]
            Initial(model.monomers[dna_name](b=None), self.k_concentration)

        # Parameters for mass action kinetics
        self.k_bind = self.parameters["k_tx_bind"]  # Binding rate
        self.k_unbind = self.parameters["k_tx_unbind"]  # Unbinding rate
        self.k_cat = self.parameters["k_tx_cat"]  # Catalytic rate
        self.k_deg = self.parameters["k_rna_deg"]  # Degradation rate

        rules = []

        # Rule 1: Polymerase binding to DNA
        rule1 = Rule(
            f"polymerase_binding_{sequence_name}",
            model.monomers[polymerase_name](b=None) + model.monomers[dna_name](b=None)
            >> model.monomers[polymerase_name](b=1) % model.monomers[dna_name](b=1),
            self.k_bind,
        )
        rules.append(rule1)

        # Rule 2: Transcription (RNA production from complex)
        rule2 = Rule(
            f"transcription_{rna.name}",
            model.monomers[polymerase_name](b=1) % model.monomers[dna_name](b=1)
            >> model.monomers[polymerase_name](b=None)
            + model.monomers[dna_name](b=None)
            + rna(state="full", sense=None, toehold=None, b=None),
            self.k_cat,
        )
        rules.append(rule2)

        # Create observable for the DNA-polymerase complex
        Observable(
            f"obs_RNAP_{dna_name}_complex",
            model.monomers[polymerase_name](b=1) % model.monomers[dna_name](b=1),
        )

        self.rules = rules


# class MassActionPulsedTranscription(ReactionComplex):
#     def __init__(self, sequence_name: str = None, model: Model = None, pulse_config: dict = None):
#         rna = RNA.get_instance(sequence_name=sequence_name, model=model)
#
#         super().__init__(substrate=None, product=rna, model=model)
#
#         polymerase_name = 'RNAPolymerase'
#         polymerase = next((m for m in model.monomers if m.name == polymerase_name), None)
#         if polymerase is None:
#             Monomer(polymerase_name, ['b'])
#             Initial(model.monomers[polymerase_name](b=None), model.parameters['rnap_0'])
#
#         dna_name = 'DNA_' + sequence_name
#         dna = next((m for m in model.monomers if m.name == dna_name), None)
#         if dna is None:
#             Monomer(dna_name, ['b'])
#
#         # Set up time tracking for pulses
#         if pulse_config and pulse_config.get('use_pulse', False):
#             time_monomer = next((m for m in model.monomers if m.name == 'Time'), None)
#
#             if time_monomer is None:
#                 # Add time monomer and parameters
#                 Monomer('Time')
#                 Parameter('Time_0', 0)
#                 Parameter('k_clock', 1)
#                 Initial(model.monomers['Time'], model.parameters['Time_0'])
#                 Rule('Clock', None >> model.monomers['Time'](), model.parameters['k_clock'])
#
#             Observable('obs_Time', model.monomers['Time'])
#
#             # Create pulsed concentration expression
#             Expression(
#                 'k_' + sequence_name + '_concentration',
#                 Piecewise(
#                     (pulse_config['base_concentration'],
#                      sp.Lt(model.observables['obs_Time'], pulse_config['pulse_start'])),
#                     (pulse_config['base_concentration'],
#                      sp.Gt(model.observables['obs_Time'], pulse_config['pulse_end'])),
#                     (pulse_config['pulse_concentration'], True)
#                 )
#             )
#
#             # Set initial concentration based on expression
#             Initial(model.monomers[dna_name](b=None), model.expressions['k_' + sequence_name + '_concentration'])
#         else:
#             # Use constant concentration
#             self.k_concentration = self.parameters[f"k_{sequence_name}_concentration"]
#             Initial(model.monomers[dna_name](b=None), self.k_concentration)
#
#         # Parameters for mass action kinetics
#         self.k_bind = self.parameters["k_tx_bind"]  # Binding rate
#         self.k_unbind = self.parameters["k_tx_unbind"]  # Unbinding rate
#         self.k_cat = self.parameters["k_tx_cat"]  # Catalytic rate
#         self.k_deg = self.parameters["k_rna_deg"]  # Degradation rate
#
#         rules = []
#
#         # Rule 1: Polymerase binding to DNA
#         rule1 = Rule(
#             f'polymerase_binding_{sequence_name}',
#             model.monomers[polymerase](b=None) + model.monomers[dna_name](b=None) >>
#             model.monomers[polymerase](b=1) % model.monomers[dna_name](b=1),
#             self.k_bind
#         )
#         rules.append(rule1)
#
#         # Rule 2: Transcription (RNA production from complex)
#         rule2 = Rule(
#             f'transcription_{rna.name}',
#             model.monomers[polymerase](b=1) % model.monomers[dna_name](b=1) >>
#             model.monomers[polymerase](b=None) + model.monomers[dna_name](b=None) + rna(state="full", sense=None,
#                                                                                         toehold=None, b=None),
#             self.k_cat
#         )
#         rules.append(rule2)
#
#         # Create observable for the DNA-polymerase complex
#         Observable(f'obs_RNAP_{dna_name}_complex',
#                    model.monomers[polymerase](b=1) % model.monomers[dna_name](b=1))
#
#         self.rules = rules


class MassActionTranslation(ReactionComplex):
    def __init__(
        self,
        rna: RNA = None,
        prot_name: str = None,
        model: Model = None,
        kinetic_parameters: dict = None,
    ):
        """

        :param rna:
        :param prot_name:
        :param model:
        :param kinetic_parameters:
        """
        sequence_name = rna.sequence_name if prot_name is None else prot_name
        protein = Protein.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=rna, product=protein, model=model)

        # Add ribosome monomer if it doesn't exist
        ribosome_name = "Ribosome"
        ribosome = next((m for m in model.monomers if m.name == ribosome_name), None)
        if ribosome is None:
            Monomer(ribosome_name, ["b"])
            Initial(
                model.monomers[ribosome_name](b=None), model.parameters["ribosome_0"]
            )

        # Parameters for mass action kinetics
        self.k_bind = self.parameters["k_tl_bind"]  # Binding rate
        self.k_unbind = self.parameters["k_tl_unbind"]  # Unbinding rate
        self.k_cat = self.parameters["k_tl_cat"]  # Catalytic rate
        self.k_mat = self.parameters["k_mat"]  # Protein maturation rate
        self.k_deg = self.parameters["k_prot_deg"]  # Protein degradation rate
        Observable(
            "obs_RNA_" + sequence_name,
            model.monomers["RNA_" + sequence_name](state="full"),
        )

        rules = []

        # Rule 1: Ribosome binding to RNA
        rule1 = Rule(
            f"ribosome_binding_{rna.name}",
            model.monomers[ribosome_name](b=None) + rna(state="full", b=None)
            >> model.monomers[ribosome_name](b=1) % rna(state="full", b=1),
            self.k_bind,
        )
        rules.append(rule1)

        # Rule 2: Translation (protein production from complex)
        rule2 = Rule(
            f"translation_of_{rna.name}_to_{protein.name}",
            model.monomers[ribosome_name](b=1) % rna(state="full", b=1)
            >> model.monomers[ribosome_name](b=None)
            + rna(state="full", b=None)
            + protein(state="immature"),
            self.k_cat,
        )
        rules.append(rule2)

        # Create observable for the RNA-ribosome complex
        Observable(
            f"obs_Ribosome_{rna.name}_complex",
            model.monomers[ribosome_name](b=1) % rna(state="full", b=1),
        )

        self.rules = rules


class TranscriptionFactory:
    @staticmethod
    def create_transcription(
        transcription_type: TranscriptionType,
        sequence_name: str,
        model: Model,
        kinetic_parameters: Dict,
        pulse_config: Optional[Dict] = None,
        kinetics_type: KineticsType = KineticsType.MICHAELIS_MENTEN,
    ) -> Union[Transcription, PulsedTranscription, MassActionTranscription]:
        """
        Factory method to create appropriate transcription instance.

        Args:
            transcription_type: Type of transcription (constant or pulsed)
            sequence_name: Name of the sequence being transcribed
            model: PySB model object
            kinetic_parameters: Kinetic parameters for transcription
            pulse_config: Configuration for pulse behavior (required if type is PULSED)
            kinetics_type: Type of kinetics to use (MICHAELIS_MENTEN or MASS_ACTION)

        Returns:
            Instance of transcription class
        """
        if kinetics_type == KineticsType.MICHAELIS_MENTEN:
            if transcription_type == TranscriptionType.CONSTANT:
                return Transcription(
                    sequence_name=sequence_name,
                    model=model,
                    kinetic_parameters=kinetic_parameters,
                )
            elif transcription_type == TranscriptionType.PULSED:
                if pulse_config is None:
                    raise ValueError(
                        "pulse_config is required for pulsed transcription"
                    )
                return PulsedTranscription(
                    sequence_name=sequence_name, model=model, pulse_config=pulse_config
                )
        elif kinetics_type == KineticsType.MASS_ACTION:
            if transcription_type == TranscriptionType.CONSTANT:
                return MassActionTranscription(sequence_name=sequence_name, model=model)
            # elif transcription_type == TranscriptionType.PULSED:
            #     if pulse_config is None:
            #         raise ValueError("pulse_config is required for pulsed transcription")
            #     return MassActionPulsedTranscription(
            #         sequence_name=sequence_name,
            #         model=model,
            #         pulse_config=pulse_config
            #     )

        raise ValueError(
            f"Unsupported combination: {transcription_type}, {kinetics_type}"
        )


class TranslationFactory:
    @staticmethod
    def create_translation(
        rna: RNA,
        prot_name: str,
        model: Model,
        kinetic_parameters: Dict,
        kinetics_type: KineticsType = KineticsType.MICHAELIS_MENTEN,
    ) -> Union[Translation, MassActionTranslation]:
        """
        Factory method to create appropriate translation instance.

        Args:
            rna: RNA object
            prot_name: Name of the protein
            model: PySB model object
            kinetics_type: Type of kinetics to use

        Returns:
            Instance of translation class
        """
        if kinetics_type == KineticsType.MICHAELIS_MENTEN:
            return Translation(
                rna=rna,
                prot_name=prot_name,
                model=model,
                kinetic_parameters=kinetic_parameters,
            )
        elif kinetics_type == KineticsType.MASS_ACTION:
            return MassActionTranslation(
                rna=rna,
                prot_name=prot_name,
                model=model,
                kinetic_parameters=kinetic_parameters,
            )
        else:
            raise ValueError(f"Unknown kinetics type: {kinetics_type}")
