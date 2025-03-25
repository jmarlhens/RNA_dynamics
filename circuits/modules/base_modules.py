from modules.molecules import RNA, Protein
from pysb import Rule, Model, Expression, Observable, Monomer, Parameter, Initial
from sympy import Piecewise
import sympy as sp
from modules.reactioncomplex import ReactionComplex
from enum import Enum
from typing import Optional, Dict, Union


class Transcription(ReactionComplex):
    def __init__(self, sequence_name: str = None, model: Model = None):
        rna = RNA.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=None, product=rna, model=model)

        self.k_tx = self.parameters["k_tx"]
        self.K_tx = self.parameters["K_tx"]
        self.k_concentration = self.parameters["k_" + sequence_name + "_concentration"]
        Expression('k_tx_plasmid_' + sequence_name, (self.k_concentration * self.k_tx)/(self.K_tx + self.k_concentration))
        self.k_deg = self.parameters["k_rna_deg"]

        rules = []
        # Transcription rule: RNA is produced in the unbound state
        rule = Rule(
            f'transcription_{rna.name}',
            None >> rna(state="full", sense=None, toehold=None),
            model.expressions['k_tx_plasmid_' + sequence_name]
        )
        rules.append(rule)

        self.rules = rules


class PulsedTranscription(ReactionComplex):
    def __init__(self, sequence_name: str = None, model: Model = None, pulse_config: dict = None):
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
        if pulse_config and pulse_config.get('use_pulse', False):
            # Check if Time monomer exists using proper PySB method
            time_monomer = next((m for m in model.monomers if m.name == 'Time'), None)

            if time_monomer is None:
                # Add time monomer and parameters if they don't exist
                Monomer('Time')
                Parameter('Time_0', 0)
                Parameter('k_clock', 1)
                Initial(model.monomers['Time'], model.parameters['Time_0'])
                Rule('Clock', None >> model.monomers['Time'](), model.parameters['k_clock'])

            Observable('obs_Time', model.monomers['Time'])

            # Create pulsed concentration expression
            Expression(
                'k_' + sequence_name + '_concentration',
                Piecewise(
                    (pulse_config['base_concentration'],
                     sp.Lt(model.observables['obs_Time'], pulse_config['pulse_start'])),
                    (pulse_config['base_concentration'],
                     sp.Gt(model.observables['obs_Time'], pulse_config['pulse_end'])),
                    (pulse_config['pulse_concentration'], True)
                )
            )
        else:
            # Use constant concentration as in original implementation
            self.k_concentration = self.parameters["k_" + sequence_name + "_concentration"]

        # Create transcription rate expression
        Expression(
            'k_tx_plasmid_' + sequence_name,
            (model.expressions['k_' + sequence_name + '_concentration'] * self.k_tx) /
            (self.K_tx + model.expressions['k_' + sequence_name + '_concentration'])
        )

        # Create transcription rule
        rules = []
        rule = Rule(
            f'transcription_{rna.name}',
            None >> rna(state="full", sense=None, toehold=None),
            model.expressions['k_tx_plasmid_' + sequence_name]
        )
        rules.append(rule)
        self.rules = rules

class Translation(ReactionComplex):
    def __init__(self, rna: RNA = None, prot_name: str = None, model: Model = None):
        sequence_name = rna.sequence_name if prot_name is None else prot_name
        protein = Protein.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=rna, product=protein, model=model)

        self.k_tl = self.parameters["k_tl"]
        self.K_tl = self.parameters["K_tl"]
        self.k_mat = self.parameters["k_mat"]
        self.k_deg = self.parameters["k_prot_deg"]
        Observable('obs_RNA_' + sequence_name, model.monomers['RNA_' + sequence_name](state='full'))
        Expression('k_tl_eff_' + sequence_name, self.k_tl / (self.K_tl + model.observables['obs_RNA_' + sequence_name]))

        rules = []

        # Translation rule: Translation occurs only if RNA is not bound at any site
        rule = Rule(
            f'translation_of_{rna.name}_to_{protein.name}',
            rna(state="full") >>
            rna(state="full") + protein(state="immature"),
            model.expressions['k_tl_eff_' + sequence_name]
        )
        rules.append(rule)

        self.rules = rules


class TranscriptionType(Enum):
    CONSTANT = "constant"
    PULSED = "pulsed"


class TranscriptionFactory:
    @staticmethod
    def create_transcription(
            transcription_type: TranscriptionType,
            sequence_name: str,
            model: Model,
            pulse_config: Optional[Dict] = None
    ) -> Union[Transcription, PulsedTranscription]:
        """
        Factory method to create appropriate transcription instance.

        Args:
            transcription_type: Type of transcription (constant or pulsed)
            sequence_name: Name of the sequence being transcribed
            model: PySB model object
            pulse_config: Configuration for pulse behavior (required if type is PULSED)

        Returns:
            Instance of either Transcription or PulsedTranscription
        """
        if transcription_type == TranscriptionType.CONSTANT:
            return Transcription(sequence_name=sequence_name, model=model)
        elif transcription_type == TranscriptionType.PULSED:
            if pulse_config is None:
                raise ValueError("pulse_config is required for pulsed transcription")
            return PulsedTranscription(
                sequence_name=sequence_name,
                model=model,
                pulse_config=pulse_config
            )
        else:
            raise ValueError(f"Unknown transcription type: {transcription_type}")
