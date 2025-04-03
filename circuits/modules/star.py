from pysb import Rule, Model, Expression, Monomer, Observable, Initial
from circuits.modules.molecules import RNA
from circuits.modules.reactioncomplex import ReactionComplex
from circuits.modules.base_modules import KineticsType


class STAR(ReactionComplex):
    def __init__(self, sequence_name: str = None, transcriptional_control: tuple = None,
                 model: Model = None, kinetics_type: KineticsType = KineticsType.MICHAELIS_MENTEN):
        assert sequence_name is not None
        assert transcriptional_control is not None

        # Get the RNA instances
        rna = RNA.get_instance(sequence_name=sequence_name, model=model)
        regulated = rna  # Regulated RNA
        regulator = RNA.get_instance(sequence_name=transcriptional_control[1], model=model)  # Regulator RNA (STAR)

        super().__init__(substrate=None, product=rna, model=model)

        self.sense_name = transcriptional_control[0]
        self.star_name = transcriptional_control[1]
        self.rna_name = rna.sequence_name
        self.regulator_name = self.star_name
        self.kinetics_type = kinetics_type

        # Choose which implementation to use based on kinetics_type
        if kinetics_type == KineticsType.MICHAELIS_MENTEN:
            self.rules = self._setup_michaelis_menten(regulated, regulator, sequence_name, model)
        elif kinetics_type == KineticsType.MASS_ACTION:
            self.rules = self._setup_mass_action(regulated, regulator, sequence_name, model)
        else:
            raise ValueError(f"Unknown kinetics type: {kinetics_type}")

    def _setup_michaelis_menten(self, regulated, regulator, sequence_name, model):
        """
        Set up STAR regulation with the original Michaelis-Menten style kinetics.
        This preserves the original implementation's behavior.
        """
        # Parameters
        self.k_init = self.parameters["k_tx_init"]
        self.k_concentration = self.parameters["k_" + sequence_name + "_concentration"]
        Expression('k_tx_plasmid_' + sequence_name, self.k_concentration * self.k_init)
        self.k_bind = self.parameters["k_star_bind"]
        self.k_unbind = self.parameters["k_star_unbind"]
        self.k_act = self.parameters["k_star_act"]
        self.k_act_reg = self.parameters["k_star_act_reg"]
        self.k_stop = self.parameters["k_star_stop"]
        self.k_stop_reg = self.parameters["k_star_stop_reg"]
        self.k_deg = self.parameters["k_rna_deg"]

        rules = []

        # RNA transcription initiation: RNA starts in unbound state (sense=None)
        transcription_initiation_rule = Rule(
            f'STAR_RNA_transcription_initiation_{regulated.name}',
            None >> regulated(state='init', sense=None, toehold=None, b=None),
            model.expressions['k_tx_plasmid_' + sequence_name]
        )
        rules.append(transcription_initiation_rule)

        # Binding of RNA regulator to the early transcript: Uses the `sense` site
        binding_rule = Rule(
            f'STAR_RNA_regulator_binding_{regulated.name}_{regulator.name}',
            regulated(sense=None) + regulator(state='full', sense=None) >>
            regulated(sense=1) % regulator(state='full', sense=1),
            self.k_bind
        )
        rules.append(binding_rule)

        # Unbinding of full and early RNA transcript and RNA regulator
        unbinding_rule = Rule(
            f'STAR_RNA_regulator_unbinding_full_{regulated.name}_{regulator.name}',
            regulated(sense=1) % regulator(state='full', sense=1) >>
            regulated(sense=None) + regulator(state='full', sense=None),
            self.k_unbind
        )
        rules.append(unbinding_rule)

        # Activation of transcription with the RNA regulator
        full_transcription_with_regulator_rule = Rule(
            f'STAR_RNA_full_transcription_reg_{regulated.name}_{regulator.name}',
            regulated(state='init', sense=1) % regulator(state='full', sense=1) >>
            regulated(state='full', sense=1) % regulator(state='full', sense=1),
            self.k_act_reg
        )
        rules.append(full_transcription_with_regulator_rule)

        # Full transcription without the RNA regulator
        full_transcription_rule = Rule(
            f'STAR_RNA_full_transcription_{regulated.name}',
            regulated(state='init', sense=None) >>
            regulated(state='full', sense=None),
            self.k_act
        )
        rules.append(full_transcription_rule)

        # Stop of transcription with the STAR RNA regulator
        partial_transcription_with_regulator_rule = Rule(
            f'STAR_RNA_partial_transcription_reg_{regulated.name}_{regulator.name}',
            regulated(state='init', sense=1) % regulator(state='full', sense=1) >>
            regulated(state='partial', sense=1) % regulator(state='full', sense=1),
            self.k_stop_reg
        )
        rules.append(partial_transcription_with_regulator_rule)

        # Partial transcription without the RNA regulator
        partial_transcription_rule = Rule(
            f'STAR_RNA_partial_transcription_{regulated.name}',
            regulated(state='init', sense=None) >>
            regulated(state='partial', sense=None),
            self.k_stop
        )
        rules.append(partial_transcription_rule)

        return rules

    def _setup_mass_action(self, regulated, regulator, sequence_name, model):
        """
        Set up STAR regulation with mass action kinetics.
        This implements a more mechanistic model with explicit RNA polymerase.
        """
        # Add RNA polymerase monomer if it doesn't exist
        polymerase_name = 'RNAPolymerase'
        polymerase = next((m for m in model.monomers if m.name == polymerase_name), None)
        if polymerase is None:
            Monomer(polymerase_name, ['b'])
            Initial(model.monomers[polymerase_name](b=None), model.parameters['rnap_0'])

        # Add DNA monomer for this sequence if it doesn't exist
        dna_name = 'DNA_' + sequence_name
        dna = next((m for m in model.monomers if m.name == dna_name), None)
        if dna is None:
            Monomer(dna_name, ['b', 'sense'])
            self.k_concentration = self.parameters[f"k_{sequence_name}_concentration"]
            Initial(model.monomers[dna_name](b=None, sense=None), self.k_concentration)

        # Parameters for mass action kinetics
        self.k_init = self.parameters["k_tx_init"]
        self.k_bind_rnap = self.parameters["k_tx_bind"]
        self.k_unbind_rnap = self.parameters.get("k_tx_unbind", 0.1)  # Default if not provided
        self.k_cat_rnap = self.parameters["k_tx_cat"]

        # STAR binding parameters (keep from original implementation)
        self.k_star_bind = self.parameters["k_star_bind"]
        self.k_star_unbind = self.parameters["k_star_unbind"]

        rules = []

        # Rule 1: Polymerase binding to DNA (without STAR)
        rule1 = Rule(
            f'STAR_polymerase_binding_without_regulator_{sequence_name}',
            model.monomers[polymerase_name](b=None) +
            model.monomers[dna_name](b=None, sense=None) >>
            model.monomers[polymerase_name](b=1) %
            model.monomers[dna_name](b=1, sense=None),
            self.k_bind_rnap
        )
        rules.append(rule1)

        # Rule 2: Transcription initiation (RNA production from complex without STAR)
        rule2 = Rule(
            f'STAR_transcription_initiation_without_regulator_{regulated.name}',
            model.monomers[polymerase_name](b=1) %
            model.monomers[dna_name](b=1, sense=None) >>
            model.monomers[polymerase_name](b=None) +
            model.monomers[dna_name](b=None, sense=None) +
            regulated(state="init", sense=None, toehold=None, b=None),
            self.k_cat_rnap
        )
        rules.append(rule2)

        # Rule 3: STAR binding to DNA 
        rule3 = Rule(
            f'STAR_binding_to_dna_{regulator.name}',
            regulator(state='full', sense=None) +
            model.monomers[dna_name](sense=None) >>
            regulator(state='full', sense=1) %
            model.monomers[dna_name](sense=1),
            self.k_star_bind
        )
        rules.append(rule3)

        # Rule 4: STAR unbinding from DNA
        rule4 = Rule(
            f'STAR_unbinding_from_dna_{regulator.name}',
            regulator(state='full', sense=1) %
            model.monomers[dna_name](sense=1) >>
            regulator(state='full', sense=None) +
            model.monomers[dna_name](sense=None),
            self.k_star_unbind
        )
        rules.append(rule4)

        # Rule 5: Polymerase binding to DNA with STAR bound
        rule5 = Rule(
            f'STAR_polymerase_binding_with_regulator_{sequence_name}',
            model.monomers[polymerase_name](b=None) +
            (regulator(state='full', sense=1) % model.monomers[dna_name](b=None, sense=1)) >>
            model.monomers[polymerase_name](b=2) %
            (regulator(state='full', sense=1) % model.monomers[dna_name](b=2, sense=1)),
            self.k_bind_rnap  # Assume STAR enhances binding
        )
        rules.append(rule5)

        # Rule 6: Enhanced transcription with STAR and polymerase bound
        rule6 = Rule(
            f'STAR_enhanced_transcription_{regulated.name}',
            model.monomers[polymerase_name](b=2) %
            (regulator(state='full', sense=1) % model.monomers[dna_name](b=2, sense=1)) >>
            model.monomers[polymerase_name](b=None) +
            (regulator(state='full', sense=1) % model.monomers[dna_name](b=None, sense=1)) +
            regulated(state="init", sense=None, toehold=None, b=None),
            self.k_cat_rnap
        )
        rules.append(rule6)

        # Rule 7: Conversion from init to full (elongation)
        rule7 = Rule(
            f'STAR_elongation_to_full_{regulated.name}',
            regulated(state='init', sense=None) >>
            regulated(state='full', sense=None),
            self.parameters["k_star_act"]
        )
        rules.append(rule7)

        # Rule 8: Conversion from init to partial (termination)
        rule8 = Rule(
            f'STAR_termination_to_partial_{regulated.name}',
            regulated(state='init', sense=None) >>
            regulated(state='partial', sense=None),
            self.parameters["k_star_stop"]
        )
        rules.append(rule8)

        # Create observables
        Observable(f'obs_STAR_{dna_name}_complex',
                   regulator(state='full', sense=1) % model.monomers[dna_name](sense=1))
        Observable(f'obs_RNAP_{dna_name}_complex',
                   model.monomers[polymerase_name](b=1) % model.monomers[dna_name](b=1))
        Observable(f'obs_RNAP_STAR_{dna_name}_complex',
                   model.monomers[polymerase_name](b=2) %
                   (regulator(state='full', sense=1) % model.monomers[dna_name](b=2, sense=1)))

        return rules
