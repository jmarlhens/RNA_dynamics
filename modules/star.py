from pysb import Rule, Model, Expression
from modules.molecules import RNA
from modules.reactioncomplex import ReactionComplex

class STAR(ReactionComplex):
    def __init__(self, sequence_name: str = None, transcriptional_control: tuple = None, model: Model = None):
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

        # RNA transcription initiation: RNA starts in unbound state (binding=None)
        transcription_initiation_rule = Rule(
            f'STAR_RNA_transcription_initiation_{regulated.name}',
            None >> regulated(state='init', binding=None),
            model.expressions['k_tx_plasmid_' + sequence_name]
        )
        rules.append(transcription_initiation_rule)

        # Binding of RNA regulator to the early transcript: Uses unified `binding` site
        binding_rule = Rule(
            f'STAR_RNA_regulator_binding_{regulated.name}_{regulator.name}',
            regulated(binding=None) + regulator(state='full', binding=None) >>
            regulated(binding=1) % regulator(state='full', binding=1),
            self.k_bind
        )
        rules.append(binding_rule)

        # Unbinding of full and early RNA transcript and RNA regulator
        unbinding_rule = Rule(
            f'STAR_RNA_regulator_unbinding_full_{regulated.name}_{regulator.name}',
            regulated(binding=1) % regulator(state='full', binding=1) >>
            regulated(binding=None) + regulator(state='full', binding=None),
            self.k_unbind
        )
        rules.append(unbinding_rule)

        # Activation of transcription with the RNA regulator
        full_transcription_with_regulator_rule = Rule(
            f'STAR_RNA_full_transcription_reg_{regulated.name}_{regulator.name}',
            regulated(state='init', binding=1) % regulator(state='full', binding=1) >>
            regulated(state='full', binding=1) % regulator(state='full', binding=1),
            self.k_act_reg
        )
        rules.append(full_transcription_with_regulator_rule)

        # Full transcription without the RNA regulator
        full_transcription_rule = Rule(
            f'STAR_RNA_full_transcription_{regulated.name}',
            regulated(state='init', binding=None) >>
            regulated(state='full', binding=None),
            self.k_act
        )
        rules.append(full_transcription_rule)

        # Stop of transcription with the STAR RNA regulator
        partial_transcription_with_regulator_rule = Rule(
            f'STAR_RNA_partial_transcription_reg_{regulated.name}_{regulator.name}',
            regulated(state='init', binding=1) % regulator(state='full', binding=1) >>
            regulated(state='partial', binding=1) % regulator(state='full', binding=1),
            self.k_stop_reg
        )
        rules.append(partial_transcription_with_regulator_rule)

        # Partial transcription without the RNA regulator
        partial_transcription_rule = Rule(
            f'STAR_RNA_partial_transcription_{regulated.name}',
            regulated(state='init', binding=None) >>
            regulated(state='partial', binding=None),
            self.k_stop
        )
        rules.append(partial_transcription_rule)

        self.rules = rules
