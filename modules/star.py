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

        # RNA transcription initiation
        rule = Rule(
            f'STAR_RNA_transcription_initiation_{regulated.name}',
            None >> regulated(state='init', sense=None, toehold=None, sequestration="free"),
            model.expressions['k_tx_plasmid_' + sequence_name]
        )
        rules.append(rule)

        # Binding of RNA regulator to the early transcript
        rule = Rule(
            f'STAR_RNA_regulator_binding_{regulated.name}_{regulator.name}',
            regulated(sense=None, sequestration="free") + regulator(state='full', sense=None, sequestration="free") >>
            regulated(sense=1, sequestration="bound") % regulator(state='full', sense=1, sequestration="bound"),
            self.k_bind
        )
        rules.append(rule)

        # Unbinding of full and early RNA transcript and RNA regulator
        rule = Rule(
            f'STAR_RNA_regulator_unbinding_full_{regulated.name}_{regulator.name}',
            regulated(sense=1, sequestration="bound") % regulator(state='full', sense=1, sequestration="bound") >>
            regulated(sense=None, sequestration="free") + regulator(state='full', sense=None, sequestration="free"),
            self.k_unbind
        )
        rules.append(rule)

        # Activation of transcription with the RNA regulator
        rule = Rule(
            f'STAR_RNA_full_transcription_reg_{regulated.name}_{regulator.name}',
            regulated(state='init', sense=1, sequestration="bound") % regulator(state='full', sense=1,
                                                                                sequestration="bound") >>
            regulated(state='full', sense=1, sequestration="bound") % regulator(state='full', sense=1,
                                                                                sequestration="bound"),
            self.k_act_reg
        )
        rules.append(rule)

        rule = Rule(
            f'STAR_RNA_full_transcription_{regulated.name}',
            regulated(state='init', sense=None, sequestration="free") >>
            regulated(state='full', sense=None, sequestration="free"),
            self.k_act
        )
        rules.append(rule)

        # Stop of transcription with RNA inhibitor
        rule = Rule(
            f'STAR_RNA_partial_transcription_reg_{regulated.name}_{regulator.name}',
            regulated(state='init', sense=1, sequestration="bound") % regulator(state='full', sense=1,
                                                                                sequestration="bound") >>
            regulated(state='partial', sense=1, sequestration="bound") % regulator(state='full', sense=1,
                                                                                   sequestration="bound"),
            self.k_stop_reg
        )
        rules.append(rule)

        rule = Rule(
            f'STAR_RNA_partial_transcription_{regulated.name}',
            regulated(state='init', sense=None, sequestration="free") >>
            regulated(state='partial', sense=None, sequestration="free"),
            self.k_stop
        )
        rules.append(rule)

        self.rules = rules
